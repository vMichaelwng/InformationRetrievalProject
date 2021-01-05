import pandas as pd

df1 = pd.read_csv('movies.csv')
df2 = pd.read_csv('ratings.csv')

df = pd.merge(df1, df2, on='movieId')
df

columns = ['userId', 'movieId', 'rating']
df_model = pd.DataFrame(df, columns = columns)
df_model

"""# Data slicing/cleaning
* Remove movie with too less reviews 


* Remove user who give too less reviews 
"""

f = ['count','mean']

df_movie_summary = df_model.groupby('movieId')['rating'].agg(f)
df_movie_summary.index = df_movie_summary.index.map(int)
movie_benchmark = round(df_movie_summary['count'].quantile(0.7), 0)
drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index

print('Movie minimum times of review: {}'.format(movie_benchmark))

df_user_summary = df.groupby('userId')['rating'].agg(f)
df_user_summary.index = df_user_summary.index.map(int)
user_benchmark = round(df_user_summary['count'].quantile(0.7),0)
drop_user_list = df_user_summary[df_user_summary['count'] < user_benchmark].index

print('User minimum times of review: {}'.format(user_benchmark))

print('Original Shape: {}'.format(df_model.shape))
df_model = df_model[~df_model['movieId'].isin(drop_movie_list)]
df_model = df_model[~df_model['userId'].isin(drop_user_list)]
print('After Trim Shape: {}'.format(df_model.shape))
print('-Data Examples-')
df_model.head(5)

"""COLLABORATIVE FILTERING"""

!pip install surprise

from surprise import Reader
from surprise import Dataset
from surprise import SVD
from surprise.model_selection import cross_validate

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df_model[['userId', 'movieId', 'rating']], reader)

"""# Matrix factorization CF using sklearn surprise SVD"""

svd = SVD()
cross_validate(svd, data, measures=['RMSE', 'MAE'])

df1.set_index('movieId', inplace = True)
df1

data_596 = df_model[(df_model['userId'] == 596) & (df_model['rating'] == 5)]
data_596 = data_596.set_index('movieId')
data_596 = data_596.join(df1)['title']
print(data_596)

data_596 = df1.copy()
data_596 = data_596.reset_index()
data_596 = data_596[~data_596['movieId'].isin(drop_movie_list)]

data = Dataset.load_from_df(df_model[['userId', 'movieId', 'rating']], reader)

trainset = data.build_full_trainset()
svd.fit(trainset)

data_596['score prediction'] = data_596['movieId'].apply(lambda x: svd.predict(596, x).est)

data_596 = data_596.drop('movieId', axis = 1)

data_596 = data_596.sort_values('score prediction', ascending=False)
print(data_596.head(10))

"""# Recommendation using pearson correlation"""

df_p = pd.pivot_table(df,values='rating',index='userId',columns='movieId')
print(df_p.shape)
df_p

def recommend(movie_title, min_count):
    print("For movie ({})".format(movie_title))
    print("- Top 10 movies recommended based on Pearsons'R correlation - ")
    i = int(df1.index[df1['title'] == movie_title][0])
    target = df_p[i]
    similar_to_target = df_p.corrwith(target)
    corr_target = pd.DataFrame(similar_to_target, columns = ['PearsonR'])
    corr_target.dropna(inplace = True)
    corr_target = corr_target.sort_values('PearsonR', ascending = False)
    corr_target.index = corr_target.index.map(int)
    corr_target = corr_target.join(df1).join(df_movie_summary)[['PearsonR', 'title', 'count', 'mean']]
    print(corr_target[corr_target['count']>min_count][:10].to_string(index=False))

recommend("Black Panther (2017)", 0)

recommend("Thor: Ragnarok (2017)", 0)

recommend("WALL·E (2008)", 0)
