import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt
import string
from collections import Counter
import nltk
#nltk.download('punkt')
#nltk.download('wordnet')
from stop_words import stop_words #import a handmade list of stopwords

# Load kaggle dataset
# 5842 rows
df = pd.read_csv('data.csv')

# missing values
# print(f'Missing values:\n{df.isnull().sum()}')
# no missing values

# encode outputs w/dictionary
sentiment_mapping = {'positive': 1, 'negative': 0, 'neutral': 2}
df['Sentiment'] = df['Sentiment'].map(sentiment_mapping)

#------------------------------------------------
# Count the number positive/negative/neutral entries

# positive_count = df[df['Sentiment'] == 1].shape[0] #1852
# negative_count = df[df['Sentiment'] == 0].shape[0] #860
# neutral_count = df[df['Sentiment'] == 2].shape[0] #3130
# print(f'Positive:{positive_count}')
# print(f'Negative:{negative_count}')
# print(f'Neutral:{neutral_count}')

#------------------------------------------------
# Join all the negative and positive words

positive_words = " ".join(df[df["Sentiment"] == 1]["Sentence"])
negative_words = " ".join(df[df["Sentiment"] == 0]["Sentence"])

#------------------------------------------------
# Preprocessing and cleaning steps

def preprocess_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    return tokens

def remove_stopwords(tokens):
    filtered_tokens = [token for token in tokens if token not in stop_words]
    return filtered_tokens

def do_lemmatize(tokens):
    lemmatizer = nltk.WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

def clean_text(text):
    tokens = preprocess_text(text)
    filtered_tokens = remove_stopwords(tokens)
    lemmatized_tokens = do_lemmatize(filtered_tokens)
    clean_text = ' '.join(lemmatized_tokens)
    return clean_text

positive_cleaned_data = clean_text(positive_words)
negative_cleaned_data = clean_text(negative_words)

#------------------------------------------------
# Generate frequency tables for positive and negative words

# # Tokenize and count positive words
# positive_words_list = positive_cleaned_data.lower().split()
# positive_word_counts = Counter(positive_words_list)
# # Tokenize and count negative words
# negative_words_list = negative_cleaned_data.lower().split()
# negative_word_counts = Counter(negative_words_list)

# def generate_table(word_counts):
#     # Step 1: Sort word counts by frequency in descending order
#     sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
#     # Step 2: Initialize an empty dictionary to store table data
#     table_data = {'Word': [], 'Count': []}
    
#     # Step 3: Iterate through sorted_word_counts and populate table_data
#     for word, count in sorted_word_counts:
#         table_data['Word'].append(word)
#         table_data['Count'].append(count)
    
#     # Step 4: Convert the dictionary to a pandas DataFrame and return it
#     return pd.DataFrame(table_data)

# positive_word_table = generate_table(positive_word_counts)
# negative_word_table = generate_table(negative_word_counts)
# print("Positive Words Table:")
# print(positive_word_table.head())
# print("\nNegative Words Table:")
# print(negative_word_table.head())

# # Check to see if this function works
# def generate_table(word_counts):
#     return pd.DataFrame.from_dict({'Word': list(word_counts.keys()), 'Count': list(word_counts.values())})



#------------------------------------------------
# WordCloud generation

p_wordcloud = WordCloud(width=800, height=600, stopwords=STOPWORDS).generate(positive_cleaned_data)
n_wordcloud = WordCloud(width=800, height=600, stopwords=STOPWORDS).generate(negative_cleaned_data)
plt.imshow(p_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
plt.imshow(n_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
#------------------------------------------------
