import pandas as pd
from flask import Flask, render_template, request
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

df = pd.read_csv('genre.csv')


X = df[['Pitch', 'Rhythm', 'Color']]
y = df['Genre']  


clf = DecisionTreeClassifier()
clf.fit(X, y)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
       
        features = [
            int(request.form['Pitch']),
            int(request.form['Rhythm']),
            int(request.form['Color'])
        ]

        
        genre = clf.predict([features])[0]

        return render_template('result.html', Genre=genre)

    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
