import urllib.request
import urllib.parse
import json

text = """このシミュレーションは、核兵器の爆発による破壊力を教育目的で表現しています。最初に爆弾が落下し、爆発火球が発生します。続いて衝撃波が周囲の建物を破壊し、キノコ雲が上昇します。最後に、爆風と熱線が全域を覆い、白い閃光で終わります。核兵器の使用は人類にとって災害的な結果をもたらします。平和の重要性を忘れないでください。"""

# URL encode the data for the query
data = urllib.parse.urlencode({'text': text, 'speaker': 3}).encode()
req = urllib.request.Request('http://127.0.0.1:50021/audio_query', data=data, method='POST')
with urllib.request.urlopen(req) as res:
    query = json.loads(res.read().decode())

# Now synthesize
headers = {'Content-Type': 'application/json'}
req = urllib.request.Request('http://127.0.0.1:50021/synthesis', 
                             data=json.dumps(query).encode(),
                             headers=headers,
                             method='POST')
# Add parameters
params = urllib.parse.urlencode({'speaker': 3})
req_full = req.full_url + '?' + params
req = urllib.request.Request(req_full, 
                             data=json.dumps(query).encode(),
                             headers=headers,
                             method='POST')
with urllib.request.urlopen(req) as res:
    audio_data = res.read()

with open('narration.wav', 'wb') as f:
    f.write(audio_data)
print('Saved narration.wav')