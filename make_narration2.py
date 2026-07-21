import urllib.request
import urllib.parse
import json

text = "このシミュレーションは、核兵器の破壊力を教育目的で表現しています。最初に爆弾が上空から落下し、次に巨大な火球が発生し、衝撃波が都市を飲み込み、建物が崩壊し、最後にキノコ雲が上がります。これにより、核兵器の甚大な被害を理解していただければ幸いです。"
# Encode the text for the query string
params = {'text': text, 'speaker': 3}
query_string = urllib.parse.urlencode(params)
url = f"http://127.0.0.1:50021/audio_query?{query_string}"
print(f"Requesting: {url}")
req = urllib.request.Request(url, method='POST')
try:
    with urllib.request.urlopen(req) as resp:
        data = json.load(resp)
except Exception as e:
    print(f"Error in audio_query: {e}")
    exit(1)

# Now synthesis
synth_params = {'speaker': 3}
synth_string = urllib.parse.urlencode(synth_params)
synth_url = f"http://127.0.0.1:50021/synthesis?{synth_string}"
headers = {'Content-Type': 'application/json'}
req2 = urllib.request.Request(synth_url, data=json.dumps(data).encode('utf-8'), headers=headers, method='POST')
try:
    with urllib.request.urlopen(req2) as resp:
        audio = resp.read()
except Exception as e:
    print(f"Error in synthesis: {e}")
    exit(1)

with open('narration.wav', 'wb') as f:
    f.write(audio)
print('Saved narration.wav')