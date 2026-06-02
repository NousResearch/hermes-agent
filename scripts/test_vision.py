#!/usr/bin/env python3
"""Test MiMo vision capability with local image."""
import urllib.request, json, os, base64

# Read MiMo API key
with open(os.path.expanduser('~/.hermes/.env')) as f:
    for line in f:
        line = line.strip()
        if line.startswith('export OPENAI_API_KEY') or line.startswith('OPENAI_API_KEY') and not line.startswith('#') and '=' in line:
            api_key = line.split('=', 1)[1].strip()
            if api_key.startswith("'"):
                api_key = api_key.strip("'")
            break
    else:
        api_key = None

if not api_key:
    print('No MiMo API key found')
    exit(1)

# Read and encode the image
with open(os.path.expanduser('~/.hermes/image_cache/img_094b9a96c8ea.jpg'), 'rb') as f:
    img_data = f.read()

b64 = base64.b64encode(img_data).decode()
print(f"Image size: {len(img_data)} bytes, base64: {len(b64)} chars")

# Try MiMo V2.5 Pro with image
payload = {
    'model': 'mimo-v2-omni',
    'messages': [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': '这张截图里显示了什么？请提取所有文字内容'},
                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}}
            ]
        }
    ],
    'max_tokens': 1000
}

req = urllib.request.Request(
    'https://api.xiaomimimo.com/v1/chat/completions',
    data=json.dumps(payload).encode(),
    headers={
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
)

try:
    resp = urllib.request.urlopen(req, timeout=60)
    result = json.loads(resp.read().decode())
    content = result['choices'][0]['message']['content']
    print('\n--- MiMo Vision Result ---')
    print(content)
    print('---')
    # Print usage info
    if 'usage' in result:
        u = result['usage']
        print(f'Tokens: {u.get("total_tokens", "?")} (input: {u.get("prompt_tokens", "?")}, output: {u.get("completion_tokens", "?")})')
except urllib.error.HTTPError as e:
    err = e.read().decode()
    print(f'HTTP {e.code}: {err[:500]}')
except Exception as e:
    print(f'Error: {e}')
