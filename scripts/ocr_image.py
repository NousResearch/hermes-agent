#!/usr/bin/env python3
"""OCR image using MiMo V2-Omni (multimodal vision model).
Usage: python3 ocr_image.py <image_path> [--prompt "custom prompt"]

Returns extracted text from the image. Cost: ~¥0.005-0.01 per call.
"""
import urllib.request, json, os, base64, sys

API_URL = 'https://api.xiaomimimo.com/v1/chat/completions'
MODEL = 'mimo-v2-omni'
ENV_PATH = os.path.expanduser('~/.hermes/.env')

def get_api_key():
    with open(ENV_PATH) as f:
        for line in f:
            line = line.strip()
            if line.startswith('export OPENAI_API_KEY') or (line.startswith('OPENAI_API_KEY') and '=' in line):
                key = line.split('=', 1)[1].strip().strip("'").strip('"')
                if key and key != '***':
                    return key
    return None

def ocr_image(image_path, prompt=None):
    if not os.path.exists(image_path):
        return f'Error: File not found: {image_path}'

    api_key = get_api_key()
    if not api_key:
        return 'Error: OPENAI_API_KEY not found in .env'

    with open(image_path, 'rb') as f:
        img_data = f.read()
    b64 = base64.b64encode(img_data).decode()

    text_prompt = prompt or '请提取这张图片中的所有文字内容，保持原有格式和排版顺序。如果包含中英文混合，请完整输出。'

    payload = {
        'model': MODEL,
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': text_prompt},
                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}}
            ]
        }],
        'max_tokens': 2000
    }

    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    )

    try:
        resp = urllib.request.urlopen(req, timeout=60)
        result = json.loads(resp.read().decode())
        return result['choices'][0]['message']['content']
    except urllib.error.HTTPError as e:
        return f'HTTP {e.code}: {e.read().decode()[:500]}'
    except Exception as e:
        return f'Error: {e}'

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 ocr_image.py <image_path> [--prompt "prompt text"]')
        sys.exit(1)

    path = sys.argv[1]
    prompt = None
    if '--prompt' in sys.argv:
        idx = sys.argv.index('--prompt')
        if idx + 1 < len(sys.argv):
            prompt = sys.argv[idx + 1]

    result = ocr_image(path, prompt)
    print(result)
