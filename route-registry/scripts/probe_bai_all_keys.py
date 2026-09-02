import json, os, sys
from pathlib import Path
sys.path.insert(0, '/home/kensei/repos/KenseiAgent')
os.environ.setdefault('HERMES_HOME', '/home/kensei/.hermes')
from hermes_cli.auth import read_credential_pool
from hermes_cli.config import load_config
import urllib.request, urllib.error

cfg = load_config()
cps = cfg.get('custom_providers') or []
entries = read_credential_pool('custom:bai') or []
keys = []
for e in entries:
    if isinstance(e, dict):
        keys.append(e.get('access_token'))
    else:
        keys.append(getattr(e, 'access_token', None))
base = next((c.get('base_url') for c in cps if isinstance(c, dict) and c.get('name') == 'bai'), '')
print('bai keys in pool:', len([k for k in keys if k]))

for idx, key in enumerate(keys):
    body = json.dumps({'model': 'glm-5.3-flash',
                       'messages': [{'role': 'user', 'content': 'Say hello in exactly two words.'}],
                       'max_tokens': 40}).encode()
    req = urllib.request.Request(base.rstrip('/') + '/chat/completions', data=body,
                                 headers={'Content-Type': 'application/json',
                                          'Authorization': 'Bearer ' + key})
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            d = json.load(resp)
            msg = (d.get('choices') or [{}])[0].get('message', {}).get('content', '')
            usage = d.get('usage')
            print(f'bai key#{idx+1}: HTTP {resp.status} reply={msg[:40]!r} usage={usage}')
    except urllib.error.HTTPError as e:
        print(f'bai key#{idx+1}: HTTP {e.code} {e.read().decode()[:140]}')
    except Exception as e:
        print(f'bai key#{idx+1}: ERR {type(e).__name__} {str(e)[:140]}')
