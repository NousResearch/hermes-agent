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
key = entries[0].get('access_token') if isinstance(entries[0], dict) else getattr(entries[0], 'access_token')
base = next((c.get('base_url') for c in cps if isinstance(c, dict) and c.get('name') == 'bai'), '')
body = json.dumps({'model': 'glm-5.3-flash',
                   'messages': [{'role': 'user', 'content': 'Say hello in exactly two words.'}],
                   'max_tokens': 200}).encode()
req = urllib.request.Request(base.rstrip('/') + '/chat/completions', data=body,
                             headers={'Content-Type': 'application/json',
                                      'Authorization': 'Bearer ' + key})
with urllib.request.urlopen(req, timeout=60) as resp:
    d = json.load(resp)
    msg = (d.get('choices') or [{}])[0].get('message', {}).get('content', '')
    print('B.AI reply:', repr(msg[:120]))
    print('usage:', d.get('usage'))
