import json, os, sys
from pathlib import Path
sys.path.insert(0, '/home/kensei/repos/KenseiAgent')
os.environ.setdefault('HERMES_HOME', '/home/kensei/.hermes')

from hermes_cli.auth import read_credential_pool  # noqa: E402
from hermes_cli.config import get_custom_provider_extra_headers, load_config  # noqa: E402

cfg = load_config()
cps = cfg.get('custom_providers') or []

def pool_key(provider):
    entries = read_credential_pool(provider) or []
    if not entries:
        return None
    e = entries[0]
    if isinstance(e, dict):
        return e.get('access_token')
    return getattr(e, 'access_token', None)

def entry_base(name):
    for c in cps:
        if isinstance(c, dict) and c.get('name') == name:
            return c.get('base_url', '')
    return ''

# Use urllib to fire a genuine chat completion with the pooled key + headers.
import urllib.request, urllib.error  # noqa: E402

CASES = [
    ('custom:xkiro-free', 'xkiro-free', 'deepseek/deepseek-v4-flash'),
    ('custom:xkiro-pro', 'xkiro-pro', 'deepseek/deepseek-v4-pro'),
    ('custom:bai', 'bai', 'glm-5.3-flash'),
]
for pool_prov, cfg_name, model in CASES:
    key = pool_key(pool_prov)
    base = entry_base(cfg_name)
    hdrs = get_custom_provider_extra_headers(base, cps, cfg) or {}
    if not key or not base:
        print(f'{pool_prov}: MISSING key/base'); continue
    body = json.dumps({'model': model, 'messages': [{'role': 'user', 'content': 'Reply OK'}], 'max_tokens': 8}).encode()
    req = urllib.request.Request(base.rstrip('/') + '/chat/completions', data=body,
                                 headers={'Content-Type': 'application/json',
                                          'Authorization': 'Bearer ' + key,
                                          **hdrs})
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            d = json.load(resp)
            msg = (d.get('choices') or [{}])[0].get('message', {}).get('content', '')[:20]
            print(f'{pool_prov} [{model}]: HTTP {resp.status} OK reply={msg!r}')
    except urllib.error.HTTPError as e:
        print(f'{pool_prov} [{model}]: HTTP {e.code} {e.read().decode()[:140]}')
    except Exception as e:
        print(f'{pool_prov} [{model}]: ERR {type(e).__name__} {str(e)[:140]}')
