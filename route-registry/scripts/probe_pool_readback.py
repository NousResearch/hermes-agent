import json, os, sys
from pathlib import Path
sys.path.insert(0, '/home/kensei/repos/KenseiAgent')
os.environ.setdefault('HERMES_HOME', '/home/kensei/.hermes')

from hermes_cli.auth import read_credential_pool  # noqa: E402

for prov in ['custom:xkiro-free', 'custom:xkiro-pro', 'custom:bai']:
    entries = read_credential_pool(prov)
    # entries may be list of dicts or list of PooledCredential objects
    out = []
    for e in (entries if isinstance(entries, list) else [entries]):
        if not isinstance(e, dict):
            e = e.__dict__ if hasattr(e, '__dict__') else {}
        tok = e.get('access_token') or e.get('api_key') or ''
        out.append({'label': e.get('label'), 'base_url': e.get('base_url'),
                    'auth_type': e.get('auth_type'),
                    'key_len': len(tok) if isinstance(tok, str) else 0})
    print(prov, json.dumps(out))
