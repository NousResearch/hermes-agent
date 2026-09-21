from pathlib import Path
import json, urllib.request, urllib.error
root=Path('/home/kensei/.hermes')
auth=json.loads((root/'auth.json').read_text())
entries=(auth.get('credential_pool') or {}).get('custom:commandcode') or []
if not entries: raise SystemExit('NO_COMMANDCODE_POOL')
# Accept either pool envelope or bare list records; never print secret.
entry=entries[0]
key=entry.get('api_key') or entry.get('token') or entry.get('credential') or entry.get('access_token')
if not key: raise SystemExit('NO_COMMANDCODE_KEY_FIELD')
req=urllib.request.Request('https://api.commandcode.ai/provider/v1/models',headers={'Authorization':'Bearer '+key,'User-Agent':'Mozilla/5.0','Accept':'application/json'})
try:
    with urllib.request.urlopen(req,timeout=30) as r: data=json.load(r)
except urllib.error.HTTPError as e:
    raise SystemExit(f'HTTP {e.code}')
ids=sorted(str(m.get('id')) for m in data.get('data',[]) if isinstance(m,dict))
hits=[x for x in ids if 'glm-5.3' in x.lower()]
print(json.dumps({'model_count':len(ids),'glm53_models':hits},indent=2))
