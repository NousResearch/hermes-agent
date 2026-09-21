import datetime, shutil
from pathlib import Path
import yaml

# --- Item 1: root config chain — add xKiro-pro + B.AI as same-model (gpt-5.6-sol)
# routes before the existing paid/local fallbacks. Root main = gpt-5.6-sol.
p = Path('/home/kensei/.hermes/config.yaml')
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
b = Path('/home/kensei/.hermes/backups') / f'config-root-chain-{stamp}.yaml'
b.parent.mkdir(exist_ok=True)
shutil.copy2(p, b)
doc = yaml.safe_load(p.read_text()) or {}
fb = doc.get('fallback_providers') or []
main = (doc.get('model') or {}).get('default')
print('root main:', main, '| backup:', b)

# New same-model premium routes first: xkiro-pro serves openai/gpt-5.6-sol;
# bai serves gpt-5.6-sol. Only add if not already present.
def prov_models():
    return {(e.get('provider'), e.get('model')) for e in fb}

existing = prov_models()
additions = []
if ('custom:xkiro-pro', 'openai/gpt-5.6-sol') not in existing:
    additions.append({'provider': 'custom:xkiro-pro', 'model': 'openai/gpt-5.6-sol',
                      'base_url': 'https://api.xkiro.com/v1'})
if ('custom:bai', 'gpt-5.6-sol') not in existing:
    additions.append({'provider': 'custom:bai', 'model': 'gpt-5.6-sol',
                      'base_url': 'https://api.b.ai/v1'})

if additions:
    # insert before the first existing fallback (after the implicit primary), i.e.
    # prepend additions to the fallback list so they are tried first.
    doc['fallback_providers'] = additions + fb
p.write_text(yaml.safe_dump(doc, sort_keys=False), encoding='utf-8')
print('added:', [a['provider'] for a in additions])
print('new chain:')
for e in doc['fallback_providers']:
    print('  ', e.get('provider'), '|', e.get('model'))
