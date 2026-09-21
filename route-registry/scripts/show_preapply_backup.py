import yaml
from pathlib import Path
backup = Path('/home/kensei/repos/KenseiAgent/route-registry/route_registry/backups/20260902T174446Z')
for name in ('denji', 'ceecee-brand', 'kensei-review'):
    p = backup / name / 'config.yaml'
    if p.exists():
        doc = yaml.safe_load(p.read_text())
        fp = doc.get('fallback_providers') or []
        print(f'=== {name} PRE-APPLY chain:')
        for e in fp:
            print('   ', e.get('provider'), '|', e.get('model'), '| base:', e.get('base_url'))
    else:
        print(name, 'NO BACKUP')
