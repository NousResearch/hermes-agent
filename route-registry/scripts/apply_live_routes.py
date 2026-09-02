import json, shutil, sys, tempfile
from pathlib import Path
sys.path.insert(0, '/home/kensei/repos/KenseiAgent/route-registry/route_registry')
sys.path.insert(0, '/home/kensei/repos/KenseiAgent/route-registry')
import generator

LIVE = Path('/home/kensei/.hermes')
REG = '/home/kensei/repos/KenseiAgent/route-registry/registry/route-slots.yaml'
SURF = '/home/kensei/repos/KenseiAgent/route-registry/registry/surfaces.yaml'
BACKUP_ROOT = Path('/home/kensei/repos/KenseiAgent/route-registry/route_registry/backups')

# build the plan against the LIVE tree directly (read-only build)
plan = generator.build_plan(LIVE, REG, SURF)
plan['env_key_bypass'] = generator.scan_env_key_bypass_tree(LIVE, generator.load_registry(REG))
blockers = generator.apply_blockers(plan)
print('apply_blockers:', blockers)
if blockers:
    raise SystemExit('REFUSED by gate: ' + '; '.join(blockers))

# apply atomically with backups under the backups root
applied = generator.apply_plan(plan, BACKUP_ROOT)
print('applied:', len(applied), 'surfaces')
print('backup_root:', BACKUP_ROOT)
for a in applied[:10]:
    print('  ', a.get('surface'), '->', a.get('backup'))
# verify: read back a couple of profile chains
for name in ('ceecee-brand', 'denji'):
    p = LIVE / 'profiles' / name / 'config.yaml'
    import yaml
    doc = yaml.safe_load(p.read_text())
    fp = doc.get('fallback_providers') or []
    print(f'--- {name} chain now:')
    for e in fp:
        print('   ', e.get('provider'), '|', e.get('model'), '| pool:', e.get('pool_accounts'))
