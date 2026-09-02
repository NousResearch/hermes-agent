import json, shutil, sys, tempfile, os
from pathlib import Path
sys.path.insert(0, '/home/kensei/repos/KenseiAgent/route-registry/route_registry')
sys.path.insert(0, '/home/kensei/repos/KenseiAgent/route-registry')
import generator

LIVE = Path('/home/kensei/.hermes')
REG = '/home/kensei/repos/KenseiAgent/route-registry/registry/route-slots.yaml'
SURF = '/home/kensei/repos/KenseiAgent/route-registry/registry/surfaces.yaml'

# Build temp home copy for a preview plan
tmp = Path(tempfile.mkdtemp(prefix='rreg-preview-'))
shutil.copy2(LIVE / 'config.yaml', tmp / 'config.yaml')
for prof in (LIVE / 'profiles').iterdir():
    cfg = prof / 'config.yaml'
    if cfg.exists():
        (tmp / 'profiles' / prof.name).mkdir(parents=True, exist_ok=True)
        shutil.copy2(cfg, tmp / 'profiles' / prof.name / 'config.yaml')
    env = prof / '.env'
    if env.exists():
        shutil.copy2(env, tmp / 'profiles' / prof.name / '.env')

plan = generator.build_plan(tmp, REG, SURF)
plan['env_key_bypass'] = generator.scan_env_key_bypass_tree(tmp, generator.load_registry(REG))
plan['apply_blockers'] = generator.apply_blockers(plan)
print('entries:', len(plan['entries']))
print('env_bypass blockers:', len(plan['env_key_bypass']['blockers']))
print('apply_blockers:', plan['apply_blockers'])
# summarize surfaces that would CHANGE
changed = [e['surface'] for e in plan['entries'] if e.get('changes')]
print('surfaces with changes:', len(changed))
for s in changed[:60]:
    print('  ', s)
# Save full preview
outp = Path('/home/kensei/repos/KenseiAgent/route-registry/out/preview-20260902-live.json')
outp.write_text(json.dumps({'mode': 'dry-run', 'plan': plan}, indent=2, sort_keys=True))
print('preview written:', outp)
