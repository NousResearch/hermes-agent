from pathlib import Path
import datetime, json, shutil, sys
import yaml

ROOT = Path('/home/kensei/.hermes')
REG = Path('/home/kensei/repos/KenseiAgent/route-registry')
STAMP = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
BACKUP = ROOT / 'backups' / f'glm53-to-flash-{STAMP}'
TARGETS = {
    'kensei-review': ('model', 'default'),
    'octacon-architect': ('model', 'default'),
    'octacon-backend': ('model', 'default'),
    'wesker': ('model', 'default'),
    'moss': ('agent', 'model'),
}

# Preflight: exact old values and expected shapes. No writes before all pass.
docs = {}
for name, path_keys in TARGETS.items():
    p = ROOT / 'profiles' / name / 'config.yaml'
    doc = yaml.safe_load(p.read_text()) or {}
    cur = doc
    for k in path_keys:
        cur = cur.get(k) if isinstance(cur, dict) else None
    if cur != 'glm-5.3':
        raise SystemExit(f'PRECHECK FAIL {name}: expected glm-5.3 at {".".join(path_keys)}, got {cur!r}')
    docs[name] = (p, doc, path_keys)

# Registry surface preflight.
surf_path = REG / 'registry' / 'surfaces.yaml'
surf_doc = yaml.safe_load(surf_path.read_text()) or {}
surfs = {s['surface']: s for s in surf_doc.get('surfaces', [])}
for name in TARGETS:
    if name not in surfs:
        raise SystemExit(f'PRECHECK FAIL registry missing surface {name}')
    if surfs[name].get('main_model') != 'glm-5.3':
        raise SystemExit(f'PRECHECK FAIL registry {name}: {surfs[name].get("main_model")!r}')

# Backup exact affected files.
BACKUP.mkdir(parents=True)
for name, (p, _, _) in docs.items():
    d = BACKUP / name
    d.mkdir()
    shutil.copy2(p, d / 'config.yaml')
(BACKUP / 'registry').mkdir()
shutil.copy2(surf_path, BACKUP / 'registry' / 'surfaces.yaml')

# Targeted writes preserving all unrelated YAML values.
for name, (p, doc, path_keys) in docs.items():
    doc[path_keys[0]][path_keys[1]] = 'glm-5.3-flash'
    p.write_text(yaml.safe_dump(doc, sort_keys=False), encoding='utf-8')

for name in TARGETS:
    surfs[name]['main_model'] = 'glm-5.3-flash'
    surfs[name]['slots'] = ['slot-glm53flash-1', 'slot-glm53flash-2', 'slot-glm53flash-3', 'slot-glm53flash-4']
surf_path.write_text(yaml.safe_dump(surf_doc, sort_keys=False), encoding='utf-8')

# Exact readback.
result = {'backup': str(BACKUP), 'profiles': {}}
for name, (p, _, path_keys) in docs.items():
    doc = yaml.safe_load(p.read_text()) or {}
    cur = doc[path_keys[0]][path_keys[1]]
    if cur != 'glm-5.3-flash':
        raise SystemExit(f'READBACK FAIL {name}: {cur!r}')
    result['profiles'][name] = cur
updated_surfs = {s['surface']: s for s in (yaml.safe_load(surf_path.read_text()) or {})['surfaces']}
for name in TARGETS:
    s = updated_surfs[name]
    if s['main_model'] != 'glm-5.3-flash' or s['slots'] != ['slot-glm53flash-1','slot-glm53flash-2','slot-glm53flash-3','slot-glm53flash-4']:
        raise SystemExit(f'REGISTRY READBACK FAIL {name}')
print(json.dumps(result, indent=2))
