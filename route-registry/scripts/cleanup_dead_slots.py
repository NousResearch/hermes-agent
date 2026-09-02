import shutil
from pathlib import Path
import yaml

# --- Item 2: remove dead glm53-3/4 slots (serve retired glm-5.3, zero surfaces ref)
reg_path = Path('/home/kensei/repos/KenseiAgent/route-registry/registry/route-slots.yaml')
b = Path('/home/kensei/repos/KenseiAgent/route-registry/route_registry/backups') / 'route-slots.yaml.bak-glm53-removal'
b.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(reg_path, b)
reg = yaml.safe_load(reg_path.read_text()) or {}
before = len(reg['slots'])
reg['slots'] = [s for s in reg['slots'] if s.get('slot') not in ('slot-glm53-3', 'slot-glm53-4')]
reg_path.write_text(yaml.safe_dump(reg, sort_keys=False), encoding='utf-8')
print(f'slots before={before} after={len(reg["slots"])} backup={b}')

# Also flip root's stale glm-5.3 fallback to glm-5.3-flash (retirement consistency)
root = Path('/home/kensei/.hermes/config.yaml')
doc = yaml.safe_load(root.read_text()) or {}
changed = False
for e in doc.get('fallback_providers') or []:
    if isinstance(e, dict) and e.get('model') == 'glm-5.3':
        e['model'] = 'glm-5.3-flash'
        changed = True
if changed:
    root.write_text(yaml.safe_dump(doc, sort_keys=False), encoding='utf-8')
print('root glm-5.3 -> flash:', changed)
