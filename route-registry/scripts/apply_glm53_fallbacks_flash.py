from pathlib import Path
import datetime, json, shutil
import yaml
ROOT=Path('/home/kensei/.hermes')
STAMP=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
BACKUP=ROOT/'backups'/f'glm53-fallbacks-to-flash-{STAMP}'
TARGETS=['kensei-review','moss','octacon','octacon-backend','octacon-frontend','quan']
OLD='zai-org/GLM-5.3'
NEW='z-ai/glm-5.3-flash'
pre={}
for name in TARGETS:
    p=ROOT/'profiles'/name/'config.yaml'
    d=yaml.safe_load(p.read_text()) or {}
    fps=d.get('fallback_providers') or []
    hits=[i for i,e in enumerate(fps) if isinstance(e,dict) and e.get('model')==OLD]
    if len(hits)!=1: raise SystemExit(f'PRECHECK {name}: expected exactly 1 {OLD}, got {hits}')
    pre[name]=(p,d,hits[0])
BACKUP.mkdir(parents=True)
for name,(p,d,i) in pre.items():
    out=BACKUP/name; out.mkdir(); shutil.copy2(p,out/'config.yaml')
for name,(p,d,i) in pre.items():
    d['fallback_providers'][i]['model']=NEW
    p.write_text(yaml.safe_dump(d,sort_keys=False),encoding='utf-8')
result={}
for name,(p,_,_) in pre.items():
    d=yaml.safe_load(p.read_text()) or {}
    old_hits=sum(1 for e in d.get('fallback_providers',[]) if isinstance(e,dict) and e.get('model')==OLD)
    new_hits=sum(1 for e in d.get('fallback_providers',[]) if isinstance(e,dict) and e.get('model')==NEW)
    if old_hits or new_hits!=1: raise SystemExit(f'READBACK FAIL {name}: old={old_hits} new={new_hits}')
    result[name]={'old_hits':old_hits,'new_hits':new_hits}
print(json.dumps({'backup':str(BACKUP),'profiles':result},indent=2))
