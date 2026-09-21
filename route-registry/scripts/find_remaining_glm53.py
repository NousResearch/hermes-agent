from pathlib import Path
import yaml
root=Path('/home/kensei/.hermes/profiles')
for cfg in sorted(root.glob('*/config.yaml')):
    doc=yaml.safe_load(cfg.read_text()) or {}
    hits=[]
    def walk(x,path=''):
        if isinstance(x,dict):
            for k,v in x.items(): walk(v,f'{path}.{k}' if path else str(k))
        elif isinstance(x,list):
            for i,v in enumerate(x): walk(v,f'{path}[{i}]')
        elif isinstance(x,str):
            norm=x.lower().replace('_','-')
            if 'glm-5.3' in norm and 'glm-5.3-flash' not in norm:
                hits.append((path,x))
    walk(doc)
    if hits: print(cfg.parent.name, hits)
