#!/usr/bin/env python3
"""Propagate Kubrick graph identity and symbolic state across storyboard frames."""
from __future__ import annotations
import argparse, json
from pathlib import Path
try:
    import yaml
except ImportError:
    raise SystemExit("pyyaml required")

def load(path):
    p=Path(path); return json.loads(p.read_text()) if p.suffix=='.json' else yaml.safe_load(p.read_text()) or {}
def dump(path,data):
    Path(path).parent.mkdir(parents=True,exist_ok=True); Path(path).write_text(yaml.safe_dump(data,sort_keys=False),encoding='utf-8')

def propagate(graph, plan):
    errors=[]; warnings=[]; frames=[]
    nodes={n['id']:n for n in graph.get('nodes',[])}
    if graph.get('validation',{}).get('status')!='VALID': errors.append('source graph is not VALID')
    prior_states={nid:n.get('initial_state','present') for nid,n in nodes.items()}
    prior_owners={nid:None for nid in nodes}
    prior_residue=list(graph.get('residue',[]))
    for ordinal,spec in enumerate(plan.get('frames',[]),start=1):
        states=dict(prior_states); states.update(spec.get('node_states',{}))
        owners=dict(prior_owners); owners.update(spec.get('motif_ownership',{}))
        residue=list(dict.fromkeys(prior_residue + spec.get('residue_add',[])))
        resets=[]
        for nid,value in states.items():
            if nid in prior_states and prior_states[nid]!=value and value==nodes.get(nid,{}).get('initial_state'):
                resets.append(f'{nid}:reset-to-initial')
        required=spec.get('required_mutations',[])
        for item in required:
            if item not in json.dumps(spec,sort_keys=True): warnings.append(f'frame {ordinal} required mutation not evidenced: {item}')
        frames.append({'frame_id':spec.get('frame_id',f'frame-{ordinal:03d}'),'ordinal':ordinal,'node_states':states,'motif_ownership':owners,'object_states':spec.get('object_states',{}),'light_states':spec.get('light_states',{}),'material_states':spec.get('material_states',{}),'residue':residue,'convergence_sites':spec.get('convergence_sites',[s.get('site_id') for s in graph.get('convergence_sites',[])]),'required_mutations':required,'prohibited_resets':resets})
        if resets and not spec.get('allow_resets',False): errors.extend(f'frame {ordinal} prohibited reset: {r}' for r in resets)
        prior_states,prior_owners,prior_residue=states,owners,residue
    if len(frames)<2: errors.append('at least two frames are required')
    return {'schema_version':'1.0.0','storyboard_id':plan.get('storyboard_id','kubrick-storyboard'),'graph_id':graph.get('graph_id','unknown'),'frames':frames,'validation':{'status':'VALID' if not errors else 'INVALID','errors':errors,'warnings':warnings,'frame_count':len(frames)}}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--graph',required=True); p.add_argument('--plan',required=True); p.add_argument('--output'); a=p.parse_args()
    result=propagate(load(a.graph),load(a.plan)); text=yaml.safe_dump(result,sort_keys=False)
    if a.output: dump(a.output,result)
    else: print(text)
    raise SystemExit(0 if result['validation']['status']=='VALID' else 1)
if __name__=='__main__': main()
