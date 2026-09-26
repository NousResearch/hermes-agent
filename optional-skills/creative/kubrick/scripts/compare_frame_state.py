#!/usr/bin/env python3
"""Compare adjacent Kubrick storyboard frames for continuity and intended mutation."""
from __future__ import annotations
import argparse, json
from pathlib import Path
try:
    import yaml
except ImportError:
    raise SystemExit("pyyaml required")

def load(path):
    p=Path(path); return json.loads(p.read_text()) if p.suffix=='.json' else yaml.safe_load(p.read_text()) or {}

def compare(storyboard):
    frames=storyboard.get('frames',[]); transitions=[]; errors=[]
    for left,right in zip(frames,frames[1:]):
        deltas={}
        for field in ('node_states','motif_ownership','object_states','light_states','material_states'):
            a=left.get(field,{}) or {}; b=right.get(field,{}) or {}
            changed={k:{'from':a.get(k),'to':b.get(k)} for k in sorted(set(a)|set(b)) if a.get(k)!=b.get(k)}
            if changed: deltas[field]=changed
        residue_lost=sorted(set(left.get('residue',[]))-set(right.get('residue',[])))
        convergence_lost=sorted(set(left.get('convergence_sites',[]))-set(right.get('convergence_sites',[])))
        resets=right.get('prohibited_resets',[])
        if residue_lost: errors.append(f"{left['frame_id']}->{right['frame_id']} lost residue: {residue_lost}")
        if resets: errors.append(f"{right['frame_id']} contains prohibited resets: {resets}")
        transitions.append({'from':left['frame_id'],'to':right['frame_id'],'deltas':deltas,'residue_lost':residue_lost,'convergence_lost':convergence_lost,'required_mutations':right.get('required_mutations',[]),'status':'VALID' if not residue_lost and not resets else 'INVALID'})
    return {'storyboard_id':storyboard.get('storyboard_id'),'transition_count':len(transitions),'transitions':transitions,'status':'VALID' if not errors else 'INVALID','errors':errors}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--storyboard',required=True); p.add_argument('--output'); a=p.parse_args()
    result=compare(load(a.storyboard)); text=json.dumps(result,indent=2)
    if a.output: Path(a.output).write_text(text,encoding='utf-8')
    print(text); raise SystemExit(0 if result['status']=='VALID' else 1)
if __name__=='__main__': main()
