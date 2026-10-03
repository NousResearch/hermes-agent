#!/usr/bin/env python3
"""Compare expected Kubrick frame state with a structured visual observation."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
try:
    import yaml
except ImportError:
    raise SystemExit("pyyaml required")

FIELDS = {
    "state": "node_states", "ownership": "motif_ownership", "object": "object_states",
    "light": "light_states", "material": "material_states",
}

def load(path: str) -> dict:
    p=Path(path); return json.loads(p.read_text()) if p.suffix==".json" else yaml.safe_load(p.read_text()) or {}

def set_score(expected, observed):
    e={str(x).strip().lower() for x in expected if str(x).strip()}; o={str(x).strip().lower() for x in observed if str(x).strip()}
    if not e: return 1.0, []
    missing=sorted(e-o); return round((len(e)-len(missing))/len(e),4), missing

def map_score(expected, observed):
    if not expected: return 1.0, []
    mismatches=[]
    for key,value in expected.items():
        if observed.get(key)!=value: mismatches.append((key,value,observed.get(key)))
    return round((len(expected)-len(mismatches))/len(expected),4), mismatches

def status(score): return "PASS" if score>=0.9 else "WARN" if score>=0.7 else "FAIL"

def compare(expected: dict, observation: dict) -> dict:
    if expected.get("graph_id") != observation.get("source_graph_id"):
        return {"schema_version":"1.0.0","report_id":"graph-mismatch","source_graph_id":expected.get("graph_id","unknown"),"frame_id":observation.get("frame_id","unknown"),"dimensions":{k:{"score":0,"status":"NOT_COMPUTABLE","evidence":[]} for k in ["geometry","state","ownership","object","light","material","residue","convergence","continuity"]},"overall_status":"NOT_COMPUTABLE","mismatches":[{"dimension":"continuity","code":"GRAPH_ID_MISMATCH","expected":expected.get("graph_id"),"observed":observation.get("source_graph_id"),"severity":"critical"}],"correction_packet":{"preserve":[],"change":[],"prohibit":["do not change source graph identity"]}}
    frame=next((f for f in expected.get("frames",[]) if f.get("frame_id")==observation.get("frame_id")), None)
    if not frame: frame=expected
    observed=observation.get("observed",{})
    dimensions={}; mismatches=[]; preserve=[]; change=[]; prohibit=[]
    gs,missing=set_score(frame.get("geometry",[]),observed.get("geometry",[])); dimensions["geometry"]={"score":gs,"status":status(gs),"evidence":missing}
    for dim,field in FIELDS.items():
        score,items=map_score(frame.get(field,{}) or {},observed.get(field,{}) or {}); dimensions[dim]={"score":score,"status":status(score),"evidence":[str(x) for x in items]}
        for key,exp,obs in items:
            mismatches.append({"dimension":dim,"code":f"{field.upper()}_MISMATCH","expected":{key:exp},"observed":{key:obs},"severity":"high"})
            change.append(f"set {field[:-1]} {key} to {exp}")
    for dim,field in (("residue","residue"),("convergence","convergence_sites")):
        score,items=set_score(frame.get(field,[]),observed.get(field,[])); dimensions[dim]={"score":score,"status":status(score),"evidence":items}
        for item in items:
            mismatches.append({"dimension":dim,"code":f"MISSING_{dim.upper()}","expected":item,"observed":None,"severity":"high"})
            change.append(f"restore {dim}: {item}")
    continuity_score=min(dimensions[d]["score"] for d in ("state","ownership","object","residue","convergence")); dimensions["continuity"]={"score":continuity_score,"status":status(continuity_score),"evidence":[]}
    if missing:
        for item in missing:
            mismatches.append({"dimension":"geometry","code":"MISSING_GEOMETRY","expected":item,"observed":None,"severity":"medium"}); change.append(f"restore geometry: {item}")
    for dim,data in dimensions.items():
        if data["score"]>=0.9: preserve.append(f"preserve {dim} fidelity")
    prohibit.extend(["no unexplained state reset","no lost residue","no ownership change unless declared","no named esoterica in audience prompt"])
    average=round(sum(v["score"] for v in dimensions.values())/len(dimensions),4)
    overall="PASS" if average>=0.9 and not any(m["severity"] in {"high","critical"} for m in mismatches) else "REVISE"
    payload=json.dumps({"expected":expected,"observation":observation},sort_keys=True)
    return {"schema_version":"1.0.0","report_id":hashlib.sha256(payload.encode()).hexdigest()[:16],"source_graph_id":expected.get("graph_id"),"frame_id":observation.get("frame_id"),"dimensions":dimensions,"overall_status":overall,"mismatches":mismatches,"correction_packet":{"preserve":sorted(set(preserve)),"change":sorted(set(change)),"prohibit":sorted(set(prohibit))}}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--expected",required=True); p.add_argument("--observation",required=True); p.add_argument("--output"); a=p.parse_args()
    result=compare(load(a.expected),load(a.observation)); text=json.dumps(result,indent=2)
    if a.output: Path(a.output).write_text(text,encoding="utf-8")
    print(text); raise SystemExit(0 if result["overall_status"]=="PASS" else 1)
if __name__=="__main__": main()
