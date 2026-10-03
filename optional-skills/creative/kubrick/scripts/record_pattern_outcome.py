#!/usr/bin/env python3
"""Record proposal-only evidence from a Kubrick compile and optional QA reports."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
try:
    import yaml
except ImportError:
    raise SystemExit("pyyaml required")

def load(path):
    p=Path(path); return json.loads(p.read_text(encoding="utf-8")) if p.suffix==".json" else yaml.safe_load(p.read_text(encoding="utf-8")) or {}

def canonical(value): return json.dumps(value,sort_keys=True,separators=(",",":"),default=str)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--compile-receipt",required=True); p.add_argument("--retrieval-receipt"); p.add_argument("--fidelity-report",action="append",default=[]); p.add_argument("--project-id",required=True); p.add_argument("--outcome",choices=["ACCEPTED","REJECTED","PARTIAL","UNREVIEWED"],default="UNREVIEWED"); p.add_argument("--note",action="append",default=[]); p.add_argument("--production-feasibility",type=float); p.add_argument("--payoff-realized",choices=["true","false"]); p.add_argument("--cultural-boundary-respected",choices=["true","false"]); p.add_argument("--output",required=True); a=p.parse_args()
    compile_receipt=load(a.compile_receipt); retrieval=load(a.retrieval_receipt) if a.retrieval_receipt else {}
    rr=retrieval.get("retrieval_receipt",retrieval)
    fidelity=[load(x) for x in a.fidelity_report]
    scores=[]
    for report in fidelity:
        dims=report.get("dimensions",{})
        vals=[d.get("score") for d in dims.values() if isinstance(d,dict) and isinstance(d.get("score"),(int,float))]
        if vals: scores.append(sum(vals)/len(vals))
    selected_primary=compile_receipt.get("selected_primary") or rr.get("selected_primary_grammar")
    payload={"schema_version":"1.0.0","project_id":a.project_id,"source_compile_receipt":str(a.compile_receipt),"compile_status":compile_receipt.get("status","NOT_COMPUTABLE"),"selected_patterns":{"primary":selected_primary,"supporting":rr.get("selected_supporting_grammars",[])},"signals":{"retrieval_confidence":float(rr.get("confidence",0.0)),"graph_valid":compile_receipt.get("status")=="COMPILED","anti_slop_pass":compile_receipt.get("status")=="COMPILED","production_feasibility":a.production_feasibility,"visual_fidelity":round(sum(scores)/len(scores),4) if scores else None,"correction_iterations":len(fidelity),"payoff_realized":None if a.payoff_realized is None else a.payoff_realized=="true","cultural_boundary_respected":None if a.cultural_boundary_respected is None else a.cultural_boundary_respected=="true"},"outcome":{"status":a.outcome,"notes":a.note},"authority":{"state":"OBSERVATION","automatic_corpus_change_allowed":False}}
    payload["receipt_id"]=hashlib.sha256(canonical(payload).encode()).hexdigest()[:16]
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(yaml.safe_dump(payload,sort_keys=False),encoding="utf-8"); print(yaml.safe_dump(payload,sort_keys=False))
if __name__=="__main__": main()
