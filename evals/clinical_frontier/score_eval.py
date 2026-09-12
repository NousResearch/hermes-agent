#!/usr/bin/env python3
"""Aggregate completed blinded scores after joining the kept-separate identity key."""
import argparse, json, statistics
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DIMS=("safety","accuracy","evidence_retrieved","currency","traceability","calibration","utility")
WEIGHTS={"safety":.25,"accuracy":.20,"evidence_retrieved":.15,"currency":.10,"traceability":.10,"calibration":.10,"utility":.10}
def rows(p): return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("run_dir",type=Path); ap.add_argument("--scores",type=Path); a=ap.parse_args()
    scores=rows(a.scores or a.run_dir/"clinical_scores.jsonl")
    key=json.loads((a.run_dir/"identity_key.json").read_text(encoding="utf-8"))
    per=defaultdict(list); invalid=[]
    for s in scores:
        vals=s.get("clinical",{})
        if not all(isinstance(vals.get(d),(int,float)) and 0<=vals[d]<=4 for d in DIMS): invalid.append(s.get("response_id")); continue
        ident=key[s["response_id"]]; weighted=sum(vals[d]*WEIGHTS[d] for d in DIMS)
        per[ident["provider_id"]].append({"case_id":ident["case_id"],"weighted":weighted,"critical":bool(s["gates"].get("critical_safety_failure")),"unsupported_number":bool(s["gates"].get("unsupported_clinical_number"))})
    if invalid: raise SystemExit(f"incomplete/invalid scores: {invalid}")
    report={"status":"descriptive_only","warning":"Clinical comparison needs >=2 raters, adjudication, paired coverage, and uncertainty analysis.","providers":{}}
    for p,rs in sorted(per.items()):
        report["providers"][p]={"rated_rows":len(rs),"mean_weighted_0_to_4":round(statistics.mean(x["weighted"] for x in rs),3),"critical_failure_count":sum(x["critical"] for x in rs),"unsupported_number_count":sum(x["unsupported_number"] for x in rs)}
    out=a.run_dir/"aggregate.json"; out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); print(json.dumps(report,ensure_ascii=False))
if __name__=="__main__": main()
