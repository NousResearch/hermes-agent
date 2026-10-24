#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess, sys, tempfile
from pathlib import Path
try:
    import yaml
except ImportError:
    raise SystemExit("pyyaml required")
ROOT=Path(__file__).resolve().parent.parent; PY=sys.executable

def run(*args,expect=0):
    p=subprocess.run([PY,*map(str,args)],cwd=ROOT,text=True,capture_output=True)
    if p.returncode!=expect: raise SystemExit(f"command failed {args}: {p.stdout}\n{p.stderr}")

def write(path,data): Path(path).write_text(yaml.safe_dump(data,sort_keys=False),encoding="utf-8")

def main():
    with tempfile.TemporaryDirectory() as d:
        td=Path(d); compile_receipt=td/"compile.json"; retrieval=td/"retrieval.yaml"; report1=td/"r1.json"; report2=td/"r2.json"
        compile_receipt.write_text(json.dumps({"status":"COMPILED","selected_primary":"doorway_ownership_transfer"}),encoding="utf-8")
        write(retrieval,{"retrieval_receipt":{"confidence":0.82,"selected_primary_grammar":"doorway_ownership_transfer","selected_supporting_grammars":[]}})
        dims1={k:{"score":0.8,"status":"WARN","evidence":[]} for k in ["geometry","state","ownership","object","light","material","residue","convergence","continuity"]}
        dims2={k:{"score":0.95,"status":"PASS","evidence":[]} for k in dims1}
        report1.write_text(json.dumps({"report_id":"r1","source_graph_id":"g1","frame_id":"f1","overall_status":"REVISE","dimensions":dims1}),encoding="utf-8")
        report2.write_text(json.dumps({"report_id":"r2","source_graph_id":"g1","frame_id":"f1","overall_status":"PASS","dimensions":dims2}),encoding="utf-8")
        receipt=td/"use.yaml"; proposal=td/"proposal.yaml"; iteration=td/"iteration.yaml"
        run(ROOT/"scripts/record_pattern_outcome.py","--compile-receipt",compile_receipt,"--retrieval-receipt",retrieval,"--fidelity-report",report2,"--project-id","smoke","--outcome","ACCEPTED","--output",receipt)
        run(ROOT/"scripts/propose_pattern_evolution.py","--pattern-id","doorway_ownership_transfer","--receipt",receipt,"--output",proposal)
        run(ROOT/"scripts/govern_correction_iteration.py","--previous",report1,"--current",report2,"--iteration","2","--output",iteration)
        run(ROOT/"scripts/validate_artifact.py","--artifact",receipt,"--schema",ROOT/"schemas/pattern-use-receipt.schema.yaml")
        run(ROOT/"scripts/validate_artifact.py","--artifact",proposal,"--schema",ROOT/"schemas/pattern-evolution-proposal.schema.yaml")
        run(ROOT/"scripts/validate_artifact.py","--artifact",iteration,"--schema",ROOT/"schemas/correction-iteration-receipt.schema.yaml")
    print("outcome governance smoke test: PASS")
if __name__=="__main__": main()
