#!/usr/bin/env python3
"""Compile the canonical example twice and compare stable artifact hashes."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
PY=sys.executable
VOLATILE_KEYS={"timestamp","updated_at","last_evolved","cached_to","logged_to"}


def load(path:Path):
    import yaml
    return json.loads(path.read_text(encoding="utf-8")) if path.suffix==".json" else yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def stable(value):
    if isinstance(value,dict): return {k:stable(v) for k,v in sorted(value.items()) if k not in VOLATILE_KEYS}
    if isinstance(value,list): return [stable(v) for v in value]
    return value


def digest(path:Path)->str:
    payload=json.dumps(stable(load(path)),sort_keys=True,separators=(",",":"),default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def main():
    p=argparse.ArgumentParser(); p.add_argument("--output"); a=p.parse_args()
    artifacts=["compile-receipt.json","motif-graph.private.yaml","structured-symbolic-packet.yaml","storyboard-symbolic-state.yaml","storyboard-transition-report.json","model-adapter-packet.yaml","grok-imagine-prompt-packet.yaml"]
    with tempfile.TemporaryDirectory() as d:
        root=Path(d); runs=[]
        for i in (1,2):
            out=root/f"run-{i}"
            cmd=[PY,str(ROOT/"scripts/kubrick.py"),"compile","--brief",str(ROOT/"examples/authority-transfer-storyboard/brief.yaml"),"--ledger",str(ROOT/"examples/authority-transfer-storyboard/symbolic-ledger.yaml"),"--mode","storyboard","--storyboard-plan",str(ROOT/"examples/authority-transfer-storyboard/storyboard-plan.yaml"),"--provider","grok-imagine","--out",str(out)]
            result=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
            if result.returncode: raise SystemExit(result.stdout+"\n"+result.stderr)
            runs.append(out)
        comparisons={name:{"run_1":digest(runs[0]/name),"run_2":digest(runs[1]/name)} for name in artifacts}
        mismatches=[name for name,data in comparisons.items() if data["run_1"]!=data["run_2"]]
        report={"status":"PASS" if not mismatches else "FAIL","artifacts":comparisons,"mismatches":mismatches}
        text=json.dumps(report,indent=2)
        if a.output:
            out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(text,encoding="utf-8")
        print(text); raise SystemExit(0 if not mismatches else 1)

if __name__=="__main__": main()
