#!/usr/bin/env python3
"""Build a provider-neutral regeneration packet from a Kubrick fidelity report."""
from __future__ import annotations
import argparse, json
from pathlib import Path
try:
    import yaml
except ImportError:
    raise SystemExit("pyyaml required")

def load(path):
    p=Path(path); return json.loads(p.read_text()) if p.suffix==".json" else yaml.safe_load(p.read_text()) or {}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--report",required=True); p.add_argument("--adapter-packet"); p.add_argument("--output"); a=p.parse_args()
    report=load(a.report); adapter=load(a.adapter_packet) if a.adapter_packet else {}
    errors=[]
    if report.get("overall_status")=="NOT_COMPUTABLE": errors.append("fidelity report is NOT_COMPUTABLE")
    graph_id=report.get("source_graph_id")
    if adapter and adapter.get("source_graph_id")!=graph_id: errors.append("adapter and fidelity graph identities differ")
    correction=report.get("correction_packet",{})
    result={
      "schema_version":"1.0.0","source_graph_id":graph_id,"frame_id":report.get("frame_id"),
      "preserve":correction.get("preserve",[]),"change":correction.get("change",[]),"prohibit":correction.get("prohibit",[]),
      "provider":adapter.get("provider","generic"),"source_adapter_id":adapter.get("adapter_id"),
      "instruction":"Regenerate the same frame identity. Preserve all listed invariants; change only listed mismatches; obey all prohibitions.",
      "validation":{"status":"VALID" if not errors else "INVALID","errors":errors}
    }
    text=yaml.safe_dump(result,sort_keys=False)
    if a.output:
        out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(text,encoding="utf-8")
    else: print(text)
    raise SystemExit(0 if not errors else 1)
if __name__=="__main__": main()
