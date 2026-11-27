#!/usr/bin/env python3
"""Assemble a self-contained Grok generation and review bundle without calling APIs."""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
try:
    import yaml
except ImportError:
    raise SystemExit("pyyaml required")

def load(path):
    p=Path(path); return json.loads(p.read_text(encoding="utf-8")) if p.suffix==".json" else yaml.safe_load(p.read_text(encoding="utf-8")) or {}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--grok-packet",required=True); p.add_argument("--expected-state",required=True); p.add_argument("--out",required=True); a=p.parse_args()
    grok=load(a.grok_packet); expected=load(a.expected_state)
    if grok.get("validation",{}).get("status")!="VALID": raise SystemExit("Grok packet is not VALID")
    graph_id=grok.get("source_graph_id")
    if expected.get("graph_id")!=graph_id: raise SystemExit("Grok packet and expected state graph identities differ")
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(a.grok_packet,out/"grok-imagine-prompt-packet.yaml"); shutil.copyfile(a.expected_state,out/"expected-frame-state.yaml")
    templates=[]
    for frame in expected.get("frames",[]):
        template={"schema_version":"1.0.0","observation_id":"FILL_AFTER_REVIEW","source_graph_id":graph_id,"frame_id":frame.get("frame_id"),"observed":{"geometry":[],"node_states":{},"motif_ownership":{},"object_states":{},"light_states":{},"material_states":{},"residue":[],"convergence_sites":[]},"confidence":0.0,"ambiguities":[],"provenance":{"observer":"operator","method":"manual","model":None}}
        path=out/f"{frame.get('frame_id')}-observation-template.yaml"; path.write_text(yaml.safe_dump(template,sort_keys=False),encoding="utf-8"); templates.append(path.name)
    manifest={"bundle_version":"1.0.0","source_graph_id":graph_id,"frame_count":len(expected.get("frames",[])),"generation_packet":"grok-imagine-prompt-packet.yaml","expected_state":"expected-frame-state.yaml","observation_templates":templates,"next_commands":["kubrick visual-compare --expected expected-frame-state.yaml --observation <filled-template> --output fidelity-report.json","kubrick visual-correct --report fidelity-report.json --adapter-packet ../model-adapter-packet.yaml --output correction-packet.yaml","kubrick correction-govern --current fidelity-report.json --iteration 1 --output iteration-receipt.yaml"]}
    (out/"workflow-manifest.yaml").write_text(yaml.safe_dump(manifest,sort_keys=False),encoding="utf-8"); print(yaml.safe_dump(manifest,sort_keys=False))
if __name__=="__main__": main()
