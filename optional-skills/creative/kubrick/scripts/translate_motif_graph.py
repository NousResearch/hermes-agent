#!/usr/bin/env python3
"""Translate a validated latent motif graph into observable Hermes output constraints."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any, Dict
try:
    import yaml
except ImportError:
    print("pyyaml required. pip install pyyaml", file=sys.stderr); raise SystemExit(1)
def load(path:Path)->Dict[str,Any]:
    with path.open("r",encoding="utf-8") as h:return json.load(h) if path.suffix==".json" else yaml.safe_load(h) or {}
def translate(graph:Dict[str,Any], mode:str)->Dict[str,Any]:
    validation=graph.get("validation",{})
    if validation.get("status")!="VALID":
        return {"status":"NOT_COMPUTABLE","reason_vector":["GRAPH_NOT_VALID"],"validation":validation}
    nodes={n["id"]:n for n in graph.get("nodes",[])}; edges=graph.get("edges",[]); sites=sorted(graph.get("convergence_sites",[]),key=lambda s:-float(s.get("mask_priority",0)))
    geometry=list(graph.get("layers",{}).get("layout_geometry",[])); semantics=list(graph.get("layers",{}).get("semantics_function",[])); attributes=list(graph.get("layers",{}).get("attributes_states",[]))
    convergence=[]
    for site in sites:
        forms=[nodes[n]["observed_form"] for n in site.get("node_ids",[]) if n in nodes]
        convergence.append(f"{site.get('observable_effect')}: " + " / ".join(forms))
    state_diffs=[f"{n['observed_form']}: {n['initial_state']} -> {n['target_state']}" for n in nodes.values()]
    light=[x for x in attributes if any(k in x.lower() for k in ("light","shadow","glow","exposure","color"))]
    material=[x for x in attributes if x not in light]
    prompt_parts=[]
    if geometry: prompt_parts.append("Geometry: "+"; ".join(geometry))
    if light: prompt_parts.append("Light: "+"; ".join(light))
    if material: prompt_parts.append("Material/state: "+"; ".join(material))
    if convergence: prompt_parts.append("Convergence: "+"; ".join(convergence))
    if graph.get("residue"): prompt_parts.append("Residue: "+"; ".join(graph["residue"]))
    output={"status":"TRANSLATED","mode":mode,"graph_id":graph.get("graph_id"),"audience_prompt":". ".join(prompt_parts),"geometry":geometry,"light":light,"material":material,"state_differentials":state_diffs,"convergence":convergence,"residue":graph.get("residue",[]),"private_semantics":semantics if mode=="diagnostic" else []}
    return output
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--graph",required=True);p.add_argument("--mode",choices=["single-frame","scene","storyboard","diagnostic"],default="single-frame");p.add_argument("--output");a=p.parse_args();o=translate(load(Path(a.graph)),a.mode);text=yaml.safe_dump(o,sort_keys=False)
    if a.output: Path(a.output).write_text(text,encoding="utf-8")
    else: print(text)
    raise SystemExit(0 if o["status"]=="TRANSLATED" else 1)
if __name__=="__main__":main()
