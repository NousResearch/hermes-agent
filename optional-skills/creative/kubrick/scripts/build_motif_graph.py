#!/usr/bin/env python3
"""Build and validate Kubrick's latent motif/structure graph IR."""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
from typing import Any, Dict, List
try:
    import yaml
except ImportError:
    print("pyyaml required. pip install pyyaml", file=sys.stderr); raise SystemExit(1)
LAYER_KEYS=("layout_geometry","semantics_function","attributes_states")
NAMED_ESOTERICA={"nigredo","albedo","rubedo","athanor","rebis","ouroboros","sephirot","tarot","qabalah","kabbalah","hermetic","alchemy","alchemical","syzygy","egregore"}
def load(path:Path)->Dict[str,Any]:
    with path.open("r",encoding="utf-8") as h:return json.load(h) if path.suffix==".json" else yaml.safe_load(h) or {}
def tokens(value:Any)->set[str]:return set(re.findall(r"[a-z0-9_]+",json.dumps(value).lower()))
def empty_surface()->Dict[str,Any]:
    return {"audience_prompt":"","geometry":[],"light":[],"material":[],"state_differentials":[],"convergence":[],"residue":[]}
def validate(graph:Dict[str,Any])->Dict[str,Any]:
    errors:List[str]=[]; unknown:List[str]=[]; nodes=graph.get("nodes",[]); edges=graph.get("edges",[]); sites=graph.get("convergence_sites",[]); layers=graph.get("layers",{})
    node_ids={n.get("id") for n in nodes}
    for i,e in enumerate(edges):
        if e.get("source") not in node_ids: unknown.append(f"edge[{i}].source:{e.get('source')}")
        if e.get("target") not in node_ids: unknown.append(f"edge[{i}].target:{e.get('target')}")
    for s in sites:
        for n in s.get("node_ids",[]):
            if n not in node_ids: unknown.append(f"site:{s.get('site_id')}.node:{n}")
        for e in s.get("edge_ids",[]):
            if not isinstance(e,int) or e<0 or e>=len(edges): unknown.append(f"site:{s.get('site_id')}.edge:{e}")
    if unknown: errors.append("unknown graph references")
    if not 1<=len(sites)<=2: errors.append("exactly one or two convergence sites are required")
    for s in sites:
        if len(set(s.get("node_ids",[])))<2 or len(set(s.get("edge_ids",[])))<1: errors.append(f"convergence site {s.get('site_id')} lacks relational density")
    leakage=[]; layer_tokens={k:tokens(layers.get(k,[])) for k in LAYER_KEYS}
    for i,left in enumerate(LAYER_KEYS):
        for right in LAYER_KEYS[i+1:]:
            for token in sorted((layer_tokens[left]&layer_tokens[right])-{"light","object","state","space","character"}): leakage.append(f"{token}:{left}<->{right}")
    if leakage: errors.append("attribute leakage detected across disentangled layers")
    surface=graph.get("surface_output",{}); named=sorted(tokens(surface)&NAMED_ESOTERICA)
    if named: errors.append("named esoterica leaked into observable output")
    density=round(len(edges)/max(1,len(nodes)),4)
    if density<0.5: errors.append("graph edge density below 0.5")
    graph["validation"]={"attribute_leakage":leakage,"convergence_density":density,"named_esoterica_surface":named,"unknown_references":unknown,"errors":errors,"status":"VALID" if not errors else "INVALID"}
    return graph
def build(spec:Dict[str,Any])->Dict[str,Any]:
    intent=spec.get("symbolic_intent",{}); observed=spec.get("observed_forms",[])
    nodes=[{"id":x.get("id",f"node_{i+1}"),"kind":x.get("kind","motif"),"observed_form":x.get("observed_form",x.get("form","")),"initial_state":x.get("initial_state","unresolved"),"target_state":x.get("target_state",intent.get("desired_state_change","transformed")),"provenance_label":x.get("provenance_label","OBSERVED"),"lexicon_links":x.get("lexicon_links",[]),"pattern_links":x.get("pattern_links",[])} for i,x in enumerate(observed)]
    edges=[{"source":r["source"],"target":r["target"],"relation":r.get("relation","opposes"),"pressure":float(r.get("pressure",0.5)),"transformation":r.get("transformation","relation changes under pressure")} for r in spec.get("relations",[])]
    graph={"schema_version":"1.1.0","graph_id":spec.get("graph_id","kubrick-graph"),"intent":{"dramatic_function":intent.get("dramatic_function","transform"),"emotional_force":intent.get("emotional_force","pressure"),"desired_state_change":intent.get("desired_state_change","transformed")},"nodes":nodes,"edges":edges,"layers":spec.get("layers",{k:[] for k in LAYER_KEYS}),"convergence_sites":spec.get("convergence_sites",[]),"residue":spec.get("residue",[]),"surface_output":spec.get("surface_output",empty_surface()),"validation":{}}
    return validate(graph)
def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--output");a=p.parse_args();g=build(load(Path(a.input)));out=yaml.safe_dump(g,sort_keys=False)
    if a.output: Path(a.output).write_text(out,encoding="utf-8")
    else: print(out)
    raise SystemExit(0 if g["validation"]["status"]=="VALID" else 1)
if __name__=="__main__":main()
