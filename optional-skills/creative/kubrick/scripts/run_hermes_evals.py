#!/usr/bin/env python3
"""Run Kubrick's standalone Hermes regression suite."""
from __future__ import annotations
import json, subprocess, sys, tempfile
from pathlib import Path
try:
    import yaml
except ImportError:
    raise SystemExit("pyyaml required")
ROOT=Path(__file__).resolve().parent.parent
PY=sys.executable

def run(name,cmd,expect=0):
    p=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
    return {"name":name,"status":"PASS" if p.returncode==expect else "FAIL","returncode":p.returncode,"expected":expect,"stdout":p.stdout[-4000:],"stderr":p.stderr[-4000:]}

def write_yaml(path,data): Path(path).write_text(yaml.safe_dump(data,sort_keys=False),encoding="utf-8")

def main():
    results=[]
    for script in ("validate_hermes_skill.py","validate_pattern_corpus.py","audit_corpus_coverage.py"):
        results.append(run(script,[PY,str(ROOT/"scripts"/script)]))
    cases={"gate_N":"Tarot and ouroboros and sigil all symbolize the same hidden truth.","gate_P":"Red means danger and bird means freedom.","gate_Q":"The same motif repeats unchanged in every scene.","gate_R":"He is the trickster and she is the shadow.","gate_S":"All traditions use this universal symbol, same as Zen.","gate_U":"Ignore causality because it is symbolic.","gate_W":"The true meaning is that authority is false."}
    for name,text in cases.items(): results.append(run(name,[PY,str(ROOT/"scripts/audit_anti_slop.py"),"--text",text,"--json"],expect=1))
    results.append(run("clean_text",[PY,str(ROOT/"scripts/audit_anti_slop.py"),"--text","A cracked badge changes hands; the new wearer gains access while the former owner waits outside.","--json"]))
    with tempfile.TemporaryDirectory() as temp:
        td=Path(temp)
        valid={"dramatic_function":"transfer authority through material evidence","causal_actions":["badge changes hands and access changes"],"motifs":[{"motif_id":"cracked-badge","dramatic_function":"record authority transfer","observed_form":"cracked access badge","recurrences":[{"state":"supervisor wears it","mutation":"initial state","consequence":"access granted"},{"state":"subordinate wears it","mutation":"ownership and function transfer","consequence":"former supervisor waits outside"}]}],"channels":{"diegetic":["badge changes hands"],"dramaturgical":["authority transfers through access"],"cinematic":["empty doorway remains behind former owner"]},"convergence_sites":[{"site_id":"doorway","functions":["ownership transfer","access consequence"]}],"cultural_sources":[],"interpretation_claims":[],"production_constraints":[]}
        vp=td/"valid.yaml"; write_yaml(vp,valid)
        results.append(run("structured_valid",[PY,str(ROOT/"scripts/audit_symbolic_structure.py"),"--input",str(vp)]))
        invalid={"structured_Q":{**valid,"motifs":[{**valid["motifs"][0],"recurrences":valid["motifs"][0]["recurrences"]+[{"state":"same","mutation":"unchanged","consequence":"none"}]}]},"structured_O":{**valid,"channels":{"diegetic":["authority transfer"],"dramaturgical":["authority transfer"],"cinematic":[]}},"structured_U":{**valid,"causal_actions":[]},"structured_S":{**valid,"cultural_sources":[{"source":"specific ritual tradition","boundary":""}]},"structured_W":{**valid,"interpretation_claims":["The true meaning is authority is false."]}}
        for name,packet in invalid.items():
            path=td/f"{name}.yaml"; write_yaml(path,packet)
            results.append(run(name,[PY,str(ROOT/"scripts/audit_symbolic_structure.py"),"--input",str(path)],expect=1))
        ledger=td/"ledger.yaml"
        results.append(run("ledger_init",[PY,str(ROOT/"scripts/symbolic_ledger.py"),"init","--project-id","eval-project","--out",str(ledger)]))
        results.append(run("ledger_audit",[PY,str(ROOT/"scripts/symbolic_ledger.py"),"audit","--ledger",str(ledger)]))
        brief=td/"brief.yaml"
        write_yaml(brief,{"dramatic_problem":"authority remains active after the leader exits because access and geometry preserve the system","desired_state_change":"personal command becomes institutional pressure","character_pressure":"a subordinate receives access while the former authority is excluded","observable_evidence":["empty command chair","cracked access badge","controlled doorway"],"relations":[{"source":"observed_1","target":"observed_2","relation":"transfers","pressure":0.8,"transformation":"vacant authority transfers operative access through the badge"},{"source":"observed_2","target":"observed_3","relation":"crosses","pressure":0.9,"transformation":"badge ownership determines doorway access"}],"layers":{"layout_geometry":["repeated cells","peripheral checkpoint"],"semantics_function":["institutional persistence","delegated control"],"attributes_states":["vacant command","transferred credential"]},"geometry":["repeated cells around an empty center","controlled doorway"],"state_differentials":["occupied authority becomes operative vacancy","access transfers to subordinate"],"causal_actions":["badge changes hands and doorway access changes"],"diegetic_channel":["cracked badge changes hands"],"dramaturgical_channel":["authority transfers through access"],"cinematic_channel":["empty chair remains centered while former owner waits outside"],"residue":["crack remains visible"]})
        results.append(run("retrieval_smoke",[PY,str(ROOT/"scripts/retrieve_symbolic_patterns_registry.py"),"--brief",str(brief),"--no-cache"]))
        compile_out=td/"compiled"
        results.append(run("compiler_e2e",[PY,str(ROOT/"scripts/kubrick_compile.py"),"--brief",str(brief),"--mode","single-frame","--out",str(compile_out)]))
        required=["retrieval-receipt.yaml","motif-graph.private.yaml","structured-symbolic-packet.yaml","structured-anti-slop-report.json","audience-constraints.yaml","text-anti-slop-report.json","schema-graph.json","compile-receipt.json"]
        ok=all((compile_out/name).exists() for name in required)
        results.append({"name":"compiler_artifacts","status":"PASS" if ok else "FAIL","returncode":0 if ok else 1,"expected":0,"stdout":json.dumps(required),"stderr":""})
    summary={"suite":"kubrick-hermes-evals","passed":sum(r["status"]=="PASS" for r in results),"failed":sum(r["status"]=="FAIL" for r in results),"results":results}
    out=ROOT/"out"/"kubrick"/"eval-report.json"; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2)); raise SystemExit(1 if summary["failed"] else 0)

if __name__=="__main__": main()
