#!/usr/bin/env python3
"""Compile a Kubrick brief into validated private, storyboard, and provider artifacts."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from diagnostics import abort, diagnostic
from runtime_identity import compile_identity
try:
    import yaml
except ImportError:
    abort(diagnostic(status="DEPENDENCY_UNAVAILABLE",code="PY_YAML_REQUIRED",exit_code=3,message="PyYAML is required for compile; install the validation runtime profile",context={"package":"PyYAML","profile":"validation"}))
ROOT=Path(__file__).resolve().parent.parent
PY=sys.executable
RUN_IDENTITY={}

def read(path):
    p=Path(path); return json.loads(p.read_text(encoding="utf-8")) if p.suffix==".json" else yaml.safe_load(p.read_text(encoding="utf-8")) or {}

def write(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(yaml.safe_dump(data,sort_keys=False),encoding="utf-8")

def run(cmd): return subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)

def fail(out,stage,reasons,diagnostic_output=None):
    payload=diagnostic(status="NOT_COMPUTABLE",code="COMPILE_NOT_COMPUTABLE",exit_code=4,message=f"compile stopped at {stage}",reason_vector=reasons,context={"stage":stage,"implementation_output":diagnostic_output[-2000:] if diagnostic_output else ""})
    final={"status":"NOT_COMPUTABLE","compiler_version":"0.3.0",**RUN_IDENTITY,"stage":stage,"reason_vector":reasons,"diagnostic":payload}
    (out/"compile-receipt.json").write_text(json.dumps(final,indent=2),encoding="utf-8"); print(json.dumps(final,indent=2)); raise SystemExit(4)

def validate(out,artifact,schema,label):
    report=out/f"schema-{label}.json"; result=run([PY,str(ROOT/"scripts/validate_artifact.py"),"--artifact",str(artifact),"--schema",str(ROOT/schema),"--output",str(report)])
    if result.returncode!=0: fail(out,"schema-validation",[f"{label.upper()}_SCHEMA_INVALID"],result.stdout or result.stderr)
    return report.name

def complete_surface(brief):
    supplied=brief.get("surface_output") or {}
    convergence=[brief.get("convergence_effect","relation becomes materially visible")]
    return {
        "audience_prompt": supplied.get("audience_prompt",brief.get("audience_prompt","")),
        "geometry": supplied.get("geometry",brief.get("geometry",[])),
        "light": supplied.get("light",brief.get("light",brief.get("lighting",[]))),
        "material": supplied.get("material",brief.get("material",brief.get("materials",[]))),
        "state_differentials": supplied.get("state_differentials",brief.get("state_differentials",[])),
        "convergence": supplied.get("convergence",convergence),
        "residue": supplied.get("residue",brief.get("residue",[])),
    }

def build_structured_packet(brief,graph,selected):
    channels=brief.get("symbolic_channels") or {"diegetic":brief.get("diegetic_channel",[]),"dramaturgical":brief.get("dramaturgical_channel",[]),"cinematic":brief.get("cinematic_channel",[])}
    motifs=brief.get("motifs",[])
    if not motifs:
        motifs=[{"motif_id":selected,"dramatic_function":brief.get("dramatic_problem","transform pressure"),"observed_form":node.get("observed_form",""),"recurrences":[{"state":node.get("initial_state","present"),"mutation":node.get("target_state","transformed"),"consequence":edge.get("transformation","") if graph.get("edges") else "state changes"}]} for node,edge in zip(graph.get("nodes",[]),graph.get("edges",[])+[{}])]
    cultural=brief.get("cultural_sources",[])
    if not cultural and brief.get("cultural_context"): cultural=[{"source":brief["cultural_context"],"boundary":brief.get("cultural_boundary","project-specific transfer only; no universal equivalence")}]
    return {"dramatic_function":brief.get("dramatic_problem","transform pressure"),"causal_actions":brief.get("causal_actions") or [e.get("transformation","") for e in graph.get("edges",[]) if e.get("transformation")],"motifs":motifs,"channels":channels,"convergence_sites":[{"site_id":s.get("site_id"),"functions":s.get("functions") or [s.get("observable_effect",""),brief.get("desired_state_change","transformed")]} for s in graph.get("convergence_sites",[])],"cultural_sources":cultural,"interpretation_claims":brief.get("interpretation_claims",[]),"production_constraints":brief.get("production_constraints",[])}

def main():
    global RUN_IDENTITY
    p=argparse.ArgumentParser(); p.add_argument("--brief",required=True); p.add_argument("--ledger"); p.add_argument("--mode",choices=["single-frame","scene","storyboard","diagnostic"],default="single-frame"); p.add_argument("--storyboard-plan"); p.add_argument("--provider",choices=["none","generic","grok-imagine","flux","sd3","midjourney"],default="none"); p.add_argument("--out",required=True); a=p.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True); brief=read(a.brief)
    if a.ledger: brief["symbolic_ledger"]=read(a.ledger)
    identity_input={"brief":brief,"storyboard_plan":read(a.storyboard_plan) if a.storyboard_plan else None}
    RUN_IDENTITY=compile_identity(identity_input,mode=a.mode,provider=a.provider)
    normalized=out/"brief.normalized.yaml"; write(normalized,brief)
    retrieval=run([PY,str(ROOT/"scripts/retrieve_symbolic_patterns_registry.py"),"--brief",str(normalized),"--no-cache"])
    try: receipt=yaml.safe_load(retrieval.stdout) or {}
    except Exception: receipt={}
    write(out/"retrieval-receipt.yaml",receipt); selected=(receipt.get("retrieval_receipt") or {}).get("selected_primary_grammar")
    if retrieval.returncode!=0 or not selected: fail(out,"retrieval",(receipt.get("retrieval_receipt") or {}).get("reason_vector",["RETRIEVAL_FAILED"]))
    forms=brief.get("observed_forms") or []
    if len(forms)<2: forms=[{"id":f"observed_{i+1}","kind":"motif","observed_form":str(v),"initial_state":"present","target_state":brief.get("desired_state_change","mutated"),"provenance_label":"OBSERVED","pattern_links":[selected]} for i,v in enumerate((brief.get("observable_evidence") or [])[:3])]
    if len(forms)<2: fail(out,"graph",["INSUFFICIENT_OBSERVED_FORMS"])
    relations=brief.get("relations") or [{"source":forms[0].get("id","observed_1"),"target":forms[1].get("id","observed_2"),"relation":"opposes","pressure":0.7,"transformation":brief.get("desired_state_change","relation changes")}]
    graph_spec={"graph_id":hashlib.sha256(json.dumps(brief,sort_keys=True).encode()).hexdigest()[:16],"symbolic_intent":{"dramatic_function":brief.get("dramatic_problem","transform pressure"),"emotional_force":brief.get("character_pressure","pressure"),"desired_state_change":brief.get("desired_state_change","transformed")},"observed_forms":forms,"relations":relations,"layers":brief.get("layers",{"layout_geometry":brief.get("geometry",[]),"semantics_function":[brief.get("dramatic_problem","")],"attributes_states":brief.get("state_differentials",[])}),"convergence_sites":brief.get("convergence_sites",[{"site_id":"primary","node_ids":[forms[0].get("id"),forms[1].get("id")],"edge_ids":[0],"observable_effect":brief.get("convergence_effect","relation becomes materially visible"),"mask_priority":0.9}]),"residue":brief.get("residue",[]),"surface_output":complete_surface(brief)}
    graph_input=out/"graph-input.yaml"; write(graph_input,graph_spec); graph_path=out/"motif-graph.private.yaml"
    graph_run=run([PY,str(ROOT/"scripts/build_motif_graph.py"),"--input",str(graph_input),"--output",str(graph_path)])
    if graph_run.returncode!=0: fail(out,"graph",["GRAPH_INVALID"],graph_run.stderr or graph_run.stdout)
    schema_reports={"graph":validate(out,graph_path,"schemas/motif-structure-graph.schema.yaml","graph")}; graph=read(graph_path)
    structured_path=out/"structured-symbolic-packet.yaml"; write(structured_path,build_structured_packet(brief,graph,selected))
    structured=run([PY,str(ROOT/"scripts/audit_symbolic_structure.py"),"--input",str(structured_path),"--output",str(out/"structured-anti-slop-report.json")])
    if structured.returncode!=0: fail(out,"structured-audit",["STRUCTURED_ANTI_SLOP_FAILED"],structured.stdout)
    translate=run([PY,str(ROOT/"scripts/translate_motif_graph.py"),"--graph",str(graph_path),"--mode",a.mode])
    try: audience=yaml.safe_load(translate.stdout) or {}
    except Exception: audience={}
    write(out/"audience-constraints.yaml",audience); text_audit=run([PY,str(ROOT/"scripts/audit_anti_slop.py"),"--text",yaml.safe_dump(audience),"--json"])
    try: text_data=json.loads(text_audit.stdout)
    except Exception: text_data={"status":"FAIL","violations":[{"gate":"UNKNOWN","repair":"inspect audit output"}]}
    (out/"text-anti-slop-report.json").write_text(json.dumps(text_data,indent=2),encoding="utf-8")
    if translate.returncode!=0 or text_audit.returncode!=0: fail(out,"translation",["TRANSLATION_OR_TEXT_AUDIT_FAILED"],translate.stderr or text_audit.stdout)
    artifacts={"retrieval":"retrieval-receipt.yaml","private_graph":graph_path.name,"structured_packet":structured_path.name,"structured_audit":"structured-anti-slop-report.json","audience":"audience-constraints.yaml","text_audit":"text-anti-slop-report.json"}; storyboard_path=None
    if a.mode=="storyboard":
        if not a.storyboard_plan: fail(out,"storyboard",["STORYBOARD_PLAN_REQUIRED"])
        storyboard_path=out/"storyboard-symbolic-state.yaml"; prop=run([PY,str(ROOT/"scripts/propagate_graph_state.py"),"--graph",str(graph_path),"--plan",a.storyboard_plan,"--output",str(storyboard_path)])
        if prop.returncode!=0: fail(out,"storyboard",["STORYBOARD_PROPAGATION_FAILED"],prop.stdout or prop.stderr)
        transition_path=out/"storyboard-transition-report.json"; comp=run([PY,str(ROOT/"scripts/compare_frame_state.py"),"--storyboard",str(storyboard_path),"--output",str(transition_path)])
        if comp.returncode!=0: fail(out,"storyboard",["STORYBOARD_CONTINUITY_FAILED"],comp.stdout or comp.stderr)
        schema_reports["storyboard"]=validate(out,storyboard_path,"schemas/storyboard-symbolic-state.schema.yaml","storyboard"); artifacts.update({"storyboard_state":storyboard_path.name,"storyboard_transitions":transition_path.name})
    if a.provider!="none":
        adapter_path=out/"model-adapter-packet.yaml"; cmd=[PY,str(ROOT/"scripts/build_model_adapter_packet.py"),"--graph",str(graph_path),"--provider",a.provider,"--output",str(adapter_path)]
        if storyboard_path: cmd.extend(["--storyboard",str(storyboard_path)])
        adapter=run(cmd)
        if adapter.returncode!=0: fail(out,"adapter",["MODEL_ADAPTER_BUILD_FAILED"],adapter.stdout or adapter.stderr)
        schema_reports["adapter"]=validate(out,adapter_path,"schemas/model-adapter-packet.schema.yaml","adapter"); artifacts["model_adapter"]=adapter_path.name
        if a.provider in {"grok-imagine","flux","sd3","midjourney"}:
            provider_path=out/f"{a.provider}-prompt-packet.yaml"
            if a.provider=="grok-imagine":
                provider=run([PY,str(ROOT/"scripts/adapt_grok_imagine.py"),"--packet",str(adapter_path),"--output",str(provider_path)])
            else:
                provider=run([PY,str(ROOT/"scripts/adapt_provider.py"),"--packet",str(adapter_path),"--provider",a.provider,"--output",str(provider_path)])
            if provider.returncode!=0: fail(out,"adapter",[f"{a.provider.upper().replace('-','_')}_ADAPTER_FAILED"],provider.stdout or provider.stderr)
            artifacts["provider_packet"]=provider_path.name
    final={"status":"COMPILED","compiler_version":"0.3.0",**RUN_IDENTITY,"timestamp":datetime.now(timezone.utc).isoformat(),"mode":a.mode,"provider":a.provider,"selected_primary":selected,"artifacts":artifacts,"schema_reports":schema_reports,"reason_vector":[]}
    (out/"compile-receipt.json").write_text(json.dumps(final,indent=2),encoding="utf-8"); print(json.dumps(final,indent=2))

if __name__=="__main__": main()
