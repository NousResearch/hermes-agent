#!/usr/bin/env python3
"""Kubrick deterministic symbolic retrieval.

Ledger-aware, collision-aware, cacheable, and fail-closed.
"""
from __future__ import annotations
import argparse, hashlib, json, re, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
try:
    import yaml
except ImportError:
    print("pyyaml required. pip install pyyaml", file=sys.stderr); raise SystemExit(1)
SCRIPT_DIR=Path(__file__).resolve().parent
SKILL_ROOT=SCRIPT_DIR.parent
PATTERNS_DIR=SKILL_ROOT/"references"/"patterns"
CACHE_DIR=SKILL_ROOT/"references"/"usage"/"cache"
RECEIPT_DIR=SKILL_ROOT/"references"/"usage"/"receipts"
THRESHOLD=0.55
MAX_SUPPORTING=2
WEIGHTS={"dramatic_fit":0.24,"character_fit":0.10,"cinematic_fit":0.12,"cultural_fit":0.10,"source_quality":0.10,"mutation_potential":0.14,"continuity_compatibility":0.10,"payoff_compatibility":0.10}
COLLISION_TYPES={"REDUNDANT","CONTRADICTORY","CULTURALLY_INCOMPATIBLE","RHYTHMICALLY_OVERLAPPING","PAYOFF_COMPETITION"}

def canonical_json(value:Any)->str:
    return json.dumps(value,sort_keys=True,separators=(",",":"),default=lambda item:item.isoformat() if hasattr(item,"isoformat") else str(item))

def _tokens(value:Any)->set[str]:
    if isinstance(value,(list,tuple,set)): value=" ".join(str(i) for i in value)
    normalized=str(value).lower().replace("-"," ").replace("_"," ")
    return set(re.findall(r"[a-z0-9]+",normalized))

def _overlap(query:Any,candidate:Any)->float:
    q=_tokens(query); c=_tokens(candidate)
    return len(q&c)/max(1,len(q)) if q and c else 0.0

def _load(path:Path)->Dict[str,Any]:
    with path.open("r",encoding="utf-8") as h: return json.load(h) if path.suffix==".json" else yaml.safe_load(h) or {}

def load_all_patterns()->Dict[str,Dict[str,Any]]:
    patterns={}
    for path in PATTERNS_DIR.rglob("*.json"):
        pattern=_load(path); pid=pattern.get("pattern_id")
        if pid: patterns[pid]=pattern
    return patterns

def ledger_snapshot(brief:Dict[str,Any])->Dict[str,Any]:
    ledger=brief.get("symbolic_ledger") or {}; active=ledger.get("active_motifs",brief.get("active_project_motifs",[]))
    if active and isinstance(active[0] if isinstance(active,list) else None,dict): active=[x.get("motif_id","") for x in active]
    return {"active":sorted(set(active)),"retired":sorted(set(ledger.get("retired_motifs",[]))),"prohibited":sorted(set(ledger.get("prohibited_motifs",brief.get("prohibited_patterns",[])))),"unresolved_payoffs":sorted(set(ledger.get("unresolved_payoffs",[]))),"saturation_score":float(ledger.get("saturation_score",0.0)),"symbolic_debt":float(ledger.get("symbolic_debt",0.0)),"collisions":ledger.get("collisions",[]),"prohibited_cultural_scopes":ledger.get("prohibited_cultural_scopes",[])}

def cache_key(brief,ledger): return hashlib.sha256(canonical_json({"brief":brief,"ledger":ledger}).encode()).hexdigest()[:24]

def production_cost(pattern:Dict[str,Any])->float:
    raw=pattern.get("production_cost")
    if isinstance(raw,(int,float)): return max(0.0,min(1.0,float(raw)))
    if isinstance(raw,str): return {"minimal":0.15,"low":0.3,"medium":0.55,"high":0.8,"very-high":0.95}.get(raw.strip().lower(),0.5)
    if isinstance(raw,dict):
        values=[float(v) for v in raw.values() if isinstance(v,(int,float))]
        return sum(values)/len(values) if values else 0.5
    return 0.5

def detect_collisions(pattern,ledger):
    collisions=[]; text=canonical_json(pattern).lower(); pid=pattern.get("pattern_id","")
    for active in ledger["active"]:
        if active.lower() in text or active.lower()==pid.lower(): collisions.append({"type":"REDUNDANT","with":active})
    for collision in ledger["collisions"]:
        if not isinstance(collision,dict): continue
        kind=collision.get("type"); participants=set(collision.get("patterns",[]))
        if kind in COLLISION_TYPES and pid in participants: collisions.append({"type":kind,"with":",".join(sorted(participants-{pid}))})
    if _tokens(pattern.get("cultural_scope",[]))&_tokens(ledger.get("prohibited_cultural_scopes",[])): collisions.append({"type":"CULTURALLY_INCOMPATIBLE","with":"prohibited cultural scope"})
    return collisions

def score_pattern(pattern,brief,ledger):
    dq=[brief.get("dramatic_problem",""),brief.get("desired_state_change",""),brief.get("symbolic_intent","")]; dc=pattern.get("dramatic_operations",[])+pattern.get("transformation_grammars",[])+[pattern.get("transferable_structure","")]
    dramatic_fit=_overlap(dq,dc); character_fit=_overlap([brief.get("character_state",""),brief.get("character_pressure","")],dc)
    cinematic_fit=max(_overlap(brief.get("preferred_encoding_vectors",[]),pattern.get("cinematic_affordances",[])),0.8 if brief.get("format") in pattern.get("applicable_formats",[]) else 0.0,0.8 if brief.get("genre","").lower() in [g.lower() for g in pattern.get("applicable_genres",[])] else 0.0)
    cultural_fit=_overlap(brief.get("cultural_context",""),pattern.get("cultural_scope",[])) if brief.get("cultural_context") else 0.65
    source_quality={"PRIMARY":0.95,"SCHOLARLY":0.85,"PRACTITIONER":0.75,"COMPARATIVE":0.65}.get(pattern.get("source_tier","POPULAR"),0.5)
    mutation=pattern.get("mutation_requirements",{}); mutation_potential=0.95 if mutation.get("required") and mutation.get("variables") else 0.35
    active_text=" ".join(ledger["active"]).lower(); redundant=any(t in active_text for t in _tokens(pattern.get("title","")))
    continuity_compatibility=max(0.0,0.9-(0.35 if redundant else 0)-min(0.5,ledger["saturation_score"]*0.35)-min(0.4,ledger["symbolic_debt"]*0.25))
    unresolved=ledger["unresolved_payoffs"]; payoff_compatibility=0.75 if not unresolved else min(1.0,0.4+_overlap(unresolved,dc))
    components={"dramatic_fit":dramatic_fit,"character_fit":character_fit,"cinematic_fit":cinematic_fit,"cultural_fit":cultural_fit,"source_quality":source_quality,"mutation_potential":mutation_potential,"continuity_compatibility":continuity_compatibility,"payoff_compatibility":payoff_compatibility}
    total=sum(components[k]*w for k,w in WEIGHTS.items()); cost=production_cost(pattern); budget=brief.get("production_cost_ceiling")
    if isinstance(budget,(int,float)) and cost>float(budget): total-=min(0.35,cost-float(budget))
    collisions=detect_collisions(pattern,ledger)
    if any(i["type"] in {"CONTRADICTORY","CULTURALLY_INCOMPATIBLE"} for i in collisions): total=0.0
    else: total-=min(0.3,0.08*len(collisions))
    return {"pattern_id":pattern["pattern_id"],"total_score":round(max(0.0,min(1.0,total)),4),"score_components":{k:round(v,4) for k,v in components.items()},"collisions":collisions,"production_cost_score":round(cost,4),"provenance_refs":pattern.get("source_refs",[]),"mutation_requirements":mutation,"lexicon_links":pattern.get("lexicon_links",[])}

def reason_vector(ranked,ledger):
    if not ranked:return ["NO_EXECUTABLE_PATTERNS"]
    top=ranked[0]; c=top["score_components"]; reasons=[]
    if c["dramatic_fit"]<=0:reasons.append("LOW_DRAMATIC_FIT")
    if c["mutation_potential"]<0.6:reasons.append("LOW_MUTATION_POTENTIAL")
    if c["continuity_compatibility"]<0.5:reasons.append("SATURATION_OR_SYMBOLIC_DEBT")
    if c["cultural_fit"]<0.25:reasons.append("LOW_CULTURAL_FIT")
    if top["production_cost_score"]>0.8:reasons.append("HIGH_PRODUCTION_COST")
    reasons.extend(sorted({i["type"] for i in top["collisions"]}))
    if ledger["saturation_score"]>=0.85:reasons.append("SYMBOLIC_OVERLOAD")
    return reasons or ["BELOW_CONFIDENCE_THRESHOLD"]

def gap_report(brief,ranked):
    top=ranked[0] if ranked else None; weak=["dramatic","character","cinematic","cultural","mutation"] if not top else [k.replace("_fit","").replace("_potential","") for k,v in top["score_components"].items() if v<0.35]
    tokens=_tokens(brief.get("dramatic_problem","")); recommended=set(brief.get("requested_domains",[]))
    if "sound" in tokens:recommended.add("sound")
    if "threshold" in tokens:recommended.add("ritual-liminal")
    return {"dramatic_problem":brief.get("dramatic_problem"),"coverage_strength":round(top["total_score"],4) if top else 0.0,"weak_dimensions":weak,"recommended_domains":sorted(recommended)}

def read_brief(path): return _load(Path(path)) if path else yaml.safe_load(sys.stdin.read()) or {}

def build_receipt(brief,ranked,ledger):
    eligible=[x for x in ranked if x["total_score"]>=THRESHOLD]; primary=eligible[0] if eligible else None; supporting=eligible[1:1+MAX_SUPPORTING]
    return {"retrieval_receipt":{"request_hash":cache_key(brief,ledger),"algorithm":"deterministic-v0.11","timestamp":datetime.now(timezone.utc).isoformat(),"selected_primary_grammar":primary["pattern_id"] if primary else None,"selected_supporting_grammars":[x["pattern_id"] for x in supporting],"ranked_patterns":ranked[:8] if primary else [],"confidence":primary["total_score"] if primary else 0.0,"status":"SELECTED" if primary else "NOT_COMPUTABLE","reason_vector":[] if primary else reason_vector(ranked,ledger),"pattern_gap_report":gap_report(brief,ranked),"ledger_snapshot":ledger,"brief":brief}}

def persist(receipt):
    CACHE_DIR.mkdir(parents=True,exist_ok=True); RECEIPT_DIR.mkdir(parents=True,exist_ok=True); key=receipt["retrieval_receipt"]["request_hash"]
    cache=CACHE_DIR/f"{key}.json"; logged=RECEIPT_DIR/f"receipt-{key}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"
    payload=json.dumps(receipt,indent=2,default=lambda item:item.isoformat() if hasattr(item,"isoformat") else str(item))+"\n"
    for p in (cache,logged):p.write_text(payload,encoding="utf-8")
    return cache,logged

def main():
    p=argparse.ArgumentParser();p.add_argument("--brief");p.add_argument("--no-cache",action="store_true");p.add_argument("--gap-report-only",action="store_true");a=p.parse_args()
    brief=read_brief(a.brief);ledger=ledger_snapshot(brief);key=cache_key(brief,ledger);cache=CACHE_DIR/f"{key}.json"
    if not a.no_cache and cache.exists():
        receipt=_load(cache);receipt["retrieval_receipt"]["cache_hit"]=True;print(yaml.safe_dump(receipt,sort_keys=False));raise SystemExit(0 if receipt["retrieval_receipt"]["status"]=="SELECTED" else 1)
    ranked=sorted((score_pattern(pattern,brief,ledger) for pattern in load_all_patterns().values()),key=lambda x:(-x["total_score"],x["pattern_id"]));receipt=build_receipt(brief,ranked,ledger);receipt["retrieval_receipt"]["cache_hit"]=False
    if a.gap_report_only:print(yaml.safe_dump(receipt["retrieval_receipt"]["pattern_gap_report"],sort_keys=False));return
    cache_path,logged=persist(receipt);receipt["retrieval_receipt"]["cached_to"]=str(cache_path);receipt["retrieval_receipt"]["logged_to"]=str(logged);print(yaml.safe_dump(receipt,sort_keys=False));raise SystemExit(0 if receipt["retrieval_receipt"]["status"]=="SELECTED" else 1)

if __name__=="__main__":main()
