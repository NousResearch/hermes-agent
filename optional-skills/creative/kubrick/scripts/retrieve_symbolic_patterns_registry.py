#!/usr/bin/env python3
"""Registry-aware Kubrick retrieval for the standalone Hermes skill."""
from __future__ import annotations
import argparse, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
try:
    import yaml
except ImportError:
    print("pyyaml required. pip install pyyaml",file=sys.stderr); raise SystemExit(1)
import retrieve_symbolic_patterns as base
SCRIPT_DIR=Path(__file__).resolve().parent
SKILL_ROOT=SCRIPT_DIR.parent
REGISTRY_PATH=SKILL_ROOT/"references"/"executable-corpus-registry.yaml"
INDEX_PATH=SKILL_ROOT/"references"/"corpus-index.yaml"
CACHE_DIR=SKILL_ROOT/"references"/"usage"/"cache-registry"
RECEIPT_DIR=SKILL_ROOT/"references"/"usage"/"receipts"
CORPUS_VERSION="0.11.5"
ROUTE_MATCH_BONUS=0.08
STRONG_DEFAULT_BONUS=0.03

def load_yaml(path:Path)->Dict[str,Any]:
    with path.open("r",encoding="utf-8") as h:return yaml.safe_load(h) or {}

def route_registry():
    registry=load_yaml(REGISTRY_PATH); routes={str(route):{str(item) for item in ids or []} for route,ids in (registry.get("default_routes") or {}).items()}; strong={str(item) for item in registry.get("strong_defaults",[])}
    return routes,strong

def route_query(brief): return base._tokens([brief.get("dramatic_problem",""),brief.get("desired_state_change",""),brief.get("symbolic_intent",""),brief.get("requested_domains",[]),brief.get("preferred_encoding_vectors",[])])

def route_matches(brief,routes):
    tokens=route_query(brief); return sorted(route for route in routes if base._tokens(route) and base._tokens(route)&tokens)

def cache_key(brief,ledger):
    payload=base.canonical_json({"algorithm":CORPUS_VERSION,"brief":brief,"ledger":ledger})
    return hashlib.sha256(payload.encode()).hexdigest()[:24]

def score_with_registry(pattern,brief,ledger,routes,matched,strong):
    scored=base.score_pattern(pattern,brief,ledger); pid=scored["pattern_id"]; direct=sorted(route for route in matched if pid in routes.get(route,set())); bonus=(ROUTE_MATCH_BONUS if direct else 0)+(STRONG_DEFAULT_BONUS if pid in strong else 0)
    scored.update({"base_score":scored["total_score"],"route_matches":direct,"route_bonus":round(bonus,4),"strong_default":pid in strong}); scored["total_score"]=round(min(1.0,scored["total_score"]+bonus),4); return scored

def gap_report(brief,ranked,matched,routes):
    report=base.gap_report(brief,ranked); report["matched_registry_routes"]=matched; report["route_candidate_counts"]={route:len(routes.get(route,set())) for route in matched}
    if not matched:report.setdefault("weak_dimensions",[]).append("registry_route")
    return report

def build_receipt(brief,ranked,ledger,matched,routes):
    eligible=[x for x in ranked if x["total_score"]>=base.THRESHOLD]; primary=eligible[0] if eligible else None; supporting=eligible[1:1+base.MAX_SUPPORTING]; key=cache_key(brief,ledger)
    return {"retrieval_receipt":{"request_hash":key,"algorithm":"registry-aware-v1","corpus_version":CORPUS_VERSION,"timestamp":datetime.now(timezone.utc).isoformat(),"selected_primary_grammar":primary["pattern_id"] if primary else None,"selected_supporting_grammars":[x["pattern_id"] for x in supporting],"ranked_patterns":ranked[:8] if primary else [],"confidence":primary["total_score"] if primary else 0.0,"status":"SELECTED" if primary else "NOT_COMPUTABLE","reason_vector":[] if primary else base.reason_vector(ranked,ledger),"pattern_gap_report":gap_report(brief,ranked,matched,routes),"matched_registry_routes":matched,"ledger_snapshot":ledger,"brief":brief}}

def load_cached(key):
    path=CACHE_DIR/f"{key}.json"; return base._load(path) if path.exists() else None

def persist(receipt):
    CACHE_DIR.mkdir(parents=True,exist_ok=True); RECEIPT_DIR.mkdir(parents=True,exist_ok=True); key=receipt["retrieval_receipt"]["request_hash"]; cache=CACHE_DIR/f"{key}.json"; logged=RECEIPT_DIR/f"receipt-registry-{key}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"
    payload=json.dumps(receipt,indent=2,default=lambda item:item.isoformat() if hasattr(item,"isoformat") else str(item))+"\n"
    for p in (cache,logged):p.write_text(payload,encoding="utf-8")
    return cache,logged

def main():
    p=argparse.ArgumentParser();p.add_argument("--brief");p.add_argument("--no-cache",action="store_true");p.add_argument("--gap-report-only",action="store_true");p.add_argument("--show-routes",action="store_true");a=p.parse_args()
    brief=base.read_brief(a.brief);ledger=base.ledger_snapshot(brief);routes,strong=route_registry();matched=route_matches(brief,routes)
    if a.show_routes:print(yaml.safe_dump({"matched_routes":matched},sort_keys=False));return
    key=cache_key(brief,ledger)
    if not a.no_cache:
        cached=load_cached(key)
        if cached:
            cached["retrieval_receipt"]["cache_hit"]=True;print(yaml.safe_dump(cached,sort_keys=False));raise SystemExit(0 if cached["retrieval_receipt"]["status"]=="SELECTED" else 1)
    patterns=base.load_all_patterns();ranked=sorted((score_with_registry(pattern,brief,ledger,routes,matched,strong) for pattern in patterns.values()),key=lambda item:(-item["total_score"],item["pattern_id"]));receipt=build_receipt(brief,ranked,ledger,matched,routes);receipt["retrieval_receipt"]["cache_hit"]=False
    if a.gap_report_only:print(yaml.safe_dump(receipt["retrieval_receipt"]["pattern_gap_report"],sort_keys=False));return
    cache,logged=persist(receipt);receipt["retrieval_receipt"]["cached_to"]=str(cache);receipt["retrieval_receipt"]["logged_to"]=str(logged);print(yaml.safe_dump(receipt,sort_keys=False));raise SystemExit(0 if receipt["retrieval_receipt"]["status"]=="SELECTED" else 1)

if __name__=="__main__":main()
