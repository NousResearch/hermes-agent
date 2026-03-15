#!/usr/bin/env python3
"""
Quality Gates — Enforce Minimum Standards with Rollback

Gates are checkpoints that research must pass before continuing.
If a gate fails, the pipeline rolls back to a previous stage.

This ensures research quality is enforced, not just hoped for.

Usage:
  python3 gates.py init                           # Initialize gate system
  python3 gates.py check --gate diversity         # Run a specific gate
  python3 gates.py status                         # Show all gate statuses
  python3 gates.py rollback --gate diversity      # Rollback after gate failure
  python3 gates.py reset                          # Reset all gates
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SKILL_DIR = os.path.dirname(SCRIPT_DIR)


def get_state_dir():
    """Return a writable state directory."""
    override = os.environ.get("HERMES_DEEP_RESEARCH_STATE_DIR")
    if override:
        return override
    hermes_home = os.environ.get("HERMES_HOME")
    if hermes_home:
        return os.path.join(hermes_home, "state", "deep-research")
    return str(Path.home() / ".hermes" / "state" / "deep-research")


STATE_DIR = get_state_dir()
GATES_FILE = os.path.join(STATE_DIR, "gates.json")


# Gate definitions with pass criteria and rollback targets
GATES = {
    "evidence_diversity": {
        "name": "Evidence Diversity Gate",
        "description": "Have I used 2+ source types?",
        "pass_criteria": "source_types_used >= 2",
        "rollback_target": "search",  # Go back to search phase
        "critical": True,
        "weight": 1.0,
    },
    "claim_verification": {
        "name": "Claim Verification Gate",
        "description": "Are key claims verified by 2+ sources?",
        "pass_criteria": "verified_claims / total_claims >= 0.5",
        "rollback_target": "gap_filling",
        "critical": True,
        "weight": 1.0,
    },
    "probability_grounding": {
        "name": "Probability Grounding Gate",
        "description": "Is probability estimate tied to prediction market, analyst forecast, or base rate?",
        "pass_criteria": "probability_sources >= 1",
        "rollback_target": "search",
        "critical": True,
        "weight": 1.0,
    },
    "source_quality": {
        "name": "Source Quality Gate",
        "description": "Are there any Tier 1-2 sources?",
        "pass_criteria": "tier1_count + tier2_count >= 1",
        "rollback_target": "search",
        "critical": True,
        "weight": 0.8,
    },
    "contrarian_evidence": {
        "name": "Contrarian Evidence Gate",
        "description": "Did I search for evidence against my hypothesis?",
        "pass_criteria": "contrarian_searches >= 1",
        "rollback_target": "reflection",
        "critical": True,
        "weight": 0.9,
    },
    "hypothesis_stated": {
        "name": "Hypothesis Stated Gate",
        "description": "Is my hypothesis explicitly stated before research?",
        "pass_criteria": "hypothesis_stated == true",
        "rollback_target": "planning",
        "critical": False,
        "weight": 0.5,
    },
    "saturation": {
        "name": "Saturation Gate",
        "description": "Have critical gaps been filled?",
        "pass_criteria": "critical_gaps_filled / critical_gaps >= 0.8",
        "rollback_target": "gap_filling",
        "critical": True,
        "weight": 1.0,
    },
    "freshness": {
        "name": "Freshness Gate",
        "description": "Is evidence from the last 7 days (for current events)?",
        "pass_criteria": "recent_sources >= 1 OR topic_not_time_sensitive",
        "rollback_target": "search",
        "critical": False,
        "weight": 0.7,
    },
}

# Gate execution order (check in this sequence)
GATE_ORDER = [
    "hypothesis_stated",
    "evidence_diversity",
    "source_quality",
    "claim_verification",
    "contrarian_evidence",
    "probability_grounding",
    "saturation",
    "freshness",
]


def timestamp():
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")


def load_gates() -> dict:
    """Load gate state."""
    if os.path.exists(GATES_FILE):
        with open(GATES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "gates": {},
        "created": timestamp(),
        "current_phase": "planning",
        "rollback_history": [],
        "proof": {},
    }


def save_gates(state: dict):
    """Save gate state."""
    os.makedirs(os.path.dirname(GATES_FILE), exist_ok=True)
    with open(GATES_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def init_gate_state(gate_id: str) -> dict:
    """Initialize state for a gate."""
    return {
        "id": gate_id,
        "name": GATES[gate_id]["name"],
        "status": "pending",  # pending, passed, failed, skipped
        "checked_at": None,
        "result": None,
        "notes": [],
        "retry_count": 0,
    }


def cmd_init(args):
    """Initialize the gate system."""
    state = {
        "gates": {gid: init_gate_state(gid) for gid in GATES},
        "created": timestamp(),
        "current_phase": "planning",
        "rollback_history": [],
        "proof": {},
    }
    save_gates(state)
    print(json.dumps({
        "initialized": True,
        "gates": len(GATES),
        "gate_order": GATE_ORDER,
    }, indent=2))
    return 0


def check_evidence_diversity(proof: dict) -> Tuple[bool, str, list]:
    """Check if 2+ source types were used."""
    types = proof.get("source_types_used", [])
    count = len(set(types))
    
    if count >= 2:
        return True, f"PASS: {count} source types used", []
    elif count == 1:
        return False, f"FAIL: Only 1 source type ({types[0]}). Need at least 2.", [
            "Add a different source type (e.g., if you used web search, add news or prediction market)"
        ]
    else:
        return False, "FAIL: No source types recorded", [
            "Record which tools you used in source_types_used array"
        ]


def check_claim_verification(proof: dict) -> Tuple[bool, str, list]:
    """Check if key claims are verified."""
    claims = proof.get("claims", [])
    if not claims:
        return False, "FAIL: No claims recorded", ["Add claims to the proof object"]
    
    total = len(claims)
    verified = sum(1 for c in claims if c.get("status") == "verified")
    supported = sum(1 for c in claims if c.get("status") == "supported")
    
    good = verified + supported
    ratio = good / total if total > 0 else 0
    
    if ratio >= 0.5:
        return True, f"PASS: {good}/{total} claims verified/supported ({ratio:.0%})", []
    else:
        return False, f"FAIL: Only {good}/{total} claims verified/supported ({ratio:.0%}). Need 50%+.", [
            f"Find sources for {total - good} unverified claims",
            "Or remove claims you cannot verify"
        ]


def check_probability_grounding(proof: dict) -> Tuple[bool, str, list]:
    """Check if probability estimate is grounded."""
    # Check if this is a forecasting question
    topic = proof.get("topic", "").lower()
    forecast_keywords = ["will", "forecast", "predict", "probability", "chance", "%", "likely"]
    is_forecast = any(kw in topic for kw in forecast_keywords)
    
    if not is_forecast:
        return True, "PASS: Not a forecasting question, probability gate not applicable", []
    
    # Check for probability sources
    prob_sources = proof.get("probability_sources", [])
    prediction_market = proof.get("prediction_market_data")
    analyst_forecast = proof.get("analyst_forecast_data")
    base_rate = proof.get("historical_base_rate")
    
    sources = []
    if prediction_market:
        sources.append("prediction_market")
    if analyst_forecast:
        sources.append("analyst_forecast")
    if base_rate:
        sources.append("base_rate")
    sources.extend(prob_sources)
    
    if len(sources) >= 1:
        return True, f"PASS: Probability grounded in {', '.join(sources)}", []
    else:
        return False, "FAIL: Probability estimate not grounded", [
            "Check Polymarket or Metaculus for crowd-sourced probability",
            "Find analyst forecasts (Goldman, JPMorgan, etc.)",
            "Research historical base rates for similar events"
        ]


def check_source_quality(proof: dict) -> Tuple[bool, str, list]:
    """Check for Tier 1-2 sources."""
    tiers = proof.get("source_tiers", {})
    t1 = tiers.get("tier1", 0)
    t2 = tiers.get("tier2", 0)
    
    if t1 + t2 >= 1:
        return True, f"PASS: {t1} Tier 1 + {t2} Tier 2 sources", []
    else:
        return False, "FAIL: No Tier 1 or Tier 2 sources found", [
            "Find primary sources (government data, official reports)",
            "Or find expert analysis (analyst reports, peer-reviewed papers)"
        ]


def check_contrarian_evidence(proof: dict) -> Tuple[bool, str, list]:
    """Check for contrarian searches."""
    contrarian = proof.get("contrarian_searches", [])
    
    if len(contrarian) >= 1:
        found_evidence = sum(1 for c in contrarian if c.get("found_evidence"))
        return True, f"PASS: {len(contrarian)} contrarian search(es), {found_evidence} found evidence", []
    else:
        return False, "FAIL: No contrarian searches performed", [
            "Search for evidence AGAINST your hypothesis",
            "Try queries like: 'why X is wrong', 'X bear case', 'X criticism'"
        ]


def check_hypothesis_stated(proof: dict) -> Tuple[bool, str, list]:
    """Check if hypothesis is explicitly stated."""
    hypothesis = proof.get("hypothesis_original")
    
    if hypothesis:
        return True, "PASS: Hypothesis stated", []
    else:
        return False, "FAIL: No hypothesis stated", [
            "State your initial hypothesis before researching",
            "Format: 'I believe X will happen because Y'"
        ]


def check_saturation(proof: dict) -> Tuple[bool, str, list]:
    """Check if critical gaps are filled."""
    gaps = proof.get("gaps_identified", [])
    
    if not gaps:
        return False, "FAIL: No gap analysis performed", [
            "Identify gaps in your research",
            "Rank each gap: critical, important, nice_to_have"
        ]
    
    critical = [g for g in gaps if g.get("rank") == "critical"]
    if not critical:
        return True, "PASS: No critical gaps identified", []
    
    filled = sum(1 for g in critical if g.get("filled"))
    ratio = filled / len(critical)
    
    if ratio >= 0.8:
        return True, f"PASS: {filled}/{len(critical)} critical gaps filled", []
    else:
        return False, f"FAIL: Only {filled}/{len(critical)} critical gaps filled. Need 80%+.", [
            f"Fill {len(critical) - filled} remaining critical gaps",
            "Or explain why gap cannot be filled"
        ]


def check_freshness(proof: dict) -> Tuple[bool, str, list]:
    """Check evidence freshness (7-day rule)."""
    topic = proof.get("topic", "").lower()
    
    # Topics that require fresh data
    time_sensitive = any(kw in topic for kw in 
        ["war", "conflict", "election", "market", "price", "current", "latest", "today", "now"])
    
    if not time_sensitive:
        return True, "PASS: Topic is not time-sensitive", []
    
    recent = proof.get("recent_sources_count", 0)
    if recent >= 1:
        return True, f"PASS: {recent} recent source(s) found", []
    else:
        return False, "FAIL: No recent sources for time-sensitive topic", [
            "Find sources from the last 7 days",
            "Check news feeds for latest developments"
        ]


CHECKERS = {
    "evidence_diversity": check_evidence_diversity,
    "claim_verification": check_claim_verification,
    "probability_grounding": check_probability_grounding,
    "source_quality": check_source_quality,
    "contrarian_evidence": check_contrarian_evidence,
    "hypothesis_stated": check_hypothesis_stated,
    "saturation": check_saturation,
    "freshness": check_freshness,
}


def cmd_check(args):
    """Run a specific gate check."""
    state = load_gates()
    
    if args.gate not in GATES:
        print(f"Unknown gate: {args.gate}")
        print(f"Available gates: {list(GATES.keys())}")
        return 1
    
    # Load proof from file if provided
    if args.proof:
        with open(args.proof, "r", encoding="utf-8") as f:
            state["proof"] = json.load(f)
    elif args.proof_json:
        state["proof"] = json.loads(args.proof_json)
    else:
        # Use existing proof in state
        pass
    
    gate_def = GATES[args.gate]
    checker = CHECKERS[args.gate]
    
    passed, message, actions = checker(state["proof"])
    
    # Update gate state
    gate_state = state["gates"][args.gate]
    gate_state["status"] = "passed" if passed else "failed"
    gate_state["checked_at"] = timestamp()
    gate_state["result"] = {
        "passed": passed,
        "message": message,
        "actions": actions,
    }
    if not passed:
        gate_state["retry_count"] += 1
    
    save_gates(state)
    
    result = {
        "gate": args.gate,
        "name": gate_def["name"],
        "passed": passed,
        "message": message,
        "actions_needed": actions if not passed else [],
        "rollback_target": gate_def["rollback_target"] if not passed else None,
        "critical": gate_def["critical"],
    }
    
    print(json.dumps(result, indent=2))
    
    if not passed and gate_def["critical"]:
        print(f"\n⚠️  CRITICAL GATE FAILED")
        print(f"   Rollback target: {gate_def['rollback_target']}")
        print(f"   Actions needed:")
        for a in actions:
            print(f"   - {a}")
    
    return 0 if passed else 1


def cmd_all(args):
    """Run all gates in order."""
    state = load_gates()
    
    if args.proof:
        with open(args.proof, "r", encoding="utf-8") as f:
            state["proof"] = json.load(f)
    
    results = []
    failed_critical = []
    
    for gate_id in GATE_ORDER:
        gate_def = GATES[gate_id]
        checker = CHECKERS[gate_id]
        
        passed, message, actions = checker(state["proof"])
        
        gate_state = state["gates"][gate_id]
        gate_state["status"] = "passed" if passed else "failed"
        gate_state["checked_at"] = timestamp()
        gate_state["result"] = {
            "passed": passed,
            "message": message,
            "actions": actions,
        }
        
        results.append({
            "gate": gate_id,
            "name": gate_def["name"],
            "passed": passed,
            "critical": gate_def["critical"],
            "message": message,
        })
        
        if not passed and gate_def["critical"]:
            failed_critical.append({
                "gate": gate_id,
                "rollback": gate_def["rollback_target"],
                "actions": actions,
            })
    
    save_gates(state)
    
    # Summary
    passed_count = sum(1 for r in results if r["passed"])
    total = len(results)
    
    print("=" * 60)
    print("  GATE CHECK RESULTS")
    print("=" * 60)
    print()
    
    for r in results:
        status = "✓ PASS" if r["passed"] else "✗ FAIL"
        critical = " [CRITICAL]" if r["critical"] else ""
        print(f"  {status}  {r['name']}{critical}")
        if not r["passed"]:
            print(f"         {r['message']}")
    
    print()
    print(f"  PASSED: {passed_count}/{total}")
    
    if failed_critical:
        print()
        print("  ⚠️  CRITICAL GATES FAILED:")
        for fc in failed_critical:
            print(f"     - {fc['gate']} → rollback to '{fc['rollback']}'")
            for a in fc["actions"]:
                print(f"       → {a}")
        print()
        print("  ACTION: Fix issues and re-run gates")
    else:
        print()
        print("  ✓ ALL CRITICAL GATES PASSED")
    
    return 0 if not failed_critical else 1


def cmd_status(args):
    """Show status of all gates."""
    state = load_gates()
    
    print(json.dumps({
        "current_phase": state.get("current_phase"),
        "gates": {
            gid: {
                "status": g["status"],
                "checked_at": g["checked_at"],
            }
            for gid, g in state["gates"].items()
        },
        "rollback_count": len(state.get("rollback_history", [])),
    }, indent=2))
    return 0


def cmd_rollback(args):
    """Record a rollback action."""
    state = load_gates()
    
    if args.gate not in GATES:
        print(f"Unknown gate: {args.gate}")
        return 1
    
    gate_def = GATES[args.gate]
    rollback = {
        "gate": args.gate,
        "rollback_to": gate_def["rollback_target"],
        "reason": args.reason or "Gate failed",
        "timestamp": timestamp(),
    }
    
    state["rollback_history"].append(rollback)
    state["current_phase"] = gate_def["rollback_target"]
    
    # Reset gate status
    state["gates"][args.gate] = init_gate_state(args.gate)
    
    save_gates(state)
    
    print(json.dumps({
        "rolled_back": True,
        "from_gate": args.gate,
        "to_phase": gate_def["rollback_target"],
        "total_rollbacks": len(state["rollback_history"]),
    }, indent=2))
    return 0


def cmd_reset(args):
    """Reset all gates."""
    state = {
        "gates": {gid: init_gate_state(gid) for gid in GATES},
        "created": timestamp(),
        "current_phase": "planning",
        "rollback_history": [],
        "proof": {},
    }
    save_gates(state)
    print(json.dumps({"reset": True, "gates": len(GATES)}, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description="Quality Gates — Enforce Minimum Research Standards")
    sub = parser.add_subparsers(dest="command")
    
    # init
    sub.add_parser("init", help="Initialize gate system")
    
    # check
    p_check = sub.add_parser("check", help="Run a specific gate")
    p_check.add_argument("--gate", required=True, choices=list(GATES.keys()), help="Gate to check")
    p_check.add_argument("--proof", help="Path to proof JSON file")
    p_check.add_argument("--proof-json", help="Proof JSON as string")
    
    # all
    p_all = sub.add_parser("all", help="Run all gates")
    p_all.add_argument("--proof", help="Path to proof JSON file")
    
    # status
    sub.add_parser("status", help="Show gate statuses")
    
    # rollback
    p_rollback = sub.add_parser("rollback", help="Record rollback after gate failure")
    p_rollback.add_argument("--gate", required=True, help="Gate that failed")
    p_rollback.add_argument("--reason", help="Reason for rollback")
    
    # reset
    sub.add_parser("reset", help="Reset all gates")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    commands = {
        "init": cmd_init,
        "check": cmd_check,
        "all": cmd_all,
        "status": cmd_status,
        "rollback": cmd_rollback,
        "reset": cmd_reset,
    }
    
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())