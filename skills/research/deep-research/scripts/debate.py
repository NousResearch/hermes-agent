#!/usr/bin/env python3
"""
Multi-Agent Debate — Structured FOR/AGAINST Analysis

Simulates a debate between multiple analysts with different perspectives.
This forces the researcher to confront counter-arguments and strengthen
their analysis.

Usage:
  python3 debate.py init --hypothesis "X will happen"    # Initialize debate
  python3 debate.py argue --side for --argument "..."    # Add argument
  python3 debate.py synthesize                            # Generate synthesis
  python3 debate.py verdict                               # Final verdict
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List

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
DEBATE_FILE = os.path.join(STATE_DIR, "debate.json")


def timestamp():
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")


def load_debate() -> dict:
    """Load debate state."""
    if os.path.exists(DEBATE_FILE):
        with open(DEBATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_debate(data: dict):
    """Save debate state."""
    os.makedirs(os.path.dirname(DEBATE_FILE), exist_ok=True)
    with open(DEBATE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def cmd_init(args):
    """Initialize a new debate."""
    debate = {
        "hypothesis": args.hypothesis,
        "created": timestamp(),
        "status": "open",
        "analysts": {
            "proponent": {
                "role": "Argues FOR the hypothesis",
                "arguments": [],
                "evidence_cited": [],
                "strength": None,
            },
            "skeptic": {
                "role": "Argues AGAINST the hypothesis",
                "arguments": [],
                "evidence_cited": [],
                "strength": None,
            },
            "alternative": {
                "role": "Proposes ALTERNATIVE explanation",
                "alternative_hypothesis": None,
                "arguments": [],
                "evidence_cited": [],
            },
        },
        "synthesis": None,
        "verdict": None,
        "rounds": 0,
        "history": [],
    }
    
    save_debate(debate)
    
    print(json.dumps({
        "initialized": True,
        "hypothesis": args.hypothesis,
        "analysts": ["proponent", "skeptic", "alternative"],
        "next_steps": [
            "Add arguments with: debate.py argue --side <proponent|skeptic|alternative> --argument '...'",
            "Each analyst should cite specific evidence",
            "Run synthesize when debate is complete",
        ]
    }, indent=2))
    return 0


def cmd_argue(args):
    """Add an argument from an analyst."""
    debate = load_debate()
    
    if not debate:
        print("No debate initialized. Run: debate.py init --hypothesis '...'")
        return 1
    
    side = args.side.lower()
    if side not in ["proponent", "skeptic", "alternative"]:
        print(f"Unknown side: {side}. Use: proponent, skeptic, or alternative")
        return 1
    
    argument = {
        "content": args.argument,
        "evidence": args.evidence,
        "strength": args.strength,
        "timestamp": timestamp(),
    }
    
    debate["analysts"][side]["arguments"].append(argument)
    
    if args.evidence:
        debate["analysts"][side]["evidence_cited"].append(args.evidence)
    
    if args.strength:
        debate["analysts"][side]["strength"] = args.strength
    
    debate["rounds"] += 1
    debate["history"].append({
        "action": "argue",
        "side": side,
        "summary": args.argument[:100],
        "timestamp": timestamp(),
    })
    
    # Handle alternative hypothesis
    if side == "alternative" and args.alternative_hypothesis:
        debate["analysts"]["alternative"]["alternative_hypothesis"] = args.alternative_hypothesis
    
    save_debate(debate)
    
    print(json.dumps({
        "added": True,
        "side": side,
        "argument_count": len(debate["analysts"][side]["arguments"]),
        "total_rounds": debate["rounds"],
    }, indent=2))
    return 0


def cmd_counter(args):
    """Add a counter-argument."""
    debate = load_debate()
    
    if not debate:
        print("No debate initialized.")
        return 1
    
    # Determine counter side
    if args.to == "proponent":
        counter_side = "skeptic"
    elif args.to == "skeptic":
        counter_side = "proponent"
    else:
        counter_side = args.to
    
    argument = {
        "content": args.argument,
        "counters": args.to,
        "evidence": args.evidence,
        "timestamp": timestamp(),
    }
    
    debate["analysts"][counter_side]["arguments"].append(argument)
    debate["rounds"] += 1
    
    if args.evidence:
        debate["analysts"][counter_side]["evidence_cited"].append(args.evidence)
    
    debate["history"].append({
        "action": "counter",
        "side": counter_side,
        "counters": args.to,
        "summary": args.argument[:100],
        "timestamp": timestamp(),
    })
    
    save_debate(debate)
    
    print(json.dumps({
        "counter_added": True,
        "from": counter_side,
        "counters": args.to,
    }, indent=2))
    return 0


def cmd_synthesize(args):
    """Generate synthesis of the debate."""
    debate = load_debate()
    
    if not debate:
        print("No debate initialized.")
        return 1
    
    proponent_args = debate["analysts"]["proponent"]["arguments"]
    skeptic_args = debate["analysts"]["skeptic"]["arguments"]
    alternative_args = debate["analysts"]["alternative"]["arguments"]
    
    # Count arguments
    p_count = len(proponent_args)
    s_count = len(skeptic_args)
    a_count = len(alternative_args)
    
    # Check if debate is balanced
    balance = abs(p_count - s_count)
    if balance > 2:
        print(f"WARNING: Debate is unbalanced. Proponent: {p_count}, Skeptic: {s_count}")
        print("Consider adding more arguments from the weaker side.")
    
    # Generate synthesis structure (agent will fill in details)
    synthesis = {
        "summary": None,  # Agent fills
        "strongest_for": None,  # Agent fills
        "strongest_against": None,  # Agent fills
        "key_evidence": {
            "for": [],
            "against": [],
        },
        "unresolved_questions": [],
        "revised_confidence": None,
        "hypothesis_should_revise": None,
        "revised_hypothesis": None,
    }
    
    debate["synthesis"] = synthesis
    debate["status"] = "synthesized"
    debate["history"].append({
        "action": "synthesize",
        "timestamp": timestamp(),
    })
    
    save_debate(debate)
    
    print(json.dumps({
        "synthesis_created": True,
        "argument_counts": {
            "proponent": p_count,
            "skeptic": s_count,
            "alternative": a_count,
        },
        "balance": "balanced" if balance <= 2 else "unbalanced",
        "synthesis_fields_to_complete": [
            "summary",
            "strongest_for",
            "strongest_against",
            "key_evidence.for",
            "key_evidence.against",
            "unresolved_questions",
            "revised_confidence",
            "hypothesis_should_revise",
        ],
        "instruction": "Agent must fill synthesis fields based on debate arguments",
    }, indent=2))
    return 0


def cmd_verdict(args):
    """Render final verdict."""
    debate = load_debate()
    
    if not debate:
        print("No debate initialized.")
        return 1
    
    if not debate.get("synthesis"):
        print("No synthesis found. Run: debate.py synthesize")
        return 1
    
    # Calculate scores based on arguments and evidence
    p_strength = debate["analysts"]["proponent"].get("strength") or len(debate["analysts"]["proponent"]["arguments"])
    s_strength = debate["analysts"]["skeptic"].get("strength") or len(debate["analysts"]["skeptic"]["arguments"])
    
    total = p_strength + s_strength
    if total == 0:
        p_ratio = 0.5
    else:
        p_ratio = p_strength / total
    
    # Determine verdict
    if p_ratio >= 0.7:
        verdict = "STRONG_SUPPORT"
        confidence = "high"
    elif p_ratio >= 0.55:
        verdict = "MODERATE_SUPPORT"
        confidence = "medium"
    elif p_ratio >= 0.45:
        verdict = "INCONCLUSIVE"
        confidence = "low"
    elif p_ratio >= 0.3:
        verdict = "MODERATE_OPPOSITION"
        confidence = "medium"
    else:
        verdict = "STRONG_OPPOSITION"
        confidence = "high"
    
    # Check for alternative hypothesis
    alt_hyp = debate["analysts"]["alternative"].get("alternative_hypothesis")
    if alt_hyp and len(debate["analysts"]["alternative"]["arguments"]) >= 2:
        alternative_strength = "moderate"
    else:
        alternative_strength = "weak"
    
    verdict_data = {
        "verdict": verdict,
        "confidence": confidence,
        "proponent_strength": p_strength,
        "skeptic_strength": s_strength,
        "hypothesis_support_ratio": round(p_ratio, 2),
        "alternative_hypothesis": alt_hyp,
        "alternative_strength": alternative_strength,
        "key_findings": args.findings.split("|") if args.findings else [],
        "recommendation": generate_recommendation(verdict, confidence, alt_hyp),
        "timestamp": timestamp(),
    }
    
    debate["verdict"] = verdict_data
    debate["status"] = "complete"
    save_debate(debate)
    
    print("=" * 60)
    print("  DEBATE VERDICT")
    print("=" * 60)
    print()
    print(f"  Hypothesis: {debate['hypothesis']}")
    print()
    print(f"  VERDICT: {verdict}")
    print(f"  Confidence: {confidence}")
    print()
    print(f"  Proponent strength: {p_strength}")
    print(f"  Skeptic strength: {s_strength}")
    print(f"  Support ratio: {p_ratio:.0%}")
    
    if alt_hyp:
        print()
        print(f"  Alternative hypothesis: {alt_hyp}")
    
    print()
    print(f"  RECOMMENDATION:")
    print(f"  {verdict_data['recommendation']}")
    
    print()
    print(json.dumps(verdict_data, indent=2))
    return 0


def generate_recommendation(verdict: str, confidence: str, alt_hyp: Optional[str]) -> str:
    """Generate recommendation based on verdict."""
    recommendations = {
        "STRONG_SUPPORT": "The evidence strongly supports the hypothesis. Proceed with high confidence. Consider documenting key supporting evidence.",
        "MODERATE_SUPPORT": "The evidence moderately supports the hypothesis. Proceed with medium confidence. Address remaining counter-arguments.",
        "INCONCLUSIVE": "The evidence is inconclusive. Either gather more data or reformulate the hypothesis. Do not proceed with high confidence forecasts.",
        "MODERATE_OPPOSITION": "The evidence moderately opposes the hypothesis. Consider revising or pivoting to an alternative explanation.",
        "STRONG_OPPOSITION": "The evidence strongly opposes the hypothesis. PIVOT required. Abandon this hypothesis and explore alternatives.",
    }
    
    base = recommendations.get(verdict, "Analyze the debate arguments and make a decision.")
    
    if alt_hyp:
        base += f" Alternative hypothesis '{alt_hyp[:50]}...' shows promise."
    
    return base


def cmd_show(args):
    """Show current debate state."""
    debate = load_debate()
    
    if not debate:
        print("No debate initialized.")
        return 1
    
    print("=" * 60)
    print("  CURRENT DEBATE")
    print("=" * 60)
    print()
    print(f"  Hypothesis: {debate['hypothesis']}")
    print(f"  Status: {debate['status']}")
    print(f"  Rounds: {debate['rounds']}")
    print()
    
    print("  PROPONENT (FOR):")
    for i, arg in enumerate(debate["analysts"]["proponent"]["arguments"][:5], 1):
        print(f"    {i}. {arg['content'][:80]}...")
    
    print()
    print("  SKEPTIC (AGAINST):")
    for i, arg in enumerate(debate["analysts"]["skeptic"]["arguments"][:5], 1):
        print(f"    {i}. {arg['content'][:80]}...")
    
    if debate["analysts"]["alternative"].get("alternative_hypothesis"):
        print()
        print("  ALTERNATIVE HYPOTHESIS:")
        print(f"    {debate['analysts']['alternative']['alternative_hypothesis']}")
        for i, arg in enumerate(debate["analysts"]["alternative"]["arguments"][:3], 1):
            print(f"    {i}. {arg['content'][:80]}...")
    
    if debate.get("verdict"):
        print()
        print("  VERDICT:")
        print(f"    {debate['verdict']['verdict']} (confidence: {debate['verdict']['confidence']})")
    
    return 0


def cmd_pivot(args):
    """Record a PIVOT decision."""
    debate = load_debate()
    
    if not debate:
        print("No debate initialized.")
        return 1
    
    pivot = {
        "from_hypothesis": debate["hypothesis"],
        "to_hypothesis": args.new_hypothesis,
        "reason": args.reason,
        "timestamp": timestamp(),
    }
    
    debate["pivots"] = debate.get("pivots", [])
    debate["pivots"].append(pivot)
    debate["hypothesis"] = args.new_hypothesis
    
    # Reset arguments
    for analyst in debate["analysts"]:
        debate["analysts"][analyst]["arguments"] = []
        debate["analysts"][analyst]["evidence_cited"] = []
    
    debate["synthesis"] = None
    debate["verdict"] = None
    debate["status"] = "open"
    debate["rounds"] = 0
    
    debate["history"].append({
        "action": "pivot",
        "from": pivot["from_hypothesis"][:50],
        "to": pivot["to_hypothesis"][:50],
        "timestamp": timestamp(),
    })
    
    save_debate(debate)
    
    print(json.dumps({
        "pivoted": True,
        "from": pivot["from_hypothesis"],
        "to": pivot["to_hypothesis"],
        "pivot_count": len(debate["pivots"]),
        "warning": "Max 2 pivots recommended. Consider gathering more evidence instead of endless pivoting.",
    }, indent=2))
    return 0


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Debate — Structured FOR/AGAINST Analysis")
    sub = parser.add_subparsers(dest="command")
    
    # init
    p_init = sub.add_parser("init", help="Initialize new debate")
    p_init.add_argument("--hypothesis", required=True, help="Hypothesis to debate")
    
    # argue
    p_argue = sub.add_parser("argue", help="Add argument from an analyst")
    p_argue.add_argument("--side", required=True, choices=["proponent", "skeptic", "alternative"])
    p_argue.add_argument("--argument", required=True, help="The argument")
    p_argue.add_argument("--evidence", help="Evidence cited")
    p_argue.add_argument("--strength", type=int, help="Argument strength (1-10)")
    p_argue.add_argument("--alternative-hypothesis", help="For alternative side: the alternative hypothesis")
    
    # counter
    p_counter = sub.add_parser("counter", help="Add counter-argument")
    p_counter.add_argument("--to", required=True, help="Side being countered")
    p_counter.add_argument("--argument", required=True)
    p_counter.add_argument("--evidence", help="Evidence cited")
    
    # synthesize
    sub.add_parser("synthesize", help="Generate debate synthesis")
    
    # verdict
    p_verdict = sub.add_parser("verdict", help="Render final verdict")
    p_verdict.add_argument("--findings", help="Key findings (pipe-separated)")
    
    # show
    sub.add_parser("show", help="Show current debate state")
    
    # pivot
    p_pivot = sub.add_parser("pivot", help="PIVOT to new hypothesis")
    p_pivot.add_argument("--new-hypothesis", required=True, help="New hypothesis")
    p_pivot.add_argument("--reason", required=True, help="Why pivoting")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    commands = {
        "init": cmd_init,
        "argue": cmd_argue,
        "counter": cmd_counter,
        "synthesize": cmd_synthesize,
        "verdict": cmd_verdict,
        "show": cmd_show,
        "pivot": cmd_pivot,
    }
    
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())