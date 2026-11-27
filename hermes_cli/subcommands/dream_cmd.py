"""``hermes dream`` — autonomous subconscious memory & reflex distillation CLI.

Run counterfactual trajectory analysis, inspect cognitive reflex rules, or prune
stale invariants.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from agent.dream_cycle import DreamCycleEngine


def cmd_dream_run(args: argparse.Namespace) -> int:
    """Execute a dream cycle over recent trajectories."""
    engine = DreamCycleEngine()
    print("🌙 Hermes entering Subconscious Dream Cycle...")
    summary = engine.execute_dream_cycle()
    print(f"✓ {summary.journal_entry}")
    print(f"  Episodes analyzed: {summary.episodes_harvested}")
    print(f"  Reflex rules active: {summary.total_active_rules}")
    print(f"  Cycle duration: {summary.duration_ms:.1f}ms")
    return 0


def cmd_dream_status(args: argparse.Namespace) -> int:
    """Display active cognitive reflex rules and heuristics."""
    engine = DreamCycleEngine()
    rules = engine.load_reflex_rules()
    if not rules:
        print("No cognitive reflex rules currently active. Run `hermes dream run` after agent sessions.")
        return 0

    print(f"🧠 Hermes Active Cognitive Reflexes ({len(rules)} rules):\n")
    for r in sorted(rules, key=lambda x: (x.category, -x.confidence)):
        print(f"  • [{r.category.upper()}] ({r.severity}, conf: {r.confidence:.2f})")
        print(f"    Trigger:   {r.trigger}")
        print(f"    Heuristic: {r.heuristic}\n")
    return 0


def cmd_dream_prune(args: argparse.Namespace) -> int:
    """Prune low-confidence or oldest overflow rules."""
    engine = DreamCycleEngine()
    rules = engine.load_reflex_rules()
    max_rules = getattr(args, "max_rules", 20) or 20
    retained, pruned = engine.prune_stale_rules(rules, max_rules=max_rules)
    engine.save_reflex_rules(retained)
    print(f"✓ Pruned {pruned} stale rules. Active rules retained: {len(retained)}.")
    return 0


def build_dream_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``hermes dream`` CLI subparser."""
    parser = subparsers.add_parser(
        "dream",
        help="Subconscious memory reflection & cognitive reflex distillation",
        description="Manage autonomous dream cycles and distilled behavioral reflexes.",
    )
    dream_subs = parser.add_subparsers(dest="dream_command")

    # run
    p_run = dream_subs.add_parser("run", help="Execute an immediate dream cycle")
    p_run.set_defaults(func=cmd_dream_run)

    # status
    p_status = dream_subs.add_parser("status", help="List active cognitive reflex rules")
    p_status.set_defaults(func=cmd_dream_status)

    # prune
    p_prune = dream_subs.add_parser("prune", help="Prune low-confidence or overflow rules")
    p_prune.add_argument("--max-rules", type=int, default=20, help="Maximum active rules to retain (default: 20)")
    p_prune.set_defaults(func=cmd_dream_prune)

    # Default to status if no subcommand provided
    parser.set_defaults(func=cmd_dream_status)
