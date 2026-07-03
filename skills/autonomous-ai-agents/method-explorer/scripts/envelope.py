#!/usr/bin/env python3
"""envelope.py — the canonical method-explorer invocation contract (single source).

The prompt framing that makes a bare `hermes -z` turn engage the method-explorer skill
and write its artifacts (plan-tree.md + journal.jsonl) where drivers and parsers expect
them. In-repo consumers (tests/helpers.py real_prompt/sim_prompt, scenario_builder) build
from here; external callers (e.g. the relentless-solve skill) keep their own runtime copy
and pin equality with a contract test against this module. test_00 / test_c1 keep their
deliberately-worded inline variants — they pin specific phrasings, not this contract.

Stdlib-only; no env reads, no imports.
"""

JSONL_CLAUSE = "(valid JSONL — one compact object per line, newline-separated)"

SKILL_HEAD_REAL = (
    "Use the method-explorer skill: skill_view to load it, then follow it "
    "INCLUDING the lean Decision Records (predict -> act -> reconcile) discipline. "
    "This is a REAL run (Simulation Mode OFF)."
)

SKILL_HEAD_SIM = (
    "Use the method-explorer skill: skill_view to load it, then follow it "
    "INCLUDING Simulation Mode and the lean Decision Records (predict -> act -> "
    "reconcile) discipline. SIMULATION MODE IS ACTIVE — read $HERMES_SIM_SCENARIO "
    "and take each stage's declared outcome; do NOT perform real actions."
)


def marker_map_block(plans_dir, slug, jsonl_clause=JSONL_CLAUSE):
    """The artifact-contract paragraph: compact marker-map plan-tree + lean JSONL journal.
    `jsonl_clause=None` ends at 'journal.jsonl.' so a caller can append a stricter clause."""
    tail = f" {jsonl_clause}." if jsonl_clause else "."
    return (
        f"Write the plan-tree to {plans_dir}/{slug}/plan-tree.md as a COMPACT MARKER MAP "
        "(STATE header; INTENT/constraints; a NODES list with markers ○/▶/✝/✓ + a "
        "one-line receipt per node; a FRONTIER line; on GUARD-HALT also one body line "
        "'GUARD-HALT: <which guard fired, open branches, smallest bump to continue>' "
        "under the header) — no Branch-log/Decision-log. "
        "Append ONE lean single-line JSON record per cycle (fields: node, q, chosen, "
        f"expected, verdict, evidence, next) to {plans_dir}/{slug}/journal.jsonl{tail}"
    )


def real_prompt(intent, slug, plans_dir, extra=""):
    """Standard REAL-mode prompt (Simulation Mode off) writing artifacts to the slug."""
    mid = f"{extra}\n" if extra else ""
    return f"{SKILL_HEAD_REAL}\nINTENT: {intent}\n{mid}{marker_map_block(plans_dir, slug)}"


def sim_prompt(intent, slug, plans_dir, extra=""):
    """Standard Simulation-Mode prompt that writes artifacts into the slug sandbox."""
    mid = f"{extra}\n" if extra else ""
    return f"{SKILL_HEAD_SIM}\nINTENT: {intent}\n{mid}{marker_map_block(plans_dir, slug)}"
