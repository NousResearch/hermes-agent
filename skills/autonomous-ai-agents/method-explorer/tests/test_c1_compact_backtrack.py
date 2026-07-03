"""Gate C1 — consolidated compact format: marker plan-tree + lean journal.

Validates the consolidation refactor BEFORE migrating the rest of the suite: does the
model, given the rewritten SKILL.md, actually PRODUCE the compact format *and* still
backtrack to success?

  - plan-tree.md is a COMPACT MARKER MAP (STATE header + NODES with ✝/✓/▶/○ markers),
    NOT the old verbose Branch-log / Decision-log layout.
  - journal.jsonl is LEAN (node, q, chosen, expected, verdict, evidence, next) — the
    helper-compatible field names survive, so the rest of the suite needs no rewrite.
  - the loop still backtracks: a fail verdict precedes a success (the behavior).

Self-contained on purpose (its own prompt, no shared-helper edits) so running it can't
disturb the in-flight suite. If this passes, promote the shared prompts in Phase 2.

Run:  python3 run.py test_c1_compact_backtrack
"""
import pytest

from helpers import (
    PLANS,
    deploy_scenario,
    is_fail,
    is_succ,
    read_file,
    run_until_journal,
    setup_sandbox,
)
from scenario_builder import build_scenario, canonical_backtrack_methods, method_list_prompt

SLUG = "test-c1-compact"

METHODS = canonical_backtrack_methods()  # alfa✗ bravo✗ charlie→delta (shared)
MEANINGS = {"alfa": "primary source", "bravo": "mirror",
            "charlie": "cache", "delta": "verify"}


def _lean_prompt():
    desc = "; ".join(f"{m.tag} = {MEANINGS[m.tag]}" for m in METHODS)
    return "\n".join([
        "Use the method-explorer skill: skill_view to load it, then follow it "
        "INCLUDING Simulation Mode and the lean Decision-Records discipline.",
        "SIMULATION MODE IS ACTIVE (read $HERMES_SIM_SCENARIO); do NOT perform real "
        "actions — take each node's declared outcome.",
        "INTENT: produce a data file with a top-level key \"ok\" (network preferred).",
        f"Methods, in preference order: {desc}.",
        method_list_prompt(METHODS),
        # Compact plan-tree (the format under test):
        f"Write the plan-tree to {PLANS}/{SLUG}/plan-tree.md as a COMPACT MARKER MAP, "
        "exactly per the skill's 'Plan-Tree Artifact' template: a `STATE:` header, "
        "INTENT/constraints, and a `NODES` list where each node carries a status marker "
        "(○ open · ▶ active · ✝ dead · ✓ done) and a one-line receipt/reason, plus a "
        "`FRONTIER:` line. Do NOT write Branch-log or Decision-log sections.",
        # Lean journal (helper-compatible field names):
        f"Append ONE lean single-line JSON record per cycle to {PLANS}/{SLUG}/journal.jsonl "
        "with fields: node, q, chosen, expected, verdict, evidence, next. Valid JSONL — "
        "one compact object per line, newline-separated; never pretty-print or concatenate.",
    ])


@pytest.mark.agent
def test_compact_format_and_backtrack():
    setup_sandbox(SLUG)
    scen = build_scenario("produce data with key ok (network preferred)", METHODS,
                          notes="Gate C1: compact-format backtrack")
    cont = deploy_scenario("c1-compact-backtrack.json", scen)

    rows, _ = run_until_journal(_lean_prompt(), SLUG, scenario=cont)
    assert rows, "persistent no-op (empty journal after retries)"

    verdicts = [r.get("verdict") for r in rows]

    # Behavior: still backtracks to success (a fail precedes a success).
    assert any(is_succ(v) for v in verdicts), "never reached a success/progress verdict"
    assert any(is_fail(v) for v in verdicts), "no failure verdict — backtrack not exercised"
    first_succ = next(i for i, v in enumerate(verdicts) if is_succ(v))
    assert any(is_fail(v) for v in verdicts[:first_succ]), \
        "success not preceded by a failure — no real backtrack"

    # Lean journal: helper-compatible fields present (so the rest of the suite still parses).
    for i, r in enumerate(rows):
        for field in ("node", "chosen", "verdict"):
            assert str(r.get(field) or "").strip(), f"row {i}: missing lean field {field!r}"

    # Format: plan-tree is the COMPACT MARKER MAP, not the old verbose layout.
    pt = read_file(f"{PLANS}/{SLUG}/plan-tree.md") or ""
    assert pt.strip(), "no plan-tree written"
    assert "STATE:" in pt, "plan-tree missing the STATE header (not the compact template)"
    assert any(g in pt for g in ("✝", "✓", "▶", "○")), \
        "plan-tree has no status markers (not the compact marker map)"
    low = pt.lower()
    assert "branch log" not in low and "decision-log" not in low, \
        "plan-tree still uses the old verbose Branch-log/Decision-log layout"
