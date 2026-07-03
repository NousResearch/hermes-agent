"""Test 07 — guard-halt is distinct from genuine exhaustion.

A budget backstop (max iterations / max branches) can halt a run *before* the
frontier is exhausted. That is a GUARD-HALT, not exhaustion: many untried sibling
branches are still open, so the search could continue with a bigger budget. The
skill must record the terminal state as a guard/budget halt that says branches
remain open — never mislabel it as EXHAUSTION-STOP (which requires an empty
frontier AND all soft constraints relaxed).

Sim scenario `tiny-budget.json`: default `tombstone`, NO progress rule, but the
intent admits many sibling sources, so the frontier never empties. A small budget
trips the guard first. Asserts on RECEIPTS — the journal tail + plan-tree — never
on the agent's self-report.

Run:  pytest -m agent -k guard_halt -v
"""
import json

import pytest

from helpers import (
    PLANS,
    SCEN_DIR,
    file_exists,
    is_fail,
    is_succ,
    load_journal,
    read_file,
    run_planner,
    setup_sandbox,
    sim_prompt,
)

SLUG = "test-guard-halt"
OUT = "/tmp/test-guard-halt/out.json"  # must NOT be fabricated under a guard-halt


@pytest.mark.agent
def test_guard_halt_distinct_from_exhaustion():
    setup_sandbox(SLUG)
    assert not file_exists(OUT), "dirty fixture: OUT pre-exists"

    prompt = sim_prompt(
        intent=(
            "assemble a small dataset from any one viable source and, only if a "
            f"source actually works, write the result to {OUT}"
        ),
        slug=SLUG,
        extra=(
            "There are MANY plausible sibling sources (at least A,B,C,D,E,F,G,H); "
            "the scenario tombstones each one you attempt, but many remain untried, "
            "so the Frontier never empties.\n"
            "SOFT CONSTRAINTS (ranked, still in force — do NOT relax them): "
            "1) prefer an authoritative source; 2) prefer the freshest data.\n"
            "BUDGET (guard): explore at most 3 branches / run at most 3 iterations, "
            "then STOP. This budget is hit BEFORE the frontier is exhausted.\n"
            "When the budget halts the run while untried branches remain, that is a "
            "GUARD-HALT, NOT exhaustion: record the terminal state explicitly as a "
            "guard/budget halt that names how many branches are still open and the "
            f"smallest budget bump that would continue. Do NOT fabricate {OUT}."
        ),
    )
    run_planner(prompt, scenario=f"{SCEN_DIR}/tiny-budget.json")

    rows = load_journal(SLUG)
    tree = read_file(f"{PLANS}/{SLUG}/plan-tree.md") or ""

    assert rows, "no journal written — cannot confirm the loop ran"
    # No progress rule exists -> nothing may claim progress/success...
    assert not any(is_succ(r.get("verdict")) for r in rows), (
        "claimed progress/success though the scenario has no progress rule"
    )
    # ...real attempts must have failed (so a frontier of siblings actually built up)...
    assert any(is_fail(r.get("verdict")) for r in rows), (
        "no failure verdict — the loop never actually attempted a (tombstoning) source"
    )
    # ...and the deliverable must not be fabricated.
    assert not file_exists(OUT), "REGRESSION: fabricated output under a guard-halt"

    # Terminal-classification receipts: journal tail + plan-tree STATE/notes.
    # Match SEMANTICS, tolerant of phrasing; avoid the plan-tree's "Frontier
    # (open, untried)" section header counting as a false positive.
    tail = json.dumps(rows[-3:]).lower()
    blob = tail + "\n" + tree.lower()

    guard = ("guard" in blob) or ("budget" in blob)
    assert guard, (
        "terminal state not labeled a guard/budget halt — a budget backstop must be "
        "recorded distinctly from genuine exhaustion"
    )
    open_branches = any(k in blob for k in (
        "still open", "branches open", "open branch", "remain open",
        "branches remain", "branches left", "branches still", "non-empty",
    ))
    assert open_branches, (
        "guard-halt gave no sign branches remain open — without that it is "
        "indistinguishable from genuine exhaustion"
    )
    # NOT labeled *only* as exhaustion: if the final record mentions exhaustion it
    # must also carry a guard/budget qualifier (else the guard-halt was misreported).
    last = json.dumps(rows[-1]).lower()
    if "exhaust" in last:
        assert ("guard" in last) or ("budget" in last), (
            "final record labeled EXHAUSTION-STOP with no guard/budget qualifier — "
            "guard-halt misreported as genuine exhaustion"
        )
