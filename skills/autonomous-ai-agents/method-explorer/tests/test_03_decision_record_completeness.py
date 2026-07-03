"""Test 03 — decision-record completeness (the show-your-work actually emits).

Real run on the same unreachable-primary + local-cache fixture as test_02. The
behavior under test is the §"Decision Records — predict -> act -> reconcile"
discipline: every loop cycle must leave a *complete*, evidence-based record, not
a story told after the fact. We assert on the journal receipts — never on the
agent's prose/self-report.

Every row must carry the lean record (node, q, chosen, expected, verdict) and an
evidence receipt. A row may omit a concrete evidence receipt ONLY when it openly
marks the verdict UNVERIFIED — the skill's honest escape hatch when no receipt can
be cited. (The verbose candidates/rationale/confidence fields were dropped in the
consolidation; deferred options now live once in the plan-tree FRONTIER.)

Run:  pytest -m agent -k completeness -v   |   python3 run.py test_03_decision_record_completeness
"""
import pytest

from helpers import (
    assert_record_complete,
    backtrack_extra,
    real_prompt,
    run_until_journal,
    setup_backtrack,
)

SLUG = "test-dr-complete"


@pytest.mark.agent
def test_every_cycle_emits_a_complete_decision_record():
    cache, out = setup_backtrack(SLUG)

    prompt = real_prompt(
        intent=f'produce {out} containing valid JSON with a top-level key "ok"',
        slug=SLUG,
        extra=backtrack_extra(cache),
    )
    rows, _ = run_until_journal(prompt, SLUG)  # no scenario -> REAL execution; no-op-resilient
    assert rows, "no journal written (persistent no-op) — cannot confirm the loop ran"

    # Lean completeness: every cycle has node, q, chosen, expected, verdict, and an
    # evidence receipt OR an UNVERIFIED verdict. (Shared with the other invariant tests.)
    assert_record_complete(rows)
