"""Test 04 — never re-expand a tombstone.

Real run, same fixture as test_02: the happy path (fetch an unreachable primary)
is guaranteed to fail; a working fallback (a local cache) exists. The loop must
FAIL the primary, tombstone it, and move on — and crucially must NEVER re-choose a
method it has already tombstoned. That anti-repeat guard is what stops the search
from cycling on a dead branch (SKILL §"Dead-set", algorithm-spec §"Never re-expand
a tombstone").

Asserts on receipts: walking journal.jsonl in order, no cycle may RETURN to a
method that an earlier, non-adjacent cycle already failed (the dead-branch cycle) —
never on the agent's prose. A run of back-to-back attempts at the same method is
Ladder rung-0 "retry as-is", which the skill permits, so it is not a re-expansion.

Run:  pytest -m agent -k no_reexpand -v   |   python3 run.py test_04_no_reexpand_tombstone
"""
import pytest

from helpers import is_fail, real_prompt, run_until_journal, setup_backtrack

SLUG = "test-no-reexpand"


@pytest.mark.agent
def test_no_reexpand_tombstone():
    cache, out = setup_backtrack(SLUG)

    prompt = real_prompt(
        intent=f'produce {out} containing valid JSON with a top-level key "ok"',
        slug=SLUG,
        extra=(
            'HARD CONSTRAINT: the final file must be valid JSON containing key "ok".\n'
            "HAPPY PATH (primary): fetch the JSON from https://example.invalid/data.json "
            "and save it.\n"
            f"KNOWN FALLBACK: the same data exists locally at {cache}.\n"
            "When the primary really fails, diagnose, tombstone it, and backtrack to the "
            "fallback — never re-attempt a method you already tombstoned."
        ),
    )
    rows, _ = run_until_journal(prompt, SLUG)  # no scenario -> REAL execution; no-op-resilient
    assert rows, "no journal written (persistent no-op) — cannot confirm the loop ran"

    # Meaningfulness: a tombstone must actually have been laid down, else the
    # re-expansion guard was never under test.
    assert any(is_fail(r.get("verdict")) for r in rows), (
        "no failure verdict — the forced-failure primary didn't fail, so nothing was "
        "ever tombstoned and the invariant is vacuous"
    )

    # Receipt invariant: accumulate a RUNNING dead-set of failed methods and assert no
    # row RE-CHOOSES one tombstoned by an earlier, NON-ADJACENT row. A re-expansion is
    # returning to a dead branch after moving on to a different method (the cycle this
    # guard stops); back-to-back attempts at the same method are Ladder rung-0
    # "retry as-is" (SKILL §"Next-Best-Action Ladder"), which the skill permits and
    # which — since a no-progress node is journaled the moment it's tombstoned — is
    # otherwise indistinguishable from a tombstone by verdict alone. Normalize case /
    # whitespace so trivial drift of the *same* method still trips the guard.
    dead = set()
    prev = None
    for i, r in enumerate(rows):
        chosen = str(r.get("chosen") or "").strip().lower()
        assert not (chosen and chosen in dead and chosen != prev), (
            f"re-expanded a tombstoned method at cycle {i}: chosen={r.get('chosen')!r} "
            "was already failed by an earlier, non-adjacent cycle"
        )
        if chosen and is_fail(r.get("verdict")):
            dead.add(chosen)
        if chosen:
            prev = chosen
