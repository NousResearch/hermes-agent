"""Test 02 — backtrack reaches success (doesn't dead-end).

Real run: the happy path (fetch an unreachable primary) is guaranteed to fail, but
a working fallback (a local cache) exists. The loop must FAIL the primary, backtrack,
and still satisfy the intent — proving it recovers instead of dead-ending.

Asserts on receipts: a failure verdict precedes a success in the journal, and the
final file actually exists and is valid JSON with key "ok".

Run:  pytest -m agent -k backtrack -v   |   python3 run.py test_02_backtrack_success
"""
import json

import pytest

from helpers import (
    backtrack_extra,
    is_fail,
    is_succ,
    read_file,
    real_prompt,
    run_until_journal,
    setup_backtrack,
)

SLUG = "test-backtrack"


@pytest.mark.agent
def test_backtrack_reaches_success():
    cache, out = setup_backtrack(SLUG)

    prompt = real_prompt(
        intent=f'produce {out} containing valid JSON with a top-level key "ok"',
        slug=SLUG,
        extra=backtrack_extra(cache),
    )
    rows, _ = run_until_journal(prompt, SLUG)  # no scenario -> REAL execution; no-op-resilient
    verdicts = [r.get("verdict") for r in rows]

    assert rows, "no journal written (persistent no-op) — cannot confirm the loop ran"
    assert any(is_fail(v) for v in verdicts), (
        "no failure verdict — the forced-failure happy path didn't fail, so backtrack "
        "was never exercised"
    )
    assert any(is_succ(v) for v in verdicts), "never reached a success verdict"

    first_success = next(i for i, v in enumerate(verdicts) if is_succ(v))
    assert any(is_fail(v) for v in verdicts[:first_success]), (
        "success was not preceded by a failure — no real backtrack happened"
    )

    # Receipt: the deliverable actually exists and satisfies the intent.
    content = read_file(out)
    assert content is not None, f"no output file produced at {out}"
    data = json.loads(content)
    assert "ok" in data, f"output JSON missing key 'ok': {content!r}"
