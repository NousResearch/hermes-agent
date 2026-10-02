"""C4 summarizer faults: a failed summary never commits, and failure is bounded and visible.

Same real chain and per-turn invariants as ``test_compaction_auto`` (see ``_helpers``), with the
summarizer scripted to answer empty / refusal / truncated (finish_reason=length) / slower than the
compression timeout / HTTP 500. For every fault: the session keeps running, the summarizer is asked a
bounded number of times per turn, the user is told, and state.db never archives history behind a bad
summary (a 500 may commit only the designed deterministic fallback -- never with
``abort_on_summary_failure``).
"""

from __future__ import annotations

import pytest

from tests.e2e.core.compaction._helpers import REFUSAL_TEXT, SEEDS, TRUNCATED_TEXT, db_rows, drive, warned

FAILING_MODES = ("empty", "refusal", "truncated", "slow")


@pytest.mark.parametrize("mode", FAILING_MODES)
@pytest.mark.parametrize("seed", SEEDS[:2])
def test_failed_summary_never_commits(make_scenario, tmp_path, seed, mode):
    sc = make_scenario(mode)
    drive(sc, seed, tmp_path / "work")
    assert sc.summary_calls, "the seeded session never crossed the compaction trigger"
    assert sc.committed_compactions() == 0, f"a {mode} summary committed a compaction (history archived)"
    bad = {"refusal": REFUSAL_TEXT, "truncated": TRUNCATED_TEXT}.get(mode)
    if bad:
        for rec in sc.turns:
            for body in rec.main_requests:
                assert not any(bad in str(m.get("content")) for m in body["messages"]), (
                    f"the {mode} summary text reached the model")
        assert not any(bad in (r["content"] or "") for r in db_rows(sc.db_path, sc.session_id)), (
            f"the {mode} summary text was written to state.db")
    # The session is over the trigger with no usable summary: the user must be told.
    assert warned(sc), f"{mode} summarizer: compaction failed silently (no warning surfaced)"


@pytest.mark.parametrize("seed", SEEDS[:1])
def test_http500_summarizer_is_bounded_and_lossless(make_scenario, tmp_path, seed):
    """A 500 may abort or commit the deterministic fallback (``abort_on_summary_failure: false``), but
    either way nothing is lost and it is surfaced."""
    sc = make_scenario("http500")
    drive(sc, seed, tmp_path / "work")
    assert sc.summary_calls
    assert warned(sc)


@pytest.mark.parametrize("seed", SEEDS[1:2])
def test_http500_with_abort_on_failure_never_commits(make_scenario, tmp_path, seed):
    sc = make_scenario("http500", extra="  abort_on_summary_failure: true\n")
    drive(sc, seed, tmp_path / "work")
    assert sc.summary_calls
    assert sc.committed_compactions() == 0
