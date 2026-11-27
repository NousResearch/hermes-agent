"""C4 manual compaction: ``/compress``, ``/compress here N`` and rolling micro-compaction, end to end.

``/compress`` goes through the TUI/Desktop choke point (``tui_gateway`` ``_compress_session_history`` ->
``compress_now``) on a real AIAgent + SessionDB after a seeded session; micro-compaction runs from the
real turn finalizer. After the compaction the same invariants as the automatic suite are asserted:

* no user message is lost from state.db (live, or archived as summarized history);
* the live rows equal the host's in-memory history (what the TUI keeps and sends next);
* the next turn from the in-memory history AND the next turn from a fresh agent resumed off state.db
  (the gateway shape) both send exactly the persisted history (+ the new message);
* a failed summary never commits: history and state.db stay byte-identical.
"""

from __future__ import annotations

import threading

import pytest

from tests.e2e.core.compaction._helpers import (
    GOOD_SUMMARY_TOKEN,
    MARKER_RE,
    Scenario,
    active_rows,
    assert_db_matches_history,
    assert_db_recoverable,
    db_rows,
    generate_transcript,
    norm_list,
    _row_msg,
)

# Manual tests keep the automatic trigger out of the way so only the command under test compacts.
HIGH_THRESHOLD = 60_000
PRE_TURNS = 6
POST_TURNS = 2


def _tui_session(sc: Scenario) -> dict:
    return {
        "agent": sc.agent, "history": list(sc.history), "history_lock": threading.Lock(),
        "history_version": 1, "session_key": sc.session_id,
    }


def _compress_via_tui(sc: Scenario, args: str) -> int:
    from tui_gateway.server import _compress_session_history

    session = _tui_session(sc)
    sc.install()
    removed, _usage = _compress_session_history(session, args)
    sc.history = session["history"]
    return removed


def _live(sc: Scenario) -> list[tuple]:
    return norm_list(_row_msg(r) for r in active_rows(db_rows(sc.db_path, sc.session_id)))


def _run_after(sc: Scenario, specs, where: str) -> None:
    """In-memory continuation, then a fresh agent resumed from state.db: both send persisted history."""
    rows = db_rows(sc.db_path, sc.session_id)
    assert_db_recoverable(rows, sc.sent_markers, where)
    assert_db_matches_history(rows, sc.history, where)
    for spec in specs[:-1]:
        sc.run_turn(spec)
    sc.resume_from_db()
    sc.run_turn(specs[-1])


# Each boundary form once; seeds alternate so both seeded session shapes meet full and partial compress.
@pytest.mark.parametrize(("seed", "args"), [
    (5, ""), (17, ""), (5, "here"), (17, "here 1"), (5, "here 2"), (17, "here 3"), (5, "keep the file names"),
])
def test_manual_compress_keeps_every_invariant(make_scenario, tmp_path, seed, args):
    sc = make_scenario("good", threshold_tokens=HIGH_THRESHOLD)
    specs = generate_transcript(seed, PRE_TURNS + POST_TURNS + 1, tmp_path / "work",
                                kinds=("tools", "parallel", "chat", "image"))
    for spec in specs[:PRE_TURNS]:
        sc.run_turn(spec)
    before_live = _live(sc)
    calls_before = len(sc.summary_calls)

    removed = _compress_via_tui(sc, args)

    assert len(sc.summary_calls) - calls_before <= 2, "manual compress made a runaway number of summarizer calls"
    assert removed > 0 and sc.committed_compactions() > 0, f"/compress {args!r} did not compact"
    assert _live(sc) != before_live
    assert any(GOOD_SUMMARY_TOKEN in str(m.get("content")) for m in sc.history), "summary missing from history"
    if args.startswith("here"):
        # The kept exchanges (the newest N user turns) stay verbatim and live.
        n = int(args.split()[1]) if len(args.split()) > 1 else 1
        kept = [s.marker for s in specs[:PRE_TURNS]][-n:]
        live_user_rows = [r for r in active_rows(db_rows(sc.db_path, sc.session_id)) if r["role"] == "user"]
        for marker in kept:
            assert any(MARKER_RE.findall(r["content"] or "") == [marker] for r in live_user_rows), (
                f"/compress {args}: kept exchange {marker} is not a live verbatim row in state.db")
    _run_after(sc, specs[PRE_TURNS:], f"after /compress {args!r}")


@pytest.mark.parametrize("mode", ("empty", "refusal", "truncated"))
@pytest.mark.parametrize("args", ["", "here 2"])
def test_manual_compress_with_failed_summary_changes_nothing(make_scenario, tmp_path, mode, args):
    sc = make_scenario(mode, threshold_tokens=HIGH_THRESHOLD)
    specs = generate_transcript(7, PRE_TURNS + POST_TURNS + 1, tmp_path / "work", kinds=("tools", "parallel", "chat"))
    for spec in specs[:PRE_TURNS]:
        sc.run_turn(spec)
    rows_before = db_rows(sc.db_path, sc.session_id)
    history_before = list(sc.history)

    removed = _compress_via_tui(sc, args)

    assert sc.summary_calls, "the summarizer was never asked"
    assert removed == 0
    assert db_rows(sc.db_path, sc.session_id) == rows_before, f"a {mode} summary changed state.db"
    assert norm_list(sc.history) == norm_list(history_before), f"a {mode} summary changed the live history"
    _run_after(sc, specs[PRE_TURNS:], f"after failed /compress {args!r}")


@pytest.mark.parametrize("seed", (3, 29))
def test_micro_compaction_keeps_every_invariant(make_scenario, tmp_path, seed):
    """Rolling micro-compaction folds one old exchange per turn into a summary marker between turns,
    carrying a non-contiguous prefix + suffix; summarized rows must be archived, carried rows kept."""
    sc = make_scenario("good", threshold_tokens=HIGH_THRESHOLD, extra="  micro_compact: true\n")
    specs = generate_transcript(seed, 10, tmp_path / "work", kinds=("tools", "parallel", "chat"))
    for i, spec in enumerate(specs):
        if i == 6:
            sc.resume_from_db()
        sc.run_turn(spec)
    assert sc.summary_calls, "micro-compaction never ran"
    assert sc.committed_compactions() > 0, "micro-compaction never archived a summarized exchange"
