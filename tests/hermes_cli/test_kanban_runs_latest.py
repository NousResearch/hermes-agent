"""Board-wide "latest run per task" query (``latest_runs_by_state``).

Exists so a consumer needing one run per task does not launch one process
per task. The contract it has to hold is equivalence: the row it returns for
a task must be the same row a per-task ``list_runs(...)[-1]`` would pick,
because callers are migrating from exactly that loop.
"""
from __future__ import annotations

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def board(tmp_path):
    db = tmp_path / "kanban.db"
    kb.init_db(db)
    conn = kb.connect(db)
    yield conn
    conn.close()


def _finished_task(conn, title, *, rounds):
    """Create a task and drive it through ``rounds`` claim/close cycles.

    ``rounds`` is a list of callables taking (conn, task_id).
    """
    tid = kb.create_task(conn, title=title, assignee="w")
    for close in rounds:
        kb.claim_task(conn, tid)
        close(conn, tid)
    conn.commit()
    return tid


def _block(conn, tid):
    kb.block_task(conn, tid, reason="r")


def _complete(conn, tid):
    kb.complete_task(conn, tid, summary="s")


def test_matches_per_task_list_runs_last(board):
    """The equivalence the query is a replacement for."""
    tid = _finished_task(board, "multi", rounds=[_block, _complete])

    per_task = [r for r in kb.list_runs(board, tid) if r.status == "done"][-1]
    board_wide = kb.latest_runs_by_state(
        board, state_type="status", state_name="done"
    )[tid]

    assert board_wide.id == per_task.id


def test_picks_the_most_recent_matching_run(board):
    """Two runs reach the same state; the later one wins.

    Uses ``blocked`` rather than ``done`` because ``done`` is terminal — a
    completed task cannot be re-claimed, so it can only ever carry one done
    run. ``blocked`` is the state a task can genuinely re-enter.
    """
    tid = kb.create_task(board, title="twice", assignee="w")
    for i in range(3):
        kb.claim_task(board, tid)
        kb.block_task(board, tid, reason=f"r{i}")
        kb.unblock_task(board, tid)
    board.commit()

    runs = [r for r in kb.list_runs(board, tid) if r.status == "blocked"]
    assert len(runs) > 1, [r.status for r in kb.list_runs(board, tid)]

    got = kb.latest_runs_by_state(
        board, state_type="status", state_name="blocked"
    )
    assert got[tid].id == runs[-1].id


def test_earlier_run_in_another_state_is_not_returned(board):
    """A task that was blocked and then completed must answer with the
    completed run for status=done, and the blocked run for status=blocked —
    the query filters, it does not just take the last run."""
    tid = _finished_task(board, "mixed", rounds=[_block, _complete])

    done = kb.latest_runs_by_state(board, state_type="status", state_name="done")
    blocked = kb.latest_runs_by_state(
        board, state_type="status", state_name="blocked"
    )
    assert done[tid].status == "done"
    assert blocked[tid].status == "blocked"
    assert done[tid].id != blocked[tid].id


def test_task_with_no_matching_run_is_absent(board):
    """Absent from the mapping rather than mapped to None, so callers can
    use ``in`` without a second None check."""
    blocked_only = _finished_task(board, "blocked only", rounds=[_block])

    got = kb.latest_runs_by_state(board, state_type="status", state_name="done")
    assert blocked_only not in got


def test_covers_every_task_on_the_board(board):
    """The point of the call is one query for the whole board, so a task
    must not be missed just because another task has more runs."""
    a = _finished_task(board, "a", rounds=[_complete])
    b = _finished_task(board, "b", rounds=[_block, _complete])
    c = _finished_task(board, "c", rounds=[_complete, _block, _complete])

    got = kb.latest_runs_by_state(board, state_type="status", state_name="done")
    assert set(got) == {a, b, c}
    for tid in (a, b, c):
        expected = [r for r in kb.list_runs(board, tid) if r.status == "done"][-1]
        assert got[tid].id == expected.id, tid


def test_outcome_axis_is_selectable(board):
    """status and outcome are different columns — the caller says which."""
    tid = _finished_task(board, "outcome", rounds=[_complete])

    by_outcome = kb.latest_runs_by_state(
        board, state_type="outcome", state_name="completed"
    )
    assert tid in by_outcome
    # 'completed' is an outcome, never a status — asking on the wrong axis
    # must not silently match.
    by_status = kb.latest_runs_by_state(
        board, state_type="status", state_name="completed"
    )
    assert by_status == {}


def test_rejects_an_unknown_state_type(board):
    with pytest.raises(ValueError):
        kb.latest_runs_by_state(board, state_type="not_a_column", state_name="done")
