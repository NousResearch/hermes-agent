"""Regressions for ``kanban_db.reopen_task`` — the done-recovery verb (#99577).

Before this, a task stuck in ``done`` (a reviewer rejection that never got
recorded, or a forged ``UPDATE ... SET status='done'``) had no CLI path out:
``reopen-review`` / ``unblock`` / ``promote`` all refuse because the task is
not in their source lane, so operators reached for raw ``sqlite3``. These tests
pin the recovery contract:

* a ``done`` task reopens to its parent-gated landing status with an audited
  ``reopened`` event carrying the required reason + actor,
* the reason is mandatory (no silent/accidental reopen),
* only ``done`` tasks are eligible,
* descendants dispatched on the retracted result are re-gated (shared
  ``invalidate_descendants_for_parent_reopen`` semantics),
* a run left open under the forged completion (the issue's stale dead-PID
  ``running`` run) is reclaimed, restoring the runs invariant,
* ``dry_run`` validates without mutating.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def conn(tmp_path: Path):
    db = kb.connect(tmp_path / "kanban.db")
    try:
        yield db
    finally:
        db.close()


def _done_task(conn, **kw) -> str:
    tid = kb.create_task(conn, title=kw.pop("title", "t"), assignee="builder", **kw)
    assert kb.complete_task(conn, tid)
    return tid


def _reopened_event(conn, task_id: str):
    for ev in kb.list_events(conn, task_id):
        if ev.kind == "reopened":
            return ev
    return None


def test_reopen_done_moves_to_ready_and_writes_audited_event(conn):
    tid = _done_task(conn)
    assert kb.get_task(conn, tid).status == "done"

    ok, err = kb.reopen_task(conn, tid, actor="alice", reason="reviewer rejected")
    assert ok is True and err is None

    task = kb.get_task(conn, tid)
    assert task.status == "ready"          # no parents -> ready
    assert task.completed_at is None        # no longer complete
    assert task.current_run_id is None

    ev = _reopened_event(conn, tid)
    assert ev is not None, "a 'reopened' event must be recorded"
    assert ev.payload["reason"] == "reviewer rejected"
    assert ev.payload["actor"] == "alice"
    assert ev.payload["from_status"] == "done"
    assert ev.payload["status"] == "ready"


def test_reopen_requires_a_reason(conn):
    tid = _done_task(conn)
    ok, err = kb.reopen_task(conn, tid, actor="alice", reason="   ")
    assert ok is False
    assert err and "reason" in err
    # State untouched — the guard is the whole point (#99577).
    assert kb.get_task(conn, tid).status == "done"
    assert _reopened_event(conn, tid) is None


def test_reopen_refuses_a_non_done_task(conn):
    tid = kb.create_task(conn, title="t", assignee="builder")  # 'ready', not done
    ok, err = kb.reopen_task(conn, tid, actor="alice", reason="oops")
    assert ok is False
    assert err and "done" in err
    assert kb.get_task(conn, tid).status == "ready"


def test_reopen_refuses_unknown_task(conn):
    ok, err = kb.reopen_task(conn, "t_missing", actor="alice", reason="x")
    assert ok is False
    assert err and "not found" in err


def test_reopen_dry_run_does_not_mutate(conn):
    tid = _done_task(conn)
    ok, err = kb.reopen_task(conn, tid, actor="alice", reason="peek", dry_run=True)
    assert ok is True and err is None
    assert kb.get_task(conn, tid).status == "done"
    assert _reopened_event(conn, tid) is None


def test_reopen_invalidates_done_descendants(conn):
    parent = _done_task(conn, title="parent")
    child = kb.create_task(conn, title="child", assignee="builder", parents=[parent])
    assert kb.complete_task(conn, child)
    assert kb.get_task(conn, child).status == "done"

    ok, err = kb.reopen_task(conn, parent, actor="alice", reason="redo parent")
    assert ok is True and err is None

    # The child was built on the parent's now-retracted result -> re-gated.
    child_task = kb.get_task(conn, child)
    assert child_task.status == "todo"
    assert child_task.completed_at is None


def test_reopen_reclaims_a_run_left_open_by_a_forged_completion(conn):
    # Simulate the issue's forged completion: a running claim raw-UPDATEd to
    # 'done' without closing the run, leaving a dangling running run + claim.
    tid = kb.create_task(conn, title="t", assignee="builder")
    claimed = kb.claim_task(conn, tid)
    assert claimed is not None and kb.get_task(conn, tid).status == "running"
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (tid,))

    # Precondition: the run is still open under the forged 'done'.
    open_before = conn.execute(
        "SELECT COUNT(*) FROM task_runs WHERE task_id = ? AND ended_at IS NULL",
        (tid,),
    ).fetchone()[0]
    assert open_before == 1

    ok, err = kb.reopen_task(conn, tid, actor="alice", reason="dead worker")
    assert ok is True and err is None

    task = kb.get_task(conn, tid)
    assert task.status in {"ready", "todo"}
    assert task.current_run_id is None
    assert task.claim_lock is None
    # The dangling run is closed — no run left running/open for this task.
    open_after = conn.execute(
        "SELECT COUNT(*) FROM task_runs WHERE task_id = ? AND ended_at IS NULL",
        (tid,),
    ).fetchone()[0]
    assert open_after == 0
