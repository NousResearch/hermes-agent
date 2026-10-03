"""Tests for ``kanban_db.reopen_task`` — the done -> blocked/todo/ready undo primitive.

``complete_task`` commits before any observer can react (kanban hooks are
observer-only; ``block_task``/``request_review`` only transition running/ready),
so a completed card cannot be taken backward through the public API. These
tests pin the domain implementation that fills the gap (M-reopen):

* a done card reopens to ``blocked``/``todo``/``ready`` in one txn,
* ``completed_at`` and ``result`` are cleared — the card no longer claims done,
* ``landing='blocked'`` is STICKY (newest blocked/unblocked event is blocked)
  so ``recompute_ready`` never auto-promotes the veto back into the pool,
* descendants promoted/completed on the card's result are invalidated
  (demoted to ``todo`` with an ancestor-naming comment), running descendants
  terminate strictly post-commit (audit trail first),
* ``consecutive_failures`` resets (deliberate operator/veto action), and
* non-done sources are refused (idempotent CAS); invalid landings/kinds raise.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def conn(tmp_path: Path):
    db = kbc.connect(tmp_path / "kanban.db")
    try:
        yield db
    finally:
        db.close()


def _done_parent_with_done_child(conn, *, parent_title="ancestor"):
    parent_id = kb.create_task(conn, title=parent_title, assignee="planner")
    assert kb.complete_task(conn, parent_id)
    child_id = kb.create_task(
        conn, title="child", assignee="builder", parents=[parent_id],
    )
    assert kb.complete_task(conn, child_id)
    return parent_id, child_id


def _blocked_events(conn, task_id):
    return [
        e for e in kb.list_events(conn, task_id)
        if e.kind in ("blocked", "unblocked")
    ]


def _status_events(conn, task_id):
    return [e for e in kb.list_events(conn, task_id) if e.kind == "status"]


def test_veto_landing_blocked_is_sticky_and_clears_completion(conn):
    parent_id, child_id = _done_parent_with_done_child(conn)

    assert kb.reopen_task(
        conn, parent_id, reason="verify-gate: tests failed",
        landing="blocked", kind="needs_input", author="verify-gate",
    ) is True

    task = kb.get_task(conn, parent_id)
    assert task is not None and task.status == "blocked"
    assert task.completed_at is None
    assert task.result is None
    assert task.block_kind == "needs_input"
    assert task.block_recurrences == 1
    assert task.consecutive_failures == 0

    # Sticky: newest blocked/unblocked event is 'blocked', so recompute_ready
    # must NOT promote the veto back into the pool.
    events = _blocked_events(conn, parent_id)
    assert events and events[-1].kind == "blocked"
    payload = events[-1].payload or {}
    assert payload.get("reason") == "verify-gate: tests failed"
    assert payload.get("kind") == "needs_input"
    assert payload.get("source_status") == "done"
    kb.recompute_ready(conn)
    assert kb.get_task(conn, parent_id).status == "blocked"

    # The child that completed on this card's result is retracted.
    child = kb.get_task(conn, child_id)
    assert child is not None and child.status == "todo"
    assert child.completed_at is None


def test_reopen_ready_returns_to_pool_with_status_event(conn):
    parent_id, _child_id = _done_parent_with_done_child(conn)

    assert kb.reopen_task(
        conn, parent_id, reason="wrongly closed; redo", landing="ready",
        author="operator",
    ) is True

    task = kb.get_task(conn, parent_id)
    assert task is not None and task.status == "ready"
    assert task.completed_at is None
    assert task.result is None
    assert task.block_kind is None

    st = _status_events(conn, parent_id)
    assert st and st[-1].kind == "status"
    payload = st[-1].payload or {}
    assert payload.get("status") == "ready"
    assert payload.get("requested_status") == "ready"
    assert payload.get("source_status") == "done"
    assert payload.get("reason") == "wrongly closed; redo"


def test_reopen_landing_todo(conn):
    task_id = kb.create_task(conn, title="leaf", assignee="builder")
    assert kb.complete_task(conn, task_id)

    assert kb.reopen_task(conn, task_id, landing="todo") is True
    task = kb.get_task(conn, task_id)
    assert task is not None and task.status == "todo"
    # A dependency-free reopened card is promoted by the next recompute.
    assert kb.recompute_ready(conn) == 1
    assert kb.get_task(conn, task_id).status == "ready"


def test_reopen_refuses_non_done_and_unknown(conn):
    ready_id = kb.create_task(conn, title="not done", assignee="builder")
    assert kb.reopen_task(conn, ready_id, landing="blocked") is False
    assert kb.get_task(conn, ready_id).status == "ready"

    assert kb.reopen_task(conn, "t_no_such_task", landing="blocked") is False

    done_id = kb.create_task(conn, title="done once", assignee="builder")
    assert kb.complete_task(conn, done_id)
    assert kb.reopen_task(conn, done_id, landing="blocked") is True
    # Idempotent CAS: a second veto is a no-op, not an error.
    assert kb.reopen_task(conn, done_id, landing="blocked") is False


def test_reopen_clears_prior_failure_counter(conn):
    task_id = kb.create_task(conn, title="flaky card", assignee="builder")
    assert kb.complete_task(conn, task_id)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET consecutive_failures = 4 WHERE id = ?", (task_id,),
        )

    assert kb.reopen_task(conn, task_id, landing="ready") is True
    assert kb.get_task(conn, task_id).consecutive_failures == 0


def test_running_descendant_event_precedes_termination(tmp_path, monkeypatch):
    db = kbc.connect(tmp_path / "kanban.db")
    try:
        parent_id = kb.create_task(conn=db, title="ancestor", assignee="planner")
        assert kb.complete_task(db, parent_id)
        child_id = kb.create_task(
            db, title="running child", assignee="builder", parents=[parent_id],
        )
        claimed = kb.claim_task(db, child_id)
        assert claimed is not None and claimed.status == "running"
        kbd._set_worker_pid(db, child_id, 424242)

        kills: list[tuple] = []

        def fake_terminate(pid, claim_lock, **kwargs):
            # The audit trail must already be durable when the kill fires:
            # standalone calls commit before terminating.
            side = kbc.connect(tmp_path / "kanban.db")
            try:
                kinds = [e.kind for e in kb.list_events(side, child_id)]
            finally:
                side.close()
            assert "descendant_invalidated" in kinds
            kills.append((pid, claim_lock))
            return {"terminated": True}

        monkeypatch.setattr(kb, "_terminate_reclaimed_worker", fake_terminate)

        assert kb.reopen_task(
            db, parent_id, reason="veto", landing="blocked", author="verify-gate",
        ) is True

        assert kills and kills[0][0] == 424242
        child = kb.get_task(db, child_id)
        assert child is not None
        assert child.status == "todo"
        assert child.current_run_id is None
        run = kb.latest_run(db, child_id)
        assert run is not None and run.outcome == "reclaimed"
    finally:
        db.close()


def test_reopen_invalid_arguments_raise(conn):
    task_id = kb.create_task(conn, title="any", assignee="builder")
    assert kb.complete_task(conn, task_id)
    with pytest.raises(ValueError):
        kb.reopen_task(conn, task_id, landing="archived")
    with pytest.raises(ValueError):
        kb.reopen_task(conn, task_id, landing="blocked", kind="not-a-kind")
    # Failed validation must not mutate the task.
    assert kb.get_task(conn, task_id).status == "done"


def test_reopen_fires_blocked_lifecycle_hook_when_landing_blocked(conn, monkeypatch):
    fired: list[tuple] = []

    def fake_fire_hook(event, task, task_id, run_id, **fields):
        fired.append((event, task_id, run_id, fields))

    monkeypatch.setattr(kb, "_fire_task_hook", fake_fire_hook)

    task_id = kb.create_task(conn, title="gate card", assignee="builder")
    assert kb.complete_task(conn, task_id)
    assert kb.reopen_task(
        conn, task_id, reason="veto", landing="blocked", author="verify-gate",
    ) is True
    assert any(
        ev == "kanban_task_blocked" and tid == task_id and rid is None
        for ev, tid, rid, _f in fired
    ), fired

    # Non-blocked landings fire no blocked lifecycle hook (they are not blocks);
    # the completion hook from complete_task is unrelated.
    fired.clear()
    task2 = kb.create_task(conn, title="redo card", assignee="builder")
    assert kb.complete_task(conn, task2)
    assert kb.reopen_task(conn, task2, landing="ready") is True
    assert [x for x in fired if x[0] == "kanban_task_blocked"] == []
