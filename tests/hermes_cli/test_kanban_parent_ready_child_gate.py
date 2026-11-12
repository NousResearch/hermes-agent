"""A parent must not complete while a DIRECT child is still ``ready``.

The rule is deliberately narrow: ``ready`` is the only child status that means
dispatchable work is queued beneath a card that is about to declare itself done.
``todo`` must be released *by* the completion, ``blocked`` is parked for a human,
and ``running``/``review`` already passed dependency admission — blocking on any
of those would deadlock the board this gate exists to protect.

Every case drives the real ``complete_task`` against a scratch board, and the
last one pins the race: a child that becomes ``ready`` between the cheap
pre-check and the parent UPDATE must still be refused, because the invariant is
re-checked inside the write transaction.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def conn(tmp_path: Path):
    db = kbc.connect(tmp_path / "kanban.db")
    try:
        yield db
    finally:
        db.close()


def _parent(conn, title: str = "fan-in parent") -> str:
    """A task with no parents, so it sits ``ready`` and can complete."""
    return kb.create_task(conn, title=title, assignee="planner")


def _child(conn, parent_id: str, title: str = "queued work") -> str:
    """A child of ``parent_id``; ``todo`` because the parent is not done."""
    child_id = kb.create_task(conn, title=title, assignee="builder", parents=[parent_id])
    assert kb.get_task(conn, child_id).status == "todo"
    return child_id


def _force_status(conn, task_id: str, status: str) -> None:
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))


def _event_kinds(conn, task_id: str) -> list[str]:
    return [e.kind for e in kb.list_events(conn, task_id)]


def test_ready_child_blocks_parent_completion_and_names_it(conn):
    parent_id = _parent(conn)
    child_id = _child(conn, parent_id)
    _force_status(conn, child_id, "ready")

    with pytest.raises(kb.ChildrenNotTerminalError) as excinfo:
        kb.complete_task(conn, parent_id, result="shipped")

    assert child_id in str(excinfo.value)
    assert parent_id in str(excinfo.value)
    parent = kb.get_task(conn, parent_id)
    assert parent.status == "ready"  # nothing mutated
    assert parent.completed_at is None and parent.current_run_id is None
    assert kb.get_task(conn, child_id).status == "ready"
    assert "completed" not in _event_kinds(conn, parent_id)


def test_todo_child_does_not_block_and_is_promoted(conn):
    parent_id = _parent(conn)
    child_id = _child(conn, parent_id)

    assert kb.complete_task(conn, parent_id, result="shipped") is True

    assert kb.get_task(conn, parent_id).status == "done"
    # Completing the parent is what releases the dependency gate.
    assert kb.get_task(conn, child_id).status == "ready"


def test_blocked_child_does_not_block(conn):
    parent_id = _parent(conn)
    child_id = kb.create_task(
        conn, title="parked for a human", assignee="builder",
        parents=[parent_id], initial_status="blocked",
    )
    assert kb.get_task(conn, child_id).status == "blocked"

    assert kb.complete_task(conn, parent_id, result="shipped") is True
    assert kb.get_task(conn, parent_id).status == "done"


@pytest.mark.parametrize("child_status", ["running", "review"])
def test_in_flight_child_does_not_block(conn, child_status):
    parent_id = _parent(conn)
    child_id = _child(conn, parent_id)
    _force_status(conn, child_id, child_status)

    assert kb.complete_task(conn, parent_id, result="shipped") is True
    assert kb.get_task(conn, parent_id).status == "done"


@pytest.mark.parametrize("child_status", ["done", "archived"])
def test_terminal_child_does_not_block(conn, child_status):
    parent_id = _parent(conn)
    child_id = _child(conn, parent_id)
    _force_status(conn, child_id, child_status)

    assert kb.complete_task(conn, parent_id, result="shipped") is True
    assert kb.get_task(conn, parent_id).status == "done"


def test_leaf_card_completes(conn):
    task_id = _parent(conn, title="leaf")
    assert kb.complete_task(conn, task_id, result="shipped") is True
    assert kb.get_task(conn, task_id).status == "done"


def test_child_becoming_ready_after_the_precheck_cannot_slip_through(
    conn, tmp_path, monkeypatch,
):
    """The check/completion race: the pre-check reads "no ready child", another
    writer promotes the child immediately afterwards, and the parent UPDATE must
    still be refused by the re-check inside the write transaction."""
    parent_id = _parent(conn)
    child_id = _child(conn, parent_id)
    real_ready_children = kb._ready_children
    calls: list[int] = []

    def racing_precheck(active_conn, task_id):
        calls.append(1)
        if len(calls) == 1:
            # A concurrent writer promotes the child right after our stale read.
            side = kbc.connect(tmp_path / "kanban.db")
            try:
                _force_status(side, child_id, "ready")
            finally:
                side.close()
            return []
        return real_ready_children(active_conn, task_id)

    monkeypatch.setattr(kb, "_ready_children", racing_precheck, raising=False)

    with pytest.raises(kb.ChildrenNotTerminalError):
        kb.complete_task(conn, parent_id, result="shipped")

    assert len(calls) >= 2  # the in-txn re-check actually ran
    assert kb.get_task(conn, parent_id).status == "ready"
    assert kb.get_task(conn, child_id).status == "ready"
