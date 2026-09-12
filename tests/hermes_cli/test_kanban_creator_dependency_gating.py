"""Creator-parent edges are provenance, not readiness dependencies (#106994).

A worker that links itself as parent while also recording ``creator_task_id``
must not deadlock: the child has to become claimable before the parent can
complete. Real prerequisite parents (no matching creator) stay gated.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _running_parent(conn, title="parent"):
    parent = kb.create_task(conn, title=title, assignee="worker")
    claimed = kb.claim_task(conn, parent)
    assert claimed is not None
    assert kb.get_task(conn, parent).status == "running"
    return parent


def test_creator_parent_does_not_gate_child_ready(kanban_home):
    with kbc.connect() as conn:
        parent = _running_parent(conn)
        child = kb.create_task(
            conn, title="child", parents=(parent,), creator_task_id=parent,
            assignee="worker",
        )
        # EXPECT on current main: child.status == "todo"  → RED
        # EXPECT after fix: child.status == "ready" (or todo→recompute_ready→ready)
        status = kb.get_task(conn, child).status
        if status == "todo":
            kb.recompute_ready(conn)
            status = kb.get_task(conn, child).status
        assert status == "ready"
        assert kb.claim_task(conn, child) is not None


def test_non_creator_parent_still_gates(kanban_home):
    with kbc.connect() as conn:
        other = _running_parent(conn, title="prerequisite")
        child = kb.create_task(
            conn, title="child", parents=(other,), assignee="worker",
        )
        assert kb.get_task(conn, child).status == "todo"
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, child).status == "todo"
        assert kb.claim_task(conn, child) is None

        assert kb.complete_task(conn, other, result="done")
        # complete_task already recomputes children; a second pass is a no-op.
        assert kb.get_task(conn, child).status == "ready"


def test_creator_parent_stuck_todo_promoted_by_recompute_ready(kanban_home):
    with kbc.connect() as conn:
        parent = _running_parent(conn)
        child = kb.create_task(
            conn, title="child", parents=(parent,), creator_task_id=parent,
            assignee="worker",
        )
        # Simulate the pre-fix stuck disk: todo child blocked only by its creator.
        conn.execute("UPDATE tasks SET status = 'todo' WHERE id = ?", (child,))
        conn.commit()
        assert kb.get_task(conn, child).status == "todo"

        promoted = kb.recompute_ready(conn)
        assert promoted == 1
        assert kb.get_task(conn, child).status == "ready"
        assert kb.claim_task(conn, child) is not None


def test_link_tasks_does_not_demote_ready_child_for_creator_parent(kanban_home):
    with kbc.connect() as conn:
        parent = _running_parent(conn)
        child = kb.create_task(
            conn, title="child", creator_task_id=parent, assignee="worker",
        )
        assert kb.get_task(conn, child).status == "ready"
        kb.link_tasks(conn, parent, child)
        assert kb.get_task(conn, child).status == "ready"
        assert kb.claim_task(conn, child) is not None


def test_mixed_parents_still_wait_on_real_prerequisite(kanban_home):
    with kbc.connect() as conn:
        creator = _running_parent(conn, title="creator")
        other = _running_parent(conn, title="prerequisite")
        child = kb.create_task(
            conn, title="child", parents=(creator, other),
            creator_task_id=creator, assignee="worker",
        )
        status = kb.get_task(conn, child).status
        if status == "todo":
            kb.recompute_ready(conn)
            status = kb.get_task(conn, child).status
        assert status == "todo"
        assert kb.claim_task(conn, child) is None

        assert kb.complete_task(conn, other, result="done")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"


def test_unparseable_created_payload_fail_open_still_gates(kanban_home):
    with kbc.connect() as conn:
        parent = _running_parent(conn)
        child = kb.create_task(
            conn, title="child", parents=(parent,), creator_task_id=parent,
            assignee="worker",
        )
        # Payload is not an object → fail-open: creator-parent still gates.
        conn.execute(
            "UPDATE task_events SET payload = '[]' WHERE task_id = ? AND kind = 'created'",
            (child,),
        )
        conn.execute("UPDATE tasks SET status = 'todo' WHERE id = ?", (child,))
        conn.commit()
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, child).status == "todo"
        assert kb.claim_task(conn, child) is None
