"""Tests for shared Kanban operator semantics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_operator as op


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _payload(event):
    if event.payload is None:
        return None
    if isinstance(event.payload, str):
        return json.loads(event.payload)
    return event.payload


def test_set_status_direct_enforces_parent_guard_and_emits_status_event(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child", parents=[parent])

        assert kb.get_task(conn, child).status == "todo"
        assert op.set_status_direct(conn, child, "ready") is False
        assert kb.get_task(conn, child).status == "todo"
        assert [event.kind for event in kb.list_events(conn, child)] == ["created"]

        assert kb.complete_task(conn, parent)
        assert kb.get_task(conn, child).status == "ready"
        assert op.set_status_direct(conn, child, "triage") is True
        assert kb.get_task(conn, child).status == "triage"

        event = kb.list_events(conn, child)[-1]
        assert event.kind == "status"
        assert _payload(event) == {"status": "triage"}


def test_set_status_direct_closes_running_run_as_reclaimed(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="work", assignee="worker")
        claimed = kb.claim_task(conn, task_id, claimer="test-worker")
        assert claimed is not None
        run_id = claimed.current_run_id

        assert op.set_status_direct(conn, task_id, "ready") is True

        task = kb.get_task(conn, task_id)
        assert task.status == "ready"
        assert task.claim_lock is None
        assert task.claim_expires is None
        assert task.worker_pid is None
        run = conn.execute("SELECT * FROM task_runs WHERE id = ?", (run_id,)).fetchone()
        assert run["status"] == "reclaimed"
        assert run["outcome"] == "reclaimed"
        assert run["summary"] == "status changed to ready (dashboard/direct)"
        event = kb.list_events(conn, task_id)[-1]
        assert event.kind == "status"
        assert event.run_id == run_id
        assert _payload(event) == {"status": "ready"}


def test_update_priority_and_edit_task_emit_dashboard_parity_events(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="old", body="old body")

        assert op.update_priority(conn, task_id, 7) is True
        assert op.edit_task(conn, task_id, title="  new title  ", body="new body") is True

        task = kb.get_task(conn, task_id)
        assert task.priority == 7
        assert task.title == "new title"
        assert task.body == "new body"
        events = kb.list_events(conn, task_id)
        assert [event.kind for event in events[-2:]] == ["reprioritized", "edited"]
        assert _payload(events[-2]) == {"priority": 7}
        assert events[-1].payload is None


def test_bulk_update_preserves_dashboard_partial_success_semantics(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child", parents=[parent])
        movable = kb.create_task(conn, title="movable")

        result = op.bulk_update(
            conn,
            ids=[child, "missing", movable],
            status="ready",
            priority=5,
        )

        assert result == {
            "results": [
                {"id": child, "ok": False, "error": "transition to 'ready' refused"},
                {"id": "missing", "ok": False, "error": "not found"},
                {"id": movable, "ok": True},
            ]
        }
        # Bulk keeps applying later requested fields even when an earlier field
        # on the same task failed, matching the existing dashboard route.
        assert kb.get_task(conn, child).status == "todo"
        assert kb.get_task(conn, child).priority == 5
        assert kb.get_task(conn, movable).status == "ready"
        assert kb.get_task(conn, movable).priority == 5
