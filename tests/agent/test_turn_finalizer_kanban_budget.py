"""Regression coverage for Kanban workers that exhaust their turn budget."""

from __future__ import annotations

import logging

from agent.turn_finalizer import _record_kanban_budget_exhausted
from hermes_cli import kanban_db as kb


def test_budget_exhaustion_parks_task_for_narrower_input(tmp_path, monkeypatch):
    """A completed process that ran out of turns must not enter a respawn loop."""
    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    with kb.connect_closing() as conn:
        task_id = kb.create_task(conn, title="needs evidence", assignee="worker")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (task_id,))
        assert kb.claim_task(conn, task_id, claimer="worker") is not None

    _record_kanban_budget_exhausted(task_id, 18, 18, logging.getLogger(__name__))

    with kb.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
        events = kb.list_events(conn, task_id)

    assert task is not None
    assert task.status == "blocked"
    assert task.consecutive_failures == 0
    blocked = [event for event in events if event.kind == "blocked"]
    assert blocked
    payload = blocked[-1].payload or {}
    assert payload["kind"] == "needs_input"
    assert "Iteration budget exhausted (18/18)" in payload["reason"]
