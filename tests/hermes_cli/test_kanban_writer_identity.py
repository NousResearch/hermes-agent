"""Invariant coverage for the Kanban SQLite writer boundary (#110080)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


def test_raw_sqlite_cannot_bypass_completion_gates(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    db_path = kb.init_db()
    with kbc.connect(db_path) as conn:
        task_id = kb.create_task(conn, title="guarded", assignee="builder")

    raw = sqlite3.connect(db_path, isolation_level=None)
    try:
        with pytest.raises(sqlite3.OperationalError, match="_hermes_kanban_writer_authorized"):
            raw.execute(
                "UPDATE tasks SET status = 'done', result = 'forged' WHERE id = ?",
                (task_id,),
            )
    finally:
        raw.close()

    with kbc.connect(db_path) as conn:
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert (task.status, task.result) == ("ready", None)
        assert kb.complete_task(conn, task_id, result="approved") is True
        completed = kb.get_task(conn, task_id)
        assert completed is not None
        assert (completed.status, completed.result) == ("done", "approved")
