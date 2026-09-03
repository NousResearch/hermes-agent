from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return home


def _running_task(conn, *, assignee: str = "dev") -> tuple[str, int]:
    task_id = kb.create_task(conn, title="bind worker session", assignee=assignee)
    now = int(time.time())
    conn.execute(
        "UPDATE tasks SET status='running', claim_lock='host:claim', "
        "claim_expires=?, worker_pid=1234, started_at=? WHERE id=?",
        (now + 300, now, task_id),
    )
    conn.execute(
        "INSERT INTO task_runs (task_id, profile, status, claim_lock, "
        "claim_expires, worker_pid, started_at) "
        "VALUES (?, ?, 'running', 'host:claim', ?, 1234, ?)",
        (task_id, assignee, now + 300, now),
    )
    run_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    conn.execute("UPDATE tasks SET current_run_id=? WHERE id=?", (run_id, task_id))
    conn.commit()
    return task_id, run_id


def test_task_runs_session_id_schema_and_row_parser(kanban_home):
    conn = kb.connect()
    try:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(task_runs)")}
        assert "session_id" in columns
        task_id, run_id = _running_task(conn)
        assert kb.bind_worker_session(conn, task_id, run_id, "session-a") is True
        run = kb.list_runs(conn, task_id)[0]
        assert run.session_id == "session-a"
    finally:
        conn.close()


def test_bind_worker_session_rejects_stale_or_ended_attempt(kanban_home):
    conn = kb.connect()
    try:
        task_id, run_id = _running_task(conn)
        assert kb.bind_worker_session(conn, task_id, run_id, "session-a") is True
        assert kb.bind_worker_session(conn, task_id, run_id + 1, "session-stale") is False
        row = conn.execute("SELECT session_id FROM task_runs WHERE id=?", (run_id,)).fetchone()
        assert row["session_id"] == "session-a"

        conn.execute("UPDATE task_runs SET status='done' WHERE id=?", (run_id,))
        conn.execute("UPDATE tasks SET status='done', current_run_id=NULL WHERE id=?", (task_id,))
        conn.commit()
        assert kb.bind_worker_session(conn, task_id, run_id, "session-late") is False
        row = conn.execute("SELECT session_id FROM task_runs WHERE id=?", (run_id,)).fetchone()
        assert row["session_id"] == "session-a"
    finally:
        conn.close()
