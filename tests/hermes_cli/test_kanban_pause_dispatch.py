"""Paused dispatcher recovery and claim-race contracts."""

import sqlite3

from hermes_cli import kanban_control
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as dispatch
from hermes_cli import kanban_ready_monitor as monitor


def _conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(kb.SCHEMA_SQL)
    monitor.ensure_schema(conn)
    return conn


def test_paused_tick_recovers_stale_and_promotes_without_spawning(monkeypatch, tmp_path):
    conn = _conn()
    conn.execute(
        "INSERT INTO tasks(id,title,status,created_at,assignee,claim_lock,claim_expires) "
        "VALUES('stale','Stale','running',1,'p1','old-worker',1)",
    )
    conn.execute(
        "INSERT INTO tasks(id,title,status,created_at) VALUES('parent','Parent','done',1)",
    )
    conn.execute(
        "INSERT INTO tasks(id,title,status,created_at,assignee) "
        "VALUES('child','Child','todo',2,'p1')",
    )
    conn.execute("INSERT INTO task_links(parent_id,child_id) VALUES('parent','child')")
    conn.commit()
    kanban_control.pause_board(conn, changed_by="operator")
    monkeypatch.setattr(dispatch, "_profile_exists_fn", lambda: lambda profile: True)
    monkeypatch.setattr(dispatch._kbw, "resolve_workspace", lambda task, board=None: str(tmp_path))
    spawned = []

    result = dispatch.dispatch_once(
        conn, spawn_fn=lambda task, workspace: spawned.append(task.id) or 123,
    )

    assert result.control_state == "paused"
    assert result.spawned == [] and spawned == []
    assert conn.execute("SELECT status FROM tasks WHERE id='stale'").fetchone()[0] == "ready"
    assert conn.execute("SELECT status FROM tasks WHERE id='child'").fetchone()[0] == "ready"
    assert conn.execute("SELECT COUNT(*) FROM kanban_claim_admissions").fetchone()[0] == 0


def test_pause_after_selection_refuses_claim_and_resume_uses_next_tick(monkeypatch, tmp_path):
    conn = _conn()
    conn.execute(
        "INSERT INTO tasks(id,title,status,created_at,assignee) "
        "VALUES('queued','Queued','ready',1,'p1')",
    )
    conn.commit()
    monkeypatch.setattr(dispatch, "_profile_exists_fn", lambda: lambda profile: True)
    monkeypatch.setattr(dispatch._kbw, "resolve_workspace", lambda task, board=None: str(tmp_path))
    original = dispatch._dispatch_lane_task
    paused = False

    def pause_before_claim(*args, **kwargs):
        nonlocal paused
        if not paused:
            paused = True
            kanban_control.pause_board(conn, changed_by="operator")
        return original(*args, **kwargs)

    monkeypatch.setattr(dispatch, "_dispatch_lane_task", pause_before_claim)
    result = dispatch.dispatch_once(conn, spawn_fn=lambda task, workspace: 123)

    assert result.control_state == "paused"
    assert result.spawned == []
    assert conn.execute("SELECT status FROM tasks WHERE id='queued'").fetchone()[0] == "ready"
    assert conn.execute("SELECT COUNT(*) FROM task_runs WHERE task_id='queued'").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM kanban_claim_admissions").fetchone()[0] == 0

    monkeypatch.setattr(dispatch, "_dispatch_lane_task", original)
    kanban_control.resume_board(conn, changed_by="operator")
    resumed = dispatch.dispatch_once(conn, spawn_fn=lambda task, workspace: 123)
    assert resumed.control_state == "running"
    assert [item[0] for item in resumed.spawned] == ["queued"]
    assert conn.execute("SELECT status FROM tasks WHERE id='queued'").fetchone()[0] == "running"
