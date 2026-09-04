"""Semantic progress timeout for Kanban workers.

Heartbeats with a non-empty note advance ``last_progress_at``; automatic
heartbeats (``note=None``) touch only ``last_heartbeat_at``. Tasks opt in
via ``progress_timeout_seconds``; NULL means no progress watchdog.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban import _cmd_create, _parse_duration


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _claim_running(conn, tid, *, pid=None):
    kb.claim_task(conn, tid)
    kb._set_worker_pid(conn, tid, pid or os.getpid())
    return kb.get_task(conn, tid)


def test_auto_heartbeat_updates_health_not_semantic_progress(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="hb", assignee="worker")
        _claim_running(conn, tid)
        task = kb.get_task(conn, tid)
        run = kb.latest_run(conn, tid)
        assert task.last_progress_at == run.started_at
        assert task.last_heartbeat_at is None

        assert kb.heartbeat_worker(conn, tid, note=None)
        task = kb.get_task(conn, tid)
        run = kb.latest_run(conn, tid)
        assert task.last_heartbeat_at is not None
        assert task.last_progress_at == run.started_at

        assert kb.heartbeat_worker(conn, tid, note="   \t")
        task = kb.get_task(conn, tid)
        run = kb.latest_run(conn, tid)
        assert task.last_progress_at == run.started_at
    finally:
        conn.close()


def test_noted_heartbeat_updates_progress_and_is_run_guarded(kanban_home, monkeypatch):
    import hermes_cli.kanban_db as _kb

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="semantic", assignee="worker")
        kb.claim_task(conn, tid)
        run1 = kb.latest_run(conn, tid)
        before = kb.get_task(conn, tid).last_progress_at

        assert kb.heartbeat_worker(conn, tid, note="still working", expected_run_id=run1.id)
        task = kb.get_task(conn, tid)
        assert task.last_heartbeat_at is not None
        assert task.last_progress_at is not None
        assert task.last_progress_at >= before

        kb._set_worker_pid(conn, tid, 98765)
        monkeypatch.setattr(_kb, "_pid_alive", lambda pid: False)
        assert kb.detect_crashed_workers(conn) == [tid]

        kb.claim_task(conn, tid)
        run2 = kb.latest_run(conn, tid)
        assert run2.id != run1.id

        assert not kb.heartbeat_worker(
            conn, tid, note="stale", expected_run_id=run1.id,
        )
        task = kb.get_task(conn, tid)
        assert task.current_run_id == run2.id
        assert task.last_progress_at == run2.started_at

        assert kb.heartbeat_worker(
            conn, tid, note="current", expected_run_id=run2.id,
        )
        assert kb.get_task(conn, tid).last_progress_at >= run2.started_at
    finally:
        conn.close()


def test_create_cli_parses_progress_timeout(kanban_home, capsys):
    args = argparse.Namespace(
        title="timed progress",
        body=None,
        assignee="worker",
        created_by="user",
        workspace="scratch",
        branch=None,
        project=None,
        tenant=None,
        priority=0,
        parent=[],
        triage=False,
        idempotency_key=None,
        max_runtime=None,
        progress_timeout="5m",
        skills=[],
        max_retries=None,
        model_override=None,
        provider_override=None,
        goal_mode=False,
        goal_max_turns=None,
        initial_status="running",
        json=True,
    )
    assert _cmd_create(args) == 0
    out = capsys.readouterr().out
    import json

    data = json.loads(out)
    assert data["progress_timeout_seconds"] == 300

    with pytest.raises(ValueError):
        _parse_duration("not-a-duration")


def test_default_task_never_reclaimed_by_progress_watchdog(kanban_home, monkeypatch):
    import hermes_cli.kanban_db as _kb

    monkeypatch.setattr(_kb, "_pid_alive", lambda pid: False)
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="no timeout", assignee="worker")
        _claim_running(conn, tid)
        old = int(time.time()) - 3600
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET last_progress_at = ? WHERE id = ?",
                (old, tid),
            )
            conn.execute(
                "UPDATE task_runs SET last_progress_at = ? "
                "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                (old, tid),
            )

        assert kb.enforce_progress_timeout(conn) == []
        assert kb.get_task(conn, tid).status == "running"
    finally:
        conn.close()


def test_recent_semantic_progress_prevents_progress_reclaim(kanban_home):
    import hermes_cli.kanban_db as _kb

    original_alive = _kb._pid_alive
    _kb._pid_alive = lambda pid: False
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title="recent progress", assignee="worker",
            progress_timeout_seconds=10,
            max_runtime_seconds=1,
        )
        _claim_running(conn, tid)
        old = int(time.time()) - 3600
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_runs SET started_at = ? "
                "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                (old, tid),
            )

        assert kb.heartbeat_worker(conn, tid, note="milestone reached")

        assert kb.enforce_progress_timeout(conn) == []
        assert kb.get_task(conn, tid).status == "running"
    finally:
        _kb._pid_alive = original_alive
        conn.close()


def test_expired_semantic_progress_terminates_and_records_event(kanban_home):
    import hermes_cli.kanban_db as _kb

    killed = []

    def _signal_fn(pid, sig):
        killed.append((pid, sig))

    original_alive = _kb._pid_alive
    _kb._pid_alive = lambda pid: False

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title="stalled", assignee="worker",
            progress_timeout_seconds=5,
        )
        _claim_running(conn, tid)
        old = int(time.time()) - 60
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET last_progress_at = ? WHERE id = ?",
                (old, tid),
            )
            conn.execute(
                "UPDATE task_runs SET last_progress_at = ?, started_at = ? "
                "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                (old, old, tid),
            )

        stalled = kb.enforce_progress_timeout(conn, signal_fn=_signal_fn)
        assert tid in stalled
        assert killed and killed[0][0] == os.getpid()

        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.worker_pid is None
        assert task.last_heartbeat_at is None
        assert task.last_progress_at is None

        events = kb.list_events(conn, tid)
        evt = next(e for e in events if e.kind == "progress_stalled")
        assert evt.payload["limit_seconds"] == 5
        assert evt.payload["progress_age_seconds"] >= 60
        assert evt.payload["last_progress_at"] == old
        assert evt.payload["retry_status"] == "ready"

        run = kb.latest_run(conn, tid)
        assert run.outcome == "progress_stalled"
        assert run.status == "progress_stalled"
    finally:
        _kb._pid_alive = original_alive
        conn.close()


def test_surviving_worker_defers_progress_reclaim(kanban_home, monkeypatch):
    import hermes_cli.kanban_db as _kb

    monkeypatch.setattr(
        _kb,
        "_terminate_reclaimed_worker",
        lambda pid, claim_lock, **kwargs: {
            "prev_pid": int(pid),
            "host_local": True,
            "termination_attempted": True,
            "terminated": False,
            "sigkill": False,
        },
    )
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title="unkillable", assignee="worker",
            progress_timeout_seconds=1,
        )
        task = _claim_running(conn, tid)
        old = int(time.time()) - 120
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET last_progress_at = ? WHERE id = ?",
                (old, tid),
            )
            conn.execute(
                "UPDATE task_runs SET last_progress_at = ? "
                "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                (old, tid),
            )

        assert kb.enforce_progress_timeout(conn) == []
        assert kb.get_task(conn, tid).status == "running"
        assert kb.get_task(conn, tid).worker_pid == task.worker_pid

        events = kb.list_events(conn, tid)
        assert any(e.kind == "reclaim_deferred" for e in events)
    finally:
        conn.close()


def test_progress_stalled_counts_toward_failure_limit(kanban_home, monkeypatch):
    import hermes_cli.kanban_db as _kb

    monkeypatch.setattr(_kb, "_pid_alive", lambda pid: False)
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn, title="breaker", assignee="worker",
            progress_timeout_seconds=1,
            max_retries=1,
        )
        for attempt in range(2):
            kb.claim_task(conn, tid)
            kb._set_worker_pid(conn, tid, 88000 + attempt)
            old = int(time.time()) - 120
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET last_progress_at = ? WHERE id = ?",
                    (old, tid),
                )
                conn.execute(
                    "UPDATE task_runs SET last_progress_at = ? "
                    "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                    (old, tid),
                )
            kb.enforce_progress_timeout(conn)

        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.consecutive_failures >= 1
        events = kb.list_events(conn, tid)
        assert any(e.kind == "gave_up" for e in events)
    finally:
        conn.close()


def test_migration_adds_progress_columns_to_legacy_db(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    db_path = home / "kanban" / "default" / "kanban.db"
    db_path.parent.mkdir(parents=True)
    conn = kb.sqlite3.connect(str(db_path))
    conn.row_factory = kb.sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            body TEXT,
            assignee TEXT,
            status TEXT NOT NULL DEFAULT 'todo',
            priority INTEGER NOT NULL DEFAULT 0,
            created_by TEXT,
            created_at INTEGER NOT NULL,
            started_at INTEGER,
            completed_at INTEGER,
            workspace_kind TEXT NOT NULL DEFAULT 'scratch',
            workspace_path TEXT,
            claim_lock TEXT,
            claim_expires INTEGER,
            tenant TEXT,
            result TEXT,
            idempotency_key TEXT,
            consecutive_failures INTEGER NOT NULL DEFAULT 0,
            worker_pid INTEGER,
            last_failure_error TEXT,
            max_runtime_seconds INTEGER,
            last_heartbeat_at INTEGER,
            current_run_id INTEGER
        );
        CREATE TABLE task_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            profile TEXT,
            step_key TEXT,
            status TEXT NOT NULL,
            claim_lock TEXT,
            claim_expires INTEGER,
            worker_pid INTEGER,
            max_runtime_seconds INTEGER,
            last_heartbeat_at INTEGER,
            started_at INTEGER NOT NULL,
            ended_at INTEGER,
            outcome TEXT,
            summary TEXT,
            metadata TEXT,
            error TEXT
        );
        CREATE TABLE task_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            run_id INTEGER,
            kind TEXT NOT NULL,
            payload TEXT,
            created_at INTEGER NOT NULL
        );
        CREATE TABLE task_links (
            parent_id TEXT NOT NULL,
            child_id TEXT NOT NULL,
            PRIMARY KEY (parent_id, child_id)
        );
        CREATE TABLE task_comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            author TEXT NOT NULL,
            body TEXT NOT NULL,
            created_at INTEGER NOT NULL
        );
        """
    )
    conn.close()

    kb.init_db(db_path=db_path)
    conn = kb.connect(db_path=db_path)
    try:
        tid = kb.create_task(
            conn, title="migrated", progress_timeout_seconds=120,
        )
        task = kb.get_task(conn, tid)
        assert task.progress_timeout_seconds == 120
        assert task.last_progress_at is None

        task_cols = {
            r["name"]
            for r in conn.execute("PRAGMA table_info(tasks)").fetchall()
        }
        run_cols = {
            r["name"]
            for r in conn.execute("PRAGMA table_info(task_runs)").fetchall()
        }
        assert "progress_timeout_seconds" in task_cols
        assert "last_progress_at" in task_cols
        assert "last_progress_at" in run_cols
        assert "progress_timeout_seconds" in run_cols
    finally:
        conn.close()
