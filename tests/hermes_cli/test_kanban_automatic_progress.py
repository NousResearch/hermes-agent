"""Automatic kanban progress evidence from completed tool calls."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


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


def test_record_automatic_progress_updates_progress_and_emits_event(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="auto", assignee="worker")
        _claim_running(conn, tid)
        run = kb.latest_run(conn, tid)
        before = kb.get_task(conn, tid).last_progress_at

        assert kb.record_automatic_progress(
            conn,
            tid,
            evidence_type="tests_passed",
            detail="tests passed",
            expected_run_id=run.id,
        )
        task = kb.get_task(conn, tid)
        assert task.last_progress_at is not None
        assert task.last_progress_at >= before
        assert task.last_heartbeat_at is None

        events = kb.list_events(conn, tid)
        evt = next(e for e in events if e.kind == "automatic_progress")
        assert evt.payload == {"evidence_type": "tests_passed", "detail": "tests passed"}
        assert evt.run_id == run.id

        run_row = kb.latest_run(conn, tid)
        assert run_row.last_progress_at == task.last_progress_at
    finally:
        conn.close()


def test_record_automatic_progress_is_run_id_guarded(kanban_home, monkeypatch):
    import hermes_cli.kanban_db as _kb

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="guarded", assignee="worker")
        kb.claim_task(conn, tid)
        run1 = kb.latest_run(conn, tid)
        before = kb.get_task(conn, tid).last_progress_at

        assert kb.record_automatic_progress(
            conn,
            tid,
            evidence_type="build_passed",
            detail="build passed",
            expected_run_id=run1.id,
        )

        kb._set_worker_pid(conn, tid, 98765)
        monkeypatch.setattr(_kb, "_pid_alive", lambda pid: False)
        assert kb.detect_crashed_workers(conn) == [tid]

        kb.claim_task(conn, tid)
        run2 = kb.latest_run(conn, tid)
        assert run2.id != run1.id

        assert not kb.record_automatic_progress(
            conn,
            tid,
            evidence_type="build_passed",
            detail="build passed",
            expected_run_id=run1.id,
        )
        task = kb.get_task(conn, tid)
        assert task.current_run_id == run2.id
        assert task.last_progress_at == run2.started_at

        assert kb.record_automatic_progress(
            conn,
            tid,
            evidence_type="lint_passed",
            detail="lint passed",
            expected_run_id=run2.id,
        )
        assert kb.get_task(conn, tid).last_progress_at >= run2.started_at
    finally:
        conn.close()


def test_record_automatic_progress_rejects_invalid_and_stale(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="reject", assignee="worker")
        _claim_running(conn, tid)
        run = kb.latest_run(conn, tid)
        before = kb.get_task(conn, tid).last_progress_at

        assert not kb.record_automatic_progress(
            conn,
            tid,
            evidence_type="",
            detail="tests passed",
            expected_run_id=run.id,
        )
        assert not kb.record_automatic_progress(
            conn,
            tid,
            evidence_type="tests_passed",
            detail="",
            expected_run_id=run.id,
        )
        assert not kb.record_automatic_progress(
            conn,
            tid,
            evidence_type="tests_passed",
            detail="pytest tests/foo.py passed",
            expected_run_id=run.id,
        )
        assert kb.get_task(conn, tid).last_progress_at == before

        kb.complete_task(conn, tid, result="done")
        assert not kb.record_automatic_progress(
            conn,
            tid,
            evidence_type="tests_passed",
            detail="tests passed",
            expected_run_id=run.id,
        )
    finally:
        conn.close()


def test_recent_automatic_progress_prevents_progress_reclaim(kanban_home):
    import hermes_cli.kanban_db as _kb

    original_alive = _kb._pid_alive
    _kb._pid_alive = lambda pid: False
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="recent auto",
            assignee="worker",
            progress_timeout_seconds=10,
            max_runtime_seconds=1,
        )
        _claim_running(conn, tid)
        run = kb.latest_run(conn, tid)
        old = int(time.time()) - 3600
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE task_runs SET started_at = ? WHERE id = ?",
                (old, run.id),
            )

        assert kb.record_automatic_progress(
            conn,
            tid,
            evidence_type="file_changed",
            detail="patch completed",
            expected_run_id=run.id,
        )

        assert kb.enforce_progress_timeout(conn) == []
        assert kb.get_task(conn, tid).status == "running"
    finally:
        _kb._pid_alive = original_alive
        conn.close()
