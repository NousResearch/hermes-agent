"""Tests for Kanban provenance, heartbeat fencing, and durable worker exit evidence.

Covers:
- P0.3A: Session provenance prefers request-scoped ContextVar over os.environ,
  and validates against persisted SessionDB.
- P0.3B: Heartbeat propagation requires both claim and worker writes, and fences
  out delegated child contexts.
- P0.3C: Machine-readable exit trailer classification fallback and stripping from
  user-facing worker log views.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent.delegation_context import delegated_child_context
from gateway.session_context import scoped_current_session_id
from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db import (
    _classify_worker_exit,
    detect_crashed_workers,
    extract_worker_exit_trailer,
    format_worker_exit_trailer,
    read_worker_log,
    strip_worker_exit_trailer,
    worker_log_path,
)
from tools import kanban_tools


@pytest.fixture()
def isolated_hermes_env(tmp_path, monkeypatch):
    """Configure isolated HERMES_HOME and clean environment for kanban tests."""
    home = tmp_path / ".hermes"
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_CLAIM_LOCK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    kanban_tools._auto_heartbeat_last_attempt = 0.0
    return home


def test_persisted_session_id_validation(isolated_hermes_env):
    """Validates session_id against active profile SessionDB read-only."""
    from hermes_state import SessionDB

    # Initially nonexistent
    assert kanban_tools._persisted_session_id("nonexistent_session") is None
    assert kanban_tools._persisted_session_id("") is None
    assert kanban_tools._persisted_session_id(None) is None

    # Create real session in SessionDB
    db = SessionDB()
    try:
        db.create_session("persisted_sess_001", source="cli")
    finally:
        db.close()

    assert kanban_tools._persisted_session_id("persisted_sess_001") == "persisted_sess_001"


def test_kanban_provenance_prefers_contextvar_and_drops_dangling(isolated_hermes_env, monkeypatch):
    """kanban_create prefers request-scoped ContextVar over stale os.environ,

    and drops nonexistent provenance rather than recording a dangling session_id.
    """
    from hermes_state import SessionDB

    db = SessionDB()
    try:
        db.create_session("active_request_session", source="gateway")
    finally:
        db.close()

    monkeypatch.setenv("HERMES_SESSION_ID", "stale_process_session")

    # Inside request session_context:
    with scoped_current_session_id("active_request_session"):
        res_raw = kanban_tools._handle_create(
            {"title": "Task with request context", "assignee": "agent", "board": "test_board"}
        )
        task_id = json.loads(res_raw)["task_id"]

        conn = kb.connect(board="test_board")
        try:
            task = kb.get_task(conn, task_id)
            assert task is not None
            assert task.session_id == "active_request_session"
        finally:
            conn.close()

    # Explicit dangling session_id is dropped
    res_dangling = kanban_tools._handle_create(
        {
            "title": "Task with dangling session",
            "assignee": "agent",
            "session_id": "ghost_session",
            "board": "test_board",
        }
    )
    task_dangling_id = json.loads(res_dangling)["task_id"]
    conn = kb.connect(board="test_board")
    try:
        task2 = kb.get_task(conn, task_dangling_id)
        assert task2 is not None
        assert task2.session_id is None
    finally:
        conn.close()


def test_heartbeat_current_worker_fences_delegated_child(isolated_hermes_env, monkeypatch):
    """Delegated child context must not heartbeat parent worker or advance timestamp."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task_parent_001")
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "lock_parent_001")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "1")

    # In delegated child context via env:
    monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", "1")
    assert kanban_tools.heartbeat_current_worker_from_env() is False
    assert kanban_tools._auto_heartbeat_last_attempt == 0.0

    # In delegated child context via ContextVar:
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    with delegated_child_context():
        assert kanban_tools.heartbeat_current_worker_from_env() is False
        assert kanban_tools._auto_heartbeat_last_attempt == 0.0


def test_heartbeat_current_worker_requires_both_writes_to_succeed(isolated_hermes_env, monkeypatch):
    """heartbeat_current_worker_from_env returns True only if BOTH claim and worker succeed."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task_worker_002")
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "lock_worker_002")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "1")

    mock_kb = MagicMock()
    mock_conn = MagicMock()

    with patch.object(kanban_tools, "_connect", return_value=(mock_kb, mock_conn)):
        # Case 1: claim succeeds, worker heartbeat fails
        mock_kb.heartbeat_claim.return_value = True
        mock_kb.heartbeat_worker.return_value = False
        assert kanban_tools.heartbeat_current_worker_from_env() is False
        assert kanban_tools._auto_heartbeat_last_attempt == 0.0

        # Case 2: claim raises, worker succeeds
        mock_kb.heartbeat_claim.side_effect = RuntimeError("DB locked")
        mock_kb.heartbeat_worker.return_value = True
        assert kanban_tools.heartbeat_current_worker_from_env() is False
        assert kanban_tools._auto_heartbeat_last_attempt == 0.0

        # Case 3: both succeed
        mock_kb.heartbeat_claim.side_effect = None
        mock_kb.heartbeat_claim.return_value = True
        mock_kb.heartbeat_worker.return_value = True
        assert kanban_tools.heartbeat_current_worker_from_env() is True
        assert kanban_tools._auto_heartbeat_last_attempt > 0.0


def test_worker_exit_trailer_formatting_and_stripping():
    """Validates trailer formatting, parsing, and user-facing stripping."""
    trailer = format_worker_exit_trailer("task_123", 0, run_id=5)
    assert trailer.startswith("__HERMES_WORKER_EXIT__:")
    assert '"task_id":"task_123"' in trailer
    assert '"exit_code":0' in trailer
    assert '"run_id":5' in trailer

    log_content = (
        "Starting task 123...\n"
        "Tool call: read_file\n"
        "Completed analysis.\n"
        f"{trailer}\n"
    )

    payload = extract_worker_exit_trailer(log_content)
    assert payload == {"task_id": "task_123", "exit_code": 0, "run_id": 5}

    stripped = strip_worker_exit_trailer(log_content)
    assert stripped == "Starting task 123...\nTool call: read_file\nCompleted analysis.\n"
    assert "__HERMES_WORKER_EXIT__" not in stripped


def test_classify_worker_exit_fallback_to_durable_trailer(isolated_hermes_env):
    """When PID is not in reap registry, _classify_worker_exit reads durable log trailer."""
    board = "trailer_board"
    task_clean = "t_clean_exit"
    task_rate = "t_rate_limited"
    task_err = "t_error_exit"
    task_unknown = "t_no_trailer"

    # Write log files
    log_clean = worker_log_path(task_clean, board=board)
    log_clean.parent.mkdir(parents=True, exist_ok=True)
    log_clean.write_text(
        "Working...\n" + format_worker_exit_trailer(task_clean, 0) + "\n",
        encoding="utf-8",
    )

    log_rate = worker_log_path(task_rate, board=board)
    log_rate.write_text(
        "Rate limit hit\n" + format_worker_exit_trailer(task_rate, kb.KANBAN_RATE_LIMIT_EXIT_CODE) + "\n",
        encoding="utf-8",
    )

    log_err = worker_log_path(task_err, board=board)
    log_err.write_text(
        "Fatal error\n" + format_worker_exit_trailer(task_err, 1) + "\n",
        encoding="utf-8",
    )

    log_unknown = worker_log_path(task_unknown, board=board)
    log_unknown.write_text("Died unexpectedly without trailer\n", encoding="utf-8")

    # PID 999999 is dead and not in _recent_worker_exits
    assert _classify_worker_exit(999999, task_id=task_clean, board=board) == ("clean_exit", 0)
    assert _classify_worker_exit(999999, task_id=task_rate, board=board) == (
        "rate_limited",
        kb.KANBAN_RATE_LIMIT_EXIT_CODE,
    )
    assert _classify_worker_exit(999999, task_id=task_err, board=board) == ("nonzero_exit", 1)
    assert _classify_worker_exit(999999, task_id=task_unknown, board=board) == ("unknown", None)
    assert _classify_worker_exit(999999, task_id="t_nonexistent", board=board) == ("unknown", None)

    # Verify read_worker_log strips trailer
    clean_user_log = read_worker_log(task_clean, board=board)
    assert clean_user_log == "Working...\n"
    assert "__HERMES_WORKER_EXIT__" not in clean_user_log


def test_detect_crashed_workers_handles_clean_exit_as_protocol_violation(isolated_hermes_env):
    """detect_crashed_workers detects clean exit via trailer as protocol violation when task is running."""
    board = "crash_board"
    conn = kb.connect(board=board)
    try:
        # Create a task with max_retries=0 to force trip immediately on violation
        task_id = kb.create_task(conn, title="Protocol violation task", assignee="agent", max_retries=0)
        # Manually set to running with dead PID 999999 and this host's claim lock
        claimer = f"{kb._claimer_id()}"
        conn.execute(
            "UPDATE tasks SET status = 'running', worker_pid = 999999, claim_lock = ?, started_at = ? WHERE id = ?",
            (claimer, time.time() - 100, task_id),
        )
        conn.commit()

        # Write log with clean exit trailer (exit_code: 0)
        log_path = worker_log_path(task_id, board=board)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "Done but forgot kanban_complete\n" + format_worker_exit_trailer(task_id, 0) + "\n",
            encoding="utf-8",
        )

        with patch("hermes_cli.kanban_db._pid_alive", return_value=False):
            crashed = detect_crashed_workers(conn, board=board)

        # Task was reclaimed as crashed/protocol-violation
        assert task_id in crashed

        # Protocol violation auto-blocks task when limit reached
        auto_blocked = getattr(detect_crashed_workers, "_last_auto_blocked", [])
        assert task_id in auto_blocked

        # Task should be blocked in DB
        task = kb.get_task(conn, task_id)
        assert task.status == "blocked"
        assert "protocol violation" in (task.last_failure_error or "").lower()

        # Event log recorded protocol_violation event
        events = conn.execute("SELECT kind FROM task_events WHERE task_id = ?", (task_id,)).fetchall()
        kinds = [row["kind"] for row in events]
        assert "protocol_violation" in kinds
    finally:
        conn.close()
