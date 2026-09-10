"""Tests for worker termination on schedule_task and block_task transitions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kbc.init_db()
    return home


def test_schedule_running_task_terminates_worker(kanban_home, monkeypatch):
    """Scheduling a running task terminates its worker process post-commit."""
    mock_terminate = MagicMock(return_value={"terminated": True, "prev_pid": 12345})
    monkeypatch.setattr(kb, "_terminate_reclaimed_worker", mock_terminate)

    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="Active task")
        # Simulate worker claim and running state
        conn.execute(
            """
            UPDATE tasks
               SET status = 'running',
                   claim_lock = 'host1:uuid1',
                   worker_pid = 12345
             WHERE id = ?
            """,
            (task_id,),
        )
        conn.commit()

        success = kb.schedule_task(conn, task_id, reason="Hold for later")
        assert success is True

        # Verify task state in database
        row = conn.execute(
            "SELECT status, claim_lock, worker_pid FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        assert row["status"] == "scheduled"
        assert row["claim_lock"] is None
        assert row["worker_pid"] is None

        # Verify worker termination was invoked
        mock_terminate.assert_called_once_with(12345, "host1:uuid1")


def test_schedule_ready_task_does_not_terminate_worker(kanban_home, monkeypatch):
    """Scheduling a ready (non-running) task does not attempt worker termination."""
    mock_terminate = MagicMock()
    monkeypatch.setattr(kb, "_terminate_reclaimed_worker", mock_terminate)

    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="Ready task")
        success = kb.schedule_task(conn, task_id, reason="Hold for later")
        assert success is True

        mock_terminate.assert_not_called()


def test_block_running_task_terminates_worker(kanban_home, monkeypatch):
    """Blocking a running task terminates its worker process post-commit."""
    mock_terminate = MagicMock(return_value={"terminated": True, "prev_pid": 54321})
    monkeypatch.setattr(kb, "_terminate_reclaimed_worker", mock_terminate)

    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="Active task to block")
        conn.execute(
            """
            UPDATE tasks
               SET status = 'running',
                   claim_lock = 'host2:uuid2',
                   worker_pid = 54321
             WHERE id = ?
            """,
            (task_id,),
        )
        conn.commit()

        success = kb.block_task(conn, task_id, reason="Missing API token", kind="needs_input")
        assert success is True

        # Verify task state in database
        row = conn.execute(
            "SELECT status, claim_lock, worker_pid, block_kind FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        assert row["status"] == "blocked"
        assert row["claim_lock"] is None
        assert row["worker_pid"] is None
        assert row["block_kind"] == "needs_input"

        # Verify worker termination was invoked
        mock_terminate.assert_called_once_with(54321, "host2:uuid2")


def test_block_ready_task_does_not_terminate_worker(kanban_home, monkeypatch):
    """Blocking a ready (non-running) task does not attempt worker termination."""
    mock_terminate = MagicMock()
    monkeypatch.setattr(kb, "_terminate_reclaimed_worker", mock_terminate)

    with kbc.connect() as conn:
        task_id = kb.create_task(conn, title="Ready task to block")
        success = kb.block_task(conn, task_id, reason="Needs info", kind="needs_input")
        assert success is True

        mock_terminate.assert_not_called()
