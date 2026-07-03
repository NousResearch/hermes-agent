"""Tests for worker process termination on task status transitions.

When a task transitions away from ``running`` (blocked, archived, done),
the associated worker subprocess must be terminated. Previously, only
``reclaim_task`` and ``release_stale_claims`` terminated workers; status
transitions via ``block_task``, ``archive_task``, ``complete_task``, and
``_set_status_direct`` cleared ``worker_pid`` in the DB but left the
subprocess alive (#57596).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _running_task_with_pid(conn, title="t", pid=12345):
    """Create a task, drive to running, and set a fake worker_pid."""
    tid = kb.create_task(conn, title=title, assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    claimed = kb.claim_task(conn, tid, claimer="worker")
    assert claimed is not None
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET worker_pid=?, claim_lock=? WHERE id=?",
            (pid, f"host:{pid}", tid),
        )
    return tid


class TestBlockTaskTerminatesWorker:
    """block_task must terminate the worker process when transitioning from running."""

    def test_block_task_terminates_running_worker(self, kanban_home):
        conn = kb.connect()
        try:
            tid = _running_task_with_pid(conn, pid=99999)
            with patch(
                "hermes_cli.kanban_db._terminate_reclaimed_worker"
            ) as mock_term:
                result = kb.block_task(conn, tid, reason="test block")
            assert result is True
            mock_term.assert_called_once_with(99999, f"host:99999")
        finally:
            conn.close()

    def test_block_task_no_termination_when_no_pid(self, kanban_home):
        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="t", assignee="worker")
            with kb.write_txn(conn):
                conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
            # No worker_pid set
            with patch(
                "hermes_cli.kanban_db._terminate_reclaimed_worker"
            ) as mock_term:
                result = kb.block_task(conn, tid, reason="test block")
            assert result is True
            mock_term.assert_not_called()
        finally:
            conn.close()

    def test_block_dependency_terminates_running_worker(self, kanban_home):
        conn = kb.connect()
        try:
            tid = _running_task_with_pid(conn, pid=88888)
            with patch(
                "hermes_cli.kanban_db._terminate_reclaimed_worker"
            ) as mock_term:
                result = kb.block_task(conn, tid, reason="dep", kind="dependency")
            assert result is True
            mock_term.assert_called_once_with(88888, f"host:88888")
        finally:
            conn.close()


class TestArchiveTaskTerminatesWorker:
    """archive_task must terminate the worker process when transitioning from running."""

    def test_archive_task_terminates_running_worker(self, kanban_home):
        conn = kb.connect()
        try:
            tid = _running_task_with_pid(conn, pid=77777)
            with patch(
                "hermes_cli.kanban_db._terminate_reclaimed_worker"
            ) as mock_term:
                result = kb.archive_task(conn, tid)
            assert result is True
            mock_term.assert_called_once_with(77777, f"host:77777")
        finally:
            conn.close()

    def test_archive_task_no_termination_when_no_pid(self, kanban_home):
        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="t", assignee="worker")
            with patch(
                "hermes_cli.kanban_db._terminate_reclaimed_worker"
            ) as mock_term:
                result = kb.archive_task(conn, tid)
            assert result is True
            mock_term.assert_not_called()
        finally:
            conn.close()


class TestCompleteTaskTerminatesWorker:
    """complete_task must terminate the worker process when transitioning from running."""

    def test_complete_task_terminates_running_worker(self, kanban_home):
        conn = kb.connect()
        try:
            tid = _running_task_with_pid(conn, pid=66666)
            with patch(
                "hermes_cli.kanban_db._terminate_reclaimed_worker"
            ) as mock_term:
                result = kb.complete_task(conn, tid, result="done")
            assert result is True
            mock_term.assert_called_once_with(66666, f"host:66666")
        finally:
            conn.close()

    def test_complete_task_no_termination_when_no_pid(self, kanban_home):
        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="t", assignee="worker")
            with kb.write_txn(conn):
                conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
            with patch(
                "hermes_cli.kanban_db._terminate_reclaimed_worker"
            ) as mock_term:
                result = kb.complete_task(conn, tid, result="done")
            assert result is True
            mock_term.assert_not_called()
        finally:
            conn.close()
