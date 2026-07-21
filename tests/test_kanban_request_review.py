"""Tests for kanban request_review transition (running → review)."""
import sqlite3
import tempfile
from pathlib import Path

import pytest


@pytest.fixture()
def kanban_conn(tmp_path):
    """Create an in-memory kanban DB with the schema applied."""
    db_path = tmp_path / "test_kanban.db"
    from hermes_cli import kanban_db as kb

    conn = kb.connect(db_path)
    yield conn
    conn.close()


def _make_running_task(conn, assignee="worker"):
    """Create and claim a task so it's in 'running' status."""
    from hermes_cli import kanban_db as kb

    task_id = kb.create_task(conn, title="Test task", assignee=assignee)
    # Promote to ready first
    kb.promote_task(conn, task_id, actor="test", force=True)
    # Claim it → running
    task = kb.claim_task(conn, task_id, claimer="test-claim")
    assert task is not None
    return task_id


class TestRequestReview:
    """Test the request_review DB function."""

    def test_running_to_review(self, kanban_conn):
        """A running task should transition to review."""
        from hermes_cli import kanban_db as kb

        task_id = _make_running_task(kanban_conn)
        ok = kb.request_review(kanban_conn, task_id, reason="Done, please review")
        assert ok is True
        task = kb.get_task(kanban_conn, task_id)
        assert task.status == "review"

    def test_releases_claim_lock(self, kanban_conn):
        """request_review should release the claim lock."""
        from hermes_cli import kanban_db as kb

        task_id = _make_running_task(kanban_conn)
        kb.request_review(kanban_conn, task_id)
        task = kb.get_task(kanban_conn, task_id)
        assert task.claim_lock is None
        assert task.claim_expires is None
        assert task.worker_pid is None

    def test_non_running_fails(self, kanban_conn):
        """A non-running task should fail the transition."""
        from hermes_cli import kanban_db as kb

        task_id = kb.create_task(conn=kanban_conn, title="Not running")
        ok = kb.request_review(kanban_conn, task_id)
        assert ok is False

    def test_expected_run_id_mismatch(self, kanban_conn):
        """Should fail when expected_run_id doesn't match."""
        from hermes_cli import kanban_db as kb

        task_id = _make_running_task(kanban_conn)
        ok = kb.request_review(kanban_conn, task_id, expected_run_id=999)
        assert ok is False
        task = kb.get_task(kanban_conn, task_id)
        assert task.status == "running"

    def test_expected_run_id_match(self, kanban_conn):
        """Should succeed when expected_run_id matches."""
        from hermes_cli import kanban_db as kb

        task_id = _make_running_task(kanban_conn)
        task = kb.get_task(kanban_conn, task_id)
        ok = kb.request_review(
            kanban_conn, task_id,
            expected_run_id=task.current_run_id,
        )
        assert ok is True
        task2 = kb.get_task(kanban_conn, task_id)
        assert task2.status == "review"

    def test_records_event(self, kanban_conn):
        """request_review should emit a review_requested event."""
        from hermes_cli import kanban_db as kb

        task_id = _make_running_task(kanban_conn)
        kb.request_review(kanban_conn, task_id, reason="All tests pass")
        events = kb.list_events(kanban_conn, task_id)
        rr_events = [e for e in events if e.kind == "review_requested"]
        assert len(rr_events) == 1
        assert rr_events[0].payload["reason"] == "All tests pass"

    def test_ends_run(self, kanban_conn):
        """request_review should close the current run."""
        from hermes_cli import kanban_db as kb

        task_id = _make_running_task(kanban_conn)
        kb.request_review(kanban_conn, task_id)
        run = kb.latest_run(kanban_conn, task_id)
        assert run is not None
        assert run.outcome == "review_requested"

    def test_nonexistent_task(self, kanban_conn):
        """Should fail gracefully for a nonexistent task."""
        from hermes_cli import kanban_db as kb

        ok = kb.request_review(kanban_conn, "t_nonexistent")
        assert ok is False

    def test_review_task_cannot_request_review(self, kanban_conn):
        """A task already in review should fail."""
        from hermes_cli import kanban_db as kb

        task_id = _make_running_task(kanban_conn)
        kb.request_review(kanban_conn, task_id)
        # Try again — should fail since it's now in review
        ok = kb.request_review(kanban_conn, task_id)
        assert ok is False
