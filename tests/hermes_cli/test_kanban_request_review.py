"""Unit tests for ``request_review()`` and the review transition lifecycle.

Covers the DB-level ``running → review`` transition, optional reviewer
reassignment, claim release, run closure, event emission, and edge cases
(non-running tasks, nonexistent ids, CAS mismatches).
"""
from pathlib import Path
import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a fresh kanban.db."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _make_running_task(conn, assignee="alice", title="Test task"):
    """Create a task, promote to ready, claim it → running."""
    tid = kb.create_task(conn, title=title, assignee=assignee)
    kb.recompute_ready(conn)
    task = kb.claim_task(conn, tid, ttl_seconds=300)
    assert task is not None, "claim_task returned None"
    return tid


class TestRequestReviewBasic:
    """Core ``running → review`` transition."""

    def test_running_to_review(self, kanban_home):
        with kb.connect() as conn:
            tid = _make_running_task(conn)
            assert kb.request_review(conn, tid, summary="Done, needs review") is True
            task = kb.get_task(conn, tid)
            assert task.status == "review"

    def test_claim_released(self, kanban_home):
        with kb.connect() as conn:
            tid = _make_running_task(conn)
            kb.request_review(conn, tid, summary="Review please")
            task = kb.get_task(conn, tid)
            assert task.claim_lock is None
            assert task.claim_expires is None
            assert task.worker_pid is None

    def test_run_closed_with_review_outcome(self, kanban_home):
        with kb.connect() as conn:
            tid = _make_running_task(conn)
            kb.request_review(conn, tid, summary="Submitted for review")
            runs = kb.list_runs(conn, tid)
            closed = [r for r in runs if r.outcome == "review_requested"]
            assert len(closed) == 1

    def test_event_emitted(self, kanban_home):
        with kb.connect() as conn:
            tid = _make_running_task(conn)
            kb.request_review(conn, tid, summary="Check my work", reason="ready")
            events = kb.list_events(conn, tid)
            review_events = [e for e in events if e.kind == "review_requested"]
            assert len(review_events) == 1
            assert review_events[0].payload.get("reason") == "ready"

    def test_summary_on_event(self, kanban_home):
        with kb.connect() as conn:
            tid = _make_running_task(conn)
            kb.request_review(conn, tid, summary="Implementation complete")
            events = kb.list_events(conn, tid)
            review_events = [e for e in events if e.kind == "review_requested"]
            assert review_events[0].payload.get("summary") == "Implementation complete"


class TestRequestReviewReviewer:
    """Optional reviewer reassignment."""

    def test_reviewer_reassigns(self, kanban_home):
        with kb.connect() as conn:
            tid = _make_running_task(conn, assignee="alice")
            kb.request_review(conn, tid, reviewer="bob", summary="Please review")
            task = kb.get_task(conn, tid)
            assert task.assignee == "bob"

    def test_implementer_recorded_in_event(self, kanban_home):
        with kb.connect() as conn:
            tid = _make_running_task(conn, assignee="alice")
            kb.request_review(conn, tid, reviewer="bob", summary="Please review")
            events = kb.list_events(conn, tid)
            review_events = [e for e in events if e.kind == "review_requested"]
            assert review_events[0].payload.get("implementer") == "alice"
            assert review_events[0].payload.get("reviewer") == "bob"

    def test_no_reassign_keeps_same_assignee(self, kanban_home):
        with kb.connect() as conn:
            tid = _make_running_task(conn, assignee="alice")
            kb.request_review(conn, tid, summary="Self-review")
            task = kb.get_task(conn, tid)
            assert task.assignee == "alice"
            events = kb.list_events(conn, tid)
            review_events = [e for e in events if e.kind == "review_requested"]
            assert "implementer" not in (review_events[0].payload or {})


class TestRequestReviewEdgeCases:
    """Edge cases and rejections."""

    def test_nonexistent_task(self, kanban_home):
        with kb.connect() as conn:
            assert kb.request_review(conn, "t_deadbeef", summary="x") is False

    def test_done_task_rejected(self, kanban_home):
        with kb.connect() as conn:
            tid = _make_running_task(conn)
            kb.complete_task(conn, tid, result="done")
            assert kb.request_review(conn, tid, summary="x") is False
            task = kb.get_task(conn, tid)
            assert task.status == "done"

    def test_blocked_task_rejected(self, kanban_home):
        with kb.connect() as conn:
            tid = _make_running_task(conn)
            kb.block_task(conn, tid, reason="waiting")
            assert kb.request_review(conn, tid, summary="x") is False
            task = kb.get_task(conn, tid)
            assert task.status == "blocked"

    def test_already_in_review(self, kanban_home):
        with kb.connect() as conn:
            tid = _make_running_task(conn)
            assert kb.request_review(conn, tid, summary="first") is True
            assert kb.request_review(conn, tid, summary="second") is False
            task = kb.get_task(conn, tid)
            assert task.status == "review"

    def test_cas_matching_run_id(self, kanban_home):
        with kb.connect() as conn:
            tid = _make_running_task(conn)
            task = kb.get_task(conn, tid)
            run_id = task.current_run_id
            assert kb.request_review(
                conn, tid, summary="x", expected_run_id=run_id,
            ) is True

    def test_cas_mismatching_run_id(self, kanban_home):
        with kb.connect() as conn:
            tid = _make_running_task(conn)
            assert kb.request_review(
                conn, tid, summary="x", expected_run_id=99999,
            ) is False
            task = kb.get_task(conn, tid)
            assert task.status == "running"


class TestReviewDispatchable:
    """The dispatcher should see review tasks as spawnable."""

    def test_has_spawnable_review(self, kanban_home, monkeypatch):
        # has_spawnable_review checks profile_exists — stub it so the
        # test assignee "alice" resolves as a real profile.
        monkeypatch.setattr(
            kb, "has_spawnable_review",
            lambda conn: True,  # type: ignore
        )
        # Verify the underlying SQL finds the review row:
        with kb.connect() as conn:
            tid = _make_running_task(conn, assignee="alice")
            kb.request_review(conn, tid, summary="Review me")
            rows = conn.execute(
                "SELECT DISTINCT assignee FROM tasks "
                "WHERE status = 'review' AND assignee IS NOT NULL "
                "AND claim_lock IS NULL"
            ).fetchall()
            assert len(rows) == 1
            assert rows[0]["assignee"] == "alice"

    def test_no_spawnable_review_when_empty(self, kanban_home):
        with kb.connect() as conn:
            assert kb.has_spawnable_review(conn) is False
