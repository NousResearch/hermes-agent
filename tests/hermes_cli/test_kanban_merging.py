"""Tests for the post-approve PR merger flow.

Covers:
- ``_extract_pr_url``: event path, comment path, no-URL path
- ``claim_merger_task``: human_review → running transition, merge_requested event
- ``approve_task`` routing:
  - PR URL present → merge_triggered, task running
  - No PR URL → done (backwards-compat path)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _claim(conn, tid):
    task = kb.claim_task(conn, tid)
    assert task is not None
    return task


def _make_human_review_task(conn, *, title="pr task", assignee="worker"):
    """Create a task that has progressed through claim → review → human_review."""
    tid = kb.create_task(conn, title=title, assignee=assignee)
    _claim(conn, tid)
    kb.move_to_review(conn, tid, reason="PR at https://github.com/owner/repo/pull/42")
    _ = kb.claim_review_task(conn, tid)
    kb.move_to_human_review(conn, tid, reason="AC verified")
    return tid


# ---------------------------------------------------------------------------
# _extract_pr_url
# ---------------------------------------------------------------------------

class TestExtractPrUrl:
    def test_finds_url_in_review_requested_event(self, kanban_home):
        """URL carried in review_requested event.reason is returned."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="x", assignee="worker")
            _claim(conn, tid)
            kb.move_to_review(
                conn, tid, reason="PR merged: https://github.com/org/repo/pull/99"
            )
            url = kb._extract_pr_url(conn, tid)
        assert url == "https://github.com/org/repo/pull/99"

    def test_falls_back_to_comments(self, kanban_home):
        """If no review_requested event, scans comments."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="x", assignee="worker")
            kb.add_comment(
                conn, tid, "worker",
                "Opened PR: https://github.com/org/repo/pull/7",
            )
            url = kb._extract_pr_url(conn, tid)
        assert url == "https://github.com/org/repo/pull/7"

    def test_returns_none_when_no_url(self, kanban_home):
        """Returns None when neither events nor comments carry a PR URL."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="x", assignee="worker")
            _claim(conn, tid)
            # A review_requested event with no URL in the reason.
            kb.move_to_review(conn, tid, reason="work done, no PR")
            url = kb._extract_pr_url(conn, tid)
        assert url is None

    def test_event_path_takes_priority_over_comments(self, kanban_home):
        """Event-path URL wins over a comment URL (events scanned first)."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="x", assignee="worker")
            _claim(conn, tid)
            # Comment has a different PR number.
            kb.add_comment(
                conn, tid, "worker",
                "Old PR: https://github.com/org/repo/pull/1",
            )
            # Event has the newer PR.
            kb.move_to_review(
                conn, tid, reason="PR https://github.com/org/repo/pull/55"
            )
            url = kb._extract_pr_url(conn, tid)
        # Should return the event URL, not the comment URL.
        assert url == "https://github.com/org/repo/pull/55"


# ---------------------------------------------------------------------------
# claim_merger_task
# ---------------------------------------------------------------------------

class TestClaimMergerTask:
    def test_human_review_to_running(self, kanban_home):
        """claim_merger_task transitions human_review → running."""
        with kb.connect() as conn:
            tid = _make_human_review_task(conn)
            task = kb.claim_merger_task(
                conn, tid, pr_url="https://github.com/org/repo/pull/10"
            )
        assert task is not None
        assert task.status == "running"

    def test_appends_merge_requested_event(self, kanban_home):
        """A merge_requested event is appended carrying the pr_url."""
        pr = "https://github.com/org/repo/pull/10"
        with kb.connect() as conn:
            tid = _make_human_review_task(conn)
            kb.claim_merger_task(conn, tid, pr_url=pr)
            events = kb.list_events(conn, tid)
        merge_events = [e for e in events if e.kind == "merge_requested"]
        assert len(merge_events) == 1
        payload = merge_events[0].payload or {}
        assert payload.get("pr_url") == pr

    def test_creates_new_run_row(self, kanban_home):
        """claim_merger_task creates a new task_run entry."""
        with kb.connect() as conn:
            tid = _make_human_review_task(conn)
            task = kb.claim_merger_task(
                conn, tid, pr_url="https://github.com/org/repo/pull/10"
            )
        assert task is not None
        assert task.current_run_id is not None

    def test_returns_none_when_already_claimed(self, kanban_home):
        """Returns None when the task has a claim lock (already running)."""
        with kb.connect() as conn:
            tid = _make_human_review_task(conn)
            # First claim succeeds.
            task1 = kb.claim_merger_task(
                conn, tid, pr_url="https://github.com/org/repo/pull/10"
            )
            assert task1 is not None
            # Second claim on the running task must fail.
            task2 = kb.claim_merger_task(
                conn, tid, pr_url="https://github.com/org/repo/pull/10"
            )
        assert task2 is None

    def test_returns_none_when_not_human_review(self, kanban_home):
        """Returns None when task is not in human_review status."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="x", assignee="worker")
            _claim(conn, tid)
            # Still running — not human_review.
            task = kb.claim_merger_task(
                conn, tid, pr_url="https://github.com/org/repo/pull/1"
            )
        assert task is None


# ---------------------------------------------------------------------------
# approve_task routing
# ---------------------------------------------------------------------------

class TestApproveTaskRouting:
    def test_routes_to_merge_triggered_when_pr_url_present(self, kanban_home):
        """When a PR URL exists in events, approve_task routes to merge_triggered."""
        with kb.connect() as conn:
            tid = _make_human_review_task(conn)
            ok, outcome, pr_url, task = kb.approve_task(conn, tid)
        assert ok
        assert outcome == "merge_triggered"
        assert pr_url == "https://github.com/owner/repo/pull/42"
        assert task is not None
        assert task.status == "running"

    def test_routes_to_done_when_no_pr_url(self, kanban_home):
        """When no PR URL is found, approve_task routes to done (backwards-compat)."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="no-pr task", assignee="worker")
            _claim(conn, tid)
            kb.move_to_human_review(conn, tid, reason="research deliverable, no PR")
            ok, outcome, pr_url, task = kb.approve_task(conn, tid)
        assert ok
        assert outcome == "done"
        assert pr_url is None
        assert task is None
        with kb.connect() as conn:
            t = kb.get_task(conn, tid)
        assert t is not None
        assert t.status == "done"
        assert t.completed_at is not None

    def test_done_path_emits_approved_event(self, kanban_home):
        """No-PR path emits an approved event (unchanged behavior)."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="x", assignee="worker")
            _claim(conn, tid)
            kb.move_to_human_review(conn, tid, reason="no PR")
            kb.approve_task(conn, tid, reason="LGTM")
            events = kb.list_events(conn, tid)
        kinds = [e.kind for e in events]
        assert "approved" in kinds

    def test_merge_triggered_path_emits_merge_requested_event(self, kanban_home):
        """PR path emits a merge_requested event with the pr_url."""
        with kb.connect() as conn:
            tid = _make_human_review_task(conn)
            kb.approve_task(conn, tid)
            events = kb.list_events(conn, tid)
        merge_events = [e for e in events if e.kind == "merge_requested"]
        assert len(merge_events) >= 1
        last_payload = merge_events[-1].payload or {}
        assert "github.com" in last_payload.get("pr_url", "")

    def test_returns_not_ok_when_not_human_review(self, kanban_home):
        """Returns (False, ...) when task is not in human_review."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="x", assignee="worker")
            _claim(conn, tid)
            # Still running.
            ok, outcome, _, __ = kb.approve_task(conn, tid)
        assert not ok
        assert outcome == "not_found"

    def test_idempotent_second_approve_fails(self, kanban_home):
        """A second approve on an already-transitioned task returns not-ok."""
        with kb.connect() as conn:
            tid = kb.create_task(conn, title="x", assignee="worker")
            _claim(conn, tid)
            kb.move_to_human_review(conn, tid, reason="no PR")
            ok1, _, __, ___ = kb.approve_task(conn, tid)
            assert ok1
            # Second call — task is already done.
            ok2, _, __, ___ = kb.approve_task(conn, tid)
            assert not ok2
