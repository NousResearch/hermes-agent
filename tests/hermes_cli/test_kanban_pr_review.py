from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_pr_review as pr


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


GREEN_VIEW = {
    "reviewDecision": "APPROVED",
    "statusCheckRollup": [
        {"name": "ci", "state": "COMPLETED", "conclusion": "SUCCESS"},
    ],
}

PENDING_CHECKS = [
    {"name": "ci", "bucket": "pending", "state": "IN_PROGRESS", "conclusion": None},
]

FAILING_CHECKS = [
    {"name": "ci", "bucket": "fail", "state": "COMPLETED", "conclusion": "FAILURE"},
]

REQUESTED_CHANGES_VIEW = {
    "reviewDecision": "CHANGES_REQUESTED",
    "reviews": [{"id": "R1", "state": "CHANGES_REQUESTED"}],
}

UNRESOLVED_THREADS = {
    "data": {
        "repository": {
            "pullRequest": {
                "reviewThreads": {
                    "nodes": [
                        {
                            "id": "T1",
                            "isResolved": False,
                            "comments": {"nodes": [{"id": "C1", "body": "fix this"}]},
                        }
                    ]
                }
            }
        }
    }
}


def test_evaluate_green_pr_merge_ready():
    result = pr.evaluate_pr_review(pr_view=GREEN_VIEW, checks=[], review_threads={})
    assert result["state"] == "merge_ready"
    assert result["actionable"] is False


def test_evaluate_pending_checks_in_review():
    result = pr.evaluate_pr_review(pr_view={}, checks=PENDING_CHECKS, review_threads={})
    assert result["state"] == "in_review"
    assert result["reason"] == "pending"


def test_evaluate_failing_checks_code_review():
    result = pr.evaluate_pr_review(pr_view={}, checks=FAILING_CHECKS, review_threads={})
    assert result["state"] == "code_review"
    assert result["actionable_ids"] == ["check:ci"]


def test_evaluate_requested_changes_code_review():
    result = pr.evaluate_pr_review(pr_view=REQUESTED_CHANGES_VIEW, checks=[], review_threads={})
    assert result["state"] == "code_review"
    assert result["actionable_ids"] == ["review:R1"]


def test_evaluate_unresolved_threads_code_review():
    result = pr.evaluate_pr_review(pr_view={}, checks=[], review_threads=UNRESOLVED_THREADS)
    assert result["state"] == "code_review"
    assert result["actionable_ids"] == ["comment:C1"]


def test_evaluate_already_seen_comments_not_actionable():
    result = pr.evaluate_pr_review(
        pr_view={},
        checks=[],
        review_threads=UNRESOLVED_THREADS,
        seen_ids={"comment:C1"},
    )
    assert result["state"] == "code_review"
    assert result["reason"] == "already_seen_feedback"
    assert result["actionable"] is False
    assert result["actionable_ids"] == []


def test_poll_task_updates_status_and_seen_ids(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="impl", assignee="worker")
        kb.complete_task(
            conn,
            tid,
            summary="PR opened",
            metadata={"pr_url": "https://github.com/acme/app/pull/42"},
        )

        def runner(args):
            if args[:2] == ["pr", "view"]:
                return json.dumps({"url": "https://github.com/acme/app/pull/42", **REQUESTED_CHANGES_VIEW})
            if args[:2] == ["pr", "checks"]:
                return json.dumps([])
            if args[:2] == ["api", "graphql"]:
                return json.dumps({})
            raise AssertionError(args)

        result = pr.poll_task(conn, tid, runner=runner)
        assert result["state"] == "code_review"
        assert result["seen_ids"] == ["review:R1"]
        assert kb.get_task(conn, tid).status == "code_review"

        events = [e for e in kb.list_events(conn, tid) if e.kind == "review_polled"]
        assert events
        assert "review:R1" in events[-1].payload["review"]["seen_ids"]
    finally:
        conn.close()


def test_poll_task_green_pr_moves_to_merge_ready(kanban_home):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="impl", assignee="worker")
        kb.complete_task(
            conn,
            tid,
            summary="PR opened",
            metadata={"pr_url": "https://github.com/acme/app/pull/42"},
        )

        def runner(args):
            if args[:2] == ["pr", "view"]:
                return json.dumps({"url": "https://github.com/acme/app/pull/42", **GREEN_VIEW})
            if args[:2] == ["pr", "checks"]:
                return json.dumps([])
            if args[:2] == ["api", "graphql"]:
                return json.dumps({})
            raise AssertionError(args)

        result = pr.poll_task(conn, tid, runner=runner)
        assert result["state"] == "merge_ready"
        task = kb.get_task(conn, tid)
        assert task.status == "merge_ready"
        assert task.completed_at is None
    finally:
        conn.close()
