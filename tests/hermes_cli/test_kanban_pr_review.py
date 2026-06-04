"""Parser coverage for Kanban PR review polling."""

from __future__ import annotations

from hermes_cli import kanban_pr_review as prp


BASE_PR = {
    "number": 42,
    "url": "https://github.com/org/repo/pull/42",
    "state": "OPEN",
    "merged": False,
    "reviewDecision": "REVIEW_REQUIRED",
    "mergeStateStatus": "CLEAN",
}


def test_parse_all_green():
    result = prp.parse_pr_review_state(
        BASE_PR,
        [{"name": "tests", "state": "COMPLETED", "conclusion": "SUCCESS"}],
        review_threads={"nodes": []},
    )
    assert result.state == "green"
    assert not result.action_items


def test_pr_reference_extracts_repo_and_number_from_url():
    task = type("Task", (), {"workspace_path": "/tmp/worktree"})()
    run = type(
        "Run",
        (),
        {"metadata": {"github": {"pr_url": "https://github.com/org/repo/pull/42"}}},
    )()
    ref = prp.pr_reference_from_task(task, run)
    assert ref.repo == "org/repo"
    assert ref.number == 42
    assert ref.cwd == "/tmp/worktree"


def test_parse_pending_checks():
    result = prp.parse_pr_review_state(
        BASE_PR,
        [{"name": "tests", "state": "IN_PROGRESS", "conclusion": ""}],
        review_threads={"nodes": []},
    )
    assert result.state == "pending"
    assert result.pending_items[0].kind == "check"


def test_parse_failing_checks():
    result = prp.parse_pr_review_state(
        BASE_PR,
        [{"name": "tests", "state": "COMPLETED", "conclusion": "FAILURE"}],
        review_threads={"nodes": []},
    )
    assert result.state == "action_required"
    assert result.action_items[0].id == "check:tests:FAILURE"


def test_parse_failing_checks_from_stdout_fixture():
    result = prp.parse_pr_review_state(
        BASE_PR,
        "lint\tFAIL\nunit\tSUCCESS\n",
        review_threads={"nodes": []},
    )
    assert result.state == "action_required"
    assert result.action_items[0].id == "check:lint:FAIL"


def test_parse_requested_changes():
    result = prp.parse_pr_review_state(
        {**BASE_PR, "reviewDecision": "CHANGES_REQUESTED"},
        [],
        review_threads={"nodes": []},
    )
    assert result.state == "action_required"
    assert any(item.kind == "review" for item in result.action_items)


def test_parse_unresolved_review_threads():
    result = prp.parse_pr_review_state(
        BASE_PR,
        [],
        review_threads={
            "nodes": [
                {
                    "id": "thread-1",
                    "isResolved": False,
                    "path": "app.py",
                    "comments": {"nodes": [{"body": "Please fix this", "url": "https://example.test"}]},
                }
            ]
        },
    )
    assert result.state == "action_required"
    assert result.action_items[0].id == "thread:thread-1"


def test_parse_already_seen_comments_and_checks_are_not_reprocessed():
    result = prp.parse_pr_review_state(
        {**BASE_PR, "reviewDecision": "CHANGES_REQUESTED"},
        [{"name": "tests", "state": "COMPLETED", "conclusion": "FAILURE"}],
        review_threads={"nodes": [{"id": "thread-1", "isResolved": False}]},
        seen_ids={
            "review-decision:changes-requested",
            "check:tests:FAILURE",
            "thread:thread-1",
        },
    )
    assert result.state == "action_required"
    assert result.action_items == []
    assert {"review-decision:changes-requested", "check:tests:FAILURE", "thread:thread-1"} <= result.seen_ids


def test_parse_closed_unmerged_pr():
    result = prp.parse_pr_review_state(
        {**BASE_PR, "state": "CLOSED", "merged": False},
        [],
        review_threads={"nodes": []},
    )
    assert result.state == "closed_unmerged"
    assert result.closed_unmerged is True
    assert result.action_items[0].kind == "pr"
