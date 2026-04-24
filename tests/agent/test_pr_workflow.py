import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent.pr_workflow import (
    AI_SIGNATURE,
    GhCliAuthError,
    GhCliCommandError,
    GhCliNotInstalledError,
    GhPrPollingAdapter,
    PRCIStatus,
    PRReviewStatus,
    PRWorkflowStatus,
    append_ai_signature,
    create_pr_workflow,
    evaluate_pr_workflow_loop,
    transition_pr_workflow,
    update_pr_workflow_from_poll,
)


def _cp(cmd, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)


def _dispatch(responses):
    calls = []

    def fake_run(cmd, **kwargs):
        key = tuple(cmd)
        calls.append(key)
        if key not in responses:
            raise AssertionError(f"unexpected command: {cmd}")
        value = responses[key]
        if isinstance(value, Exception):
            raise value
        return value

    return fake_run, calls


def _pr_view_payload(**overrides):
    payload = {
        "number": 123,
        "title": "feat: polling adapter",
        "url": "https://github.com/acme/widgets/pull/123",
        "state": "OPEN",
        "isDraft": False,
        "reviewDecision": "APPROVED",
        "mergeable": "MERGEABLE",
        "headRefName": "feat/polling-adapter",
        "headRefOid": "abc123",
        "baseRefName": "main",
        "statusCheckRollup": [
            {
                "__typename": "CheckRun",
                "name": "pytest",
                "status": "COMPLETED",
                "conclusion": "SUCCESS",
                "detailsUrl": "https://github.com/acme/widgets/actions/runs/100",
            }
        ],
    }
    payload.update(overrides)
    return payload


def _responses(pr_payload, reviews):
    return {
        ("gh", "auth", "status", "--hostname", "github.com"): _cp(
            ["gh", "auth", "status", "--hostname", "github.com"], stdout="ok"
        ),
        (
            "gh",
            "pr",
            "view",
            "123",
            "--repo",
            "acme/widgets",
            "--json",
            "number,title,url,state,isDraft,reviewDecision,mergeable,headRefName,headRefOid,baseRefName,statusCheckRollup",
        ): _cp(["gh", "pr", "view"], stdout=json.dumps(pr_payload)),
        ("gh", "api", "repos/acme/widgets/pulls/123/reviews?per_page=100"): _cp(
            ["gh", "api"], stdout=json.dumps(reviews)
        ),
    }


def test_create_pr_workflow_defaults_to_setup_and_links_task_branch_worktree():
    record = create_pr_workflow(task_id="task-1", branch="feature/pr", worktree_path="/tmp/wt", now=10.0)

    assert record.task_id == "task-1"
    assert record.branch == "feature/pr"
    assert record.worktree_path == "/tmp/wt"
    assert record.status == PRWorkflowStatus.setup
    assert record.ci_status == PRCIStatus.unknown
    assert record.review_status == PRReviewStatus.unknown
    assert record.created_at == 10.0
    assert record.updated_at == 10.0


@pytest.mark.parametrize("field", ["task_id", "branch", "worktree_path"])
def test_create_pr_workflow_rejects_blank_identity_fields(field):
    kwargs = {"task_id": "task-1", "branch": "feature/pr", "worktree_path": "/tmp/wt"}
    kwargs[field] = "  "

    with pytest.raises(ValidationError):
        create_pr_workflow(**kwargs)


def test_transition_to_pr_created_requires_pr_number_and_url():
    record = create_pr_workflow(task_id="task-1", branch="feature/pr", worktree_path="/tmp/wt", now=1.0)
    record = transition_pr_workflow(record, PRWorkflowStatus.implementing, now=2.0)

    with pytest.raises(ValueError, match="pr_number and pr_url"):
        transition_pr_workflow(record, PRWorkflowStatus.pr_created, now=3.0)

    created = transition_pr_workflow(
        record,
        PRWorkflowStatus.pr_created,
        pr_number=123,
        pr_url="https://github.com/acme/widgets/pull/123",
        now=3.0,
    )

    assert created.pr_number == 123
    assert created.pr_url == "https://github.com/acme/widgets/pull/123"
    assert created.status == PRWorkflowStatus.pr_created


def test_ready_to_merge_requires_ci_review_and_local_verification_gates():
    record = create_pr_workflow(task_id="task-1", branch="feature/pr", worktree_path="/tmp/wt")
    record = transition_pr_workflow(record, PRWorkflowStatus.implementing)
    record = transition_pr_workflow(
        record,
        PRWorkflowStatus.pr_created,
        pr_number=123,
        pr_url="https://github.com/acme/widgets/pull/123",
    )
    record = transition_pr_workflow(record, PRWorkflowStatus.review_waiting, review_status=PRReviewStatus.waiting)
    approved = transition_pr_workflow(record, PRWorkflowStatus.approved, review_status=PRReviewStatus.approved)

    with pytest.raises(ValueError, match="ready_to_merge requires"):
        transition_pr_workflow(approved, PRWorkflowStatus.ready_to_merge)

    ready = transition_pr_workflow(
        approved,
        PRWorkflowStatus.ready_to_merge,
        ci_status=PRCIStatus.passing,
        review_status=PRReviewStatus.approved,
        local_verification_passed=True,
    )

    assert ready.status == PRWorkflowStatus.ready_to_merge


def test_model_forbids_secret_fields_in_persisted_state():
    with pytest.raises(ValidationError):
        create_pr_workflow(
            task_id="task-1",
            branch="feature/pr",
            worktree_path="/tmp/wt",
            github_token="ghp_secret",
        )


def test_ai_signature_is_appended_once_and_survives_empty_body():
    assert append_ai_signature("LGTM").endswith(AI_SIGNATURE)
    assert append_ai_signature(f"LGTM\n\n{AI_SIGNATURE}").count(AI_SIGNATURE) == 1
    assert append_ai_signature("   ") == AI_SIGNATURE


def test_missing_gh_raises_actionable_install_error():
    runner, _ = _dispatch({("gh", "auth", "status", "--hostname", "github.com"): FileNotFoundError("gh")})
    adapter = GhPrPollingAdapter(runner=runner)

    with pytest.raises(GhCliNotInstalledError, match="GitHub CLI"):
        adapter.poll_pull_request("acme/widgets", 123)


def test_unauthenticated_gh_raises_actionable_login_error():
    runner, _ = _dispatch(
        {
            ("gh", "auth", "status", "--hostname", "github.com"): _cp(
                ["gh", "auth"], returncode=1, stderr="not logged into any GitHub hosts"
            )
        }
    )
    adapter = GhPrPollingAdapter(runner=runner)

    with pytest.raises(GhCliAuthError, match="gh auth login"):
        adapter.poll_pull_request("acme/widgets", 123)


def test_ci_failure_creates_remediation_request_without_merging():
    payload = _pr_view_payload(
        reviewDecision="REVIEW_REQUIRED",
        statusCheckRollup=[
            {
                "__typename": "CheckRun",
                "name": "pytest",
                "status": "COMPLETED",
                "conclusion": "FAILURE",
                "detailsUrl": "https://github.com/acme/widgets/actions/runs/100",
            }
        ],
    )
    runner, calls = _dispatch(_responses(payload, []))
    result = GhPrPollingAdapter(runner=runner).poll_pull_request("acme/widgets", 123)

    assert result.status == PRWorkflowStatus.ci_failed
    assert result.remediation_requests[0].reason == "ci_failure"
    assert "pytest" in result.remediation_requests[0].summary
    assert not any(call[:3] == ("gh", "pr", "merge") for call in calls)


def test_review_changes_requested_creates_remediation_request():
    reviews = [
        {
            "state": "CHANGES_REQUESTED",
            "body": "Please add a regression test.",
            "submitted_at": "2026-04-24T16:00:00Z",
            "html_url": "https://github.com/acme/widgets/pull/123#pullrequestreview-2",
            "user": {"login": "alice", "type": "User"},
        }
    ]
    runner, _ = _dispatch(_responses(_pr_view_payload(reviewDecision="CHANGES_REQUESTED"), reviews))

    result = GhPrPollingAdapter(runner=runner).poll_pull_request("acme/widgets", 123)

    assert result.status == PRWorkflowStatus.changes_requested
    assert result.remediation_requests[0].reason == "review_feedback"
    assert result.review_feedback[0].reviewer == "alice"


def test_approved_green_and_local_verification_marks_ready_to_merge():
    reviews = [
        {
            "state": "APPROVED",
            "body": "Looks good.",
            "submitted_at": "2026-04-24T16:00:00Z",
            "html_url": "https://github.com/acme/widgets/pull/123#pullrequestreview-3",
            "user": {"login": "alice", "type": "User"},
        }
    ]
    runner, _ = _dispatch(_responses(_pr_view_payload(), reviews))

    result = GhPrPollingAdapter(runner=runner).poll_pull_request(
        "acme/widgets", 123, local_verification_passed=True
    )

    assert result.status == PRWorkflowStatus.ready_to_merge
    assert result.ready_to_merge is True
    assert result.remediation_requests == []


def test_polling_result_requires_local_verification_before_ready_to_merge():
    runner, _ = _dispatch(_responses(_pr_view_payload(), []))

    result = GhPrPollingAdapter(runner=runner).poll_pull_request("acme/widgets", 123)

    assert result.status == PRWorkflowStatus.approved
    assert result.ready_to_merge is False


def test_latest_human_review_wins_and_bot_reviews_are_ignored():
    reviews = [
        {
            "state": "CHANGES_REQUESTED",
            "body": "Needs work.",
            "submitted_at": "2026-04-24T15:00:00Z",
            "user": {"login": "alice", "type": "User"},
        },
        {
            "state": "APPROVED",
            "body": "Fixed.",
            "submitted_at": "2026-04-24T16:00:00Z",
            "user": {"login": "alice", "type": "User"},
        },
        {
            "state": "CHANGES_REQUESTED",
            "body": "bot noise",
            "submitted_at": "2026-04-24T17:00:00Z",
            "user": {"login": "review-bot", "type": "Bot"},
        },
    ]
    runner, _ = _dispatch(_responses(_pr_view_payload(), reviews))

    result = GhPrPollingAdapter(runner=runner).poll_pull_request("acme/widgets", 123)

    assert result.status == PRWorkflowStatus.approved
    assert [approval.reviewer for approval in result.approvals] == ["alice"]
    assert result.review_feedback == []


def test_update_pr_workflow_from_poll_maps_ci_failure_to_state():
    record = create_pr_workflow(task_id="task-1", branch="feature/pr", worktree_path="/tmp/wt")
    payload = _pr_view_payload(
        reviewDecision="REVIEW_REQUIRED",
        statusCheckRollup=[
            {"__typename": "CheckRun", "name": "pytest", "status": "COMPLETED", "conclusion": "FAILURE"}
        ],
    )
    runner, _ = _dispatch(_responses(payload, []))
    poll = GhPrPollingAdapter(runner=runner).poll_pull_request("acme/widgets", 123)

    updated = update_pr_workflow_from_poll(record, poll, now=20.0)

    assert updated.status == PRWorkflowStatus.ci_failed
    assert updated.pr_number == 123
    assert updated.pr_url == "https://github.com/acme/widgets/pull/123"
    assert updated.ci_status == PRCIStatus.failing
    assert updated.updated_at == 20.0


def test_update_pr_workflow_from_poll_rejects_unreachable_transition():
    record = create_pr_workflow(task_id="task-1", branch="feature/pr", worktree_path="/tmp/wt")
    closed = transition_pr_workflow(record, PRWorkflowStatus.closed, closed_reason="abandoned")
    reviews = [
        {
            "state": "APPROVED",
            "body": "Looks good.",
            "submitted_at": "2026-04-24T16:00:00Z",
            "html_url": "https://github.com/acme/widgets/pull/123#pullrequestreview-3",
            "user": {"login": "alice", "type": "User"},
        }
    ]
    runner, _ = _dispatch(_responses(_pr_view_payload(), reviews))
    poll = GhPrPollingAdapter(runner=runner).poll_pull_request("acme/widgets", 123, local_verification_passed=True)

    with pytest.raises(ValueError, match="invalid PR workflow transition"):
        update_pr_workflow_from_poll(closed, poll)


def test_closed_pr_maps_to_closed_status():
    runner, _ = _dispatch(_responses(_pr_view_payload(state="CLOSED"), []))

    result = GhPrPollingAdapter(runner=runner).poll_pull_request("acme/widgets", 123)

    assert result.status == PRWorkflowStatus.closed
    assert result.ready_to_merge is False


def test_changes_requested_without_body_still_creates_remediation_request():
    reviews = [
        {
            "state": "CHANGES_REQUESTED",
            "body": "",
            "submitted_at": "2026-04-24T16:00:00Z",
            "html_url": "https://github.com/acme/widgets/pull/123#pullrequestreview-2",
            "user": {"login": "alice", "type": "User"},
        }
    ]
    runner, _ = _dispatch(_responses(_pr_view_payload(reviewDecision="CHANGES_REQUESTED"), reviews))

    result = GhPrPollingAdapter(runner=runner).poll_pull_request("acme/widgets", 123)

    assert result.status == PRWorkflowStatus.changes_requested
    assert result.review_status == PRReviewStatus.changes_requested
    assert result.remediation_requests[0].reason == "review_feedback"
    assert result.review_feedback[0].reviewer == "alice"


def test_gh_command_errors_redact_token_like_stderr():
    runner, _ = _dispatch(
        {
            ("gh", "auth", "status", "--hostname", "github.com"): _cp(
                ["gh", "auth"], returncode=2, stderr="failed with ghp_secretTOKEN1234567890 and xoxb-secret"
            )
        }
    )

    with pytest.raises(GhCliCommandError) as exc:
        GhPrPollingAdapter(runner=runner).poll_pull_request("acme/widgets", 123)

    message = str(exc.value)
    assert "[REDACTED]" in message
    assert "ghp_secret" not in message
    assert "xoxb-secret" not in message


def test_work_with_pr_loop_stops_as_stuck_after_max_rounds():
    record = create_pr_workflow(task_id="task-1", branch="feature/pr", worktree_path="/tmp/wt")
    payload = _pr_view_payload(
        reviewDecision="REVIEW_REQUIRED",
        statusCheckRollup=[
            {"__typename": "CheckRun", "name": "pytest", "status": "COMPLETED", "conclusion": "FAILURE"}
        ],
    )
    runner, _ = _dispatch(_responses(payload, []))
    poll = GhPrPollingAdapter(runner=runner).poll_pull_request("acme/widgets", 123)

    result = evaluate_pr_workflow_loop(record, poll, round_index=3, max_rounds=3, now=30.0)

    assert result.record.status == PRWorkflowStatus.stuck
    assert result.stop is True
    assert result.reason == "max_rounds_exceeded"
    assert "maximum PR workflow rounds" in result.record.stuck_reason
    assert result.remediation_requests == poll.remediation_requests


def test_work_with_pr_loop_requires_final_review_gate_before_ready_to_merge():
    record = create_pr_workflow(task_id="task-1", branch="feature/pr", worktree_path="/tmp/wt")
    reviews = [
        {
            "state": "APPROVED",
            "body": "Looks good.",
            "submitted_at": "2026-04-24T16:00:00Z",
            "html_url": "https://github.com/acme/widgets/pull/123#pullrequestreview-3",
            "user": {"login": "alice", "type": "User"},
        }
    ]
    runner, _ = _dispatch(_responses(_pr_view_payload(), reviews))
    poll = GhPrPollingAdapter(runner=runner).poll_pull_request("acme/widgets", 123, local_verification_passed=True)

    pending = evaluate_pr_workflow_loop(
        record,
        poll,
        round_index=1,
        max_rounds=3,
        final_review_required=True,
        final_review_passed=False,
    )
    ready = evaluate_pr_workflow_loop(
        record,
        poll,
        round_index=1,
        max_rounds=3,
        final_review_required=True,
        final_review_passed=True,
    )

    assert pending.record.status == PRWorkflowStatus.approved
    assert pending.stop is False
    assert pending.reason == "final_review_required"
    assert pending.ready_to_merge is False
    assert pending.remediation_requests[0].reason == "final_review_required"
    assert ready.record.status == PRWorkflowStatus.ready_to_merge
    assert ready.stop is True
    assert ready.ready_to_merge is True
