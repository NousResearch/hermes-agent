"""Tests for the barebones GitHub issue resolution lane."""

import json
from pathlib import Path

import pytest

from gateway.issue_resolution import (
    AiderRole,
    CompletedProcess,
    EpicTask,
    IssueMetadata,
    IssueResolutionRequest,
    IssueSelectionRequest,
    IssueRun,
    IssueRunStatus,
    IssueRunType,
    PullRequestMetadata,
    cancel_issue_resolution,
    CODER_TIERS,
    ModelProvider,
    ReviewFindingsRetry,
    ReviewLoopCircuitBreaker,
    ReviewMergeGateError,
    ReviewTagParseError,
    REVIEWER_TIERS,
    ReviewSuggestionStats,
    IssueStateStore,
    _execute_master_issue,
    _execute_single_issue,
    _assert_issue_branch_ready_for_pr,
    _guard_managed_repo_before_issue_dispatch,
    _inspect_managed_repo,
    _prepare_issue_branch_from_synced_default,
    _find_existing_sub_issue,
    _merge_ready_pr,
    _issue_branch_name,
    _local_coder_prompt,
    _load_pr_review_suggestion_stats,
    _load_next_open_issue,
    allowed_issue_repos,
    _pr_body,
    build_aider_invocation,
    can_merge_pr,
    is_copilot_review_author,
    is_review_findings_for_coder,
    plan_pr_manager_next_action,
    parse_review_routing_tag,
    summarize_review_suggestions,
    github_issue_webhook_command,
    is_master_issue,
    parse_decomposition_response,
    parse_issue_cancel_command_args,
    parse_issue_command_args,
    _push_issue_branch,
    submit_issue_resolution,
    parse_issue_next_command_args,
)
from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS, resolve_command


def test_issue_command_is_gateway_known():
    """The gateway should recognize /issue as a first-class command."""
    command = resolve_command("issue")

    assert command is not None
    assert command.name == "issue"
    assert "issue" in GATEWAY_KNOWN_COMMANDS


def test_parse_issue_command_owner_repo_and_number(tmp_path):
    """Parse the normal Telegram form: /issue owner/repo #10."""
    request = parse_issue_command_args(
        f"m0nklabs/cryptotrader #10 --workdir {tmp_path} --branch issue/10-test "
        "--kanban-task t_abc123 --kanban-board production"
    )

    assert request.repo == "m0nklabs/cryptotrader"
    assert request.issue_number == 10
    assert request.workdir == tmp_path
    assert request.branch == "issue/10-test"
    assert request.kanban_task_id == "t_abc123"
    assert request.kanban_board == "production"


def test_parse_issue_command_github_url(tmp_path):
    """Parse a direct GitHub issue URL."""
    request = parse_issue_command_args(
        f"https://github.com/m0nklabs/cryptotrader/issues/42 --workdir {tmp_path}"
    )

    assert request.repo == "m0nklabs/cryptotrader"
    assert request.issue_number == 42


def test_github_issue_webhook_command_ignores_pull_requests():
    """GitHub issues events for PRs should not start the issue lane."""
    payload = {
        "repository": {"full_name": "m0nklabs/cryptotrader"},
        "issue": {"number": 5, "pull_request": {"url": "https://api.github.com/pr"}},
    }

    assert github_issue_webhook_command(payload) is None


def test_github_issue_webhook_command_builds_slash_command():
    """A GitHub issues webhook can be converted into a gateway /issue command."""
    payload = {
        "repository": {"full_name": "m0nklabs/cryptotrader"},
        "issue": {"number": 5},
    }

    assert (
        github_issue_webhook_command(payload)
        == "/issue --repo m0nklabs/cryptotrader --issue 5"
    )


def test_parse_issue_cancel_command_args():
    """Issue cancellation accepts a run id plus optional audit reason."""
    assert parse_issue_cancel_command_args("#42 stop spend") == (42, "stop spend")
    assert parse_issue_cancel_command_args("42") == (
        42,
        "operator requested cancellation",
    )


@pytest.mark.asyncio
async def test_cancel_issue_resolution_marks_queued_run(tmp_path, monkeypatch):
    """Operators should be able to cancel queued work before execution starts."""
    store = IssueStateStore(tmp_path / "issues.db")
    run = store.enqueue_run(
        IssueResolutionRequest("m0nklabs/cryptotrader", 42, tmp_path),
        run_type=IssueRunType.ISSUE,
    )
    messages: list[str] = []
    comments: list[tuple[int, str]] = []

    async def notify(message: str):
        messages.append(message)

    async def fake_post_issue_audit_comment(_repo, issue_number, body):
        comments.append((issue_number, body))

    monkeypatch.setattr("gateway.issue_resolution.IssueStateStore", lambda: store)
    monkeypatch.setattr(
        "gateway.issue_resolution._post_issue_audit_comment",
        fake_post_issue_audit_comment,
    )

    result = await cancel_issue_resolution(run.id, reason="stop spend", notify=notify)

    cancelled = store.get_run(run.id)
    assert result.cancelled is True
    assert result.status is IssueRunStatus.CANCELLED
    assert cancelled.status is IssueRunStatus.CANCELLED
    assert cancelled.error == "cancelled by operator: stop spend"
    assert any("Cancelled issue run" in message for message in messages)
    assert comments == [
        (
            42,
            f"Hermes audit: run #{run.id} cancelled by operator. Reason: stop spend",
        )
    ]


def test_cancel_run_ignores_terminal_runs(tmp_path):
    """Completed or failed runs should not be moved back into cancellation state."""
    store = IssueStateStore(tmp_path / "issues.db")
    run = store.enqueue_run(
        IssueResolutionRequest("m0nklabs/cryptotrader", 42, tmp_path),
        run_type=IssueRunType.ISSUE,
    )
    store.mark_completed(run.id)

    result = store.cancel_run(run.id, "too late")

    assert result.cancelled is False
    assert result.status is IssueRunStatus.COMPLETED
    assert store.get_run(run.id).status is IssueRunStatus.COMPLETED


def test_issue_repo_allowlist_uses_safe_default(monkeypatch):
    """Issue automation should default to the controlled CryptoTrader playground."""
    monkeypatch.delenv("HERMES_ISSUE_ALLOWED_REPOS", raising=False)

    assert allowed_issue_repos() == ("m0nklabs/cryptotrader",)


@pytest.mark.asyncio
async def test_submit_issue_resolution_blocks_disallowed_repo(tmp_path, monkeypatch):
    """Issue execution should fail before loading unapproved repositories."""
    monkeypatch.delenv("HERMES_ISSUE_ALLOWED_REPOS", raising=False)

    async def fake_load_issue(_repo, _issue_number):
        raise AssertionError("disallowed repositories must not be loaded")

    async def notify(_message: str):
        return None

    monkeypatch.setattr("gateway.issue_resolution._load_issue", fake_load_issue)

    with pytest.raises(RuntimeError, match="not allowed for evil/example"):
        await submit_issue_resolution(
            IssueResolutionRequest("evil/example", 1, tmp_path), notify=notify
        )


@pytest.mark.asyncio
async def test_submit_issue_resolution_accepts_configured_repo(tmp_path, monkeypatch):
    """Operators can explicitly allow additional issue automation repositories."""
    monkeypatch.setenv("HERMES_ISSUE_ALLOWED_REPOS", "m0nklabs/cryptotrader,team/app")

    async def fake_load_issue(_repo, issue_number):
        return IssueMetadata(
            number=issue_number,
            title="Allowed",
            body="body",
            url="https://github.com/team/app/issues/1",
        )

    async def fake_ensure_issue_queue_worker(*, notify=None):
        return None

    async def notify(_message: str):
        return None

    monkeypatch.setattr("gateway.issue_resolution._load_issue", fake_load_issue)
    monkeypatch.setattr(
        "gateway.issue_resolution.ensure_issue_queue_worker",
        fake_ensure_issue_queue_worker,
    )
    store = IssueStateStore(tmp_path / "allowed.db")
    monkeypatch.setattr("gateway.issue_resolution.IssueStateStore", lambda: store)

    result = await submit_issue_resolution(
        IssueResolutionRequest("team/app", 1, tmp_path), notify=notify
    )

    assert result.reused is False
    assert result.status is IssueRunStatus.QUEUED


def test_issue_branch_name_includes_managed_repo_slug():
    """Managed issue branches should be repo-scoped for auditability."""
    issue = IssueMetadata(
        number=42,
        title="Add PnL summary!",
        body="body",
        url="https://github.com/m0nklabs/cryptotrader/issues/42",
    )

    assert (
        _issue_branch_name(issue, "m0nklabs/cryptotrader")
        == "issue/cryptotrader-42-add-pnl-summary"
    )


def test_pr_body_records_issue_run_validation_risk_and_review_contract(tmp_path):
    """Hermes PR bodies should be reviewable without local SQLite context."""
    store = IssueStateStore(tmp_path / "issues.db")
    run = store.enqueue_run(
        IssueResolutionRequest("m0nklabs/cryptotrader", 42, tmp_path),
        run_type=IssueRunType.ISSUE,
    )
    issue = IssueMetadata(
        number=42,
        title="Add PnL summary",
        body="body",
        url="https://github.com/m0nklabs/cryptotrader/issues/42",
    )

    body = _pr_body(
        "m0nklabs/cryptotrader",
        issue,
        "issue/cryptotrader-42-add-pnl-summary",
        "master",
        run=run,
    )

    assert "Closes #42" in body
    assert "Issue: #42 — https://github.com/m0nklabs/cryptotrader/issues/42" in body
    assert "Branch: `issue/cryptotrader-42-add-pnl-summary`" in body
    assert f"Hermes issue run: #{run.id}" in body
    assert "## Validation evidence" in body
    assert "Validation is pending" in body
    assert "## Risk notes" in body
    assert (
        "All implementation changes outside the KyberM0nk framework scope must be submitted through this PR"
        in body
    )
    assert (
        "Direct implementation drift on protected CryptoTrader `master`/`main` is forbidden"
        in body
    )
    assert "## Review handoff" in body
    assert "State: `ready_for_review`" in body
    assert "Reviewer lane: `cloud_reviewer`" in body


def test_local_coder_prompt_requires_pr_branch_and_issue_link():
    """Local coder instructions should prohibit direct downstream edits."""
    issue = IssueMetadata(
        number=42,
        title="Add PnL summary",
        body="body",
        url="https://github.com/m0nklabs/cryptotrader/issues/42",
    )

    prompt = _local_coder_prompt(
        "m0nklabs/cryptotrader",
        issue,
        "issue/cryptotrader-42-add-pnl-summary",
    )

    assert "All changes outside the KyberM0nk framework scope" in prompt
    assert "branch -> PR -> review" in prompt
    assert "Do not switch to `main`, `master`, or any unrelated branch" in prompt
    assert "Issue #42" in prompt


def test_parse_review_routing_tag_detects_findings_for_coder():
    """Reviewer kyber-tags should route fix findings back to the coder lane."""
    tag = parse_review_routing_tag(
        """
        Summary: fixes needed.
        kyber-tag.state=review_findings
        next_action=coding_subagent
        head_ref_oid=abc123
        """
    )

    assert tag is not None
    assert tag.state == "review_findings"
    assert tag.next_action == "coding_subagent"
    assert tag.head_ref_oid == "abc123"
    assert is_review_findings_for_coder(tag) is True


def test_parse_review_routing_tag_ignores_non_coder_next_action():
    """Only coding_subagent findings should create same-branch fix work."""
    tag = parse_review_routing_tag(
        "kyber-tag.state=review_findings\nnext_action=rerun_reviewer\nhead_ref_oid=abc123"
    )

    assert is_review_findings_for_coder(tag) is False


def test_can_merge_pr_requires_ready_for_current_head():
    """Merge gates should fail closed on stale or findings-bearing review tags."""
    pr = PullRequestMetadata(
        number=77,
        url="https://github.com/m0nklabs/cryptotrader/pull/77",
        head_ref_name="issue/cryptotrader-42-add-pnl-summary",
        head_ref_oid="abc123",
    )

    assert (
        can_merge_pr(
            parse_review_routing_tag(
                "kyber-tag.state=ready_for_merge\nnext_action=ready_for_merge\nhead_ref_oid=abc123"
            ),
            pr,
        )
        is True
    )
    assert (
        can_merge_pr(
            parse_review_routing_tag(
                "kyber-tag.state=ready_for_merge\nnext_action=ready_for_merge\nhead_ref_oid=stale"
            ),
            pr,
        )
        is False
    )
    assert (
        can_merge_pr(
            parse_review_routing_tag(
                "kyber-tag.state=ready_for_merge\nnext_action=rerun_reviewer\nhead_ref_oid=abc123"
            ),
            pr,
        )
        is False
    )
    assert (
        can_merge_pr(
            parse_review_routing_tag(
                "kyber-tag.state=review_findings\nnext_action=coding_subagent\nhead_ref_oid=abc123"
            ),
            pr,
        )
        is False
    )
    assert can_merge_pr(None, pr) is False


def test_parse_review_routing_tag_rejects_malformed_tags():
    """Malformed reviewer tags should fail closed instead of implying success."""
    assert parse_review_routing_tag("review looks fine") is None
    assert (
        parse_review_routing_tag(
            "kyber-tag.state=review_clean\nnext_action=ready_for_merge\nhead_ref_oid=abc123"
        )
        is None
    )


def test_parse_review_routing_tag_keeps_tier_and_suggestion_metadata():
    """Reviewer routing tags should preserve tier telemetry fields."""
    tag = parse_review_routing_tag(
        "\n".join(
            [
                "kyber-tag.state=review_findings",
                "next_action=coding_subagent",
                "review_tier=tier1",
                "suggestions_count=3",
                "source=hermes-pr-manager",
                "head_ref_oid=abc123",
            ]
        )
    )

    assert tag is not None
    assert tag.tier == "tier1"
    assert tag.suggestions_count == 3
    assert tag.source == "hermes-pr-manager"


def test_copilot_review_suggestions_route_to_coder_tier1():
    """Actionable Copilot/code-quality comments should route to the coder lane."""
    stats = summarize_review_suggestions(
        reviews=[
            {
                "author": {"login": "copilot-pull-request-reviewer[bot]"},
                "body": "Found one issue.",
            }
        ],
        comments=[
            {
                "user": {"login": "github-code-quality[bot]"},
                "path": "api/routes/execution.py",
                "line": 42,
                "body": "This can break the endpoint; please fix it.",
            }
        ],
    )
    decision = plan_pr_manager_next_action(
        current_reviewer_tier="tier1",
        suggestion_stats=stats,
        findings_cycles=0,
    )

    assert is_copilot_review_author("github-code-quality[bot]") is True
    assert stats.copilot_review_detected is True
    assert stats.total_suggestions_count == 1
    assert decision.next_action == "coding_subagent"
    assert decision.coder_tier == CODER_TIERS[0]


def test_complex_review_findings_escalate_to_coder_tier2():
    """Hard findings should skip the cheap coder after repeated or complex signals."""
    stats = ReviewSuggestionStats(
        internal_suggestions_count=1,
        copilot_suggestions_count=0,
        copilot_review_detected=False,
        complex_findings_detected=True,
    )
    decision = plan_pr_manager_next_action(
        current_reviewer_tier="tier2",
        suggestion_stats=stats,
        findings_cycles=1,
    )

    assert decision.next_action == "coding_subagent"
    assert decision.coder_tier == CODER_TIERS[1]


def test_clean_tier1_escalates_and_clean_tier2_is_ready():
    """Tier1 clean should continue review; Tier2 clean should produce exact merge-ready text."""
    clean_stats = ReviewSuggestionStats(0, 0, False, False)

    tier1_decision = plan_pr_manager_next_action(
        current_reviewer_tier="tier1",
        suggestion_stats=clean_stats,
    )
    tier2_decision = plan_pr_manager_next_action(
        current_reviewer_tier="tier2",
        suggestion_stats=clean_stats,
    )

    assert tier1_decision.next_action == "tier2_review"
    assert tier1_decision.reviewer_tier is not None
    assert tier1_decision.reviewer_tier.provider is ModelProvider.GUARDIAN
    assert tier2_decision.next_action == "ready_for_merge"
    assert tier2_decision.ready_comment == "Ready for merge"
    assert (
        parse_review_routing_tag(
            "kyber-tag.state=ready_for_merge\nnext_action=merge_now\nhead_ref_oid=abc123"
        )
        is None
    )
    assert (
        parse_review_routing_tag(
            "kyber-tag.state=ready_for_merge\nnext_action=ready_for_merge"
        )
        is None
    )


@pytest.mark.asyncio
async def test_execute_single_issue_queues_fix_run_on_review_findings(
    tmp_path, monkeypatch
):
    """review_findings + coding_subagent should create same-branch fix work."""
    store = IssueStateStore(tmp_path / "issues.db")
    store.enqueue_run(
        IssueResolutionRequest("m0nklabs/cryptotrader", 42, tmp_path),
        run_type=IssueRunType.ISSUE,
    )
    run = store.claim_next_run()
    assert run is not None
    issue = IssueMetadata(
        number=42,
        title="Add PnL summary",
        body="body",
        url="https://github.com/m0nklabs/cryptotrader/issues/42",
    )
    messages: list[str] = []

    async def notify(message: str):
        messages.append(message)

    async def fake_run(command, *, cwd=None, env=None, check=True):
        if command[:4] == ["gh", "repo", "view", "m0nklabs/cryptotrader"]:
            return CompletedProcess(
                command=command,
                returncode=0,
                stdout=json.dumps({"defaultBranchRef": {"name": "master"}}),
                stderr="",
            )
        if command == ["git", "checkout", "master"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command == ["git", "pull", "--ff-only", "origin", "master"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command == ["git", "branch", "--show-current"]:
            return CompletedProcess(
                command,
                0,
                stdout="issue/cryptotrader-42-add-pnl-summary\n",
                stderr="",
            )
        if command == ["git", "status", "--porcelain"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["git", "checkout", "-B"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["git", "push", "-u"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["gh", "pr", "create"]:
            return CompletedProcess(
                command,
                0,
                stdout="https://github.com/m0nklabs/cryptotrader/pull/77\n",
                stderr="",
            )
        if command[:3] == ["gh", "pr", "list"]:
            payload = [
                {
                    "number": 77,
                    "url": "https://github.com/m0nklabs/cryptotrader/pull/77",
                    "headRefName": "issue/cryptotrader-42-add-pnl-summary",
                    "headRefOid": "abc123",
                }
            ]
            return CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")
        if command[:3] == ["gh", "pr", "diff"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["gh", "pr", "view"]:
            payload = {
                "state": "OPEN",
                "isDraft": False,
                "mergeStateStatus": "CLEAN",
                "headRefOid": "abc123",
            }
            return CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")
        if command[:3] == ["gh", "issue", "comment"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["gh", "pr", "comment"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["gh", "pr", "review"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if "--message" in command:
            message = command[command.index("--message") + 1]
            if "Review this new PR" in message:
                return CompletedProcess(
                    command,
                    0,
                    stdout=(
                        "kyber-tag.state=review_findings\n"
                        "next_action=coding_subagent\n"
                        "head_ref_oid=abc123\n"
                    ),
                    stderr="",
                )
            return CompletedProcess(command, 0, stdout="local coder done", stderr="")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("gateway.issue_resolution._run", fake_run)
    monkeypatch.setenv("AIDER_BIN", "/opt/aider/bin/aider")
    monkeypatch.setenv("AIDER_GUARDIAN_API_KEY", "local-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "cloud-key")
    monkeypatch.setenv("KYBERM0NK_ENV", str(tmp_path / "missing.env"))

    with pytest.raises(ReviewFindingsRetry):
        await _execute_single_issue(store, run, issue, notify)

    original = store.get_run(run.id)
    assert original.status is IssueRunStatus.RUNNING
    assert original.branch == "issue/cryptotrader-42-add-pnl-summary"
    assert original.review_findings_count == 1
    assert any("keeping run #" in message for message in messages)


@pytest.mark.asyncio
async def test_execute_single_issue_trips_review_findings_circuit_breaker(
    tmp_path, monkeypatch
):
    """Repeated review findings should escalate instead of ping-ponging forever."""
    store = IssueStateStore(tmp_path / "issues.db")
    store.enqueue_run(
        IssueResolutionRequest("m0nklabs/cryptotrader", 42, tmp_path),
        run_type=IssueRunType.ISSUE,
    )
    run = store.claim_next_run()
    assert run is not None
    store.record_review_findings(run.id)
    store.record_review_findings(run.id)
    issue = IssueMetadata(
        number=42,
        title="Add PnL summary",
        body="body",
        url="https://github.com/m0nklabs/cryptotrader/issues/42",
    )

    async def notify(_message: str):
        return None

    async def fake_run(command, *, cwd=None, env=None, check=True):
        if command[:4] == ["gh", "repo", "view", "m0nklabs/cryptotrader"]:
            return CompletedProcess(
                command=command,
                returncode=0,
                stdout=json.dumps({"defaultBranchRef": {"name": "master"}}),
                stderr="",
            )
        if command == ["git", "checkout", "master"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command == ["git", "pull", "--ff-only", "origin", "master"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command == ["git", "branch", "--show-current"]:
            return CompletedProcess(
                command,
                0,
                stdout="issue/cryptotrader-42-add-pnl-summary\n",
                stderr="",
            )
        if command == ["git", "status", "--porcelain"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["git", "checkout", "-B"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["git", "push", "-u"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["gh", "pr", "create"]:
            return CompletedProcess(
                command,
                0,
                stdout="https://github.com/m0nklabs/cryptotrader/pull/77\n",
                stderr="",
            )
        if command[:3] == ["gh", "pr", "list"]:
            payload = [
                {
                    "number": 77,
                    "url": "https://github.com/m0nklabs/cryptotrader/pull/77",
                    "headRefName": "issue/cryptotrader-42-add-pnl-summary",
                    "headRefOid": "abc123",
                }
            ]
            return CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")
        if command[:3] == ["gh", "pr", "diff"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["gh", "pr", "view"]:
            payload = {
                "state": "OPEN",
                "isDraft": False,
                "mergeStateStatus": "CLEAN",
                "headRefOid": "abc123",
            }
            return CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")
        if command[:3] == ["gh", "issue", "comment"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["gh", "pr", "comment"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["gh", "pr", "review"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if "--message" in command:
            message = command[command.index("--message") + 1]
            if "Review this new PR" in message:
                return CompletedProcess(
                    command,
                    0,
                    stdout=(
                        "kyber-tag.state=review_findings\n"
                        "next_action=coding_subagent\n"
                        "head_ref_oid=abc123\n"
                    ),
                    stderr="",
                )
            return CompletedProcess(command, 0, stdout="local coder done", stderr="")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("gateway.issue_resolution._run", fake_run)
    monkeypatch.setenv("AIDER_BIN", "/opt/aider/bin/aider")
    monkeypatch.setenv("AIDER_GUARDIAN_API_KEY", "local-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "cloud-key")
    monkeypatch.setenv("KYBERM0NK_ENV", str(tmp_path / "missing.env"))

    with pytest.raises(ReviewLoopCircuitBreaker):
        await _execute_single_issue(store, run, issue, notify)

    assert store.get_run(run.id).review_findings_count == 3


@pytest.mark.asyncio
async def test_execute_single_issue_merges_and_closes_ready_for_merge(
    tmp_path, monkeypatch
):
    """Current-head ready_for_merge reviews should merge PRs and close issues."""
    from hermes_cli import kanban_db as kb

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setenv("HERMES_ISSUE_AUTO_MERGE_ENABLED", "1")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db(board="production")
    with kb.connect(board="production") as conn:
        kanban_task_id = kb.create_task(conn, title="Resolve issue 42")

    store = IssueStateStore(tmp_path / "issues.db")
    store.enqueue_run(
        IssueResolutionRequest(
            "m0nklabs/cryptotrader",
            42,
            tmp_path,
            kanban_task_id=kanban_task_id,
            kanban_board="production",
        ),
        run_type=IssueRunType.ISSUE,
    )
    run = store.claim_next_run()
    assert run is not None
    issue = IssueMetadata(
        number=42,
        title="Add PnL summary",
        body="body",
        url="https://github.com/m0nklabs/cryptotrader/issues/42",
    )
    commands: list[list[str]] = []
    messages: list[str] = []
    pr_list_calls = 0

    async def notify(message: str):
        messages.append(message)

    async def fake_run(command, *, cwd=None, env=None, check=True):
        nonlocal pr_list_calls
        commands.append(command)
        if command[:4] == ["gh", "repo", "view", "m0nklabs/cryptotrader"]:
            return CompletedProcess(
                command=command,
                returncode=0,
                stdout=json.dumps({"defaultBranchRef": {"name": "master"}}),
                stderr="",
            )
        if command == ["git", "checkout", "master"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command == ["git", "pull", "--ff-only", "origin", "master"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command == ["git", "branch", "--show-current"]:
            return CompletedProcess(
                command,
                0,
                stdout="issue/cryptotrader-42-add-pnl-summary\n",
                stderr="",
            )
        if command == ["git", "status", "--porcelain"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["git", "checkout", "-B"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["git", "push", "-u"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["gh", "pr", "create"]:
            return CompletedProcess(
                command,
                0,
                stdout="https://github.com/m0nklabs/cryptotrader/pull/77\n",
                stderr="",
            )
        if command[:3] == ["gh", "pr", "list"]:
            pr_list_calls += 1
            if pr_list_calls == 1:
                return CompletedProcess(command, 0, stdout="[]", stderr="")
            payload = [
                {
                    "number": 77,
                    "url": "https://github.com/m0nklabs/cryptotrader/pull/77",
                    "headRefName": "issue/cryptotrader-42-add-pnl-summary",
                    "headRefOid": "abc123",
                }
            ]
            return CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")
        if command[:3] == ["gh", "pr", "diff"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["gh", "pr", "view"]:
            payload = {
                "state": "OPEN",
                "isDraft": False,
                "mergeStateStatus": "CLEAN",
                "headRefOid": "abc123",
            }
            return CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")
        if command[:3] == ["gh", "issue", "comment"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["gh", "pr", "comment"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["gh", "pr", "review"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["gh", "pr", "merge"]:
            return CompletedProcess(command, 0, stdout="merged", stderr="")
        if command[:3] == ["gh", "issue", "close"]:
            return CompletedProcess(command, 0, stdout="closed", stderr="")
        if "--message" in command:
            message = command[command.index("--message") + 1]
            if "Review this new PR" in message:
                return CompletedProcess(
                    command,
                    0,
                    stdout=(
                        "kyber-tag.state=ready_for_merge\n"
                        "next_action=ready_for_merge\n"
                        "head_ref_oid=abc123\n"
                    ),
                    stderr="",
                )
            return CompletedProcess(command, 0, stdout="local coder done", stderr="")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("gateway.issue_resolution._run", fake_run)
    monkeypatch.setenv("AIDER_BIN", "/opt/aider/bin/aider")
    monkeypatch.setenv("AIDER_GUARDIAN_API_KEY", "local-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "cloud-key")
    monkeypatch.setenv("KYBERM0NK_ENV", str(tmp_path / "missing.env"))

    await _execute_single_issue(store, run, issue, notify)

    assert store.get_run(run.id).status is IssueRunStatus.COMPLETED
    assert any(command[:3] == ["gh", "pr", "merge"] for command in commands)
    assert any(command[:3] == ["gh", "issue", "close"] for command in commands)
    assert any("merged and Issue #42 closed" in message for message in messages)
    with kb.connect(board="production") as conn:
        task = kb.get_task(conn, kanban_task_id)
        events = kb.list_events(conn, kanban_task_id)
        comments = kb.list_comments(conn, kanban_task_id)

    assert task is not None
    assert task.status == "done"
    event_kinds = [event.kind for event in events]
    assert "issue_run_claimed" in event_kinds
    assert "pr_opened" in event_kinds
    assert "review_requested" in event_kinds
    assert "merge_started" in event_kinds
    assert "pr_merged" in event_kinds
    assert "issue_closed" in event_kinds
    assert "issue_run_completed" in event_kinds
    assert "completed" in event_kinds
    assert any("PR #77 merged automatically" in comment.body for comment in comments)
    assert any("GitHub issue m0nklabs/cryptotrader#42 closed" in comment.body for comment in comments)


@pytest.mark.asyncio
async def test_merge_ready_pr_blocks_when_auto_merge_disabled(tmp_path, monkeypatch):
    """The PR manager should not merge unless explicitly enabled."""
    monkeypatch.delenv("HERMES_ISSUE_AUTO_MERGE_ENABLED", raising=False)
    comments: list[str] = []

    async def fake_post_pr_audit_comment(_repo, _pr, body):
        comments.append(body)

    monkeypatch.setattr(
        "gateway.issue_resolution._post_pr_audit_comment",
        fake_post_pr_audit_comment,
    )
    run = IssueRun(
        id=123,
        repo="m0nklabs/cryptotrader",
        issue_number=42,
        workdir=tmp_path,
        branch="issue/42-test",
        kanban_task_id=None,
        kanban_board=None,
        status=IssueRunStatus.RUNNING,
        run_type=IssueRunType.ISSUE,
        parent_run_id=None,
        master_issue_number=None,
        pr_number=77,
        pr_url="https://github.com/m0nklabs/cryptotrader/pull/77",
        error=None,
    )
    issue = IssueMetadata(42, "Issue", "body", "https://example.invalid/issue")
    pr = PullRequestMetadata(77, "https://example.invalid/pr", "issue/42-test", "abc123")

    with pytest.raises(ReviewMergeGateError, match="automatic merge disabled"):
        await _merge_ready_pr("m0nklabs/cryptotrader", issue, pr, run)

    assert any("automatic merge disabled" in comment for comment in comments)


@pytest.mark.asyncio
async def test_execute_single_issue_blocks_stale_ready_for_merge_tag(
    tmp_path, monkeypatch
):
    """Stale ready_for_merge tags should fail closed without merging."""
    store = IssueStateStore(tmp_path / "issues.db")
    store.enqueue_run(
        IssueResolutionRequest("m0nklabs/cryptotrader", 42, tmp_path),
        run_type=IssueRunType.ISSUE,
    )
    run = store.claim_next_run()
    assert run is not None
    issue = IssueMetadata(
        number=42,
        title="Add PnL summary",
        body="body",
        url="https://github.com/m0nklabs/cryptotrader/issues/42",
    )
    commands: list[list[str]] = []

    async def notify(_message: str):
        return None

    async def fake_run(command, *, cwd=None, env=None, check=True):
        commands.append(command)
        if command[:4] == ["gh", "repo", "view", "m0nklabs/cryptotrader"]:
            return CompletedProcess(
                command=command,
                returncode=0,
                stdout=json.dumps({"defaultBranchRef": {"name": "master"}}),
                stderr="",
            )
        if command == ["git", "checkout", "master"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command == ["git", "pull", "--ff-only", "origin", "master"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command == ["git", "branch", "--show-current"]:
            return CompletedProcess(
                command,
                0,
                stdout="issue/cryptotrader-42-add-pnl-summary\n",
                stderr="",
            )
        if command == ["git", "status", "--porcelain"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["git", "checkout", "-B"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["git", "push", "-u"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["gh", "pr", "create"]:
            return CompletedProcess(
                command,
                0,
                stdout="https://github.com/m0nklabs/cryptotrader/pull/77\n",
                stderr="",
            )
        if command[:3] == ["gh", "pr", "list"]:
            payload = [
                {
                    "number": 77,
                    "url": "https://github.com/m0nklabs/cryptotrader/pull/77",
                    "headRefName": "issue/cryptotrader-42-add-pnl-summary",
                    "headRefOid": "abc123",
                }
            ]
            return CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")
        if command[:3] == ["gh", "pr", "diff"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["gh", "pr", "view"]:
            payload = {
                "state": "OPEN",
                "isDraft": False,
                "mergeStateStatus": "CLEAN",
                "headRefOid": "abc123",
            }
            return CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")
        if command[:3] == ["gh", "issue", "comment"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["gh", "pr", "comment"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["gh", "pr", "review"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if "--message" in command:
            message = command[command.index("--message") + 1]
            if "Review this new PR" in message:
                return CompletedProcess(
                    command,
                    0,
                    stdout=(
                        "kyber-tag.state=ready_for_merge\n"
                        "next_action=ready_for_merge\n"
                        "head_ref_oid=stale\n"
                    ),
                    stderr="",
                )
            return CompletedProcess(command, 0, stdout="local coder done", stderr="")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("gateway.issue_resolution._run", fake_run)
    monkeypatch.setenv("AIDER_BIN", "/opt/aider/bin/aider")
    monkeypatch.setenv("AIDER_GUARDIAN_API_KEY", "local-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "cloud-key")
    monkeypatch.setenv("KYBERM0NK_ENV", str(tmp_path / "missing.env"))

    with pytest.raises(ReviewMergeGateError):
        await _execute_single_issue(store, run, issue, notify)

    assert store.get_run(run.id).status is IssueRunStatus.RUNNING
    assert not any(command[:3] == ["gh", "pr", "merge"] for command in commands)
    assert not any(command[:3] == ["gh", "issue", "close"] for command in commands)


@pytest.mark.asyncio
async def test_execute_single_issue_retries_malformed_review_tag_once(
    tmp_path, monkeypatch
):
    """Malformed review tags should get one bounded reviewer rerun."""
    monkeypatch.setenv("HERMES_ISSUE_AUTO_MERGE_ENABLED", "1")
    store = IssueStateStore(tmp_path / "issues.db")
    store.enqueue_run(
        IssueResolutionRequest("m0nklabs/cryptotrader", 42, tmp_path),
        run_type=IssueRunType.ISSUE,
    )
    run = store.claim_next_run()
    assert run is not None
    issue = IssueMetadata(
        number=42,
        title="Add PnL summary",
        body="body",
        url="https://github.com/m0nklabs/cryptotrader/issues/42",
    )
    messages: list[str] = []
    review_calls = 0

    async def notify(message: str):
        messages.append(message)

    async def fake_run(command, *, cwd=None, env=None, check=True):
        nonlocal review_calls
        if command[:4] == ["gh", "repo", "view", "m0nklabs/cryptotrader"]:
            return CompletedProcess(
                command=command,
                returncode=0,
                stdout=json.dumps({"defaultBranchRef": {"name": "master"}}),
                stderr="",
            )
        if command == ["git", "checkout", "master"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command == ["git", "pull", "--ff-only", "origin", "master"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command == ["git", "branch", "--show-current"]:
            return CompletedProcess(
                command,
                0,
                stdout="issue/cryptotrader-42-add-pnl-summary\n",
                stderr="",
            )
        if command == ["git", "status", "--porcelain"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["git", "checkout", "-B"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["git", "push", "-u"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["gh", "pr", "create"]:
            return CompletedProcess(
                command,
                0,
                stdout="https://github.com/m0nklabs/cryptotrader/pull/77\n",
                stderr="",
            )
        if command[:3] == ["gh", "pr", "list"]:
            payload = [
                {
                    "number": 77,
                    "url": "https://github.com/m0nklabs/cryptotrader/pull/77",
                    "headRefName": "issue/cryptotrader-42-add-pnl-summary",
                    "headRefOid": "abc123",
                }
            ]
            return CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")
        if command[:3] == ["gh", "pr", "diff"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["gh", "pr", "view"]:
            payload = {
                "state": "OPEN",
                "isDraft": False,
                "mergeStateStatus": "CLEAN",
                "headRefOid": "abc123",
            }
            return CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")
        if command[:3] == ["gh", "issue", "comment"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["gh", "pr", "comment"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["gh", "pr", "review"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["gh", "pr", "merge"]:
            return CompletedProcess(command, 0, stdout="merged", stderr="")
        if command[:3] == ["gh", "issue", "close"]:
            return CompletedProcess(command, 0, stdout="closed", stderr="")
        if "--message" in command:
            message = command[command.index("--message") + 1]
            if "Review this new PR" in message:
                review_calls += 1
                if review_calls == 1:
                    return CompletedProcess(command, 0, stdout="looks clean", stderr="")
                return CompletedProcess(
                    command,
                    0,
                    stdout=(
                        "kyber-tag.state=ready_for_merge\n"
                        "next_action=ready_for_merge\n"
                        "head_ref_oid=abc123\n"
                    ),
                    stderr="",
                )
            return CompletedProcess(command, 0, stdout="local coder done", stderr="")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("gateway.issue_resolution._run", fake_run)
    monkeypatch.setenv("AIDER_BIN", "/opt/aider/bin/aider")
    monkeypatch.setenv("AIDER_GUARDIAN_API_KEY", "local-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "cloud-key")
    monkeypatch.setenv("KYBERM0NK_ENV", str(tmp_path / "missing.env"))

    await _execute_single_issue(store, run, issue, notify)

    assert review_calls == 3
    assert store.get_run(run.id).status is IssueRunStatus.COMPLETED
    assert any("retrying once" in message for message in messages)


@pytest.mark.asyncio
async def test_execute_single_issue_fails_after_repeated_malformed_review_tag(
    tmp_path, monkeypatch
):
    """Repeated malformed tags should escalate as a failed review parse."""
    store = IssueStateStore(tmp_path / "issues.db")
    store.enqueue_run(
        IssueResolutionRequest("m0nklabs/cryptotrader", 42, tmp_path),
        run_type=IssueRunType.ISSUE,
    )
    run = store.claim_next_run()
    assert run is not None
    issue = IssueMetadata(
        number=42,
        title="Add PnL summary",
        body="body",
        url="https://github.com/m0nklabs/cryptotrader/issues/42",
    )
    review_calls = 0

    async def fake_run(command, *, cwd=None, env=None, check=True):
        nonlocal review_calls
        if command[:4] == ["gh", "repo", "view", "m0nklabs/cryptotrader"]:
            return CompletedProcess(
                command=command,
                returncode=0,
                stdout=json.dumps({"defaultBranchRef": {"name": "master"}}),
                stderr="",
            )
        if command == ["git", "checkout", "master"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command == ["git", "pull", "--ff-only", "origin", "master"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command == ["git", "branch", "--show-current"]:
            return CompletedProcess(
                command,
                0,
                stdout="issue/cryptotrader-42-add-pnl-summary\n",
                stderr="",
            )
        if command == ["git", "status", "--porcelain"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["git", "checkout", "-B"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["git", "push", "-u"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["gh", "pr", "create"]:
            return CompletedProcess(
                command,
                0,
                stdout="https://github.com/m0nklabs/cryptotrader/pull/77\n",
                stderr="",
            )
        if command[:3] == ["gh", "pr", "list"]:
            payload = [
                {
                    "number": 77,
                    "url": "https://github.com/m0nklabs/cryptotrader/pull/77",
                    "headRefName": "issue/cryptotrader-42-add-pnl-summary",
                    "headRefOid": "abc123",
                }
            ]
            return CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")
        if command[:3] == ["gh", "issue", "comment"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["gh", "pr", "comment"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if "--message" in command:
            message = command[command.index("--message") + 1]
            if "Review this new PR" in message:
                review_calls += 1
                return CompletedProcess(command, 0, stdout="still malformed", stderr="")
            return CompletedProcess(command, 0, stdout="local coder done", stderr="")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("gateway.issue_resolution._run", fake_run)
    monkeypatch.setenv("AIDER_BIN", "/opt/aider/bin/aider")
    monkeypatch.setenv("AIDER_GUARDIAN_API_KEY", "local-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "cloud-key")
    monkeypatch.setenv("KYBERM0NK_ENV", str(tmp_path / "missing.env"))

    async def notify(_message: str):
        return None

    with pytest.raises(ReviewTagParseError):
        await _execute_single_issue(store, run, issue, notify)

    assert review_calls == 6


def test_master_issue_label_detection():
    """The master-plan label promotes an issue to the epic lane."""
    issue = IssueMetadata(
        number=1,
        title="Roadmap",
        body="normal body",
        url="https://github.com/m0nklabs/cryptotrader/issues/1",
        labels=("master-plan",),
    )

    assert is_master_issue(issue) is True


def test_master_issue_heading_detection():
    """A master-plan heading promotes an issue even without labels."""
    issue = IssueMetadata(
        number=1,
        title="Roadmap",
        body="# Master Project Plan\n\nBuild the thing.",
        url="https://github.com/m0nklabs/cryptotrader/issues/1",
    )

    assert is_master_issue(issue) is True


def test_parse_issue_next_command_parses_repo_and_workdir(tmp_path):
    """The next-open-issue command should accept a repo plus workdir."""
    request = parse_issue_next_command_args(
        f"m0nklabs/cryptotrader --workdir {tmp_path}"
    )

    assert request == IssueSelectionRequest(
        repo="m0nklabs/cryptotrader",
        workdir=tmp_path,
        branch=None,
    )


def test_parse_decomposition_response_json_object():
    """Guardian JSON object responses become ordered EpicTask records."""
    tasks = parse_decomposition_response(
        '{"tasks":[{"title":"Add API","body":"Create endpoint"},"Add tests"]}'
    )

    assert tasks == [
        EpicTask(title="Add API", body="Create endpoint"),
        EpicTask(title="Add tests", body="Add tests"),
    ]


def test_issue_state_store_persists_fifo_and_resets_running(tmp_path):
    """SQLite state stores queued runs and resets interrupted local coder work."""
    store = IssueStateStore(tmp_path / "issues.db")
    first = store.enqueue_run(
        IssueResolutionRequest(
            "m0nklabs/cryptotrader",
            1,
            tmp_path,
            kanban_task_id="t_abc123",
            kanban_board="production",
        ),
        run_type=IssueRunType.ISSUE,
    )
    second = store.enqueue_run(
        IssueResolutionRequest("m0nklabs/cryptotrader", 2, tmp_path),
        run_type=IssueRunType.ISSUE,
    )

    claimed = store.claim_next_run()

    assert claimed is not None
    assert claimed.id == first.id
    assert claimed.status is IssueRunStatus.RUNNING
    assert claimed.kanban_task_id == "t_abc123"
    assert claimed.kanban_board == "production"
    assert store.get_run(second.id).status is IssueRunStatus.QUEUED
    assert store.reset_interrupted_runs() == 1
    assert store.get_run(first.id).status is IssueRunStatus.QUEUED


def test_issue_state_store_retries_failed_runs_with_backoff(tmp_path, monkeypatch):
    """Failed issue runs should be delayed and retried before permanent failure."""
    current_time = 1000.0
    monkeypatch.setattr("gateway.issue_resolution._now", lambda: current_time)
    store = IssueStateStore(tmp_path / "issues.db")
    run = store.enqueue_run(
        IssueResolutionRequest("m0nklabs/cryptotrader", 1, tmp_path),
        run_type=IssueRunType.ISSUE,
    )

    claimed = store.claim_next_run()

    assert claimed is not None
    assert store.mark_retry_or_failed(claimed, "temporary failure") is True
    retrying = store.get_run(run.id)
    assert retrying.status is IssueRunStatus.QUEUED
    assert retrying.attempt_count == 1
    assert retrying.next_attempt_at == current_time + 60
    assert store.claim_next_run() is None
    assert store.next_queued_delay() == 60


def test_issue_state_store_fails_after_retry_budget(tmp_path, monkeypatch):
    """Retry budget prevents a broken issue run from looping forever."""
    current_time = 1000.0

    def now():
        return current_time

    monkeypatch.setattr("gateway.issue_resolution._now", now)
    store = IssueStateStore(tmp_path / "issues.db")
    store.enqueue_run(
        IssueResolutionRequest("m0nklabs/cryptotrader", 1, tmp_path),
        run_type=IssueRunType.ISSUE,
    )

    first = store.claim_next_run()
    assert first is not None
    assert store.mark_retry_or_failed(first, "temporary failure") is True

    current_time += 60
    second = store.claim_next_run()
    assert second is not None
    assert store.mark_retry_or_failed(second, "temporary failure") is True

    current_time += 300
    third = store.claim_next_run()
    assert third is not None
    assert store.mark_retry_or_failed(third, "permanent failure") is False

    failed = store.get_run(third.id)
    assert failed.status is IssueRunStatus.FAILED
    assert failed.attempt_count == 3


@pytest.mark.asyncio
async def test_find_existing_sub_issue_matches_master_task(monkeypatch):
    """Crash retries should reuse already-created Master Epic sub-issues."""
    task = EpicTask(title="Task one", body="Do one")
    master = IssueMetadata(
        number=10,
        title="Master",
        body="# Master Project Plan\n\nShip it.",
        url="https://github.com/m0nklabs/cryptotrader/issues/10",
    )

    async def fake_run(command, *, cwd=None, env=None, check=True):
        assert command[:3] == ["gh", "issue", "list"]
        payload = [
            {
                "number": 182,
                "title": "Task one",
                "body": "Part of Master Issue #10.\n\n## Task 1\n\nDo one",
                "url": "https://github.com/m0nklabs/cryptotrader/issues/182",
            }
        ]
        return CompletedProcess(command, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr("gateway.issue_resolution._run", fake_run)

    existing = await _find_existing_sub_issue("m0nklabs/cryptotrader", master, task, 1)

    assert existing is not None
    assert existing.number == 182
    assert existing.title == "Task one"


@pytest.mark.asyncio
async def test_master_expansion_creates_persisted_subissue_runs(tmp_path, monkeypatch):
    """Master expansion creates GitHub sub-issues and queues each one."""
    store = IssueStateStore(tmp_path / "issues.db")
    run = store.enqueue_run(
        IssueResolutionRequest("m0nklabs/cryptotrader", 10, tmp_path),
        run_type=IssueRunType.MASTER,
    )
    issue = IssueMetadata(
        number=10,
        title="Master",
        body="# Master Project Plan\n\nShip it.",
        url="https://github.com/m0nklabs/cryptotrader/issues/10",
    )

    async def fake_decompose(_issue):
        return [
            EpicTask(title="Task one", body="Do one"),
            EpicTask(title="Task two", body="Do two"),
        ]

    async def fake_find_existing_sub_issue(_repo, _master_issue, _task, _position):
        return None

    async def fake_create_sub_issue(_repo, _master_issue, task, position):
        return IssueMetadata(
            number=100 + position,
            title=task.title,
            body=task.body,
            url=f"https://github.com/m0nklabs/cryptotrader/issues/{100 + position}",
        )

    messages: list[str] = []

    async def notify(message: str):
        messages.append(message)

    monkeypatch.setattr(
        "gateway.issue_resolution.decompose_master_plan", fake_decompose
    )
    monkeypatch.setattr(
        "gateway.issue_resolution._find_existing_sub_issue",
        fake_find_existing_sub_issue,
    )
    monkeypatch.setattr(
        "gateway.issue_resolution._create_sub_issue", fake_create_sub_issue
    )

    await _execute_master_issue(store, run, issue, notify)

    master = store.get_run(run.id)
    children = store.list_child_runs(run.id)

    assert master.status is IssueRunStatus.EXPANDED
    assert [child.issue_number for child in children] == [101, 102]
    assert [child.status for child in children] == [
        IssueRunStatus.QUEUED,
        IssueRunStatus.QUEUED,
    ]
    assert [child.run_type for child in children] == [
        IssueRunType.SUB_ISSUE,
        IssueRunType.SUB_ISSUE,
    ]
    assert any("expanded into 2" in message for message in messages)


@pytest.mark.asyncio
async def test_master_expansion_reuses_existing_subissues(tmp_path, monkeypatch):
    """Master expansion should not duplicate a sub-issue found after a crash."""
    store = IssueStateStore(tmp_path / "issues.db")
    run = store.enqueue_run(
        IssueResolutionRequest("m0nklabs/cryptotrader", 10, tmp_path),
        run_type=IssueRunType.MASTER,
    )
    issue = IssueMetadata(
        number=10,
        title="Master",
        body="# Master Project Plan\n\nShip it.",
        url="https://github.com/m0nklabs/cryptotrader/issues/10",
    )
    created: list[EpicTask] = []

    async def fake_decompose(_issue):
        return [EpicTask(title="Task one", body="Do one")]

    async def fake_find_existing_sub_issue(_repo, _master_issue, task, _position):
        return IssueMetadata(
            number=182,
            title=task.title,
            body=task.body,
            url="https://github.com/m0nklabs/cryptotrader/issues/182",
        )

    async def fake_create_sub_issue(_repo, _master_issue, task, _position):
        created.append(task)
        return IssueMetadata(
            number=999,
            title=task.title,
            body=task.body,
            url="https://github.com/m0nklabs/cryptotrader/issues/999",
        )

    async def notify(_message: str):
        return None

    monkeypatch.setattr(
        "gateway.issue_resolution.decompose_master_plan", fake_decompose
    )
    monkeypatch.setattr(
        "gateway.issue_resolution._find_existing_sub_issue",
        fake_find_existing_sub_issue,
    )
    monkeypatch.setattr(
        "gateway.issue_resolution._create_sub_issue", fake_create_sub_issue
    )

    await _execute_master_issue(store, run, issue, notify)

    children = store.list_child_runs(run.id)

    assert created == []
    assert [child.issue_number for child in children] == [182]


@pytest.mark.asyncio
async def test_managed_repo_guard_blocks_dirty_protected_branch(tmp_path, monkeypatch):
    """CryptoTrader implementation must not start from dirty master/main."""
    issue = IssueMetadata(
        number=12,
        title="Fix drift",
        body="body",
        url="https://github.com/m0nklabs/cryptotrader/issues/12",
    )
    run = IssueStateStore(tmp_path / "issues.db").enqueue_run(
        IssueResolutionRequest("m0nklabs/cryptotrader", issue.number, tmp_path),
        run_type=IssueRunType.ISSUE,
    )

    async def fake_run(command, *, cwd=None, env=None, check=True):
        if command == ["git", "branch", "--show-current"]:
            return CompletedProcess(
                command=command, returncode=0, stdout="master\n", stderr=""
            )
        if command == ["git", "status", "--porcelain"]:
            return CompletedProcess(
                command=command, returncode=0, stdout=" M core/trading.py\n", stderr=""
            )
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("gateway.issue_resolution._run", fake_run)

    with pytest.raises(
        RuntimeError, match="protected branch 'master' has implementation drift"
    ):
        await _guard_managed_repo_before_issue_dispatch(
            run, issue, "issue/12-fix-drift"
        )


@pytest.mark.asyncio
async def test_managed_repo_guard_allows_only_aider_noise_on_protected_branch(
    tmp_path, monkeypatch
):
    """Aider history/cache files are allowed local noise on protected branches."""

    async def fake_run(command, *, cwd=None, env=None, check=True):
        if command == ["git", "branch", "--show-current"]:
            return CompletedProcess(
                command=command, returncode=0, stdout="master\n", stderr=""
            )
        if command == ["git", "status", "--porcelain"]:
            return CompletedProcess(
                command=command,
                returncode=0,
                stdout="?? .aider.chat.history.md\n?? .aider.tags.cache.v4/tags\n",
                stderr="",
            )
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("gateway.issue_resolution._run", fake_run)

    status = await _inspect_managed_repo("m0nklabs/cryptotrader", tmp_path)

    assert status is not None
    assert status.ok is True
    assert status.violating_paths == ()
    assert status.ignored_paths == (
        ".aider.chat.history.md",
        ".aider.tags.cache.v4/tags",
    )


@pytest.mark.asyncio
async def test_issue_branch_ready_guard_blocks_branch_escape(tmp_path, monkeypatch):
    """Coder output must still be on the expected issue branch before PR creation."""

    async def fake_run(command, *, cwd=None, env=None, check=True):
        if command == ["git", "branch", "--show-current"]:
            return CompletedProcess(command=command, returncode=0, stdout="master\n", stderr="")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("gateway.issue_resolution._run", fake_run)

    with pytest.raises(RuntimeError, match="expected issue branch"):
        await _assert_issue_branch_ready_for_pr(
            "m0nklabs/cryptotrader",
            tmp_path,
            "issue/cryptotrader-12-fix-drift",
            "master",
        )


@pytest.mark.asyncio
async def test_issue_branch_ready_guard_blocks_uncommitted_drift(
    tmp_path, monkeypatch
):
    """Implementation changes must be committed to the issue branch before PR handoff."""

    async def fake_run(command, *, cwd=None, env=None, check=True):
        if command == ["git", "branch", "--show-current"]:
            return CompletedProcess(
                command=command,
                returncode=0,
                stdout="issue/cryptotrader-12-fix-drift\n",
                stderr="",
            )
        if command == ["git", "status", "--porcelain"]:
            return CompletedProcess(
                command=command,
                returncode=0,
                stdout=" M core/trading.py\n?? .aider.chat.history.md\n",
                stderr="",
            )
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("gateway.issue_resolution._run", fake_run)

    with pytest.raises(RuntimeError, match="uncommitted implementation drift"):
        await _assert_issue_branch_ready_for_pr(
            "m0nklabs/cryptotrader",
            tmp_path,
            "issue/cryptotrader-12-fix-drift",
            "master",
        )


@pytest.mark.asyncio
async def test_prepare_issue_branch_from_synced_default_refreshes_base_first(
    tmp_path, monkeypatch
):
    """Every new issue branch must start from a freshly pulled default branch."""
    commands: list[list[str]] = []

    async def fake_run(command, *, cwd=None, env=None, check=True):
        commands.append(command)
        return CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("gateway.issue_resolution._run", fake_run)

    await _prepare_issue_branch_from_synced_default(
        tmp_path,
        "master",
        "issue/cryptotrader-12-fix-drift",
    )

    assert commands == [
        ["git", "checkout", "master"],
        ["git", "pull", "--ff-only", "origin", "master"],
        ["git", "checkout", "-B", "issue/cryptotrader-12-fix-drift", "master"],
    ]


@pytest.mark.asyncio
async def test_push_issue_branch_blocks_direct_master_pushes(tmp_path, monkeypatch):
    """Hermes must abort if a push is attempted from master/main."""

    async def fake_run(command, *, cwd=None, env=None, check=True):
        if command == ["git", "branch", "--show-current"]:
            return CompletedProcess(command, 0, stdout="master\n", stderr="")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("gateway.issue_resolution._run", fake_run)

    with pytest.raises(
        RuntimeError,
        match="ERROR: Direct pushes to master are strictly forbidden by operator flip\\.",
    ):
        await _push_issue_branch(tmp_path, "issue/cryptotrader-12-fix-drift")


@pytest.mark.asyncio
async def test_managed_repo_guard_skips_unmanaged_repos(tmp_path):
    """Only configured managed repos should receive the CryptoTrader guard."""
    status = await _inspect_managed_repo("m0nklabs/other", tmp_path)

    assert status is None


@pytest.mark.asyncio
async def test_load_next_open_issue_selects_oldest_issue(monkeypatch):
    """Hermes should pick the oldest open issue, not a random one."""

    async def fake_run(command, *, cwd=None, env=None, check=True):
        assert command[:4] == ["gh", "issue", "list", "--repo"]
        payload = [
            {"number": 12, "createdAt": "2026-05-29T12:00:00Z"},
            {"number": 7, "createdAt": "2026-05-28T12:00:00Z"},
        ]
        return CompletedProcess(
            command=command, returncode=0, stdout=json.dumps(payload), stderr=""
        )

    async def fake_load_issue(repo, issue_number):
        return IssueMetadata(
            number=issue_number,
            title=f"Issue {issue_number}",
            body="body",
            url=f"https://github.com/{repo}/issues/{issue_number}",
        )

    monkeypatch.setattr("gateway.issue_resolution._run", fake_run)
    monkeypatch.setattr("gateway.issue_resolution._load_issue", fake_load_issue)

    issue = await _load_next_open_issue("m0nklabs/cryptotrader")

    assert issue.number == 7


def test_local_aider_invocation_targets_guardian(tmp_path):
    """The local coder should force Aider through Guardian."""
    invocation = build_aider_invocation(
        AiderRole.LOCAL_CODER,
        tmp_path,
        "fix it",
        runtime_env={
            "AIDER_BIN": "/opt/aider/bin/aider",
            "AIDER_GUARDIAN_API_KEY": "local-key",
            "GUARDIAN_BASE_URL": "http://host.docker.internal:11434/v1",
            "KYBERM0NK_ENV": str(tmp_path / "missing.env"),
        },
    )

    assert invocation.cwd == tmp_path
    assert invocation.command[:4] == [
        "/opt/aider/bin/aider",
        "--model",
        "openai/qwen3-35b-uncensored",
        "--yes",
    ]
    assert invocation.env["OPENAI_API_BASE"] == "http://127.0.0.1:11434/v1"
    assert invocation.env["OPENAI_API_KEY"] == "local-key"


def test_cloud_aider_invocation_targets_openrouter(tmp_path):
    """The cloud reviewer should force Aider through OpenRouter without commits."""
    invocation = build_aider_invocation(
        AiderRole.CLOUD_REVIEWER,
        Path(tmp_path),
        "review it",
        runtime_env={
            "AIDER_BIN": "/opt/aider/bin/aider",
            "OPENROUTER_API_KEY": "cloud-key",
            "OPENAI_API_BASE": "http://127.0.0.1:11434/v1",
            "KYBERM0NK_ENV": str(tmp_path / "missing.env"),
        },
    )

    assert invocation.command[:3] == [
        "/opt/aider/bin/aider",
        "--model",
        "openrouter/deepseek/deepseek-v4-flash",
    ]
    assert "--cache-prompts" in invocation.command
    assert "--no-auto-commits" in invocation.command
    assert "OPENAI_API_BASE" not in invocation.env
    assert invocation.env["OPENROUTER_API_KEY"] == "cloud-key"
    assert invocation.env["OPENAI_API_KEY"] == "cloud-key"


def test_tier2_reviewer_invocation_targets_guardian(tmp_path):
    """Tier2 reviewer should use the local Guardian route, not OpenRouter."""
    invocation = build_aider_invocation(
        AiderRole.CLOUD_REVIEWER,
        Path(tmp_path),
        "review it",
        runtime_env={
            "AIDER_BIN": "/opt/aider/bin/aider",
            "AIDER_GUARDIAN_API_KEY": "local-key",
            "GUARDIAN_BASE_URL": "http://host.docker.internal:11434/v1",
            "KYBERM0NK_ENV": str(tmp_path / "missing.env"),
        },
        reviewer_tier=next(tier for tier in REVIEWER_TIERS if tier.name == "tier2"),
    )

    assert invocation.command[:3] == [
        "/opt/aider/bin/aider",
        "--model",
        "openai/gemma4-26b-a4b",
    ]
    assert invocation.env["OPENAI_API_BASE"] == "http://127.0.0.1:11434/v1"
    assert invocation.env["OPENAI_API_KEY"] == "local-key"


def test_tier1_review_fix_coder_invocation_targets_openrouter(tmp_path):
    """Review-fix coder Tier1 should use the cheap OpenRouter route."""
    invocation = build_aider_invocation(
        AiderRole.LOCAL_CODER,
        Path(tmp_path),
        "fix review findings",
        runtime_env={
            "AIDER_BIN": "/opt/aider/bin/aider",
            "OPENROUTER_API_KEY": "cloud-key",
            "KYBERM0NK_ENV": str(tmp_path / "missing.env"),
        },
        coder_tier=CODER_TIERS[0],
    )

    assert invocation.command[:3] == [
        "/opt/aider/bin/aider",
        "--model",
        "openrouter/deepseek/deepseek-v4-flash",
    ]
    assert "OPENAI_API_BASE" not in invocation.env
    assert invocation.env["OPENROUTER_API_KEY"] == "cloud-key"
    assert invocation.env["OPENAI_API_KEY"] == "cloud-key"


@pytest.mark.asyncio
async def test_pr_review_metadata_loader_counts_current_head_copilot_comments(monkeypatch):
    """Live PR-manager metadata should include only current-head Copilot suggestions."""
    pr = PullRequestMetadata(
        number=325,
        url="https://github.com/m0nklabs/cryptotrader/pull/325",
        head_ref_name="hermes/fix",
        head_ref_oid="abc123",
    )

    async def fake_run(command, *, cwd=None, env=None, check=True):
        if command == ["gh", "api", "repos/m0nklabs/cryptotrader/pulls/325/reviews"]:
            return CompletedProcess(command, 0, stdout="[]", stderr="")
        if command == ["gh", "api", "repos/m0nklabs/cryptotrader/pulls/325/comments"]:
            return CompletedProcess(
                command,
                0,
                stdout=json.dumps(
                    [
                        {
                            "user": {"login": "github-code-quality[bot]"},
                            "commit_id": "abc123",
                            "path": "api/routes/execution.py",
                            "line": 42,
                            "body": "This DB execution path can fail.",
                        },
                        {
                            "user": {"login": "github-code-quality[bot]"},
                            "commit_id": "oldsha",
                            "path": "api/routes/execution.py",
                            "line": 41,
                            "body": "Stale finding.",
                        },
                    ]
                ),
                stderr="",
            )
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr("gateway.issue_resolution._run", fake_run)

    stats = await _load_pr_review_suggestion_stats("m0nklabs/cryptotrader", pr)

    assert stats.copilot_review_detected is True
    assert stats.copilot_suggestions_count == 1
    assert stats.total_suggestions_count == 1
