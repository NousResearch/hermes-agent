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
    IssueRunStatus,
    IssueRunType,
    PullRequestMetadata,
    ReviewFindingsRetry,
    ReviewLoopCircuitBreaker,
    ReviewTagParseError,
    IssueStateStore,
    _execute_master_issue,
    _execute_single_issue,
    _guard_managed_repo_before_issue_dispatch,
    _inspect_managed_repo,
    _find_existing_sub_issue,
    _issue_branch_name,
    _load_next_open_issue,
    allowed_issue_repos,
    _pr_body,
    build_aider_invocation,
    can_merge_pr,
    is_review_findings_for_coder,
    parse_review_routing_tag,
    github_issue_webhook_command,
    is_master_issue,
    parse_decomposition_response,
    parse_issue_command_args,
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
        f"m0nklabs/cryptotrader #10 --workdir {tmp_path} --branch issue/10-test"
    )

    assert request.repo == "m0nklabs/cryptotrader"
    assert request.issue_number == 10
    assert request.workdir == tmp_path
    assert request.branch == "issue/10-test"


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
    monkeypatch.setattr("gateway.issue_resolution.IssueStateStore", IssueStateStore)

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
        "Direct implementation drift on protected CryptoTrader `master`/`main` is forbidden"
        in body
    )
    assert "## Review handoff" in body
    assert "State: `ready_for_review`" in body
    assert "Reviewer lane: `cloud_reviewer`" in body


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
        if command == ["git", "branch", "--show-current"]:
            return CompletedProcess(command, 0, stdout="feature\n", stderr="")
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
        if command == ["git", "branch", "--show-current"]:
            return CompletedProcess(command, 0, stdout="feature\n", stderr="")
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
async def test_execute_single_issue_retries_malformed_review_tag_once(
    tmp_path, monkeypatch
):
    """Malformed review tags should get one bounded reviewer rerun."""
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
        if command == ["git", "branch", "--show-current"]:
            return CompletedProcess(command, 0, stdout="feature\n", stderr="")
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
        if command[:3] == ["gh", "issue", "comment"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["gh", "pr", "comment"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
        if command[:3] == ["gh", "pr", "review"]:
            return CompletedProcess(command, 0, stdout="", stderr="")
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

    assert review_calls == 2
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
        if command == ["git", "branch", "--show-current"]:
            return CompletedProcess(command, 0, stdout="feature\n", stderr="")
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

    assert review_calls == 2


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
        IssueResolutionRequest("m0nklabs/cryptotrader", 1, tmp_path),
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
