"""Tests for the barebones GitHub issue resolution lane."""

from pathlib import Path

import pytest

from gateway.issue_resolution import (
    AiderRole,
    EpicTask,
    IssueMetadata,
    IssueResolutionRequest,
    IssueRunStatus,
    IssueRunType,
    IssueStateStore,
    _execute_master_issue,
    build_aider_invocation,
    github_issue_webhook_command,
    is_master_issue,
    parse_decomposition_response,
    parse_issue_command_args,
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

    assert github_issue_webhook_command(payload) == "/issue --repo m0nklabs/cryptotrader --issue 5"


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

    monkeypatch.setattr("gateway.issue_resolution.decompose_master_plan", fake_decompose)
    monkeypatch.setattr("gateway.issue_resolution._create_sub_issue", fake_create_sub_issue)

    await _execute_master_issue(store, run, issue, notify)

    master = store.get_run(run.id)
    children = store.list_child_runs(run.id)

    assert master.status is IssueRunStatus.EXPANDED
    assert [child.issue_number for child in children] == [101, 102]
    assert [child.status for child in children] == [IssueRunStatus.QUEUED, IssueRunStatus.QUEUED]
    assert [child.run_type for child in children] == [IssueRunType.SUB_ISSUE, IssueRunType.SUB_ISSUE]
    assert any("expanded into 2" in message for message in messages)


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
    assert invocation.command[:4] == ["/opt/aider/bin/aider", "--model", "openai/qwen3-35b-uncensored", "--yes"]
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

    assert invocation.command[:3] == ["/opt/aider/bin/aider", "--model", "openrouter/deepseek/deepseek-v4-flash"]
    assert "--cache-prompts" in invocation.command
    assert "--no-auto-commits" in invocation.command
    assert "OPENAI_API_BASE" not in invocation.env
    assert invocation.env["OPENROUTER_API_KEY"] == "cloud-key"
    assert invocation.env["OPENAI_API_KEY"] == "cloud-key"