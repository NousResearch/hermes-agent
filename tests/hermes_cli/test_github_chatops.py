"""Tests for GitHub ChatOps parsing and orchestration bridge."""

from __future__ import annotations

import time

from hermes_cli.code.github_chatops import GitHubChatOpsService, parse_chatops_commands
from hermes_cli.code.github_integration import GitHubIntegrationStore
from hermes_state import SessionDB


def test_parse_supported_commands():
    text = "@hermes plan do this\n@hermes review\n@hermes fix now\n@hermes explain why\n@hermes status"
    parsed = parse_chatops_commands(text)
    assert [item.command for item in parsed] == ["plan", "review", "fix", "explain", "status"]


def test_non_hermes_comment_is_ignored():
    assert parse_chatops_commands("hello world") == []


def test_create_command_from_comment_and_run(tmp_path):
    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).close()
    service = GitHubChatOpsService(db_path=db_path)
    commands = service.create_commands_from_comment(
        delivery_id="d1",
        repo_full_name="acme/repo",
        issue_number=7,
        pr_number=None,
        comment_id=100,
        sender_login="alice",
        body="@hermes plan implement auth",
    )
    assert len(commands) == 1
    cmd = commands[0]
    result = service.run_command(cmd["id"])
    assert result["run"] is not None
    assert result["command"]["orchestrated_run_id"] is not None


def test_fix_command_moves_to_approval(tmp_path):
    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).close()
    service = GitHubChatOpsService(db_path=db_path)
    cmd = service.create_commands_from_comment(
        delivery_id="d2",
        repo_full_name="acme/repo",
        issue_number=8,
        pr_number=None,
        comment_id=101,
        sender_login="bob",
        body="@hermes fix failing tests",
    )[0]
    result = service.run_command(cmd["id"])
    assert result["run"]["state"] == "approval"


def test_status_links_existing_run_when_possible(tmp_path):
    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).close()
    service = GitHubChatOpsService(db_path=db_path)

    first = service.create_commands_from_comment(
        delivery_id="d3",
        repo_full_name="acme/repo",
        issue_number=9,
        pr_number=None,
        comment_id=102,
        sender_login="alice",
        body="@hermes plan do x",
    )[0]
    first_result = service.run_command(first["id"])
    run_id = first_result["run"]["id"]

    second = service.create_commands_from_comment(
        delivery_id="d4",
        repo_full_name="acme/repo",
        issue_number=9,
        pr_number=None,
        comment_id=103,
        sender_login="alice",
        body="@hermes status",
    )[0]
    second_result = service.run_command(second["id"])
    assert second_result["run"]["id"] == run_id
    assert second_result["resumed"] is True


def test_command_links_repo_issue_pr_sender_fields(tmp_path):
    db_path = tmp_path / "state.db"
    SessionDB(db_path=db_path).close()
    service = GitHubChatOpsService(db_path=db_path)
    created = service.create_commands_from_comment(
        delivery_id="d5",
        repo_full_name="acme/repo",
        issue_number=1,
        pr_number=2,
        comment_id=99,
        sender_login="charlie",
        body="@hermes review",
    )[0]
    assert created["repo_full_name"] == "acme/repo"
    assert created["issue_number"] == 1
    assert created["pr_number"] == 2
    assert created["comment_id"] == 99
    assert created["sender_login"] == "charlie"
