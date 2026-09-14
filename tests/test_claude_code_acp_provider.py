"""Regression tests for the Claude Code ACP provider wiring."""

from __future__ import annotations

from agent.copilot_acp_client import CopilotACPClient
from hermes_cli.auth import (
    get_auth_status,
    get_external_process_provider_status,
    resolve_external_process_provider_credentials,
)


def test_claude_code_acp_client_uses_claude_agent_command(monkeypatch):
    monkeypatch.delenv("HERMES_CLAUDE_CODE_ACP_COMMAND", raising=False)
    monkeypatch.delenv("HERMES_CLAUDE_CODE_ACP_ARGS", raising=False)

    client = CopilotACPClient(base_url="acp://claude-code")

    assert client._acp_command == "claude-agent-acp"
    assert client._acp_args == []


def test_claude_code_acp_uses_explicit_command_and_empty_args(monkeypatch):
    monkeypatch.setenv("HERMES_CLAUDE_CODE_ACP_COMMAND", "/tmp/claude-agent-acp")
    monkeypatch.delenv("HERMES_CLAUDE_CODE_ACP_ARGS", raising=False)
    monkeypatch.setattr("hermes_cli.auth.shutil.which", lambda command: command)

    creds = resolve_external_process_provider_credentials("claude-code-acp")

    assert creds["provider"] == "claude-code-acp"
    assert creds["base_url"] == "acp://claude-code"
    assert creds["command"] == "/tmp/claude-agent-acp"
    assert creds["args"] == []
    assert creds["api_key"] == "claude-code-acp"


def test_claude_code_acp_args_are_shell_split(monkeypatch):
    monkeypatch.setenv("HERMES_CLAUDE_CODE_ACP_COMMAND", "/tmp/claude-agent-acp")
    monkeypatch.setenv("HERMES_CLAUDE_CODE_ACP_ARGS", "--flag 'value with spaces'")
    monkeypatch.setattr("hermes_cli.auth.shutil.which", lambda command: command)

    creds = resolve_external_process_provider_credentials("claude-code-acp")

    assert creds["args"] == ["--flag", "value with spaces"]


def test_claude_code_acp_status_uses_claude_agent_command(monkeypatch):
    monkeypatch.delenv("HERMES_CLAUDE_CODE_ACP_COMMAND", raising=False)
    monkeypatch.setattr("hermes_cli.auth.shutil.which", lambda command: "/opt/bin/claude-agent-acp")

    status = get_external_process_provider_status("claude-code-acp")

    assert status["configured"] is True
    assert status["logged_in"] is True
    assert status["command"] == "claude-agent-acp"
    assert status["args"] == []


def test_auth_status_dispatches_claude_code_acp(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.auth.get_external_process_provider_status",
        lambda provider: {"logged_in": True, "provider": provider},
    )

    assert get_auth_status("claude-code-acp") == {
        "logged_in": True,
        "provider": "claude-code-acp",
    }
