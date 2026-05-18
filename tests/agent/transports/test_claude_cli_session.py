"""Tests for the Claude Code CLI turn adapter."""

from __future__ import annotations

import json
import subprocess

from agent.transports.claude_cli_session import ClaudeCliSession


class _Completed:
    returncode = 0
    stdout = "OK\n"
    stderr = ""


def test_kanban_worker_invocation_wires_hermes_mcp_tools(monkeypatch, tmp_path):
    seen: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["kwargs"] = kwargs
        return _Completed()

    monkeypatch.setenv("HERMES_HOME", "/tmp/hermes-profile")
    monkeypatch.setenv("HERMES_PROFILE", "coding-agent")
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_12345678")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "ai-stack")
    monkeypatch.setenv("HERMES_KANBAN_DB", "/tmp/kanban.db")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", str(tmp_path))
    monkeypatch.setattr(subprocess, "run", fake_run)

    session = ClaudeCliSession(
        cwd=str(tmp_path),
        claude_bin="claude",
        model="sonnet",
    )

    result = session.run_turn(messages=[], user_input="work kanban task")

    assert result.final_text == "OK"
    cmd = seen["cmd"]
    assert isinstance(cmd, list)
    assert "--mcp-config" in cmd
    mcp_config = json.loads(cmd[cmd.index("--mcp-config") + 1])
    server = mcp_config["mcpServers"]["hermes-tools"]
    assert server["args"] == ["-m", "agent.transports.hermes_tools_mcp_server"]
    assert server["env"]["HERMES_KANBAN_TASK"] == "t_12345678"
    assert server["env"]["HERMES_KANBAN_BOARD"] == "ai-stack"
    assert server["env"]["HERMES_PROFILE"] == "coding-agent"

    assert "--allowedTools" in cmd
    allowed_tools = cmd[cmd.index("--allowedTools") + 1]
    assert "mcp__hermes-tools__kanban_show" in allowed_tools
    assert "mcp__hermes-tools__kanban_complete" in allowed_tools
    assert "mcp__hermes-tools__kanban_block" in allowed_tools
