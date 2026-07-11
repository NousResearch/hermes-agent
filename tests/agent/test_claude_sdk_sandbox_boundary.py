import asyncio

import pytest

from agent.claude_sdk_session import build_claude_agent_options


def _tool(name):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _invoke(hook, tool_name, tool_input):
    return asyncio.run(
        hook(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": tool_name,
                "tool_input": tool_input,
                "session_id": "session",
                "transcript_path": "transcript",
                "cwd": "/workspace",
                "agent_id": "agent",
                "agent_type": "main",
                "tool_use_id": "tool",
            },
            "tool-id",
            {"signal": None},
        )
    )


def test_real_sdk_options_hook_blocks_host_and_bash_but_allows_worktree(tmp_path):
    sdk = pytest.importorskip("claude_agent_sdk")
    host_home = tmp_path / "host"
    workspace = host_home / "worktree"
    workspace.mkdir(parents=True)
    (host_home / "sentinel").write_text("secret", encoding="utf-8")
    (workspace / "allowed.txt").write_text("ok", encoding="utf-8")
    options = build_claude_agent_options(
        sdk=sdk,
        model="claude-sonnet-4-6",
        system_prompt="prompt",
        workspace=workspace,
        host_home=host_home,
        profile_home=tmp_path / "profile",
        inherited_env={"PATH": "/usr/bin", "USER": "worker", "LOGNAME": "worker"},
        tool_definitions=[_tool("kanban_complete"), _tool("read_file")],
        dispatch=lambda *args, **kwargs: "ok",
        effective_task_id="run",
        kanban_task_id="BUILD-392",
        cli_path=tmp_path / "exact-env-wrapper",
    )
    matcher = options.hooks["PreToolUse"][0]
    assert isinstance(matcher, sdk.HookMatcher)
    hook = matcher.hooks[0]

    allowed = _invoke(hook, "mcp__hermes__read_file", {"path": "allowed.txt"})
    host_read = _invoke(hook, "Read", {"file_path": str(host_home / "sentinel")})
    host_write = _invoke(hook, "Write", {"file_path": str(host_home / "new")})
    bash = _invoke(hook, "Bash", {"command": "cat $HOME/sentinel"})

    assert allowed["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert host_read["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert host_write["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert bash["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "Bash" not in options.tools
    assert options.sandbox is None
    assert not {"Read", "Glob", "Grep", "Edit", "Write"}.intersection(options.tools)
