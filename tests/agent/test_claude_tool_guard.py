import asyncio
from pathlib import Path

import pytest

from agent.claude_tool_guard import create_workspace_pre_tool_hook


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


def _decision(output):
    return output["hookSpecificOutput"]["permissionDecision"]


@pytest.mark.parametrize(
    ("tool_name", "tool_input"),
    [
        ("Read", {"file_path": "README.md"}),
        ("Edit", {"file_path": "src/app.py"}),
        ("Write", {"file_path": "src/new.py"}),
        ("Glob", {"path": ".", "pattern": "**/*.py"}),
        ("Grep", {"path": "src", "pattern": "needle"}),
        ("mcp__hermes__kanban_complete", {"summary": "done"}),
    ],
)
def test_supported_workspace_tools_are_allowed(tmp_path, tool_name, tool_input):
    workspace = tmp_path / "work"
    (workspace / "src").mkdir(parents=True)
    (workspace / "README.md").write_text("readme", encoding="utf-8")
    hook = create_workspace_pre_tool_hook(workspace)

    assert _decision(_invoke(hook, tool_name, tool_input)) == "allow"


@pytest.mark.parametrize(
    ("tool_name", "tool_input"),
    [
        ("Read", {"file_path": "/etc/passwd"}),
        ("Read", {"file_path": "../host-sentinel"}),
        ("Edit", {"file_path": "/tmp/outside"}),
        ("Write", {"file_path": "../../outside"}),
        ("Glob", {"path": "/", "pattern": "**/*"}),
        ("Glob", {"path": ".", "pattern": "../*"}),
        ("Grep", {"path": "..", "pattern": "secret"}),
        ("Bash", {"command": "cat $HOME/host-sentinel"}),
        ("NotebookEdit", {"notebook_path": "note.ipynb"}),
        ("mcp__evil__read", {}),
    ],
)
def test_host_capable_or_unknown_tools_are_denied(tmp_path, tool_name, tool_input):
    workspace = tmp_path / "work"
    workspace.mkdir()
    hook = create_workspace_pre_tool_hook(workspace)

    output = _invoke(hook, tool_name, tool_input)

    assert _decision(output) == "deny"
    assert output["hookSpecificOutput"]["permissionDecisionReason"]


def test_symlink_escape_is_denied(tmp_path):
    workspace = tmp_path / "work"
    outside = tmp_path / "host-home"
    workspace.mkdir()
    outside.mkdir()
    (workspace / "escape").symlink_to(outside, target_is_directory=True)
    hook = create_workspace_pre_tool_hook(workspace)

    assert _decision(_invoke(hook, "Read", {"file_path": "escape/sentinel"})) == "deny"
