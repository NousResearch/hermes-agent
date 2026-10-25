"""Provider-bound terminal output regression for #50807."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from tests.agent.test_run_agent import (
    _mock_assistant_msg,
    _mock_tool_call,
    _mock_plugin_discovery as _mock_plugin_discovery,
    agent as agent,
)


def test_terminal_large_stdout_is_bounded_before_provider(agent):
    """Regression for #50807: terminal's 50K stdout cap must not be sent inline."""
    import tools.terminal_tool as terminal_tool_module

    mock_env = MagicMock()
    mock_env.execute.return_value = {
        "output": "x" * 100_000,
        "returncode": 0,
    }
    terminal_config = {
        "env_type": "local",
        "cwd": str(Path.cwd()),
        "timeout": 30,
    }
    with (
        patch(
            "tools.terminal_tool._get_env_config",
            return_value=terminal_config,
        ),
        patch("tools.terminal_tool._start_cleanup_thread"),
        patch(
            "tools.terminal_tool._active_environments",
            {"default": mock_env},
        ),
        patch("tools.terminal_tool._last_activity", {"default": 0}),
        patch(
            "tools.terminal_tool._check_all_guards",
            return_value={"approved": True},
        ),
    ):
        terminal_result = terminal_tool_module.terminal_tool(
            command="python3 -c \"print('x'*100000)\"",
            timeout=30,
        )

    tc = _mock_tool_call(
        name="terminal",
        arguments=json.dumps({
            "command": "python3 -c \"print('x'*100000)\"",
            "timeout": 30,
        }),
        call_id="huge_stdout_probe",
    )
    mock_msg = _mock_assistant_msg(content="", tool_calls=[tc])
    messages = []

    with patch("model_tools.handle_function_call", return_value=terminal_result):
        agent._execute_tool_calls_sequential(mock_msg, messages, "task-1")

    assert len(messages) == 1
    content = messages[0]["content"]
    assert len(content.encode("utf-8")) < 8_192
    assert "Truncated" in content or "<persisted-output>" in content
    assert "x" * 9_000 not in content
    # The exit status must survive truncation so the model still knows
    # whether the command passed (regression guard for the head-cut).
    assert '"exit_code"' in content
