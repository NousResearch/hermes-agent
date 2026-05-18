"""Integration test for the claude_cli runtime path through AIAgent."""

from __future__ import annotations

from unittest.mock import patch

import run_agent
from agent.transports.claude_cli_session import ClaudeCliSession, ClaudeCliTurnResult


def _make_claude_cli_agent():
    return run_agent.AIAgent(
        provider="claude-cli",
        api_mode="claude_cli",
        model="sonnet",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


def test_claude_cli_mode_does_not_require_provider_api_key(monkeypatch):
    def _unexpected_openai_client(**kwargs):
        raise AssertionError("claude_cli should not initialize an OpenAI client")

    monkeypatch.setattr(run_agent, "OpenAI", _unexpected_openai_client)

    agent = _make_claude_cli_agent()

    assert agent.api_mode == "claude_cli"
    assert agent.provider == "claude-cli"
    assert agent.api_key == ""
    assert agent.client is None


def test_run_conversation_returns_claude_cli_shape(monkeypatch):
    seen = {}

    def fake_run_turn(self, *, messages, user_input):
        seen["messages"] = messages
        seen["user_input"] = user_input
        return ClaudeCliTurnResult(final_text=f"cli: {user_input}")

    monkeypatch.setattr(ClaudeCliSession, "run_turn", fake_run_turn)

    agent = _make_claude_cli_agent()
    with patch.object(agent, "_spawn_background_review", return_value=None):
        result = agent.run_conversation("hello from max")

    assert result["final_response"] == "cli: hello from max"
    assert result["completed"] is True
    assert result["partial"] is False
    assert result["error"] is None
    assert result["api_calls"] == 1
    assert result["messages"][-1] == {
        "role": "assistant",
        "content": "cli: hello from max",
    }
    assert seen["user_input"] == "hello from max"
    assert any(m.get("role") == "system" for m in seen["messages"])
    assert any(m.get("role") == "user" and m.get("content") == "hello from max" for m in seen["messages"])


def test_kanban_worker_text_fallback_blocks_task(monkeypatch):
    """If Claude CLI returns prose without a terminal Kanban MCP call,
    Hermes should block the task itself rather than creating a dispatcher
    protocol violation."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_test_task_123")
    monkeypatch.setattr(
        run_agent.AIAgent,
        "_kanban_task_needs_terminal_transition",
        lambda self, task_id: True,
        raising=False,
    )

    def fake_run_turn(self, *, messages, user_input):
        return ClaudeCliTurnResult(final_text="I could not write files.")

    monkeypatch.setattr(ClaudeCliSession, "run_turn", fake_run_turn)

    agent = _make_claude_cli_agent()
    with (
        patch.object(agent, "_spawn_background_review", return_value=None),
        patch("run_agent.handle_function_call", return_value='{"ok": true}') as hfc,
    ):
        result = agent.run_conversation("work kanban task t_test_task_123")

    assert result["completed"] is False
    kanban_block_calls = [
        c for c in hfc.call_args_list if c.args and c.args[0] == "kanban_block"
    ]
    assert len(kanban_block_calls) == 1
    args = kanban_block_calls[0].args[1]
    assert args["task_id"] == "t_test_task_123"
    assert "external-runtime-prose" in args["reason"]
