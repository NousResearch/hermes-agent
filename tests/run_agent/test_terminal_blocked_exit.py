"""Tests for terminal command denied early-exit fix — issue #24806.

When a user denies a terminal command (approval dialog returns
status: "blocked"), the agent loop should exit early instead of
continuing to process remaining tool calls.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list:
    """Build minimal tool definition list accepted by AIAgent.__init__."""
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _make_agent():
    """Minimal AIAgent for testing — mocks OpenAI and tool loading."""
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("terminal", "web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        return agent


def _mock_tool_call(name: str, arguments: str, call_id: str):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_assistant_msg(content: str = "", tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls or [])


BLOCKED_RESULT = '{"output": "", "exit_code": -1, "error": "BLOCKED: Command denied by user.", "status": "blocked"}'
NORMAL_RESULT = '{"output": "/home/user", "exit_code": 0, "status": "completed"}'


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_agent_halt_on_blocked_terminal_sets_decision():
    """Calling _agent_halt_on_blocked_terminal() sets _blocked_terminal_halt_decision."""
    agent = _make_agent()
    assert agent._blocked_terminal_halt_decision is None

    agent._agent_halt_on_blocked_terminal()

    assert agent._blocked_terminal_halt_decision is not None
    assert agent._blocked_terminal_halt_decision.tool_name == "terminal"
    assert agent._blocked_terminal_halt_decision.action == "halt"
    assert agent._blocked_terminal_halt_decision.code == "terminal_denied"
    assert agent._blocked_terminal_halt_decision.should_halt is True


def test_agent_halt_on_blocked_terminal_idempotent():
    """Second call is idempotent — same decision object is reused."""
    agent = _make_agent()
    agent._agent_halt_on_blocked_terminal()
    first = agent._blocked_terminal_halt_decision

    agent._agent_halt_on_blocked_terminal()
    assert agent._blocked_terminal_halt_decision is first


def test_blocked_terminal_halt_decision_reset_between_turns():
    """_blocked_terminal_halt_decision is cleared at the start of each turn."""
    agent = _make_agent()
    agent._agent_halt_on_blocked_terminal()
    assert agent._blocked_terminal_halt_decision is not None

    # Simulate new turn: both guardrail decisions are reset
    agent._tool_guardrails.reset_for_turn()
    agent._tool_guardrail_halt_decision = None
    agent._blocked_terminal_halt_decision = None

    assert agent._blocked_terminal_halt_decision is None


def test_execute_tool_calls_sequential_exits_early_on_blocked_terminal():
    """A terminal result with status: blocked triggers early break and halt."""
    agent = _make_agent()

    tc = _mock_tool_call(name="terminal", arguments='{"command":"rm -rf /"}', call_id="c1")
    mock_msg = _mock_assistant_msg(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value=BLOCKED_RESULT):
        with patch.object(agent, "_vprint"):
            agent._execute_tool_calls_sequential(mock_msg, messages, "task-1")

    assert agent._blocked_terminal_halt_decision is not None
    assert agent._blocked_terminal_halt_decision.action == "halt"
    assert agent._blocked_terminal_halt_decision.code == "terminal_denied"


def test_execute_tool_calls_sequential_continues_on_non_blocked_terminal():
    """A terminal result without status: blocked does NOT trigger early exit."""
    agent = _make_agent()

    tc = _mock_tool_call(name="terminal", arguments='{"command":"pwd"}', call_id="c1")
    mock_msg = _mock_assistant_msg(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value=NORMAL_RESULT):
        agent._execute_tool_calls_sequential(mock_msg, messages, "task-1")

    assert agent._blocked_terminal_halt_decision is None


def test_execute_tool_calls_sequential_ignores_non_terminal_blocked():
    """status: blocked from non-terminal tool does NOT trigger early exit."""
    agent = _make_agent()
    blocked_non_terminal = '{"output": "", "error": "BLOCKED", "status": "blocked"}'

    tc = _mock_tool_call(name="web_search", arguments='{"query":"test"}', call_id="c1")
    mock_msg = _mock_assistant_msg(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value=blocked_non_terminal):
        agent._execute_tool_calls_sequential(mock_msg, messages, "task-1")

    assert agent._blocked_terminal_halt_decision is None


def test_execute_tool_calls_sequential_breaks_after_blocked_without_running_subsequent():
    """When terminal is blocked, subsequent tool calls in the same turn are skipped."""
    agent = _make_agent()
    call_log = []

    def fake_handle(function_name, function_args, task_id=None, tool_call_id=None, **kwargs):
        call_log.append(function_name)
        return BLOCKED_RESULT if len(call_log) == 1 else NORMAL_RESULT

    tc1 = _mock_tool_call(name="terminal", arguments='{"command":"rm -rf /"}', call_id="c1")
    tc2 = _mock_tool_call(name="terminal", arguments='{"command":"echo done"}', call_id="c2")
    mock_msg = _mock_assistant_msg(content="", tool_calls=[tc1, tc2])
    messages = []

    with patch("run_agent.handle_function_call", side_effect=fake_handle):
        with patch.object(agent, "_vprint"):
            agent._execute_tool_calls_sequential(mock_msg, messages, "task-1")

    # Only the first (blocked) call should have been made — the second is skipped
    assert call_log == ["terminal"]
    assert agent._blocked_terminal_halt_decision is not None


def test_execute_tool_calls_sequential_blocked_result_is_added_to_messages():
    """The blocked result is appended to messages before the loop breaks."""
    agent = _make_agent()

    tc = _mock_tool_call(name="terminal", arguments='{"command":"rm -rf /"}', call_id="c1")
    mock_msg = _mock_assistant_msg(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value=BLOCKED_RESULT):
        with patch.object(agent, "_vprint"):
            agent._execute_tool_calls_sequential(mock_msg, messages, "task-1")

    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "c1"
    assert '"status": "blocked"' in messages[0]["content"]


def test_vprint_called_on_blocked_terminal():
    """_vprint is invoked with the blocked terminal message when denied."""
    agent = _make_agent()
    vprint_args = []

    tc = _mock_tool_call(name="terminal", arguments='{"command":"rm -rf /"}', call_id="c1")
    mock_msg = _mock_assistant_msg(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value=BLOCKED_RESULT):
        with patch.object(agent, "_vprint", side_effect=lambda *a, **kw: vprint_args.append(a)):
            agent._execute_tool_calls_sequential(mock_msg, messages, "task-1")

    assert len(vprint_args) == 1
    combined = " ".join(str(a) for a in vprint_args[0])
    assert "denied" in combined.lower() or "blocked" in combined.lower()
