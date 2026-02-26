"""Tests for agent.tool_executor -- extracted execute_tool_calls function.

Tests the module-level execute_tool_calls(), ToolExecConfig dataclass,
and the AGENT_LOOP_TOOLS frozenset.

Run with:
    python -m pytest tests/agent/test_tool_executor.py -v
"""

import dataclasses
import json
import time
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Mock helpers -- lightweight stand-ins for OpenAI tool_call objects
# ---------------------------------------------------------------------------

class MockFunction:
    def __init__(self, name, arguments="{}"):
        self.name = name
        self.arguments = arguments


class MockToolCall:
    def __init__(self, id, name, arguments="{}"):
        self.id = id
        self.function = MockFunction(name, arguments)


class MockAssistantMessage:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides):
    """Build a ToolExecConfig with sensible mock defaults."""
    from agent.tool_executor import ToolExecConfig

    defaults = dict(
        todo_store=MagicMock(),
        memory_store=MagicMock(),
        session_db=MagicMock(),
        clarify_callback=MagicMock(),
        tool_delay=0.0,
        tool_progress_callback=None,
        quiet_mode=True,
        verbose_logging=False,
        log_prefix="",
        log_prefix_chars=100,
    )
    defaults.update(overrides)
    return ToolExecConfig(**defaults)


def _run(tool_calls, *, config=None, is_interrupted=None,
         parent_agent=None, messages=None):
    """Convenience wrapper around execute_tool_calls with mocked callbacks."""
    from agent.tool_executor import execute_tool_calls

    if config is None:
        config = _make_config()
    if messages is None:
        messages = []
    if is_interrupted is None:
        is_interrupted = lambda: False

    log_mock = MagicMock()
    on_executed_mock = MagicMock()

    assistant_msg = MockAssistantMessage(tool_calls)
    execute_tool_calls(
        config,
        assistant_msg,
        messages,
        "test-task-id",
        is_interrupted=is_interrupted,
        log_msg_to_db=log_mock,
        on_tool_executed=on_executed_mock,
        parent_agent=parent_agent,
    )
    return messages, log_mock, on_executed_mock


# ===========================================================================
# Tests
# ===========================================================================


class TestAgentLoopToolsCanonicalSet:
    """AGENT_LOOP_TOOLS must be exactly the 5 known agent-loop tools."""

    def test_agent_loop_tools_canonical_set(self):
        from agent.tool_executor import AGENT_LOOP_TOOLS

        expected = {"todo", "memory", "session_search", "clarify", "delegate_task"}
        assert AGENT_LOOP_TOOLS == expected

    def test_agent_loop_tools_is_frozenset(self):
        from agent.tool_executor import AGENT_LOOP_TOOLS

        assert isinstance(AGENT_LOOP_TOOLS, frozenset)


class TestAgentLoopToolRouting:
    """Agent-loop tools must bypass handle_function_call."""

    @pytest.mark.parametrize("tool_name", [
        "todo", "memory", "session_search", "clarify", "delegate_task",
    ])
    @patch("model_tools.handle_function_call")
    def test_agent_loop_tools_not_through_registry(self, mock_hfc, tool_name):
        """Each agent-loop tool must NOT go through handle_function_call."""
        # Mock the underlying tool functions so they don't touch real stores
        with patch("tools.todo_tool.todo_tool", return_value='{"ok": true}'), \
             patch("tools.memory_tool.memory_tool", return_value='{"ok": true}'), \
             patch("tools.session_search_tool.session_search", return_value='{"ok": true}'), \
             patch("tools.clarify_tool.clarify_tool", return_value='{"ok": true}'):
            tc = MockToolCall(f"tc_{tool_name}", tool_name)
            messages, _, _ = _run([tc])

        mock_hfc.assert_not_called()
        # Must still produce exactly one tool response
        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert messages[0]["tool_call_id"] == f"tc_{tool_name}"

    @patch("model_tools.handle_function_call", return_value='{"ok": true}')
    def test_other_tools_through_registry(self, mock_hfc):
        """Non-agent-loop tools must go through handle_function_call."""
        tc = MockToolCall("tc_web", "web_search", '{"query": "test"}')
        messages, _, _ = _run([tc])

        mock_hfc.assert_called_once()
        assert messages[0]["content"] == '{"ok": true}'


class TestResponseCompleteness:
    """Every tool_call must produce exactly one tool response message."""

    @patch("model_tools.handle_function_call", return_value='{"ok": true}')
    def test_every_tool_call_gets_response(self, mock_hfc):
        with patch("tools.todo_tool.todo_tool", return_value='{"ok": true}'), \
             patch("tools.memory_tool.memory_tool", return_value='{"ok": true}'):
            tool_calls = [
                MockToolCall("tc_1", "todo"),
                MockToolCall("tc_2", "web_search"),
                MockToolCall("tc_3", "memory"),
            ]
            messages, log_mock, _ = _run(tool_calls)

        assert len(messages) == 3
        ids = [m["tool_call_id"] for m in messages]
        assert ids == ["tc_1", "tc_2", "tc_3"]
        for m in messages:
            assert m["role"] == "tool"


class TestInterruptBehavior:
    """Interrupt flag must cause remaining tools to be skipped."""

    def test_interrupt_before_first_skips_all(self):
        """When interrupted before the first tool, all get cancel messages."""
        tool_calls = [
            MockToolCall("tc_1", "terminal", '{"command": "ls"}'),
            MockToolCall("tc_2", "web_search", '{"query": "x"}'),
            MockToolCall("tc_3", "todo"),
        ]
        messages, log_mock, on_exec = _run(
            tool_calls, is_interrupted=lambda: True
        )

        assert len(messages) == 3
        for m in messages:
            assert m["role"] == "tool"
            assert "cancel" in m["content"].lower()

        # log_msg_to_db called for each skip message
        assert log_mock.call_count == 3

    @patch("model_tools.handle_function_call", return_value='{"ok": true}')
    def test_interrupt_after_nth_skips_remaining(self, mock_hfc):
        """After the Nth tool completes, remaining are skipped if interrupted."""
        call_count = {"n": 0}

        def interrupted_after_first():
            return call_count["n"] >= 1

        original_run = _run

        # We need to intercept to count how many tools have executed.
        # Use a wrapper that patches handle_function_call to track calls.
        from agent.tool_executor import execute_tool_calls, ToolExecConfig

        config = _make_config()
        messages = []
        log_mock = MagicMock()
        on_exec = MagicMock()

        # Track execution via on_tool_executed
        def track_executed(name):
            call_count["n"] += 1

        tool_calls = [
            MockToolCall("tc_1", "web_search", '{"query": "a"}'),
            MockToolCall("tc_2", "web_search", '{"query": "b"}'),
            MockToolCall("tc_3", "web_search", '{"query": "c"}'),
        ]

        assistant_msg = MockAssistantMessage(tool_calls)
        execute_tool_calls(
            config,
            assistant_msg,
            messages,
            "test-task-id",
            is_interrupted=interrupted_after_first,
            log_msg_to_db=log_mock,
            on_tool_executed=track_executed,
        )

        # First tool executes, then interrupt triggers -> 2 skipped
        assert len(messages) == 3
        # First message is a real result
        assert "cancel" not in messages[0]["content"].lower()
        assert "skip" not in messages[0]["content"].lower()
        # Remaining are skipped
        for m in messages[1:]:
            content_lower = m["content"].lower()
            assert "cancel" in content_lower or "skip" in content_lower


class TestResultTruncation:
    """Results exceeding MAX_TOOL_RESULT_CHARS must be truncated."""

    @patch("model_tools.handle_function_call")
    def test_result_truncation(self, mock_hfc):
        from agent.tool_executor import MAX_TOOL_RESULT_CHARS

        huge_result = "x" * (MAX_TOOL_RESULT_CHARS + 5000)
        mock_hfc.return_value = huge_result

        tc = MockToolCall("tc_1", "web_search", '{"query": "big"}')
        messages, _, _ = _run([tc])

        content = messages[0]["content"]
        # Must be truncated
        assert len(content) < len(huge_result)
        assert "Truncated" in content
        # The first MAX_TOOL_RESULT_CHARS chars should be preserved
        assert content[:MAX_TOOL_RESULT_CHARS] == "x" * MAX_TOOL_RESULT_CHARS


class TestInvalidJsonFallback:
    """Invalid JSON in tool_call arguments should fallback to {}."""

    @patch("model_tools.handle_function_call", return_value='{"ok": true}')
    def test_invalid_json_args_fallback(self, mock_hfc):
        tc = MockToolCall("tc_bad", "web_search", "NOT VALID JSON {{{")
        messages, _, _ = _run([tc])

        # Should still execute (with empty args)
        assert len(messages) == 1
        # handle_function_call should have been called with {} as fallback
        call_args = mock_hfc.call_args
        assert call_args[0][1] == {}  # second positional arg = function_args


class TestToolDelay:
    """tool_delay must be applied between calls but not after the last."""

    @patch("model_tools.handle_function_call", return_value='{"ok": true}')
    @patch("agent.tool_executor.time.sleep")
    def test_tool_delay_between_calls(self, mock_sleep, mock_hfc):
        # Use quiet_mode=False to avoid KawaiiSpinner (which also sleeps)
        config = _make_config(tool_delay=0.5, quiet_mode=False)
        tool_calls = [
            MockToolCall("tc_1", "web_search", '{"query": "a"}'),
            MockToolCall("tc_2", "web_search", '{"query": "b"}'),
            MockToolCall("tc_3", "web_search", '{"query": "c"}'),
        ]
        _run(tool_calls, config=config)

        # Sleep called between calls: 2 times for 3 calls (not after last)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(0.5)


class TestCallbacks:
    """Verify log_msg_to_db and on_tool_executed callbacks."""

    @patch("model_tools.handle_function_call", return_value='{"ok": true}')
    def test_log_msg_callback_called(self, mock_hfc):
        with patch("tools.todo_tool.todo_tool", return_value='{"ok": true}'):
            tool_calls = [
                MockToolCall("tc_1", "todo"),
                MockToolCall("tc_2", "web_search"),
            ]
            messages, log_mock, _ = _run(tool_calls)

        # log_msg_to_db called once per appended message
        assert log_mock.call_count == 2
        for c in log_mock.call_args_list:
            msg = c[0][0]
            assert msg["role"] == "tool"

    @patch("model_tools.handle_function_call", return_value='{"ok": true}')
    def test_on_tool_executed_callback(self, mock_hfc):
        with patch("tools.todo_tool.todo_tool", return_value='{"ok": true}'):
            tool_calls = [
                MockToolCall("tc_1", "todo"),
                MockToolCall("tc_2", "web_search"),
            ]
            _, _, on_exec = _run(tool_calls)

        assert on_exec.call_count == 2
        on_exec.assert_any_call("todo")
        on_exec.assert_any_call("web_search")

    def test_on_tool_executed_not_called_for_skipped(self):
        """on_tool_executed must NOT be called for cancelled/skipped tools."""
        tool_calls = [
            MockToolCall("tc_1", "terminal", '{"command": "ls"}'),
            MockToolCall("tc_2", "web_search", '{"query": "x"}'),
        ]
        _, _, on_exec = _run(tool_calls, is_interrupted=lambda: True)

        on_exec.assert_not_called()


class TestDelegateTaskGuard:
    """delegate_task with parent_agent=None must return error JSON."""

    def test_delegate_task_none_guard(self):
        tc = MockToolCall("tc_del", "delegate_task", '{"goal": "do stuff"}')
        messages, _, _ = _run([tc], parent_agent=None)

        assert len(messages) == 1
        content = messages[0]["content"]
        parsed = json.loads(content)
        assert "error" in parsed


class TestToolExecConfigFrozen:
    """ToolExecConfig must be frozen (immutable)."""

    def test_config_is_frozen(self):
        config = _make_config()
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.tool_delay = 999.0
