"""Tests for steer breakout in tool execution — fixes #28172.

When /steer is consumed mid-batch, remaining tools should be deferred so the
model can process the steer before more tools run.
"""

import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubAgent:
    """Minimal stub that mimics AIAgent for tool_executor tests.
    Avoids MagicMock's implicit truthy behavior on every attribute access.
    """
    def __init__(self):
        self._pending_steer = None
        self._pending_steer_lock = None
        self._interrupt_requested = False
        self._interrupt_message = None
        self._current_tool = None
        self.quiet_mode = True
        self.verbose_logging = False
        self.log_prefix = ""
        self.log_prefix_chars = 200
        self.tool_delay = 0
        self.valid_tool_names = None
        self.session_id = "test-session"
        self.tool_progress_callback = None
        self.tool_start_callback = None
        self.tool_complete_callback = None
        self._checkpoint_mgr = MagicMock(enabled=False)
        self._subdirectory_hints = MagicMock()
        self._subdirectory_hints.check_tool_call.return_value = None
        self._tool_guardrails = MagicMock()
        self._tool_guardrails.before_call.return_value = MagicMock(allows_execution=True)
        self._memory_manager = None  # Not a MagicMock — no implicit truthy
        self._context_engine_tool_names = None
        self._print_fn = print
        self._steer_result_log = []  # Track steer apply calls
        self._tool_worker_threads = set()
        self._tool_worker_threads_lock = threading.Lock()
        self._active_children = []
        self._active_children_lock = threading.Lock()
        self._last_activity = 0

    def _touch_activity(self, desc):
        pass

    def _vprint(self, msg, force=False):
        pass

    def _should_emit_quiet_tool_messages(self):
        return False

    def _should_start_quiet_spinner(self):
        return False

    def _has_stream_consumers(self):
        return False

    def _append_guardrail_observation(self, name, args, result, failed=False):
        return result

    def _tool_result_content_for_active_model(self, name, result):
        return result

    def _record_file_mutation_result(self, *args, **kwargs):
        pass

    def _apply_pending_steer_to_tool_results(self, messages, n):
        """Default: drain steer if set, append to last tool message."""
        text = self._drain_pending_steer()
        if not text:
            self._steer_result_log.append(("apply", n, False))
            return False
        # Append steer text to the last tool message
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "tool":
                msg["content"] = msg.get("content", "") + "\n\nUser guidance: " + text
                break
        self._steer_result_log.append(("apply", n, True))
        return True

    def _drain_pending_steer(self):
        text = self._pending_steer
        self._pending_steer = None
        return text

    def _invoke_tool(self, *args, **kwargs):
        return '{"status": "ok"}'


class _FakeToolCall:
    def __init__(self, name, args="{}", call_id=None):
        self.function = MagicMock(name=name, arguments=args)
        self.function.name = name
        self.id = call_id or f"call_{name}"


class _FakeAssistantMsg:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls


def _count_tool_messages(messages):
    return [m for m in messages if isinstance(m, dict) and m.get("role") == "tool"]


# ---------------------------------------------------------------------------
# apply_pending_steer_to_tool_results returns bool
# ---------------------------------------------------------------------------

class TestSteerReturnBool:
    """Verify the helper returns True when steer is consumed."""

    def test_returns_false_when_no_steer(self):
        from agent.agent_runtime_helpers import apply_pending_steer_to_tool_results
        agent = _StubAgent()
        messages = [{"role": "tool", "content": "result", "tool_call_id": "t1"}]
        result = apply_pending_steer_to_tool_results(agent, messages, 1)
        assert result is False

    def test_returns_true_when_steer_consumed(self):
        from agent.agent_runtime_helpers import apply_pending_steer_to_tool_results
        agent = _StubAgent()
        agent._pending_steer = "use Python"
        messages = [{"role": "tool", "content": "result", "tool_call_id": "t1"}]
        result = apply_pending_steer_to_tool_results(agent, messages, 1)
        assert result is True
        assert "use Python" in messages[0]["content"]

    def test_returns_false_when_no_tool_messages(self):
        from agent.agent_runtime_helpers import apply_pending_steer_to_tool_results
        agent = _StubAgent()
        agent._pending_steer = "use Python"
        result = apply_pending_steer_to_tool_results(agent, [], 0)
        assert result is False

    def test_returns_false_empty_messages(self):
        from agent.agent_runtime_helpers import apply_pending_steer_to_tool_results
        agent = _StubAgent()
        result = apply_pending_steer_to_tool_results(agent, [], 1)
        assert result is False


# ---------------------------------------------------------------------------
# Sequential path: steer breakout
# ---------------------------------------------------------------------------

class TestSequentialSteerBreakout:
    """After /steer is consumed, remaining tools are deferred."""

    @patch("agent.tool_executor._ra")
    def test_sequential_steer_breaks_loop(self, mock_ra):
        """When steer consumed after tool 2 of 4, tools 3-4 are deferred."""
        from agent.tool_executor import execute_tool_calls_sequential

        agent = _StubAgent()
        call_count = [0]

        def fake_steer(messages, n):
            call_count[0] += 1
            if call_count[0] >= 2:
                # Inject steer text to simulate drain
                for msg in reversed(messages):
                    if isinstance(msg, dict) and msg.get("role") == "tool":
                        msg["content"] += "\n\nUser guidance: use Python"
                        return True
            return False

        agent._apply_pending_steer_to_tool_results = fake_steer
        mock_ra.return_value.handle_function_call.return_value = "ok"

        tool_calls = [_FakeToolCall(f"tool_{i}") for i in range(4)]
        assistant_msg = _FakeAssistantMsg(tool_calls)
        messages = []

        execute_tool_calls_sequential(agent, assistant_msg, messages, "task1")

        tool_msgs = _count_tool_messages(messages)
        assert len(tool_msgs) == 4

        # First two executed normally
        assert "ok" in tool_msgs[0]["content"]
        assert "ok" in tool_msgs[1]["content"]

        # Last two deferred due to steer
        assert "deferred" in tool_msgs[2]["content"].lower() or "steer" in tool_msgs[2]["content"].lower()
        assert "deferred" in tool_msgs[3]["content"].lower() or "steer" in tool_msgs[3]["content"].lower()

    @patch("agent.tool_executor._ra")
    def test_sequential_no_steer_all_tools_execute(self, mock_ra):
        """Without steer, all tools execute normally."""
        from agent.tool_executor import execute_tool_calls_sequential

        agent = _StubAgent()
        mock_ra.return_value.handle_function_call.return_value = "ok"

        tool_calls = [_FakeToolCall(f"tool_{i}") for i in range(3)]
        assistant_msg = _FakeAssistantMsg(tool_calls)
        messages = []

        execute_tool_calls_sequential(agent, assistant_msg, messages, "task1")

        tool_msgs = _count_tool_messages(messages)
        assert len(tool_msgs) == 3
        for msg in tool_msgs:
            assert "ok" in msg["content"]


# ---------------------------------------------------------------------------
# Concurrent path: steer cancels pending futures
# ---------------------------------------------------------------------------

class TestConcurrentSteerCancel:
    """When steer is pending during concurrent execution, unstarted futures are cancelled."""

    def test_concurrent_steer_pending_triggers_cancel(self):
        """Steer pending during concurrent execution: wait loop detects it.
        With 3 slow tools, the wait loop (5s poll) fires once while tools
        are still running. All futures are already running so cancel() has
        no effect, but the steer is drained in the post-collection phase.
        """
        from agent.tool_executor import execute_tool_calls_concurrent

        agent = _StubAgent()
        agent._pending_steer = "use Python"

        # Make tools slow so the wait loop (5s poll) triggers at least once
        import time
        def slow_tool(*args, **kwargs):
            time.sleep(10)
            return '{"status": "ok"}'
        agent._invoke_tool = slow_tool

        tool_calls = [_FakeToolCall(f"slow_tool_{i}") for i in range(3)]
        assistant_msg = _FakeAssistantMsg(tool_calls)
        messages = []

        execute_tool_calls_concurrent(agent, assistant_msg, messages, "task1")

        # All tools complete (they were already running, can't be cancelled)
        tool_msgs = _count_tool_messages(messages)
        assert len(tool_msgs) == 3

        # Steer was drained by the post-collection apply_pending_steer
        assert agent._pending_steer is None or agent._pending_steer == "", \
            f"Expected steer to be drained, got: {agent._pending_steer!r}"

    def test_concurrent_no_steer_all_tools_complete(self):
        """Without steer, all concurrent tools complete normally."""
        from agent.tool_executor import execute_tool_calls_concurrent

        agent = _StubAgent()

        tool_calls = [_FakeToolCall(f"tool_{i}") for i in range(3)]
        assistant_msg = _FakeAssistantMsg(tool_calls)
        messages = []

        execute_tool_calls_concurrent(agent, assistant_msg, messages, "task1")

        tool_msgs = _count_tool_messages(messages)
        assert len(tool_msgs) == 3


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestSteerBreakoutEdgeCases:
    """Edge cases for steer breakout."""

    @patch("agent.tool_executor._ra")
    def test_single_tool_steer_no_skip(self, mock_ra):
        """Single tool call: steer consumed after it, no tools to skip."""
        from agent.tool_executor import execute_tool_calls_sequential

        agent = _StubAgent()
        agent._apply_pending_steer_to_tool_results = MagicMock(return_value=True)
        mock_ra.return_value.handle_function_call.return_value = "ok"

        tool_calls = [_FakeToolCall("single_tool")]
        assistant_msg = _FakeAssistantMsg(tool_calls)
        messages = []

        execute_tool_calls_sequential(agent, assistant_msg, messages, "task1")

        tool_msgs = _count_tool_messages(messages)
        assert len(tool_msgs) == 1
        assert "ok" in tool_msgs[0]["content"]

    @patch("agent.tool_executor._ra")
    def test_steer_on_last_tool_no_skip(self, mock_ra):
        """Steer consumed after the last tool: no tools to skip."""
        from agent.tool_executor import execute_tool_calls_sequential

        agent = _StubAgent()
        call_count = [0]

        def fake_steer(messages, n):
            call_count[0] += 1
            if call_count[0] == 2:
                for msg in reversed(messages):
                    if isinstance(msg, dict) and msg.get("role") == "tool":
                        msg["content"] += "\n\nUser guidance: faster"
                        return True
            return False

        agent._apply_pending_steer_to_tool_results = fake_steer
        mock_ra.return_value.handle_function_call.return_value = "ok"

        tool_calls = [_FakeToolCall(f"tool_{i}") for i in range(2)]
        assistant_msg = _FakeAssistantMsg(tool_calls)
        messages = []

        execute_tool_calls_sequential(agent, assistant_msg, messages, "task1")

        tool_msgs = _count_tool_messages(messages)
        assert len(tool_msgs) == 2
        assert "ok" in tool_msgs[0]["content"]
        assert "ok" in tool_msgs[1]["content"]

    @patch("agent.tool_executor._ra")
    def test_steer_breakout_before_interrupt(self, mock_ra):
        """When steer is consumed, breakout fires before interrupt check.
        Steer drain (L898) runs first, so steer breakout (L904) preempts
        the interrupt check (L931). This is correct behavior — steer is
        a softer signal that should be processed before a hard interrupt.
        """
        from agent.tool_executor import execute_tool_calls_sequential

        agent = _StubAgent()
        # Steer will be consumed after first tool
        call_count = [0]
        def fake_steer(messages, n):
            call_count[0] += 1
            if call_count[0] >= 1:
                for msg in reversed(messages):
                    if isinstance(msg, dict) and msg.get("role") == "tool":
                        msg["content"] += "\n\nUser guidance: change plan"
                        return True
            return False
        agent._apply_pending_steer_to_tool_results = fake_steer
        mock_ra.return_value.handle_function_call.return_value = "ok"

        tool_calls = [_FakeToolCall(f"tool_{i}") for i in range(4)]
        assistant_msg = _FakeAssistantMsg(tool_calls)
        messages = []

        # Also set interrupt — but steer should fire first
        def set_interrupt_after_first(*args, **kwargs):
            if len(messages) >= 1:
                agent._interrupt_requested = True
            return "ok"
        mock_ra.return_value.handle_function_call.side_effect = set_interrupt_after_first

        execute_tool_calls_sequential(agent, assistant_msg, messages, "task1")

        tool_msgs = _count_tool_messages(messages)
        assert len(tool_msgs) == 4
        # Steer breakout fires — remaining tools deferred, not interrupted
        assert "steer" in tool_msgs[1]["content"].lower() or "deferred" in tool_msgs[1]["content"].lower()

    @patch("agent.tool_executor._ra")
    def test_interrupt_works_without_steer(self, mock_ra):
        """When no steer but interrupt is set, interrupt skips remaining tools."""
        from agent.tool_executor import execute_tool_calls_sequential

        agent = _StubAgent()
        # No steer consumed (default False)
        mock_ra.return_value.handle_function_call.return_value = "ok"

        tool_calls = [_FakeToolCall(f"tool_{i}") for i in range(4)]
        assistant_msg = _FakeAssistantMsg(tool_calls)
        messages = []

        call_count = [0]
        def interrupt_on_second(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] >= 2:
                agent._interrupt_requested = True
            return "ok"
        mock_ra.return_value.handle_function_call.side_effect = interrupt_on_second

        execute_tool_calls_sequential(agent, assistant_msg, messages, "task1")

        tool_msgs = _count_tool_messages(messages)
        assert len(tool_msgs) == 4
        # First 2 tools execute normally, last 2 interrupted
        assert "ok" in tool_msgs[0]["content"]
        assert "ok" in tool_msgs[1]["content"]
        assert "skipped" in tool_msgs[2]["content"].lower() or "interrupt" in tool_msgs[2]["content"].lower()
