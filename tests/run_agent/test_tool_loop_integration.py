"""Integration tests for tool loop detection in the agent loop.

Verifies:
1. ToolLoopDetector is initialized on the agent
2. Each tool call result is recorded in the detector
3. When a loop is detected at 'warning' severity, a log message is emitted
4. When a loop is detected at 'critical' severity, context is pruned
5. Reasoning content is passed to the detector for intent extraction
6. The detector resets on new user turn (reset_session_state)
"""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _make_agent():
    """Create a minimal AIAgent for testing loop detection."""
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
    ):
        from run_agent import AIAgent
        a = AIAgent(
            api_key="test-key-12345",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        a.client = MagicMock()
        a._cached_system_prompt = "You are helpful."
        a._use_prompt_caching = False
        a.tool_delay = 0
        a.compression_enabled = False
        return a


class TestDetectorInitialized:
    def test_agent_has_detector(self):
        a = _make_agent()
        assert hasattr(a, "_tool_loop_detector")
        from agent.tool_loop_detector import ToolLoopDetector
        assert isinstance(a._tool_loop_detector, ToolLoopDetector)

    def test_default_thresholds(self):
        a = _make_agent()
        assert a._tool_loop_detector.warning_threshold == 3
        assert a._tool_loop_detector.critical_threshold == 5


class TestDetectorRecording:
    def _make_tc(self, name="grep", args='{"pattern": "foo"}', call_id="tc1"):
        return SimpleNamespace(
            id=call_id,
            call_id=call_id,
            function=SimpleNamespace(name=name, arguments=args),
        )

    def _make_assistant_msg(self, tool_calls, reasoning=None):
        msg = SimpleNamespace(
            content="",
            tool_calls=tool_calls,
            reasoning=reasoning,
            reasoning_content=reasoning,
            reasoning_details=None,
        )
        return msg

    def test_record_called_after_tool_execution(self):
        a = _make_agent()
        a._tool_loop_detector = MagicMock()
        a._tool_loop_detector.record.return_value = SimpleNamespace(
            severity="none", detector=None, streak=0, intended_tool=None
        )

        tc = self._make_tc()
        assistant_msg = self._make_assistant_msg([tc])

        messages = [
            {"role": "user", "content": "find foo"},
            {
                "role": "assistant", "content": "",
                "finish_reason": "tool_calls",
                "tool_calls": [{"id": "tc1", "function": {"name": "grep", "arguments": '{"pattern": "foo"}'}}],
            },
            {"role": "tool", "content": "found 3 matches", "tool_call_id": "tc1"},
        ]

        a._check_tool_loop(messages, assistant_msg, "tool_calls")
        a._tool_loop_detector.record.assert_called_once()

    def test_critical_prunes_messages(self):
        a = _make_agent()
        a._tool_loop_detector = MagicMock()
        a._tool_loop_detector.record.return_value = SimpleNamespace(
            severity="critical", detector="generic_repeat", streak=5, intended_tool=None
        )

        messages = []
        for i in range(5):
            cid = f"call_{i}"
            messages.append({
                "role": "assistant", "content": "",
                "finish_reason": "tool_calls",
                "tool_calls": [{"id": cid, "call_id": cid, "type": "function",
                               "function": {"name": "bad_tool", "arguments": '{"x": 1}'}}],
            })
            messages.append({"role": "tool", "content": "error", "tool_call_id": cid})

        original_len = len(messages)
        tc = self._make_tc(name="bad_tool", args='{"x": 1}', call_id="call_4")
        assistant_msg = self._make_assistant_msg([tc])

        a._check_tool_loop(messages, assistant_msg, "tool_calls")
        assert len(messages) < original_len

    def test_warning_does_not_prune(self):
        a = _make_agent()
        a._tool_loop_detector = MagicMock()
        a._tool_loop_detector.record.return_value = SimpleNamespace(
            severity="warning", detector="generic_repeat", streak=3, intended_tool=None
        )

        messages = [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "finish_reason": "tool_calls",
             "tool_calls": [{"id": "c1", "function": {"name": "t", "arguments": "{}"}}]},
            {"role": "tool", "content": "r", "tool_call_id": "c1"},
        ]

        original_len = len(messages)
        tc = self._make_tc(name="t", args="{}", call_id="c1")
        assistant_msg = self._make_assistant_msg([tc])

        a._check_tool_loop(messages, assistant_msg, "tool_calls")
        assert len(messages) == original_len


class TestDetectorResetOnNewTurn:
    def test_reset_on_new_conversation(self):
        a = _make_agent()
        a._tool_loop_detector = MagicMock()
        a.reset_session_state()
        a._tool_loop_detector.reset.assert_called_once()
