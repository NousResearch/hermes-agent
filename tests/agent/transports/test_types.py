"""Tests for agent/transports/types.py — dataclass construction + helpers."""

import json

from agent.transports.types import (
    NormalizedResponse,
    ToolCall,
    build_tool_call,
    map_finish_reason,
)


# ---------------------------------------------------------------------------
# ToolCall
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# NormalizedResponse
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# build_tool_call
# ---------------------------------------------------------------------------

class TestBuildToolCall:
    def test_dict_arguments_serialized(self):
        tc = build_tool_call(id="call_1", name="terminal", arguments={"cmd": "ls"})
        assert tc.arguments == json.dumps({"cmd": "ls"})
        assert tc.provider_data is None


# ---------------------------------------------------------------------------
# map_finish_reason
# ---------------------------------------------------------------------------

class TestMapFinishReason:
    ANTHROPIC_MAP = {
        "end_turn": "stop",
        "tool_use": "tool_calls",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "refusal": "content_filter",
    }

    def test_known_reason(self):
        assert map_finish_reason("end_turn", self.ANTHROPIC_MAP) == "stop"
        assert map_finish_reason("tool_use", self.ANTHROPIC_MAP) == "tool_calls"
        assert map_finish_reason("max_tokens", self.ANTHROPIC_MAP) == "length"
        assert map_finish_reason("refusal", self.ANTHROPIC_MAP) == "content_filter"

    def test_unknown_reason_defaults_to_stop(self):
        assert map_finish_reason("something_new", self.ANTHROPIC_MAP) == "stop"

    def test_none_reason(self):
        assert map_finish_reason(None, self.ANTHROPIC_MAP) == "stop"


# ---------------------------------------------------------------------------
# Backward-compat property tests
# ---------------------------------------------------------------------------

class TestToolCallBackwardCompat:
    """Test duck-typing properties that let ToolCall pass through code expecting
    the old SimpleNamespace(id, type, function=SimpleNamespace(name, arguments)) shape."""


    def test_function_name_matches(self):
        tc = ToolCall(id="1", name="search", arguments='{"q":"test"}')
        assert tc.function.name == "search"
        assert tc.function.name == tc.name

    def test_function_arguments_matches(self):
        tc = ToolCall(id="1", name="search", arguments='{"q":"test"}')
        assert tc.function.arguments == '{"q":"test"}'
        assert tc.function.arguments == tc.arguments


    def test_getattr_pattern_matches_agent_loop(self):
        """run_agent.py uses getattr(tool_call, 'call_id', None) — verify it works."""
        tc = ToolCall(id="1", name="fn", arguments="{}", provider_data={"call_id": "c1"})
        assert getattr(tc, "call_id", None) == "c1"
        tc_no_pd = ToolCall(id="1", name="fn", arguments="{}")
        assert getattr(tc_no_pd, "call_id", None) is None


    def test_extra_content_getattr_pattern(self):
        """_build_assistant_message uses getattr(tc, 'extra_content', None).

        This is the exact pattern that was broken before the extra_content
        property was added — ToolCall lacked the property so getattr always
        returned None, silently dropping the Gemini thought_signature and
        causing HTTP 400 on subsequent turns (issue #14488).
        """
        ec = {"google": {"thought_signature": "SIG_ABC123"}}
        tc = ToolCall(id="1", name="fn", arguments="{}", provider_data={"extra_content": ec})
        assert getattr(tc, "extra_content", None) == ec

        tc_no_extra = ToolCall(id="1", name="fn", arguments="{}")
        assert getattr(tc_no_extra, "extra_content", None) is None


class TestNormalizedResponseBackwardCompat:
    """Test properties that replaced _nr_to_assistant_message() shim."""

    def test_reasoning_content_from_provider_data(self):
        nr = NormalizedResponse(
            content="hi", tool_calls=None, finish_reason="stop",
            provider_data={"reasoning_content": "thought process"},
        )
        assert nr.reasoning_content == "thought process"

    def test_reasoning_content_none_when_absent(self):
        nr = NormalizedResponse(content="hi", tool_calls=None, finish_reason="stop")
        assert nr.reasoning_content is None

    def test_reasoning_details_from_provider_data(self):
        details = [{"type": "thinking", "thinking": "hmm"}]
        nr = NormalizedResponse(
            content="hi", tool_calls=None, finish_reason="stop",
            provider_data={"reasoning_details": details},
        )
        assert nr.reasoning_details == details
