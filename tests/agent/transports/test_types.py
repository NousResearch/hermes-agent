"""Tests for agent/transports/types.py — dataclass construction + helpers."""

import json
import pytest

from agent.transports.types import (
    NormalizedResponse,
    ToolCall,
    Usage,
    build_tool_call,
    map_finish_reason,
)


# ---------------------------------------------------------------------------
# ToolCall
# ---------------------------------------------------------------------------

class TestToolCall:
    def test_basic_construction(self):
        tc = ToolCall(id="call_abc", name="terminal", arguments='{"cmd": "ls"}')
        assert tc.id == "call_abc"
        assert tc.name == "terminal"
        assert tc.arguments == '{"cmd": "ls"}'
        assert tc.provider_data is None

    def test_none_id(self):
        tc = ToolCall(id=None, name="read_file", arguments="{}")
        assert tc.id is None

    def test_provider_data(self):
        tc = ToolCall(
            id="call_x",
            name="t",
            arguments="{}",
            provider_data={"call_id": "call_x", "response_item_id": "fc_x"},
        )
        assert tc.provider_data["call_id"] == "call_x"
        assert tc.provider_data["response_item_id"] == "fc_x"


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

class TestUsage:
    def test_defaults(self):
        u = Usage()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_tokens == 0
        assert u.cached_tokens == 0

    def test_explicit(self):
        u = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150, cached_tokens=80)
        assert u.total_tokens == 150


# ---------------------------------------------------------------------------
# NormalizedResponse
# ---------------------------------------------------------------------------

class TestNormalizedResponse:
    def test_text_only(self):
        r = NormalizedResponse(content="hello", tool_calls=None, finish_reason="stop")
        assert r.content == "hello"
        assert r.tool_calls is None
        assert r.finish_reason == "stop"
        assert r.reasoning is None
        assert r.usage is None
        assert r.provider_data is None

    def test_with_tool_calls(self):
        tcs = [ToolCall(id="call_1", name="terminal", arguments='{"cmd":"pwd"}')]
        r = NormalizedResponse(content=None, tool_calls=tcs, finish_reason="tool_calls")
        assert r.finish_reason == "tool_calls"
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].name == "terminal"

    def test_with_reasoning(self):
        r = NormalizedResponse(
            content="answer",
            tool_calls=None,
            finish_reason="stop",
            reasoning="I thought about it",
        )
        assert r.reasoning == "I thought about it"

    def test_with_provider_data(self):
        r = NormalizedResponse(
            content=None,
            tool_calls=None,
            finish_reason="stop",
            provider_data={"reasoning_details": [{"type": "thinking", "thinking": "hmm"}]},
        )
        assert r.provider_data["reasoning_details"][0]["type"] == "thinking"


# ---------------------------------------------------------------------------
# build_tool_call
# ---------------------------------------------------------------------------

class TestBuildToolCall:
    def test_dict_arguments_serialized(self):
        tc = build_tool_call(id="call_1", name="terminal", arguments={"cmd": "ls"})
        assert tc.arguments == json.dumps({"cmd": "ls"})
        assert tc.provider_data is None

    def test_string_arguments_passthrough(self):
        tc = build_tool_call(id="call_2", name="read_file", arguments='{"path": "/tmp"}')
        assert tc.arguments == '{"path": "/tmp"}'

    def test_provider_fields(self):
        tc = build_tool_call(
            id="call_3",
            name="terminal",
            arguments="{}",
            call_id="call_3",
            response_item_id="fc_3",
        )
        assert tc.provider_data == {"call_id": "call_3", "response_item_id": "fc_3"}

    def test_none_id(self):
        tc = build_tool_call(id=None, name="t", arguments="{}")
        assert tc.id is None


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

    def test_type_is_function(self):
        tc = ToolCall(id="1", name="search", arguments='{"q":"test"}')
        assert tc.type == "function"

    def test_function_returns_self(self):
        tc = ToolCall(id="1", name="search", arguments='{"q":"test"}')
        assert tc.function is tc

    def test_function_name_matches(self):
        tc = ToolCall(id="1", name="search", arguments='{"q":"test"}')
        assert tc.function.name == "search"
        assert tc.function.name == tc.name

    def test_function_arguments_matches(self):
        tc = ToolCall(id="1", name="search", arguments='{"q":"test"}')
        assert tc.function.arguments == '{"q":"test"}'
        assert tc.function.arguments == tc.arguments

    def test_call_id_from_provider_data(self):
        tc = ToolCall(id="1", name="fn", arguments="{}", provider_data={"call_id": "c1"})
        assert tc.call_id == "c1"

    def test_call_id_none_when_no_provider_data(self):
        tc = ToolCall(id="1", name="fn", arguments="{}", provider_data=None)
        assert tc.call_id is None

    def test_response_item_id_from_provider_data(self):
        tc = ToolCall(id="1", name="fn", arguments="{}", provider_data={"response_item_id": "r1"})
        assert tc.response_item_id == "r1"

    def test_response_item_id_none_when_missing(self):
        tc = ToolCall(id="1", name="fn", arguments="{}", provider_data={"call_id": "c1"})
        assert tc.response_item_id is None

    def test_getattr_pattern_matches_agent_loop(self):
        """run_agent.py uses getattr(tool_call, 'call_id', None) — verify it works."""
        tc = ToolCall(id="1", name="fn", arguments="{}", provider_data={"call_id": "c1"})
        assert getattr(tc, "call_id", None) == "c1"
        tc_no_pd = ToolCall(id="1", name="fn", arguments="{}")
        assert getattr(tc_no_pd, "call_id", None) is None

    def test_extra_content_from_provider_data(self):
        """Gemini thought_signature stored in provider_data is exposed via property."""
        ec = {"google": {"thought_signature": "SIG_ABC123"}}
        tc = ToolCall(id="1", name="fn", arguments="{}", provider_data={"extra_content": ec})
        assert tc.extra_content == ec

    def test_extra_content_none_when_no_provider_data(self):
        tc = ToolCall(id="1", name="fn", arguments="{}", provider_data=None)
        assert tc.extra_content is None

    def test_extra_content_none_when_key_absent(self):
        tc = ToolCall(id="1", name="fn", arguments="{}", provider_data={"call_id": "c1"})
        assert tc.extra_content is None

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

    def test_reasoning_details_none_when_no_provider_data(self):
        nr = NormalizedResponse(
            content="hi", tool_calls=None, finish_reason="stop",
            provider_data=None,
        )
        assert nr.reasoning_details is None

    def test_codex_reasoning_items_from_provider_data(self):
        items = ["item1", "item2"]
        nr = NormalizedResponse(
            content="hi", tool_calls=None, finish_reason="stop",
            provider_data={"codex_reasoning_items": items},
        )
        assert nr.codex_reasoning_items == items

    def test_codex_reasoning_items_none_when_absent(self):
        nr = NormalizedResponse(content="hi", tool_calls=None, finish_reason="stop")
        assert nr.codex_reasoning_items is None
