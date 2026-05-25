"""Tests for agent.transports.types — ToolCall, Usage, NormalizedResponse, factories."""

from __future__ import annotations

import json

import pytest

from agent.transports.types import (
    NormalizedResponse,
    ToolCall,
    Usage,
    build_tool_call,
    map_finish_reason,
)


# ============================================================================
# ToolCall
# ============================================================================
class TestToolCall:
    def test_basic_construction(self):
        tc = ToolCall(id="call_1", name="search", arguments='{"q":"hi"}')
        assert tc.id == "call_1"
        assert tc.name == "search"
        assert tc.arguments == '{"q":"hi"}'

    def test_default_provider_data_is_none(self):
        tc = ToolCall(id="c1", name="f", arguments="{}")
        assert tc.provider_data is None

    def test_type_property_returns_function(self):
        tc = ToolCall(id="c1", name="f", arguments="{}")
        assert tc.type == "function"

    def test_function_property_returns_self(self):
        """tc.function.name and tc.function.arguments should work."""
        tc = ToolCall(id="c1", name="myfunc", arguments='{"x":1}')
        assert tc.function is tc
        assert tc.function.name == "myfunc"
        assert tc.function.arguments == '{"x":1}'

    def test_call_id_from_provider_data(self):
        tc = ToolCall(id="c1", name="f", arguments="{}",
                      provider_data={"call_id": "call_ABC"})
        assert tc.call_id == "call_ABC"

    def test_call_id_none_when_no_provider_data(self):
        tc = ToolCall(id="c1", name="f", arguments="{}")
        assert tc.call_id is None

    def test_call_id_none_when_empty_provider_data(self):
        tc = ToolCall(id="c1", name="f", arguments="{}", provider_data={})
        assert tc.call_id is None

    def test_response_item_id_from_provider_data(self):
        tc = ToolCall(id="c1", name="f", arguments="{}",
                      provider_data={"response_item_id": "fc_XYZ"})
        assert tc.response_item_id == "fc_XYZ"

    def test_response_item_id_none_when_missing(self):
        tc = ToolCall(id="c1", name="f", arguments="{}",
                      provider_data={"call_id": "x"})
        assert tc.response_item_id is None

    def test_extra_content_from_provider_data(self):
        tc = ToolCall(id="c1", name="f", arguments="{}",
                      provider_data={"extra_content": {"google": {"thought_signature": "sig"}}})
        assert tc.extra_content == {"google": {"thought_signature": "sig"}}

    def test_extra_content_none_when_missing(self):
        tc = ToolCall(id="c1", name="f", arguments="{}")
        assert tc.extra_content is None

    def test_repr_does_not_expose_provider_data(self):
        tc = ToolCall(id="c1", name="f", arguments="{}",
                      provider_data={"secret": "shh"})
        rep = repr(tc)
        assert "secret" not in rep
        assert "provider_data" not in rep

    def test_id_can_be_none(self):
        tc = ToolCall(id=None, name="f", arguments="{}")
        assert tc.id is None


# ============================================================================
# Usage
# ============================================================================
class TestUsage:
    def test_default_construction(self):
        u = Usage()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_tokens == 0
        assert u.cached_tokens == 0

    def test_partial_construction(self):
        u = Usage(prompt_tokens=100, completion_tokens=50)
        assert u.prompt_tokens == 100
        assert u.completion_tokens == 50
        assert u.total_tokens == 0
        assert u.cached_tokens == 0

    def test_full_construction(self):
        u = Usage(prompt_tokens=100, completion_tokens=50,
                  total_tokens=150, cached_tokens=20)
        assert u.total_tokens == 150
        assert u.cached_tokens == 20

    def test_is_dataclass(self):
        u = Usage(prompt_tokens=10)
        # Can access by attribute
        assert u.prompt_tokens == 10


# ============================================================================
# NormalizedResponse
# ============================================================================
class TestNormalizedResponse:
    def test_basic_construction(self):
        nr = NormalizedResponse(
            content="hello",
            tool_calls=None,
            finish_reason="stop",
        )
        assert nr.content == "hello"
        assert nr.tool_calls is None
        assert nr.finish_reason == "stop"
        assert nr.reasoning is None
        assert nr.usage is None
        assert nr.provider_data is None

    def test_with_tool_calls(self):
        tc = ToolCall(id="c1", name="f", arguments="{}")
        nr = NormalizedResponse(
            content=None,
            tool_calls=[tc],
            finish_reason="tool_calls",
        )
        assert nr.tool_calls == [tc]
        assert nr.content is None

    def test_reasoning_content_from_provider_data(self):
        nr = NormalizedResponse(
            content="ok",
            tool_calls=None,
            finish_reason="stop",
            provider_data={"reasoning_content": "let me think..."},
        )
        assert nr.reasoning_content == "let me think..."

    def test_reasoning_content_none_when_missing(self):
        nr = NormalizedResponse(content="ok", tool_calls=None, finish_reason="stop")
        assert nr.reasoning_content is None

    def test_reasoning_content_none_when_empty_provider_data(self):
        nr = NormalizedResponse(content="ok", tool_calls=None, finish_reason="stop",
                                provider_data={})
        assert nr.reasoning_content is None

    def test_reasoning_details_from_provider_data(self):
        nr = NormalizedResponse(
            content="ok", tool_calls=None, finish_reason="stop",
            provider_data={"reasoning_details": [{"type": "thinking"}]},
        )
        assert nr.reasoning_details == [{"type": "thinking"}]

    def test_reasoning_details_none_when_missing(self):
        nr = NormalizedResponse(content="ok", tool_calls=None, finish_reason="stop")
        assert nr.reasoning_details is None

    def test_codex_reasoning_items_from_provider_data(self):
        nr = NormalizedResponse(
            content="ok", tool_calls=None, finish_reason="stop",
            provider_data={"codex_reasoning_items": ["r1", "r2"]},
        )
        assert nr.codex_reasoning_items == ["r1", "r2"]

    def test_codex_reasoning_items_none_when_missing(self):
        nr = NormalizedResponse(content="ok", tool_calls=None, finish_reason="stop")
        assert nr.codex_reasoning_items is None

    def test_codex_message_items_from_provider_data(self):
        nr = NormalizedResponse(
            content="ok", tool_calls=None, finish_reason="stop",
            provider_data={"codex_message_items": ["m1"]},
        )
        assert nr.codex_message_items == ["m1"]

    def test_codex_message_items_none_when_missing(self):
        nr = NormalizedResponse(content="ok", tool_calls=None, finish_reason="stop")
        assert nr.codex_message_items is None

    def test_repr_does_not_expose_provider_data(self):
        nr = NormalizedResponse(
            content="ok", tool_calls=None, finish_reason="stop",
            provider_data={"secret": "hidden"},
        )
        rep = repr(nr)
        assert "secret" not in rep

    def test_with_usage(self):
        u = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        nr = NormalizedResponse(content="ok", tool_calls=None,
                                finish_reason="stop", usage=u)
        assert nr.usage.prompt_tokens == 10

    def test_with_reasoning(self):
        nr = NormalizedResponse(content="ok", tool_calls=None,
                                finish_reason="stop", reasoning="think...")
        assert nr.reasoning == "think..."


# ============================================================================
# build_tool_call
# ============================================================================
class TestBuildToolCall:
    def test_dict_arguments_serialized(self):
        tc = build_tool_call("c1", "search", {"q": "hello", "limit": 5})
        assert tc.name == "search"
        assert tc.id == "c1"
        # arguments should be JSON string
        parsed = json.loads(tc.arguments)
        assert parsed == {"q": "hello", "limit": 5}

    def test_string_arguments_preserved(self):
        tc = build_tool_call("c1", "f", "already a string")
        assert tc.arguments == "already a string"

    def test_int_arguments_str_converted(self):
        tc = build_tool_call("c1", "f", 42)
        assert tc.arguments == "42"

    def test_list_arguments_str_converted(self):
        tc = build_tool_call("c1", "f", [1, 2, 3])
        assert tc.arguments == "[1, 2, 3]"

    def test_none_id(self):
        tc = build_tool_call(None, "f", {})
        assert tc.id is None

    def test_no_provider_fields_gives_none_provider_data(self):
        tc = build_tool_call("c1", "f", {})
        assert tc.provider_data is None

    def test_extra_kwargs_become_provider_data(self):
        tc = build_tool_call("c1", "f", {}, call_id="call_X",
                             response_item_id="fc_Y")
        assert tc.provider_data == {"call_id": "call_X", "response_item_id": "fc_Y"}

    def test_single_provider_field(self):
        tc = build_tool_call("c1", "f", {}, extra_content={"sig": "x"})
        assert tc.provider_data == {"extra_content": {"sig": "x"}}

    def test_empty_dict_arguments(self):
        tc = build_tool_call("c1", "f", {})
        assert tc.arguments == "{}"


# ============================================================================
# map_finish_reason
# ============================================================================
class TestMapFinishReason:
    def test_known_reason_mapped(self):
        mapping = {"end_turn": "stop", "max_tokens": "length"}
        assert map_finish_reason("end_turn", mapping) == "stop"

    def test_unknown_reason_falls_back_to_stop(self):
        mapping = {"end_turn": "stop"}
        assert map_finish_reason("weird_reason", mapping) == "stop"

    def test_none_reason_falls_back_to_stop(self):
        assert map_finish_reason(None, {}) == "stop"

    def test_empty_mapping_falls_back_to_stop(self):
        assert map_finish_reason("anything", {}) == "stop"

    def test_multiple_mappings(self):
        mapping = {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "content_filter"}
        assert map_finish_reason("STOP", mapping) == "stop"
        assert map_finish_reason("MAX_TOKENS", mapping) == "length"
        assert map_finish_reason("SAFETY", mapping) == "content_filter"

    def test_exact_match_required(self):
        mapping = {"Stop": "stop"}
        # Case-sensitive — "stop" != "Stop"
        assert map_finish_reason("stop", mapping) == "stop"
