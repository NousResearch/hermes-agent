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
