"""Regression tests: normalize_anthropic_response_v2 vs v1.

Constructs mock Anthropic responses and asserts that the v2 function
(returning NormalizedResponse) produces identical field values to the
original v1 function (returning SimpleNamespace + finish_reason).
"""

import json
import pytest
from types import SimpleNamespace

from agent.anthropic_adapter import (
    normalize_anthropic_response,
    normalize_anthropic_response_v2,
)
from agent.transports.types import NormalizedResponse, ToolCall


# ---------------------------------------------------------------------------
# Helpers to build mock Anthropic SDK responses
# ---------------------------------------------------------------------------

def _text_block(text: str):
    return SimpleNamespace(type="text", text=text)


def _thinking_block(thinking: str, signature: str = "sig_abc"):
    return SimpleNamespace(type="thinking", thinking=thinking, signature=signature)


def _tool_use_block(id: str, name: str, input: dict):
    return SimpleNamespace(type="tool_use", id=id, name=name, input=input)


def _response(content_blocks, stop_reason="end_turn"):
    return SimpleNamespace(
        content=content_blocks,
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTextOnly:
    """Text-only response — no tools, no thinking."""

    def setup_method(self):
        self.resp = _response([_text_block("Hello world")])
        self.v1_msg, self.v1_finish = normalize_anthropic_response(self.resp)
        self.v2 = normalize_anthropic_response_v2(self.resp)

    def test_type(self):
        assert isinstance(self.v2, NormalizedResponse)

    def test_content_matches(self):
        assert self.v2.content == self.v1_msg.content

    def test_finish_reason_matches(self):
        assert self.v2.finish_reason == self.v1_finish

    def test_no_tool_calls(self):
        assert self.v2.tool_calls is None
        assert self.v1_msg.tool_calls is None

    def test_no_reasoning(self):
        assert self.v2.reasoning is None
        assert self.v1_msg.reasoning is None


class TestWithToolCalls:
    """Response with tool calls."""

    def setup_method(self):
        self.resp = _response(
            [
                _text_block("I'll check that"),
                _tool_use_block("toolu_abc", "terminal", {"command": "ls"}),
                _tool_use_block("toolu_def", "read_file", {"path": "/tmp"}),
            ],
            stop_reason="tool_use",
        )
        self.v1_msg, self.v1_finish = normalize_anthropic_response(self.resp)
        self.v2 = normalize_anthropic_response_v2(self.resp)

    def test_finish_reason(self):
        assert self.v2.finish_reason == "tool_calls"
        assert self.v1_finish == "tool_calls"

    def test_tool_call_count(self):
        assert len(self.v2.tool_calls) == 2
        assert len(self.v1_msg.tool_calls) == 2

    def test_tool_call_ids_match(self):
        for i in range(2):
            assert self.v2.tool_calls[i].id == self.v1_msg.tool_calls[i].id

    def test_tool_call_names_match(self):
        assert self.v2.tool_calls[0].name == "terminal"
        assert self.v2.tool_calls[1].name == "read_file"
        for i in range(2):
            assert self.v2.tool_calls[i].name == self.v1_msg.tool_calls[i].function.name

    def test_tool_call_arguments_match(self):
        for i in range(2):
            assert self.v2.tool_calls[i].arguments == self.v1_msg.tool_calls[i].function.arguments

    def test_content_preserved(self):
        assert self.v2.content == self.v1_msg.content
        assert "check that" in self.v2.content


class TestWithThinking:
    """Response with thinking blocks (Claude 3.5+ extended thinking)."""

    def setup_method(self):
        self.resp = _response([
            _thinking_block("Let me think about this carefully..."),
            _text_block("The answer is 42."),
        ])
        self.v1_msg, self.v1_finish = normalize_anthropic_response(self.resp)
        self.v2 = normalize_anthropic_response_v2(self.resp)

    def test_reasoning_matches(self):
        assert self.v2.reasoning == self.v1_msg.reasoning
        assert "think about this" in self.v2.reasoning

    def test_reasoning_details_in_provider_data(self):
        v1_details = self.v1_msg.reasoning_details
        v2_details = self.v2.provider_data.get("reasoning_details") if self.v2.provider_data else None
        assert v1_details is not None
        assert v2_details is not None
        assert len(v2_details) == len(v1_details)

    def test_content_excludes_thinking(self):
        assert self.v2.content == "The answer is 42."


class TestMixed:
    """Response with thinking + text + tool calls."""

    def setup_method(self):
        self.resp = _response(
            [
                _thinking_block("Planning my approach..."),
                _text_block("I'll run the command"),
                _tool_use_block("toolu_xyz", "terminal", {"command": "pwd"}),
            ],
            stop_reason="tool_use",
        )
        self.v1_msg, self.v1_finish = normalize_anthropic_response(self.resp)
        self.v2 = normalize_anthropic_response_v2(self.resp)

    def test_all_fields_present(self):
        assert self.v2.content is not None
        assert self.v2.tool_calls is not None
        assert self.v2.reasoning is not None
        assert self.v2.finish_reason == "tool_calls"

    def test_content_matches(self):
        assert self.v2.content == self.v1_msg.content

    def test_reasoning_matches(self):
        assert self.v2.reasoning == self.v1_msg.reasoning

    def test_tool_call_matches(self):
        assert self.v2.tool_calls[0].id == self.v1_msg.tool_calls[0].id
        assert self.v2.tool_calls[0].name == self.v1_msg.tool_calls[0].function.name


class TestStopReasons:
    """Verify finish_reason mapping matches between v1 and v2."""

    @pytest.mark.parametrize("stop_reason,expected", [
        ("end_turn", "stop"),
        ("tool_use", "tool_calls"),
        ("max_tokens", "length"),
        ("stop_sequence", "stop"),
        ("refusal", "content_filter"),
        ("model_context_window_exceeded", "length"),
        ("unknown_future_reason", "stop"),
    ])
    def test_stop_reason_mapping(self, stop_reason, expected):
        resp = _response([_text_block("x")], stop_reason=stop_reason)
        v1_msg, v1_finish = normalize_anthropic_response(resp)
        v2 = normalize_anthropic_response_v2(resp)
        assert v2.finish_reason == v1_finish == expected


class TestStripToolPrefix:
    """Verify mcp_ prefix stripping works identically."""

    def test_prefix_stripped(self):
        resp = _response(
            [_tool_use_block("toolu_1", "mcp_terminal", {"cmd": "ls"})],
            stop_reason="tool_use",
        )
        v1_msg, _ = normalize_anthropic_response(resp, strip_tool_prefix=True)
        v2 = normalize_anthropic_response_v2(resp, strip_tool_prefix=True)
        assert v1_msg.tool_calls[0].function.name == "terminal"
        assert v2.tool_calls[0].name == "terminal"

    def test_prefix_kept(self):
        resp = _response(
            [_tool_use_block("toolu_1", "mcp_terminal", {"cmd": "ls"})],
            stop_reason="tool_use",
        )
        v1_msg, _ = normalize_anthropic_response(resp, strip_tool_prefix=False)
        v2 = normalize_anthropic_response_v2(resp, strip_tool_prefix=False)
        assert v1_msg.tool_calls[0].function.name == "mcp_terminal"
        assert v2.tool_calls[0].name == "mcp_terminal"


class TestEdgeCases:
    """Edge cases: empty content, no blocks, etc."""

    def test_empty_content_blocks(self):
        resp = _response([])
        v1_msg, v1_finish = normalize_anthropic_response(resp)
        v2 = normalize_anthropic_response_v2(resp)
        assert v2.content == v1_msg.content
        assert v2.content is None

    def test_no_reasoning_details_means_none_provider_data(self):
        resp = _response([_text_block("hi")])
        v2 = normalize_anthropic_response_v2(resp)
        assert v2.provider_data is None

    def test_v2_returns_dataclass_not_namespace(self):
        resp = _response([_text_block("hi")])
        v2 = normalize_anthropic_response_v2(resp)
        assert isinstance(v2, NormalizedResponse)
        assert not isinstance(v2, SimpleNamespace)
