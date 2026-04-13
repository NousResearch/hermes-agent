"""Tests for HermesToolCallParser."""
from __future__ import annotations

from environments.tool_call_parsers import get_parser

from .conftest import args_as_dict


class TestHermesParser:
    def setup_method(self):
        self.parser = get_parser("hermes")

    def test_no_tool_call_returns_raw_text(self):
        text = "This is just a normal response."
        content, tool_calls = self.parser.parse(text)
        assert content == text
        assert tool_calls is None

    def test_single_tool_call(self):
        text = '<tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_call>'
        content, tool_calls = self.parser.parse(text)
        assert content is None
        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "get_weather"
        assert args_as_dict(tool_calls[0]) == {"city": "NYC"}

    def test_multiple_tool_calls(self):
        text = (
            '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
            '<tool_call>{"name": "b", "arguments": {"x": 1}}</tool_call>'
        )
        content, tool_calls = self.parser.parse(text)
        assert content is None
        assert [tc.function.name for tc in tool_calls] == ["a", "b"]
        assert args_as_dict(tool_calls[1]) == {"x": 1}

    def test_content_preserved_before_tool_call(self):
        text = (
            "Let me check that. "
            '<tool_call>{"name": "get_weather", "arguments": {"city": "NYC"}}</tool_call>'
        )
        content, tool_calls = self.parser.parse(text)
        assert content == "Let me check that."
        assert len(tool_calls) == 1

    def test_malformed_json_returns_raw_text(self):
        text = '<tool_call>{"name": not-json}</tool_call>'
        content, tool_calls = self.parser.parse(text)
        # Graceful fall-through: whole input returned as content, no tool_calls.
        assert content == text
        assert tool_calls is None

    def test_unclosed_tag_truncated_generation(self):
        text = '<tool_call>{"name": "truncated", "arguments": {}}'
        content, tool_calls = self.parser.parse(text)
        # Either extracts the truncated call OR treats as malformed — both
        # are acceptable; the key invariant is no crash and JSON validity.
        if tool_calls:
            assert tool_calls[0].function.name == "truncated"
        else:
            assert content == text

    def test_arguments_missing_treated_as_empty(self):
        text = '<tool_call>{"name": "ping"}</tool_call>'
        content, tool_calls = self.parser.parse(text)
        assert len(tool_calls) == 1
        assert args_as_dict(tool_calls[0]) == {}
