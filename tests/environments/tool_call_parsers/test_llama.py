"""Tests for LlamaToolCallParser (Llama3/4 JSON + <|python_tag|>)."""
from __future__ import annotations

from environments.tool_call_parsers import get_parser

from .conftest import args_as_dict


class TestLlamaParser:
    def setup_method(self):
        self.parser = get_parser("llama3_json")

    def test_no_json_or_tag(self):
        content, tool_calls = self.parser.parse("Just chat.")
        assert content == "Just chat."
        assert tool_calls is None

    def test_bare_json_object(self):
        text = '{"name": "get_weather", "arguments": {"city": "NYC"}}'
        content, tool_calls = self.parser.parse(text)
        assert tool_calls[0].function.name == "get_weather"
        assert args_as_dict(tool_calls[0]) == {"city": "NYC"}

    def test_python_tag_prefix(self):
        text = '<|python_tag|>{"name": "ping", "arguments": {}}'
        content, tool_calls = self.parser.parse(text)
        assert tool_calls[0].function.name == "ping"

    def test_parameters_key_accepted(self):
        """Some llama fine-tunes emit 'parameters' instead of 'arguments'."""
        text = '{"name": "search", "parameters": {"q": "x"}}'
        _, tool_calls = self.parser.parse(text)
        assert args_as_dict(tool_calls[0]) == {"q": "x"}

    def test_multiple_json_objects(self):
        text = (
            '{"name":"a","arguments":{"x":1}} '
            '{"name":"b","arguments":{"y":2}}'
        )
        _, tool_calls = self.parser.parse(text)
        assert [tc.function.name for tc in tool_calls] == ["a", "b"]

    def test_content_before_first_call(self):
        text = 'preamble {"name":"a","arguments":{}}'
        content, tool_calls = self.parser.parse(text)
        assert content == "preamble"
        assert tool_calls[0].function.name == "a"

    def test_json_without_name_ignored(self):
        """Random JSON that lacks 'name' must not produce a tool call."""
        text = '{"foo": "bar"}'
        content, tool_calls = self.parser.parse(text)
        assert tool_calls is None
        assert content == text

    def test_llama4_alias(self):
        p4 = get_parser("llama4_json")
        text = '{"name":"x","arguments":{}}'
        _, tool_calls = p4.parse(text)
        assert tool_calls[0].function.name == "x"
