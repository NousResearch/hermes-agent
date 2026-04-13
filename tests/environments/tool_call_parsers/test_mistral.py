"""Tests for MistralToolCallParser (pre-v11 and v11+ formats)."""
from __future__ import annotations

from environments.tool_call_parsers import get_parser

from .conftest import args_as_dict


class TestMistralParser:
    def setup_method(self):
        self.parser = get_parser("mistral")

    def test_no_bot_token(self):
        content, tool_calls = self.parser.parse("Just a reply.")
        assert content == "Just a reply."
        assert tool_calls is None

    def test_pre_v11_array_format(self):
        text = (
            'content before[TOOL_CALLS] [{"name":"a","arguments":{"x":1}}, '
            '{"name":"b","arguments":{"y":2}}]'
        )
        content, tool_calls = self.parser.parse(text)
        assert content == "content before"
        assert [tc.function.name for tc in tool_calls] == ["a", "b"]
        assert args_as_dict(tool_calls[0]) == {"x": 1}
        assert args_as_dict(tool_calls[1]) == {"y": 2}

    def test_pre_v11_single_dict(self):
        text = 'hi[TOOL_CALLS] {"name":"a","arguments":{"x":1}}'
        content, tool_calls = self.parser.parse(text)
        assert content == "hi"
        assert tool_calls[0].function.name == "a"

    def test_v11_format_name_then_brace(self):
        text = 'pre[TOOL_CALLS]get_weather{"city":"NYC"}[TOOL_CALLS]ping{}'
        content, tool_calls = self.parser.parse(text)
        assert content == "pre"
        assert [tc.function.name for tc in tool_calls] == ["get_weather", "ping"]
        assert args_as_dict(tool_calls[0]) == {"city": "NYC"}

    def test_malformed_pre_v11_falls_through(self):
        text = "x[TOOL_CALLS] [{bad]"
        content, tool_calls = self.parser.parse(text)
        # Raw-decode fallback finds no valid JSON → no tool calls.
        assert tool_calls is None or tool_calls == []
        if tool_calls is None:
            assert content == text

    def test_mistral_ids_are_9_char_alnum(self):
        text = '[TOOL_CALLS] [{"name":"a","arguments":{}}]'
        _, tool_calls = self.parser.parse(text)
        assert len(tool_calls[0].id) == 9
        assert tool_calls[0].id.isalnum()
