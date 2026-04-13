"""Tests for LongcatToolCallParser."""
from __future__ import annotations

from environments.tool_call_parsers import get_parser

from .conftest import args_as_dict


class TestLongcatParser:
    def setup_method(self):
        self.parser = get_parser("longcat")

    def test_no_tool_call(self):
        content, tool_calls = self.parser.parse("just text")
        assert content == "just text"
        assert tool_calls is None

    def test_single_tool_call(self):
        text = '<longcat_tool_call>{"name": "search", "arguments": {"q": "hi"}}</longcat_tool_call>'
        content, tool_calls = self.parser.parse(text)
        assert content is None
        assert tool_calls[0].function.name == "search"
        assert args_as_dict(tool_calls[0]) == {"q": "hi"}

    def test_multiple_calls(self):
        text = (
            '<longcat_tool_call>{"name":"a","arguments":{}}</longcat_tool_call>'
            '<longcat_tool_call>{"name":"b","arguments":{}}</longcat_tool_call>'
        )
        _, tool_calls = self.parser.parse(text)
        assert [tc.function.name for tc in tool_calls] == ["a", "b"]

    def test_content_before_call(self):
        text = 'Preamble. <longcat_tool_call>{"name":"x","arguments":{}}</longcat_tool_call>'
        content, tool_calls = self.parser.parse(text)
        assert content == "Preamble."
        assert len(tool_calls) == 1

    def test_malformed_json_fallthrough(self):
        text = '<longcat_tool_call>{bad}</longcat_tool_call>'
        content, tool_calls = self.parser.parse(text)
        assert content == text
        assert tool_calls is None

    def test_does_not_match_hermes_tag(self):
        """Longcat parser must ignore <tool_call> (that's hermes-format)."""
        text = '<tool_call>{"name":"x","arguments":{}}</tool_call>'
        content, tool_calls = self.parser.parse(text)
        assert content == text
        assert tool_calls is None
