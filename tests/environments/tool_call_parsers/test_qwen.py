"""Tests for QwenToolCallParser (alias for hermes format)."""
from __future__ import annotations

from environments.tool_call_parsers import get_parser

from .conftest import args_as_dict


class TestQwenParser:
    def setup_method(self):
        self.parser = get_parser("qwen")

    def test_qwen_uses_hermes_format(self):
        text = '<tool_call>{"name": "x", "arguments": {"a": 1}}</tool_call>'
        content, tool_calls = self.parser.parse(text)
        assert content is None
        assert tool_calls[0].function.name == "x"
        assert args_as_dict(tool_calls[0]) == {"a": 1}

    def test_qwen_no_tool(self):
        content, tool_calls = self.parser.parse("plain")
        assert content == "plain"
        assert tool_calls is None
