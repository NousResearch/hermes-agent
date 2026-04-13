"""Tests for Qwen3CoderToolCallParser — XML-style nested tags."""
from __future__ import annotations

from environments.tool_call_parsers import get_parser

from .conftest import args_as_dict


class TestQwen3CoderParser:
    def setup_method(self):
        self.parser = get_parser("qwen3_coder")

    def test_no_function_prefix(self):
        content, tool_calls = self.parser.parse("plain response")
        assert content == "plain response"
        assert tool_calls is None

    def test_single_function_with_string_param(self):
        text = (
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=city>NYC</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        _, tool_calls = self.parser.parse(text)
        assert tool_calls[0].function.name == "get_weather"
        assert args_as_dict(tool_calls[0]) == {"city": "NYC"}

    def test_numeric_value_converted(self):
        text = (
            "<tool_call>\n"
            "<function=count>\n"
            "<parameter=n>42</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        _, tool_calls = self.parser.parse(text)
        assert args_as_dict(tool_calls[0]) == {"n": 42}

    def test_null_value_converted(self):
        text = (
            "<tool_call>\n"
            "<function=optional>\n"
            "<parameter=x>null</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        _, tool_calls = self.parser.parse(text)
        assert args_as_dict(tool_calls[0]) == {"x": None}

    def test_multiple_params(self):
        text = (
            "<tool_call>\n"
            "<function=f>\n"
            "<parameter=a>1</parameter>\n"
            "<parameter=b>two</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        _, tool_calls = self.parser.parse(text)
        assert args_as_dict(tool_calls[0]) == {"a": 1, "b": "two"}
