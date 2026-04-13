"""Tests for Glm45ToolCallParser — custom arg_key/arg_value tags."""
from __future__ import annotations

from environments.tool_call_parsers import get_parser

from .conftest import args_as_dict


class TestGlm45Parser:
    def setup_method(self):
        self.parser = get_parser("glm45")

    def test_no_tool_call(self):
        content, tool_calls = self.parser.parse("plain text")
        assert content == "plain text"
        assert tool_calls is None

    def test_single_call_with_args(self):
        text = (
            "<tool_call>get_weather\n"
            "<arg_key>city</arg_key><arg_value>NYC</arg_value>\n"
            "<arg_key>units</arg_key><arg_value>metric</arg_value>\n"
            "</tool_call>"
        )
        _, tool_calls = self.parser.parse(text)
        assert tool_calls[0].function.name == "get_weather"
        assert args_as_dict(tool_calls[0]) == {"city": "NYC", "units": "metric"}

    def test_value_deserialization_to_int(self):
        text = (
            "<tool_call>count\n"
            "<arg_key>n</arg_key><arg_value>42</arg_value>\n"
            "</tool_call>"
        )
        _, tool_calls = self.parser.parse(text)
        # json.loads or literal_eval should turn "42" into 42.
        assert args_as_dict(tool_calls[0]) == {"n": 42}

    def test_value_deserialization_to_list(self):
        text = (
            "<tool_call>batch\n"
            '<arg_key>items</arg_key><arg_value>[1, 2, 3]</arg_value>\n'
            "</tool_call>"
        )
        _, tool_calls = self.parser.parse(text)
        assert args_as_dict(tool_calls[0]) == {"items": [1, 2, 3]}

    def test_multiple_calls(self):
        text = (
            "<tool_call>a\n</tool_call>"
            "<tool_call>b\n<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>"
        )
        _, tool_calls = self.parser.parse(text)
        assert [tc.function.name for tc in tool_calls] == ["a", "b"]
        assert args_as_dict(tool_calls[1]) == {"x": 1}

    def test_content_before_call(self):
        text = "thinking...\n<tool_call>x\n</tool_call>"
        content, _ = self.parser.parse(text)
        assert content == "thinking..."
