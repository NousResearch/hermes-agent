"""Tests for Glm47ToolCallParser — inherits GLM 4.5 with updated regex."""
from __future__ import annotations

from environments.tool_call_parsers import get_parser

from .conftest import args_as_dict


class TestGlm47Parser:
    def setup_method(self):
        self.parser = get_parser("glm47")

    def test_no_tool_call(self):
        content, tool_calls = self.parser.parse("plain")
        assert content == "plain"
        assert tool_calls is None

    def test_single_call_with_args(self):
        text = (
            "<tool_call>get_weather\n"
            "<arg_key>city</arg_key><arg_value>NYC</arg_value>\n"
            "</tool_call>"
        )
        _, tool_calls = self.parser.parse(text)
        assert tool_calls[0].function.name == "get_weather"
        assert args_as_dict(tool_calls[0]) == {"city": "NYC"}

    def test_inherits_from_glm45(self):
        from environments.tool_call_parsers.glm45_parser import Glm45ToolCallParser
        assert isinstance(self.parser, Glm45ToolCallParser)
