"""Tests for KimiK2ToolCallParser."""
from __future__ import annotations

from environments.tool_call_parsers import get_parser


class TestKimiK2Parser:
    def setup_method(self):
        self.parser = get_parser("kimi_k2")

    def test_no_section_token(self):
        content, tool_calls = self.parser.parse("plain")
        assert content == "plain"
        assert tool_calls is None

    def test_single_call_extracts_name_from_id(self):
        text = (
            "<|tool_calls_section_begin|>"
            '<|tool_call_begin|>functions.get_weather:0'
            '<|tool_call_argument_begin|>{"city":"NYC"}<|tool_call_end|>'
            "<|tool_calls_section_end|>"
        )
        content, tool_calls = self.parser.parse(text)
        assert tool_calls[0].function.name == "get_weather"
        assert tool_calls[0].function.arguments == '{"city":"NYC"}'
        assert tool_calls[0].id == "functions.get_weather:0"

    def test_multiple_calls(self):
        text = (
            "<|tool_calls_section_begin|>"
            '<|tool_call_begin|>functions.a:0<|tool_call_argument_begin|>{}<|tool_call_end|>'
            '<|tool_call_begin|>functions.b:1<|tool_call_argument_begin|>{"x":1}<|tool_call_end|>'
            "<|tool_calls_section_end|>"
        )
        _, tool_calls = self.parser.parse(text)
        assert [tc.function.name for tc in tool_calls] == ["a", "b"]

    def test_id_without_module_prefix(self):
        """Per parser doc: 'func_name:index' form also accepted."""
        text = (
            "<|tool_calls_section_begin|>"
            '<|tool_call_begin|>ping:0<|tool_call_argument_begin|>{}<|tool_call_end|>'
        )
        _, tool_calls = self.parser.parse(text)
        assert tool_calls[0].function.name == "ping"

    def test_singular_section_variant(self):
        text = (
            "<|tool_call_section_begin|>"
            '<|tool_call_begin|>functions.ping:0<|tool_call_argument_begin|>{}<|tool_call_end|>'
        )
        _, tool_calls = self.parser.parse(text)
        assert tool_calls[0].function.name == "ping"

    def test_start_token_without_bodies(self):
        text = "<|tool_calls_section_begin|> nothing inside"
        content, tool_calls = self.parser.parse(text)
        assert tool_calls is None
        assert content == text
