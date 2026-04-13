"""Tests for DeepSeekV31ToolCallParser."""
from __future__ import annotations

from environments.tool_call_parsers import get_parser


OPEN_ALL = "<｜tool▁calls▁begin｜>"
OPEN_ONE = "<｜tool▁call▁begin｜>"
CLOSE_ONE = "<｜tool▁call▁end｜>"
SEP = "<｜tool▁sep｜>"


class TestDeepSeekV31Parser:
    def setup_method(self):
        self.parser = get_parser("deepseek_v3_1")

    def test_no_start_token(self):
        content, tool_calls = self.parser.parse("plain")
        assert content == "plain"
        assert tool_calls is None

    def test_single_call_name_then_args(self):
        text = f'{OPEN_ALL}\n{OPEN_ONE}search{SEP}{{"q": "x"}}{CLOSE_ONE}'
        content, tool_calls = self.parser.parse(text)
        assert tool_calls[0].function.name == "search"
        assert tool_calls[0].function.arguments == '{"q": "x"}'

    def test_multiple_calls(self):
        text = (
            f"{OPEN_ALL}\n"
            f"{OPEN_ONE}a{SEP}{{}}{CLOSE_ONE}\n"
            f"{OPEN_ONE}b{SEP}{{}}{CLOSE_ONE}"
        )
        _, tool_calls = self.parser.parse(text)
        assert [tc.function.name for tc in tool_calls] == ["a", "b"]

    def test_content_before(self):
        text = f'preface {OPEN_ALL}{OPEN_ONE}x{SEP}{{}}{CLOSE_ONE}'
        content, _ = self.parser.parse(text)
        assert content == "preface"

    def test_alias_deepseek_v31(self):
        p = get_parser("deepseek_v31")
        assert type(p).__name__ == "DeepSeekV31ToolCallParser"
