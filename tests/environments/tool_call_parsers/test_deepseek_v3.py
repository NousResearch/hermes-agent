"""Tests for DeepSeekV3ToolCallParser — unicode-token bracketed format."""
from __future__ import annotations

from environments.tool_call_parsers import get_parser

from .conftest import args_as_dict


OPEN_ALL = "<｜tool▁calls▁begin｜>"
CLOSE_ALL = "<｜tool▁calls▁end｜>"
OPEN_ONE = "<｜tool▁call▁begin｜>"
CLOSE_ONE = "<｜tool▁call▁end｜>"
SEP = "<｜tool▁sep｜>"


def _call(name: str, args_json: str, type_="function"):
    return (
        f"{OPEN_ONE}{type_}{SEP}{name}\n"
        f"```json\n{args_json}\n```\n"
        f"{CLOSE_ONE}"
    )


class TestDeepSeekV3Parser:
    def setup_method(self):
        self.parser = get_parser("deepseek_v3")

    def test_no_start_token(self):
        content, tool_calls = self.parser.parse("plain response")
        assert content == "plain response"
        assert tool_calls is None

    def test_single_tool_call(self):
        text = f"{OPEN_ALL}\n" + _call("get_weather", '{"city": "NYC"}') + f"\n{CLOSE_ALL}"
        content, tool_calls = self.parser.parse(text)
        assert tool_calls[0].function.name == "get_weather"
        assert args_as_dict(tool_calls[0]) == {"city": "NYC"}

    def test_multiple_tool_calls(self):
        text = (
            f"{OPEN_ALL}\n"
            + _call("a", "{}")
            + "\n"
            + _call("b", '{"y": 2}')
            + f"\n{CLOSE_ALL}"
        )
        _, tool_calls = self.parser.parse(text)
        assert [tc.function.name for tc in tool_calls] == ["a", "b"]
        assert args_as_dict(tool_calls[1]) == {"y": 2}

    def test_content_before(self):
        text = (
            "thinking... "
            + f"{OPEN_ALL}\n"
            + _call("x", "{}")
            + f"\n{CLOSE_ALL}"
        )
        content, _ = self.parser.parse(text)
        assert content == "thinking..."

    def test_unmatched_begin_token_falls_through(self):
        text = f"{OPEN_ALL} but no call blocks"
        content, tool_calls = self.parser.parse(text)
        assert tool_calls is None
        assert content == text
