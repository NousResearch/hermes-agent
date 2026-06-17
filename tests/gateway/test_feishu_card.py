"""Tests for gateway.platforms.feishu_card."""
from __future__ import annotations
import pytest
from gateway.platforms.feishu_card import format_token_count

class TestFormatTokenCount:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (0, "0"),
            (48, "48"),
            (999, "999"),
            (1000, "1.0k"),
            (1100, "1.1k"),
            (11100, "11.1k"),
            (999999, "1000.0k"),
            (1000000, "1.0M"),
            (3400000, "3.4M"),
        ],
    )
    def test_format_token_count(self, value, expected):
        assert format_token_count(value) == expected

    def test_format_token_count_negative_returns_zero(self):
        assert format_token_count(-5) == "0"


from gateway.platforms.feishu_card import get_tool_display

class TestGetToolDisplay:
    @pytest.mark.parametrize(
        "tool_name, expected",
        [
            ("Read", "Read · 阅读文件"),
            ("read_file", "Read · 阅读文件"),
            ("Bash", "Bash · 执行命令"),
            ("terminal", "Bash · 执行命令"),
            ("Edit", "Edit · 改代码"),
            ("edit_file", "Edit · 改代码"),
            ("Write", "Write · 写文件"),
            ("write_file", "Write · 写文件"),
            ("MultiEdit", "MultiEdit · 批量改代码"),
            ("Grep", "Grep · 搜索代码"),
            ("Glob", "Glob · 查找文件"),
            ("WebFetch", "WebFetch · 抓取网页"),
            ("web_fetch", "WebFetch · 抓取网页"),
            ("WebSearch", "WebSearch · 搜索网络"),
            ("web_search", "WebSearch · 搜索网络"),
            ("Task", "Agent · 派出子任务"),
            ("Agent", "Agent · 派出子任务"),
            ("TodoWrite", "TodoWrite · 更新任务"),
            ("unknown_tool_xyz", "unknown_tool_xyz"),
        ],
    )
    def test_get_tool_display(self, tool_name, expected):
        assert get_tool_display(tool_name) == expected
