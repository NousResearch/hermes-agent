import pytest

from cli import _display_width, _display_ljust, _display_wrap


class TestDisplayWidth:
    def test_ascii_string(self):
        assert _display_width("hello") == 5

    def test_empty_string(self):
        assert _display_width("") == 0

    def test_none_input(self):
        assert _display_width(None) == 0

    def test_cjk_characters(self):
        # Each CJK character occupies 2 terminal columns
        assert _display_width("危险") == 4

    def test_mixed_ascii_and_cjk(self):
        # "hi" = 2 columns, "危险" = 4 columns
        assert _display_width("hi危险") == 6

    def test_emoji(self):
        # Exact width depends on get_cwidth implementation; at minimum 1 column
        assert _display_width("⚠️") >= 1

    def test_wider_than_len(self):
        s = "危险命令"
        assert _display_width(s) >= len(s)


class TestDisplayLjust:
    def test_ascii_padding(self):
        result = _display_ljust("hi", 10)
        assert result.startswith("hi")
        assert len(result) == 10

    def test_cjk_padding(self):
        # "危险" has display width 4; padding to 10 means 6 spaces appended
        result = _display_ljust("危险", 10)
        assert _display_width(result) == 10

    def test_already_wide_enough(self):
        result = _display_ljust("hello world", 5)
        assert result == "hello world"

    def test_exact_width(self):
        result = _display_ljust("hello", 5)
        assert result == "hello"


class TestDisplayWrap:
    def test_short_text_no_wrap(self):
        assert _display_wrap("hello world", 20) == ["hello world"]

    def test_wraps_at_width(self):
        lines = _display_wrap("one two three four", 10)
        assert len(lines) > 1
        for line in lines:
            assert _display_width(line) <= 10

    def test_cjk_wraps_correctly(self):
        # Each word ("危险", "命令", "执行") is 4 columns wide; two words = 9 cols
        # which exceeds width=6, so each word should land on its own line
        lines = _display_wrap("危险 命令 执行", 6)
        for line in lines:
            assert _display_width(line) <= 6

    def test_subsequent_indent(self):
        lines = _display_wrap("one two three", 8, subsequent_indent="  ")
        assert len(lines) > 1
        for line in lines[1:]:
            assert line.startswith("  ")

    def test_empty_string(self):
        assert _display_wrap("", 20) == [""]

    def test_minimum_width_clamp(self):
        # Width of 2 is clamped to 8 internally; must not raise
        lines = _display_wrap("hello world", 2)
        assert isinstance(lines, list)
        assert len(lines) >= 1
