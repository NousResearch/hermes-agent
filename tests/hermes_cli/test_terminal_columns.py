"""Tests for hermes_cli/terminal_columns.py (CJK-aware width calculation)."""

import pytest

from hermes_cli.terminal_columns import disp_width, truncate_for_width, pad_left, pad_right


class TestDispWidth:
    """Tests for display width calculation."""

    def test_ascii(self):
        """ASCII characters have width 1 each."""
        assert disp_width("Hello") == 5
        assert disp_width("123abc") == 6

    def test_cjk_chinese(self):
        """Chinese characters have width 2 each."""
        assert disp_width("中文") == 4
        assert disp_width("你好世界") == 8

    def test_cjk_japanese(self):
        """Japanese characters have width 2 each."""
        assert disp_width("日本語") == 6
        assert disp_width("こんにちは") == 10

    def test_cjk_korean(self):
        """Korean characters have width 2 each."""
        assert disp_width("한글") == 4

    def test_mixed_ascii_cjk(self):
        """Mixed ASCII and CJK content."""
        assert disp_width("Hello世界") == 9  # 5 + 4
        assert disp_width("测试Test") == 8  # 4 + 4

    def test_emoji(self):
        """Emoji have width 2 (most common emojis)."""
        assert disp_width("👍") == 2
        assert disp_width("🎉🎊") == 4

    def test_empty_string(self):
        """Empty string has width 0."""
        assert disp_width("") == 0

    def test_control_characters(self):
        """Control characters return 0 width (handled by wcswidth returning -1)."""
        # wcswidth returns -1 for control chars, we clamp to 0
        assert disp_width("\x00") == 0
        assert disp_width("\x1b") == 0


class TestTruncateForWidth:
    """Tests for string truncation."""

    def test_no_truncation_needed(self):
        """String fits within width, no truncation."""
        assert truncate_for_width("Hello", 10) == "Hello"
        assert truncate_for_width("中文", 10) == "中文"

    def test_truncate_ascii(self):
        """Truncate ASCII string."""
        result = truncate_for_width("Hello World", 8)
        assert disp_width(result) <= 8
        assert result.endswith("…")

    def test_truncate_cjk(self):
        """Truncate CJK string."""
        result = truncate_for_width("这是一个很长的中文句子", 10)
        assert disp_width(result) <= 10
        assert result.endswith("…")

    def test_truncate_mixed(self):
        """Truncate mixed ASCII/CJK string."""
        result = truncate_for_width("Hello世界测试Test", 12)
        assert disp_width(result) <= 12
        assert result.endswith("…")

    def test_truncate_preserves_chars(self):
        """Truncation does not cut characters in half."""
        # Chinese chars are 2-width each
        result = truncate_for_width("中文测试", 5)
        # Should be "中文…" (2 + 2 + 1 = 5), fitting within max_width=5
        # Not "中" + half of "文"
        assert result == "中文…"

    def test_ellipsis_width(self):
        """Result width does not exceed max_width."""
        for s in ["这是一个很长的中文句子用于测试", "Very long English sentence for testing"]:
            for width in [5, 10, 15, 20]:
                result = truncate_for_width(s, width)
                assert disp_width(result) <= width, f"Width {width}, result '{result}'"

    def test_zero_width(self):
        """Zero max_width returns empty or minimal."""
        result = truncate_for_width("Hello", 0)
        assert disp_width(result) <= 0

    def test_custom_ellipsis(self):
        """Custom ellipsis character."""
        result = truncate_for_width("这是一个测试", 8, ellipsis="...")
        assert result.endswith("...")

    def test_exact_fit_no_ellipsis(self):
        """String exactly fits width, no ellipsis."""
        # "中文" = 4 width
        result = truncate_for_width("中文", 4)
        assert result == "中文"
        assert "…" not in result


class TestPadLeft:
    """Tests for right-alignment (pad left)."""

    def test_pad_ascii(self):
        """Pad ASCII string on left."""
        result = pad_left("Hi", 5)
        assert result == "   Hi"
        assert len(result) == 5

    def test_pad_cjk(self):
        """Pad CJK string on left."""
        result = pad_left("中文", 6)
        assert result == "  中文"
        assert disp_width(result) == 6

    def test_no_pad_needed(self):
        """String already at width."""
        result = pad_left("Hello", 5)
        assert result == "Hello"

    def test_truncate_and_pad(self):
        """Truncate then pad if string too wide."""
        result = pad_left("这是一个长句子", 8)
        assert disp_width(result) <= 8

    def test_mixed_content(self):
        """Pad mixed ASCII/CJK content."""
        result = pad_left("Hi世界", 8)
        assert disp_width(result) == 8  # 2 + 4 + 2 spaces


class TestPadRight:
    """Tests for left-alignment (pad right)."""

    def test_pad_ascii(self):
        """Pad ASCII string on right."""
        result = pad_right("Hi", 5)
        assert result == "Hi   "
        assert len(result) == 5

    def test_pad_cjk(self):
        """Pad CJK string on right."""
        result = pad_right("中文", 6)
        assert result == "中文  "
        assert disp_width(result) == 6

    def test_no_pad_needed(self):
        """String already at width."""
        result = pad_right("Hello", 5)
        assert result == "Hello"

    def test_truncate_and_pad(self):
        """Truncate then pad if string too wide."""
        result = pad_right("这是一个长句子", 8)
        assert disp_width(result) <= 8

    def test_mixed_content(self):
        """Pad mixed ASCII/CJK content."""
        result = pad_right("Hi世界", 8)
        assert disp_width(result) == 8

    def test_sessions_list_cjk_title_truncation(self):
        """Sessions list truncates CJK titles without cutting chars in half.

        Regression test: narrow terminals should truncate long CJK session
        titles correctly, preserving character boundaries.
        """
        # Simulate narrow terminal column widths (60 cols → col_title ≈ 20)
        col_title = 20
        long_cjk_title = "中文测试标题这是一个很长的标题用于测试截断效果"

        result = pad_right(long_cjk_title, col_title)

        # Width must not exceed the column limit
        assert disp_width(result) <= col_title, f"Result '{result}' exceeds width {col_title}"

        # Must contain ellipsis (truncated) and not cut any character in half
        # Note: pad_right may add trailing spaces, so check for '…' presence
        assert "…" in result, f"Expected truncation ellipsis, got '{result}'"

        # No partial characters should appear (valid UTF-8)
        result.encode("utf-8").decode("utf-8")  # raises if invalid

    def test_sessions_list_cjk_preview_alignment(self):
        """Sessions list aligns CJK previews correctly with ASCII columns."""
        col_preview = 15
        cjk_preview = "这是一个中文预览内容"

        result = pad_right(cjk_preview, col_preview)

        # Should be truncated and fit within column
        assert disp_width(result) <= col_preview

        # If truncated, must end with ellipsis
        if disp_width(cjk_preview) > col_preview:
            assert result.endswith("…")

    def test_sessions_list_mixed_row_alignment(self):
        """Full table row with mixed CJK/ASCII aligns correctly."""
        col_title = 18
        col_preview = 12
        col_last_active = 13

        title = pad_right("中文标题English混合", col_title)
        preview = pad_right("预览Preview", col_preview)
        last_active = pad_right("2h ago", col_last_active)

        # All columns should fit their widths
        assert disp_width(title) <= col_title
        assert disp_width(preview) <= col_preview
        assert disp_width(last_active) <= col_last_active


class TestIntegration:
    """Integration tests for table formatting scenarios."""

    def test_table_row_alignment(self):
        """Simulate a table row with CJK content."""
        col_title = 20
        col_preview = 15

        title = pad_right("这是一个中文标题", col_title)
        preview = pad_right("预览内容", col_preview)

        assert disp_width(title) <= col_title
        assert disp_width(preview) <= col_preview

    def test_long_mixed_content(self):
        """Handle long mixed content gracefully."""
        s = "This is a 混合mixed 内容content string with 各种characters"
        result = pad_right(s, 30)
        assert disp_width(result) <= 30