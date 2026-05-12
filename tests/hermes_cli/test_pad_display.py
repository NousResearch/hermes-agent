"""Tests for _pad_display — CJK-aware column padding."""

import pytest

from hermes_cli.main import _pad_display


def test_pad_display_ascii_exact():
    """ASCII string at exact width — no padding or truncation."""
    assert _pad_display("Hello", 5) == "Hello"


def test_pad_display_ascii_needs_padding():
    """ASCII string shorter than width — right-padded with spaces."""
    assert _pad_display("Hi", 10) == "Hi        "


def test_pad_display_ascii_needs_truncation():
    """ASCII string longer than width — truncated."""
    assert _pad_display("Hello World!", 5) == "Hello"


def test_pad_display_cjk_exact():
    """CJK-only string at exact display width (2 per char)."""
    # "中文" = 4 display columns
    assert _pad_display("中文", 4) == "中文"


def test_pad_display_cjk_needs_padding():
    """CJK string shorter than width — padded with spaces by display width."""
    result = _pad_display("中文", 8)
    assert result == "中文    "  # 4 display cols of chars + 4 space padding


def test_pad_display_cjk_needs_truncation():
    """CJK string truncated by display width (preserves whole characters)."""
    result = _pad_display("中文测试", 6)
    # "中" (2) + "文" (2) + "测" (2) = 6, "试" would make 8
    assert result == "中文测"


def test_pad_display_mixed_cjk_ascii():
    """Mixed CJK+ASCII — correct display width accounting."""
    # "Hi中文" = 2 + 4 = 6 display columns
    result = _pad_display("Hi中文", 10)
    assert result == "Hi中文    "  # 6 chars + 4 spaces


def test_pad_display_emoji_padding():
    """Emoji (typically 2-column) handled correctly."""
    assert _pad_display("😀", 4) == "😀  "  # emoji = 2 cols, pad 2


def test_pad_display_empty_string():
    """Empty string is padded to full width."""
    assert _pad_display("", 10) == "          "
    assert _pad_display("", 0) == ""


def test_pad_display_wide_edge():
    """Edge: char-by-char truncation lands at the right spot."""
    # "测试" = 4 cols, need 3 cols — must drop last char entirely
    result = _pad_display("测试", 3)
    # "测" = 2 cols, leaving 1 col — can't fit another full-width char
    # So result should be "测 " (2 cols + 1 space) or just "测" if we then pad
    # Actually: cur=4, width=3. cur>width, so loop: cell_len("测试")=4>3 → s="测", cell_len("测")=2 <= 3 → stop. Then width-cur=1, so add 1 space.
    assert result == "测 "


def test_pad_display_zero_width():
    """Zero-width target."""
    assert _pad_display("abc", 0) == ""
    assert _pad_display("中文", 0) == ""


def test_pad_display_negative_becomes_zero():
    """Negative width behaves like zero (guard in caller)."""
    # _pad_display handles this gracefully via range
    assert _pad_display("abc", -1) == ""
