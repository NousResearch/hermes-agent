"""Tests for Telegram MarkdownV2 formatting in gateway/platforms/telegram.py.

Covers: _escape_mdv2 (pure function), format_message (markdown-to-MarkdownV2
conversion pipeline), and edge cases that could produce invalid MarkdownV2
or corrupt user-visible content.
"""

import re
import sys
from unittest.mock import MagicMock

import pytest

from gateway.config import PlatformConfig


# ---------------------------------------------------------------------------
# Mock the telegram package if it's not installed
# ---------------------------------------------------------------------------

def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    mod = MagicMock()
    mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    mod.constants.ChatType.GROUP = "group"
    mod.constants.ChatType.SUPERGROUP = "supergroup"
    mod.constants.ChatType.CHANNEL = "channel"
    mod.constants.ChatType.PRIVATE = "private"
    for name in ("telegram", "telegram.ext", "telegram.constants"):
        sys.modules.setdefault(name, mod)


_ensure_telegram_mock()

from gateway.platforms.telegram import TelegramAdapter, _escape_mdv2  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def adapter():
    config = PlatformConfig(enabled=True, token="fake-token")
    return TelegramAdapter(config)


# =========================================================================
# _escape_mdv2
# =========================================================================


class TestEscapeMdv2:
    def test_escapes_all_special_characters(self):
        special = r'_*[]()~`>#+-=|{}.!\ '
        escaped = _escape_mdv2(special)
        # Every special char should be preceded by backslash
        for ch in r'_*[]()~`>#+-=|{}.!\  ':
            if ch == ' ':
                continue
            assert f'\\{ch}' in escaped

    def test_empty_string(self):
        assert _escape_mdv2("") == ""

    def test_no_special_characters(self):
        assert _escape_mdv2("hello world 123") == "hello world 123"

    def test_backslash_escaped(self):
        assert _escape_mdv2("a\\b") == "a\\\\b"

    def test_dot_escaped(self):
        assert _escape_mdv2("v2.0") == "v2\\.0"

    def test_exclamation_escaped(self):
        assert _escape_mdv2("wow!") == "wow\\!"

    def test_mixed_text_and_specials(self):
        result = _escape_mdv2("Hello (world)!")
        assert result == "Hello \\(world\\)\\!"


# =========================================================================
# format_message - basic conversions
# =========================================================================


class TestFormatMessageBasic:
    def test_empty_string(self, adapter):
        assert adapter.format_message("") == ""

    def test_none_input(self, adapter):
        # content is falsy, returned as-is
        assert adapter.format_message(None) is None

    def test_plain_text_specials_escaped(self, adapter):
        result = adapter.format_message("Price is $5.00!")
        assert "\\." in result
        assert "\\!" in result

    def test_plain_text_no_markdown(self, adapter):
        result = adapter.format_message("Hello world")
        assert result == "Hello world"


# =========================================================================
# format_message - code blocks
# =========================================================================


class TestFormatMessageCodeBlocks:
    def test_fenced_code_block_preserved(self, adapter):
        text = "Before\n```python\nprint('hello')\n```\nAfter"
        result = adapter.format_message(text)
        # Code block contents must NOT be escaped
        assert "```python\nprint('hello')\n```" in result
        # But "After" should have no escaping needed (plain text)
        assert "After" in result

    def test_inline_code_preserved(self, adapter):
        text = "Use `my_var` here"
        result = adapter.format_message(text)
        # Inline code content must NOT be escaped
        assert "`my_var`" in result
        # The surrounding text's underscore-free content should be fine
        assert "Use" in result

    def test_code_block_special_chars_not_escaped(self, adapter):
        text = "```\nif (x > 0) { return !x; }\n```"
        result = adapter.format_message(text)
        # Inside code block, > and ! and { should NOT be escaped
        assert "if (x > 0) { return !x; }" in result

    def test_inline_code_special_chars_not_escaped(self, adapter):
        text = "Run `rm -rf ./*` carefully"
        result = adapter.format_message(text)
        assert "`rm -rf ./*`" in result

    def test_multiple_code_blocks(self, adapter):
        text = "```\nblock1\n```\ntext\n```\nblock2\n```"
        result = adapter.format_message(text)
        assert "block1" in result
        assert "block2" in result
        # "text" between blocks should be present
        assert "text" in result


# =========================================================================
# format_message - bold and italic
# =========================================================================


class TestFormatMessageBoldItalic:
    def test_bold_converted(self, adapter):
        result = adapter.format_message("This is **bold** text")
        # MarkdownV2 bold uses single *
        assert "*bold*" in result
        # Original ** should be gone
        assert "**" not in result

    def test_italic_converted(self, adapter):
        result = adapter.format_message("This is *italic* text")
        # MarkdownV2 italic uses _
        assert "_italic_" in result

    def test_bold_with_special_chars(self, adapter):
        result = adapter.format_message("**hello.world!**")
        # Content inside bold should be escaped
        assert "*hello\\.world\\!*" in result

    def test_italic_with_special_chars(self, adapter):
        result = adapter.format_message("*hello.world*")
        assert "_hello\\.world_" in result

    def test_bold_and_italic_in_same_line(self, adapter):
        result = adapter.format_message("**bold** and *italic*")
        assert "*bold*" in result
        assert "_italic_" in result


# =========================================================================
# format_message - headers
# =========================================================================


class TestFormatMessageHeaders:
    def test_h1_converted_to_bold(self, adapter):
        result = adapter.format_message("# Title")
        # Header becomes bold in MarkdownV2
        assert "*Title*" in result
        # Hash should be removed
        assert "#" not in result

    def test_h2_converted(self, adapter):
        result = adapter.format_message("## Subtitle")
        assert "*Subtitle*" in result

    def test_header_with_inner_bold_stripped(self, adapter):
        # Headers strip redundant **...** inside
        result = adapter.format_message("## **Important**")
        # Should be *Important* not ***Important***
        assert "*Important*" in result
        count = result.count("*")
        # Should have exactly 2 asterisks (open + close)
        assert count == 2

    def test_header_with_special_chars(self, adapter):
        result = adapter.format_message("# Hello (World)!")
        assert "\\(" in result
        assert "\\)" in result
        assert "\\!" in result

    def test_multiline_headers(self, adapter):
        text = "# First\nSome text\n## Second"
        result = adapter.format_message(text)
        assert "*First*" in result
        assert "*Second*" in result
        assert "Some text" in result


# =========================================================================
# format_message - links
# =========================================================================


class TestFormatMessageLinks:
    def test_markdown_link_converted(self, adapter):
        result = adapter.format_message("[Click here](https://example.com)")
        assert "[Click here](https://example.com)" in result

    def test_link_display_text_escaped(self, adapter):
        result = adapter.format_message("[Hello!](https://example.com)")
        # The ! in display text should be escaped
        assert "Hello\\!" in result

    def test_link_url_parentheses_escaped(self, adapter):
        result = adapter.format_message("[link](https://example.com/path_(1))")
        # The ) in URL should be escaped
        assert "\\)" in result

    def test_link_with_surrounding_text(self, adapter):
        result = adapter.format_message("Visit [Google](https://google.com) today.")
        assert "[Google](https://google.com)" in result
        assert "today\\." in result


# =========================================================================
# format_message - BUG: italic regex spans newlines
# =========================================================================


class TestItalicNewlineBug:
    r"""Italic regex ``\*([^*]+)\*`` matched across newlines, corrupting content.

    This affects bullet lists using * markers and any text where * appears
    at the end of one line and start of another.
    """

    def test_bullet_list_not_corrupted(self, adapter):
        """Bullet list items using * must NOT be merged into italic."""
        text = "* Item one\n* Item two\n* Item three"
        result = adapter.format_message(text)
        # Each item should appear in the output (not eaten by italic conversion)
        assert "Item one" in result
        assert "Item two" in result
        assert "Item three" in result
        # Should NOT contain _ (italic markers) wrapping list items
        assert "_" not in result or "Item" not in result.split("_")[1] if "_" in result else True

    def test_asterisk_list_items_preserved(self, adapter):
        """Each * list item should remain as a separate line, not become italic."""
        text = "* Alpha\n* Beta"
        result = adapter.format_message(text)
        # Both items must be present in output
        assert "Alpha" in result
        assert "Beta" in result
        # The text between first * and second * must NOT become italic
        lines = result.split("\n")
        assert len(lines) >= 2

    def test_italic_does_not_span_lines(self, adapter):
        """*text on\nmultiple lines* should NOT become italic."""
        text = "Start *across\nlines* end"
        result = adapter.format_message(text)
        # Should NOT have underscore italic markers wrapping cross-line text
        # If this fails, the italic regex is matching across newlines
        assert "_across\nlines_" not in result

    def test_single_line_italic_still_works(self, adapter):
        """Normal single-line italic must still convert correctly."""
        text = "This is *italic* text"
        result = adapter.format_message(text)
        assert "_italic_" in result


# =========================================================================
# format_message - mixed/complex
# =========================================================================


class TestFormatMessageComplex:
    def test_code_block_with_bold_outside(self, adapter):
        text = "**Note:**\n```\ncode here\n```"
        result = adapter.format_message(text)
        assert "*Note:*" in result or "*Note\\:*" in result
        assert "```\ncode here\n```" in result

    def test_bold_inside_code_not_converted(self, adapter):
        """Bold markers inside code blocks should not be converted."""
        text = "```\n**not bold**\n```"
        result = adapter.format_message(text)
        assert "**not bold**" in result

    def test_link_inside_code_not_converted(self, adapter):
        text = "`[not a link](url)`"
        result = adapter.format_message(text)
        assert "`[not a link](url)`" in result

    def test_header_after_code_block(self, adapter):
        text = "```\ncode\n```\n## Title"
        result = adapter.format_message(text)
        assert "*Title*" in result
        assert "```\ncode\n```" in result

    def test_multiple_bold_segments(self, adapter):
        result = adapter.format_message("**a** and **b** and **c**")
        assert result.count("*") >= 6  # 3 bold pairs = 6 asterisks

    def test_special_chars_in_plain_text(self, adapter):
        result = adapter.format_message("Price: $5.00 (50% off!)")
        assert "\\." in result
        assert "\\(" in result
        assert "\\)" in result
        assert "\\!" in result

    def test_empty_bold(self, adapter):
        """**** (empty bold) should not crash."""
        result = adapter.format_message("****")
        assert result is not None

    def test_empty_code_block(self, adapter):
        result = adapter.format_message("```\n```")
        assert "```" in result

    def test_placeholder_collision(self, adapter):
        """Many formatting elements should not cause placeholder collisions."""
        text = (
            "# Header\n"
            "**bold1** *italic1* `code1`\n"
            "**bold2** *italic2* `code2`\n"
            "```\nblock\n```\n"
            "[link](https://url.com)"
        )
        result = adapter.format_message(text)
        # No placeholder tokens should leak into output
        assert "\x00" not in result
        # All elements should be present
        assert "Header" in result
        assert "block" in result
        assert "url.com" in result
