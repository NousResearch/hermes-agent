"""Tests for Telegram HTML formatting via TelegramAdapter.format_message().

Covers the markdown-to-HTML conversion pipeline used by the adapter's send
and edit methods, including edge cases that could produce invalid HTML or
corrupt user-visible content.
"""

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
    mod.constants.ParseMode.HTML = "HTML"
    mod.constants.ChatType.GROUP = "group"
    mod.constants.ChatType.SUPERGROUP = "supergroup"
    mod.constants.ChatType.CHANNEL = "channel"
    mod.constants.ChatType.PRIVATE = "private"
    for name in ("telegram", "telegram.ext", "telegram.constants"):
        sys.modules.setdefault(name, mod)


_ensure_telegram_mock()

from gateway.platforms.telegram import TelegramAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def adapter():
    config = PlatformConfig(enabled=True, token="fake-token")
    return TelegramAdapter(config)


# =========================================================================
# format_message - basic conversions
# =========================================================================


class TestFormatMessageBasic:
    def test_empty_string(self, adapter):
        assert adapter.format_message("") == ""

    def test_none_input(self, adapter):
        assert adapter.format_message(None) is None

    def test_plain_text_no_markdown(self, adapter):
        result = adapter.format_message("Hello world")
        assert result == "Hello world"

    def test_html_entities_escaped(self, adapter):
        result = adapter.format_message("a < b > c & d")
        assert "&lt;" in result
        assert "&gt;" in result
        assert "&amp;" in result


# =========================================================================
# format_message - code blocks
# =========================================================================


class TestFormatMessageCodeBlocks:
    def test_fenced_code_block_wrapped_in_pre(self, adapter):
        text = "Before\n```python\nprint('hello')\n```\nAfter"
        result = adapter.format_message(text)
        assert '<pre><code class="language-python">' in result
        assert "</code></pre>" in result
        assert "Before" in result
        assert "After" in result

    def test_inline_code_wrapped(self, adapter):
        text = "Use `my_var` here"
        result = adapter.format_message(text)
        assert "<code>my_var</code>" in result

    def test_code_block_html_escaped(self, adapter):
        text = "```\nif (x > 0) { return !x; }\n```"
        result = adapter.format_message(text)
        assert "&gt;" in result
        assert "<b>" not in result

    def test_inline_code_html_escaped(self, adapter):
        text = "Run `a<b>c` carefully"
        result = adapter.format_message(text)
        assert "<code>a&lt;b&gt;c</code>" in result

    def test_multiple_code_blocks(self, adapter):
        text = "```\nblock1\n```\ntext\n```\nblock2\n```"
        result = adapter.format_message(text)
        assert "block1" in result
        assert "block2" in result
        assert "text" in result


# =========================================================================
# format_message - bold and italic
# =========================================================================


class TestFormatMessageBoldItalic:
    def test_bold_converted(self, adapter):
        result = adapter.format_message("This is **bold** text")
        assert "<b>bold</b>" in result

    def test_italic_asterisk_converted(self, adapter):
        result = adapter.format_message("This is *italic* text")
        assert "<i>italic</i>" in result

    def test_italic_underscore_converted(self, adapter):
        result = adapter.format_message("This is _italic_ text")
        assert "<i>italic</i>" in result

    def test_bold_and_italic_in_same_line(self, adapter):
        result = adapter.format_message("**bold** and *italic*")
        assert "<b>bold</b>" in result
        assert "<i>italic</i>" in result


# =========================================================================
# format_message - headers
# =========================================================================


class TestFormatMessageHeaders:
    def test_h1_converted_to_bold(self, adapter):
        result = adapter.format_message("# Title")
        assert "<b>Title</b>" in result

    def test_h2_converted(self, adapter):
        result = adapter.format_message("## Subtitle")
        assert "<b>Subtitle</b>" in result

    def test_header_with_inner_bold_stripped(self, adapter):
        result = adapter.format_message("## **Important**")
        assert "<b>Important</b>" in result
        # Should not have nested <b> tags
        assert result.count("<b>") == 1

    def test_multiline_headers(self, adapter):
        text = "# First\nSome text\n## Second"
        result = adapter.format_message(text)
        assert "<b>First</b>" in result
        assert "<b>Second</b>" in result
        assert "Some text" in result


# =========================================================================
# format_message - links
# =========================================================================


class TestFormatMessageLinks:
    def test_markdown_link_converted(self, adapter):
        result = adapter.format_message("[Click here](https://example.com)")
        assert '<a href="https://example.com">Click here</a>' in result

    def test_link_with_surrounding_text(self, adapter):
        result = adapter.format_message("Visit [Google](https://google.com) today.")
        assert '<a href="https://google.com">Google</a>' in result
        assert "Visit" in result
        assert "today." in result


# =========================================================================
# format_message - italic does not span newlines
# =========================================================================


class TestItalicNewlineBug:
    r"""Italic regex must not match across newlines."""

    def test_bullet_list_not_corrupted(self, adapter):
        text = "* Item one\n* Item two\n* Item three"
        result = adapter.format_message(text)
        assert "Item one" in result
        assert "Item two" in result
        assert "Item three" in result

    def test_italic_does_not_span_lines(self, adapter):
        text = "Start *across\nlines* end"
        result = adapter.format_message(text)
        assert "<i>across\nlines</i>" not in result

    def test_single_line_italic_still_works(self, adapter):
        text = "This is *italic* text"
        result = adapter.format_message(text)
        assert "<i>italic</i>" in result


# =========================================================================
# format_message - mixed/complex
# =========================================================================


class TestFormatMessageComplex:
    def test_code_block_with_bold_outside(self, adapter):
        text = "**Note:**\n```\ncode here\n```"
        result = adapter.format_message(text)
        assert "<b>" in result
        assert "<pre>" in result

    def test_bold_inside_code_not_converted(self, adapter):
        text = "```\n**not bold**\n```"
        result = adapter.format_message(text)
        assert "<b>" not in result
        assert "**not bold**" in result

    def test_link_inside_code_not_converted(self, adapter):
        text = "`[not a link](url)`"
        result = adapter.format_message(text)
        assert "<a " not in result
        assert "[not a link](url)" in result

    def test_header_after_code_block(self, adapter):
        text = "```\ncode\n```\n## Title"
        result = adapter.format_message(text)
        assert "<b>Title</b>" in result

    def test_multiple_bold_segments(self, adapter):
        result = adapter.format_message("**a** and **b** and **c**")
        assert result.count("<b>") == 3

    def test_empty_bold(self, adapter):
        result = adapter.format_message("****")
        assert result is not None

    def test_empty_code_block(self, adapter):
        result = adapter.format_message("```\n```")
        assert "<pre>" in result

    def test_placeholder_collision(self, adapter):
        text = (
            "# Header\n"
            "**bold1** *italic1* `code1`\n"
            "**bold2** *italic2* `code2`\n"
            "```\nblock\n```\n"
            "[link](https://url.com)"
        )
        result = adapter.format_message(text)
        assert "\x00" not in result
        assert "Header" in result
        assert "block" in result
        assert "url.com" in result

    def test_snake_case_not_italicized(self, adapter):
        result = adapter.format_message("my_variable_name")
        assert "<i>" not in result
        assert "my_variable_name" in result
