"""Tests for Telegram HTML formatting in gateway/platforms/telegram.py.

Covers: _escape_html (pure function), format_message (markdown-to-HTML
conversion pipeline), _strip_html (plain-text fallback), and edge cases
that could produce invalid HTML or corrupt user-visible content.
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
    mod.constants.ParseMode.HTML = "HTML"
    mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    mod.constants.ChatType.GROUP = "group"
    mod.constants.ChatType.SUPERGROUP = "supergroup"
    mod.constants.ChatType.CHANNEL = "channel"
    mod.constants.ChatType.PRIVATE = "private"
    for name in ("telegram", "telegram.ext", "telegram.constants"):
        sys.modules.setdefault(name, mod)


_ensure_telegram_mock()

from gateway.platforms.telegram import TelegramAdapter, _escape_html, _strip_html  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def adapter():
    config = PlatformConfig(enabled=True, token="fake-token")
    return TelegramAdapter(config)


# =========================================================================
# _escape_html
# =========================================================================


class TestEscapeHtml:
    def test_escapes_ampersand(self):
        assert _escape_html("a & b") == "a &amp; b"

    def test_escapes_angle_brackets(self):
        assert _escape_html("<tag>") == "&lt;tag&gt;"

    def test_empty_string(self):
        assert _escape_html("") == ""

    def test_no_special_characters(self):
        assert _escape_html("hello world 123") == "hello world 123"

    def test_no_double_escaping(self):
        # & must be escaped first; pre-escaped entities should not collapse
        assert _escape_html("&amp;") == "&amp;amp;"

    def test_all_three(self):
        assert _escape_html("a < b & c > d") == "a &lt; b &amp; c &gt; d"

    def test_non_html_specials_unchanged(self):
        # Only &, <, > are special in HTML — everything else passes through
        result = _escape_html("Price is $5.00! (50% off)")
        assert result == "Price is $5.00! (50% off)"


# =========================================================================
# format_message - basic conversions
# =========================================================================


class TestFormatMessageBasic:
    def test_empty_string(self, adapter):
        assert adapter.format_message("") == ""

    def test_none_input(self, adapter):
        # content is falsy, returned as-is
        assert adapter.format_message(None) is None

    def test_plain_text_unchanged(self, adapter):
        assert adapter.format_message("Hello world") == "Hello world"

    def test_angle_brackets_escaped(self, adapter):
        # Literal < and > in prose must become HTML entities
        result = adapter.format_message("x < 5 and y > 3")
        assert "&lt;" in result
        assert "&gt;" in result

    def test_ampersand_escaped(self, adapter):
        result = adapter.format_message("Tom & Jerry")
        assert "&amp;" in result

    def test_dots_and_exclamation_not_escaped(self, adapter):
        # Dots and exclamation marks are not HTML-special
        result = adapter.format_message("Price is $5.00!")
        assert "\\." not in result
        assert "\\!" not in result
        assert "5.00!" in result


# =========================================================================
# format_message - code blocks
# =========================================================================


class TestFormatMessageCodeBlocks:
    def test_fenced_code_block_with_language(self, adapter):
        text = "Before\n```python\nprint('hello')\n```\nAfter"
        result = adapter.format_message(text)
        # Language tag should appear as class attribute on <code>
        assert '<pre><code class="language-python">print(\'hello\')</code></pre>' in result
        assert "After" in result

    def test_fenced_code_block_no_language(self, adapter):
        text = "```\nsome code\n```"
        result = adapter.format_message(text)
        # No language -> plain <pre> without nested <code>
        assert "<pre>some code</pre>" in result

    def test_inline_code(self, adapter):
        result = adapter.format_message("Use `my_var` here")
        # Inline code content must NOT be converted or double-escaped
        assert "<code>my_var</code>" in result

    def test_code_block_html_entities_escaped(self, adapter):
        """HTML-special chars inside code blocks must be entity-escaped."""
        text = "```\nif (x > 0) { return !x; }\n```"
        result = adapter.format_message(text)
        assert "&gt;" in result
        assert "<pre>" in result

    def test_inline_code_html_entities_escaped(self, adapter):
        """HTML-special chars inside inline code must be entity-escaped."""
        text = "Run `a < b & c` carefully"
        result = adapter.format_message(text)
        assert "<code>a &lt; b &amp; c</code>" in result

    def test_multiple_code_blocks(self, adapter):
        text = "```\nblock1\n```\ntext\n```\nblock2\n```"
        result = adapter.format_message(text)
        assert "block1" in result
        assert "block2" in result
        # Text between blocks should be present
        assert "text" in result

    def test_bold_inside_code_not_converted(self, adapter):
        """Bold markers inside code blocks should not be converted."""
        text = "```\n**not bold**\n```"
        result = adapter.format_message(text)
        assert "**not bold**" in result
        assert "<b>" not in result.split("<pre>")[1].split("</pre>")[0]


# =========================================================================
# format_message - bold and italic
# =========================================================================


class TestFormatMessageBoldItalic:
    def test_bold_converted(self, adapter):
        result = adapter.format_message("This is **bold** text")
        assert "<b>bold</b>" in result
        # Original ** markers should be gone
        assert "**" not in result

    def test_italic_converted(self, adapter):
        result = adapter.format_message("This is *italic* text")
        assert "<i>italic</i>" in result

    def test_bold_with_special_chars(self, adapter):
        # Content inside bold should be HTML-escaped
        result = adapter.format_message("**hello<world>**")
        assert "<b>hello&lt;world&gt;</b>" in result

    def test_italic_with_special_chars(self, adapter):
        result = adapter.format_message("*a & b*")
        assert "<i>a &amp; b</i>" in result

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
        # Header becomes bold in HTML
        assert "<b>Title</b>" in result
        # Hash should be removed
        assert "#" not in result

    def test_h2_converted(self, adapter):
        result = adapter.format_message("## Subtitle")
        assert "<b>Subtitle</b>" in result

    def test_header_with_inner_bold_stripped(self, adapter):
        """Headers strip redundant **...** inside to avoid double-wrapping."""
        result = adapter.format_message("## **Important**")
        # Should be <b>Important</b>, not <b><b>Important</b></b>
        assert "<b>Important</b>" in result
        assert result.count("<b>") == 1

    def test_header_with_special_chars(self, adapter):
        result = adapter.format_message("# Hello (World)!")
        assert "<b>Hello (World)!</b>" in result

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

    def test_link_display_text_escaped(self, adapter):
        # The & in display text should be HTML-escaped
        result = adapter.format_message("[A & B](https://example.com)")
        assert "A &amp; B" in result
        assert "<a href" in result

    def test_link_url_escaped(self, adapter):
        # The & in the URL should be HTML-escaped
        result = adapter.format_message("[link](https://example.com?a=1&b=2)")
        assert "&amp;" in result

    def test_link_with_surrounding_text(self, adapter):
        result = adapter.format_message("Visit [Google](https://google.com) today.")
        assert '<a href="https://google.com">Google</a>' in result
        assert "today." in result


# =========================================================================
# format_message - strikethrough
# =========================================================================


class TestFormatMessageStrikethrough:
    def test_strikethrough_converted(self, adapter):
        result = adapter.format_message("~~deleted~~")
        assert "<s>deleted</s>" in result

    def test_strikethrough_with_special_chars(self, adapter):
        result = adapter.format_message("~~a < b~~")
        assert "<s>a &lt; b</s>" in result


# =========================================================================
# format_message - blockquotes
# =========================================================================


class TestFormatMessageBlockquotes:
    def test_single_line_blockquote(self, adapter):
        result = adapter.format_message("> Hello world")
        assert "<blockquote>Hello world</blockquote>" in result

    def test_multi_line_blockquote(self, adapter):
        """Consecutive > lines should merge into a single <blockquote>."""
        text = "> Line one\n> Line two\n> Line three"
        result = adapter.format_message(text)
        assert "<blockquote>" in result
        assert "Line one\nLine two\nLine three" in result
        assert result.count("<blockquote>") == 1

    def test_blockquote_with_special_chars(self, adapter):
        result = adapter.format_message("> a < b & c > d")
        assert "<blockquote>" in result
        assert "&lt;" in result
        assert "&amp;" in result


# =========================================================================
# format_message - horizontal rules
# =========================================================================


class TestFormatMessageHorizontalRules:
    def test_triple_dash(self, adapter):
        # --- on its own line -> em-dash separator (no native HR in Telegram)
        result = adapter.format_message("above\n---\nbelow")
        assert "\u2014\u2014\u2014" in result
        assert "---" not in result

    def test_triple_asterisk(self, adapter):
        result = adapter.format_message("above\n***\nbelow")
        assert "\u2014\u2014\u2014" in result

    def test_triple_underscore(self, adapter):
        result = adapter.format_message("above\n___\nbelow")
        assert "\u2014\u2014\u2014" in result


# =========================================================================
# format_message - tables
# =========================================================================


class TestFormatMessageTables:
    def test_simple_table(self, adapter):
        """Markdown tables should be wrapped in <pre> for monospace alignment."""
        text = "| A | B |\n|---|---|\n| 1 | 2 |"
        result = adapter.format_message(text)
        assert "<pre>" in result
        assert "| A | B |" in result
        assert "</pre>" in result

    def test_table_with_special_chars(self, adapter):
        """HTML-special chars inside tables must be entity-escaped."""
        text = "| A & B | C |\n|---|---|\n| <x> | y |"
        result = adapter.format_message(text)
        assert "<pre>" in result
        assert "&amp;" in result
        assert "&lt;x&gt;" in result


# =========================================================================
# format_message - italic must not span newlines
# =========================================================================


class TestItalicNewlineBug:
    r"""The italic regex uses [^*\n]+ to prevent matching across newlines.

    Without this restriction, bullet lists using * markers and any text
    where * appears at the end of one line and start of another would be
    incorrectly wrapped as italic.
    """

    def test_bullet_list_not_corrupted(self, adapter):
        """Bullet list items using * must NOT be merged into italic."""
        text = "* Item one\n* Item two\n* Item three"
        result = adapter.format_message(text)
        # Each item should appear in the output (not eaten by italic conversion)
        assert "Item one" in result
        assert "Item two" in result
        assert "Item three" in result

    def test_asterisk_list_items_preserved(self, adapter):
        """Each * list item should remain as a separate line, not become italic."""
        text = "* Alpha\n* Beta"
        result = adapter.format_message(text)
        assert "Alpha" in result
        assert "Beta" in result
        lines = result.split("\n")
        assert len(lines) >= 2

    def test_italic_does_not_span_lines(self, adapter):
        """*text on\\nmultiple lines* should NOT become italic."""
        text = "Start *across\nlines* end"
        result = adapter.format_message(text)
        # Should NOT have <i> wrapping cross-line text
        assert "<i>across\nlines</i>" not in result

    def test_single_line_italic_still_works(self, adapter):
        """Normal single-line italic must still convert correctly."""
        result = adapter.format_message("This is *italic* text")
        assert "<i>italic</i>" in result


# =========================================================================
# format_message - lists (pass-through, no conversion needed)
# =========================================================================


class TestFormatMessageLists:
    def test_dash_list_preserved(self, adapter):
        """Dash lists pass through as plain text — Telegram renders them fine."""
        text = "- Item one\n- Item two"
        result = adapter.format_message(text)
        assert "- Item one" in result
        assert "- Item two" in result

    def test_numbered_list_preserved(self, adapter):
        """Numbered lists pass through as plain text."""
        text = "1. First\n2. Second"
        result = adapter.format_message(text)
        assert "1. First" in result
        assert "2. Second" in result


# =========================================================================
# format_message - chunked digest content
#
# Long messages (e.g. news digests) are chunked before formatting. Each
# chunk must produce valid HTML independently, including chunk indicators
# like (1/2) that are appended by the base chunker.
# =========================================================================


class TestChunkedDigestContent:
    def test_digest_chunks_format_independently(self, adapter):
        """A digest-style message that exceeds 4096 chars should produce
        valid HTML in every chunk."""
        # Build a realistic digest that exceeds the chunk limit
        items = []
        for i in range(30):
            items.append(
                f"- [Article title number {i} with enough length to be realistic]"
                f"(https://www.example.com/news/longer-article-slug-{i}-detail-{i*11}0784.html)"
                f" \u2014 Summary of article {i} with additional context and description."
            )
        text = (
            "## Monday, 14.03.2026\n\n"
            "**Security & Privacy**\n"
            + "\n".join(items[:8])
            + "\n\n**Tech & Business**\n"
            + "\n".join(items[8:16])
            + "\n\n**Science & Culture**\n"
            + "\n".join(items[16:])
            + "\n\n*25 articles total.*"
        )
        assert len(text) > 4096, "test input must exceed Telegram's limit"

        chunks = adapter.truncate_message(
            text, adapter.MAX_MESSAGE_LENGTH - adapter.HTML_OVERHEAD_RESERVE
        )
        assert len(chunks) >= 2

        for i, chunk in enumerate(chunks):
            formatted = adapter.format_message(chunk)
            assert "\x00" not in formatted, f"chunk {i} leaked placeholders"
            # Every chunk should contain valid HTML, not raw markdown artifacts
            assert "\\" not in formatted, f"chunk {i} has backslash escapes"

    def test_digest_section_headers_and_links(self, adapter):
        """Bold section headers followed by link list items — the typical
        pattern in digest output — should produce clean HTML."""
        text = (
            "**Security & Privacy**\n"
            "- [Update released for browser](https://example.com/news/update-1234.html)"
            " \u2014 Vendor follows up after initial fix was insufficient.\n"
            "- [Hidden fees result in fine](https://example.com/news/fees-5678.html)"
            " \u2014 Company settles lawsuit.\n"
            "\n**Tech & Business**\n"
            "- [Cloud service turns 20](https://example.com/opinion/cloud-20-years.html)"
            " \u2014 Industry standard and vendor lock-in."
        )
        result = adapter.format_message(text)
        assert "<b>Security &amp; Privacy</b>" in result
        assert "<b>Tech &amp; Business</b>" in result
        assert result.count("<a href") == 3
        assert "\x00" not in result

    def test_digest_italic_trailing_line(self, adapter):
        """Italic article count at the end of a digest — the italic regex
        must not be disrupted by preceding link-heavy content."""
        text = (
            "- [First item](https://example.com/1.html) \u2014 Summary.\n"
            "- [Second item](https://example.com/2.html) \u2014 Details.\n"
            "\n*12 articles total.*"
        )
        result = adapter.format_message(text)
        assert "<i>12 articles total.</i>" in result
        assert result.count("<a href") == 2
        assert "\x00" not in result


# =========================================================================
# format_message - mixed/complex scenarios
# =========================================================================


class TestFormatMessageComplex:
    def test_code_block_with_bold_outside(self, adapter):
        text = "**Note:**\n```\ncode here\n```"
        result = adapter.format_message(text)
        assert "<b>" in result
        assert "<pre>" in result

    def test_link_inside_code_not_converted(self, adapter):
        """Link syntax inside inline code should not become an <a> tag."""
        text = "`[not a link](url)`"
        result = adapter.format_message(text)
        assert "<code>" in result
        assert "<a href" not in result

    def test_header_after_code_block(self, adapter):
        text = "```\ncode\n```\n## Title"
        result = adapter.format_message(text)
        assert "<b>Title</b>" in result
        assert "<pre>" in result

    def test_multiple_bold_segments(self, adapter):
        result = adapter.format_message("**a** and **b** and **c**")
        assert result.count("<b>") == 3
        assert result.count("</b>") == 3

    def test_empty_bold(self, adapter):
        """**** (empty bold) should not crash."""
        result = adapter.format_message("****")
        assert result is not None

    def test_empty_code_block(self, adapter):
        result = adapter.format_message("```\n```")
        assert "<pre>" in result

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


# =========================================================================
# _strip_html — plaintext fallback
# =========================================================================


class TestStripHtml:
    def test_strips_bold_tags(self):
        assert _strip_html("<b>bold text</b>") == "bold text"

    def test_strips_italic_tags(self):
        assert _strip_html("<i>italic</i>") == "italic"

    def test_strips_pre_tags(self):
        assert _strip_html("<pre>code</pre>") == "code"

    def test_strips_nested_tags(self):
        assert _strip_html('<pre><code class="language-py">x</code></pre>') == "x"

    def test_unescapes_html_entities(self):
        # Entity reversal must restore original characters
        assert _strip_html("a &lt; b &amp; c &gt; d") == "a < b & c > d"

    def test_plain_text_unchanged(self):
        assert _strip_html("plain text") == "plain text"

    def test_empty_string(self):
        assert _strip_html("") == ""

    def test_strips_link_tags(self):
        assert _strip_html('<a href="https://x.com">link</a>') == "link"
