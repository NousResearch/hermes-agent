"""Tests for Telegram HTML formatting helpers in gateway/platforms/telegram.py."""

import sys
from unittest.mock import MagicMock


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

from gateway.platforms.telegram import (  # noqa: E402
    _markdown_to_telegram_html as markdown_to_telegram_html,
    _split_html_chunks as split_html_chunks,
    _strip_html_tags as strip_html_tags,
)


class TestFencedCodeBlocks:
    def test_basic_fenced_block(self):
        md = "```\nprint('hi')\n```"
        result = markdown_to_telegram_html(md)
        assert "<pre>" in result
        assert "print(" in result
        assert "</pre>" in result

    def test_fenced_block_with_language(self):
        md = "```python\nprint('hi')\n```"
        result = markdown_to_telegram_html(md)
        assert '<pre><code class="language-python">' in result
        assert "print(" in result
        assert "</code></pre>" in result

    def test_fenced_block_escapes_html(self):
        md = "```\n<div>a & b</div>\n```"
        result = markdown_to_telegram_html(md)
        assert "&lt;div&gt;" in result
        assert "&amp;" in result

    def test_fenced_block_preserves_markdown(self):
        md = "```\n**not bold** and *not italic*\n```"
        result = markdown_to_telegram_html(md)
        assert "<b>" not in result
        assert "<i>" not in result
        assert "**not bold**" in result

    def test_multiple_code_blocks(self):
        md = "Before\n```\nblock1\n```\nMiddle\n```\nblock2\n```\nAfter"
        result = markdown_to_telegram_html(md)
        assert result.count("<pre>") == 2
        assert result.count("</pre>") == 2
        assert "Before" in result
        assert "Middle" in result
        assert "After" in result


class TestInlineCode:
    def test_inline_code(self):
        assert markdown_to_telegram_html("use `foo()`") == "use <code>foo()</code>"

    def test_inline_code_escapes_html(self):
        result = markdown_to_telegram_html("use `a<b>c`")
        assert result == "use <code>a&lt;b&gt;c</code>"

    def test_inline_code_not_processed_for_formatting(self):
        result = markdown_to_telegram_html("`**bold**`")
        assert "<b>" not in result
        assert "**bold**" in result


class TestBold:
    def test_bold(self):
        assert markdown_to_telegram_html("**hello**") == "<b>hello</b>"

    def test_bold_mid_sentence(self):
        result = markdown_to_telegram_html("this is **important** stuff")
        assert result == "this is <b>important</b> stuff"

    def test_multiple_bold(self):
        result = markdown_to_telegram_html("**a** and **b**")
        assert result == "<b>a</b> and <b>b</b>"


class TestItalic:
    def test_italic_asterisk(self):
        assert markdown_to_telegram_html("*hello*") == "<i>hello</i>"

    def test_italic_underscore(self):
        assert markdown_to_telegram_html("_hello_") == "<i>hello</i>"

    def test_no_italic_in_snake_case(self):
        result = markdown_to_telegram_html("my_variable_name")
        assert "<i>" not in result
        assert "my_variable_name" in result

    def test_no_italic_mid_word(self):
        result = markdown_to_telegram_html("some*thing*here")
        assert "<i>" not in result


class TestStrikethrough:
    def test_strikethrough(self):
        assert markdown_to_telegram_html("~~deleted~~") == "<s>deleted</s>"

    def test_strikethrough_mid_sentence(self):
        result = markdown_to_telegram_html("this is ~~wrong~~ right")
        assert result == "this is <s>wrong</s> right"


class TestLinks:
    def test_basic_link(self):
        result = markdown_to_telegram_html("[click](https://example.com)")
        assert result == '<a href="https://example.com">click</a>'

    def test_link_with_special_chars(self):
        result = markdown_to_telegram_html("[a&b](https://example.com?a=1&b=2)")
        assert "a&amp;b" in result
        assert "a=1&amp;b=2" in result

    def test_link_display_text_not_formatted(self):
        result = markdown_to_telegram_html("[**bold link**](https://example.com)")
        # The display text should be escaped, not formatted
        assert "**bold link**" in result


class TestHeaders:
    def test_h1(self):
        assert markdown_to_telegram_html("# Title") == "<b>Title</b>"

    def test_h3(self):
        assert markdown_to_telegram_html("### Section") == "<b>Section</b>"

    def test_header_strips_nested_bold(self):
        result = markdown_to_telegram_html("## **Bold Title**")
        assert result == "<b>Bold Title</b>"

    def test_header_mid_text(self):
        result = markdown_to_telegram_html("Before\n## Header\nAfter")
        assert "<b>Header</b>" in result
        assert "Before" in result
        assert "After" in result


class TestBlockquotes:
    def test_single_line_blockquote(self):
        result = markdown_to_telegram_html("> quote")
        assert "<blockquote>" in result
        assert "quote" in result
        assert "</blockquote>" in result

    def test_multiline_blockquote(self):
        result = markdown_to_telegram_html("> line1\n> line2")
        # Should be grouped into one blockquote
        assert result.count("<blockquote>") == 1
        assert "line1" in result
        assert "line2" in result


class TestHorizontalRules:
    def test_triple_dash(self):
        result = markdown_to_telegram_html("---")
        assert "\u2015" in result

    def test_triple_asterisk(self):
        result = markdown_to_telegram_html("***")
        # *** could be interpreted as bold italic empty, but as a line by
        # itself it should be a horizontal rule
        assert "\u2015" in result or "<i>" in result

    def test_rule_between_text(self):
        result = markdown_to_telegram_html("above\n---\nbelow")
        assert "above" in result
        assert "below" in result


class TestHtmlEscaping:
    def test_angle_brackets_escaped(self):
        result = markdown_to_telegram_html("a < b > c")
        assert "&lt;" in result
        assert "&gt;" in result

    def test_ampersand_escaped(self):
        result = markdown_to_telegram_html("a & b")
        assert "&amp;" in result

    def test_html_tags_in_plain_text_escaped(self):
        result = markdown_to_telegram_html("<script>alert('xss')</script>")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result


class TestCombined:
    def test_bold_and_code(self):
        result = markdown_to_telegram_html("**bold** and `code`")
        assert "<b>bold</b>" in result
        assert "<code>code</code>" in result

    def test_header_with_code_block(self):
        md = "## Setup\n\n```bash\npip install foo\n```"
        result = markdown_to_telegram_html(md)
        assert "<b>Setup</b>" in result
        assert '<pre><code class="language-bash">' in result

    def test_realistic_llm_output(self):
        md = (
            "## Summary\n\n"
            "Here's a **quick** overview:\n\n"
            "- First item\n"
            "- Second *important* item\n\n"
            "```python\ndef hello():\n    print('world')\n```\n\n"
            "See [docs](https://example.com) for more."
        )
        result = markdown_to_telegram_html(md)
        assert "<b>Summary</b>" in result
        assert "<b>quick</b>" in result
        assert "<i>important</i>" in result
        assert '<pre><code class="language-python">' in result
        assert '<a href="https://example.com">docs</a>' in result

    def test_empty_input(self):
        assert markdown_to_telegram_html("") == ""
        assert markdown_to_telegram_html(None) is None

    def test_plain_text_passthrough(self):
        assert markdown_to_telegram_html("just plain text") == "just plain text"


class TestStripHtmlTags:
    def test_strips_tags(self):
        assert strip_html_tags("<b>bold</b>") == "bold"

    def test_strips_nested_tags(self):
        assert strip_html_tags("<b><i>text</i></b>") == "text"

    def test_preserves_plain_text(self):
        assert strip_html_tags("no tags here") == "no tags here"

    def test_preserves_entities(self):
        assert strip_html_tags("a &amp; b") == "a &amp; b"


class TestSplitHtmlChunks:
    def test_short_message_no_split(self):
        chunks = split_html_chunks("hello", 100)
        assert chunks == ["hello"]

    def test_long_message_splits(self):
        text = "word " * 1000  # ~5000 chars
        chunks = split_html_chunks(text, 100)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 100

    def test_split_adds_indicators(self):
        text = "a " * 2500  # ~5000 chars
        chunks = split_html_chunks(text, 100)
        assert "(1/" in chunks[0]
        assert f"({len(chunks)}/{len(chunks)})" in chunks[-1]

    def test_split_preserves_pre_blocks(self):
        text = "before\n<pre>" + "x" * 200 + "</pre>\nafter"
        chunks = split_html_chunks(text, 150)
        # If a chunk ends mid-<pre>, it should close and reopen
        for chunk in chunks:
            if "<pre>" in chunk:
                assert "</pre>" in chunk
