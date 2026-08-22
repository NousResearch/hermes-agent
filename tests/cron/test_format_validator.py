"""Tests for cron.format_validator — TKT-0033 Phase A.

A deterministic, code-block-aware HTML-leak validator: when a cron job
declares ``text/markdown`` (the default) but its delivery content contains
raw HTML tags OUTSIDE code fences, the send must hard-fail into a dead-letter
record instead of leaking literal tags to the user.
"""

from cron.format_validator import (
    find_html_leak,
    should_deadletter,
    strip_code_blocks,
)


class TestStripCodeBlocks:
    def test_removes_fenced_block(self):
        text = "before\n```\n<b>not a leak</b>\n```\nafter"
        stripped = strip_code_blocks(text)
        assert "<b>" not in stripped
        assert "before" in stripped
        assert "after" in stripped

    def test_removes_fenced_block_with_language(self):
        text = "look:\n```html\n<i>italics</i>\n```\ndone"
        stripped = strip_code_blocks(text)
        assert "<i>" not in stripped
        assert "done" in stripped

    def test_removes_inline_code_span(self):
        text = "use `<b>bold</b>` for bold"
        stripped = strip_code_blocks(text)
        assert "<b>" not in stripped
        assert "for bold" in stripped

    def test_preserves_plain_text(self):
        text = "Status: **OK** and x < y"
        assert strip_code_blocks(text) == text


class TestFindHtmlLeak:
    def test_fenced_block_containing_html_is_not_a_leak(self):
        text = "example:\n```\n<b>hi</b>\n```"
        assert find_html_leak(text) is None

    def test_inline_code_html_is_not_a_leak(self):
        text = "run `<i>` to italicize"
        assert find_html_leak(text) is None

    def test_html_inside_html_fence_is_not_a_leak(self):
        text = "```html\n<div class=\"x\">hello</div>\n```"
        assert find_html_leak(text) is None

    def test_raw_bold_tag_is_a_leak(self):
        text = "Status: <b>OK</b>"
        leak = find_html_leak(text)
        assert leak is not None
        assert leak.startswith("<b")

    def test_raw_closing_tag_is_a_leak(self):
        text = "value is </i> here"
        leak = find_html_leak(text)
        assert leak is not None
        assert leak == "</i>"

    def test_tag_with_attributes_is_a_leak(self):
        text = 'see <a href="https://x">link</a>'
        leak = find_html_leak(text)
        assert leak is not None
        assert leak.startswith("<a ")

    def test_clean_markdown_is_not_a_leak(self):
        text = "**bold** and x < y and a -> b"
        assert find_html_leak(text) is None

    def test_arrow_is_not_a_tag(self):
        assert find_html_leak("a -> b and c -> d") is None

    def test_comparison_is_not_a_tag(self):
        assert find_html_leak("if x < y and y > z") is None

    def test_leak_after_fence_is_found(self):
        text = "```\n<b>safe</b>\n```\nbut <u>this</u> leaks"
        leak = find_html_leak(text)
        assert leak is not None
        assert "<u>" in leak


class TestShouldDeadletter:
    def test_markdown_with_leak_is_deadlettered(self):
        assert should_deadletter("text/markdown", "Status: <b>OK</b>") is True

    def test_markdown_clean_delivers_normally(self):
        assert should_deadletter("text/markdown", "**bold** and x < y") is False

    def test_html_payload_skips_check(self):
        assert should_deadletter("text/html", "<b>anything</b> goes") is False

    def test_html_payload_clean_also_skips(self):
        assert should_deadletter("text/html", "plain text") is False
