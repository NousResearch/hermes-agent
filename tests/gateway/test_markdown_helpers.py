"""Tests for shared platform Markdown formatting helpers."""

from gateway.platforms.helpers import format_markdown_link


def test_format_markdown_link_escapes_label_and_destination_delimiters():
    assert format_markdown_link(
        r"docs [beta]", "https://example.com/a_(draft)"
    ) == r"[docs \[beta\]](https://example.com/a_%28draft%29)"
