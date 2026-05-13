"""Tests for shared gateway platform helpers."""

from gateway.platforms.helpers import strip_markdown


class TestStripMarkdown:
    def test_strips_underscore_italic_at_word_boundaries(self):
        assert strip_markdown("_italic_ text") == "italic text"

    def test_preserves_intraword_underscores(self):
        assert strip_markdown("foo_bar_baz") == "foo_bar_baz"
