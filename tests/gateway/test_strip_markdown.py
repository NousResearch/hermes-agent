"""Tests for the shared ``strip_markdown`` plain-text helper.

Regression coverage for markdown image syntax. The link regex alone treated
an image (``![alt](url)``) as a link prefixed with ``!``, so it left a stray
``!`` (``![alt](url)`` → ``!alt``) and skipped empty-alt images
(``![](url)``) entirely. Images must be collapsed to their alt text before
the link pass runs.
"""

from gateway.platforms.helpers import strip_markdown


class TestStripMarkdownImages:
    """Image syntax should reduce to alt text with no leftover markers."""

    def test_image_with_alt_keeps_only_alt_text(self):
        assert strip_markdown("![alt](http://x.png)") == "alt"

    def test_image_inline_keeps_surrounding_text(self):
        assert strip_markdown("See ![diagram](http://y.png) here") == "See diagram here"

    def test_image_without_alt_is_removed(self):
        assert strip_markdown("![](http://noalt.png)") == ""

    def test_plain_link_unaffected(self):
        assert strip_markdown("text [link](http://z.com) end") == "text link end"

    def test_mixed_link_and_image(self):
        assert (
            strip_markdown("**bold** and [a](b) and ![img](c)")
            == "bold and a and img"
        )


class TestStripMarkdownBasics:
    """Guard the existing inline-formatting behaviour against regressions."""

    def test_bold_italic_and_code(self):
        assert strip_markdown("**hello**") == "hello"
        assert strip_markdown("*hello*") == "hello"
        assert strip_markdown("run `ls -la`") == "run ls -la"

    def test_heading_stripped(self):
        assert strip_markdown("# Title") == "Title"
