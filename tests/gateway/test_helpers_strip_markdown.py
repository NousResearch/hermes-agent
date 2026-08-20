"""Regression tests for ``gateway.platforms.helpers.strip_markdown``.

The shared helper strips Markdown formatting for plain-text platforms
(SMS, iMessage/BlueBubbles, Feishu, QQ Bot, Photon). The star emphasis
regexes used to lack the inside-edge guards that the underscore variants
already had, so ``*`` list bullets and literal asterisks were swallowed as
if they were italic spans. These tests pin the corrected behaviour and the
emphasis stripping that must keep working.
"""

from gateway.platforms.helpers import strip_markdown


class TestStripMarkdownPreservesBulletsAndLiterals:
    def test_simple_star_bullet_list_is_preserved(self):
        text = "* first item\n* second item"
        assert strip_markdown(text) == text

    def test_star_bullet_list_with_lead_in_is_preserved(self):
        text = "Here are steps:\n* Install deps\n* Run tests\n* Ship it"
        assert strip_markdown(text) == text

    def test_literal_asterisks_with_spaces_are_preserved(self):
        text = "Compute a * b * c for the area"
        assert strip_markdown(text) == text

    def test_wildcard_and_double_star_glob_are_preserved(self):
        text = "Use the * wildcard and the ** glob"
        assert strip_markdown(text) == text


class TestStripMarkdownStillStripsEmphasis:
    def test_bold_star_is_stripped(self):
        assert strip_markdown("**bold**") == "bold"

    def test_italic_star_is_stripped(self):
        assert strip_markdown("This is *very* important") == "This is very important"

    def test_bold_underscore_is_stripped(self):
        assert strip_markdown("__bold__") == "bold"

    def test_italic_underscore_is_stripped(self):
        assert strip_markdown("_italic_") == "italic"

    def test_bold_inside_dash_bullet_is_stripped_marker_kept(self):
        text = "- **Step 1**: do this\n- **Step 2**: do that"
        assert strip_markdown(text) == "- Step 1: do this\n- Step 2: do that"

    def test_inline_code_and_heading_and_link_still_work(self):
        assert strip_markdown("`code`") == "code"
        assert strip_markdown("# Heading") == "Heading"
        assert strip_markdown("[click here](http://example.com)") == "click here"
