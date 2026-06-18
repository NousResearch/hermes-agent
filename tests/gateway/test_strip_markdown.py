"""Tests for the shared ``strip_markdown`` helper in ``gateway/platforms/helpers.py``.

Covers both correct stripping of real emphasis and preservation of
literal asterisks (bullet lists, math, globs).  Regression for #48150.
"""

import re

import pytest

from gateway.platforms.helpers import strip_markdown, _RE_BOLD, _RE_ITALIC_STAR


class TestStripMarkdownPreservesBulletLists:
    """Bullet-list markers must pass through unchanged."""

    def test_bullet_list_no_loss(self):
        text = "Here are steps:\n* Install deps\n* Run tests\n* Ship it"
        assert strip_markdown(text) == text

    def test_consecutive_bullets(self):
        assert strip_markdown("* first item\n* second item") == "* first item\n* second item"

    def test_single_bullet(self):
        assert strip_markdown("* Just one item") == "* Just one item"

    def test_mixed_bullets_and_emphasis(self):
        result = strip_markdown("* **urgent** fix this\n* *maybe* later")
        assert result == "* urgent fix this\n* maybe later"


class TestStripMarkdownPreservesLiteralAsterisks:
    """Literal asterisks (math, globs, etc.) must not be stripped."""

    def test_multiplication(self):
        assert strip_markdown("Compute a * b * c for the area") == "Compute a * b * c for the area"

    def test_glob_patterns(self):
        assert strip_markdown("Use the * wildcard and the ** glob") == "Use the * wildcard and the ** glob"

    def test_edge_asterisks_parenthesis(self):
        assert strip_markdown("(*) denotes significance") == "(*) denotes significance"


class TestStripMarkdownStillStripsRealEmphasis:
    """Real emphasis spans must still be unwrapped."""

    def test_bold(self):
        assert strip_markdown("This is **bold** text") == "This is bold text"

    def test_italic(self):
        assert strip_markdown("This is *italic* text") == "This is italic text"

    def test_bold_and_italic(self):
        assert strip_markdown("**bold** and *italic*") == "bold and italic"

    def test_bold_italic_three_stars(self):
        result = strip_markdown("A ***bold italic*** span")
        assert result == "A bold italic span"

    def test_underscore_bold(self):
        assert strip_markdown("This is __bold__ text") == "This is bold text"

    def test_underscore_italic(self):
        assert strip_markdown("This is _italic_ text") == "This is italic text"

    def test_title_with_bold(self):
        assert strip_markdown("**important** items: *one*, *two*") == "important items: one, two"


class TestRegexEdgeCases:
    """Boundary cases for the emphasis regexes."""

    @pytest.mark.parametrize("text", [
        "*a*",           # single char italic
        "**a**",         # single char bold
        "*ab*",          # two char italic
        "x *y* z",       # italic within text
        "x **y** z",     # bold within text
    ])
    def test_minimal_emphasis_stripped(self, text):
        result = strip_markdown(text)
        assert "*" not in result

    @pytest.mark.parametrize("text", [
        " * a * b",      # spaces inside
        "a * b * c",     # literal asterisks with spaces
        "* ",            # lone bullet marker (trailing space)
        " *",            # leading space then asterisk (not emphasis)
        "** **",         # two bold markers with space
    ])
    def test_not_emphasis_preserved(self, text):
        """Patterns that look like emphasis but aren't should be preserved."""
        result = strip_markdown(text)
        # The asterisks should still be present.
        assert "*" in result


class TestStripMarkdownRegression:
    """Known regression cases from #48150."""

    def test_issue_bullet_list(self):
        """The exact reproduction case from the issue."""
        text = "Here are steps:\n* Install deps\n* Run tests\n* Ship it"
        assert strip_markdown(text) == text

    def test_issue_two_bullets(self):
        assert strip_markdown("* first item\n* second item") == "* first item\n* second item"

    def test_issue_multiplication(self):
        assert strip_markdown("Compute a * b * c for the area") == "Compute a * b * c for the area"

    def test_issue_glob(self):
        assert strip_markdown("Use the * wildcard and the ** glob") == "Use the * wildcard and the ** glob"
