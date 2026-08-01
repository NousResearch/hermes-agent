"""Tests for rich-message newline normalization (issue #46070).

When Bot API 10.1 ``sendRichMessage`` is available, slash-command responses
are sent through the rich path with RAW markdown.  Standard Markdown treats
a lone ``\\n`` as a soft line break (renders as whitespace), so multi-line
command output collapses into a single paragraph on Telegram.

``_rich_message_payload`` must normalize single newlines to Markdown hard
breaks (two trailing spaces + ``\\n``) so they render as visible line breaks.
Paragraph breaks (``\\n\\n``) and fenced code blocks must be preserved.

The ``telegram`` package is mocked by ``tests/gateway/conftest.py``, so these
tests construct a real ``TelegramAdapter``.
"""

import pytest

from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.fixture()
def adapter():
    """Bare adapter instance — _rich_message_payload doesn't use self."""
    return object.__new__(TelegramAdapter)


class TestRichMessageNewlineNormalization:
    """Verify _rich_message_payload normalizes single \\n to hard breaks."""

    def test_single_newlines_become_hard_breaks(self, adapter):
        """A lone \\n must gain two trailing spaces (Markdown hard break).

        Standard Markdown soft-break rendering causes Bot API 10.1
        ``sendRichMessage`` to collapse multi-line content into one paragraph.
        """
        content = "Line 1\nLine 2\nLine 3"
        payload = adapter._rich_message_payload(content)
        md = payload["markdown"]
        # Each single \n should now be "  \n" (two spaces + newline)
        assert "  \n" in md, f"Expected hard break '  \\n' in {md!r}"
        assert "Line 1  \nLine 2  \nLine 3" == md

    def test_paragraph_breaks_preserved(self, adapter):
        """Double newlines (paragraph breaks) must NOT gain extra spaces."""
        content = "Paragraph 1\n\nParagraph 2"
        payload = adapter._rich_message_payload(content)
        md = payload["markdown"]
        # \n\n should remain as-is — no trailing spaces injected
        assert "Paragraph 1\n\nParagraph 2" == md

    def test_mixed_single_and_double_newlines(self, adapter):
        """Content with both list items and paragraph breaks must be handled correctly."""
        content = (
            "Header\n\n"
            "`/new` -- Start\n"
            "`/model` -- Switch\n"
            "`/reset` -- Reset\n\n"
            "Footer"
        )
        payload = adapter._rich_message_payload(content)
        md = payload["markdown"]
        # Paragraph breaks preserved
        assert "Header\n\n" in md
        assert "\n\nFooter" in md
        # Single newlines converted to hard breaks
        assert "`/new` -- Start  \n`/model` -- Switch  \n`/reset` -- Reset" in md


class TestRichMessageTableProtection:
    """Hard-break injection must not corrupt GFM tables (rendered natively)."""

    def test_table_rows_keep_bare_newlines(self, adapter):
        """Table block newlines must stay bare — no '  \\n' inside the table."""
        content = "| Col A | Col B |\n|-------|-------|\n| 1 | 2 |\n| 3 | 4 |"
        md = adapter._rich_message_payload(content)["markdown"]
        assert "  \n" not in md
        assert md == content



class TestRichBlockConstructProtection:
    """Hard-break markers must not be stamped onto/around block constructs.

    A single newline adjacent to a heading, list item, blockquote opener,
    table, or fence is already a hard block boundary; "  \\n" there is
    meaningless in CommonMark and malformed in lenient parsers. #69444's
    truncation repro stopped exactly at a heading and at a paragraph→list
    boundary — the two shapes stamped by the old unconditional injection.
    """

    def test_heading_line_gets_no_trailing_hard_break(self, adapter):
        content = "## What RevenueOS has that Buzz does not\nFirst paragraph."
        md = adapter._rich_message_payload(content)["markdown"]
        assert md == content  # heading→prose newline untouched

    def test_prose_to_heading_gets_no_hard_break(self, adapter):
        content = "Intro text\n### Next heading"
        md = adapter._rich_message_payload(content)["markdown"]
        assert md == content

    def test_paragraph_to_list_boundary_gets_no_hard_break(self, adapter):
        content = "RevenueOS answers:\n- point one\n- point two"
        md = adapter._rich_message_payload(content)["markdown"]
        assert md == content  # list opener and item→item newlines untouched

    def test_ordered_list_boundary_gets_no_hard_break(self, adapter):
        content = "Steps:\n1. first\n2. second"
        md = adapter._rich_message_payload(content)["markdown"]
        assert md == content

    def test_lazy_list_continuation_keeps_hard_break(self, adapter):
        # A plain-text line after a list item is a lazy continuation that
        # would soft-wrap into the item — the marker is load-bearing there.
        content = "- item one\ncontinued text\n- item two"
        md = adapter._rich_message_payload(content)["markdown"]
        assert md == "- item one  \ncontinued text\n- item two"

    def test_blockquote_interior_keeps_hard_breaks(self, adapter):
        # "> a\n> b" soft-wraps into one line without markers.
        content = "> first line\n> second line"
        md = adapter._rich_message_payload(content)["markdown"]
        assert md == "> first line  \n> second line"

    def test_prose_to_blockquote_gets_no_hard_break(self, adapter):
        content = "Intro text\n> a quote"
        md = adapter._rich_message_payload(content)["markdown"]
        assert md == content

    def test_prose_to_table_gets_no_hard_break_weld(self, adapter):
        content = "Scores:\n| A | B |\n|---|---|\n| 1 | 2 |"
        md = adapter._rich_message_payload(content)["markdown"]
        assert md == content  # no "  " stamped before the header row

    def test_prose_to_fence_gets_no_hard_break_weld(self, adapter):
        content = "Example:\n```py\nx = 1\n```"
        md = adapter._rich_message_payload(content)["markdown"]
        assert md == content

    def test_plain_prose_lines_still_get_hard_breaks(self, adapter):
        content = "Line 1\nLine 2\nLine 3"
        md = adapter._rich_message_payload(content)["markdown"]
        assert md == "Line 1  \nLine 2  \nLine 3"

    def test_hashtag_line_is_not_treated_as_heading(self, adapter):
        # "#tag" (no space) is not an ATX heading — still prose.
        content = "#hashtag one\nplain line"
        md = adapter._rich_message_payload(content)["markdown"]
        assert md == "#hashtag one  \nplain line"

    def test_69444_repro_shape_stays_clean(self, adapter):
        content = (
            "## What RevenueOS has that Buzz does not\n"
            "First differentiator paragraph.\n"
            "RevenueOS answers:\n"
            "- point one\n"
            "- point two\n\n"
            "> An important quote\n"
            "> second quote line\n\n"
            "### Next heading\n"
            "More text."
        )
        md = adapter._rich_message_payload(content)["markdown"]
        # The ONLY hard break left is the prose→prose one after the first
        # paragraph; every block-adjacent newline stays bare.
        assert "## What RevenueOS has that Buzz does not\n" in md
        assert "First differentiator paragraph.  \nRevenueOS answers:\n" in md
        assert "RevenueOS answers:\n- point one\n- point two\n\n" in md
        assert "> An important quote  \n> second quote line\n\n" in md
        assert "### Next heading\nMore text." in md
