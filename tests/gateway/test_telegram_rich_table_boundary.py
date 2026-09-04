"""A GFM table needs a blank line around it, not a hard break.

A table written directly under a line of prose — no blank line between them —
arrived in Telegram as a single run-on paragraph of literal ``| a | b |`` text,
while an otherwise identical table with a blank line above it rendered
natively.  ``sendRichMessage`` echoes the parsed block structure back, which
shows the rule directly: a table cannot be opened from inside an open
paragraph, and a bare newline above it is not enough either.

``_rich_normalize_linebreaks`` protects the *inside* of a table block but used
to hard-break the newline that leads into it, so the caption's paragraph stayed
open and the table could never start a block of its own. The mirror case —
prose written directly under the last table row — was absorbed as one more
table row and also picked up two stray trailing spaces inside the final cell.

Fenced code blocks deliberately keep their hard-break boundary: a fence *can*
interrupt an open paragraph, so promoting that newline would only add a blank
line nobody asked for.

The ``telegram`` package is mocked by ``tests/gateway/conftest.py``, so these
tests construct a real ``TelegramAdapter``.
"""

import pytest

from plugins.platforms.telegram.adapter import TelegramAdapter


@pytest.fixture()
def adapter():
    """Bare adapter instance — _rich_message_payload doesn't use self."""
    return object.__new__(TelegramAdapter)


def _md(adapter, content: str) -> str:
    return adapter._rich_message_payload(content)["markdown"]


class TestTableBoundaryPromotedToParagraphBreak:
    """The newline between prose and a table must become a blank line."""

    def test_table_under_caption_gets_blank_line(self, adapter):
        """Caption + table with no blank line: the reported break."""
        content = (
            "**Knowledge base**\n"
            "| # | What |\n"
            "|---|---|\n"
            "| 201 | Backlinks |\n"
            "| 202 | Search |"
        )
        md = _md(adapter, content)
        assert md.startswith("**Knowledge base**\n\n|"), md
        # No hard break may survive on the boundary — that is what kept the
        # caption's paragraph open and swallowed the table.
        assert "  \n" not in md, md

    def test_prose_under_table_gets_blank_line(self, adapter):
        """Prose directly under the last row must not become a table row."""
        content = "| # | What |\n|---|---|\n| 201 | a |\ntrailing prose"
        md = _md(adapter, content)
        assert md == "| # | What |\n|---|---|\n| 201 | a |\n\ntrailing prose", md
        # The old code left "| 201 | a |  \n" — two spaces inside the last cell.
        assert "|  \n" not in md, md

    def test_every_table_in_a_multi_table_message(self, adapter):
        """The reported message had four tables; each boundary is independent."""
        content = (
            "## Report\n"
            "| a | b |\n|---|---|\n| 1 | 2 |\n"
            "\n"
            "**Second**\n"
            "| c | d |\n|---|---|\n| 3 | 4 |"
        )
        md = _md(adapter, content)
        assert "## Report\n\n| a | b |" in md, md
        assert "**Second**\n\n| c | d |" in md, md

    def test_existing_blank_line_is_left_alone(self, adapter):
        """The one table that already rendered must come out byte-identical."""
        content = "**Knowledge base**\n\n| # | What |\n|---|---|\n| 201 | Backlinks |"
        assert _md(adapter, content) == content


class TestBoundaryPromotionStaysNarrow:
    """Guards against the fix reaching past table boundaries."""

    def test_fence_boundary_keeps_its_hard_break(self, adapter):
        """A fence can interrupt a paragraph — no blank line to inject."""
        content = "Look:\n```bash\nls -la\n```\nafter"
        md = _md(adapter, content)
        assert md == "Look:  \n```bash\nls -la\n```  \nafter", md

    def test_plain_prose_still_gets_hard_breaks(self, adapter):
        """The original #46070 behaviour must survive untouched."""
        assert _md(adapter, "Line 1\nLine 2\nLine 3") == "Line 1  \nLine 2  \nLine 3"

    def test_fence_directly_above_table_still_gets_the_blank_line(self, adapter):
        """The only path where the boundary is decided by the *following* region.

        The closing fence is not prose, so nothing above needs a hard break —
        but the table below still cannot open inside whatever precedes it.
        """
        content = "```py\nx = 1\n```\n| a | b |\n|---|---|\n| 1 | 2 |"
        md = _md(adapter, content)
        assert md == "```py\nx = 1\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |", md

    def test_table_at_message_start_needs_no_leading_blank(self, adapter):
        """Nothing above the table — do not prepend a stray blank line."""
        content = "| a | b |\n|---|---|\n| 1 | 2 |"
        assert _md(adapter, content) == content
