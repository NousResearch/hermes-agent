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

    def test_fenced_code_block_newlines_preserved(self, adapter):
        """Newlines inside fenced code blocks must NOT gain trailing spaces."""
        content = "Before\n```\ncode line 1\ncode line 2\n```\nAfter"
        payload = adapter._rich_message_payload(content)
        md = payload["markdown"]
        # Code block content should be untouched
        assert "```\ncode line 1\ncode line 2\n```" in md
        # But the \n before ``` and after ``` should be hard breaks
        assert "Before  \n```" in md
        assert "```  \nAfter" in md

    def test_realistic_command_output(self, adapter):
        """Simulates /commands output: header + list items + nav line."""
        lines = [
            "📊 Commands (24 total, page 1/2)",
            "",
            "`/new` -- Start a new session",
            "`/model` -- Switch model",
            "`/stop` -- Stop the agent",
            "",
            "Use /commands 2 for next page | /commands 1 for prev",
        ]
        content = "\n".join(lines)
        payload = adapter._rich_message_payload(content)
        md = payload["markdown"]
        # Header paragraph break preserved
        assert "📊 Commands (24 total, page 1/2)\n\n" in md
        # List items have hard breaks
        assert "`/new` -- Start a new session  \n" in md
        assert "`/model` -- Switch model  \n" in md
        # Nav paragraph break preserved
        assert "\n\nUse /commands 2" in md

    def test_no_trailing_space_on_last_line(self, adapter):
        """The final line should not get trailing spaces (no newline after it)."""
        content = "Line 1\nLine 2"
        payload = adapter._rich_message_payload(content)
        md = payload["markdown"]
        # No trailing spaces at end of string
        assert md == "Line 1  \nLine 2"
        assert not md.endswith("  ")

    def test_empty_and_single_line_unchanged(self, adapter):
        """Empty string and single-line content should pass through."""
        assert adapter._rich_message_payload("")["markdown"] == ""
        assert adapter._rich_message_payload("Single line")["markdown"] == "Single line"

    def test_skip_entity_detection_flag_preserved(self, adapter):
        """The skip_entity_detection flag must still work after normalization."""
        payload = adapter._rich_message_payload("Line 1\nLine 2", skip_entity_detection=True)
        assert payload.get("skip_entity_detection") is True


class TestRichMessageTableProtection:
    """Hard-break injection must not corrupt GFM tables (rendered natively)."""

    def test_table_rows_keep_bare_newlines(self, adapter):
        """Table block newlines must stay bare — no '  \\n' inside the table."""
        content = "| Col A | Col B |\n|-------|-------|\n| 1 | 2 |\n| 3 | 4 |"
        md = adapter._rich_message_payload(content)["markdown"]
        assert "  \n" not in md
        assert md == content

    def test_text_around_table_still_gets_hard_breaks(self, adapter):
        """Prose lines outside the table keep getting hard breaks."""
        content = (
            "Intro line one\n"
            "Intro line two\n"
            "| H1 | H2 |\n"
            "|----|----|\n"
            "| a | b |\n"
            "Outro line"
        )
        md = adapter._rich_message_payload(content)["markdown"]
        # Prose-to-prose newline becomes a hard break.
        assert "Intro line one  \nIntro line two" in md
        # Table rows stay bare.
        assert "| H1 | H2 |\n|----|----|\n| a | b |" in md
        # Prose lines around the table still hard-break; only the table's own
        # header/delimiter/data-row newlines stay bare.
        assert "Intro line two  \n| H1 | H2 |" in md
        assert "| a | b |  \nOutro line" in md


class TestRichMessageBlockMathProtection:
    """Hard-break injection must not corrupt block math (rendered natively)."""

    def test_block_math_interior_keeps_bare_newlines(self, adapter):
        """Newlines inside a $$...$$ display-math span must stay bare — a
        '  \\n' between the rows of an aligned/cases environment is invalid
        LaTeX. Block math (`$$`) is one of the constructs that triggers the
        rich path (_needs_rich_rendering), so it must be protected like the
        natively-rendered fenced-code and table blocks."""
        content = (
            "Solve:\n"
            "$$\n"
            "\\begin{aligned}\n"
            "a &= b \\\\\n"
            "c &= d\n"
            "\\end{aligned}\n"
            "$$\n"
            "Done"
        )
        md = adapter._rich_message_payload(content)["markdown"]
        # The display-math block is preserved verbatim (no '  \n' injected).
        assert (
            "$$\n\\begin{aligned}\na &= b \\\\\nc &= d\n\\end{aligned}\n$$" in md
        ), f"block math interior must stay bare, got {md!r}"
        # Prose around the math still hard-breaks.
        assert "Solve:  \n$$" in md
        assert "$$  \nDone" in md

    def test_single_line_block_math_unaffected(self, adapter):
        """A single-line $$x$$ has no interior newline, so its rendering is
        unchanged; surrounding prose still hard-breaks."""
        content = "inline $$x = 1$$ here\nnext"
        md = adapter._rich_message_payload(content)["markdown"]
        assert md == "inline $$x = 1$$ here  \nnext"

    def test_stray_dollars_before_code_block_do_not_corrupt_it(self, adapter):
        """A stray/inline ``$$`` in prose must not pair with a ``$$`` inside a
        later fenced code block: block math is recognized only outside the
        protected fenced-code/table regions, so the code block stays verbatim
        and only the prose around it hard-breaks."""
        content = "Cost is $$5 today\n```\nx = $$y\nz = $$w\n```\nafter"
        md = adapter._rich_message_payload(content)["markdown"]
        # Code block (incl. its literal $$) is untouched.
        assert "```\nx = $$y\nz = $$w\n```" in md
        # The stray-$$ prose line before the fence still hard-breaks.
        assert "Cost is $$5 today  \n```" in md

    def test_currency_lines_hard_break_normally(self, adapter):
        """Lines that merely contain ``$$`` (currency, not a lone block
        delimiter) are not treated as block math; they hard-break like prose."""
        content = "Price $$5\nPlus $$10\nmore"
        md = adapter._rich_message_payload(content)["markdown"]
        assert md == "Price $$5  \nPlus $$10  \nmore"
