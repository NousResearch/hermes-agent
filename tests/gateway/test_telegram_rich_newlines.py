"""Tests for rich-message newline normalization (issue #46070).

When Bot API 10.1 ``sendRichMessage`` is available, slash-command responses
are sent through the rich path with RAW markdown.  Standard Markdown treats
a lone ``\\n`` as a soft line break (renders as whitespace), so multi-line
command output collapses into a single paragraph on Telegram.

``_rich_message_payload`` must normalize single newlines to Markdown hard
breaks (two trailing spaces + ``\\n``) and materialize prose paragraph breaks
as exactly one visible spacer row. Fenced code blocks and tables stay raw.

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

    def test_prose_paragraph_breaks_materialize_one_spacer_row(self, adapter):
        """A prose paragraph boundary becomes one hard-broken NBSP row."""
        content = "Paragraph 1\n\nParagraph 2"
        payload = adapter._rich_message_payload(content)
        md = payload["markdown"]
        assert "Paragraph 1  \n\u00a0  \nParagraph 2" == md

    def test_extra_blank_lines_do_not_expand_the_visual_spacer(self, adapter):
        content = "Paragraph 1\n\n\n\nParagraph 2"

        md = adapter._rich_message_payload(content)["markdown"]

        assert md == "Paragraph 1  \n\u00a0  \nParagraph 2"
        assert md.count("\u00a0") == 1

    def test_normalization_is_idempotent(self, adapter):
        once = adapter._rich_message_payload("Paragraph 1\n\nParagraph 2")[
            "markdown"
        ]

        twice = adapter._rich_message_payload(once)["markdown"]

        assert twice == once

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
        # Prose paragraph breaks get one explicit visual spacer.
        assert "Header  \n\u00a0  \n" in md
        assert "\n\u00a0  \nFooter" in md
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

    def test_table_boundaries_and_rows_stay_raw(self, adapter):
        content = (
            "Intro\n\n"
            "| Col A | Col B |\n"
            "|-------|-------|\n"
            "| 1 | 2 |\n\n"
            "Outro"
        )

        md = adapter._rich_message_payload(content)["markdown"]

        assert md == content


class TestRichMessageStructuralBoundaries:
    """Paragraph materialization must not erase Markdown block boundaries."""

    @pytest.mark.parametrize(
        "content",
        [
            "## Heading\n\nBody",
            "Body\n\n- list item",
            "Body\n\n> quote",
            "<details>\n<summary>S</summary>\n\nFirst\n\nSecond\n\n</details>",
            "$$\na = 1\n\nb = 2\n$$",
            "\\[\na = 1\n\nb = 2\n\\]",
            "~~~python\na = 1\n\nb = 2\n~~~",
            "    indented code\n\n    second code block",
            " nested list prose\n\n second nested paragraph",
        ],
    )
    def test_structural_paragraph_boundaries_stay_raw(self, adapter, content):
        md = adapter._rich_message_payload(content)["markdown"]

        assert "\u00a0" not in md
        assert "\n\n" in md

    @pytest.mark.parametrize(
        "content",
        [
            "<details>\n<summary>S</summary>\n\nFirst\n\nSecond\n\n</details>",
            "$$\na = 1\n\nb = 2\n$$",
            "\\[\na = 1\n\nb = 2\n\\]",
            "~~~python\na = 1\n\nb = 2\n~~~",
        ],
    )
    def test_protected_structural_regions_stay_byte_for_byte_raw(
        self, adapter, content
    ):
        assert adapter._rich_message_payload(content)["markdown"] == content

    @pytest.mark.parametrize(
        "second_paragraph",
        ["<https://example.com>", "<em>Inline HTML paragraph.</em>"],
    )
    def test_inline_angle_bracket_prose_gets_spacing(self, adapter, second_paragraph):
        md = adapter._rich_message_payload(
            f"First paragraph.\n\n{second_paragraph}"
        )["markdown"]

        assert md == f"First paragraph.  \n\u00a0  \n{second_paragraph}"


def test_rich_limit_counts_materialized_paragraph_payload(adapter):
    content = ("a\n\n" * 10_000) + "| A | B |\n|---|---|\n| 1 | 2 |"

    assert len(content) < adapter.RICH_MESSAGE_MAX_CHARS
    assert len(adapter._rich_message_payload(content)["markdown"]) > (
        adapter.RICH_MESSAGE_MAX_CHARS
    )
    assert adapter._content_fits_rich_limits(content) is False

