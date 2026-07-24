"""Shared GFM table → bullet conversion helpers."""

from gateway.platforms.helpers import (
    TABLE_SEPARATOR_RE,
    is_table_row,
    split_markdown_table_row,
    convert_table_to_bullets,
)


class TestTablePrimitives:

    def test_separator_re_matches_basic(self):
        assert TABLE_SEPARATOR_RE.match("|---|---|")

    def test_separator_re_matches_alignment(self):
        assert TABLE_SEPARATOR_RE.match("|:-----|----:|:----:|")

    def test_separator_re_rejects_lone_rule(self):
        assert not TABLE_SEPARATOR_RE.match("---")

    def test_is_table_row_with_pipe(self):
        assert is_table_row("| Alice | 150 |")

    def test_is_table_row_blank(self):
        assert not is_table_row("")

    def test_split_row_strips_outer_pipes(self):
        assert split_markdown_table_row("| a | b | c |") == ["a", "b", "c"]

    def test_split_row_no_outer_pipes(self):
        assert split_markdown_table_row("a | b | c") == ["a", "b", "c"]


class TestConvertTableToBullets:

    def test_basic_table(self):
        text = (
            "| Player | Score |\n"
            "|--------|-------|\n"
            "| Alice  | 150   |\n"
            "| Bob    | 120   |"
        )
        out = convert_table_to_bullets(text)
        assert "**Alice**" in out
        assert "• Score: 150" in out
        assert "**Bob**" in out
        assert "• Score: 120" in out
        assert "• Player: Alice" not in out

    def test_bold_cell_heading_not_double_wrapped(self):
        """Regression: a ``**bold**`` cell used to be wrapped again into
        ``****bold****``, which the MarkdownV2 bold converter downstream
        mangles into an escaped ``\\*\\*`` that Telegram renders as a literal
        ``**``.  The heading's existing ``*`` markers must be stripped before
        wrapping so each bold cell ends up wrapped exactly once.
        """
        text = (
            "| 항목 | 설명 |\n"
            "|------|------|\n"
            "| **OCR** | 이미지에서 텍스트 |\n"
            "| **LLM** | 대형 언어 모델 |"
        )
        out = convert_table_to_bullets(text)
        # No double-wrap survives anywhere.
        assert "****" not in out
        # Each bold heading is wrapped exactly once.
        assert out.count("**OCR**") == 1
        assert out.count("**LLM**") == 1

    def test_italic_marker_in_heading_stripped(self):
        """A single-``*`` italic marker in a heading cell is stripped too,
        so ``*foo*`` wraps to ``**foo**`` instead of ``***foo***``."""
        text = (
            "| Type | Count |\n"
            "|------|-------|\n"
            "| *foo* | 3 |"
        )
        out = convert_table_to_bullets(text)
        assert "***" not in out
        assert "**foo**" in out

    def test_three_column_table(self):
        text = (
            "| Name | Age | City |\n"
            "|:-----|----:|:----:|\n"
            "| Ada  |  30 | NYC  |"
        )
        out = convert_table_to_bullets(text)
        assert "**Ada**" in out
        assert "• Name: Ada" not in out
        assert "• Age: 30" in out
        assert "• City: NYC" in out
        assert "**Ada**\n• Age: 30\n• City: NYC" in out

    def test_row_label_column(self):
        text = (
            "|        | Score | Rank |\n"
            "|--------|-------|------|\n"
            "| Alice  | 150   | 1    |\n"
            "| Bob    | 120   | 2    |"
        )
        out = convert_table_to_bullets(text)
        assert "**Alice**" in out
        assert "• Score: 150" in out
        assert "• Rank: 1" in out
        assert "**Alice**\n• Score: 150\n• Rank: 1" in out

    def test_bare_pipe_table(self):
        text = "head1 | head2\n--- | ---\na | b\nc | d"
        out = convert_table_to_bullets(text)
        assert "**a**" in out
        assert "• head1: a" not in out
        assert "• head2: b" in out

    def test_two_consecutive_tables(self):
        text = (
            "| A | B |\n"
            "|---|---|\n"
            "| 1 | 2 |\n"
            "\n"
            "| X | Y |\n"
            "|---|---|\n"
            "| 9 | 8 |"
        )
        out = convert_table_to_bullets(text)
        assert out.count("**1**") == 1
        assert out.count("**9**") == 1
        assert "• B: 2" in out
        assert "• Y: 8" in out

    def test_surrounding_prose_preserved(self):
        text = (
            "Scores:\n\n"
            "| Player | Score |\n"
            "|--------|-------|\n"
            "| Alice  | 150   |\n"
            "\nEnd."
        )
        out = convert_table_to_bullets(text)
        assert out.startswith("Scores:")
        assert out.endswith("End.")

    def test_table_inside_code_fence_untouched(self):
        text = "```\n| a | b |\n|---|---|\n| 1 | 2 |\n```"
        assert convert_table_to_bullets(text) == text

    def test_plain_text_with_pipes_untouched(self):
        text = "Use the | pipe operator to chain."
        assert convert_table_to_bullets(text) == text

    def test_horizontal_rule_not_matched(self):
        text = "Section A\n\n---\n\nSection B"
        assert convert_table_to_bullets(text) == text

    def test_no_pipe_short_circuits(self):
        text = "Plain **bold** text."
        assert convert_table_to_bullets(text) == text

    def test_row_groups_separated_by_blank_line(self):
        text = (
            "| A | B |\n"
            "|---|---|\n"
            "| x | 1 |\n"
            "| y | 2 |"
        )
        out = convert_table_to_bullets(text)
        assert "• B: 1\n\n**y**" in out
        assert "\n\n• " not in out
