"""Shared GFM table conversion helpers."""

from gateway.platforms.helpers import (
    TABLE_SEPARATOR_RE,
    _display_width,
    is_table_row,
    split_markdown_table_row,
    convert_table_to_bullets,
    convert_table_to_codeblock,
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


class TestConvertTableToCodeblock:

    def test_basic_table(self):
        text = (
            "Here is a table:\n\n"
            "| Name  | Score | Grade |\n"
            "|-------|-------|-------|\n"
            "| Alice | 95    | A     |\n"
            "| Bob   | 82    | B     |\n\n"
            "Done."
        )
        out = convert_table_to_codeblock(text)
        assert "|-------|" not in out
        assert "```\n┌" in out
        assert "│ Name  │ Score │ Grade │" in out
        assert "│ Alice │ 95    │ A     │" in out
        assert out.startswith("Here is a table:")
        assert out.endswith("Done.")

    def test_table_inside_code_fence_untouched(self):
        text = "```\n| a | b |\n|---|---|\n| 1 | 2 |\n```"
        assert convert_table_to_codeblock(text) == text

    def test_fenced_table_kept_while_real_table_converted(self):
        text = (
            "```\n"
            "| Not | A | Table |\n"
            "|-----|---|-------|\n"
            "| x   | y | z     |\n"
            "```\n\n"
            "| Real | Table |\n"
            "|------|-------|\n"
            "| a    | b     |"
        )
        out = convert_table_to_codeblock(text)
        assert "| Not | A | Table |" in out
        assert "|-----|---|-------|" in out
        assert "│ Real │ Table │" in out
        assert "│ a    │ b     │" in out
        assert out.count("┌") == 1

    def test_cjk_wide_characters_stay_aligned(self):
        text = (
            "| 항목 | 설명 |\n"
            "|------|------|\n"
            "| CPU  | 중앙처리장치 |\n"
            "| RAM  | 메모리 |"
        )
        out = convert_table_to_codeblock(text)
        table_lines = [
            line for line in out.splitlines()
            if line and line[0] in "┌├└│"
        ]
        assert len({_display_width(line) for line in table_lines}) == 1
        assert "│ CPU  │ 중앙처리장치 │" in out
        assert "│ RAM  │ 메모리       │" in out

    def test_ragged_rows_padded_and_clipped(self):
        text = (
            "| A | B | C |\n"
            "|---|---|---|\n"
            "| 1 |\n"
            "| 2 | 3 | 4 | 5 |"
        )
        out = convert_table_to_codeblock(text)
        assert "│ 1   │     │     │" in out
        assert "│ 2   │ 3   │ 4   │" in out
        assert "5" not in out

    def test_plain_text_with_pipes_untouched(self):
        text = "Use the | operator for bitwise OR."
        assert convert_table_to_codeblock(text) == text

    def test_horizontal_rule_not_matched(self):
        text = "Section A\n\n---\n\nSection B"
        assert convert_table_to_codeblock(text) == text

    def test_single_column_table_untouched(self):
        text = "| only |\n|------|\n| one  |"
        assert convert_table_to_codeblock(text) == text
