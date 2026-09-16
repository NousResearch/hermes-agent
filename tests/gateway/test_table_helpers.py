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


    def test_is_table_row_with_pipe(self):
        assert is_table_row("| Alice | 150 |")


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


