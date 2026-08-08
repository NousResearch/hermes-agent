"""Regression tests for hermes_cli/cli_display_formatters.py (c2 cluster).

Wave 1 godfile extraction, shard s1 cluster c2: the lazy table-formatting
wrappers moved verbatim from cli.py. These tests pin the delegation contract
— each wrapper must return exactly what the underlying
``agent.markdown_tables`` function returns for the same input.
"""

from agent.markdown_tables import (
    is_table_divider as _real_is_table_divider,
    looks_like_table_row as _real_looks_like_table_row,
    realign_markdown_tables as _real_realign_markdown_tables,
)
from hermes_cli.cli_display_formatters import (
    is_table_divider,
    looks_like_table_row,
    realign_markdown_tables,
)


class TestIsTableDivider:
    def test_divider_row(self):
        line = "| --- | --- |"
        assert is_table_divider(line) is _real_is_table_divider(line)
        assert is_table_divider(line) is True

    def test_non_divider(self):
        line = "| a | b |"
        assert is_table_divider(line) is _real_is_table_divider(line)
        assert is_table_divider(line) is False

    def test_empty(self):
        assert is_table_divider("") is _real_is_table_divider("")


class TestLooksLikeTableRow:
    def test_leading_pipe(self):
        line = "| a | b |"
        assert looks_like_table_row(line) is _real_looks_like_table_row(line)
        assert looks_like_table_row(line) is True

    def test_two_pipes_no_leading(self):
        line = "a | b | c"
        assert looks_like_table_row(line) is _real_looks_like_table_row(line)
        assert looks_like_table_row(line) is True

    def test_not_a_table(self):
        line = "not a table"
        assert looks_like_table_row(line) is _real_looks_like_table_row(line)
        assert looks_like_table_row(line) is False


class TestRealignMarkdownTables:
    def test_realigns_underpadded_table(self):
        src = "| a | b |\n|---|---|\n| 1 | 2 |"
        expected = _real_realign_markdown_tables(src, available_width=40)
        assert realign_markdown_tables(src, available_width=40) == expected
        assert "| a " in expected

    def test_passthrough_of_kwargs(self):
        src = "| x | y |\n|---|---|\n| 1 | 2 |"
        assert realign_markdown_tables(src) == _real_realign_markdown_tables(src)
        assert realign_markdown_tables(
            src, available_width=20
        ) == _real_realign_markdown_tables(src, available_width=20)
