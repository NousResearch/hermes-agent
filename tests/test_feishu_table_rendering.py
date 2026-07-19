"""Unit tests for Feishu markdown table parsing and CardKit v2 card building."""
import json
import pytest
from plugins.platforms.feishu.adapter import (
    _parse_markdown_table,
    _build_table_card,
    _build_interactive_card_with_tables,
    _MARKDOWN_TABLE_RE,
)


class TestParseMarkdownTable:
    """Test _parse_markdown_table extracts tables correctly."""

    def test_simple_table(self):
        text = "| A | B |\n|---|---|\n| 1 | 2 |"
        segs = _parse_markdown_table(text)
        assert len(segs) == 1
        assert segs[0]["type"] == "table"
        assert segs[0]["headers"] == ["A", "B"]
        assert segs[0]["rows"] == [["1", "2"]]

    def test_no_table_returns_text(self):
        text = "Hello world, no table here."
        segs = _parse_markdown_table(text)
        assert len(segs) == 1
        assert segs[0]["type"] == "text"

    def test_no_pipe_returns_text(self):
        text = "Plain text without any pipes."
        segs = _parse_markdown_table(text)
        assert len(segs) == 1
        assert segs[0]["type"] == "text"

    def test_pipe_in_code_block_not_table(self):
        text = "```\n| a | b |\n```"
        segs = _parse_markdown_table(text)
        assert all(s["type"] == "text" for s in segs)

    def test_table_with_surrounding_text(self):
        text = "Here's a table:\n\n| Name | Age |\n|------|-----|\n| Alice | 30 |\n\nDone."
        segs = _parse_markdown_table(text)
        assert len(segs) == 3  # text, table, text
        assert segs[0]["type"] == "text"
        assert segs[1]["type"] == "table"
        assert segs[1]["headers"] == ["Name", "Age"]
        assert segs[1]["rows"] == [["Alice", "30"]]
        assert segs[2]["type"] == "text"

    def test_large_table_10_rows(self):
        header = "| # | Name |\n|---|------|\n"
        rows = "".join(f"| {i} | User{i} |\n" for i in range(1, 11))
        segs = _parse_markdown_table(header + rows)
        assert len(segs) == 1
        assert segs[0]["type"] == "table"
        assert len(segs[0]["rows"]) == 10
        assert segs[0]["rows"][0] == ["1", "User1"]
        assert segs[0]["rows"][9] == ["10", "User10"]

    def test_table_with_alignment_divider(self):
        text = "| Left | Center | Right |\n|:-----|:------:|------:|\n| a | b | c |"
        segs = _parse_markdown_table(text)
        assert segs[0]["type"] == "table"
        assert segs[0]["headers"] == ["Left", "Center", "Right"]
        assert segs[0]["rows"] == [["a", "b", "c"]]

    def test_uneven_row_columns_padded(self):
        """Rows with fewer cells than headers should be padded with empty strings."""
        text = "| A | B | C |\n|---|---|---|\n| 1 | 2 |"
        segs = _parse_markdown_table(text)
        assert segs[0]["rows"][0] == ["1", "2", ""]

    def test_uneven_row_columns_truncated(self):
        """Rows with more cells than headers should be truncated."""
        text = "| A | B |\n|---|---|\n| 1 | 2 | 3 |"
        segs = _parse_markdown_table(text)
        assert segs[0]["rows"][0] == ["1", "2"]

    def test_malformed_table_double_header_before_divider(self):
        """When two header-like rows appear before the divider, the divider
        is still recognized and stripped — no literal '---' in cells."""
        text = "| A | B |\n| C | D |\n|---|---|\n| 1 | 2 |"
        segs = _parse_markdown_table(text)
        assert segs[0]["type"] == "table"
        assert segs[0]["headers"] == ["A", "B"]
        # C/D row and 1/2 row are both data; divider is not
        assert segs[0]["rows"] == [["C", "D"], ["1", "2"]]
        # No literal dashes leak into any cell
        for row in segs[0]["rows"]:
            for cell in row:
                assert "---" not in cell

    def test_bare_pipe_not_table(self):
        """A single bare '|' should not produce a table with zero columns."""
        text = "Just a pipe: |"
        segs = _parse_markdown_table(text)
        assert all(s["type"] == "text" for s in segs)


class TestBuildTableCard:
    """Test _build_table_card generates valid CardKit v2 table JSON."""

    def test_card_structure(self):
        card = _build_table_card(["Name", "Age"], [["Alice", "30"]])
        assert card["tag"] == "table"
        assert len(card["columns"]) == 2
        assert card["columns"][0]["name"] == "col_0"
        assert card["columns"][0]["display_name"] == "Name"
        assert card["columns"][0]["data_type"] == "text"
        assert card["rows"][0]["col_0"] == "Alice"
        assert card["rows"][0]["col_1"] == "30"

    def test_header_style_is_bold(self):
        card = _build_table_card(["A"], [["1"]])
        assert card["header_style"]["bold"] is True

    def test_bold_markers_stripped_from_cells(self):
        """CardKit v2 doesn't support markdown in cells."""
        card = _build_table_card(["**Bold**"], [["**value**"]])
        assert card["columns"][0]["display_name"] == "Bold"
        assert card["rows"][0]["col_0"] == "value"

    def test_empty_cell_becomes_space(self):
        """Empty cells should become a single space, not empty string."""
        card = _build_table_card(["A", "B"], [["val", ""]])
        assert card["rows"][0]["col_1"] == " "

    def test_empty_headers_returns_none(self):
        """Degenerate table with no columns should return None, not a broken card."""
        assert _build_table_card([], []) is None
        assert _build_table_card([], [["1"]]) is None


class TestBuildInteractiveCardWithTables:
    """Test _build_interactive_card_with_tables builds complete cards."""

    def test_returns_none_without_table(self):
        assert _build_interactive_card_with_tables("No table here") is None

    def test_card_schema(self):
        text = "| A | B |\n|---|---|\n| 1 | 2 |"
        card = _build_interactive_card_with_tables(text)
        assert card is not None
        assert card["schema"] == "2.0"
        assert card["config"]["wide_screen_mode"] is True
        assert len(card["body"]["elements"]) == 1
        assert card["body"]["elements"][0]["tag"] == "table"

    def test_mixed_text_and_table(self):
        text = "Intro text\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\nOutro text"
        card = _build_interactive_card_with_tables(text)
        assert card is not None
        elements = card["body"]["elements"]
        # Should have: markdown element, table element, markdown element
        assert len(elements) == 3
        assert elements[0]["tag"] == "markdown"
        assert elements[1]["tag"] == "table"
        assert elements[2]["tag"] == "markdown"
