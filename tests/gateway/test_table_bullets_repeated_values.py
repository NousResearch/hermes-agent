"""Converting a GFM table to bullets must not delete columns.

``_render_table_block`` renders each row as a bold heading plus one bullet per
column, and drops the bullet that merely repeats the heading. It identified that
bullet by comparing VALUES::

    if not has_row_label_col and value == heading:
        continue

The heading is the row's first non-empty cell, so any OTHER cell holding the
same string matched too and was silently dropped. Repeated values are the norm
in the status and comparison tables agents emit, so a real column disappeared
from the delivered message with no error and no trace.

Skipping by position keeps the intended de-duplication and nothing else.
"""

from __future__ import annotations

import pytest

from gateway.platforms.helpers import convert_table_to_bullets


class TestRepeatedValuesKeepTheirColumns:
    def test_value_equal_to_the_row_heading_is_still_rendered(self):
        out = convert_table_to_bullets(
            "| Env | Region | Status |\n"
            "|---|---|---|\n"
            "| prod | us-east | prod |\n"
        )
        assert "• Region: us-east" in out
        assert "• Status: prod" in out          # was silently dropped
        assert out.startswith("**prod**")
        assert "• Env: prod" not in out         # the heading's own bullet

    def test_second_column_repeating_the_first(self):
        out = convert_table_to_bullets(
            "| Name | Alias |\n|---|---|\n| bob | bob |\n"
        )
        assert "• Alias: bob" in out
        assert "• Name: bob" not in out

    def test_three_way_repeat_keeps_both_non_heading_cells(self):
        out = convert_table_to_bullets(
            "| A | B | C |\n|---|---|---|\n| x | x | x |\n"
        )
        assert out.count("• ") == 2
        assert "• B: x" in out
        assert "• C: x" in out

    def test_every_data_cell_reaches_the_output(self):
        """Contract: one bullet per column, minus the heading's own."""
        table = (
            "| Model | Vision | Tools | Notes |\n"
            "|---|---|---|---|\n"
            "| kimi | yes | yes | kimi |\n"
        )
        out = convert_table_to_bullets(table)
        assert out.count("• ") == 3
        for expected in ("• Vision: yes", "• Tools: yes", "• Notes: kimi"):
            assert expected in out


class TestUnchangedBehaviour:
    def test_ordinary_table_is_unaffected(self):
        out = convert_table_to_bullets(
            "| Feature | Free | Pro |\n"
            "|---|---|---|\n"
            "| Export | Yes | Yes |\n"
            "| API | No | Yes |\n"
        )
        assert "**Export**" in out and "**API**" in out
        assert "• Free: Yes" in out and "• Pro: Yes" in out
        assert "• Feature: Export" not in out
        assert "• Free: No" in out

    def test_leading_empty_cell_still_picks_the_first_non_empty_heading(self):
        out = convert_table_to_bullets(
            "| A | B | C |\n|---|---|---|\n|  | x | y |\n"
        )
        assert out.startswith("**x**")
        assert "• B: x" not in out       # B produced the heading
        assert "• C: y" in out
        assert "• A: " in out

    def test_row_label_column_layout_is_untouched(self):
        """One extra leading cell => explicit row-label mode, no value match."""
        out = convert_table_to_bullets(
            "| Free | Pro |\n"
            "|---|---|\n"
            "| Export | Yes | Yes |\n"
        )
        assert out.startswith("**Export**")
        assert "• Free: Yes" in out and "• Pro: Yes" in out

    @pytest.mark.parametrize(
        "text",
        [
            "no table here at all",
            "| only | header |\n|---|---|\n",
            "```\n| A | B |\n|---|---|\n| x | x |\n```\n",
        ],
    )
    def test_non_tables_and_fenced_tables_pass_through(self, text):
        assert convert_table_to_bullets(text) == text
