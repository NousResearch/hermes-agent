"""Tests for `agent.markdown_tables.realign_markdown_tables`.

These cover the alignment guarantee on CJK / wide-character tables and
the conservative no-op behaviour on non-table input.
"""

from __future__ import annotations

from textwrap import dedent

from wcwidth import wcswidth

from agent.markdown_tables import (
    is_table_divider,
    looks_like_table_row,
    realign_markdown_tables,
    split_table_row,
)


def _column_offsets(line: str) -> list[int]:
    """Return the display-cell index of every ``|`` in ``line``."""

    cells: list[int] = []
    width = 0
    for ch in line:
        if ch == "|":
            cells.append(width)
        # wcswidth on a single char; clamp negatives.
        w = wcswidth(ch)
        width += w if w > 0 else 1
    return cells


# ---------------------------------------------------------------------------
# split_table_row / is_table_divider / looks_like_table_row
# ---------------------------------------------------------------------------


def test_split_strips_outer_pipes_and_trims():
    assert split_table_row("| a | b | c |") == ["a", "b", "c"]
    assert split_table_row("|配置|状态|") == ["配置", "状态"]
    assert split_table_row("a | b | c") == ["a", "b", "c"]




def test_looks_like_table_row():
    assert looks_like_table_row("| a | b |")
    assert looks_like_table_row("a | b | c")          # no leading pipe, ≥2 pipes
    assert not looks_like_table_row("not a table")
    assert not looks_like_table_row("a | b")          # one pipe, no leading pipe
    assert not looks_like_table_row("")


# ---------------------------------------------------------------------------
# realign_markdown_tables
# ---------------------------------------------------------------------------














# ---------------------------------------------------------------------------
# Vertical fallback for tables wider than the terminal
# ---------------------------------------------------------------------------




def test_horizontal_kept_when_table_fits():
    """A table that fits the terminal must keep the horizontal
    pipe-bordered rendering — vertical fallback only kicks in when
    soft-wrap is unavoidable."""

    src = dedent(
        """\
        | Name | Age |
        |------|-----|
        | Alice | 30 |
        | Bob | 25 |
        """
    )

    out = realign_markdown_tables(src, available_width=100)

    # Pipe-bordered rendering survives.
    body_rows = [ln for ln in out.split("\n") if ln.strip().startswith("|")]
    assert len(body_rows) == 4
    offsets = [_column_offsets(r) for r in body_rows]
    assert all(o == offsets[0] for o in offsets)


def test_vertical_fallback_wraps_long_cell_text_with_indent():
    src = dedent(
        """\
        | Key | Value |
        |-----|-------|
        | x | this value is long enough that wrapping the value to fit a narrow terminal width is required even in vertical mode |
        """
    )

    out = realign_markdown_tables(src, available_width=60)

    lines = out.split("\n")
    assert lines[0].startswith("Key: x")
    # First "Value:" line + at least one continuation indented by 2 spaces.
    value_idx = next(i for i, l in enumerate(lines) if l.startswith("Value:"))
    assert lines[value_idx + 1].startswith("  ")
    # Every line still fits the budget.
    for line in lines:
        assert wcswidth(line) <= 60


def test_header_only_table_preserves_headers_at_narrow_and_wide_widths():
    headers = ["Available support channels", "Response time expectations"]
    source = "| " + " | ".join(headers) + " |\n| --- | --- |"

    for width in (40, 100):
        rendered = realign_markdown_tables(source, available_width=width)
        assert all(header in rendered for header in headers)
        assert rendered.index(headers[0]) < rendered.index(headers[1])
        assert all(wcswidth(line) <= width for line in rendered.splitlines())


def test_header_only_fallback_preserves_surrounding_prose_and_other_tables():
    headers = ["Available support channels", "Response time expectations"]
    empty_table = "| " + " | ".join(headers) + " |\n| --- | --- |"
    populated_table = "| Channel | Response |\n| --- | --- |\n| Email | Within a working day |"
    prefix, middle, suffix = "Available fields:\n\n", "\n\nCurrent entries:\n\n", "\n\nEnd."
    rendered = realign_markdown_tables(prefix + empty_table + middle + populated_table + suffix, 40)

    assert rendered.startswith(prefix)
    assert middle in rendered
    assert rendered.endswith(suffix)
    assert all(header in rendered for header in headers)
    assert realign_markdown_tables(populated_table, 40) in rendered
    assert realign_markdown_tables(prefix + suffix, 40) == prefix + suffix






def test_multiple_tables_in_one_text():
    src = dedent(
        """\
        First:

        | 配置 | 值 |
        |------|----|
        | 通义 | 1 |

        Second:

        | model | n |
        |-------|---|
        | gpt   | 2 |
        """
    )
    out = realign_markdown_tables(src)
    # Each table block individually aligns.
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in out.split("\n"):
        if "|" in line:
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)

    assert len(blocks) == 2
    for block in blocks:
        offsets = [_column_offsets(row) for row in block]
        assert all(o == offsets[0] for o in offsets), (
            "block did not align:\n" + "\n".join(block)
        )
