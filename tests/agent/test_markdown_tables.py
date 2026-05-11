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


def test_is_table_divider_handles_alignment_colons():
    assert is_table_divider("|---|---|")
    assert is_table_divider("| :--- | ---: | :---: |")
    assert not is_table_divider("| - | - |")          # 1 dash is not a divider
    assert not is_table_divider("| a | b |")
    assert not is_table_divider("---")                # single column, no pipes


def test_looks_like_table_row():
    assert looks_like_table_row("| a | b |")
    assert looks_like_table_row("a | b | c")          # no leading pipe, ≥2 pipes
    assert not looks_like_table_row("not a table")
    assert not looks_like_table_row("a | b")          # one pipe, no leading pipe
    assert not looks_like_table_row("")


# ---------------------------------------------------------------------------
# realign_markdown_tables
# ---------------------------------------------------------------------------


def test_no_op_on_text_without_tables():
    text = "Hello world\nThis has no | pipes table.\n"
    assert realign_markdown_tables(text) == text


def test_no_op_when_pipes_but_no_divider():
    text = "echo a | grep b\necho c | wc -l\n"
    assert realign_markdown_tables(text) == text


def test_cjk_table_pipes_align_across_rows():
    # Model-emitted (under-padded for CJK) input.
    src = dedent(
        """\
        | 配置 | Config | 论文 (%) | 复现 (%) | 差值 | 状态 |
        |------|--------|---------|---------|------|------|
        | Vicuna (report) | dense | 79.30 | 未完成 | - | × |
        | ChatGLM | chat | 37.60 | 37.82 | +0.22 | ✓ |
        | 通义千问 | qwen | (无) | 报错 | - | × |
        """
    )

    out = realign_markdown_tables(src).rstrip("\n").split("\n")

    # All rows in the rebuilt block must have pipes at identical display
    # columns — that's the alignment guarantee.
    offsets = [_column_offsets(row) for row in out]
    assert all(o == offsets[0] for o in offsets), (
        "rebuilt table rows do not share pipe column offsets:\n"
        + "\n".join(out)
    )
    # And we expect 7 pipes per row (6 columns + outer borders).
    assert len(offsets[0]) == 7


def test_emoji_with_cjk_table_aligns():
    src = dedent(
        """\
        | 模型 | 状态 | 备注 |
        |------|------|------|
        | 千问 | ✅ | 通过 |
        | Claude | ✅ | 推理强 |
        | 文心一言 | ❌ | 报错 |
        """
    )

    out = realign_markdown_tables(src).rstrip("\n").split("\n")
    offsets = [_column_offsets(row) for row in out]
    # The emoji-with-variation-selector case (⚠️) intentionally tolerates
    # 1-cell drift; bare emoji like ✅ / ❌ have stable wcwidth and must
    # align.  Use bare emoji here so the assertion is hard.
    assert all(o == offsets[0] for o in offsets), (
        "emoji+CJK rows do not share pipe column offsets:\n" + "\n".join(out)
    )


def test_already_aligned_ascii_table_remains_aligned():
    src = dedent(
        """\
        | a   | b   |
        |-----|-----|
        | 1   | 2   |
        | foo | bar |
        """
    )
    out = realign_markdown_tables(src).rstrip("\n").split("\n")
    offsets = [_column_offsets(row) for row in out]
    assert all(o == offsets[0] for o in offsets)


def test_passes_non_table_lines_through_around_a_table():
    src = dedent(
        """\
        Here is a comparison:

        | 模型 | 状态 |
        |------|------|
        | 千问 | 通过 |

        And some prose after.
        """
    )

    out = realign_markdown_tables(src)
    assert out.startswith("Here is a comparison:\n")
    assert out.endswith("And some prose after.\n")
    # And the table lines are aligned.
    block = [ln for ln in out.split("\n") if "|" in ln]
    offsets = [_column_offsets(row) for row in block]
    assert all(o == offsets[0] for o in offsets)


# ---------------------------------------------------------------------------
# Vertical fallback for tables wider than the terminal
# ---------------------------------------------------------------------------


def test_overflow_falls_back_to_vertical_when_table_wider_than_terminal():
    """A horizontal table that would exceed the available width must
    drop to vertical key-value rendering so the terminal does not
    soft-wrap mid-cell (which destroys column alignment visually)."""

    src = dedent(
        """\
        | Item | Description | Notes |
        |------|-------------|-------|
        | a | short | ok |
        | b | this is a much longer description that stretches the column wider than the others by a lot | fine |
        | c | tiny | - |
        """
    )

    out = realign_markdown_tables(src, available_width=100)

    # No horizontal pipe-bordered rows: vertical mode emits "Header: value"
    # lines and a ─ separator instead.
    assert "|" not in out
    assert "Item: a" in out
    assert "Description: short" in out
    assert "Notes: ok" in out
    # Body rows separated by ─ rule
    assert "──" in out

    # Every emitted line fits the available width.
    for line in out.split("\n"):
        assert wcswidth(line) <= 100, f"line wider than budget: {line!r}"


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


def test_overflow_falls_back_to_vertical_for_cjk_too():
    """CJK content can also push a table over the terminal budget;
    the vertical fallback should kick in regardless of script."""

    src = dedent(
        """\
        | 模型 | 描述 | 备注 |
        |------|------|------|
        | 千问 | 一个相当长的描述用于把列宽撑得超过可用终端宽度从而触发竖排回退 | 通过 |
        | 文心 | 短 | × |
        """
    )

    out = realign_markdown_tables(src, available_width=50)

    assert "|" not in out
    assert "模型: 千问" in out
    assert "模型: 文心" in out
    for line in out.split("\n"):
        assert wcswidth(line) <= 50, f"line wider than budget: {line!r}"


def test_handles_ragged_rows_by_padding_short_rows():
    src = dedent(
        """\
        | a | b | c |
        |---|---|---|
        | 1 | 2 |
        | x | y | z |
        """
    )
    out = realign_markdown_tables(src).rstrip("\n").split("\n")
    offsets = [_column_offsets(row) for row in out]
    # Short rows must be padded out so they have the same pipe count
    # and column positions as the header.
    assert all(len(o) == len(offsets[0]) for o in offsets)
    assert all(o == offsets[0] for o in offsets)


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
            f"block did not align:\n" + "\n".join(block)
        )
