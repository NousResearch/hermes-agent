"""The classic CLI lays panels out in terminal CELLS, not code points.

A CJK character is two cells wide but one code point, so ``len()``, ``str.ljust`` and
``textwrap`` all under-measured Chinese text: panel borders drifted right, a wide row
overflowed the terminal, and space-less CJK clauses only wrapped at the next ASCII
space (leaving most of the row blank). These tests pin the cell-based contract.
"""
from wcwidth import wcswidth

from cli import _append_panel_line, _panel_box_width, _wrap_panel_text
from hermes_cli import cli_render

PARAGRAPH = (
    "这条来自我最早那个 PID 写错的监听器（P=295447），但它读的是同一份 topdirs19.txt，"
    "数据与真监听器（23:37:28）完全一致——验证结论不变。"
)


def cells(text: str) -> int:
    """Independent width oracle: terminal cells, not code points."""
    return wcswidth(text)


def test_display_width_counts_terminal_cells():
    assert cli_render._display_width("中文") == 4
    assert cli_render._display_width("abc") == 3
    assert cli_render._display_width("，") == 2


def test_pad_display_fills_to_cells_not_characters():
    padded = cli_render._pad_display("中文", 10)

    assert padded == "中文" + " " * 6
    assert cells(padded) == 10


def test_wrap_fills_a_row_instead_of_breaking_at_the_previous_space():
    rows = _wrap_panel_text(PARAGRAPH, 60)

    assert len(rows) > 1
    assert all(cells(row) <= 60 for row in rows), rows
    # The old wrapper stopped after "同一份" (28 cells of a 60-cell row).
    assert cells(rows[0]) >= 55, rows


def test_wrap_never_starts_a_row_with_closing_punctuation():
    rows = _wrap_panel_text("这是一句用于测试标点不能出现在行首的中文，后面继续写一些字。", 20)

    assert len(rows) > 1
    assert all(row[0] not in "，。、" for row in rows[1:]), rows


def test_wrap_honours_the_subsequent_indent_within_the_width():
    rows = _wrap_panel_text(PARAGRAPH, 40, subsequent_indent="    ")

    assert all(cells(row) <= 40 for row in rows), rows
    assert all(row.startswith("    ") for row in rows[1:]), rows


def test_panel_rows_pad_to_the_same_cell_width():
    lines: list[tuple[str, str]] = []

    _append_panel_line(lines, "border", "content", "中文内容", 20)

    text = "".join(chunk for _style, chunk in lines)

    assert cells(text.rstrip("\n")) == 22  # "│ " + 18 cells + " │"


def test_panel_box_width_measures_cjk_titles_in_cells():
    # 30 ideographs = 60 cells, so the box has to be wider than the 46-column floor.
    wide = _panel_box_width("汉" * 30, [], min_width=46, max_width=76)

    assert wide >= 60
    assert wide <= 76


def test_panel_borders_and_rows_share_one_cell_width():
    from hermes_cli.cli_tui_mixin import _Panel

    panel = _Panel("border", 30, "中文标题", "title")
    panel.row("content", "中文内容")
    rendered = panel.close()
    lines = [line for line in "".join(chunk for _style, chunk in rendered).split("\n") if line]

    assert {cells(line) for line in lines} == {32}, lines  # box_width + 2 borders, every line


def test_wrap_line_cells_keeps_every_ideograph_it_breaks_on():
    """A row break must not swallow the character that triggered it.

    ``_wrap_line_cells`` used to flush the finished row and reset ``current``
    without re-emitting the ideograph that did not fit, so any CJK text landing
    exactly on a row boundary lost a character and gained a spurious empty row.
    """
    source = "汉" * 11

    rows = cli_render._wrap_line_cells(source, 20, 20, "")

    assert "".join(rows) == source, rows
    assert rows == ["汉" * 10, "汉"], rows


def test_kinsoku_carries_the_ideograph_down_with_the_punctuation():
    # Ten ideographs fill the row exactly; the comma that does not fit must
    # drag the last ideograph with it instead of opening the row by itself.
    rows = cli_render._wrap_line_cells("汉" * 10 + "，", 20, 20, "")

    assert rows == ["汉" * 9, "汉，"], rows
