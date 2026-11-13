"""Shared panel wrap helpers count terminal cells, so CJK rows never push a panel's border out."""

import pytest
from prompt_toolkit.utils import get_cwidth

from hermes_agent.hermes_cli.cli_render import _append_panel_line, _wrap_panel_text


@pytest.mark.parametrize("text", ["Which option?\n1) keep\n2) drop", "选哪个?\n1) 保留\n2) 删除"])
def test_panel_wrap_never_keeps_a_newline_inside_a_row(text):
    """Shared by clarify/approval/palette panels: a raw newline in a row breaks the right border."""
    rows = _wrap_panel_text(text, 60)
    assert rows and not any("\n" in row or "\t" in row for row in rows)


def test_cjk_text_wraps_and_pads_by_terminal_cells():
    text = "请帮我把这个函数重构成更小的部分，并解释每一步为什么这样做，同时保留原有的行为和测试。" * 2
    rows = _wrap_panel_text(text, 30)
    assert len(rows) > 1 and all(get_cwidth(row) <= 30 for row in rows)
    lines = []
    for row in rows:
        _append_panel_line(lines, "b", "c", row, 34)
    rendered = "".join(text for _style, text in lines).rstrip("\n").split("\n")
    assert {get_cwidth(line) for line in rendered} == {36}


def test_plain_ascii_wraps_exactly_like_textwrap():
    import textwrap
    text = " ".join(f"word{i}" for i in range(40))
    assert _wrap_panel_text(text, 30) == textwrap.wrap(text, 30, break_long_words=False, break_on_hyphens=False)


def test_a_combining_accent_stays_with_its_letter_and_rows_fit():
    rows = _wrap_panel_text("中中中中e\u0301", 8)
    assert all(get_cwidth(row) <= 8 for row in rows)
    assert "".join(rows) == "中中中中e\u0301" and not rows[-1].startswith("\u0301")


def test_other_panels_keep_long_words_whole():
    """Only the backtrack pager hard-splits: approval/clarify panels keep main's textwrap output."""
    url = "https://example.com/" + "a" * 120
    assert _wrap_panel_text("see " + url, 40) == ["see", url]


def test_a_long_cjk_prompt_wraps_in_one_pass(monkeypatch):
    """Each character's width is read a bounded number of times, not once per wrapped row."""
    from hermes_agent.hermes_cli import cli_render
    calls = []
    real = cli_render._cells
    monkeypatch.setattr(cli_render, "_cells", lambda text: calls.append(len(text)) or real(text))
    text = "请帮我把这个函数重构成更小的部分" * 375  # 6,000 characters, ~130 rows at 92 cells
    rows = cli_render._wrap_panel_text(text, 92, break_long_words=True)
    assert len(rows) > 100 and all(get_cwidth(row) <= 92 for row in rows)
    assert sum(calls) <= 3 * len(text)


@pytest.mark.parametrize("keep_ws", [False, True])
def test_fullwidth_spaces_count_as_two_cells(keep_ws):
    """U+3000 is whitespace but two cells wide: a row of them must still fit the panel."""
    rows = _wrap_panel_text("a\u3000" * 30, 40, keep_ws=keep_ws)
    assert len(rows) > 1 and all(get_cwidth(row) <= 40 for row in rows)
