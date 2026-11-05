"""Shared panel wrap helpers count terminal cells, so CJK rows never push a panel's border out."""

import pytest
from prompt_toolkit.utils import get_cwidth

from hermes_cli.cli_render import _append_panel_line, _wrap_panel_text


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
