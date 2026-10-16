"""Layout of the compact banner shown on narrow terminals."""

import os

import pytest
from rich.cells import cell_len
from rich.text import Text
from unittest.mock import patch

from cli import _build_compact_banner


def _banner_lines(columns):
    with patch("cli.shutil.get_terminal_size", return_value=os.terminal_size((columns, 40))), \
         patch.dict(_build_compact_banner.__globals__, {"format_banner_version_label": lambda: "Hermes Agent v0.1.0 (test)"}):
        banner = _build_compact_banner()
    return [line for line in Text.from_markup(banner).plain.split("\n") if line]


@pytest.mark.parametrize("columns", [40, 69, 90, 200])
def test_compact_banner_box_edges_line_up(columns):
    lines = _banner_lines(columns)

    assert len(lines) == 4
    widths = [cell_len(line) for line in lines]
    # The top and bottom borders and both content rows must be the same
    # width, or the right-hand edge of the box is ragged.
    assert len(set(widths)) == 1, widths
    assert widths[0] <= columns
    assert lines[0][0] == "╔" and lines[0][-1] == "╗"
    assert lines[-1][0] == "╚" and lines[-1][-1] == "╝"
    for row in lines[1:3]:
        assert row[0] == "║" and row[-1] == "║"
