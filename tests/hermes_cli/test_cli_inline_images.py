"""Inline images in the classic CLI scrollback (hermes_cli/cli_inline_images.py)."""

from __future__ import annotations

import io

import pytest
from prompt_toolkit.data_structures import Size
from prompt_toolkit.formatted_text import ANSI, fragment_list_width, to_formatted_text
from prompt_toolkit.output.vt100 import Vt100_Output
from prompt_toolkit.renderer import print_formatted_text
from prompt_toolkit.styles import Style

from agent.terminal_image import render_inline_image
from hermes_cli.cli_inline_images import image_lines

PIL = pytest.importorskip("PIL.Image")


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "photo.png"
    PIL.new("RGBA", (640, 320), (10, 120, 200, 255)).save(path)
    return str(path)


def _painted(line: str) -> str:
    """Bytes the CLI's scrollback printer writes for *line* at prompt_toolkit's default color depth."""
    buf = io.StringIO()
    output = Vt100_Output(buf, lambda: Size(rows=40, columns=120), term="xterm-kitty")
    print_formatted_text(output, ANSI(line), Style([]))
    output.flush()
    return buf.getvalue()


def test_kitty_rows_reach_the_terminal_with_exact_id_color_and_transmit(photo):
    image = render_inline_image(photo, mode="kitty", max_cols=40, max_rows=10)
    lines = image_lines(image)
    assert len(lines) == image.rows
    r, g, b = bytes.fromhex(image.color[1:])
    first = _painted(lines[0])
    # The 256-color default would round the color and lose the image id kitty decodes from it.
    assert f"\x1b[38;2;{r};{g};{b}m" in first
    assert image.transmit in first
    assert all(f"\x1b[38;2;{r};{g};{b}m" in _painted(line) for line in lines[1:])


@pytest.mark.parametrize("mode", ["kitty", "unicode"])
def test_image_rows_measure_as_their_cell_width(photo, mode):
    # Scrollback replay sizes rows by measured width: the transmit must count as zero cells.
    image = render_inline_image(photo, mode=mode, max_cols=40, max_rows=10)
    widths = {fragment_list_width(to_formatted_text(ANSI(line))) for line in image_lines(image)}
    assert widths == {image.cols + 2}


def test_iterm_rows_are_reserved_blank_and_the_image_draws_over_them(photo):
    # Konsole/iTerm2/WezTerm: the image owns no text cells, so the scrollback reserves exactly
    # ``rows`` blank lines and the last one jumps back up, draws, and restores the cursor.
    image = render_inline_image(photo, mode="iterm", max_cols=40, max_rows=10)
    assert image is not None
    lines = image_lines(image)
    assert len(lines) == image.rows > 1
    assert [fragment_list_width(to_formatted_text(ANSI(line))) for line in lines] == [0] * (image.rows - 1) + [2]
    painted = _painted(lines[-1])
    assert f"\x1b7\x1b[{image.rows - 1}A\x1b]1337;File=inline=1;" in painted
    assert f"width={image.cols};height={image.rows}" in painted and "\x07\x1b8" in painted
