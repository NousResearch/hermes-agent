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
import cli as cli_mod
from hermes_cli.cli_inline_images import image_rows, print_inline_images
from hermes_cli.cli_render import _output_tail_fitting

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
    rows = image_rows(image)
    assert len(rows) == image.rows
    r, g, b = bytes.fromhex(image.color[1:])
    # The 256-color default would round the color and lose the image id kitty decodes from it.
    assert all(f"\x1b[38;2;{r};{g};{b}m" in _painted(row) for row, _raw in rows)
    assert [raw for _row, raw in rows] == [image.transmit] + [""] * (image.rows - 1)


@pytest.mark.parametrize("mode", ["kitty", "unicode"])
def test_image_rows_measure_as_their_cell_width(photo, mode):
    image = render_inline_image(photo, mode=mode, max_cols=40, max_rows=10)
    widths = {fragment_list_width(to_formatted_text(ANSI(row))) for row, _raw in image_rows(image)}
    assert widths == {image.cols + 2}


def test_iterm_rows_are_reserved_blank_and_the_image_draws_over_them(photo):
    # Konsole/iTerm2/WezTerm: the image owns no text cells, so the scrollback reserves exactly
    # ``rows`` blank lines and the last one jumps back up, draws, and restores the cursor.
    image = render_inline_image(photo, mode="iterm", max_cols=40, max_rows=10)
    assert image is not None
    rows = image_rows(image)
    assert len(rows) == image.rows > 1
    assert [row for row, _raw in rows] == [""] * image.rows
    draw = rows[-1][1]
    assert draw.startswith(f"\x1b7\x1b[{image.rows - 1}A\x1b[2C\x1b]1337;File=inline=1;") and draw.endswith("\x07\x1b8")
    assert f"width={image.cols};height={image.rows}" in draw


@pytest.fixture
def scrollback(monkeypatch):
    """The CLI's real scrollback printer into a buffer, with output history on, on a TTY."""
    buf = io.StringIO()
    output = Vt100_Output(buf, lambda: Size(rows=40, columns=120), term="xterm-kitty")
    monkeypatch.setattr(cli_mod, "_pt_print", lambda text: print_formatted_text(output, text, Style([])))
    monkeypatch.setattr("sys.__stdout__", type("Tty", (), {"isatty": staticmethod(lambda: True)})())
    for key in ("TMUX", "STY"):
        monkeypatch.delenv(key, raising=False)
    cli_mod._configure_output_history(True, 200)
    yield buf
    cli_mod._configure_output_history(True, 200)


@pytest.mark.parametrize(("mode", "transmit_start"), [("kitty", "\x1b_G"), ("iterm", "\x1b]1337;")])
def test_transmit_is_written_once_and_kept_out_of_replayed_history(scrollback, photo, mode, transmit_start):
    # A resize replay re-prints (and may crop) history rows: a transmit stored there was cut into
    # base64 text, re-sent whole on every redraw, or redrawn an image over replayed text.
    print_inline_images(type("Cli", (), {"_inline_images_mode": mode})(), [photo])
    assert scrollback.getvalue().count(transmit_start) == 1
    history = list(cli_mod._OUTPUT_HISTORY)
    assert history and not any(transmit_start in line or "\x1b7" in line for line in history)
    assert not any(transmit_start in line for line in _output_tail_fitting(history, 3, 120))


def test_cropped_kitty_row_keeps_its_exact_id_color(photo):
    # A partly kept placeholder row must still paint in the truecolor that names its image.
    image = render_inline_image(photo, mode="kitty", max_cols=40, max_rows=10)
    row = cli_mod._PaintedLine(image_rows(image)[0][0])
    row.width = 20  # painted at 20 columns: three rows, of which the replay keeps the last
    [kept] = _output_tail_fitting([row], 1, 20)
    r, g, b = bytes.fromhex(image.color[1:])
    assert f"\x1b[38;2;{r};{g};{b}m" in _painted(kept)


def test_pinned_kitty_in_tmux_wraps_the_transmit_for_passthrough(scrollback, photo, monkeypatch):
    # tmux forwards an APC to the outer kitty only inside its DCS passthrough wrapper.
    monkeypatch.setenv("TMUX", "tmux-socket,1,0")
    print_inline_images(type("Cli", (), {"_inline_images_mode": "kitty"})(), [photo])
    painted = scrollback.getvalue()
    assert painted.count("\x1bPtmux;\x1b\x1b_G") == 1 and "\x1b_G" not in painted.replace("\x1b\x1b_G", "")
