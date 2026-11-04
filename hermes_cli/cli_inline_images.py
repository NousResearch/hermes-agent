"""Inline images in the classic CLI scrollback: the user's attachments and image tool results.

Images print as ordinary text rows (kitty Unicode placeholders, half-blocks, or blank rows an
iTerm2 image is drawn over) so the scrollback replay and its row accounting keep working. The
transmit itself goes out once, raw, beside its row (``_cprint(raw=...)``) and never enters the
output history: a resize replay repaints the rows alone, so it can't crop a 1 MB escape into
text or re-send it. kitty keeps the image by id, so replayed placeholders still show it. An
iTerm2 image is pixels drawn over its reserved rows; a replay repaints those rows blank, so the
picture is lost in the redrawn viewport (scrollback above keeps it) rather than painted over
whatever text the replay put above a partly kept image.

The kitty id color rides in prompt_toolkit's zero-width ``\\001…\\002`` brackets: its ANSI
printer would otherwise quantize the truecolor that carries the image id to the output's color
depth (256 by default). Half-block colors go through that quantization on purpose, like the rest
of the scrollback.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable

from agent.terminal_image import InlineImage, render_inline_image, resolve_inline_mode

_INDENT = "  "
_MAX_COLS = 64
_MAX_ROWS = 16


def _zero_width(seq: str) -> str:
    return f"\001{seq}\002"


def _half_block(top, bottom) -> str:
    (tr, tg, tb, ta), (br, bg, bb, ba) = top, bottom
    if ta < 32 and ba < 32:
        return "\x1b[0m "
    if ba < 32:
        return f"\x1b[0m\x1b[38;2;{tr};{tg};{tb}m▀"
    if ta < 32:
        return f"\x1b[0m\x1b[38;2;{br};{bg};{bb}m▄"
    return f"\x1b[38;2;{tr};{tg};{tb}m\x1b[48;2;{br};{bg};{bb}m▀"


def _passthrough(seq: str) -> str:
    from hermes_cli.cli_terminal_mixin import _tmux_passthrough
    return _tmux_passthrough(seq) if os.environ.get("TMUX") else seq


def image_rows(image: InlineImage) -> list[tuple[str, str]]:
    """*image* as ``(row, raw)`` pairs: ``row`` is history-safe scrollback text, ``raw`` the
    escape written verbatim right after it (the transmit: first kitty row, last iTerm2 row)."""
    if image.mode == "kitty":
        r, g, b = bytes.fromhex(image.color[1:])
        color, reset = _zero_width(f"\x1b[38;2;{r};{g};{b}m"), _zero_width("\x1b[39m")
        rows = [(_INDENT + color + row + reset, "") for row in image.placeholder]
        rows[0] = (rows[0][0], _passthrough(image.transmit))
        return rows
    if image.mode == "iterm":
        # Reserve the rows as empty lines (text written over an image cell erases it), then from
        # the start of the last one jump to the first, past the indent, draw, and restore.
        up = f"\x1b[{image.rows - 1}A" if image.rows > 1 else ""
        draw = f"\x1b7{up}\x1b[{len(_INDENT)}C{_passthrough(image.transmit)}\x1b8"
        return [("", "")] * (image.rows - 1) + [("", draw)]
    return [(_INDENT + "".join(_half_block(t, b) for t, b in row) + "\x1b[0m", "") for row in image.cells]


def print_inline_images(cli, paths: Iterable) -> None:
    """Print each image in *paths* under the current scrollback line, per ``display.inline_images``."""
    from cli import _cprint, _terminal_columns
    stdout = sys.__stdout__
    if stdout is None or not stdout.isatty():
        return
    mode = resolve_inline_mode(getattr(cli, "_inline_images_mode", "auto"), overlays=True)
    if mode == "off":
        return
    max_cols = min(_MAX_COLS, _terminal_columns() - len(_INDENT) - 1)
    for path in paths:
        image = render_inline_image(path, mode=mode, max_cols=max_cols, max_rows=_MAX_ROWS)
        # One print per row, like the diff printer: a whole truecolor image in a single write
        # overruns the Windows console write and the tail rows are lost.
        for row, raw in image_rows(image) if image is not None else ():
            _cprint(row, raw=raw)
