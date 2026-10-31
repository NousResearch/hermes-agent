"""Inline images in the classic CLI scrollback: the user's attachments and image tool results.

Images print as ordinary text rows (kitty Unicode placeholders, or half-blocks) so the scrollback
replay and its row accounting keep working. The kitty escapes ride in prompt_toolkit's zero-width
``\\001…\\002`` brackets: its ANSI printer would otherwise quantize the truecolor that carries the
image id to the output's color depth (256 by default), and strip the APC transmit outright.
Half-block colors go through that quantization on purpose, like the rest of the scrollback.
"""

from __future__ import annotations

import shutil
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


def image_lines(image: InlineImage) -> list[str]:
    """*image* as scrollback lines; the first one carries the kitty transmit."""
    if image.mode == "kitty":
        r, g, b = bytes.fromhex(image.color[1:])
        color, reset = _zero_width(f"\x1b[38;2;{r};{g};{b}m"), _zero_width("\x1b[39m")
        lines = [_INDENT + color + row + reset for row in image.placeholder]
        lines[0] = _zero_width(image.transmit) + lines[0]
        return lines
    return [_INDENT + "".join(_half_block(t, b) for t, b in row) + "\x1b[0m" for row in image.cells]


def print_inline_images(cli, paths: Iterable) -> None:
    """Print each image in *paths* under the current scrollback line, per ``display.inline_images``."""
    from cli import _cprint
    stdout = sys.__stdout__
    if stdout is None or not stdout.isatty():
        return
    mode = resolve_inline_mode(getattr(cli, "_inline_images_mode", "auto"))
    if mode == "off":
        return
    max_cols = min(_MAX_COLS, shutil.get_terminal_size().columns - len(_INDENT) - 1)
    for path in paths:
        image = render_inline_image(path, mode=mode, max_cols=max_cols, max_rows=_MAX_ROWS)
        # One print per row, like the diff printer: a whole truecolor image in a single write
        # overruns the Windows console write and the tail rows are lost.
        for line in image_lines(image) if image is not None else ():
            _cprint(line)
