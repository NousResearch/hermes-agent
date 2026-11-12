"""Terminal graphics: capability detection and image encoders for the CLI and the TUI.

One home for the escape-level work so the pet mascot and inline images share it. Protocols
in fidelity order: ``kitty`` (kitty, Ghostty, WezTerm), ``iterm`` (iTerm2, WezTerm),
``sixel`` (xterm -ti vt340, foot, mlterm, …), ``unicode`` (24-bit half-blocks).

Inline images (:func:`render_inline_image`) use kitty Unicode placeholders and half-blocks, whose
cells a line-based renderer can measure, plus iTerm2 images for a painter that reserves blank rows
and draws over them (the classic CLI; Ink can't). Sixel terminals get half-blocks. PIL is
imported lazily; a missing Pillow or an unreadable file yields ``None``.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import warnings
import zlib
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DA1_REPLY = re.compile(rb"\x1b\[\?[0-9;]*c")


def _is_wezterm() -> bool:
    return os.environ.get("TERM_PROGRAM", "").lower() == "wezterm" or bool(os.environ.get("WEZTERM_PANE"))


def detect_terminal_graphics() -> str:
    """Richest protocol (``kitty``/``iterm``/``sixel``/``unicode``) from env vars only — never a DA1 query that could hang a pipe."""
    term = os.environ.get("TERM", "").lower()
    term_program = os.environ.get("TERM_PROGRAM", "").lower()
    # VS Code/Cursor set TERM_PROGRAM=vscode but don't scrub inherited
    # ITERM_SESSION_ID/KITTY_WINDOW_ID; trusting those emits a protocol xterm.js
    # can't show (blank frame). Inline images there are opt-in, so default to
    # half-blocks; users who enabled them can pin the render mode in config.
    if term_program == "vscode":
        return "unicode"
    if os.environ.get("KITTY_WINDOW_ID") or "kitty" in term or "ghostty" in term or term_program == "ghostty" or _is_wezterm():
        return "kitty"  # WezTerm speaks kitty and iterm; kitty has richer placement
    if term_program == "iterm.app" or os.environ.get("ITERM_SESSION_ID"):
        return "iterm"
    if term_program == "mintty" or "foot" in term or "mlterm" in term or "sixel" in term:
        return "sixel"
    return "unicode"


def supports_iterm_images() -> bool:
    """True when the terminal paints iTerm2 inline images (OSC 1337): iTerm2, WezTerm, Konsole,
    and VS Code once :func:`probe_vscode_images` saw its image support switched on."""
    if os.environ.get("TERM_PROGRAM", "").lower() == "vscode":
        return _VSCODE_IMAGES is True
    return detect_terminal_graphics() == "iterm" or _is_wezterm() or "KONSOLE_VERSION" in os.environ


# VS Code draws images only with ``terminal.integrated.enableImages`` on, and nothing in the
# environment says so. Its image addon is the only part that answers XTSMGRAPHICS, so one query
# fenced by DA1 tells: measured live, the reply lands in ~7 ms and every run at a 10 ms deadline
# was right; 8 ms missed some and let a late reply leak into the next read. 50 ms leaves headroom
# without being felt, since the read ends as soon as the DA1 fence arrives.
_VSCODE_IMAGES: bool | None = None
_VSCODE_PROBE_TIMEOUT_S = 0.05
_VSCODE_PROBE = "\x1b[?1;1;0S\x1b[c"
_SIXEL_REPLY = b"\x1b[?1;0;"


def probe_vscode_images() -> bool | None:
    """Whether VS Code's terminal will draw images; ``None`` outside a local VS Code terminal.

    Run once before prompt_toolkit owns the tty (the reply is read from stdin). Skipped over SSH,
    where the reply often misses the deadline (the prompt's parser then drops it, see
    ``hermes_agent.hermes_cli.pt_input_extras``). Cached.
    """
    global _VSCODE_IMAGES
    if _VSCODE_IMAGES is not None or os.environ.get("TERM_PROGRAM", "").lower() != "vscode":
        return _VSCODE_IMAGES
    if any(os.environ.get(v) for v in ("SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY")):
        return None
    try:
        import select
        import sys
        import termios
        import time
        import tty
    except ImportError:  # Windows: no termios
        return None
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    buf = b""
    try:
        tty.setcbreak(fd)
        sys.stdout.write(_VSCODE_PROBE)
        sys.stdout.flush()
        deadline = time.monotonic() + _VSCODE_PROBE_TIMEOUT_S
        while (left := deadline - time.monotonic()) > 0 and select.select([fd], [], [], left)[0]:
            chunk = os.read(fd, 256)
            buf += chunk
            if not chunk or _DA1_REPLY.search(buf):
                break
    finally:
        # A reply past the deadline: drain it while still in cbreak (in canonical mode a reply
        # without a newline is invisible to select and would reach the prompt as typed text).
        # Skipped once the DA1 fence arrived, so a normal start pays no extra wait.
        drain = time.monotonic() + (0 if _DA1_REPLY.search(buf) else 0.05)
        while (left := drain - time.monotonic()) > 0 and select.select([fd], [], [], left)[0]:
            chunk = os.read(fd, 256)
            buf += chunk
            if not chunk or _DA1_REPLY.search(buf):
                break
        termios.tcsetattr(fd, termios.TCSAFLUSH, old)
    _VSCODE_IMAGES = _SIXEL_REPLY in buf
    return _VSCODE_IMAGES


def supports_kitty_placeholders() -> bool:
    """True when the terminal paints kitty Unicode placeholders; WezTerm speaks kitty APC but renders placeholders as tofu."""
    return detect_terminal_graphics() == "kitty" and not _is_wezterm()


def _png_b64(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


# Nominal terminal cell size in pixels. kitty fits an image to its cell rectangle
# preserving aspect, so a frame that isn't a whole cell multiple rounds up, clipping
# the bottom row ("clipped feet"); snapping to an exact multiple avoids that.
CELL_W, CELL_H = 8, 16


def cell_box(image) -> tuple[int, int]:
    """Terminal cell box (~8×16 px per cell) for an image at its current pixel size.

    kitty stretches the image to fill ``c``×``r`` cells, so track the scaled pixel size, not a native-aspect column count (that upscales small pets).
    """
    return max(1, image.width // CELL_W), max(1, image.height // CELL_H)


def _kitty_apc(ctrl: str, data: str) -> str:
    """kitty APC escape for *data*, chunked into ≤4096-byte ``m`` pieces (``m=1`` = more chunks follow)."""
    pieces = [data[i : i + 4096] for i in range(0, len(data), 4096)] or [""]
    last = len(pieces) - 1
    return "".join(f"\x1b_G{ctrl + ',' if i == 0 else ''}m={int(i != last)};{piece}\x1b\\" for i, piece in enumerate(pieces))


def encode_kitty(image) -> str:
    """kitty transmit+display at the cursor; ``c``/``r`` pin the cell box so frames overwrite each other."""
    cols, rows = cell_box(image)
    return _kitty_apc(f"f=100,a=T,q=2,c={cols},r={rows}", _png_b64(image))


# kitty Unicode placeholders: Ink owns the screen and measures every cell, so it
# can't host raw kitty image escapes. Transmit once as a virtual placement (U=1),
# then print ordinary-width placeholder cells (U+10EEEE + row diacritic) whose
# foreground color encodes the image id; the terminal paints the image underneath.
#   https://sw.kovidgoyal.net/kitty/graphics-protocol/#unicode-placeholders
_KITTY_PLACEHOLDER = "\U0010eeee"
# Row diacritics by index, verbatim from kitty's gen/rowcolumn-diacritics.txt.
_ROWCOL_DIACRITICS: tuple[int, ...] = (
    0x0305, 0x030D, 0x030E, 0x0310, 0x0312, 0x033D, 0x033E, 0x033F, 0x0346, 0x034A, 0x034B, 0x034C, 0x0350, 0x0351, 0x0352, 0x0357, 0x035B, 0x0363, 0x0364, 0x0365,
    0x0366, 0x0367, 0x0368, 0x0369, 0x036A, 0x036B, 0x036C, 0x036D, 0x036E, 0x036F, 0x0483, 0x0484, 0x0485, 0x0486, 0x0487, 0x0592, 0x0593, 0x0594, 0x0595, 0x0597,
    0x0598, 0x0599, 0x059C, 0x059D, 0x059E, 0x059F, 0x05A0, 0x05A1, 0x05A8, 0x05A9, 0x05AB, 0x05AC, 0x05AF, 0x05C4, 0x0610, 0x0611, 0x0612, 0x0613, 0x0614, 0x0615,
    0x0616, 0x0617, 0x0657, 0x0658, 0x0659, 0x065A, 0x065B, 0x065D, 0x065E, 0x06D6, 0x06D7, 0x06D8, 0x06D9, 0x06DA, 0x06DB, 0x06DC, 0x06DF, 0x06E0, 0x06E1, 0x06E2,
    0x06E4, 0x06E7, 0x06E8, 0x06EB, 0x06EC, 0x0730, 0x0732, 0x0733, 0x0735, 0x0736, 0x073A, 0x073D, 0x073F, 0x0740, 0x0741, 0x0743, 0x0745, 0x0747, 0x0749, 0x074A,
    0x07EB, 0x07EC, 0x07ED, 0x07EE, 0x07EF, 0x07F0, 0x07F1, 0x07F3, 0x0816, 0x0817, 0x0818, 0x0819, 0x081B, 0x081C, 0x081D, 0x081E, 0x081F, 0x0820, 0x0821, 0x0822,
    0x0823, 0x0825, 0x0826, 0x0827, 0x0829, 0x082A, 0x082B, 0x082C, 0x082D, 0x0951, 0x0953, 0x0954, 0x0F82, 0x0F83, 0x0F86, 0x0F87, 0x135D, 0x135E, 0x135F, 0x17DD,
    0x193A, 0x1A17, 0x1A75, 0x1A76, 0x1A77, 0x1A78, 0x1A79, 0x1A7A, 0x1A7B, 0x1A7C, 0x1B6B, 0x1B6D, 0x1B6E, 0x1B6F, 0x1B70, 0x1B71, 0x1B72, 0x1B73, 0x1CD0, 0x1CD1,
    0x1CD2, 0x1CDA, 0x1CDB, 0x1CE0, 0x1DC0, 0x1DC1, 0x1DC3, 0x1DC4, 0x1DC5, 0x1DC6, 0x1DC7, 0x1DC8, 0x1DC9, 0x1DCB, 0x1DCC, 0x1DD1, 0x1DD2, 0x1DD3, 0x1DD4, 0x1DD5,
    0x1DD6, 0x1DD7, 0x1DD8, 0x1DD9, 0x1DDA, 0x1DDB, 0x1DDC, 0x1DDD, 0x1DDE, 0x1DDF, 0x1DE0, 0x1DE1, 0x1DE2, 0x1DE3, 0x1DE4, 0x1DE5, 0x1DE6, 0x1DFE, 0x20D0, 0x20D1,
    0x20D4, 0x20D5, 0x20D6, 0x20D7, 0x20DB, 0x20DC, 0x20E1, 0x20E7, 0x20E9, 0x20F0, 0x2CEF, 0x2CF0, 0x2CF1, 0x2DE0, 0x2DE1, 0x2DE2, 0x2DE3, 0x2DE4, 0x2DE5, 0x2DE6,
    0x2DE7, 0x2DE8, 0x2DE9, 0x2DEA, 0x2DEB, 0x2DEC, 0x2DED, 0x2DEE, 0x2DEF, 0x2DF0, 0x2DF1, 0x2DF2, 0x2DF3, 0x2DF4, 0x2DF5, 0x2DF6, 0x2DF7, 0x2DF8, 0x2DF9, 0x2DFA,
    0x2DFB, 0x2DFC, 0x2DFD, 0x2DFE, 0x2DFF, 0xA66F, 0xA67C, 0xA67D, 0xA6F0, 0xA6F1, 0xA8E0, 0xA8E1, 0xA8E2, 0xA8E3, 0xA8E4, 0xA8E5, 0xA8E6, 0xA8E7, 0xA8E8, 0xA8E9,
    0xA8EA, 0xA8EB, 0xA8EC, 0xA8ED, 0xA8EE, 0xA8EF, 0xA8F0, 0xA8F1, 0xAAB0, 0xAAB2, 0xAAB3, 0xAAB7, 0xAAB8, 0xAABE, 0xAABF, 0xAAC1, 0xFE20, 0xFE21, 0xFE22, 0xFE23,
    0xFE24, 0xFE25, 0xFE26, 0x10A0F, 0x10A38, 0x1D185, 0x1D186, 0x1D187, 0x1D188, 0x1D189, 0x1D1AA, 0x1D1AB, 0x1D1AC, 0x1D1AD, 0x1D242, 0x1D243, 0x1D244,
)


def kitty_color_hex(image_id: int) -> str:
    """Hex foreground color (``#rrggbb``) that encodes *image_id* for kitty."""
    return "#%06x" % (image_id & 0xFFFFFF)


def kitty_placeholder_rows(cols: int, rows: int) -> list[str]:
    """Placeholder text grid: first cell carries the row diacritic, the rest auto-increment the column (fg color applied by the painter)."""
    cols = max(1, cols)
    return [
        _KITTY_PLACEHOLDER + chr(_ROWCOL_DIACRITICS[min(r, len(_ROWCOL_DIACRITICS) - 1)]) + _KITTY_PLACEHOLDER * (cols - 1)
        for r in range(max(1, rows))
    ]


def encode_kitty_virtual(image, *, image_id: int, cols: int, rows: int) -> str:
    """kitty virtual placement (``U=1``; ``q=2`` mutes replies that would corrupt the screen). Re-sending the same ``i`` replaces the image in place."""
    return _kitty_apc(f"a=T,U=1,i={image_id},c={cols},r={rows},f=100,q=2", _png_b64(image))


def encode_iterm(image, *, cols: int | None = None, rows: int | None = None) -> str:
    """iTerm2 inline image (OSC 1337 File) pinned to a cell box (default: the image's own).

    ``size`` is the decoded byte count; xterm.js (VS Code) drops an image whose ``size`` is the
    base64 length.
    """
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    payload = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    if cols is None or rows is None:
        cols, rows = cell_box(image)
    return f"\x1b]1337;File=inline=1;size={buf.tell()};preserveAspectRatio=1;width={cols};height={rows}:{payload}\x07"


def encode_sixel(image) -> str:
    """DEC sixel via a compact hand-rolled encoder (Pillow has none); ≤255 adaptive colors, transparent pixels skipped."""
    from PIL import Image

    pal = image.convert("RGB").quantize(colors=255, method=Image.MEDIANCUT)
    palette = pal.getpalette() or []
    px = pal.load()
    alpha = image.getchannel("A").load()
    w, h = pal.size
    out = ["\x1bP0;1;0q", f'"1;1;{w:d};{h:d}']
    used = sorted({px[x, y] for y in range(h) for x in range(w)})
    for idx in used:  # color registers on a 0..100 scale
        r, g, b = (palette[idx * 3 + c] if idx * 3 + c < len(palette) else 0 for c in range(3))
        out.append(f'#{idx:d};2;{r * 100 // 255:d};{g * 100 // 255:d};{b * 100 // 255:d}')

    for band in range(0, h, 6):
        ys = range(band, min(band + 6, h))
        for color_idx in used:
            chars = [chr(63 + sum(1 << (y - band) for y in ys if alpha[x, y] > 32 and px[x, y] == color_idx)) for x in range(w)]
            runs = ((ch, len(list(group))) for ch, group in groupby(chars))  # run-length: ``!<n><ch>`` for runs longer than 3
            out.append(f'#{color_idx:d}' + "".join(f"!{n:d}{ch}" if n > 3 else ch * n for ch, n in runs) + "$")  # ``$`` = band CR
        out.append("-")  # next band
    return "".join(out) + "\x1b\\"


# A single half-block cell: top pixel + bottom pixel as (r, g, b, a) tuples.
Cell = tuple[tuple[int, int, int, int], tuple[int, int, int, int]]


def downscale_cells(image, *, target_cols: int, rows: int | None = None) -> list[list[Cell]]:
    """Downscale to rows of half-block cells (one terminal row = two pixel rows); shared by the ANSI encoder and Ink.

    Without *rows* the grid keeps the image's aspect at a 4-column floor; with it the grid is exactly ``target_cols``×``rows``.
    """
    from PIL import Image

    if rows is None:
        target_cols = max(4, target_cols)
        target_rows = max(2, round(target_cols * (image.height / max(1, image.width)) * 0.5) * 2)
    else:
        target_cols, target_rows = max(1, target_cols), max(1, rows) * 2
    px = image.resize((target_cols, target_rows), Image.LANCZOS).convert("RGBA").load()
    return [
        [(px[x, y], px[x, y + 1] if y + 1 < target_rows else (0, 0, 0, 0)) for x in range(target_cols)]
        for y in range(0, target_rows, 2)
    ]


def half_block(top, bottom) -> str:
    """One truecolor ANSI half-block cell; a transparent half shows the terminal background."""
    (tr, tg, tb, ta), (br, bg, bb, ba) = top, bottom
    if ta < 32 and ba < 32:
        return "\x1b[0m "
    if ba < 32:
        return f"\x1b[0m\x1b[38;2;{tr};{tg};{tb}m▀"
    if ta < 32:
        return f"\x1b[0m\x1b[38;2;{br};{bg};{bb}m▄"
    return f"\x1b[38;2;{tr};{tg};{tb}m\x1b[48;2;{br};{bg};{bb}m▀"


def encode_unicode(image, *, target_cols: int) -> str:
    """Truecolor ANSI half-blocks (one char = 2 vertical pixels)."""
    return "\n".join("".join(half_block(t, b) for t, b in row) + "\x1b[0m"
                     for row in downscale_cells(image, target_cols=target_cols))


# ── Inline images ────────────────────────────────────────────────────────────

# Names accepted by ``display.inline_images``.
_INLINE_IMAGE_MODES = ("auto", "kitty", "iterm", "unicode", "off")
# The image files Pillow decodes: what the CLI and the TUI both draw (attachments and tool images).
_RENDERABLE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".tif", ".ico"})

# Pet ids live in [1, 0x7FFF] (``hermes_agent.agent.pet.render.kitty_image_id``); images take the rest of
# the 24-bit space the placeholder foreground color can carry, so neither replaces the other.
_IMAGE_ID_BASE, _IMAGE_ID_SPAN = 0x8000, 0xFFFFFF - 0x8000
# Multiplexers that swallow image transmits (and inherit the outer terminal's KITTY_WINDOW_ID).
_MULTIPLEXER_ENV = ("TMUX", "STY")
# Transmit at up to 2 px per nominal cell pixel: sharp on HiDPI cells without shipping
# a full-resolution photo through the tty.
_PIXELS_PER_CELL_PIXEL = 2
# Refuse to decode a preview past this many pixels (after JPEG draft scaling): a 169 MP PNG is
# under Pillow's bomb error and would fully decode (seconds, >1 GB) on the CLI's agent thread.
_MAX_DECODE_PIXELS = 40_000_000


def resolve_inline_mode(
    configured: Any, *, graphics: bool = True, overlays: bool = False, detected: str | None = None
) -> str:
    """Effective ``kitty``/``iterm``/``unicode``/``off`` for ``display.inline_images``.

    ``graphics`` says the painter owns a real terminal it can send escapes to; ``overlays`` that
    it can reserve blank rows and paint a cell-pinned image over them (the classic CLI's
    scrollback can, Ink's re-laid-out frame can't). ``auto`` picks kitty placeholders where they
    paint (kitty, Ghostty), then iTerm2 images over reserved rows where the painter allows it
    (iTerm2, WezTerm, Konsole), else half-blocks. tmux and GNU screen swallow both transmits,
    so ``auto`` stays on half-blocks there. Pinning a protocol opts in: the classic CLI wraps its
    transmits for tmux's ``allow-passthrough on``; the TUI and GNU screen send them unwrapped.

    ``detected`` is a remote painter's own ``auto`` pick (``kitty``/``unicode``), made from its
    terminal's env: a shared gateway serves many clients, and its own env names none of them.
    """
    if configured is False:
        return "off"
    mode = "auto" if configured in (None, True) else str(configured).strip().lower()
    mode = mode if mode in _INLINE_IMAGE_MODES else "auto"
    if mode == "iterm" and not overlays:
        mode = "auto"
    if mode == "off":
        return "off"
    if mode == "unicode" or not graphics:
        return "unicode"
    if mode in ("kitty", "iterm"):
        return mode
    if detected is not None:
        return "kitty" if detected == "kitty" else "unicode"
    if any(os.environ.get(name) for name in _MULTIPLEXER_ENV):
        return "unicode"
    if supports_kitty_placeholders():
        return "kitty"
    return "iterm" if overlays and supports_iterm_images() else "unicode"


def local_image_path(candidate: Any) -> str | None:
    """*candidate* as an existing local image file path, else ``None`` (URLs and data URLs are not fetched)."""
    if not isinstance(candidate, str) or not candidate.strip() or "://" in candidate or candidate.startswith("data:"):
        return None
    path = Path(os.path.expanduser(candidate.strip()))
    try:
        return str(path) if path.suffix.lower() in _RENDERABLE_EXTENSIONS and path.is_file() else None
    except (OSError, ValueError):
        return None


# Result fields that name an image the call produced (browser_exec, computer_use), read from the
# result or from a multimodal envelope's ``meta``.
_RESULT_IMAGE_KEYS = ("image_path", "screenshot_path")
# Tools whose image is what the call showed or made: (``args``/``result``, key). Nothing else's
# args count: a ``path`` is just as often a write, a refused patch or a read that can't show it.
_TOOL_IMAGE_FIELDS: dict[str, tuple[tuple[str, str], ...]] = {
    "vision_analyze": (("args", "image_url"),),
    "image_generate": (("result", "image"),),
}


def tool_image_path(tool_name: str, args: Any, result: Any, *, is_error: bool) -> str | None:
    """The local image a successful tool call produced or looked at, else ``None``.

    Named fields only, so a tool that lists image paths in free text never floods the transcript,
    and nothing for a failed call: a refused write or an unreadable file must not look like success.
    """
    if is_error:
        return None
    if isinstance(result, str):
        from hermes_agent.utils import safe_json_loads

        result = safe_json_loads(result)
    result = result if isinstance(result, dict) else {}
    meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
    sources = {"args": args if isinstance(args, dict) else {}, "result": result}
    for source, key in (*_TOOL_IMAGE_FIELDS.get(tool_name, ()), *(("result", key) for key in _RESULT_IMAGE_KEYS)):
        for candidate in (sources[source].get(key), meta.get(key) if source == "result" else None):
            if path := local_image_path(candidate):
                return path
    return None


@dataclass(frozen=True)
class InlineImage:
    """One image laid out on a ``cols``×``rows`` cell grid.

    kitty mode: send ``transmit`` once, then paint ``placeholder`` rows in ``color``.
    unicode mode: paint ``cells`` as half-blocks.
    """

    mode: str
    cols: int
    rows: int
    color: str = ""
    transmit: str = ""
    placeholder: list[str] = field(default_factory=list)
    cells: list[list[Cell]] = field(default_factory=list)


def _image_id(path: Path, cols: int, rows: int) -> int:
    """Stable kitty image id per file revision and cell grid: re-renders reuse the terminal copy, an
    edited file gets a new one, and two sizes on screen at once each keep their own placement (kitty
    holds one virtual placement per id)."""
    st = path.stat()
    key = f"{path.resolve()}:{st.st_mtime_ns}:{st.st_size}:{cols}x{rows}"
    return _IMAGE_ID_BASE + zlib.crc32(key.encode()) % _IMAGE_ID_SPAN


def _grid_for(width: int, height: int, *, max_cols: int, max_rows: int) -> tuple[int, int]:
    """Cell grid for a *width*×*height* image: never wider than its natural size, never past the caps."""
    cols = max(1, min(max_cols, -(-width // CELL_W)))
    rows = max(1, round(cols * height / width * CELL_W / CELL_H))
    if rows > max_rows:
        rows = max_rows
        cols = max(1, min(cols, round(rows * width / height * CELL_H / CELL_W)))
    return cols, rows


def _decode_preview(source: Path, *, max_cols: int, max_rows: int):
    """``(image, cols, rows)``: *source* decoded no larger than its preview needs, upright, RGBA.

    The grid comes from the header size; JPEG decodes straight at the target scale (``draft``);
    a pixel count past ``_MAX_DECODE_PIXELS`` or a Pillow bomb warning refuses (raises)
    before any pixel is decoded.
    """
    from PIL import Image, ImageOps

    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(source) as opened:
            width, height = opened.size
            # Only ``draft`` (JPEG) can shrink the decode, and EXIF on a PNG is read by decoding it.
            if width * height > _MAX_DECODE_PIXELS and type(opened).draft is Image.Image.draft:
                raise ValueError(f"{width}x{height} is past the preview decode cap")
            swapped = opened.getexif().get(0x0112) in (5, 6, 7, 8)  # EXIF orientations that turn 90°
            if swapped:
                width, height = height, width
            cols, rows = _grid_for(width, height, max_cols=max_cols, max_rows=max_rows)
            box = (cols * CELL_W * _PIXELS_PER_CELL_PIXEL, rows * CELL_H * _PIXELS_PER_CELL_PIXEL)
            box = box[::-1] if swapped else box
            opened.draft("RGB", box)
            if opened.width * opened.height > _MAX_DECODE_PIXELS:
                raise ValueError(f"{opened.width}x{opened.height} is past the preview decode cap")
            opened.thumbnail(box, Image.LANCZOS)
            image = ImageOps.exif_transpose(opened).convert("RGBA")
    return image, cols, rows


def render_inline_image(path: str | Path, *, mode: str, max_cols: int, max_rows: int) -> InlineImage | None:
    """Lay out the image at *path* for *mode* (``kitty``/``iterm``/``unicode``); ``None`` when it can't be shown.

    ``iterm`` returns a ``transmit`` pinned to exactly ``cols``×``rows`` cells for a painter that
    reserved those rows; it draws at the cursor and owns no text cells.
    """
    if mode not in ("kitty", "iterm", "unicode") or max_cols < 1 or max_rows < 1:
        return None
    try:
        source = Path(path)
        image, cols, rows = _decode_preview(source, max_cols=max_cols, max_rows=max_rows)
        if mode == "unicode":
            return InlineImage(mode=mode, cols=cols, rows=rows, cells=downscale_cells(image, target_cols=cols, rows=rows))
        if mode == "iterm":
            return InlineImage(mode=mode, cols=cols, rows=rows, transmit=encode_iterm(image, cols=cols, rows=rows))
        image_id = _image_id(source, cols, rows)
        return InlineImage(
            mode=mode, cols=cols, rows=rows, color=kitty_color_hex(image_id),
            transmit=encode_kitty_virtual(image, image_id=image_id, cols=cols, rows=rows),
            placeholder=kitty_placeholder_rows(cols, rows))
    except Exception as exc:  # health: allow BLE001 -- PIL raises arbitrary types decoding an untrusted file; display-only
        logger.debug("inline image render failed (%s): %s", path, exc)
        return None
