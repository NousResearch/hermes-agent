"""Decode a pet spritesheet and encode frames for a terminal.

Shared by the base CLI (writes the escape bytes to its own stdout) and the
TUI (``tui_gateway`` ships the encoded bytes to Ink, which writes them) so the
decode + capability-detection + protocol-encoding logic exists exactly once.

Supported output modes, in fidelity order:

- ``kitty``   — the kitty graphics protocol (kitty, Ghostty, WezTerm).
- ``iterm``   — iTerm2 inline images (iTerm2, WezTerm).
- ``sixel``   — DEC sixel (xterm -ti vt340, foot, mlterm, WezTerm, …).
- ``unicode`` — 24-bit half-block downscale; works in any truecolor terminal.

Frame decoding requires Pillow (a core Hermes dependency).  If Pillow or the
spritesheet is unavailable the renderer degrades to ``unicode`` text or an
empty string rather than raising.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import sys
from functools import lru_cache
from pathlib import Path

from agent.pet.constants import (
    DEFAULT_SCALE,
    FRAME_H,
    FRAME_W,
    FRAMES_PER_STATE,
    PetState,
    state_row_index,
)

logger = logging.getLogger(__name__)

# Public render-mode names accepted by ``display.pet.render_mode``.
RENDER_MODES = ("auto", "kitty", "iterm", "sixel", "unicode", "off")


# ─────────────────────────────────────────────────────────────────────────
# Terminal capability detection
# ─────────────────────────────────────────────────────────────────────────

def detect_terminal_graphics() -> str:
    """Best-effort detection of the richest graphics protocol available.

    Env-based (non-blocking — we never issue a DA1/terminal query that could
    hang a pipe).  Returns one of ``kitty`` / ``iterm`` / ``sixel`` /
    ``unicode``.  Conservative: unknown terminals get ``unicode``, which works
    anywhere with truecolor.
    """
    term = os.environ.get("TERM", "").lower()
    term_program = os.environ.get("TERM_PROGRAM", "").lower()

    # The VS Code / Cursor integrated terminal sets TERM_PROGRAM=vscode
    # authoritatively but does NOT scrub the terminal env vars it inherits when
    # launched from another emulator (ITERM_SESSION_ID, KITTY_WINDOW_ID, …).
    # Trusting those leaks emits an image protocol the embedded xterm.js can't
    # display — you get a blank frame. Inline images there are opt-in
    # (terminal.integrated.enableImages), so default to half-blocks, which
    # always render in its truecolor grid. Users who enabled images can pin
    # display.pet.render_mode explicitly.
    if term_program == "vscode":
        return "unicode"

    # kitty graphics protocol
    if os.environ.get("KITTY_WINDOW_ID") or "kitty" in term or "ghostty" in term:
        return "kitty"
    if term_program in {"ghostty"}:
        return "kitty"

    # WezTerm speaks both kitty and iterm; prefer kitty (richer placement).
    if term_program == "wezterm" or os.environ.get("WEZTERM_PANE"):
        return "kitty"

    # iTerm2 inline images
    if term_program == "iterm.app" or os.environ.get("ITERM_SESSION_ID"):
        return "iterm"

    # sixel-capable terminals (env heuristics only)
    if term_program in {"mintty"} or "foot" in term or "mlterm" in term:
        return "sixel"
    if "sixel" in term:
        return "sixel"

    return "unicode"


def resolve_mode(configured: str | None, *, stream=None) -> str:
    """Resolve the effective render mode from config + the environment.

    ``configured`` is ``display.pet.render_mode`` (``auto`` → detect).  Returns
    ``off`` when not attached to a TTY (no point emitting graphics into a pipe
    or logfile).
    """
    mode = (configured or "auto").strip().lower()
    if mode not in RENDER_MODES:
        mode = "auto"
    if mode == "off":
        return "off"

    stream = stream or sys.stdout
    try:
        if not (hasattr(stream, "isatty") and stream.isatty()):
            return "off"
    except (ValueError, OSError):
        return "off"

    if mode == "auto":
        return detect_terminal_graphics()
    return mode


# ─────────────────────────────────────────────────────────────────────────
# Frame decoding
# ─────────────────────────────────────────────────────────────────────────

def _open_sheet(path: Path):
    from PIL import Image

    img = Image.open(path)
    return img.convert("RGBA")


# Max alpha at/below which a frame counts as blank padding.  petdex sheets are
# left-packed: a state with fewer real frames than ``FRAMES_PER_STATE`` fills
# the trailing columns with fully transparent cells.  Animating into one flashes
# the pet blank, so we stop the row at the first such gap.
_BLANK_ALPHA = 8


def _frame_is_blank(frame) -> bool:
    """True if *frame* has no meaningfully opaque pixel (transparent padding)."""
    return frame.getchannel("A").getextrema()[1] <= _BLANK_ALPHA


@lru_cache(maxsize=16)
def _raw_frames(
    sheet_path: str,
    state_value: str,
    frame_w: int,
    frame_h: int,
    frames_per_state: int,
) -> tuple:
    """Cropped, padding-trimmed RGBA frames for one state row (unscaled).

    Steps across the row until the first blank column so pets with ragged
    per-state frame counts never animate into empty padding.  Cached; returns
    ``()`` on any decode failure.
    """
    try:
        sheet = _open_sheet(Path(sheet_path))
        cols = max(1, sheet.width // frame_w)
        rows = max(1, sheet.height // frame_h)
        row = state_row_index(state_value, rows)
        top = row * frame_h
        # Clamp the row to the sheet (some pets ship fewer rows than the 8 the
        # taxonomy reserves).
        if top + frame_h > sheet.height:
            top = max(0, sheet.height - frame_h)

        frames = []
        for i in range(min(frames_per_state, cols)):
            left = i * frame_w
            frame = sheet.crop((left, top, left + frame_w, top + frame_h))
            if _frame_is_blank(frame):
                break  # trailing transparent padding — real frames end here
            frames.append(frame)
        return tuple(frames)
    except Exception as exc:  # noqa: BLE001 - cosmetic feature, never fatal
        logger.debug("pet frame decode failed (%s, %s): %s", sheet_path, state_value, exc)
        return ()


@lru_cache(maxsize=8)
def _frames_for(
    sheet_path: str,
    state_value: str,
    frame_w: int,
    frame_h: int,
    frames_per_state: int,
    scale_w: int,
    scale_h: int,
):
    """Return padding-trimmed RGBA frames for one state row, scaled.

    Thin scaling layer over :func:`_raw_frames`; both are cached so repeated
    frame requests during animation are free.
    """
    raw = _raw_frames(sheet_path, state_value, frame_w, frame_h, frames_per_state)
    if not raw or (scale_w, scale_h) == (frame_w, frame_h):
        return list(raw)
    from PIL import Image

    return [f.resize((scale_w, scale_h), Image.LANCZOS) for f in raw]


def state_frame_counts(
    sheet_path: str | Path,
    *,
    frame_w: int = FRAME_W,
    frame_h: int = FRAME_H,
    frames_per_state: int = FRAMES_PER_STATE,
) -> dict[str, int]:
    """Map each driven :class:`PetState` → its real (padding-trimmed) frame count.

    The single source of truth for "how many frames does this state actually
    have?".  The CLI/TUI consume the trimmed frame lists directly; the gateway
    ships this map to the desktop canvas, which steps its own loop.
    """
    return {
        state.value: len(
            _raw_frames(str(sheet_path), state.value, frame_w, frame_h, frames_per_state)
        )
        for state in PetState
    }


# ─────────────────────────────────────────────────────────────────────────
# Encoders
# ─────────────────────────────────────────────────────────────────────────

def _png_bytes(frame) -> bytes:
    buf = io.BytesIO()
    frame.save(buf, format="PNG")
    return buf.getvalue()


def _kitty_apc(ctrl: str, data: str) -> str:
    """Emit a kitty APC escape for *data*, chunked into ≤4096-byte ``m`` pieces."""
    chunk = 4096
    if len(data) <= chunk:
        return f"\x1b_G{ctrl},m=0;{data}\x1b\\"
    out = [f"\x1b_G{ctrl},m=1;{data[:chunk]}\x1b\\"]
    rest = data[chunk:]
    while rest:
        piece, rest = rest[:chunk], rest[chunk:]
        out.append(f"\x1b_Gm={1 if rest else 0};{piece}\x1b\\")
    return "".join(out)


def _encode_kitty(frame, *, cell_cols: int | None = None, cell_rows: int | None = None) -> str:
    """Encode one frame via the kitty graphics protocol (transmit + display).

    ``a=T`` transmits & displays at the cursor; ``c``/``r`` request a display
    box in terminal cells so successive frames overwrite the same area.
    """
    ctrl = "f=100,a=T,q=2"
    if cell_cols:
        ctrl += f",c={cell_cols}"
    if cell_rows:
        ctrl += f",r={cell_rows}"
    return _kitty_apc(ctrl, base64.standard_b64encode(_png_bytes(frame)).decode("ascii"))


# ─────────────────────────────────────────────────────────────────────────
# kitty Unicode placeholders
#
# Ink (the TUI's React-for-terminal layer) owns the screen and measures every
# cell's width, so it can't host raw kitty image escapes (no width to count,
# clobbered on the next repaint). kitty's *Unicode placeholder* protocol is the
# grid-safe path: transmit the image once (q=2, virtual placement U=1), then the
# host app prints ordinary-width placeholder cells (U+10EEEE + diacritics) whose
# foreground color encodes the image id. Ink counts those as width-1 text, so
# layout stays correct and the terminal paints the image underneath.
#   https://sw.kovidgoyal.net/kitty/graphics-protocol/#unicode-placeholders
# ─────────────────────────────────────────────────────────────────────────

_KITTY_PLACEHOLDER = "\U0010eeee"

# Row/column diacritics, in order (index → diacritic). Verbatim from kitty's
# gen/rowcolumn-diacritics.txt (Unicode 6.0.0, combining class 230). Index i is
# the diacritic that encodes the number i; we only ever need the row index.
_ROWCOL_DIACRITICS: tuple[int, ...] = (
    0x0305, 0x030D, 0x030E, 0x0310, 0x0312, 0x033D, 0x033E, 0x033F, 0x0346, 0x034A,
    0x034B, 0x034C, 0x0350, 0x0351, 0x0352, 0x0357, 0x035B, 0x0363, 0x0364, 0x0365,
    0x0366, 0x0367, 0x0368, 0x0369, 0x036A, 0x036B, 0x036C, 0x036D, 0x036E, 0x036F,
    0x0483, 0x0484, 0x0485, 0x0486, 0x0487, 0x0592, 0x0593, 0x0594, 0x0595, 0x0597,
    0x0598, 0x0599, 0x059C, 0x059D, 0x059E, 0x059F, 0x05A0, 0x05A1, 0x05A8, 0x05A9,
    0x05AB, 0x05AC, 0x05AF, 0x05C4, 0x0610, 0x0611, 0x0612, 0x0613, 0x0614, 0x0615,
    0x0616, 0x0617, 0x0657, 0x0658, 0x0659, 0x065A, 0x065B, 0x065D, 0x065E, 0x06D6,
    0x06D7, 0x06D8, 0x06D9, 0x06DA, 0x06DB, 0x06DC, 0x06DF, 0x06E0, 0x06E1, 0x06E2,
    0x06E4, 0x06E7, 0x06E8, 0x06EB, 0x06EC, 0x0730, 0x0732, 0x0733, 0x0735, 0x0736,
    0x073A, 0x073D, 0x073F, 0x0740, 0x0741, 0x0743, 0x0745, 0x0747, 0x0749, 0x074A,
    0x07EB, 0x07EC, 0x07ED, 0x07EE, 0x07EF, 0x07F0, 0x07F1, 0x07F3, 0x0816, 0x0817,
    0x0818, 0x0819, 0x081B, 0x081C, 0x081D, 0x081E, 0x081F, 0x0820, 0x0821, 0x0822,
    0x0823, 0x0825, 0x0826, 0x0827, 0x0829, 0x082A, 0x082B, 0x082C, 0x082D, 0x0951,
    0x0953, 0x0954, 0x0F82, 0x0F83, 0x0F86, 0x0F87, 0x135D, 0x135E, 0x135F, 0x17DD,
    0x193A, 0x1A17, 0x1A75, 0x1A76, 0x1A77, 0x1A78, 0x1A79, 0x1A7A, 0x1A7B, 0x1A7C,
    0x1B6B, 0x1B6D, 0x1B6E, 0x1B6F, 0x1B70, 0x1B71, 0x1B72, 0x1B73, 0x1CD0, 0x1CD1,
    0x1CD2, 0x1CDA, 0x1CDB, 0x1CE0, 0x1DC0, 0x1DC1, 0x1DC3, 0x1DC4, 0x1DC5, 0x1DC6,
    0x1DC7, 0x1DC8, 0x1DC9, 0x1DCB, 0x1DCC, 0x1DD1, 0x1DD2, 0x1DD3, 0x1DD4, 0x1DD5,
    0x1DD6, 0x1DD7, 0x1DD8, 0x1DD9, 0x1DDA, 0x1DDB, 0x1DDC, 0x1DDD, 0x1DDE, 0x1DDF,
    0x1DE0, 0x1DE1, 0x1DE2, 0x1DE3, 0x1DE4, 0x1DE5, 0x1DE6, 0x1DFE, 0x20D0, 0x20D1,
    0x20D4, 0x20D5, 0x20D6, 0x20D7, 0x20DB, 0x20DC, 0x20E1, 0x20E7, 0x20E9, 0x20F0,
    0x2CEF, 0x2CF0, 0x2CF1, 0x2DE0, 0x2DE1, 0x2DE2, 0x2DE3, 0x2DE4, 0x2DE5, 0x2DE6,
    0x2DE7, 0x2DE8, 0x2DE9, 0x2DEA, 0x2DEB, 0x2DEC, 0x2DED, 0x2DEE, 0x2DEF, 0x2DF0,
    0x2DF1, 0x2DF2, 0x2DF3, 0x2DF4, 0x2DF5, 0x2DF6, 0x2DF7, 0x2DF8, 0x2DF9, 0x2DFA,
    0x2DFB, 0x2DFC, 0x2DFD, 0x2DFE, 0x2DFF, 0xA66F, 0xA67C, 0xA67D, 0xA6F0, 0xA6F1,
    0xA8E0, 0xA8E1, 0xA8E2, 0xA8E3, 0xA8E4, 0xA8E5, 0xA8E6, 0xA8E7, 0xA8E8, 0xA8E9,
    0xA8EA, 0xA8EB, 0xA8EC, 0xA8ED, 0xA8EE, 0xA8EF, 0xA8F0, 0xA8F1, 0xAAB0, 0xAAB2,
    0xAAB3, 0xAAB7, 0xAAB8, 0xAABE, 0xAABF, 0xAAC1, 0xFE20, 0xFE21, 0xFE22, 0xFE23,
    0xFE24, 0xFE25, 0xFE26, 0x10A0F, 0x10A38, 0x1D185, 0x1D186, 0x1D187, 0x1D188,
    0x1D189, 0x1D1AA, 0x1D1AB, 0x1D1AC, 0x1D1AD, 0x1D242, 0x1D243, 0x1D244,
)


def kitty_image_id(slug: str) -> int:
    """Stable per-pet image id in ``[1, 0x7FFF]``.

    The id is encoded in the placeholder's 24-bit foreground color, so it must
    be non-zero and fit comfortably under ``0xFFFFFF``. A small CRC keeps it
    deterministic per slug (so re-renders reuse the same terminal-side image)
    while making collisions between two different pets unlikely.
    """
    import zlib

    return (zlib.crc32(slug.encode("utf-8")) % 0x7FFE) + 1


def kitty_color_hex(image_id: int) -> str:
    """Hex foreground color (``#rrggbb``) that encodes *image_id* for kitty."""
    return "#%06x" % (image_id & 0xFFFFFF)


def kitty_placeholder_rows(cols: int, rows: int) -> list[str]:
    """Build the placeholder text grid for an *rows*×*cols* image.

    Each line is one row of the grid: the first cell carries the row diacritic
    (column defaults to 0), and the remaining ``cols-1`` bare placeholders let
    the terminal auto-increment the column. The foreground color (the image id)
    is applied by the caller / Ink, not embedded here.
    """
    cols = max(1, cols)
    out: list[str] = []
    for r in range(max(1, rows)):
        idx = min(r, len(_ROWCOL_DIACRITICS) - 1)
        first = _KITTY_PLACEHOLDER + chr(_ROWCOL_DIACRITICS[idx])
        out.append(first + _KITTY_PLACEHOLDER * (cols - 1))
    return out


def _encode_kitty_virtual(frame, *, image_id: int, cols: int, rows: int) -> str:
    """Transmit a frame as a kitty *virtual* placement for Unicode placeholders.

    ``a=T`` transmits and creates the placement in one shot; ``U=1`` marks it
    virtual (no on-screen output, cursor untouched); ``q=2`` suppresses the
    terminal's OK/error replies that would otherwise corrupt the host app's
    output. Re-sending with the same ``i`` replaces the image, so the static
    placeholder cells animate underneath.
    """
    ctrl = f"a=T,U=1,i={image_id},c={cols},r={rows},f=100,q=2"
    return _kitty_apc(ctrl, base64.standard_b64encode(_png_bytes(frame)).decode("ascii"))


def _encode_iterm(frame, *, cell_cols: int | None = None, cell_rows: int | None = None) -> str:
    """Encode one frame as an iTerm2 inline image (OSC 1337 File)."""
    payload = base64.standard_b64encode(_png_bytes(frame)).decode("ascii")
    size = len(payload)
    args = [f"inline=1", f"size={size}", "preserveAspectRatio=1"]
    if cell_cols:
        args.append(f"width={cell_cols}")
    if cell_rows:
        args.append(f"height={cell_rows}")
    return f"\x1b]1337;File={';'.join(args)}:{payload}\x07"


def _encode_sixel(frame) -> str:
    """Encode one frame as DEC sixel.

    Quantizes to an adaptive palette (≤255 colors) and emits the sixel band
    stream.  Pillow has no sixel writer, so this is a compact hand-rolled
    encoder.  Transparent pixels render as background (color register skipped).
    """
    from PIL import Image

    rgba = frame
    # Composite onto transparent-as-skip: track alpha to decide background.
    pal = rgba.convert("RGB").quantize(colors=255, method=Image.MEDIANCUT)
    palette = pal.getpalette() or []
    px = pal.load()
    alpha = rgba.getchannel("A").load()
    w, h = pal.size

    out = ["\x1bP0;1;0q", '"1;1;%d;%d' % (w, h)]
    # Color register definitions (sixel uses 0..100 scale).
    used = sorted({px[x, y] for y in range(h) for x in range(w)})
    for idx in used:
        r = palette[idx * 3] if idx * 3 < len(palette) else 0
        g = palette[idx * 3 + 1] if idx * 3 + 1 < len(palette) else 0
        b = palette[idx * 3 + 2] if idx * 3 + 2 < len(palette) else 0
        out.append("#%d;2;%d;%d;%d" % (idx, r * 100 // 255, g * 100 // 255, b * 100 // 255))

    # Emit in 6-row bands.
    for band in range(0, h, 6):
        for color_idx in used:
            line = ["#%d" % color_idx]
            run_char = None
            run_len = 0

            def flush():
                nonlocal run_char, run_len
                if run_char is None:
                    return
                if run_len > 3:
                    line.append("!%d%s" % (run_len, run_char))
                else:
                    line.append(run_char * run_len)
                run_char, run_len = None, 0

            for x in range(w):
                bits = 0
                for bit in range(6):
                    y = band + bit
                    if y < h and alpha[x, y] > 32 and px[x, y] == color_idx:
                        bits |= 1 << bit
                ch = chr(63 + bits)
                if ch == run_char:
                    run_len += 1
                else:
                    flush()
                    run_char, run_len = ch, 1
            flush()
            out.append("".join(line) + "$")  # carriage return within band
        out.append("-")  # next band
    out.append("\x1b\\")
    return "".join(out)


_HALF_BLOCK = "▀"

# A single half-block cell: top pixel + bottom pixel as (r, g, b, a) tuples.
Cell = tuple[tuple[int, int, int, int], tuple[int, int, int, int]]


def _downscale_cells(frame, *, target_cols: int) -> list[list[Cell]]:
    """Downscale a frame to a grid of half-block cells.

    Each cell pairs a top and bottom pixel so one terminal row encodes two
    pixel rows.  Returns rows of ``((tr,tg,tb,ta),(br,bg,bb,ba))`` — the
    framework-neutral representation shared by the ANSI encoder (CLI) and the
    structured ``cells`` API (Ink).
    """
    from PIL import Image

    target_cols = max(4, target_cols)
    aspect = frame.height / max(1, frame.width)
    target_rows = max(2, int(round(target_cols * aspect * 0.5)) * 2)
    small = frame.resize((target_cols, target_rows), Image.LANCZOS).convert("RGBA")
    px = small.load()

    grid: list[list[Cell]] = []
    for y in range(0, target_rows, 2):
        row: list[Cell] = []
        for x in range(target_cols):
            top = px[x, y]
            bottom = px[x, y + 1] if y + 1 < target_rows else (0, 0, 0, 0)
            row.append((top, bottom))
        grid.append(row)
    return grid


def _encode_unicode(frame, *, target_cols: int) -> str:
    """Downscale to truecolor ANSI half-blocks (one char = 2 vertical pixels)."""
    lines: list[str] = []
    for row in _downscale_cells(frame, target_cols=target_cols):
        cells: list[str] = []
        for (tr, tg, tb, ta), (br, bg, bb, ba) in row:
            if ta < 32 and ba < 32:
                cells.append("\x1b[0m ")  # fully transparent → blank
                continue
            cells.append(f"\x1b[38;2;{tr};{tg};{tb}m\x1b[48;2;{br};{bg};{bb}m{_HALF_BLOCK}")
        lines.append("".join(cells) + "\x1b[0m")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────
# Public renderer
# ─────────────────────────────────────────────────────────────────────────

class PetRenderer:
    """Holds a pet's spritesheet and yields encoded frames per (state, index).

    Construct once per pet, then call :meth:`frame` on an animation timer.
    Cheap to call repeatedly — decoded frames are cached.
    """

    def __init__(
        self,
        spritesheet: str | Path,
        *,
        mode: str = "unicode",
        scale: float = DEFAULT_SCALE,
        unicode_cols: int = 20,
        frame_w: int = FRAME_W,
        frame_h: int = FRAME_H,
        frames_per_state: int = FRAMES_PER_STATE,
    ) -> None:
        self.spritesheet = str(spritesheet)
        self.mode = mode if mode in RENDER_MODES else "unicode"
        self.scale = scale
        self.unicode_cols = unicode_cols
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.frames_per_state = frames_per_state

    @property
    def available(self) -> bool:
        return self.mode != "off" and Path(self.spritesheet).is_file()

    def frame_count(self, state: PetState | str) -> int:
        return len(self._frames(state))

    def _frames(self, state: PetState | str):
        value = state.value if isinstance(state, PetState) else str(state)
        scale_w = max(1, int(self.frame_w * self.scale))
        scale_h = max(1, int(self.frame_h * self.scale))
        return _frames_for(
            self.spritesheet,
            value,
            self.frame_w,
            self.frame_h,
            self.frames_per_state,
            scale_w,
            scale_h,
        )

    def cells(self, state: PetState | str, index: int, *, cols: int | None = None) -> list[list[Cell]]:
        """Return one frame as a half-block cell grid (framework-neutral).

        Used by the TUI, which renders the grid with native Ink color props
        instead of raw ANSI.  Returns ``[]`` when no frame is available.
        """
        frames = self._frames(state)
        if not frames:
            return []
        frame = frames[index % len(frames)]
        return _downscale_cells(frame, target_cols=cols or self.unicode_cols)

    def _cell_box(self, frame) -> tuple[int, int]:
        """Terminal cell box for a scaled frame (~8×16 px per cell).

        Must match :meth:`frame` graphics sizing — kitty stretches the image to
        fill ``c``×``r`` cells, so these must reflect the scaled pixel
        dimensions, not a native-aspect column count (that upscales small pets).
        """
        return max(1, frame.width // 8), max(1, frame.height // 16)

    def kitty_payload(self, state: PetState | str, *, image_id: int) -> dict | None:
        """Build the kitty Unicode-placeholder payload for one state.

        Returns ``{cols, rows, placeholder, frames}`` where ``frames`` is a
        list of transmit escapes (one per animation frame, all reusing
        ``image_id``) and ``placeholder`` is the static text grid Ink paints.
        Placement geometry is derived from the scaled frame pixels (via
        :meth:`_cell_box`), not ``unicode_cols`` — kitty upscales to fill
        ``c``×``r`` cells. ``None`` when no frame is available.
        """
        frames = self._frames(state)
        if not frames:
            return None
        cols, rows = self._cell_box(frames[0])
        return {
            "cols": cols,
            "rows": rows,
            "placeholder": kitty_placeholder_rows(cols, rows),
            "frames": [
                _encode_kitty_virtual(f, image_id=image_id, cols=cols, rows=rows) for f in frames
            ],
        }

    def frame(self, state: PetState | str, index: int) -> str:
        """Return the encoded escape string for one frame, or ``""``.

        ``index`` is taken modulo the available frame count so callers can pass
        a free-running counter.
        """
        if self.mode == "off":
            return ""
        frames = self._frames(state)
        if not frames:
            return ""
        frame = frames[index % len(frames)]
        cell_cols, cell_rows = self._cell_box(frame)

        try:
            if self.mode == "kitty":
                return _encode_kitty(frame, cell_cols=cell_cols, cell_rows=cell_rows)
            if self.mode == "iterm":
                return _encode_iterm(frame, cell_cols=cell_cols, cell_rows=cell_rows)
            if self.mode == "sixel":
                return _encode_sixel(frame)
            return _encode_unicode(frame, target_cols=self.unicode_cols)
        except Exception as exc:  # noqa: BLE001 - degrade silently
            logger.debug("pet frame encode failed (mode=%s): %s", self.mode, exc)
            return ""


def build_renderer(
    spritesheet: str | Path,
    *,
    configured_mode: str | None = None,
    scale: float = DEFAULT_SCALE,
    unicode_cols: int = 20,
    stream=None,
) -> PetRenderer:
    """Convenience factory: resolve the mode from config+env, then construct."""
    mode = resolve_mode(configured_mode, stream=stream)
    return PetRenderer(
        spritesheet,
        mode=mode,
        scale=scale,
        unicode_cols=unicode_cols,
    )
