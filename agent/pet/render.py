"""Decode a pet spritesheet and encode frames for a terminal.

Shared by the base CLI (escape bytes to stdout) and the TUI (bytes shipped to
Ink) so decode + capability detection + protocol encoding exist once. Modes in
fidelity order: ``kitty`` (kitty, Ghostty, WezTerm), ``iterm`` (iTerm2, WezTerm),
``sixel`` (xterm -ti vt340, foot, mlterm, …), ``unicode`` (24-bit half-blocks).
Missing Pillow or spritesheet degrades to an empty string rather than raising
(PIL is imported lazily on purpose).
"""

from __future__ import annotations

import base64
import io
import logging
import os
import sys
import zlib
from dataclasses import KW_ONLY, dataclass
from functools import lru_cache
from itertools import groupby, takewhile
from pathlib import Path

from agent.pet.constants import DEFAULT_SCALE, FRAME_H, FRAME_W, FRAMES_PER_STATE, PetState, state_row_index

logger = logging.getLogger(__name__)

# Public render-mode names accepted by ``display.pet.render_mode``.
RENDER_MODES = ("auto", "kitty", "iterm", "sixel", "unicode", "off")


def _is_wezterm() -> bool:
    return os.environ.get("TERM_PROGRAM", "").lower() == "wezterm" or bool(os.environ.get("WEZTERM_PANE"))


def detect_terminal_graphics() -> str:
    """Richest protocol (``kitty``/``iterm``/``sixel``/``unicode``) from env vars only — never a DA1 query that could hang a pipe."""
    term = os.environ.get("TERM", "").lower()
    term_program = os.environ.get("TERM_PROGRAM", "").lower()
    # VS Code/Cursor set TERM_PROGRAM=vscode but don't scrub inherited
    # ITERM_SESSION_ID/KITTY_WINDOW_ID; trusting those emits a protocol xterm.js
    # can't show (blank frame). Inline images there are opt-in, so default to
    # half-blocks; users who enabled them can pin display.pet.render_mode.
    if term_program == "vscode":
        return "unicode"
    if os.environ.get("KITTY_WINDOW_ID") or "kitty" in term or "ghostty" in term or term_program == "ghostty" or _is_wezterm():
        return "kitty"  # WezTerm speaks kitty and iterm; kitty has richer placement
    if term_program == "iterm.app" or os.environ.get("ITERM_SESSION_ID"):
        return "iterm"
    if term_program == "mintty" or "foot" in term or "mlterm" in term or "sixel" in term:
        return "sixel"
    return "unicode"


def supports_kitty_placeholders() -> bool:
    """True when the terminal paints kitty Unicode placeholders; WezTerm speaks kitty APC but renders placeholders as tofu."""
    return detect_terminal_graphics() == "kitty" and not _is_wezterm()


def resolve_mode(configured: str | None, *, stream=None) -> str:
    """Effective render mode from ``display.pet.render_mode`` + env; ``off`` when not a TTY."""
    mode = (configured or "auto").strip().lower()
    mode = mode if mode in RENDER_MODES else "auto"
    stream = stream or sys.stdout
    try:
        if mode == "off" or not (hasattr(stream, "isatty") and stream.isatty()):
            return "off"
    except (ValueError, OSError):
        return "off"
    return detect_terminal_graphics() if mode == "auto" else mode


# Max alpha at/below which a frame is blank padding: petdex sheets are left-packed,
# so short states have transparent trailing cells; animating into one flashes blank.
_BLANK_ALPHA = 8


def _frame_is_blank(frame) -> bool:
    return frame.getchannel("A").getextrema()[1] <= _BLANK_ALPHA


@lru_cache(maxsize=16)
def _raw_frames(sheet_path: str, state_value: str, frame_w: int, frame_h: int, frames_per_state: int) -> tuple:
    """Cropped RGBA frames for one state row, stopping at the first blank column; ``()`` on any decode failure."""
    try:
        from PIL import Image

        sheet = Image.open(Path(sheet_path)).convert("RGBA")
        cols, rows = max(1, sheet.width // frame_w), max(1, sheet.height // frame_h)
        # Clamp to the sheet: some pets ship fewer rows than the taxonomy reserves.
        top = min(state_row_index(state_value, rows) * frame_h, max(0, sheet.height - frame_h))
        crops = (sheet.crop((i * frame_w, top, (i + 1) * frame_w, top + frame_h)) for i in range(min(frames_per_state, cols)))
        return tuple(takewhile(lambda f: not _frame_is_blank(f), crops))
    except Exception as exc:  # noqa: BLE001 - cosmetic feature, never fatal
        logger.debug("pet frame decode failed (%s, %s): %s", sheet_path, state_value, exc)
        return ()


@lru_cache(maxsize=8)
def _frames_for(sheet_path: str, state_value: str, frame_w: int, frame_h: int, frames_per_state: int, scale_w: int, scale_h: int):
    """Scaled :func:`_raw_frames` (both cached, so animation-time requests are free)."""
    raw = _raw_frames(sheet_path, state_value, frame_w, frame_h, frames_per_state)
    if not raw or (scale_w, scale_h) == (frame_w, frame_h):
        return list(raw)
    from PIL import Image

    return [f.resize((scale_w, scale_h), Image.LANCZOS) for f in raw]


def state_frame_counts(
    sheet_path: str | Path, *, frame_w: int = FRAME_W, frame_h: int = FRAME_H, frames_per_state: int = FRAMES_PER_STATE
) -> dict[str, int]:
    """Each driven :class:`PetState` → its real (padding-trimmed) frame count (shipped to the desktop canvas)."""
    return {s.value: len(_raw_frames(str(sheet_path), s.value, frame_w, frame_h, frames_per_state)) for s in PetState}


def _png_b64(frame) -> str:
    buf = io.BytesIO()
    frame.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


# Nominal terminal cell size in pixels. kitty fits an image to its cell rectangle
# preserving aspect, so a frame that isn't a whole cell multiple rounds up, clipping
# the bottom row ("clipped feet"); snapping to an exact multiple avoids that.
_CELL_W, _CELL_H = 8, 16


def _fit_frames_to_cell_grid(frames):
    """Crop *frames* (non-empty) to their union opaque bbox, then resize so width/height are exact cell-box multiples.

    kitty paints transparent margins too, so an untrimmed pet looks small and adrift.
    """
    from PIL import Image

    boxes = [b for b in (f.getchannel("A").getbbox() for f in frames) if b]
    if boxes:
        union = (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))
        frames = [f.crop(union) for f in frames]
    w, h = frames[0].size
    target = (max(1, round(w / _CELL_W)) * _CELL_W, max(1, round(h / _CELL_H)) * _CELL_H)
    return frames if (w, h) == target else [f.resize(target, Image.LANCZOS) for f in frames]


def _kitty_apc(ctrl: str, data: str) -> str:
    """kitty APC escape for *data*, chunked into ≤4096-byte ``m`` pieces (``m=1`` = more chunks follow)."""
    pieces = [data[i : i + 4096] for i in range(0, len(data), 4096)] or [""]
    last = len(pieces) - 1
    return "".join(f"\x1b_G{ctrl + ',' if i == 0 else ''}m={int(i != last)};{piece}\x1b\\" for i, piece in enumerate(pieces))


def _encode_kitty(frame) -> str:
    """kitty transmit+display at the cursor; ``c``/``r`` pin the cell box so frames overwrite each other."""
    cols, rows = _cell_box(frame)
    return _kitty_apc(f"f=100,a=T,q=2,c={cols},r={rows}", _png_b64(frame))


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


def kitty_image_id(slug: str) -> int:
    """Deterministic per-slug image id in ``[1, 0x7FFF]`` (non-zero; encoded in the placeholder fg color) so re-renders reuse the terminal-side image."""
    return (zlib.crc32(slug.encode("utf-8")) % 0x7FFE) + 1


def kitty_color_hex(image_id: int) -> str:
    """Hex foreground color (``#rrggbb``) that encodes *image_id* for kitty."""
    return "#%06x" % (image_id & 0xFFFFFF)


def kitty_placeholder_rows(cols: int, rows: int) -> list[str]:
    """Placeholder text grid: first cell carries the row diacritic, the rest auto-increment the column (fg color applied by Ink)."""
    cols = max(1, cols)
    return [
        _KITTY_PLACEHOLDER + chr(_ROWCOL_DIACRITICS[min(r, len(_ROWCOL_DIACRITICS) - 1)]) + _KITTY_PLACEHOLDER * (cols - 1)
        for r in range(max(1, rows))
    ]


def _encode_kitty_virtual(frame, *, image_id: int, cols: int, rows: int) -> str:
    """kitty virtual placement (``U=1``; ``q=2`` mutes replies that would corrupt Ink). Re-sending the same ``i`` animates in place."""
    return _kitty_apc(f"a=T,U=1,i={image_id},c={cols},r={rows},f=100,q=2", _png_b64(frame))


def _encode_iterm(frame) -> str:
    """iTerm2 inline image (OSC 1337 File) pinned to the frame's cell box."""
    payload = _png_b64(frame)
    cols, rows = _cell_box(frame)
    return f"\x1b]1337;File=inline=1;size={len(payload)};preserveAspectRatio=1;width={cols};height={rows}:{payload}\x07"


def _encode_sixel(frame) -> str:
    """DEC sixel via a compact hand-rolled encoder (Pillow has none); ≤255 adaptive colors, transparent pixels skipped."""
    from PIL import Image

    pal = frame.convert("RGB").quantize(colors=255, method=Image.MEDIANCUT)
    palette = pal.getpalette() or []
    px = pal.load()
    alpha = frame.getchannel("A").load()
    w, h = pal.size
    out = ["\x1bP0;1;0q", '"1;1;%d;%d' % (w, h)]
    used = sorted({px[x, y] for y in range(h) for x in range(w)})
    for idx in used:  # color registers on a 0..100 scale
        r, g, b = (palette[idx * 3 + c] if idx * 3 + c < len(palette) else 0 for c in range(3))
        out.append("#%d;2;%d;%d;%d" % (idx, r * 100 // 255, g * 100 // 255, b * 100 // 255))

    for band in range(0, h, 6):
        ys = range(band, min(band + 6, h))
        for color_idx in used:
            chars = [chr(63 + sum(1 << (y - band) for y in ys if alpha[x, y] > 32 and px[x, y] == color_idx)) for x in range(w)]
            runs = ((ch, len(list(group))) for ch, group in groupby(chars))  # run-length: ``!<n><ch>`` for runs longer than 3
            out.append("#%d" % color_idx + "".join("!%d%s" % (n, ch) if n > 3 else ch * n for ch, n in runs) + "$")  # ``$`` = band CR
        out.append("-")  # next band
    return "".join(out) + "\x1b\\"


# A single half-block cell: top pixel + bottom pixel as (r, g, b, a) tuples.
Cell = tuple[tuple[int, int, int, int], tuple[int, int, int, int]]


def _downscale_cells(frame, *, target_cols: int) -> list[list[Cell]]:
    """Downscale to rows of half-block cells (one terminal row = two pixel rows); shared by the ANSI encoder and Ink."""
    from PIL import Image

    target_cols = max(4, target_cols)
    target_rows = max(2, int(round(target_cols * (frame.height / max(1, frame.width)) * 0.5)) * 2)
    px = frame.resize((target_cols, target_rows), Image.LANCZOS).convert("RGBA").load()
    return [
        [(px[x, y], px[x, y + 1] if y + 1 < target_rows else (0, 0, 0, 0)) for x in range(target_cols)]
        for y in range(0, target_rows, 2)
    ]


def _encode_unicode(frame, *, target_cols: int) -> str:
    """Truecolor ANSI half-blocks (one char = 2 vertical pixels)."""
    def cell(top, bottom) -> str:
        (tr, tg, tb, ta), (br, bg, bb, ba) = top, bottom
        return "\x1b[0m " if ta < 32 and ba < 32 else f"\x1b[38;2;{tr};{tg};{tb}m\x1b[48;2;{br};{bg};{bb}m▀"

    return "\n".join("".join(cell(t, b) for t, b in row) + "\x1b[0m" for row in _downscale_cells(frame, target_cols=target_cols))


def _cell_box(frame) -> tuple[int, int]:
    """Terminal cell box (~8×16 px per cell) for a scaled frame.

    kitty stretches the image to fill ``c``×``r`` cells, so track the scaled pixel size, not a native-aspect column count (that upscales small pets).
    """
    return max(1, frame.width // _CELL_W), max(1, frame.height // _CELL_H)


_ENCODERS = {"kitty": _encode_kitty, "iterm": _encode_iterm, "sixel": _encode_sixel}


@dataclass(eq=False)
class PetRenderer:
    """Holds a pet's spritesheet and yields encoded frames per (state, index); decoded frames are cached."""

    spritesheet: str | Path
    _: KW_ONLY
    mode: str = "unicode"
    scale: float = DEFAULT_SCALE
    unicode_cols: int = 20
    frame_w: int = FRAME_W
    frame_h: int = FRAME_H
    frames_per_state: int = FRAMES_PER_STATE

    def __post_init__(self) -> None:
        self.spritesheet = str(self.spritesheet)
        self.mode = self.mode if self.mode in RENDER_MODES else "unicode"

    @property
    def available(self) -> bool:
        return self.mode != "off" and Path(self.spritesheet).is_file()

    def frame_count(self, state: PetState | str) -> int:
        return len(self._frames(state))

    def _frames(self, state: PetState | str):
        name = state.value if isinstance(state, PetState) else str(state)
        scaled = max(1, int(self.frame_w * self.scale)), max(1, int(self.frame_h * self.scale))
        return _frames_for(self.spritesheet, name, self.frame_w, self.frame_h, self.frames_per_state, *scaled)

    def cells(self, state: PetState | str, index: int, *, cols: int | None = None) -> list[list[Cell]]:
        """One frame as a half-block cell grid for Ink's native color props; ``[]`` when unavailable."""
        frames = self._frames(state)
        return _downscale_cells(frames[index % len(frames)], target_cols=cols or self.unicode_cols) if frames else []

    def kitty_payload(self, state: PetState | str, *, image_id: int) -> dict | None:
        """kitty placeholder payload ``{cols, rows, placeholder, frames}`` (transmit escapes + static text grid); ``None`` if no frames."""
        if not (frames := self._frames(state)):
            return None
        frames = _fit_frames_to_cell_grid(frames)
        cols, rows = _cell_box(frames[0])
        encoded = [_encode_kitty_virtual(f, image_id=image_id, cols=cols, rows=rows) for f in frames]
        return {"cols": cols, "rows": rows, "placeholder": kitty_placeholder_rows(cols, rows), "frames": encoded}

    def frame(self, state: PetState | str, index: int) -> str:
        """Encoded escape string for one frame (``index`` taken modulo the frame count), or ``""``."""
        if self.mode == "off" or not (frames := self._frames(state)):
            return ""
        frame = frames[index % len(frames)]
        try:
            if self.mode in _ENCODERS:
                return _ENCODERS[self.mode](frame)
            return _encode_unicode(frame, target_cols=self.unicode_cols)
        except Exception as exc:  # noqa: BLE001 - degrade silently
            logger.debug("pet frame encode failed (mode=%s): %s", self.mode, exc)
            return ""


def build_renderer(
    spritesheet: str | Path, *, configured_mode: str | None = None, scale: float = DEFAULT_SCALE, unicode_cols: int = 20, stream=None
) -> PetRenderer:
    """Resolve the mode from config+env, then construct a :class:`PetRenderer`."""
    return PetRenderer(spritesheet, mode=resolve_mode(configured_mode, stream=stream), scale=scale, unicode_cols=unicode_cols)
