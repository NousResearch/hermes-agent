"""Decode a pet spritesheet and encode frames for a terminal.

Shared by the base CLI (escape bytes to stdout) and the TUI (bytes shipped to
Ink). Capability detection and the protocol encoders live in
:mod:`agent.terminal_image`, which inline images use too. Missing Pillow or
spritesheet degrades to an empty string rather than raising (PIL is imported
lazily on purpose).
"""

from __future__ import annotations

import logging
import sys
import zlib
from dataclasses import KW_ONLY, dataclass
from functools import lru_cache
from itertools import takewhile
from pathlib import Path

from agent.pet.constants import DEFAULT_SCALE, FRAME_H, FRAME_W, FRAMES_PER_STATE, PetState, state_row_index
from agent.terminal_image import (
    CELL_H,
    CELL_W,
    Cell,
    cell_box,
    detect_terminal_graphics,
    downscale_cells,
    encode_iterm,
    encode_kitty,
    encode_kitty_virtual,
    encode_sixel,
    encode_unicode,
    kitty_placeholder_rows,
)

logger = logging.getLogger(__name__)

# Public render-mode names accepted by ``display.pet.render_mode``.
RENDER_MODES = ("auto", "kitty", "iterm", "sixel", "unicode", "off")


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
    except Exception as exc:
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
    target = (max(1, round(w / CELL_W)) * CELL_W, max(1, round(h / CELL_H)) * CELL_H)
    return frames if (w, h) == target else [f.resize(target, Image.LANCZOS) for f in frames]


def kitty_image_id(slug: str) -> int:
    """Deterministic per-slug image id in ``[1, 0x7FFF]`` (non-zero; encoded in the placeholder fg color) so re-renders reuse the terminal-side image."""
    return (zlib.crc32(slug.encode("utf-8")) % 0x7FFE) + 1


_ENCODERS = {"kitty": encode_kitty, "iterm": encode_iterm, "sixel": encode_sixel}


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
        return downscale_cells(frames[index % len(frames)], target_cols=cols or self.unicode_cols) if frames else []

    def kitty_payload(self, state: PetState | str, *, image_id: int) -> dict | None:
        """kitty placeholder payload ``{cols, rows, placeholder, frames}`` (transmit escapes + static text grid); ``None`` if no frames."""
        if not (frames := self._frames(state)):
            return None
        frames = _fit_frames_to_cell_grid(frames)
        cols, rows = cell_box(frames[0])
        encoded = [encode_kitty_virtual(f, image_id=image_id, cols=cols, rows=rows) for f in frames]
        return {"cols": cols, "rows": rows, "placeholder": kitty_placeholder_rows(cols, rows), "frames": encoded}

    def frame(self, state: PetState | str, index: int) -> str:
        """Encoded escape string for one frame (``index`` taken modulo the frame count), or ``""``."""
        if self.mode == "off" or not (frames := self._frames(state)):
            return ""
        frame = frames[index % len(frames)]
        try:
            if self.mode in _ENCODERS:
                return _ENCODERS[self.mode](frame)
            return encode_unicode(frame, target_cols=self.unicode_cols)
        except Exception as exc:
            logger.debug("pet frame encode failed (mode=%s): %s", self.mode, exc)
            return ""


def build_renderer(
    spritesheet: str | Path, *, configured_mode: str | None = None, scale: float = DEFAULT_SCALE, unicode_cols: int = 20, stream=None
) -> PetRenderer:
    """Resolve the mode from config+env, then construct a :class:`PetRenderer`."""
    return PetRenderer(spritesheet, mode=resolve_mode(configured_mode, stream=stream), scale=scale, unicode_cols=unicode_cols)
