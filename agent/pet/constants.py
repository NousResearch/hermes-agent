"""Pet sprite geometry + animation-state taxonomy.

Common petdex/Codex pet geometry. ``pet.json`` usually only carries
``id``/``displayName``/``description``/``spritesheetPath``; row taxonomy is
inferred from the atlas shape so Hermes renders both legacy 8-row sheets and
current 9-row Codex sheets.
"""

from __future__ import annotations

from enum import Enum

# Frame geometry (pixels). Codex/petdex sheets are 8 cols x 9 rows (1536x1872);
# older sheets 9 x 8 (1728x1664). Renderers derive taxonomy/columns from the sheet.
FRAME_W = 192
FRAME_H = 208
# Frames stepped per state (petdex CSS ``steps(6)``); extra physical columns are ignored.
FRAMES_PER_STATE = 6
LOOP_MS = 1100  # full-loop duration for one state, ms (petdex default)

# ``display.pet.scale`` is the single master scalar: the desktop canvas multiplies
# native pixels by it and every terminal surface derives its column width from it
# (:func:`cols_for_scale`). petdex clients render at 0.7; we default smaller so the
# mascot stays a glanceable corner sprite (half-blocks clamp to ``UNICODE_MIN_COLS``).
DEFAULT_SCALE = 0.33
# User-settable bounds (``/pet scale``, desktop slider): floor keeps the pet
# clickable/visible; ceiling stops a fat-fingered value from filling the screen.
MIN_SCALE = 0.1
MAX_SCALE = 3.0


def clamp_scale(scale: float) -> float:
    """Clamp *scale* to ``[MIN_SCALE, MAX_SCALE]`` (the single validation point)."""
    return max(MIN_SCALE, min(MAX_SCALE, scale))


# Cells one native frame spans at ``scale == 1.0`` (~8px cells → 24); mirrors the
# kitty placement (``scaled_px // 8``) so at full scale every renderer agrees.
BASE_UNICODE_COLS = FRAME_W // 8
# Legibility floor for half-blocks: a cell samples 1 horizontal + 2 vertical taps,
# so below this width the pet is an unreadable blob regardless of scale (kitty/GUI
# draw true pixels, no floor). ``scale`` shrinks the unicode pet TO this floor, not past it.
UNICODE_MIN_COLS = 16


def cols_for_scale(scale: float) -> int:
    """Half-block width implied by *scale*, clamped to the legibility floor (tracks the kitty cell box above it)."""
    return max(UNICODE_MIN_COLS, round(BASE_UNICODE_COLS * (scale or DEFAULT_SCALE)))


def resolve_cols(scale: float, unicode_cols: int = 0) -> int:
    """Resolve terminal width: explicit *unicode_cols* override, else from *scale*."""
    return int(unicode_cols) if unicode_cols and int(unicode_cols) > 0 else cols_for_scale(scale)


class PetState(str, Enum):
    """Animation state a pet can be shown in (Hermes names; Codex rows say ``jumping``/``running`` for ``jump``/``run``)."""

    IDLE = "idle"
    WAVE = "wave"
    RUN = "run"
    FAILED = "failed"
    REVIEW = "review"
    JUMP = "jump"
    WAITING = "waiting"


# Legacy Hermes/petdex row order (top -> bottom) for the older 8-row, 9-column atlas.
LEGACY_STATE_ROWS: list[str] = ["idle", "wave", "run", "failed", "review", "jump", "extra1", "extra2"]

# Current Petdex row order (top -> bottom) for 1536x1872 atlases (8 cols x 9 rows).
CODEX_STATE_ROWS: list[str] = ["idle", "running-right", "running-left", "waving", "jumping", "failed", "waiting", "running", "review"]

# Default for callers without a sheet: generated pets and the Codex contract use 9 rows.
STATE_ROWS: list[str] = CODEX_STATE_ROWS
# Canonical Hermes names -> accepted row-name aliases in descending preference.
_CODEX_NAMES = {"wave": "waving", "jump": "jumping", "run": "running"}
STATE_ALIASES: dict[str, tuple[str, ...]] = {
    s: (s, _CODEX_NAMES[s]) if s in _CODEX_NAMES else (s,) for s in ("idle", "wave", "jump", "run", "failed", "review", "waiting")
}


def state_aliases_for(state: "PetState | str") -> tuple[str, ...]:
    """Return accepted row-name aliases for *state* (always non-empty)."""
    value = state.value if isinstance(state, PetState) else str(state)
    return STATE_ALIASES.get(value) or (value,)


def state_rows_for_grid(row_count: int | None) -> list[str]:
    """Return the row taxonomy for a spritesheet with *row_count* rows."""
    try:
        rows = int(row_count or 0)
    except (TypeError, ValueError):
        rows = 0
    return CODEX_STATE_ROWS if rows >= len(CODEX_STATE_ROWS) else LEGACY_STATE_ROWS


def state_row_index(state: "PetState | str", row_count: int | None = None) -> int:
    """Return the spritesheet row index for *state* (clamped, never raises)."""
    rows = state_rows_for_grid(row_count)
    return next((rows.index(name) for name in state_aliases_for(state) if name in rows), 0)  # 0 = idle row
