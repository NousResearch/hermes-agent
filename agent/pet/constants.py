"""Pet sprite geometry + animation-state taxonomy.

These values are the common petdex/Codex pet geometry. The real ``pet.json``
usually only carries ``id``/``displayName``/``description``/``spritesheetPath``;
row taxonomy is inferred from the atlas shape so Hermes can render both legacy
8-row sheets and current 9-row Codex sheets.
"""

from __future__ import annotations

from enum import Enum

# Frame geometry (pixels). Current Codex/petdex spritesheets are 8 columns x 9
# rows (1536x1872), while older Hermes/petdex sheets used 9 columns x 8 rows
# (1728x1664). Renderers derive both row taxonomy and real column count from the
# concrete sheet, so either shape works.
FRAME_W = 192
FRAME_H = 208

# Frames consumed per animation state (the petdex web app uses CSS
# ``steps(6)``).  A sheet may physically contain more columns; we only step
# through the first ``FRAMES_PER_STATE``.
FRAMES_PER_STATE = 6

# Full-loop duration for one state, milliseconds (petdex default).
LOOP_MS = 1100

# Default on-screen scale relative to native frame size.  ``display.pet.scale``
# is the single master scalar: the desktop canvas multiplies its native pixels
# by it and every terminal surface derives its half-block/kitty column width
# from it (see :func:`cols_for_scale`), so one number shrinks all three
# interfaces together.  (petdex's own clients render at 0.7; we default smaller
# so the kitty/GUI mascot stays a glanceable corner sprite.  The half-block
# fallback can't shrink as far — see ``UNICODE_MIN_COLS`` — and clamps to its
# legibility floor instead.)
DEFAULT_SCALE = 0.33

# User-settable scale bounds (``/pet scale``, desktop slider).  Floor keeps the
# pet clickable/visible; ceiling stops a fat-fingered value from filling the
# screen.  The unicode fallback additionally clamps to ``UNICODE_MIN_COLS``.
MIN_SCALE = 0.1
MAX_SCALE = 3.0


def clamp_scale(scale: float) -> float:
    """Clamp *scale* to ``[MIN_SCALE, MAX_SCALE]`` (the single validation point)."""
    return max(MIN_SCALE, min(MAX_SCALE, scale))

# Terminal cells one native frame spans at ``scale == 1.0``.  A cell is ~8px
# wide, a frame is ``FRAME_W`` (192) px → 24 cells.  This mirrors the kitty
# graphics placement (``scaled_px // 8``) so at full scale every renderer agrees.
BASE_UNICODE_COLS = FRAME_W // 8

# Legibility floor for the half-block fallback.  A half-block cell samples the
# sprite at only 1 horizontal + 2 vertical taps, so below this width a 192×208
# pet collapses into an unreadable blob *regardless* of scale.  kitty/GUI draw
# true pixels and have no such floor — that's why the same ``scale: 0.33`` is
# crisp there but mush in half-blocks.  ``scale`` shrinks the unicode pet down
# TO this floor (and grows it above), instead of past it into noise.
UNICODE_MIN_COLS = 16


def cols_for_scale(scale: float) -> int:
    """Half-block width implied by *scale*, clamped to the legibility floor.

    Above the floor it tracks the kitty cell box (``scaled_px // 8``) so the two
    renderers converge at larger sizes; below it the floor keeps the sprite
    readable rather than letting it devolve into a blob.
    """
    return max(UNICODE_MIN_COLS, round(BASE_UNICODE_COLS * (scale or DEFAULT_SCALE)))


def resolve_cols(scale: float, unicode_cols: int = 0) -> int:
    """Resolve terminal width: explicit *unicode_cols* override, else from *scale*."""
    return int(unicode_cols) if unicode_cols and int(unicode_cols) > 0 else cols_for_scale(scale)


class PetState(str, Enum):
    """Animation state a pet can be shown in.

    These are Hermes' activity state names. They are not always identical to the
    source atlas row names: Codex-format pets use rows like ``jumping`` /
    ``running`` while the UI keeps the shorter ``jump`` / ``run`` names.
    """

    IDLE = "idle"
    WAVE = "wave"
    RUN = "run"
    FAILED = "failed"
    REVIEW = "review"
    JUMP = "jump"
    WAITING = "waiting"


# Legacy Hermes/petdex row order (top -> bottom) used by the older 8-row,
# 9-column atlas shape.
LEGACY_STATE_ROWS: list[str] = [
    PetState.IDLE.value,
    PetState.WAVE.value,
    PetState.RUN.value,
    PetState.FAILED.value,
    PetState.REVIEW.value,
    PetState.JUMP.value,
    "extra1",
    "extra2",
]

# Current Petdex row order (top -> bottom) used by 1536x1872 atlases:
# 8 columns x 9 rows of 192x208 cells.
CODEX_STATE_ROWS: list[str] = [
    PetState.IDLE.value,
    "running-right",
    "running-left",
    "waving",
    "jumping",
    PetState.FAILED.value,
    PetState.WAITING.value,
    "running",
    PetState.REVIEW.value,
]

# Default/fallback for callers without a sheet. Prefer the current 9-row Codex
# format because generated pets and the public Codex pet contract use it.
STATE_ROWS: list[str] = CODEX_STATE_ROWS

# Canonical Hermes activity names -> accepted row-name aliases in descending
# preference. This keeps our internal state names stable (`wave`/`jump`/`run`)
# while matching Petdex's current `waving`/`jumping`/`running` taxonomy.
STATE_ALIASES: dict[str, tuple[str, ...]] = {
    PetState.IDLE.value: (PetState.IDLE.value,),
    PetState.WAVE.value: (PetState.WAVE.value, "waving"),
    PetState.JUMP.value: (PetState.JUMP.value, "jumping"),
    PetState.RUN.value: (PetState.RUN.value, "running"),
    PetState.FAILED.value: (PetState.FAILED.value,),
    PetState.REVIEW.value: (PetState.REVIEW.value,),
    PetState.WAITING.value: (PetState.WAITING.value,),
}


def state_aliases_for(state: "PetState | str") -> tuple[str, ...]:
    """Return accepted row-name aliases for *state* (always non-empty)."""
    value = state.value if isinstance(state, PetState) else str(state)
    aliases = STATE_ALIASES.get(value)
    return aliases if aliases else (value,)


def state_rows_for_grid(row_count: int | None) -> list[str]:
    """Return the row taxonomy for a spritesheet with *row_count* rows."""
    try:
        rows = int(row_count or 0)
    except (TypeError, ValueError):
        rows = 0

    if rows >= len(CODEX_STATE_ROWS):
        return CODEX_STATE_ROWS
    return LEGACY_STATE_ROWS


def state_row_index(state: "PetState | str", row_count: int | None = None) -> int:
    """Return the spritesheet row index for *state* (clamped, never raises)."""
    rows = state_rows_for_grid(row_count)
    for name in state_aliases_for(state):
        try:
            return rows.index(name)
        except ValueError:
            continue
    return 0  # fall back to the idle row
