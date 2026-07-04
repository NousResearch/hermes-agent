"""Petdex pet engine — shared core for the CLI, TUI, and desktop surfaces.

Petdex (https://github.com/crafter-station/petdex) is a public gallery of
animated sprite "pets" for coding agents.  Each pet is a ``pet.json`` plus a
``spritesheet.{webp,png}`` of 192×208 px cells. Current Codex/petdex sheets use
an 8-column × 9-row atlas; older Hermes/petdex sheets used an 8-row atlas.
Hermes infers the row taxonomy from the sheet and maps agent activity onto
idle/run/review/failed/wave/jump.

This package is the **single source of truth** for the feature so the base
CLI (Python) and TUI (Ink, via ``tui_gateway``) never duplicate the hard
parts:

- :mod:`agent.pet.constants` — frame geometry + the :class:`PetState` enum.
- :mod:`agent.pet.state`     — map agent activity → a :class:`PetState`.
- :mod:`agent.pet.manifest`  — fetch the public petdex manifest.
- :mod:`agent.pet.store`     — install / list / resolve pets on disk
                               (profile-aware via ``get_hermes_home()``).
- :mod:`agent.pet.render`    — decode a spritesheet and encode frames for a
                               terminal (kitty / iTerm2 / sixel graphics
                               protocols, with a Unicode half-block
                               fallback).

Rendering in the Electron desktop is necessarily TypeScript (canvas), but it
reuses the same on-disk store and the same state semantics.

The whole feature is a *display* concern: it adds no model tool, mutates no
system prompt or toolset, and therefore has zero effect on prompt caching.
"""

from agent.pet.constants import (
    DEFAULT_SCALE,
    FRAME_H,
    FRAME_W,
    FRAMES_PER_STATE,
    LOOP_MS,
    STATE_ROWS,
    PetState,
)
from agent.pet.state import derive_pet_state

__all__ = [
    "DEFAULT_SCALE",
    "FRAME_H",
    "FRAME_W",
    "FRAMES_PER_STATE",
    "LOOP_MS",
    "STATE_ROWS",
    "PetState",
    "derive_pet_state",
]
