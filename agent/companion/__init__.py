"""Companiondex companion engine — shared core for the CLI, TUI, and desktop surfaces.

Companiondex (https://github.com/crafter-station/companiondex) is a public gallery of
animated sprite "companions": a ``companion.json`` plus a ``spritesheet.{webp,png}`` of
192×208 px cells (8-col × 9-row Codex atlas, or the older 8-row atlas). This
package is the single source of truth for the feature so the CLI (Python) and
TUI (Ink, via ``tui_gateway``) never duplicate the hard parts: ``constants``
(geometry + :class:`CompanionState`), ``state`` (activity → state), ``manifest``,
``store`` (on-disk companions), ``render`` (kitty / iTerm2 / sixel / half-blocks).
A pure display concern: no model tool, no prompt/toolset mutation, so zero
effect on prompt caching.
"""

from agent.companion.constants import DEFAULT_SCALE, FRAME_H, FRAME_W, FRAMES_PER_STATE, LOOP_MS, STATE_ROWS, CompanionState
from agent.companion.state import derive_companion_state

__all__ = ["DEFAULT_SCALE", "FRAMES_PER_STATE", "FRAME_H", "FRAME_W", "LOOP_MS", "STATE_ROWS", "CompanionState", "derive_companion_state"]
