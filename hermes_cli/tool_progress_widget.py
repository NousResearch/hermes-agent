"""Multi-lane tool progress widget for the prompt_toolkit TUI.

Inspired by SubQ Code's research-progress component. Shows parallel
tool/agent activity as distinct lanes with animated spinners,
completion checkmarks, and tool call counts.

Integrates with the existing TUI via ConditionalContainer in cli.py
and is driven by the _on_tool_progress / _on_tool_complete callbacks.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

# Lane states
LANE_WAITING = "waiting"
LANE_ACTIVE = "active"
LANE_DONE = "done"
LANE_ERROR = "error"

from hermes_cli.braille_animations import ANIMATIONS as _ANIMATIONS

SPINNER_FRAMES = _ANIMATIONS["braille"]["frames"]
_SPINNER_INTERVAL_MS = _ANIMATIONS["braille"]["interval_ms"]

STATUS_ICONS = {
    LANE_WAITING: "◌",
    LANE_DONE: "✓",
    LANE_ERROR: "✗",
}

LANE_NAME_WIDTH = 14
PAD_X = 2
THROTTLE_MS = 50


@dataclass
class LaneState:
    name: str
    status: str = LANE_WAITING
    action: str = "Waiting..."
    tool_calls: int = 0


class ToolProgressTracker:
    """Thread-safe tracker for tool execution state.

    Usage from cli.py::

        tracker = ToolProgressTracker()
        tracker.on_tool_start("terminal", "whoami")
        tracker.on_tool_complete("terminal")
        tracker.on_turn_end()  # clears all lanes
    """

    def __init__(self) -> None:
        self._lanes: Dict[str, LaneState] = {}
        self._lock = threading.Lock()
        self._turn_tool_counts: Dict[str, int] = {}
        self._invalidate_cb: Optional[Callable[[], None]] = None
        self._last_invalidate: float = 0.0

    def set_invalidate_callback(self, cb: Callable[[], None]) -> None:
        self._invalidate_cb = cb

    def on_tool_start(self, tool_name: str, preview: str = "") -> None:
        with self._lock:
            if tool_name not in self._lanes:
                self._lanes[tool_name] = LaneState(
                    name=tool_name,
                    status=LANE_ACTIVE,
                    action=preview or f"Running {tool_name}...",
                )
            else:
                lane = self._lanes[tool_name]
                lane.status = LANE_ACTIVE
                lane.action = preview or f"Running {tool_name}..."

            self._lanes[tool_name].tool_calls += 1
            self._turn_tool_counts[tool_name] = (
                self._turn_tool_counts.get(tool_name, 0) + 1
            )
        self._throttled_invalidate()

    def on_tool_complete(self, tool_name: str, success: bool = True) -> None:
        with self._lock:
            if tool_name in self._lanes:
                lane = self._lanes[tool_name]
                lane.status = LANE_DONE if success else LANE_ERROR
                lane.action = "Done" if success else "Failed"
        self._throttled_invalidate()

    def on_turn_end(self) -> None:
        """Reset all lanes at end of agent turn."""
        with self._lock:
            self._lanes.clear()
            self._turn_tool_counts.clear()
        self._throttled_invalidate()

    def get_active_lanes(self) -> List[LaneState]:
        with self._lock:
            return list(self._lanes.values())

    def get_turn_summary(self) -> Dict[str, int]:
        """Return {tool_name: call_count} for session stats."""
        with self._lock:
            return dict(self._turn_tool_counts)

    def has_activity(self) -> bool:
        with self._lock:
            return len(self._lanes) > 0

    def _throttled_invalidate(self) -> None:
        cb = None
        with self._lock:
            now = time.monotonic() * 1000
            if now - self._last_invalidate >= THROTTLE_MS and self._invalidate_cb:
                self._last_invalidate = now
                cb = self._invalidate_cb
        if cb:
            cb()


def build_progress_fragments(
    tracker: ToolProgressTracker,
    width: int,
) -> List[Tuple[str, str]]:
    """Build prompt_toolkit (style, text) fragments for the progress widget.

    Returns an empty list when there are no active lanes.
    """
    lanes = tracker.get_active_lanes()
    if not lanes:
        return []

    fragments: List[Tuple[str, str]] = []
    inner = width - PAD_X * 2
    pad = " " * PAD_X
    border = "─" * max(inner, 1)

    # Header
    fragments.append(("class:hint", f"{pad}{border}\n"))
    fragments.append(("class:status-bar-strong", f"{pad}Tool Activity"))
    fragments.append(("class:hint", f" ({len(lanes)} tools)\n"))
    fragments.append(("class:hint", f"{pad}{border}\n"))

    for lane in lanes:
        # Status icon
        if lane.status == LANE_ACTIVE:
            frame_idx = int(time.monotonic() * 1000) // _SPINNER_INTERVAL_MS % len(SPINNER_FRAMES)
            icon = SPINNER_FRAMES[frame_idx]
            icon_style = "class:status-bar-strong"
        elif lane.status == LANE_DONE:
            icon = STATUS_ICONS[LANE_DONE]
            icon_style = "class:status-bar-good"
        elif lane.status == LANE_ERROR:
            icon = STATUS_ICONS[LANE_ERROR]
            icon_style = "class:status-bar-bad"
        else:
            icon = STATUS_ICONS[LANE_WAITING]
            icon_style = "class:hint"

        # Lane name
        name = lane.name[:LANE_NAME_WIDTH].ljust(LANE_NAME_WIDTH)
        name_style = "class:hint" if lane.status == LANE_DONE else "class:status-bar"

        # Action text
        action_max = max(10, inner - LANE_NAME_WIDTH - 12)
        action = lane.action[:action_max]
        action_style = "class:hint"

        # Counter
        counter = f"{lane.tool_calls}x" if lane.tool_calls > 0 else ""
        counter = counter.rjust(6)

        action_pad = max(0, action_max - len(action))
        fragments.append((icon_style, f"{pad}{icon} "))
        fragments.append((name_style, name))
        fragments.append((action_style, action))
        fragments.append(("", " " * action_pad))
        fragments.append(("class:hint", f" {counter}\n"))

    fragments.append(("class:hint", f"{pad}{border}\n"))

    return fragments
