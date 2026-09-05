"""Ctrl+S prompt stash — pure state machine for the classic CLI composer.

No prompt_toolkit imports so it can be unit tested directly; ``cli.py`` owns only the keybinding
and the rendering. Newest-first: index 0 is always the most recently stashed draft, so "undo my
last Ctrl+S" is a single keystroke.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence, Tuple

# Single-line preview length for the browse panel.
PREVIEW_WIDTH = 60

# Cap the stack so a user leaning on Ctrl+S can't grow it without bound.
MAX_STASH_ITEMS = 20


def build_preview(text: str, width: int = PREVIEW_WIDTH) -> str:
    """Collapse a possibly multi-line draft into one preview line."""
    if not text:
        return ""
    flat = text.replace("\r\n", "\n").replace("\r", "\n")
    flat = flat.replace("\n", " ⏎ ").replace("\t", " ")
    flat = " ".join(flat.split())
    if width > 1 and len(flat) > width:
        return flat[: width - 1] + "…"
    return flat


@dataclass
class StashEntry:
    """One parked draft: exact text plus any images that were attached."""

    text: str
    images: List[Any] = field(default_factory=list)
    stashed_at: float = 0.0
    preview: str = ""

    def as_dict(self) -> dict:
        """Shape ``HermesCLI._render_stash_panel`` consumes."""
        return {"text": self.text, "images": list(self.images), "stashed_at": self.stashed_at, "preview": self.preview}


class PromptStash:
    """Session-scoped stack of parked composer drafts."""

    def __init__(self, *, max_items: int = MAX_STASH_ITEMS, clock=None):
        self._items: List[StashEntry] = []
        self._max_items = max(1, int(max_items))
        self._clock = clock or time.monotonic
        self.panel_open = False
        self.panel_cursor = 0

    def __len__(self) -> int:
        return len(self._items)

    @property
    def items(self) -> List[StashEntry]:
        """Newest-first list of entries (a copy — mutate via the API)."""
        return list(self._items)

    def panel_rows(self) -> List[dict]:
        return [e.as_dict() for e in self._items]

    def indicator(self) -> str:
        """Status-bar indicator: ``📌 2`` idle, ``📌 2 ▲`` while browsing, ``""`` when empty."""
        n = len(self._items)
        return "" if not n else f"📌 {n} ▲" if self.panel_open else f"📌 {n}"

    def placeholder_hint(self) -> str:
        """Composer placeholder text advertising the stashed draft."""
        n = len(self._items)
        if n == 1:
            return f"Ctrl+S to restore: {self._items[0].preview}"
        return f"Ctrl+S to browse {n} stashed drafts" if n else ""

    def stash(self, text: str, images: Optional[Sequence[Any]] = None) -> bool:
        """Push a draft. A blank buffer with no images is a no-op (returns False) so Ctrl+S on an
        empty composer triggers the restore half of the gesture instead of pushing junk.
        """
        if not (text or "").strip() and not images:
            return False

        entry = StashEntry(text=text or "", images=list(images or []), stashed_at=self._clock(),
                           preview=build_preview(text or "") or "(images only)")
        self._items.insert(0, entry)
        del self._items[self._max_items:]  # drop the oldest past the cap
        self.close_panel()  # a push invalidates any open browse session
        return True

    def pop(self, index: int = 0) -> Optional[Tuple[str, List[Any]]]:
        """Remove and return ``(text, images)`` at ``index``, or None."""
        if not 0 <= index < len(self._items):
            return None
        entry = self._items.pop(index)
        if not self._items:
            self.panel_open = False
        self.panel_cursor = self._clamp_cursor(self.panel_cursor)
        return entry.text, list(entry.images)

    def peek(self, index: int = 0) -> Optional[StashEntry]:
        return self._items[index] if 0 <= index < len(self._items) else None

    def clear(self) -> None:
        self._items.clear()
        self.close_panel()

    # ------------------------------------------------------------ panel state

    def _clamp_cursor(self, value: int) -> int:
        return max(0, min(int(value), len(self._items) - 1)) if self._items else 0

    def open_panel(self) -> bool:
        """Open the browse panel. False when there is nothing to browse."""
        if not self._items:
            return False
        self.panel_open = True
        self.panel_cursor = 0
        return True

    def close_panel(self) -> None:
        self.panel_open = False
        self.panel_cursor = 0

    def move_cursor(self, delta: int) -> int:
        """Move the panel cursor, clamped to the list bounds."""
        self.panel_cursor = self._clamp_cursor(self.panel_cursor + int(delta))
        return self.panel_cursor

    def delete_at_cursor(self) -> bool:
        """Delete the highlighted entry. False when there was nothing to drop."""
        if not self._items:
            return False
        idx = self._clamp_cursor(self.panel_cursor)
        self._items.pop(idx)
        if not self._items:
            self.close_panel()
        else:
            self.panel_cursor = self._clamp_cursor(idx)
        return True

    def restore_at_cursor(self) -> Optional[Tuple[str, List[Any]]]:
        """Pop the highlighted entry and close the panel."""
        if not self._items:
            return None
        result = self.pop(self._clamp_cursor(self.panel_cursor))
        self.close_panel()
        return result


# Outcomes of a single Ctrl+S press.
ACTION_NOOP = "noop"
ACTION_STASHED = "stashed"
ACTION_RESTORED = "restored"
ACTION_OPEN_PANEL = "open_panel"
ACTION_CLOSE_PANEL = "close_panel"


def resolve_ctrl_s(
    stash: PromptStash, buffer_text: str, images: Optional[Sequence[Any]] = None
) -> Tuple[str, Optional[Tuple[str, List[Any]]]]:
    """Decide what one Ctrl+S press does. Returns ``(action, payload)`` where ``payload`` is
    ``(text, images)`` for :data:`ACTION_RESTORED`, else None.
    """
    # Panel open → Ctrl+S is the "close it" escape hatch.
    if stash.panel_open:
        stash.close_panel()
        return ACTION_CLOSE_PANEL, None

    # Something to park → park it (onto a stack, so an earlier draft is still reachable).
    if (buffer_text or "").strip() or images:
        return (ACTION_STASHED if stash.stash(buffer_text, images) else ACTION_NOOP), None

    # Empty buffer → restore half of the gesture.
    count = len(stash)
    if count == 0:
        return ACTION_NOOP, None
    if count == 1:
        return ACTION_RESTORED, stash.pop(0)
    stash.open_panel()
    return ACTION_OPEN_PANEL, None
