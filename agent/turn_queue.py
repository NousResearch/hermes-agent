"""TurnQueue — a thread-safe FIFO of prompts to run as the next turn(s).

This is the agent-side queue that unifies message queueing across CLI, TUI
gateway, desktop, and messaging platforms.  It lives on the AIAgent instance,
alongside the existing ``_pending_steer`` mechanism.

Unlike steer (which injects into the *current* turn's tool results), queued
prompts become the *next* user turn after the current one finishes.  The
drain point is the agent-loop / gateway-runner tail — wherever the current
turn ends and ``running`` flips to False.

Thread-safety: all mutations go through ``self._lock``.  Safe to call from
gateway threads, CLI process_loop, and the agent execution thread.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class QueuedTurn:
    """A single queued prompt entry."""

    id: str
    text: str
    mode: str = "queue"  # "queue" | "interrupt" | "steer"
    transport: Any = None  # pinned transport for the drained turn
    queued_at: float = field(default_factory=time.time)
    attachments: list = field(default_factory=list)
    # Where the entry came from: "queue" (explicit session.queue.add — the
    # client shows it in a queue panel, not the transcript) or "busy_submit"
    # (a prompt.submit that landed mid-turn — the client already echoed it
    # as an optimistic user message). Lets drain events tell clients whether
    # the text still needs painting.
    source: str = "queue"

    def to_dict(self) -> dict:
        """Serialise for RPC / event emission (no transport internals)."""
        return {
            "id": self.id,
            "text": self.text,
            "mode": self.mode,
            "queued_at": self.queued_at,
            "attachments": self.attachments,
            "source": self.source,
        }


class TurnQueue:
    """Thread-safe FIFO of prompts waiting to become the next turn.

    Replaces the ad-hoc ``session["queued_prompt"]`` dict slot in the TUI
    gateway and the ``localStorage``-backed queue in the desktop renderer.
    The queue lives in the agent process so it drains even when no client
    window is open.
    """

    def __init__(self) -> None:
        self._entries: list[QueuedTurn] = []
        self._lock = threading.Lock()

    # ── enqueue ──────────────────────────────────────────────

    def enqueue(
        self,
        text: str,
        mode: str = "queue",
        transport: Any = None,
        attachments: list | None = None,
        source: str = "queue",
    ) -> QueuedTurn:
        """Add a prompt to the back of the queue.

        Consecutive text entries are *not* merged here — the gateway's
        ``_handle_busy_submit`` still does merge if it wants to (mirroring
        ``repair_message_sequence``).  Callers that want merge semantics
        should check ``peek()`` first.

        Returns the created :class:`QueuedTurn`.
        """
        entry = QueuedTurn(
            id=str(uuid.uuid4()),
            text=text,
            mode=mode,
            transport=transport,
            attachments=list(attachments) if attachments else [],
            source=source,
        )
        with self._lock:
            self._entries.append(entry)
        return entry

    def enqueue_front(
        self,
        text: str,
        mode: str = "queue",
        transport: Any = None,
        attachments: list | None = None,
    ) -> QueuedTurn:
        """Add a prompt to the *front* of the queue (priority send)."""
        entry = QueuedTurn(
            id=str(uuid.uuid4()),
            text=text,
            mode=mode,
            transport=transport,
            attachments=list(attachments) if attachments else [],
        )
        with self._lock:
            self._entries.insert(0, entry)
        return entry

    # ── drain ────────────────────────────────────────────────

    def drain(self) -> Optional[QueuedTurn]:
        """Pop and return the head entry, or None if the queue is empty."""
        with self._lock:
            if not self._entries:
                return None
            return self._entries.pop(0)

    # ── peek / inspect ───────────────────────────────────────

    def peek(self) -> list[QueuedTurn]:
        """Return a snapshot copy of pending entries (for UI sync)."""
        with self._lock:
            return list(self._entries)

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._entries) == 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __bool__(self) -> bool:
        return len(self) > 0

    # ── mutate ───────────────────────────────────────────────

    def remove(self, entry_id: str) -> bool:
        """Remove a specific entry by id. Returns True if found."""
        with self._lock:
            before = len(self._entries)
            self._entries = [e for e in self._entries if e.id != entry_id]
            return len(self._entries) < before

    def promote(self, entry_id: str) -> bool:
        """Move an entry to the front of the queue. Returns True if found."""
        with self._lock:
            idx = None
            for i, e in enumerate(self._entries):
                if e.id == entry_id:
                    idx = i
                    break
            if idx is None or idx == 0:
                return False
            entry = self._entries.pop(idx)
            self._entries.insert(0, entry)
            return True

    def update_text(self, entry_id: str, text: str) -> bool:
        """Update the text of a queued entry. Returns True if found."""
        with self._lock:
            for e in self._entries:
                if e.id == entry_id:
                    e.text = text
                    return True
            return False

    def clear(self) -> int:
        """Clear all entries. Returns the number removed."""
        with self._lock:
            n = len(self._entries)
            self._entries.clear()
            return n

    # ── serialisation ────────────────────────────────────────

    def to_list(self) -> list[dict]:
        """Return a list of plain dicts for RPC / event emission."""
        with self._lock:
            return [e.to_dict() for e in self._entries]
