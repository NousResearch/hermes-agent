"""Punctuation-based message windowing for Daimon.

Accumulates messages between @mentions in a per-thread ring buffer.
On @mention (the "punctuation event"), the buffer is flushed and all
accumulated messages become context for the agent's response.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime



@dataclass(frozen=True)
class BufferedMessage:
    """A single message accumulated between @mentions."""

    author_name: str
    author_id: str
    content: str
    timestamp: datetime
    has_attachments: bool = False


class WindowBuffer:
    """Per-thread ring buffer accumulating messages between @mentions.

    Thread-safe. Each thread_id gets its own bounded deque.
    When a thread exceeds MAX_PER_THREAD, oldest messages are evicted.
    When total tracked threads exceed MAX_THREADS, the least-recently-used
    thread buffer is evicted entirely.
    """

    def __init__(self, max_per_thread: int = 50, max_threads: int = 5000) -> None:
        self._max_per_thread = max_per_thread
        self._max_threads = max_threads
        self._lock = threading.Lock()
        self._buffers: dict[str, deque[BufferedMessage]] = {}
        # Idempotency: track recent message IDs to prevent double-processing
        self._seen_ids: dict[str, deque[str]] = {}  # thread_id → recent message IDs
        _SEEN_IDS_MAX = 100  # per thread

    def has_seen(self, thread_id: str, message_id: str) -> bool:
        """Check if a message ID has already been processed (dedup)."""
        with self._lock:
            seen = self._seen_ids.get(thread_id)
            if seen and message_id in seen:
                return True
            return False

    def mark_seen(self, thread_id: str, message_id: str) -> None:
        """Mark a message ID as processed."""
        with self._lock:
            if thread_id not in self._seen_ids:
                self._seen_ids[thread_id] = deque(maxlen=100)
            self._seen_ids[thread_id].append(message_id)

    def append(self, thread_id: str, msg: BufferedMessage) -> None:
        """Add a message to the thread's buffer. Evicts oldest if at cap."""
        with self._lock:
            if thread_id not in self._buffers:
                # Evict oldest thread if at capacity
                if len(self._buffers) >= self._max_threads:
                    oldest_key = next(iter(self._buffers))
                    del self._buffers[oldest_key]
                self._buffers[thread_id] = deque(maxlen=self._max_per_thread)
            self._buffers[thread_id].append(msg)

    def flush(self, thread_id: str) -> list[BufferedMessage]:
        """Return all buffered messages for a thread and clear the buffer.

        Returns empty list if no messages buffered.
        """
        with self._lock:
            buf = self._buffers.pop(thread_id, None)
            if buf is None:
                return []
            return list(buf)

    def clear(self, thread_id: str) -> None:
        """Remove buffer and seen IDs for a thread (cleanup on close/archive)."""
        with self._lock:
            self._buffers.pop(thread_id, None)
            self._seen_ids.pop(thread_id, None)

    @property
    def tracked_threads(self) -> int:
        """Number of threads with active buffers."""
        with self._lock:
            return len(self._buffers)

    def peek_count(self, thread_id: str) -> int:
        """Return number of buffered messages for a thread without flushing."""
        with self._lock:
            buf = self._buffers.get(thread_id)
            return len(buf) if buf else 0


def format_window_context(buffered: list[BufferedMessage], trigger_author: str = "") -> str:
    """Format buffered messages into context string prepended to the trigger.

    Returns empty string if no buffered messages (trigger message is sufficient).
    """
    if not buffered:
        return ""

    parts = ["[Messages since last response]"]
    for msg in buffered:
        line = f"{msg.author_name}: {msg.content}"
        if msg.has_attachments:
            line += " [+attachments]"
        parts.append(line)
    parts.append("[Current request:]")
    return "\n".join(parts) + "\n\n"
