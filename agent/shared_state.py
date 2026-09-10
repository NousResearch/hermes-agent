"""
SharedStateStore — Sub-Agent State Sharing Layer (MI: Module Independence).

Provides a thread-safe, EventBus-backed shared state store that enables sub-agents
to read/write shared state during execution without tight coupling.

Design principles (Module Independence MI):
  - The store is parent-owned — sub-agents never hold a direct reference
  - All communication goes through EventBus events — fire-and-forget by default
  - The store itself is a pure in-memory dict with no sub-agent awareness
  - All mutations are observable via EventBus (traceable)
  - Backward compatible: no changes to SubagentCoordinator core

Event flow:
  Sub-agent emits → shared_state.write → SharedStateStore handler (parent thread)
    → writes to _store → emits shared_state.changed → all EventBus subscribers notified

Usage:

  # Parent: create and attach
  store = SharedStateStore(session_id="abc")
  store.attach_to_event_bus(parent_agent._event_bus)

  # Sub-agent: emit write event (no direct store reference)
  parent_agent._event_bus.emit_event(
      "shared_state.write",
      {"key": "findings", "value": {...}, "writer_id": "subagent-0"}
  )

  # Sub-agent: emit read event (returns via changed event to all subscribers)
  parent_agent._event_bus.emit_event(
      "shared_state.read",
      {"key": "findings"}
  )

  # Subscribe to all state changes (e.g., in parent's delegation loop)
  parent_agent._event_bus.subscribe(
      "shared_state.changed",
      lambda event: print(event.payload)
  )
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─── Change Record ─────────────────────────────────────────────────────────────


@dataclass
class StateChangeRecord:
    """
    Immutable record of a single shared state mutation.

    Emitted as part of the shared_state.changed event payload so that
    consumers can track the full history of state transitions.
    """
    key: str
    value: Any
    writer_id: str          # Identifies which sub-agent wrote this (or read it for reads)
    session_id: str = ""    # Session in which this write occurred
    timestamp: float = field(default_factory=time.time)
    seq: int = 0           # Monotonic sequence number for ordering
    old_value: Any = None  # Previous value before this mutation (None for first write)
    is_read: bool = False  # True if this was triggered by a read event, not a write


# ─── SharedStateStore ───────────────────────────────────────────────────────────


class SharedStateStore:
    """
    Thread-safe shared state store owned by the parent agent.

    Sub-agents communicate exclusively via EventBus events — they never
    hold a reference to this store. The parent is the sole writer to _store;
    sub-agents only emit events that the parent's handler processes.

    Thread safety:
        Uses threading.RLock for all _store mutations and _subscribers access.
        The EventBus itself is thread-safe (RLock in emit/subscribe).
    """

    # Event types for sub-agent → store communication
    EVENT_WRITE = "shared_state.write"
    EVENT_READ = "shared_state.read"
    # Event type emitted by the store to all subscribers
    EVENT_CHANGED = "shared_state.changed"
    EVENT_CLEARED = "shared_state.cleared"

    def __init__(self, session_id: str = ""):
        self._session_id = session_id
        self._store: Dict[str, Any] = {}
        self._lock = threading.RLock()
        # Per-key subscribers: key → list of callbacks
        self._subscribers: Dict[str, List[Callable[[StateChangeRecord], None]]] = {}
        self._sub_lock = threading.Lock()
        # Global subscribers: notified on ANY change
        self._global_subscribers: List[Callable[[StateChangeRecord], None]] = []
        # Sequence counter for total-ordering of writes
        self._seq: int = 0
        # EventBus reference (set by attach_to_event_bus)
        self._event_bus: Optional[Any] = None
        # Bound handler methods for EventBus subscription
        self._on_write = self._handle_write_event
        self._on_read = self._handle_read_event

    # ── EventBus Attachment ──────────────────────────────────────────────────

    def attach_to_event_bus(self, event_bus: Any) -> None:
        """
        Attach this store to an EventBus and start listening for sub-agent events.

        Safe to call multiple times (idempotent — only attaches once).
        """
        if self._event_bus is not None:
            logger.debug("SharedStateStore already attached to an EventBus")
            return
        self._event_bus = event_bus
        event_bus.subscribe(self.EVENT_WRITE, self._on_write)
        event_bus.subscribe(self.EVENT_READ, self._on_read)
        logger.debug("SharedStateStore attached to EventBus (session_id=%s)", self._session_id)

    def detach_from_event_bus(self) -> None:
        """
        Detach this store from its EventBus and stop listening.

        After detachment, sub-agent events are dropped (no-op).
        """
        if self._event_bus is None:
            return
        try:
            self._event_bus.unsubscribe(self.EVENT_WRITE, self._on_write)
            self._event_bus.unsubscribe(self.EVENT_READ, self._on_read)
        except Exception as e:
            logger.debug("Error unsubscribing SharedStateStore from EventBus: %s", e)
        self._event_bus = None
        logger.debug("SharedStateStore detached from EventBus")

    # ── Event Handlers (called from parent thread via EventBus) ────────────

    def _handle_write_event(self, event: Any) -> None:
        """
        Handle a shared_state.write event from a sub-agent.

        Extracts key/value/writer_id from the event payload and updates _store.
        Then emits shared_state.changed to notify all subscribers.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else dict(event)
            key = str(payload.get("key", ""))
            value = payload.get("value")
            writer_id = str(payload.get("writer_id", "unknown"))
            timestamp = payload.get("timestamp", time.time())

            if not key:
                logger.debug("shared_state.write: empty key, ignored")
                return

            with self._lock:
                self._seq += 1
                old_value = self._store.get(key)
                self._store[key] = value
                record = StateChangeRecord(
                    key=key,
                    value=value,
                    old_value=old_value,
                    writer_id=writer_id,
                    session_id=self._session_id,
                    timestamp=timestamp,
                    seq=self._seq,
                    is_read=False,
                )

            logger.debug(
                "SharedStateStore: %s wrote key=%r (seq=%d)",
                writer_id, key, self._seq
            )

            # Emit changed event to all EventBus subscribers
            self._emit_changed(record)

        except Exception as e:
            logger.warning("shared_state.write handler failed: %s", e)

    def _handle_read_event(self, event: Any) -> None:
        """
        Handle a shared_state.read event from a sub-agent.

        Emits a shared_state.changed event with the current value so the
        requesting sub-agent (and all others) can observe the read as a
        "read notification". This lets subscribers track what keys have
        been accessed without blocking the sub-agent.
        """
        try:
            payload = event.payload if hasattr(event, "payload") else dict(event)
            key = str(payload.get("key", ""))
            reader_id = str(payload.get("reader_id", "unknown"))

            if not key:
                logger.debug("shared_state.read: empty key, ignored")
                return

            with self._lock:
                value = self._store.get(key)
                self._seq += 1
                record = StateChangeRecord(
                    key=key,
                    value=value,
                    writer_id=reader_id,  # Reuse writer_id field for reader
                    session_id=self._session_id,
                    timestamp=time.time(),
                    seq=self._seq,
                    is_read=True,
                )

            logger.debug(
                "SharedStateStore: %s read key=%r (seq=%d)",
                reader_id, key, self._seq
            )

            # Emit as a changed event so all subscribers are notified
            self._emit_changed(record)

        except Exception as e:
            logger.warning("shared_state.read handler failed: %s", e)

    def _emit_changed(self, record: StateChangeRecord) -> None:
        """Emit a shared_state.changed event to EventBus and notify in-store subscribers."""
        # First notify in-store subscribers (thread-safe)
        self._notify_subscribers(record)
        # Then emit to EventBus for external subscribers
        if self._event_bus is None:
            return
        try:
            self._event_bus.emit_event(
                self.EVENT_CHANGED,
                {
                    "key": record.key,
                    "value": record.value,
                    "old_value": getattr(record, "old_value", None),
                    "writer_id": record.writer_id,
                    "session_id": record.session_id,
                    "timestamp": record.timestamp,
                    "seq": record.seq,
                    "is_read": getattr(record, "is_read", False),
                },
                session_id=self._session_id,
            )
        except Exception as e:
            logger.debug("Failed to emit shared_state.changed: %s", e)

    # ── Subscriber API ──────────────────────────────────────────────────────

    def subscribe(
        self,
        key: str,
        handler: Callable[[StateChangeRecord], None],
    ) -> None:
        """
        Subscribe to changes for a specific key.

        The handler is called synchronously in the EventBus emit thread
        whenever the key's value changes (or is read).

        Args:
            key: The state key to watch. Use "*" to watch all changes.
            handler: Callback receiving the StateChangeRecord.
        """
        with self._sub_lock:
            if key == "*":
                self._global_subscribers.append(handler)
            else:
                self._subscribers.setdefault(key, []).append(handler)
        logger.debug("Subscribed handler to shared_state key=%r", key)

    def unsubscribe(
        self,
        key: str,
        handler: Callable[[StateChangeRecord], None],
    ) -> None:
        """Unsubscribe a previously registered handler."""
        with self._sub_lock:
            if key == "*":
                self._global_subscribers = [
                    h for h in self._global_subscribers if h != handler
                ]
            elif key in self._subscribers:
                self._subscribers[key] = [
                    h for h in self._subscribers[key] if h != handler
                ]
        logger.debug("Unsubscribed handler from shared_state key=%r", key)

    def _notify_subscribers(self, record: StateChangeRecord) -> None:
        """Notify key-specific and global subscribers of a change."""
        with self._sub_lock:
            key_handlers = list(self._subscribers.get(record.key, []))
            global_handlers = list(self._global_subscribers)

        for handler in key_handlers + global_handlers:
            try:
                handler(record)
            except Exception as e:
                logger.warning(
                    "SharedStateStore subscriber failed for key=%s: %s",
                    record.key, e,
                )

    # ── Direct Store Access (for parent agent only) ─────────────────────────

    def read(self, key: str, default: Any = None) -> Any:
        """
        Read a value from the store (parent thread only).

        Args:
            key: State key to read.
            default: Value returned if key is not present.

        Returns:
            The stored value, or ``default`` if not found.
        """
        with self._lock:
            return self._store.get(key, default)

    def write(self, key: str, value: Any, writer_id: str = "parent") -> None:
        """
        Write a value to the store (parent thread only).

        Emits shared_state.changed so all subscribers are notified.

        Args:
            key: State key to write.
            value: Value to store.
            writer_id: Identifier of the writer (default: "parent").
        """
        with self._lock:
            self._seq += 1
            old_value = self._store.get(key)
            self._store[key] = value
            record = StateChangeRecord(
                key=key,
                value=value,
                old_value=old_value,
                writer_id=writer_id,
                session_id=self._session_id,
                timestamp=time.time(),
                seq=self._seq,
            )
        self._emit_changed(record)

    def clear(self, key: Optional[str] = None) -> None:
        """
        Clear the store.

        Args:
            key: If provided, only that key is removed. If None, entire store is cleared.
        """
        with self._lock:
            if key is not None:
                self._store.pop(key, None)
            else:
                self._store.clear()
                self._seq += 1

        if self._event_bus is not None:
            self._event_bus.emit_event(
                self.EVENT_CLEARED,
                {"key": key, "session_id": self._session_id},
                session_id=self._session_id,
            )
        logger.debug("SharedStateStore cleared (key=%s)", key)

    def keys(self) -> List[str]:
        """Return a snapshot of all keys in the store."""
        with self._lock:
            return list(self._store.keys())

    def get_all(self) -> Dict[str, Any]:
        """Return a snapshot of the entire store."""
        with self._lock:
            return dict(self._store)

    @property
    def seq(self) -> int:
        """Current sequence number (total-ordering of all mutations)."""
        with self._lock:
            return self._seq

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __repr__(self) -> str:
        with self._lock:
            return f"SharedStateStore(session_id={self._session_id!r}, keys={list(self._store.keys())}, seq={self._seq})"
