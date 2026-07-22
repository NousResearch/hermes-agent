"""Monitoring emitter: fire-and-forget queue + background dispatcher.

The emitter is the single seam between producers (gateway status hooks, the
diagnostic log handler) and consumers (the OTLP streamers). Its contract is
the hot-path invariant:

    ``emit()`` MUST return in O(microseconds), MUST NOT block on disk/network,
    and MUST NEVER raise into the caller. A monitoring failure is logged
    locally and dropped — it can never affect the gateway or a session.

Mechanism:
  * ``emit(event)`` does a non-blocking ``queue.put_nowait`` wrapped in a bare
    except. On a full queue it drops the *oldest* event and counts the drop.
  * A daemon thread drains the queue and fans each batch out to subscribers
    (the OTLP metric/span/log streamers). Each subscriber is fail-isolated —
    a slow or raising subscriber never affects the hot path or its peers.

Nothing is persisted here. Monitoring is an egress path, not a local store;
if no subscriber is attached, events simply age out of the ring buffer.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_MAX_QUEUE = 10_000  # ring-buffer depth; oldest dropped when full
_DRAIN_BATCH = 256


class MonitoringEmitter:
    """Owns the queue, the dispatcher thread, and the subscriber list."""

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = enabled
        self._q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=_MAX_QUEUE)
        self._dropped = 0
        self._dispatched = 0
        self._stop = threading.Event()
        self._started = False
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        # Live subscribers (the OTLP streamers). Called from the dispatcher
        # thread, fully fail-isolated. Each subscriber is callable(batch: list[dict]).
        self._subscribers: list = []

    # ── public API (hot path) ───────────────────────────────────────────────
    def emit(self, event: Any) -> None:
        """Enqueue an event. Never blocks, never raises.

        ``event`` may be a dataclass with ``to_dict()`` or a plain dict.
        """
        if not self._enabled:
            return
        try:
            payload = event.to_dict() if hasattr(event, "to_dict") else dict(event)
            payload.setdefault("ts_ns", time.time_ns())
            self._ensure_started()
            try:
                self._q.put_nowait(payload)
            except queue.Full:
                # Drop oldest to make room — bounded memory, newest-wins.
                try:
                    self._q.get_nowait()
                    self._dropped += 1
                    self._q.put_nowait(payload)
                except Exception:
                    self._dropped += 1
        except Exception:  # the hot-path invariant: never propagate
            logger.debug("monitoring emit failed", exc_info=True)

    # ── lifecycle ───────────────────────────────────────────────────────────
    def _ensure_started(self) -> None:
        if self._started:
            return
        with self._lock:
            if self._started:
                return
            self._thread = threading.Thread(
                target=self._run, name="hermes-monitoring-dispatch", daemon=True
            )
            self._thread.start()
            self._started = True

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                first = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            batch = [first]
            while len(batch) < _DRAIN_BATCH:
                try:
                    batch.append(self._q.get_nowait())
                except queue.Empty:
                    break
            self._dispatch(batch)

    def _dispatch(self, batch) -> None:
        # Fan-out to subscribers (OTLP streamers) — fully fail-isolated.
        for sub in list(self._subscribers):
            try:
                sub(batch)
            except Exception:
                logger.debug("monitoring subscriber failed", exc_info=True)
        self._dispatched += len(batch)

    def subscribe(self, callback) -> None:
        """Register a live batch subscriber (callable(batch: list[dict]))."""
        if callback not in self._subscribers:
            self._subscribers.append(callback)
        self._enabled = True

    def unsubscribe(self, callback) -> None:
        try:
            self._subscribers.remove(callback)
        except ValueError:
            pass
        if not self._subscribers:
            self._enabled = False

    # ── introspection / shutdown (tests, CLI) ───────────────────────────────
    def flush(self, timeout: float = 2.0) -> None:
        """Block until the queue drains (test/CLI helper, NOT the hot path)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._q.empty():
                # give the dispatcher a tick to finish the in-flight batch
                time.sleep(0.05)
                if self._q.empty():
                    return
            time.sleep(0.02)

    def stats(self) -> Dict[str, int]:
        return {
            "queued": self._q.qsize(),
            "dispatched": self._dispatched,
            "dropped": self._dropped,
            "subscribers": len(self._subscribers),
        }

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._started = False


# ── process-wide singleton ──────────────────────────────────────────────────
_EMITTER: Optional[MonitoringEmitter] = None
_EMITTER_LOCK = threading.Lock()


def get_emitter() -> MonitoringEmitter:
    """Return the process-wide monitoring emitter."""
    global _EMITTER
    if _EMITTER is not None:
        return _EMITTER
    with _EMITTER_LOCK:
        if _EMITTER is None:
            # Collection is opt-in. A plane exporter enables the singleton by
            # attaching its first subscriber; until then producers are no-ops.
            _EMITTER = MonitoringEmitter(enabled=False)
    return _EMITTER


def emit(event: Any) -> None:
    """Module-level convenience: emit via the singleton."""
    get_emitter().emit(event)


def reset_emitter_for_tests(emitter: Optional[MonitoringEmitter] = None) -> None:
    """Swap the singleton (tests only)."""
    global _EMITTER
    with _EMITTER_LOCK:
        if _EMITTER is not None and emitter is not _EMITTER:
            try:
                _EMITTER.close()
            except Exception:
                pass
        _EMITTER = emitter


# Back-compat alias for the salvaged class name used in emozilla's tests.
TelemetryEmitter = MonitoringEmitter

__all__ = [
    "MonitoringEmitter",
    "TelemetryEmitter",
    "get_emitter",
    "emit",
    "reset_emitter_for_tests",
]
