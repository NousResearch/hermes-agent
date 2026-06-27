"""Local telemetry emitter: fire-and-forget queue + background writer.

The emitter is the single seam between instrumentation (the telemetry plugin's hook
callbacks) and durable storage. Its contract is the hot-path invariant:

    ``emit()`` MUST return in O(microseconds), MUST NOT block on disk/network, and
    MUST NEVER raise into the caller. A telemetry failure is logged locally and
    dropped — it can never affect a model call, a tool call, or a session.

Mechanism:
  * ``emit(event)`` does a non-blocking ``queue.put_nowait`` wrapped in a bare except.
    On a full queue it drops the *oldest* event and counts the drop.
  * A daemon thread drains the queue and writes each event to two places:
      1. the append-only JSONL log (source of truth)
      2. the ``tel_*`` SQLite tables in state.db (rebuildable index)
  * The writer uses its own sqlite connection to state.db, separate from SessionDB,
    so telemetry writes never contend with or corrupt session writes.

Local telemetry only. Nothing here uploads anywhere.
"""

from __future__ import annotations

import json
import logging
import queue
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_MAX_QUEUE = 10_000  # ring-buffer depth; oldest dropped when full
_DRAIN_BATCH = 256


def _default_dir() -> Path:
    """Resolve the telemetry dir under the active HERMES_HOME (profile-safe)."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "telemetry"


def _default_db_path() -> Path:
    """Resolve state.db under the active HERMES_HOME (profile-safe)."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "state.db"


# Map a telemetry event dict (its "event" tag) to (table, column-ordered insert).
# Only the columns the indexer knows about are written; unknown keys are ignored,
# so an event carrying extra fields never breaks the insert.
_TABLE_COLUMNS: Dict[str, tuple] = {
    "run": (
        "tel_runs",
        ("run_id", "trace_id", "session_id", "profile_id", "entrypoint",
         "platform", "start_ns", "end_ns", "end_reason",
         "model_call_count", "tool_call_count", "error_count",
         "estimated_cost_usd", "cost_status"),
    ),
    "span": (
        "tel_spans",
        ("span_id", "trace_id", "run_id", "parent_span_id", "name", "kind",
         "start_ns", "end_ns", "status"),
    ),
    "model_call": (
        "tel_model_calls",
        ("span_id", "run_id", "provider", "model", "base_url",
         "input_tokens", "output_tokens", "cache_read_tokens",
         "cache_write_tokens", "reasoning_tokens", "latency_ms", "ttft_ms",
         "estimated_cost_usd", "cost_status", "cost_source", "end_reason",
         "retry_count"),
    ),
    "tool_call": (
        "tel_tool_calls",
        ("span_id", "run_id", "tool_name", "backend",
         "duration_ms", "result_class", "retry_count", "approval"),
    ),
    "error": (
        "tel_error_events",
        ("run_id", "error_class", "subsystem", "recovery", "ts_ns"),
    ),
}


class TelemetryEmitter:
    """Owns the queue, the writer thread, and the telemetry sqlite connection."""

    def __init__(
        self,
        *,
        events_path: Optional[Path] = None,
        db_path: Optional[Path] = None,
        enabled: bool = True,
    ) -> None:
        self._dir = (events_path.parent if events_path else _default_dir())
        self._events_path = events_path or (self._dir / "events.jsonl")
        self._db_path = db_path or _default_db_path()
        self._enabled = enabled
        self._q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=_MAX_QUEUE)
        self._dropped = 0
        self._written = 0
        self._stop = threading.Event()
        self._started = False
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._thread: Optional[threading.Thread] = None
        # Optional live subscribers (e.g. OTLP exporter). Called from the writer
        # thread AFTER durable writes, fully fail-isolated — a subscriber that
        # raises or blocks can never affect the JSONL/SQLite source of truth or
        # the hot path. Each subscriber is callable(batch: list[dict]).
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
            logger.debug("telemetry emit failed", exc_info=True)

    # ── lifecycle ───────────────────────────────────────────────────────────
    def _ensure_started(self) -> None:
        if self._started:
            return
        with self._lock:
            if self._started:
                return
            try:
                self._dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                logger.debug("telemetry dir create failed", exc_info=True)
            self._thread = threading.Thread(
                target=self._run, name="hermes-telemetry-writer", daemon=True
            )
            self._thread.start()
            self._started = True

    def _open_conn(self) -> Optional[sqlite3.Connection]:
        if self._conn is not None:
            return self._conn
        try:
            conn = sqlite3.connect(str(self._db_path), isolation_level=None, timeout=5.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            self._conn = conn
        except Exception:
            logger.debug("telemetry db open failed", exc_info=True)
            self._conn = None
        return self._conn

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
            self._write_batch(batch)

    def _write_batch(self, batch) -> None:
        # JSONL append (source of truth) — best effort.
        try:
            with open(self._events_path, "a", encoding="utf-8") as fh:
                for ev in batch:
                    fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
        except Exception:
            logger.debug("telemetry jsonl append failed", exc_info=True)

        # SQLite index — best effort, per-event so one bad row can't lose the batch.
        conn = self._open_conn()
        if conn is None:
            return
        for ev in batch:
            try:
                self._index_one(conn, ev)
                self._written += 1
            except Exception:
                logger.debug("telemetry index row failed", exc_info=True)

        # Live fan-out (e.g. OTLP) — AFTER durable writes, fully fail-isolated.
        # A slow/raising subscriber never affects JSONL/SQLite or the hot path.
        for sub in self._subscribers:
            try:
                sub(batch)
            except Exception:
                logger.debug("telemetry subscriber failed", exc_info=True)

    def subscribe(self, callback) -> None:
        """Register a live batch subscriber (callable(batch: list[dict])).

        Called from the writer thread after durable writes. Used by the OTLP
        exporter for continuous streaming. Fail-isolated; never on the hot path.
        """
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def unsubscribe(self, callback) -> None:
        try:
            self._subscribers.remove(callback)
        except ValueError:
            pass

    def _index_one(self, conn: sqlite3.Connection, ev: Dict[str, Any]) -> None:
        kind = ev.get("event")
        spec = _TABLE_COLUMNS.get(kind)
        if spec is None:
            return
        table, cols = spec
        values = [ev.get(c) for c in cols]
        placeholders = ", ".join("?" for _ in cols)
        collist = ", ".join(cols)
        conn.execute(
            f"INSERT OR REPLACE INTO {table} ({collist}) VALUES ({placeholders})",
            values,
        )

    # ── introspection / shutdown (tests, CLI) ───────────────────────────────
    def flush(self, timeout: float = 2.0) -> None:
        """Block until the queue drains (test/CLI helper, NOT the hot path)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._q.empty():
                # give the writer a tick to finish the in-flight batch
                time.sleep(0.05)
                if self._q.empty():
                    return
            time.sleep(0.02)

    def stats(self) -> Dict[str, int]:
        return {
            "queued": self._q.qsize(),
            "written": self._written,
            "dropped": self._dropped,
        }

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        self._started = False


# ── process-wide singleton ──────────────────────────────────────────────────
_EMITTER: Optional[TelemetryEmitter] = None
_EMITTER_LOCK = threading.Lock()


def get_emitter() -> TelemetryEmitter:
    """Return the process-wide emitter, honoring telemetry.local config."""
    global _EMITTER
    if _EMITTER is not None:
        return _EMITTER
    with _EMITTER_LOCK:
        if _EMITTER is None:
            enabled = _local_enabled()
            _EMITTER = TelemetryEmitter(enabled=enabled)
    return _EMITTER


def _local_enabled() -> bool:
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        tel = cfg.get("telemetry") if isinstance(cfg, dict) else {}
        return bool((tel or {}).get("local", True))
    except Exception:
        return True


def emit(event: Any) -> None:
    """Module-level convenience: emit via the singleton."""
    get_emitter().emit(event)


def reset_emitter_for_tests(emitter: Optional[TelemetryEmitter] = None) -> None:
    """Swap the singleton (tests only)."""
    global _EMITTER
    with _EMITTER_LOCK:
        if _EMITTER is not None and emitter is not _EMITTER:
            try:
                _EMITTER.close()
            except Exception:
                pass
        _EMITTER = emitter


__all__ = [
    "TelemetryEmitter",
    "get_emitter",
    "emit",
    "reset_emitter_for_tests",
]
