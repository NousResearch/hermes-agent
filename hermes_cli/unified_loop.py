"""Unified loop architecture — single SQLite-backed state store, stateless ticker.

Replaces the dual LoopScheduler (in-process) + cron (external) backends with:
  - ``UnifiedLoopTicker``  — one background thread per process, DB-driven
  - ``UnifiedLoopManager`` — per-session CRUD API, all surfaces share this

The ``loops`` table in SessionDB stores all state.  Cross-process safety
is via SQLite optimistic locking (``WHERE last_fired_at = ?``), not
Python thread locks.
"""

from __future__ import annotations

import heapq
import logging
import os
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Re-export formatting helpers so callers don't need to import loop.py
# ──────────────────────────────────────────────────────────────────────

from hermes_cli.loop import (
    DEFAULT_MAX_TURNS,
    MIN_INTERVAL_SECONDS,
    _format_countdown,
    format_interval,
    _parse_interval,
)

# ──────────────────────────────────────────────────────────────────────
# UID generation (matches loop.py convention)
# ──────────────────────────────────────────────────────────────────────


def _new_loop_uid() -> str:
    """Generate a short, collision-resistant loop UID."""
    return uuid.uuid4().hex[:8]


# ──────────────────────────────────────────────────────────────────────
# UnifiedLoopTicker — singleton background thread
# ──────────────────────────────────────────────────────────────────────


class UnifiedLoopTicker:
    """One background thread that ticks ALL loops across ALL sessions.

    NOTE: This singleton is per-process.  Cross-process safety is ensured
    by SQLite optimistic locking, not by Python locks.
    """

    _instance: Optional["UnifiedLoopTicker"] = None
    _lock = threading.Lock()

    def __init__(self, hermes_home: str) -> None:
        from hermes_state import SessionDB

        self.hermes_home = hermes_home
        self.db = SessionDB()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._health_thread: Optional[threading.Thread] = None

        # Priority queue: (next_fire_at, session_id, uid)
        self._heap: List[Tuple[float, str, str]] = []
        self._heap_lock = threading.Lock()
        self._heap_dirty = threading.Event()

        # Dispatcher registry: backend -> handler(session_id, body, *, source_json)
        # Keys: "cli", "gateway", "tui" — NOT platform names.
        # The loops table ``platform`` field stores the backend key, not the
        # messaging platform (telegram/discord/etc).  The ``source_json`` field
        # carries the full SessionSource for routing.
        self._dispatchers: Dict[str, Callable] = {}
        self._dispatcher_lock = threading.Lock()

    @classmethod
    def get_instance(cls, hermes_home: str) -> "UnifiedLoopTicker":
        """Return the singleton ticker, creating it on first call."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(hermes_home)
            return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Tear down the singleton (for tests only)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.stop(timeout=2.0)
            cls._instance = None

    # ── Dispatcher registry ──────────────────────────────────────────

    def register_dispatcher(
        self,
        backend: str,
        handler: Callable,
    ) -> None:
        """Register a dispatch handler for a backend.

        Handler signature::

            handler(session_id: str, body: str, *, source_json: Optional[str] = None)

        Backend keys: ``"cli"``, ``"gateway"``, ``"tui"`` — one per dispatch
        mechanism.  NOT platform names.  The loops table ``platform`` field
        stores the backend key.

        CLI registers:     ``register_dispatcher("cli", cli_dispatch)``
        Gateway registers: ``register_dispatcher("gateway", gateway_dispatch)``
        TUI registers:     ``register_dispatcher("tui", tui_dispatch)``
        """
        with self._dispatcher_lock:
            self._dispatchers[backend] = handler
        self._heap_dirty.set()

    def unregister_dispatcher(self, backend: str) -> None:
        """Remove a dispatch handler."""
        with self._dispatcher_lock:
            self._dispatchers.pop(backend, None)

    # ── Lifecycle ────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the ticker thread.  Idempotent — no-op if already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        is_cli = os.environ.get("HERMES_CLI_MODE", "false").lower() == "true"
        self._thread = threading.Thread(
            target=self._run,
            name="unified-loop-ticker",
            daemon=is_cli,
        )
        self._thread.start()
        if is_cli:
            import atexit

            atexit.register(self.stop)
        # Start health check thread (non-daemon, short-lived)
        self._health_thread = threading.Thread(
            target=self._health_check,
            name="unified-loop-health",
            daemon=True,
        )
        self._health_thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the ticker to stop and wait for it to finish."""
        self._stop.set()
        self._heap_dirty.set()  # wake _run() if sleeping
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("UnifiedLoopTicker thread did not stop gracefully")

    def _health_check(self) -> None:
        """Respawn ticker thread if it dies unexpectedly."""
        respawn_count = 0
        max_respawns = 5
        while not self._stop.is_set():
            self._stop.wait(timeout=30.0)
            if self._stop.is_set():
                break
            if self._thread and not self._thread.is_alive():
                respawn_count += 1
                if respawn_count > max_respawns:
                    logger.error(
                        "UnifiedLoopTicker respawned %d times — giving up",
                        respawn_count,
                    )
                    break
                logger.error(
                    "UnifiedLoopTicker died — respawning (%d/%d)",
                    respawn_count,
                    max_respawns,
                )
                self.start()

    # ── Main loop ────────────────────────────────────────────────────

    def _run(self) -> None:
        """Ticker main loop — rebuild heap, sleep until next fire, dispatch."""
        while not self._stop.is_set():
            try:
                self._rebuild_heap_if_dirty()
                sleep_seconds = self._compute_sleep()
                self._stop.wait(timeout=sleep_seconds)
                if not self._stop.is_set():
                    self._tick_due()
            except Exception:
                logger.exception("UnifiedLoopTicker tick failed")
                self._stop.wait(timeout=1.0)

    def _rebuild_heap_if_dirty(self) -> None:
        """Rebuild the in-memory heap from the DB if flagged dirty."""
        if not self._heap_dirty.is_set():
            return
        now = time.time()

        def _do(conn: Any) -> Any:
            rows = conn.execute(
                "SELECT session_id, uid, last_fired_at, interval_seconds "
                "FROM loops WHERE status = 'active'"
            ).fetchall()
            return rows

        rows = self.db._execute_read(_do)
        with self._heap_lock:
            self._heap = [
                (
                    r["last_fired_at"] + r["interval_seconds"],
                    r["session_id"],
                    r["uid"],
                )
                for r in rows
            ]
            heapq.heapify(self._heap)
        self._heap_dirty.clear()

    def _compute_sleep(self) -> float:
        """Return seconds until the next loop is due (capped at 60s)."""
        with self._heap_lock:
            if not self._heap:
                return 5.0  # no loops — sleep longer to reduce CPU
            next_fire, _, _ = self._heap[0]
        return max(0.1, min(next_fire - time.time(), 60.0))

    def _tick_due(self) -> None:
        """Pop due entries from heap, fire them, dispatch outside lock."""
        now = time.time()
        # Pop due entries from heap under lock (fast, no DB)
        candidates: List[Tuple[float, str, str]] = []
        with self._heap_lock:
            while self._heap and self._heap[0][0] <= now:
                candidates.append(heapq.heappop(self._heap))
            self._heap_dirty.set()

        # Try fire outside lock — DB writes may block
        due: List[Tuple[str, str, dict]] = []
        for _next_fire, session_id, uid in candidates:
            row_data = self._try_fire_loop(session_id, uid, now)
            if row_data is not None:
                due.append((session_id, uid, row_data))

        # Dispatch outside lock — handlers may block
        for session_id, uid, row_data in due:
            self._dispatch_loop(session_id, uid, row_data)

        if due:
            logger.debug("Fired %d loops", len(due))

    # ── SQLite optimistic locking ────────────────────────────────────

    def _try_fire_loop(
        self,
        session_id: str,
        uid: str,
        now: float,
    ) -> Optional[dict]:
        """Atomically check interval, enforce max_turns, and update last_fired_at.

        Returns the row data dict if the loop was fired (caller must dispatch),
        ``None`` if another process already fired it, interval not elapsed,
        or done.

        NOTE: The SELECT and UPDATE must be in the same ``_execute_write`` call
        (same SQLite transaction) to prevent TOCTOU races.  If we read in one
        transaction and write in another, another process could fire between them.
        """

        def _do(conn: Any) -> Optional[dict]:
            row = conn.execute(
                "SELECT last_fired_at, interval_seconds, status, "
                "turns_completed, max_turns, prompt, platform, source_json "
                "FROM loops WHERE session_id = ? AND uid = ?",
                (session_id, uid),
            ).fetchone()
            if row is None:
                return None

            last_fired, interval, status, turns, max_turns = (
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
            )
            if status != "active":
                return None
            if turns >= max_turns:
                conn.execute(
                    "UPDATE loops SET status = 'done', reason = 'max_turns reached' "
                    "WHERE session_id = ? AND uid = ?",
                    (session_id, uid),
                )
                return None
            if last_fired > 0 and (now - last_fired) < interval:
                return None

            # Optimistic lock: only update if last_fired_at hasn't changed
            cur = conn.execute(
                "UPDATE loops SET last_fired_at = ?, turns_completed = turns_completed + 1 "
                "WHERE session_id = ? AND uid = ? AND last_fired_at = ?",
                (now, session_id, uid, last_fired),
            )
            if cur.rowcount > 0:
                return {
                    "prompt": row[5],
                    "platform": row[6],
                    "source_json": row[7],
                }
            return None

        return self.db._execute_write(_do)

    # ── Dispatch ─────────────────────────────────────────────────────

    def _dispatch_loop(
        self,
        session_id: str,
        uid: str,
        row_data: dict,
    ) -> None:
        """Dispatch a fired loop to the registered backend handler.

        ``row_data`` is the dict returned by ``_try_fire_loop`` — no second
        DB read.

        NOTE: ``turns_completed`` was already incremented by ``_try_fire_loop``.
        This is at-most-once delivery — if dispatch fails, the turn is lost.
        The existing ``LoopScheduler`` retries on busy without counting the turn.
        This is a deliberate simplification: loops are best-effort, not guaranteed.
        """
        backend = row_data.get("platform", "cli")  # column is still "platform" in DB
        body = row_data.get("prompt", "")
        source_json = row_data.get("source_json")

        with self._dispatcher_lock:
            handler = self._dispatchers.get(backend)

        if handler:
            try:
                handler(session_id, body, source_json=source_json)
            except Exception:
                logger.exception(
                    "Loop dispatch failed for %s:%s", session_id, uid
                )
        else:
            logger.warning(
                "No dispatcher for backend %r — loop prompt dropped", backend
            )

    def mark_heap_dirty(self) -> None:
        """Signal the ticker to rebuild its heap from the DB."""
        self._heap_dirty.set()


# ──────────────────────────────────────────────────────────────────────
# UnifiedLoopManager — per-session CRUD API
# ──────────────────────────────────────────────────────────────────────


class UnifiedLoopManager:
    """Unified loop management API.  All surfaces use this."""

    def __init__(self, session_id: str, hermes_home: str) -> None:
        from hermes_state import SessionDB

        self.session_id = session_id
        self.db = SessionDB()
        self.ticker = UnifiedLoopTicker.get_instance(hermes_home)
        # Formatting helpers (from hermes_cli/loop.py)
        self._format_interval = format_interval
        self._format_countdown = _format_countdown
        self._MIN_INTERVAL_SECONDS = MIN_INTERVAL_SECONDS
        self._DEFAULT_MAX_TURNS = DEFAULT_MAX_TURNS

    def create(
        self,
        body: str,
        interval_seconds: int,
        source_json: Optional[str] = None,
        platform: str = "cli",
        max_turns: Optional[int] = None,
        fire_now: bool = False,
    ) -> str:
        """Create a new loop.  Returns UID.

        ``source_json``: full ``SessionSource.to_dict()`` JSON for dispatch
        routing.  CLI passes ``None`` (queue-based, no routing needed).
        Gateway passes ``event.source.to_dict()`` as JSON.

        ``fire_now=True``: first fire happens on next tick (CLI backward compat).
        ``fire_now=False``: first fire happens after ``interval_seconds``
        (Gateway default).
        """
        body = (body or "").strip()
        if not body:
            raise ValueError("loop prompt is empty")
        interval_seconds = max(int(interval_seconds), self._MIN_INTERVAL_SECONDS)
        if max_turns is None:
            max_turns = self._DEFAULT_MAX_TURNS
        uid = _new_loop_uid()
        now = time.time()

        def _do(conn: Any) -> None:
            conn.execute(
                "INSERT INTO loops (session_id, uid, prompt, interval_seconds, "
                "last_fired_at, turns_completed, max_turns, status, created_at, "
                "source_json, platform, reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self.session_id,
                    uid,
                    body,
                    interval_seconds,
                    0.0 if fire_now else now,
                    0,
                    max_turns,
                    "active",
                    now,
                    source_json,
                    platform,
                    None,
                ),
            )

        self.db._execute_write(_do)
        self.ticker.mark_heap_dirty()
        return uid

    def delete(self, uid: str) -> bool:
        """Delete a loop by UID."""

        def _do(conn: Any) -> bool:
            cur = conn.execute(
                "DELETE FROM loops WHERE session_id = ? AND uid = ?",
                (self.session_id, uid),
            )
            return cur.rowcount > 0

        result = self.db._execute_write(_do)
        self.ticker.mark_heap_dirty()
        return result

    def delete_all(self) -> int:
        """Delete all loops for this session.  Returns count removed."""

        def _do(conn: Any) -> int:
            cur = conn.execute(
                "DELETE FROM loops WHERE session_id = ?",
                (self.session_id,),
            )
            return cur.rowcount

        count = self.db._execute_write(_do)
        self.ticker.mark_heap_dirty()
        return count

    def pause(self, uid: str, reason: Optional[str] = None) -> bool:
        """Pause a loop.  Stores reason for display in ``status_line()``."""

        def _do(conn: Any) -> bool:
            cur = conn.execute(
                "UPDATE loops SET status = 'paused', reason = ? "
                "WHERE session_id = ? AND uid = ?",
                (reason, self.session_id, uid),
            )
            return cur.rowcount > 0

        result = self.db._execute_write(_do)
        self.ticker.mark_heap_dirty()
        return result

    def resume(self, uid: str, fire_now: bool = False) -> bool:
        """Resume a paused loop."""
        now = time.time()

        def _do(conn: Any) -> bool:
            cur = conn.execute(
                "UPDATE loops SET status = 'active', last_fired_at = ? "
                "WHERE session_id = ? AND uid = ?",
                (0.0 if fire_now else now, self.session_id, uid),
            )
            return cur.rowcount > 0

        result = self.db._execute_write(_do)
        self.ticker.mark_heap_dirty()
        return result

    def list(self) -> List[dict]:
        """Return all loops for this session as a list of dicts."""

        def _do(conn: Any) -> List[dict]:
            rows = conn.execute(
                "SELECT * FROM loops WHERE session_id = ?",
                (self.session_id,),
            ).fetchall()
            return [dict(r) for r in rows]

        return self.db._execute_read(_do)

    # ── Introspection ──────────────────────────────────────────────

    @property
    def active_loop(self) -> Optional[dict]:
        """Return first active loop, or first paused, or ``None``."""
        loops = self.list()
        for loop in loops:
            if loop["status"] == "active":
                return loop
        for loop in loops:
            if loop["status"] == "paused":
                return loop
        return None

    def is_active(self) -> bool:
        """``True`` if any loop is active."""
        return any(lp["status"] == "active" for lp in self.list())

    def status_line(self) -> str:
        """Multi-line status listing all loops with UIDs and countdown."""
        loops = self.list()
        if not loops:
            return "No active loops. Set one with /loop [interval] <prompt>."
        now = time.time()
        lines: List[str] = []
        for loop in sorted(loops, key=lambda l: l.get("uid", "")):
            uid = loop["uid"]
            turns = f"{loop['turns_completed']}/{loop['max_turns']} turns"
            if loop["status"] == "active":
                remaining = (
                    int((loop["last_fired_at"] + loop["interval_seconds"]) - now)
                    if loop["last_fired_at"] > 0
                    else 0
                )
                countdown = self._format_countdown(remaining)
                interval_str = self._format_interval(loop["interval_seconds"])
                lines.append(
                    f"⊙ #{uid} (active, every {interval_str}, {countdown}, {turns}): {loop['prompt']}"
                )
            elif loop["status"] == "paused":
                interval_str = self._format_interval(loop["interval_seconds"])
                reason = f" — {loop['reason']}" if loop.get("reason") else ""
                lines.append(
                    f"⏸ #{uid} (paused, every {interval_str}, {turns}{reason}): {loop['prompt']}"
                )
        return "\n".join(lines)

    # ── Lifecycle ──────────────────────────────────────────────────

    def shutdown(self) -> None:
        """Clean shutdown — mark heap dirty so ticker picks up changes."""
        self.ticker.mark_heap_dirty()
