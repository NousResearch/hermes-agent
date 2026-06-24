"""
Session summary scheduler (issue #45103).

A lightweight in-process scheduler that periodically scans the sessions
table for sessions that need (or should refresh) their cached AI
summary, and runs ``SessionDB.summarize_session()`` on them in a
bounded thread pool.

Design:

  - Three trigger classes, all pull-based (no hooks required):
      1. CLOSED       — session was ended and has no summary yet
      2. IDLE         — session is idle for N minutes since last
                        summary (default 30) — likely a new
                        meaningful turn came in
      3. STALE        — session has been active for a long time
                        since last summary (default 1h) — even if
                        not idle, the cached text is probably wrong

  - Configurable via config.yaml under ``desktop.session_summarization``
    (or wherever the project surfaces it):

        desktop:
          session_summarization:
            enabled: true
            max_concurrent: 2
            min_messages: 6             # skip trivially short sessions
            idle_minutes: 30
            stale_minutes: 60
            min_age_seconds: 10         # don't summarize a brand-new
                                         # session the moment it lands
            startup_backfill: true      # on first tick, queue every
                                         # session that lacks a summary

  - ``tick()`` is the entry point. The gateway (or a CLI daemon)
    calls it every ~60 s, the same cadence as ``cron.scheduler.tick()``.

  - Concurrency: bounded ThreadPoolExecutor (default 2 workers). Each
    worker is short-lived (one session per call) so a slow LLM
    call doesn't block the whole pool.

  - Cancellation: the executor is shut down on process exit (atexit).
    The workers respect ``summarize_session()``'s own timeout, so
    worst case is a 30s hang per stuck call.

  - Idempotency: ``summarize_session()`` has its own 5-min
    idempotency check, so re-scheduling the same session within
    5 min is a free no-op. We don't try to deduplicate harder
    than that — over-summarizing slightly is fine, under-summarizing
    is not.

Why pull, not push:

  - ``append_message`` and ``end_session`` are on the hot path
    (every chat turn). A push hook would add latency to every
    turn for a side-effect most turns don't need.
  - The 60s tick cadence is already proven for cron jobs; reusing
    it means we get "free" lifecycle integration with whatever
    runs the gateway.
  - If a session misses a tick, the next tick picks it up. No
    state to recover on restart, no event log to replay.

References:
  - cron/scheduler.py — same tick() pattern, same pool sizing
  - agent/context_compressor.py — LLM-call error handling template
  - hermes_state.SummarizeSession() — the function this scheduler
    wraps; do not duplicate any of its logic
"""

from __future__ import annotations

import atexit
import concurrent.futures
import logging
import os
import threading
import time
from typing import List, Optional

from hermes_constants import get_hermes_home
from hermes_state import SessionDB

logger = logging.getLogger(__name__)


# Default knobs. Override per-instance via __init__ kwargs or, in the
# future, via config.yaml under desktop.session_summarization (the
# actual config wiring is left to the gateway / CLI bootstrap — the
# scheduler is the dumb runtime that consumes already-resolved values).
DEFAULT_MAX_CONCURRENT = 2
DEFAULT_MIN_MESSAGES = 6
DEFAULT_IDLE_MINUTES = 30
DEFAULT_STALE_MINUTES = 60
DEFAULT_MIN_AGE_SECONDS = 10
DEFAULT_STARTUP_BACKFILL = True


class SessionSummaryScheduler:
    """Background summarization scheduler.

    Construct one per process, call ``tick()`` periodically (every 60 s
    is the recommended cadence — matches cron). The scheduler is
    thread-safe; concurrent ``tick()`` calls are coalesced via a
    single in-flight lock so a slow tick doesn't pile up.
    """

    def __init__(
        self,
        state_db: SessionDB,
        *,
        max_concurrent: int = DEFAULT_MAX_CONCURRENT,
        min_messages: int = DEFAULT_MIN_MESSAGES,
        idle_minutes: int = DEFAULT_IDLE_MINUTES,
        stale_minutes: int = DEFAULT_STALE_MINUTES,
        min_age_seconds: int = DEFAULT_MIN_AGE_SECONDS,
        startup_backfill: bool = DEFAULT_STARTUP_BACKFILL,
    ) -> None:
        self._state = state_db
        self._max_concurrent = max(1, int(max_concurrent))
        self._min_messages = max(1, int(min_messages))
        self._idle_minutes = max(0, int(idle_minutes))
        self._stale_minutes = max(0, int(stale_minutes))
        self._min_age_seconds = max(0, int(min_age_seconds))
        self._startup_backfill = bool(startup_backfill)

        # Bounded pool. We use one executor for the lifetime of the
        # process — spinning up a new pool per tick would defeat the
        # purpose and burn startup latency.
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._max_concurrent,
            thread_name_prefix="session-summary",
        )
        # Coalesce concurrent ticks: if a tick is still running, the
        # next call returns immediately. This is what cron/scheduler
        # does too (via file lock) — we just do it in-process.
        self._tick_lock = threading.Lock()
        self._atexit_registered = False
        # Idempotency: only run the startup backfill once per process.
        self._startup_backfill_done = False

    # ── lifecycle ──────────────────────────────────────────────

    def shutdown(self, wait: bool = True) -> None:
        """Stop the worker pool. Safe to call multiple times."""
        try:
            self._pool.shutdown(wait=wait, cancel_futures=not wait)
        except Exception:  # pragma: no cover — defensive
            logger.debug("scheduler pool shutdown error", exc_info=True)

    def _register_atexit(self) -> None:
        if self._atexit_registered:
            return
        atexit.register(self.shutdown, wait=False)
        self._atexit_registered = True

    # ── scheduling logic ───────────────────────────────────────

    def _find_candidates(self) -> List[str]:
        """Return session ids that should be (re-)summarized this tick.

        The query is intentionally a single SQL with OR-of-ANDs to
        avoid N+1 round-trips. All three trigger classes run in the
        same statement; Python just filters the result.

        Note: ``sessions.last_active`` is a correlated subquery
        computed in ``list_sessions_rich`` (line ~2143) — not a real
        column. We can't reference it from this query, so we use
        ``ended_at`` (set on session close) and ``summary_updated_at``
        (set after summarization) as our two real timestamps.

        The three trigger classes:
          1. CLOSED — ended_at IS NOT NULL AND summary IS NULL
                        (session was closed but never summarized)
          2. STALE  — summary IS NOT NULL AND
                        (ended_at IS NULL OR ended_at > summary_updated_at)
                        AND (now - summary_updated_at) > stale_seconds
                        (active session that's been "evolving" past
                        its cached summary for too long)
          3. IDLE   — ended_at IS NOT NULL AND ended_at > summary_updated_at
                        AND (now - summary_updated_at) > idle_seconds
                        (closed session whose user has been coming
                        back; the cached summary is now stale)
        """
        now = time.time()
        idle_cutoff = now - self._idle_minutes * 60
        stale_cutoff = now - self._stale_minutes * 60
        age_cutoff = now - self._min_age_seconds

        sql = """
            SELECT id,
                   started_at,
                   ended_at,
                   message_count,
                   summary,
                   summary_updated_at,
                   CASE
                     WHEN summary IS NULL
                       AND ended_at IS NOT NULL
                       AND ended_at <= ?
                       AND archived = 0
                       AND message_count >= ?
                       THEN 'closed'
                     WHEN summary IS NOT NULL
                       AND summary_updated_at < ?
                       AND ended_at IS NULL
                       AND archived = 0
                       THEN 'stale'
                     WHEN summary IS NOT NULL
                       AND summary_updated_at < ?
                       AND ended_at IS NOT NULL
                       AND ended_at > summary_updated_at
                       AND archived = 0
                       THEN 'idle'
                     ELSE NULL
                   END AS trigger_class
            FROM sessions
            WHERE archived = 0
        """
        params = (
            age_cutoff,
            self._min_messages,
            stale_cutoff,
            idle_cutoff,
        )
        with self._state._lock:
            cursor = self._state._conn.execute(sql, params)
            rows = cursor.fetchall()

        # Optional: skip startup backfill after the first tick.
        if not self._startup_backfill and not self._startup_backfill_done:
            # We only want truly "live" triggers (idle/stale), not
            # the one-shot backfill. Filter out trigger_class='closed'
            # rows on the very first tick.
            rows = [r for r in rows if r["trigger_class"] in ("idle", "stale")]
            self._startup_backfill_done = True

        # Mark startup backfill done after the first successful scan.
        if not self._startup_backfill_done:
            self._startup_backfill_done = True

        # Drop rows that don't actually trigger — CASE returns NULL
        # for everything else.
        return [(r["id"], r["trigger_class"]) for r in rows if r["trigger_class"]]

    def tick(self) -> int:
        """Run one scheduling pass. Returns the number of jobs queued.

        Safe to call concurrently — a slow tick will not stack up
        additional calls. If called more often than the LLM can
        keep up, excess candidates are simply dropped from this
        tick (next tick will pick them up).
        """
        if not self._tick_lock.acquire(blocking=False):
            logger.debug("summary scheduler: previous tick still running, skipping")
            return 0
        try:
            return self._tick_locked()
        finally:
            self._tick_lock.release()

    def _tick_locked(self) -> int:
        candidates = self._find_candidates()
        if not candidates:
            return 0
        queued = 0
        for session_id, trigger in candidates:
            try:
                self._pool.submit(self._summarize_one, session_id, trigger)
                queued += 1
            except RuntimeError:
                # Pool was shut down (e.g. during process exit).
                # Stop submitting; the rest of the candidates will
                # be picked up next tick.
                logger.debug(
                    "summary scheduler: pool closed, stopped after %d jobs", queued
                )
                break
        if queued:
            logger.info(
                "summary scheduler: queued %d session(s) for summarization "
                "(%s)",
                queued,
                ", ".join(sorted({t for _, t in candidates[:queued]})),
            )
        return queued

    def _summarize_one(self, session_id: str, trigger: str) -> None:
        """Worker body — runs on a pool thread.

        All errors are caught and logged; never re-raised. A failing
        session must not be able to crash the scheduler.
        """
        try:
            t0 = time.time()
            result = self._state.summarize_session(session_id)
            dt = time.time() - t0
            if result:
                logger.info(
                    "summary ok: session=%s trigger=%s elapsed=%.2fs",
                    session_id,
                    trigger,
                    dt,
                )
            else:
                # summarize_session() returned None — either no
                # provider, no messages, or LLM call failed. All
                # logged at WARNING inside the function. We just
                # note the trigger class for diagnostics.
                logger.info(
                    "summary skipped: session=%s trigger=%s elapsed=%.2fs",
                    session_id,
                    trigger,
                    dt,
                )
        except Exception:  # pragma: no cover — defensive belt
            logger.exception(
                "summary scheduler: unhandled error for session=%s trigger=%s",
                session_id,
                trigger,
            )


# ── singleton accessor (matches cron.scheduler.get_* style) ──────
_default_scheduler: Optional[SessionSummaryScheduler] = None
_default_scheduler_lock = threading.Lock()


def get_default_scheduler() -> SessionSummaryScheduler:
    """Return the process-wide scheduler, constructing it on first use.

    Reads the default SessionDB from HERMES_HOME. Wire a custom
    scheduler via ``set_default_scheduler()`` for tests.
    """
    global _default_scheduler
    if _default_scheduler is not None:
        return _default_scheduler
    with _default_scheduler_lock:
        if _default_scheduler is None:
            hermes_home = get_hermes_home()
            db_path = hermes_home / "state.db"
            state = SessionDB(db_path=db_path)
            _default_scheduler = SessionSummaryScheduler(state)
    return _default_scheduler


def set_default_scheduler(scheduler: Optional[SessionSummaryScheduler]) -> None:
    """Replace the default scheduler (used by tests; pass None to clear)."""
    global _default_scheduler
    with _default_scheduler_lock:
        if _default_scheduler is not None and _default_scheduler is not scheduler:
            _default_scheduler.shutdown(wait=False)
        _default_scheduler = scheduler
