"""Periodic in-process write-path health logging for the gateway.

Follow-up from the 2026-11-10/11 artifact-loss incident (RCA §10, fix #1).
The external ``hermes-artifact-loss-watchdog`` CronJob (``*/15m``) already
detects a stalled write path from *outside* the process, but by the time it
fires the stall has been invisible in the gateway's own logs/metrics for up
to 15 minutes (in the actual incident: 9+ hours, since the external watchdog
hadn't shipped yet). This module closes that gap by emitting a single
grep-friendly ``[WRITEPATH]`` line every N seconds (default 60) with:

  * A live write-lock probe against ``state.db``: open a short-lived
    connection with a known ``busy_timeout``, then ``BEGIN IMMEDIATE;
    COMMIT;``. This is deliberately a FRESH connection rather than reaching
    into the gateway's internal ``AsyncSessionDB``/``SessionDB`` connection
    pool — it exercises the write path the same way any other hermes
    process (CLI, cron, worktree agent) would, and avoids coupling this
    watchdog to internals that are refactored independently of this file.
  * Seconds since the cron-output writer's last successful write (tracked
    via :func:`record_cron_output_write`, called from
    ``cron/jobs.py::mark_job_run`` after every job run) — this is the exact
    write path that silently stalled for 9+ hours in the incident before
    any external signal caught it.

Design notes (parity with ``gateway/memory_monitor.py``, the existing
precedent for a grep-friendly periodic gateway.log line from a daemon
thread):
  * Grep-friendly single-line format beginning ``[WRITEPATH]``.
  * Baseline snapshot logged immediately on start; final snapshot on stop.
  * Daemon thread — never blocks process exit.
  * Entirely best-effort: any internal failure is caught and logged at
    DEBUG so this watchdog can never itself take down the gateway.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_monitor_thread: Optional[threading.Thread] = None
_stop_event: Optional[threading.Event] = None
_start_time: Optional[float] = None
_interval_seconds: float = 60.0
_lock = threading.Lock()
_db_path: Optional[Path] = None

# Probe-connection busy_timeout. Matches the short-timeout + app-level-retry
# pattern ``hermes_state.SessionDB`` uses for its real writer connection
# (see ``_WRITE_RETRY_*`` tuning there) — a probe that waited SQLite's
# default 30s busy handler would itself look like a stall.
_PROBE_BUSY_TIMEOUT_MS = 2000

# --- cron-output write tracking ------------------------------------------
_cron_write_lock = threading.Lock()
_last_cron_write_ts: Optional[float] = None
_last_cron_write_ok: Optional[bool] = None
_last_cron_write_job_id: Optional[str] = None


def record_cron_output_write(job_id: str, success: bool) -> None:
    """Record that the cron-output writer just attempted a write.

    Called from ``cron/jobs.py::mark_job_run`` immediately after
    ``save_jobs()`` returns — i.e. right at the point the 2026-11-10/11
    incident's write path silently stalled for 9+ hours before any
    external signal caught it.
    """
    global _last_cron_write_ts, _last_cron_write_ok, _last_cron_write_job_id
    with _cron_write_lock:
        _last_cron_write_ts = time.monotonic()
        _last_cron_write_ok = bool(success)
        _last_cron_write_job_id = job_id


def _cron_write_age_s() -> Optional[float]:
    with _cron_write_lock:
        if _last_cron_write_ts is None:
            return None
        return time.monotonic() - _last_cron_write_ts


def _sqlite_probe(db_path: Optional[Path]) -> str:
    """Open a short-lived connection to ``state.db`` and probe the write lock.

    Returns a short human-readable status string; never raises.
    """
    if db_path is None:
        return "no_db_path"
    if not db_path.exists():
        return f"db_missing:{db_path}"

    probe_start = time.monotonic()
    try:
        conn = sqlite3.connect(
            str(db_path), timeout=_PROBE_BUSY_TIMEOUT_MS / 1000.0, isolation_level=None,
        )
    except Exception as e:
        return f"connect_failed:{e}"

    try:
        conn.execute(f"PRAGMA busy_timeout={_PROBE_BUSY_TIMEOUT_MS}")
        # BEGIN IMMEDIATE acquires the write lock immediately (vs. a deferred
        # BEGIN, which waits until the first write statement) — this is the
        # exact signal we want: is *something* holding state.db's write lock
        # right now, for longer than a healthy writer ever should.
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("COMMIT")
        probe_ms = (time.monotonic() - probe_start) * 1000
        return f"busy_timeout={_PROBE_BUSY_TIMEOUT_MS}ms write_probe=ok({probe_ms:.1f}ms)"
    except Exception as e:
        probe_ms = (time.monotonic() - probe_start) * 1000
        return (
            f"busy_timeout={_PROBE_BUSY_TIMEOUT_MS}ms "
            f"write_probe=FAILED({probe_ms:.1f}ms): {e}"
        )
    finally:
        try:
            conn.close()
        except Exception:
            pass


def log_writepath_status(prefix: str = "") -> None:
    """Emit one ``[WRITEPATH] ...`` line. Safe to call on-demand."""
    tag = f"{prefix} " if prefix else ""
    sqlite_status = _sqlite_probe(_db_path)
    age = _cron_write_age_s()
    age_str = f"{age:.0f}s" if age is not None else "never"
    with _cron_write_lock:
        last_ok = _last_cron_write_ok
        last_job = _last_cron_write_job_id
    logger.info(
        "[WRITEPATH] %ssqlite(%s) cron_last_write_age=%s cron_last_write_ok=%s cron_last_job=%s",
        tag, sqlite_status, age_str, last_ok, last_job,
    )


def _monitor_loop(stop_event: threading.Event, interval: float) -> None:
    while not stop_event.wait(interval):
        try:
            log_writepath_status()
        except Exception as e:
            logger.debug("Write-path watchdog iteration failed: %s", e)


def start_writepath_monitoring(db_path, interval_seconds: float = 60.0) -> bool:
    """Start periodic write-path health logging in a daemon thread.

    Parameters
    ----------
    db_path
        Path to the ``state.db`` file to probe (e.g.
        ``get_hermes_home() / "state.db"``). May be ``None`` — the monitor
        still runs and reports cron-write staleness, just without the
        SQLite probe.
    interval_seconds
        How often to log. Default 60s (1 minute).
    """
    global _monitor_thread, _stop_event, _start_time, _interval_seconds, _db_path

    with _lock:
        if _monitor_thread is not None and _monitor_thread.is_alive():
            return False

        _db_path = Path(db_path) if db_path is not None else None
        _start_time = time.monotonic()
        _interval_seconds = float(interval_seconds)
        _stop_event = threading.Event()

        log_writepath_status(prefix="baseline")

        _monitor_thread = threading.Thread(
            target=_monitor_loop,
            args=(_stop_event, _interval_seconds),
            name="gateway-writepath-watchdog",
            daemon=True,
        )
        _monitor_thread.start()

        logger.info(
            "[WRITEPATH] Periodic write-path watchdog started (interval: %ds)",
            int(_interval_seconds),
        )
        return True


def stop_writepath_monitoring(timeout: float = 2.0) -> None:
    """Stop the monitor thread and log a final snapshot."""
    global _monitor_thread, _stop_event

    with _lock:
        if _stop_event is None or _monitor_thread is None:
            return
        try:
            log_writepath_status(prefix="shutdown")
        except Exception:
            pass
        _stop_event.set()
        thread = _monitor_thread
        _monitor_thread = None
        _stop_event = None

    try:
        thread.join(timeout=timeout)
    except Exception:
        pass

    logger.info("[WRITEPATH] Periodic write-path watchdog stopped")


def is_running() -> bool:
    with _lock:
        return _monitor_thread is not None and _monitor_thread.is_alive()
