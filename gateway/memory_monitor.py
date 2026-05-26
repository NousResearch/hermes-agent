"""Periodic process memory/resource usage logging for the gateway.

Ported from cline/cline#10343 (src/standalone/memory-monitor.ts).

The gateway is a long-lived process that accumulates memory as it caches
agent instances, session transcripts, tool schemas, memory providers, MCP
connections, etc.  A slow leak in any of those subsystems is invisible
in a single log line — you only see it by watching RSS climb over hours.

This module emits a single structured ``[MEMORY] ...`` line every N
minutes (default 5) so maintainers investigating a suspected leak can
grep ``agent.log`` / ``gateway.log`` for a time series of RSS, open file
descriptors, and Python GC stats.  The timer runs in a background thread
and shuts down cleanly with the gateway.

Design notes (parity with the Cline port):
  * Grep-friendly single-line format beginning ``[MEMORY]``.
  * Final snapshot logged on shutdown so "last RSS before exit" is
    always in the log.
  * Baseline snapshot logged immediately on start.
  * Daemon thread — never blocks process exit.
  * Uses ``psutil`` for current RSS when available and falls back to
    ``resource``. Both are optional; when neither works we emit a single
    WARNING and disable the monitor rather than crashing the gateway.
  * Triggers active idle-client cleanup on fd or RSS pressure; the log line is
    telemetry, not the remediation mechanism.

Config: ``logging.memory_monitor`` in ``config.yaml`` — see
``hermes_cli/config.py`` for the defaults block.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import threading
import time
from typing import Callable, Optional, Tuple

logger = logging.getLogger(__name__)

_BYTES_TO_MB = 1024 * 1024
_FD_WARN_RATIO = 0.70
_FD_CRITICAL_RATIO = 0.90
_RSS_HIGH_MB_DEFAULT = 1024
_RSS_CRITICAL_MB_DEFAULT = 1536

_monitor_thread: Optional[threading.Thread] = None
_stop_event: Optional[threading.Event] = None
_start_time: Optional[float] = None
_interval_seconds: float = 300.0  # 5 minutes
_rss_high_mb: Optional[int] = _RSS_HIGH_MB_DEFAULT
_rss_critical_mb: Optional[int] = _RSS_CRITICAL_MB_DEFAULT
_lock = threading.Lock()
_pressure_handler_lock = threading.Lock()
_pressure_handler: Optional[Callable[[str, int, Optional[int], Optional[float]], Optional[str]]] = None


def _get_rss_mb() -> Optional[int]:
    """Return current process resident set size in MB, or None if unavailable.

    Tries ``psutil`` first for current RSS, then falls back to
    ``resource.getrusage``. On Linux/macOS ``ru_maxrss`` is a high-water mark;
    useful for logging, but too pessimistic for release-after-cleanup pressure
    decisions.
    """
    try:
        import psutil  # type: ignore

        rss = psutil.Process(os.getpid()).memory_info().rss
        return int(rss / _BYTES_TO_MB)
    except Exception:
        pass

    # Linux / macOS — resource is stdlib. On Linux ru_maxrss is in KB,
    # on macOS it is in bytes (yes, really).
    try:
        import resource

        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return int(maxrss / _BYTES_TO_MB)
        # Linux / other unices: KB
        return int(maxrss / 1024)
    except Exception:
        return None


def _get_open_fd_count() -> Optional[int]:
    """Return current open file descriptor count, or None if unavailable."""
    try:
        import psutil  # type: ignore

        return int(psutil.Process(os.getpid()).num_fds())
    except Exception:
        pass

    for fd_dir in ("/proc/self/fd", "/dev/fd"):
        try:
            return len(os.listdir(fd_dir))
        except Exception:
            continue
    return None


def _get_fd_soft_limit() -> Optional[int]:
    """Return RLIMIT_NOFILE soft limit, or None when not meaningful."""
    try:
        import resource

        soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft == getattr(resource, "RLIM_INFINITY", -1):
            return None
        return int(soft)
    except Exception:
        return None


def _get_fd_usage() -> Optional[Tuple[int, Optional[int], Optional[float]]]:
    """Return (open_fds, soft_limit, ratio), or None if count is unavailable."""
    open_fds = _get_open_fd_count()
    if open_fds is None:
        return None
    soft_limit = _get_fd_soft_limit()
    if soft_limit is None or soft_limit <= 0:
        return (open_fds, soft_limit, None)
    return (open_fds, soft_limit, open_fds / soft_limit)


def _format_fd_usage(fd_usage: Optional[Tuple[int, Optional[int], Optional[float]]]) -> str:
    if fd_usage is None:
        return "fds=unavailable"
    open_fds, soft_limit, ratio = fd_usage
    if soft_limit is None or ratio is None:
        return f"fds={open_fds}/unlimited"
    return f"fds={open_fds}/{soft_limit} ({ratio:.1%})"


def set_resource_pressure_handler(
    handler: Optional[Callable[[str, int, Optional[int], Optional[float]], Optional[str]]],
) -> None:
    """Install a callback that actively releases resources under pressure."""
    global _pressure_handler
    _pressure_handler = handler


def _coerce_threshold_mb(value: Optional[int], default: Optional[int]) -> Optional[int]:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else None


def _set_rss_pressure_thresholds(
    high_mb: Optional[int],
    critical_mb: Optional[int],
) -> None:
    global _rss_high_mb, _rss_critical_mb
    high = _coerce_threshold_mb(high_mb, _RSS_HIGH_MB_DEFAULT)
    critical = _coerce_threshold_mb(critical_mb, _RSS_CRITICAL_MB_DEFAULT)
    if high is not None and critical is not None and critical < high:
        critical = high
    _rss_high_mb = high
    _rss_critical_mb = critical


def _run_pressure_handler(
    prefix: str,
    level: str,
    open_fds: int,
    soft_limit: Optional[int],
    ratio: Optional[float],
) -> None:
    handler = _pressure_handler
    if handler is None:
        logger.warning("[%s] no resource-pressure handler registered; cleanup skipped", prefix)
        return
    if not _pressure_handler_lock.acquire(blocking=False):
        logger.info("[%s] resource-pressure cleanup already running; skipping duplicate trigger", prefix)
        return
    try:
        detail = handler(level, open_fds, soft_limit, ratio)
        logger.warning(
            "[%s] resource-pressure cleanup completed%s",
            prefix,
            f": {detail}" if detail else "",
        )
    except Exception as exc:
        logger.exception("[%s] resource-pressure cleanup failed: %s", prefix, exc)
    finally:
        _pressure_handler_lock.release()


def _handle_fd_pressure(fd_usage: Optional[Tuple[int, Optional[int], Optional[float]]]) -> None:
    if fd_usage is None:
        return
    open_fds, soft_limit, ratio = fd_usage
    if soft_limit is None or ratio is None:
        return
    if ratio >= _FD_CRITICAL_RATIO:
        level = "critical"
        logger.error(
            "[FD] open file descriptor usage critical: %d/%d (%.1f%%); "
            "running resource-pressure cleanup before EMFILE",
            open_fds,
            soft_limit,
            ratio * 100,
        )
    elif ratio >= _FD_WARN_RATIO:
        level = "high"
        logger.warning(
            "[FD] open file descriptor usage high: %d/%d (%.1f%%); "
            "running resource-pressure cleanup before EMFILE",
            open_fds,
            soft_limit,
            ratio * 100,
        )
    else:
        return

    _run_pressure_handler("FD", level, open_fds, soft_limit, ratio)


def _handle_rss_pressure(
    rss_mb: Optional[int],
    fd_usage: Optional[Tuple[int, Optional[int], Optional[float]]],
) -> None:
    if rss_mb is None:
        return
    if _rss_critical_mb is not None and rss_mb >= _rss_critical_mb:
        level = "memory_critical"
        logger.error(
            "[MEMORY] rss usage critical: %dMB >= %dMB; "
            "running resource-pressure cleanup before provider/runtime pressure escalates",
            rss_mb,
            _rss_critical_mb,
        )
    elif _rss_high_mb is not None and rss_mb >= _rss_high_mb:
        level = "memory_high"
        logger.warning(
            "[MEMORY] rss usage high: %dMB >= %dMB; "
            "running resource-pressure cleanup before provider/runtime pressure escalates",
            rss_mb,
            _rss_high_mb,
        )
    else:
        return

    open_fds, soft_limit, ratio = fd_usage if fd_usage is not None else (0, None, None)
    _run_pressure_handler("MEMORY", level, open_fds, soft_limit, ratio)


def log_memory_usage(prefix: str = "") -> None:
    """Log current memory usage in a grep-friendly ``[MEMORY] ...`` line.

    Safe to call on-demand from any thread at important lifecycle
    moments (after shutdown, after context compression, etc.).

    Parameters
    ----------
    prefix
        Optional extra tag inserted after ``[MEMORY]`` — e.g.
        ``"baseline"``, ``"shutdown"``.
    """
    rss = _get_rss_mb()
    fd_usage = _get_fd_usage()
    fd_text = _format_fd_usage(fd_usage)
    _handle_fd_pressure(fd_usage)
    _handle_rss_pressure(rss, fd_usage)
    uptime = int(time.monotonic() - _start_time) if _start_time else 0
    # gc.get_stats() returns per-generation collection counts; the sum
    # is a cheap proxy for "how much garbage have we created".
    try:
        gc_counts = gc.get_count()  # (gen0, gen1, gen2)
    except Exception:
        gc_counts = (0, 0, 0)
    # Thread count is a handy correlate when diagnosing thread leaks.
    try:
        thread_count = threading.active_count()
    except Exception:
        thread_count = 0

    tag = f"{prefix} " if prefix else ""
    if rss is None:
        logger.info(
            "[MEMORY] %srss=unavailable %s gc=%s threads=%d uptime=%ds",
            tag,
            fd_text,
            gc_counts,
            thread_count,
            uptime,
        )
    else:
        logger.info(
            "[MEMORY] %srss=%dMB %s gc=%s threads=%d uptime=%ds",
            tag,
            rss,
            fd_text,
            gc_counts,
            thread_count,
            uptime,
        )


def _monitor_loop(stop_event: threading.Event, interval: float) -> None:
    """Background thread body — log every ``interval`` seconds until stopped."""
    while not stop_event.wait(interval):
        try:
            log_memory_usage()
        except Exception as e:
            # Never let the monitor crash the gateway; just log and carry on.
            logger.debug("Memory monitor iteration failed: %s", e)


def start_memory_monitoring(
    interval_seconds: float = 300.0,
    pressure_handler: Optional[Callable[[str, int, Optional[int], Optional[float]], Optional[str]]] = None,
    rss_high_mb: Optional[int] = None,
    rss_critical_mb: Optional[int] = None,
) -> bool:
    """Start periodic memory usage logging in a daemon thread.

    Logs immediately to capture a baseline, then every ``interval_seconds``.
    Safe to call multiple times — subsequent calls are no-ops while the
    first monitor is still running.

    Parameters
    ----------
    interval_seconds
        How often to log.  Default 300s (5 minutes), matching the
        upstream cline/cline implementation.

    Returns
    -------
    bool
        True if a fresh monitor thread was started, False if one was
        already running or if memory introspection isn't available.
    """
    global _monitor_thread, _stop_event, _start_time, _interval_seconds
    if pressure_handler is not None:
        set_resource_pressure_handler(pressure_handler)
    _set_rss_pressure_thresholds(rss_high_mb, rss_critical_mb)

    with _lock:
        if _monitor_thread is not None and _monitor_thread.is_alive():
            return False

        # Sanity-check that we can read RSS at all.  If neither resource
        # nor psutil works, no point spinning a thread that can only log
        # "rss=unavailable" forever — warn once and bail.
        if _get_rss_mb() is None:
            logger.warning(
                "[MEMORY] Memory monitoring unavailable: neither resource.getrusage "
                "nor psutil could read process RSS — skipping periodic logging.",
            )
            return False

        _start_time = time.monotonic()
        _interval_seconds = float(interval_seconds)
        _stop_event = threading.Event()

        # Baseline snapshot before the loop starts.
        log_memory_usage(prefix="baseline")

        _monitor_thread = threading.Thread(
            target=_monitor_loop,
            args=(_stop_event, _interval_seconds),
            name="gateway-memory-monitor",
            daemon=True,
        )
        _monitor_thread.start()

        logger.info(
            "[MEMORY] Periodic memory monitoring started (interval: %ds)",
            int(_interval_seconds),
        )
        return True


def stop_memory_monitoring(timeout: float = 2.0) -> None:
    """Stop the monitor thread and log a final snapshot.

    Safe to call even if ``start_memory_monitoring()`` was never called.
    """
    global _monitor_thread, _stop_event

    with _lock:
        if _stop_event is None or _monitor_thread is None:
            set_resource_pressure_handler(None)
            return

        # Final snapshot before teardown so "last RSS" is always in the log.
        try:
            log_memory_usage(prefix="shutdown")
        except Exception:
            pass

        _stop_event.set()
        thread = _monitor_thread
        _monitor_thread = None
        _stop_event = None

    # Join outside the lock so a stuck log call can't deadlock shutdown.
    try:
        thread.join(timeout=timeout)
    except Exception:
        pass

    set_resource_pressure_handler(None)
    logger.info("[MEMORY] Periodic memory monitoring stopped")


def is_running() -> bool:
    """True if the background monitor thread is alive."""
    with _lock:
        return _monitor_thread is not None and _monitor_thread.is_alive()
