"""Periodic process memory usage logging for the gateway.

Emits one grep-friendly ``[MEMORY] ...`` line every N seconds (default 300)
from a daemon thread so slow leaks show up as an RSS time series in the logs.
A baseline snapshot is logged on start and a final one on stop.  Uses stdlib
``resource`` first, ``psutil`` as fallback (Windows); if neither works the
monitor warns once and stays disabled.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import os
import sys
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

_BYTES_TO_MB = 1024 * 1024

_monitor_thread: Optional[threading.Thread] = None
_stop_event: Optional[threading.Event] = None
_start_time: Optional[float] = None
_lock = threading.Lock()


def _get_rss_mb() -> Optional[int]:
    """Return process RSS high-water mark in MB, or None if unavailable."""
    try:
        import resource

        # ru_maxrss is KB on Linux but bytes on macOS.
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (_BYTES_TO_MB if sys.platform == "darwin" else 1024))
    except Exception:
        pass
    try:
        import psutil  # type: ignore

        return int(psutil.Process(os.getpid()).memory_info().rss / _BYTES_TO_MB)
    except Exception:
        return None


def log_memory_usage(prefix: str = "") -> None:
    """Log ``[MEMORY] [<prefix> ]rss=... gc=... threads=... uptime=...``; safe from any thread."""
    rss = _get_rss_mb()
    logger.info(
        "[MEMORY] %srss=%s gc=%s threads=%d uptime=%ds", f"{prefix} " if prefix else "",
        "unavailable" if rss is None else f"{rss}MB", gc.get_count(), threading.active_count(),
        int(time.monotonic() - _start_time) if _start_time else 0,
    )


def _monitor_loop(stop_event: threading.Event, interval: float) -> None:
    """Background thread body — log every ``interval`` seconds until stopped."""
    while not stop_event.wait(interval):
        try:
            log_memory_usage()
        except Exception as e:  # never let the monitor crash the gateway
            logger.debug("Memory monitor iteration failed: %s", e)


def start_memory_monitoring(interval_seconds: float = 300.0) -> bool:
    """Start periodic logging in a daemon thread (baseline logged immediately).  False if
    already running or RSS introspection is unavailable (warned once)."""
    global _monitor_thread, _stop_event, _start_time

    with _lock:
        if _monitor_thread is not None and _monitor_thread.is_alive():
            return False
        if _get_rss_mb() is None:
            logger.warning(
                "[MEMORY] Memory monitoring unavailable: neither resource.getrusage nor psutil could read process RSS "
                "— skipping periodic logging.",
            )
            return False
        _start_time = time.monotonic()
        _stop_event = threading.Event()
        log_memory_usage(prefix="baseline")
        _monitor_thread = threading.Thread(
            target=_monitor_loop, args=(_stop_event, float(interval_seconds)), name="gateway-memory-monitor", daemon=True
        )
        _monitor_thread.start()
        logger.info("[MEMORY] Periodic memory monitoring started (interval: %ds)", int(interval_seconds))
        return True


def stop_memory_monitoring(timeout: float = 2.0) -> None:
    """Stop the monitor thread and log a final ``shutdown`` snapshot. No-op if never started."""
    global _monitor_thread, _stop_event

    with _lock:
        if _stop_event is None or _monitor_thread is None:
            return
        with contextlib.suppress(Exception):
            log_memory_usage(prefix="shutdown")
        _stop_event.set()
        thread, _monitor_thread, _stop_event = _monitor_thread, None, None
    with contextlib.suppress(Exception):  # join outside the lock so a stuck log call can't deadlock the stop path
        thread.join(timeout=timeout)
    logger.info("[MEMORY] Periodic memory monitoring stopped")


def is_running() -> bool:
    """True if the background monitor thread is alive."""
    with _lock:
        return _monitor_thread is not None and _monitor_thread.is_alive()
