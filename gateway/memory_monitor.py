"""Periodic process memory usage logging for the gateway.

Ported from cline/cline#10343 (src/standalone/memory-monitor.ts).

The gateway is a long-lived process that accumulates memory as it caches
agent instances, session transcripts, tool schemas, memory providers, MCP
connections, etc.  A slow leak in any of those subsystems is invisible
in a single log line — you only see it by watching RSS climb over hours.

This module emits a single structured ``[MEMORY] ...`` line every N
minutes (default 5) so maintainers investigating a suspected leak can
grep ``agent.log`` / ``gateway.log`` for a time series of RSS + Python
GC stats.  The timer runs in a background thread and shuts down cleanly
with the gateway.

Design notes (parity with the Cline port):
  * Grep-friendly single-line format beginning ``[MEMORY]``.
  * Final snapshot logged on shutdown so "last RSS before exit" is
    always in the log.
  * Baseline snapshot logged immediately on start.
  * Daemon thread — never blocks process exit.
  * Uses ``resource`` (stdlib, Linux/macOS) first and falls back to
    ``psutil`` when ``resource`` isn't available (Windows).  Both are
    optional; when neither works we emit a single WARNING and disable
    the monitor rather than crashing the gateway.

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
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

_BYTES_TO_MB = 1024 * 1024

# ── Memory Pressure Levels ─────────────────────────────────────

class MemoryPressureLevel(Enum):
    NORMAL = auto()       # < 70% — no action
    ELEVATED = auto()     # 70-80% — GC only
    WARNING = auto()      # 80-90% — GC + evict idle caches
    CRITICAL = auto()     # 90-95% — aggressive eviction
    EMERGENCY = auto()    # > 95% — prepare restart

# ── Eviction Hook ──────────────────────────────────────────────

@dataclass
class _EvictionHook:
    """A registered callback that fires when memory pressure crosses thresholds."""
    callback: Callable[[MemoryPressureLevel, float], None]
    name: str
    min_interval: float = 30.0  # minimum seconds between fires
    last_fired: float = 0.0

# ── Module State ───────────────────────────────────────────────

_monitor_thread: Optional[threading.Thread] = None
_stop_event: Optional[threading.Event] = None
_start_time: Optional[float] = None
_interval_seconds: float = 300.0  # 5 minutes
_lock = threading.Lock()

# Phase 2 additions
_eviction_hooks: List[_EvictionHook] = []
_system_memory_mb: int = 0  # cached total system RAM
_high_watermark_mb: int = 0  # highest RSS ever seen
_high_watermark_pct: float = 0.0
_pressure_level: MemoryPressureLevel = MemoryPressureLevel.NORMAL
_pressure_entered_at: float = 0.0  # monotonic time when current pressure level was entered

# Configurable thresholds (as percentages of system RAM)
_WARNING_THRESHOLD: float = 80.0
_CRITICAL_THRESHOLD: float = 90.0
_EMERGENCY_THRESHOLD: float = 95.0
_ELEVATED_THRESHOLD: float = 70.0


def _get_system_memory_mb() -> int:
    """Return total system RAM in MB, or 0 if unknown.

    Caches result after first successful read. Cross-platform:
    uses ``os.sysconf`` (Linux), ``sysctl`` (macOS), or
    ``psutil`` (Windows / fallback).
    """
    global _system_memory_mb

    if _system_memory_mb > 0:
        return _system_memory_mb

    # Linux: os.sysconf is stdlib, no extra deps
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        _system_memory_mb = int(total / _BYTES_TO_MB)
        return _system_memory_mb
    except Exception:
        pass

    # macOS: sysctl via subprocess (no resource.hw_memsize in sysconf)
    try:
        import subprocess
        out = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], stderr=subprocess.DEVNULL
        )
        _system_memory_mb = int(int(out.strip()) / _BYTES_TO_MB)
        return _system_memory_mb
    except Exception:
        pass

    # Windows / fallback: psutil
    try:
        import psutil  # type: ignore
        total = psutil.virtual_memory().total
        _system_memory_mb = int(total / _BYTES_TO_MB)
        return _system_memory_mb
    except Exception:
        pass

    return 0


def _get_rss_pct() -> float:
    """Return RSS as a percentage of total system RAM, or 0.0 if unknown."""
    rss = _get_rss_mb()
    system = _get_system_memory_mb()
    if rss is None or system <= 0:
        return 0.0
    return (rss / system) * 100.0


def _compute_pressure_level(pct: float) -> MemoryPressureLevel:
    """Map a memory percentage to a pressure level."""
    if pct >= _EMERGENCY_THRESHOLD:
        return MemoryPressureLevel.EMERGENCY
    if pct >= _CRITICAL_THRESHOLD:
        return MemoryPressureLevel.CRITICAL
    if pct >= _WARNING_THRESHOLD:
        return MemoryPressureLevel.WARNING
    if pct >= _ELEVATED_THRESHOLD:
        return MemoryPressureLevel.ELEVATED
    return MemoryPressureLevel.NORMAL


def register_eviction_hook(
    callback: Callable[[MemoryPressureLevel, float], None],
    name: str,
    min_interval: float = 30.0,
) -> bool:
    """Register a callback that fires when memory pressure exceeds NORMAL.

    ``callback`` receives ``(level, rss_pct)`` when pressure thresholds are
    crossed.  ``min_interval`` prevents the same hook from being called more
    often than every N seconds (default 30s), limiting eviction churn.

    Returns True if the hook was registered; False if a hook with the same
    name already exists (idempotent).

    Usage in GatewayRunner::

        memory_monitor.register_eviction_hook(
            callback=lambda level, pct: self._evict_caches(level),
            name="gateway-runner",
        )
    """
    global _eviction_hooks

    with _lock:
        for hook in _eviction_hooks:
            if hook.name == name:
                return False
        _eviction_hooks.append(_EvictionHook(
            callback=callback,
            name=name,
            min_interval=min_interval,
        ))
        logger.debug(
            "[MEMORY] Registered eviction hook '%s' (min_interval=%.0fs)",
            name, min_interval,
        )
        return True


def unregister_eviction_hook(name: str) -> bool:
    """Remove a previously registered eviction hook. Returns True if removed."""
    global _eviction_hooks

    with _lock:
        for i, hook in enumerate(_eviction_hooks):
            if hook.name == name:
                _eviction_hooks.pop(i)
                return True
        return False


def get_memory_pressure() -> MemoryPressureLevel:
    """Return the current memory pressure level (thread-safe snapshot)."""
    with _lock:
        return _pressure_level


def get_high_watermark_mb() -> int:
    """Return the highest RSS (in MB) ever observed by this monitor."""
    with _lock:
        return _high_watermark_mb


def get_high_watermark_pct() -> float:
    """Return the highest RSS (% of system RAM) ever observed."""
    with _lock:
        return _high_watermark_pct


def configure_thresholds(
    warning: float = 80.0,
    critical: float = 90.0,
    emergency: float = 95.0,
    elevated: float = 70.0,
) -> None:
    """Override pressure thresholds (called from config-aware startup code).

    Defaults match the Phase 2 architecture spec.
    """
    global _WARNING_THRESHOLD, _CRITICAL_THRESHOLD, _EMERGENCY_THRESHOLD, _ELEVATED_THRESHOLD
    with _lock:
        _WARNING_THRESHOLD = warning
        _CRITICAL_THRESHOLD = critical
        _EMERGENCY_THRESHOLD = emergency
        _ELEVATED_THRESHOLD = elevated


def _get_rss_mb() -> Optional[int]:
    """Return current process resident set size in MB, or None if unavailable.

    Tries ``resource.getrusage`` first (Linux/macOS, no extra deps), then
    falls back to ``psutil`` which is an optional hermes-agent dep.
    """
    # Linux / macOS — resource is stdlib.  On Linux ru_maxrss is in KB,
    # on macOS it is in bytes (yes, really).  We use it as a cheap
    # "current" RSS — ru_maxrss reports the high-water mark for the
    # process, which is what you actually want for leak detection.
    try:
        import resource

        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return int(maxrss / _BYTES_TO_MB)
        # Linux / other unices: KB
        return int(maxrss / 1024)
    except Exception as _res_err:
        logger.debug("resource.getrusage failed, falling back to psutil: %s", _res_err)

    # Fallback: psutil (Windows, or unusual unix without resource).
    try:
        import psutil  # type: ignore

        rss = psutil.Process(os.getpid()).memory_info().rss
        return int(rss / _BYTES_TO_MB)
    except Exception as _psutil_err:
        return None


def log_memory_usage(prefix: str = "") -> None:
    """Log current memory usage in a grep-friendly ``[MEMORY] ...`` line.

    Safe to call on-demand from any thread at important lifecycle
    moments (after shutdown, after context compression, etc.).

    Parameters
    ----------
    prefix
        Optional extra tag inserted after ``[MEMORY]`` — e.g.
        ``"baseline"``, ``"shutdown"``.

    Side effects (Phase 2+)
        Updates high-watermark globals and recomputes the pressure level.
    """
    global _high_watermark_mb, _high_watermark_pct, _pressure_level, _pressure_entered_at

    rss = _get_rss_mb()
    rss_pct = _get_rss_pct()
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

    # High-watermark tracking
    if rss is not None and rss > _high_watermark_mb:
        _high_watermark_mb = rss
    if rss_pct > _high_watermark_pct:
        _high_watermark_pct = rss_pct

    # Recompute pressure level
    new_level = _compute_pressure_level(rss_pct) if rss_pct > 0 else MemoryPressureLevel.NORMAL
    if new_level != _pressure_level:
        old_level = _pressure_level
        _pressure_level = new_level
        _pressure_entered_at = time.monotonic()
        logger.warning(
            "[MEMORY] Pressure level changed: %s → %s (%.1f%% of %dMB system RAM)",
            old_level.name, new_level.name, rss_pct, _get_system_memory_mb(),
        )

    tag = f"{prefix} " if prefix else ""
    pressure_tag = f"level={_pressure_level.name} " if rss_pct > 0 else ""
    if rss is None:
        logger.info(
            "[MEMORY] %srss=unavailable gc=%s threads=%d uptime=%ds",
            tag,
            gc_counts,
            thread_count,
            uptime,
        )
    else:
        logger.info(
            "[MEMORY] %s%srss=%dMB(%.1f%%) gc=%s threads=%d uptime=%ds",
            tag,
            pressure_tag,
            rss,
            rss_pct,
            gc_counts,
            thread_count,
            uptime,
        )


def _fire_pressure_hooks(level: MemoryPressureLevel) -> None:
    """Call registered eviction hooks when pressure exceeds NORMAL.

    Rate-limited per-hook via ``min_interval`` to reduce churn.
    Never lets a hook exception escape to the caller.
    """
    now = time.monotonic()
    rss_pct = _get_rss_pct()

    with _lock:
        hooks = list(_eviction_hooks)  # copy for safe iteration

    for hook in hooks:
        if now - hook.last_fired < hook.min_interval:
            continue
        try:
            hook.callback(level, rss_pct)
            hook.last_fired = now
        except Exception as _hook_err:
            logger.warning(
                "[MEMORY] Eviction hook '%s' failed: %s",
                hook.name, _hook_err,
            )


def _check_and_evict(rss_mb: Optional[int] = None, rss_pct: float = 0.0) -> None:
    """Evaluate memory pressure and take action.

    Actions (ascending severity):
      NORMAL    — nothing
      ELEVATED  — run gc.collect()
      WARNING   — gc.collect() + fire eviction hooks
      CRITICAL  — gc.collect() + fire eviction hooks + LOG CRITICAL
      EMERGENCY — gc.collect() + fire hooks + LOG CRITICAL + alert

    Called from the monitor loop on each tick.
    """
    global _pressure_level

    if rss_pct < _ELEVATED_THRESHOLD and _pressure_level == MemoryPressureLevel.NORMAL:
        return

    level = _pressure_level
    if rss_mb is None:
        rss_mb = _get_rss_mb() or 0

    if level == MemoryPressureLevel.NORMAL:
        return

    # Always trigger GC at ELEVATED and above
    try:
        gc.collect()
    except Exception:
        pass

    if level == MemoryPressureLevel.ELEVATED:
        logger.debug(
            "[MEMORY] ELEVATED pressure — gc.collect() performed "
            "(rss=%dMB, %.1f%%)", rss_mb, rss_pct,
        )
        return

    # WARNING and above: fire eviction hooks
    _fire_pressure_hooks(level)

    if level == MemoryPressureLevel.WARNING:
        logger.warning(
            "[MEMORY] WARNING pressure — eviction hooks fired "
            "(rss=%dMB, %.1f%%)", rss_mb, rss_pct,
        )
    elif level == MemoryPressureLevel.CRITICAL:
        logger.critical(
            "[MEMORY] CRITICAL pressure — aggressive eviction completed "
            "(rss=%dMB, %.1f%%). Consider restarting the gateway if "
            "memory does not drop.", rss_mb, rss_pct,
        )
    elif level == MemoryPressureLevel.EMERGENCY:
        logger.critical(
            "[MEMORY] EMERGENCY pressure — gateway is at risk of OOM "
            "(rss=%dMB, %.1f%%)! Eviction hooks fired. "
            "If memory does not drop within 60s, a managed restart may be "
            "required.", rss_mb, rss_pct,
        )


def _monitor_loop(stop_event: threading.Event, interval: float) -> None:
    """Background thread body — log every ``interval`` seconds until stopped."""
    while not stop_event.wait(interval):
        try:
            log_memory_usage()
            _check_and_evict()
        except Exception as e:
            # Never let the monitor crash the gateway; just log and carry on.
            logger.debug("Memory monitor iteration failed: %s", e)


def start_memory_monitoring(interval_seconds: float = 300.0) -> bool:
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
            return

        # Final snapshot before teardown so "last RSS" is always in the log.
        try:
            log_memory_usage(prefix="shutdown")
        except Exception as _log_err:
            logger.debug("Final memory log failed during shutdown: %s", _log_err)

        _stop_event.set()
        thread = _monitor_thread
        _monitor_thread = None
        _stop_event = None

    # Join outside the lock so a stuck log call can't deadlock shutdown.
    try:
        thread.join(timeout=timeout)
    except Exception as _join_err:
        logger.debug("Memory monitor thread join failed: %s", _join_err)

    logger.info("[MEMORY] Periodic memory monitoring stopped")


def is_running() -> bool:
    """True if the background monitor thread is alive."""
    with _lock:
        return _monitor_thread is not None and _monitor_thread.is_alive()
