"""Rate-limited heap release for long-lived Hermes gateway processes.

On Linux/glibc, ``malloc_trim(0)`` can return pages from freed Python/C
allocations to the OS.  Other platforms and allocators are safe no-ops.
Behavior is configured under ``context.memory_trim`` in ``config.yaml``.
"""

from __future__ import annotations

import ctypes
import gc
import logging
import platform
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_COOLDOWN_SECONDS = 60.0
_DEFAULT_LOG_EVERY_N = 1
_DEFAULT_INFO_LOG_MIN_DELTA_MB = 0.0
_trim_lock = threading.Lock()
_last_trim_monotonic = 0.0
_probe_done = False
_malloc_trim: Callable[[int], int] | None = None
_trim_call_count = 0


def _config_settings() -> tuple[bool, float, int, float]:
    """Return fail-open settings from the normal Hermes config path."""
    enabled = True
    cooldown: Any = _DEFAULT_COOLDOWN_SECONDS
    log_every_n: Any = _DEFAULT_LOG_EVERY_N
    info_log_min_delta_mb: Any = _DEFAULT_INFO_LOG_MIN_DELTA_MB
    try:
        from hermes_cli.config import load_config

        config = load_config() or {}
        context = config.get("context") if isinstance(config, dict) else None
        settings = context.get("memory_trim") if isinstance(context, dict) else None
        if isinstance(settings, dict):
            configured_enabled = settings.get("enabled")
            if isinstance(configured_enabled, bool):
                enabled = configured_enabled
            cooldown = settings.get("cooldown_seconds", _DEFAULT_COOLDOWN_SECONDS)
            log_every_n = settings.get("log_every_n", _DEFAULT_LOG_EVERY_N)
            info_log_min_delta_mb = settings.get(
                "info_log_min_delta_mb", _DEFAULT_INFO_LOG_MIN_DELTA_MB
            )
    except Exception:
        pass
    return (
        enabled,
        _cooldown_seconds(cooldown),
        _log_every_n(log_every_n),
        _nonnegative_float(info_log_min_delta_mb, _DEFAULT_INFO_LOG_MIN_DELTA_MB),
    )


def _cooldown_seconds(value: Any) -> float:
    if isinstance(value, bool):
        return _DEFAULT_COOLDOWN_SECONDS
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return _DEFAULT_COOLDOWN_SECONDS


def _log_every_n(value: Any) -> int:
    if isinstance(value, bool):
        return _DEFAULT_LOG_EVERY_N
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return _DEFAULT_LOG_EVERY_N


def _nonnegative_float(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return default


def _read_proc_status() -> str | None:
    """Read Linux process status without making non-Linux callers special-case."""
    if sys.platform != "linux":
        return None
    try:
        return Path("/proc/self/status").read_text(encoding="utf-8")
    except OSError:
        return None


def collect_memory_snapshot(history_bytes: int | None = None) -> dict[str, int | None]:
    """Return lightweight process-memory telemetry for trim logs and canaries.

    ``VmRSS`` and ``RssAnon`` are Linux-only best effort fields.  The helper is
    intentionally dependency-free so allocation recovery never requires psutil.
    """
    snapshot: dict[str, int | None] = {
        "rss_kib": None,
        "rss_anon_kib": None,
        "thread_count": threading.active_count(),
    }
    status = _read_proc_status()
    if status:
        for line in status.splitlines():
            key, separator, raw_value = line.partition(":")
            if not separator or key not in {"VmRSS", "RssAnon"}:
                continue
            value = raw_value.strip().split(maxsplit=1)
            if value and value[0].isdigit():
                snapshot["rss_kib" if key == "VmRSS" else "rss_anon_kib"] = int(value[0])
    if isinstance(history_bytes, int) and history_bytes >= 0:
        snapshot["history_bytes"] = history_bytes
    return snapshot


def _should_log_trim(
    *, force: bool, log_every_n: int, call_count: int, before: dict[str, int | None],
    after: dict[str, int | None], info_log_min_delta_mb: float,
) -> bool:
    # trim_memory calls this only after malloc_trim reported success. A forced
    # successful trim is an explicit observability event, regardless of RSS.
    if force:
        return True
    if not force and call_count % log_every_n:
        return False
    before_rss = before.get("rss_kib")
    after_rss = after.get("rss_kib")
    if before_rss is None or after_rss is None:
        return True
    return abs(after_rss - before_rss) >= info_log_min_delta_mb * 1024


def _probe_glibc_malloc_trim() -> Callable[[int], int] | None:
    """Resolve glibc's malloc_trim once; return None on unsupported systems."""
    global _malloc_trim, _probe_done
    if _probe_done:
        return _malloc_trim
    _probe_done = True
    if sys.platform != "linux":
        return None
    try:
        if platform.libc_ver()[0].lower() != "glibc":
            return None
        libc = ctypes.CDLL(None)
        trim = libc.malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        _malloc_trim = trim
    except Exception as exc:
        logger.debug("malloc_trim unavailable: %s", exc)
    return _malloc_trim


def trim_memory(
    *,
    force: bool = False,
    reason: str = "",
    cooldown_seconds: float | None = None,
) -> bool:
    """Collect cycles and ask glibc to release free heap pages.

    Returns ``True`` only when ``malloc_trim(0)`` ran and reported success.
    Unsupported allocators, the config kill switch, cooldown suppression, and all
    runtime errors return ``False`` without affecting the caller.
    """
    (
        enabled,
        configured_cooldown,
        log_every_n,
        info_log_min_delta_mb,
    ) = _config_settings()
    if not enabled:
        return False

    global _last_trim_monotonic, _trim_call_count
    with _trim_lock:
        trim = _probe_glibc_malloc_trim()
        if trim is None:
            return False
        now = time.monotonic()
        cooldown = (
            configured_cooldown
            if cooldown_seconds is None
            else _cooldown_seconds(cooldown_seconds)
        )
        if not force and _last_trim_monotonic and now - _last_trim_monotonic < cooldown:
            return False
        # Record the attempt before calling into libc so repeated failures do not
        # turn every turn boundary into an expensive full collection.
        _last_trim_monotonic = now
        try:
            before = collect_memory_snapshot()
            started = time.perf_counter()
            gc.collect()
            trim_result = trim(0)
            released = bool(trim_result)
            after = collect_memory_snapshot()
            duration_ms = (time.perf_counter() - started) * 1000
            _trim_call_count += 1
            if released and _should_log_trim(
                force=force,
                log_every_n=log_every_n,
                call_count=_trim_call_count,
                before=before,
                after=after,
                info_log_min_delta_mb=info_log_min_delta_mb,
            ):
                logger.info(
                    "memory trim: reason=%s malloc_trim=%s rss_kib=%s->%s "
                    "rss_anon_kib=%s->%s threads=%s duration_ms=%.1f",
                    reason or "cleanup",
                    trim_result,
                    before.get("rss_kib"),
                    after.get("rss_kib"),
                    before.get("rss_anon_kib"),
                    after.get("rss_anon_kib"),
                    after.get("thread_count"),
                    duration_ms,
                )
            return released
        except Exception as exc:
            logger.warning(
                "memory trim failed after %s: %s: %s",
                reason or "cleanup",
                type(exc).__name__,
                exc,
            )
            return False
