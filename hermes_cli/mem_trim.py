"""Rate-limited heap release for long-lived Hermes gateway processes.

``malloc_trim(0)`` (Linux/glibc) and ``malloc_zone_pressure_relief`` (macOS 10.15+) can
return pages from freed Python/C allocations to the OS. Unsupported platforms still get
the rate-limited ``gc.collect()`` pass; page release is simply a safe no-op there.
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
# Even forced trims honor a short floor: AIAgent.close() forces a trim, and delegate
# batches close N child subagents back-to-back in the SAME process — without a floor
# that stacks N+1 uncooled full gc.collect() passes (50-500ms each in a large gateway
# process). 5s coalesces the burst while keeping the parent's final close-trim effective.
_FORCE_FLOOR_SECONDS = 5.0
_trim_lock = threading.Lock()
_last_trim_monotonic = 0.0
_probe_done = False
_malloc_trim: Callable[[int], int] | None = None
_darwin_probe_done = False
_darwin_relief: Callable[[int | None, int, int], None] | None = None
_trim_call_count = 0


def _config_settings() -> tuple[bool, float, int, float]:
    """Return fail-open ``(enabled, cooldown, log_every_n, info_log_min_delta_mb)`` from config."""
    settings: Any = None
    try:
        # Read-only, no-deepcopy variant: this runs on EVERY trim attempt (before the
        # cooldown check), and a full-config deepcopy per attempt is exactly the
        # allocator garbage this module exists to release.
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly() or {}
        context = config.get("context") if isinstance(config, dict) else None
        settings = context.get("memory_trim") if isinstance(context, dict) else None
    except Exception:
        pass
    if not isinstance(settings, dict):
        settings = {}
    enabled = settings["enabled"] if isinstance(settings.get("enabled"), bool) else True
    return (
        enabled,
        _cooldown_seconds(settings.get("cooldown_seconds")),
        _coerce(settings.get("log_every_n"), _DEFAULT_LOG_EVERY_N, int, 1),
        _coerce(settings.get("info_log_min_delta_mb"), _DEFAULT_INFO_LOG_MIN_DELTA_MB, float, 0.0))


def _coerce(value: Any, default, cast, floor):
    """``cast(value)`` clamped to ``floor``; bools and unparseable values fall back to ``default``."""
    if isinstance(value, bool):
        return default
    try:
        return max(floor, cast(value))
    except (TypeError, ValueError):
        return default


def _cooldown_seconds(value: Any) -> float:
    return _coerce(value, _DEFAULT_COOLDOWN_SECONDS, float, 0.0)


def _read_proc_status() -> str | None:
    """Read Linux process status without making non-Linux callers special-case."""
    if sys.platform != "linux":
        return None
    try:
        return Path("/proc/self/status").read_text(encoding="utf-8")
    except OSError:
        return None


def collect_memory_snapshot(history_bytes: int | None = None) -> dict[str, int | None]:
    """Lightweight process-memory telemetry for trim logs and canaries.

    ``VmRSS`` / ``RssAnon`` are Linux-only best effort; deliberately psutil-free.
    """
    snapshot: dict[str, int | None] = {
        "rss_kib": None, "rss_anon_kib": None, "thread_count": threading.active_count()}
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
    after: dict[str, int | None], info_log_min_delta_mb: float) -> bool:
    # Called only after the platform page-release primitive reported success (glibc
    # malloc_trim non-zero, darwin pressure relief executed without raising); a forced
    # successful trim is an explicit observability event regardless of RSS.
    if force:
        return True
    if call_count % log_every_n:
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
        trim = ctypes.CDLL(None).malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        _malloc_trim = trim
    except Exception as exc:
        logger.debug("malloc_trim unavailable: %s", exc)
    return _malloc_trim


def _probe_darwin_pressure_relief() -> Callable[[int | None, int, int], None] | None:
    """Resolve libSystem's malloc_zone_pressure_relief once; None off macOS.

    Void API (macOS 10.15+): ``relief(zone, length, flags)`` has no return code, so
    "executed without raising" is the success contract.
    """
    global _darwin_probe_done, _darwin_relief
    if _darwin_probe_done:
        return _darwin_relief
    _darwin_probe_done = True
    if sys.platform != "darwin":
        return None
    try:
        relief = ctypes.CDLL("libSystem.B.dylib").malloc_zone_pressure_relief
        relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        relief.restype = None
        _darwin_relief = relief
    except Exception as exc:
        logger.debug("malloc_zone_pressure_relief unavailable: %s", exc)
    return _darwin_relief


def trim_memory(
    *, force: bool = False, reason: str = "", cooldown_seconds: float | None = None) -> bool:
    """Collect cycles and release free heap pages back to the OS.

    Returns ``True`` only when the platform's page-release primitive ran successfully:
    glibc ``malloc_trim(0)`` reporting non-zero, or macOS ``malloc_zone_pressure_relief``
    executing without raising (a void API, so execution IS the success contract). The
    rate-limited ``gc.collect()`` runs on every platform; unsupported page-release
    targets return ``False`` after still collecting. The config kill switch, cooldown
    suppression, and runtime errors return ``False`` without affecting the caller.
    """
    enabled, configured_cooldown, log_every_n, info_log_min_delta_mb = _config_settings()
    if not enabled:
        return False

    global _last_trim_monotonic, _trim_call_count
    with _trim_lock:
        trim = _probe_glibc_malloc_trim()
        relief = _probe_darwin_pressure_relief() if sys.platform == "darwin" else None
        now = time.monotonic()
        cooldown = configured_cooldown if cooldown_seconds is None else _cooldown_seconds(cooldown_seconds)
        since_last = now - _last_trim_monotonic
        if _last_trim_monotonic and since_last < (_FORCE_FLOOR_SECONDS if force else cooldown):
            return False
        # Record the attempt before calling into libc so repeated failures do not
        # turn every turn boundary into an expensive full collection.
        _last_trim_monotonic = now
        try:
            before = collect_memory_snapshot()
            started = time.perf_counter()
            gc.collect()
            if trim is not None:
                trim_result = trim(0)
                released = bool(trim_result)
            elif relief is not None:
                # All zones (None), no byte cap (0), default flags (0): full
                # heuristic pressure relief across the process.
                relief(None, 0, 0)
                trim_result = True
                released = True
            else:
                # No page-release primitive here: the gc pass above is still worth
                # its cost, only the success flag stays False.
                trim_result = False
                released = False
            after = collect_memory_snapshot()
            duration_ms = (time.perf_counter() - started) * 1000
            _trim_call_count += 1
            if released and _should_log_trim(
                force=force, log_every_n=log_every_n, call_count=_trim_call_count,
                before=before, after=after, info_log_min_delta_mb=info_log_min_delta_mb):
                logger.info(
                    "memory trim: reason=%s malloc_trim=%s rss_kib=%s->%s "
                    "rss_anon_kib=%s->%s threads=%s duration_ms=%.1f",
                    reason or "cleanup", trim_result,
                    before.get("rss_kib"), after.get("rss_kib"),
                    before.get("rss_anon_kib"), after.get("rss_anon_kib"),
                    after.get("thread_count"), duration_ms)
            return released
        except Exception as exc:
            logger.warning(
                "memory trim failed after %s: %s: %s", reason or "cleanup", type(exc).__name__, exc)
            return False
