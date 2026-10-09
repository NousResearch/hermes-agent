"""Rate-limited heap release for long-lived Hermes gateway processes.

On Linux/glibc, ``malloc_trim(0)`` can return pages from freed Python/C allocations to the OS. Every
platform runs the rate-limited ``gc.collect()``; only the page release is glibc-only. Behavior is configured
under ``context.memory_trim`` in ``config.yaml``.
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
_DEFAULT_LOG_EVERY_N = 60
_DEFAULT_INFO_LOG_MIN_DELTA_MB = 0.0
# Even forced trims honor a short floor: AIAgent.close() forces a trim, and delegate
# batches close N child subagents back-to-back in the SAME process — without a floor
# that stacks N+1 uncooled full gc.collect() passes (50-500ms each in a large gateway
# process). 5s coalesces the burst while keeping the parent's final close-trim effective.
_FORCE_FLOOR_SECONDS = 5.0
_trim_lock = threading.Lock()
_last_trim_monotonic = 0.0
# ``_last_trim_monotonic`` stamps collecting trims only, so an allocator-only trim (housekeeping tick, agent
# close) never pushes a later forced collection (cache-pressure eviction) inside its floor.
_last_alloc_trim_monotonic = 0.0
_probe_done = False
_malloc_trim: Callable[[int], int] | None = None
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
    # Called only after malloc_trim reported success; a forced successful trim is an
    # explicit observability event regardless of RSS.
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


def trim_memory(
    *, force: bool = False, reason: str = "", cooldown_seconds: float | None = None,
    collect: bool = True) -> bool:
    """Collect cycles and ask glibc to release free heap pages.

    Returns ``True`` only when ``malloc_trim(0)`` ran and reported success. Unsupported allocators,
    the config kill switch, cooldown suppression, and all runtime errors return ``False`` without
    affecting the caller.

    ``collect=False`` drops the forced full ``gc.collect()`` and only asks the allocator to return
    already-free pages. Every call site on a busy path of a long-lived process passes it (gateway
    housekeeping tick, agent close, TUI turn completion, post-compression): a full collection walks
    the whole shared heap with the GIL held, 250-750 ms on a multi-million-object gateway, freezing
    every thread (adapters, streams, API requests) while it runs. Most of what those paths free goes
    by reference counting; cycles wait for ``trim_memory_when_idle``, the timer lanes' one full
    collection per quiet period. The forced collection otherwise stays only where memory is the
    emergency (agent-cache pressure eviction) or the process is single-session (slash worker).
    """
    enabled, configured_cooldown, log_every_n, info_log_min_delta_mb = _config_settings()
    if not enabled:
        return False

    global _last_trim_monotonic, _last_alloc_trim_monotonic, _trim_call_count
    with _trim_lock:
        trim = _probe_glibc_malloc_trim()
        now = time.monotonic()
        cooldown = configured_cooldown if cooldown_seconds is None else _cooldown_seconds(cooldown_seconds)
        last = _last_trim_monotonic if collect else max(_last_trim_monotonic, _last_alloc_trim_monotonic)
        if last and now - last < (_FORCE_FLOOR_SECONDS if force else cooldown):
            return False
        # Record the attempt before calling into libc so repeated failures do not
        # turn every turn boundary into an expensive full collection.
        _last_alloc_trim_monotonic = now
        if collect:
            _last_trim_monotonic = now
        try:
            before = collect_memory_snapshot()
            started = time.perf_counter()
            if collect:
                gc.collect()
            # Cycle collection is allocator-independent; only the page release needs glibc.
            trim_result = trim(0) if trim is not None else 0
            released = bool(trim_result)
            after = collect_memory_snapshot()
            duration_ms = (time.perf_counter() - started) * 1000
            _trim_call_count += 1
            if released:
                level = logging.INFO if _should_log_trim(
                    force=force, log_every_n=log_every_n, call_count=_trim_call_count,
                    before=before, after=after, info_log_min_delta_mb=info_log_min_delta_mb,
                ) else logging.DEBUG
                logger.log(
                    level,
                    "memory trim: reason=%s malloc_trim=%s collected=%s rss_kib=%s->%s "
                    "rss_anon_kib=%s->%s threads=%s duration_ms=%.1f",
                    reason or "cleanup", trim_result, collect,
                    before.get("rss_kib"), after.get("rss_kib"),
                    before.get("rss_anon_kib"), after.get("rss_anon_kib"),
                    after.get("thread_count"), duration_ms)
            return released
        except Exception as exc:
            logger.warning(
                "memory trim failed after %s: %s: %s", reason or "cleanup", type(exc).__name__, exc)
            return False


# A gateway that has been turn-free this long has nobody waiting on it, so the full collection the busy-path
# trims skip can run here without freezing anyone. Python 3.14's incremental collector advances only on new
# allocations, so a quiet process otherwise keeps closed agents' reference cycles (and their pages) forever.
IDLE_COLLECT_AFTER_SECONDS = 300.0
_idle_collect_lock = threading.Lock()
_idle_collected_for: float | None = None  # quiet-period stamp the last idle collection covered
_agent_closed_since_idle_collect = False


def note_agent_closed() -> None:
    """A closed agent may leave reference cycles: re-arm the idle collection even within one quiet period."""
    global _agent_closed_since_idle_collect
    _agent_closed_since_idle_collect = True


def trim_memory_when_idle(*, reason: str) -> bool:
    """Timer-lane trim: allocator release only, plus ONE full collection per quiet period once no turn has run
    in this process for ``IDLE_COLLECT_AFTER_SECONDS`` (again only after a later agent close)."""
    global _idle_collected_for, _agent_closed_since_idle_collect
    from agent.review_idle_queue import QUEUE

    quiet_for, quiet_since = QUEUE.quiet_period()
    with _idle_collect_lock:
        due = quiet_since is not None and quiet_for >= IDLE_COLLECT_AFTER_SECONDS and (
            quiet_since != _idle_collected_for or _agent_closed_since_idle_collect)
        if due:
            _idle_collected_for, _agent_closed_since_idle_collect = quiet_since, False
    if not due or not _config_settings()[0]:
        return trim_memory(reason=reason, collect=False)
    started = time.perf_counter()
    freed = gc.collect()
    logger.info("idle full collection: reason=%s quiet_s=%d unreachable=%d duration_ms=%.1f",
                reason, quiet_for, freed, (time.perf_counter() - started) * 1000)
    return trim_memory(force=True, reason=f"{reason} after idle collection", collect=False)
