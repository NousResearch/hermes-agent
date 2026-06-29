"""Periodic process memory sampling + heap-dump-on-threshold for the gateway.

Ported from cline/cline#10343 (src/standalone/memory-monitor.ts).

The gateway is a long-lived process that accumulates memory as it caches
agent instances, session transcripts, tool schemas, memory providers, MCP
connections, etc.  A slow leak in any of those subsystems is invisible
in a single log line — you only see it by watching RSS climb over hours.

This module does two things:

1. **Periodic RSS sampling** — every N seconds (default 60) log a structured
   ``[MEMORY] rss=...MB peak=...MB ...`` line with PID, current RSS,
   peak RSS, GC counts, and thread count.  Baseline snapshot on start,
   final snapshot on shutdown.

2. **Heap dump on threshold** — when current RSS crosses a configurable
   threshold (default 4 GiB, well below the observed 25.5 GB peak),
   a heap snapshot is captured via ``tracemalloc`` and pickled into
   ``~/.hermes/logs/heap-dumps/`` with rotation (oldest evicted).

Design notes:
  * Grep-friendly single-line format beginning ``[MEMORY]``.
  * ``[MEMORY] HEAP_DUMP`` line on every dump with path + size.
  * Daemon thread — never blocks process exit.
  * Linux: reads ``/proc/self/statm`` for current RSS (reliable,
    always-current) and ``resource.getrusage().ru_maxrss`` for peak.
  * Windows / macOS / BSD: uses ``psutil`` (already an optional dep
    for memory introspection on those platforms).
  * Heap dumps use ``tracemalloc`` (stdlib) — captures Python object
    allocations, not native/C memory.  Enough to find a Python-level
    leak; if the leak is in a C extension, the dump still shows
    Python-side attribution that points to the offender.
  * Threshold=0 disables the trigger entirely (use during benchmark
    runs where you only want sampling, not dumps).

Config: ``logging.memory_monitor`` block in ``config.yaml`` — see
``hermes_cli/config.py`` for the defaults.  Tunable knobs:

    logging:
      memory_monitor:
        enabled: true           # master switch (default: true)
        interval_seconds: 60    # how often to sample RSS
        heap_dump_threshold_mb: 4096   # 4 GiB; 0 disables
        heap_dump_dir: null     # defaults to ~/.hermes/logs/heap-dumps
        heap_dump_top_n: 50     # top allocation sites captured per dump
        heap_dump_max_files: 5  # rotation count
"""

from __future__ import annotations

import gc
import glob
import itertools
import logging
import os
import pickle
import sys
import threading
import time
import tracemalloc
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_BYTES_TO_MB = 1024 * 1024
_KB_TO_MB = 1024 * 1024

_monitor_thread: Optional[threading.Thread] = None
_stop_event: Optional[threading.Event] = None
_start_time: Optional[float] = None
_interval_seconds: float = 60.0
_lock = threading.Lock()

# Heap-dump trigger state — module-level so the periodic loop can read/write
# without holding the long-lived lock for the (slow) dump itself.
_heap_dump_threshold_mb: int = 0  # 0 = disabled
_heap_dump_dir: Optional[Path] = None
_heap_dump_top_n: int = 50
_heap_dump_max_files: int = 5
_heap_dump_in_flight: bool = False  # guard against re-entrance if dump is slow
_peak_rss_mb: int = 0  # running peak across the lifetime of this process

# Monotonic counter for sub-second uniqueness in dump filenames.  Two dumps
# in the same wall-clock second get distinct filenames so rotation can see
# each one.  itertools.count is process-local and never reused — combined
# with the timestamp, two dumps will never collide on name within a
# process's lifetime.
_dump_seq = itertools.count(1)


# ---------------------------------------------------------------------------
# RSS source — Linux /proc/self/statm + resource.getrusage, psutil fallback
# ---------------------------------------------------------------------------


def _read_proc_self_rss_kb() -> Optional[int]:
    """Read current RSS (in KiB) from /proc/self/statm.  Linux-only.

    /proc/self/statm fields are: size resident shared text lib data dt
    All values are in pages (multiply by page size for bytes).  Field 1
    (resident) is the current RSS — what we want, not ru_maxrss which
    is a high-water mark that only ever climbs.
    """
    try:
        with open("/proc/self/statm", "r", encoding="ascii") as f:
            line = f.readline()
        # Field 1 is RSS in pages.  Avoid a getconf syscall by reading
        # /proc/self/smaps_rollup is overkill; just trust the standard
        # 4 KiB page size — wrong on PowerPC / aarch64 with 64K pages,
        # but Linux x86_64 + arm64 server kernels all use 4K.
        rss_pages = int(line.split()[1])
        return rss_pages * 4  # 4 KiB → KiB
    except Exception:
        return None


def _get_rss_mb() -> Optional[int]:
    """Return current process RSS in MB, or None if unavailable.

    Linux:   /proc/self/statm (current, reliable).
    macOS:   resource.getrusage().ru_maxrss is in BYTES on macOS — gives
             peak, not current, but it's the only stdlib option.  Acceptable
             for trend monitoring.
    Windows: psutil fallback.
    """
    if sys.platform.startswith("linux"):
        kb = _read_proc_self_rss_kb()
        if kb is not None:
            return int(kb / 1024)

    # Fallback for macOS / non-Linux: resource gives peak RSS only.
    # Good enough for trend detection — we expose peak alongside current
    # so the operator can tell which signal they're seeing.
    try:
        import resource

        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return int(maxrss / _BYTES_TO_MB)
        # Other unices (BSD): bytes
        return int(maxrss / _BYTES_TO_MB) if maxrss > 10 * 1024 * 1024 else int(maxrss / 1024)
    except Exception:
        pass

    try:
        import psutil  # type: ignore

        rss = psutil.Process(os.getpid()).memory_info().rss
        return int(rss / _BYTES_TO_MB)
    except Exception:
        return None


def _get_peak_rss_mb() -> Optional[int]:
    """Return process peak RSS in MB, or None if unavailable.

    Uses ``resource.getrusage().ru_maxrss`` which is the OS-reported
    high-water mark for the lifetime of the process.  Unit is
    KB on Linux, bytes on macOS, bytes on Windows.
    """
    try:
        import resource

        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return int(maxrss / _BYTES_TO_MB)
        # Linux / other unices: KB
        return int(maxrss / 1024)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Configuration helper — reads logging.memory_monitor from config.yaml
# ---------------------------------------------------------------------------


def _resolve_hermes_home() -> Optional[Path]:
    """Best-effort resolve of ~/.hermes for log directory fallback.

    Imports lazily so this module can be unit-tested without the full
    Hermes config stack present.
    """
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home())
    except Exception:
        pass
    env = os.environ.get("HERMES_HOME")
    if env:
        return Path(env)
    return Path.home() / ".hermes"


def _resolve_memory_monitor_config() -> dict[str, Any]:
    """Read the ``logging.memory_monitor`` block from config.yaml.

    Returns a dict with all expected keys; missing keys get sensible
    defaults so callers don't have to defensively ``.get()`` everywhere.

    We never raise — the monitor is best-effort observability, and a
    malformed config must not prevent the gateway from starting.
    """
    defaults: dict[str, Any] = {
        "enabled": True,
        "interval_seconds": 60,
        "heap_dump_threshold_mb": 4096,
        "heap_dump_dir": None,
        "heap_dump_top_n": 50,
        "heap_dump_max_files": 5,
    }
    try:
        from hermes_cli.config import read_raw_config  # type: ignore

        raw = read_raw_config() or {}
        logging_block = raw.get("logging") or {}
        if not isinstance(logging_block, dict):
            return defaults
        mm = logging_block.get("memory_monitor")
        if not isinstance(mm, dict):
            return defaults
        for key in defaults:
            if key in mm:
                defaults[key] = mm[key]
    except Exception as e:
        logger.debug("[MEMORY] Could not read logging.memory_monitor config: %s", e)
    return defaults


# ---------------------------------------------------------------------------
# Heap dump machinery
# ---------------------------------------------------------------------------


def _rotate_heap_dumps(directory: Path, max_files: int) -> None:
    """Keep only the ``max_files`` most recent ``.pkl`` dumps in ``directory``.

    Eviction is purely age-based — assumes dump filenames encode timestamp
    prefix (which they do — ``heap-YYYYMMDD-HHMMSS.pkl``), so ``glob``
    sorted by name gives chronological order.
    """
    try:
        files = sorted(directory.glob("heap-*.pkl"))
    except Exception:
        return
    excess = len(files) - max_files
    if excess <= 0:
        return
    for old in files[:excess]:
        try:
            old.unlink()
        except Exception:
            pass


def _take_heap_dump(top_n: int) -> Optional[Path]:
    """Capture a tracemalloc snapshot and persist it to disk.

    Returns the path written, or None on failure.  Always tries to
    stop any prior tracemalloc session before starting — leaks tend to
    be diagnosed in production where the cost of leaving tracemalloc
    running is acceptable during the dump window itself.
    """
    global _heap_dump_dir

    if _heap_dump_dir is None:
        hermes_home = _resolve_hermes_home()
        if hermes_home is None:
            logger.warning("[MEMORY] HEAP_DUMP skipped — no HERMES_HOME to write to")
            return None
        _heap_dump_dir = hermes_home / "logs" / "heap-dumps"
    try:
        _heap_dump_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning("[MEMORY] HEAP_DUMP skipped — cannot create %s: %s", _heap_dump_dir, e)
        return None

    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()

    try:
        snap = tracemalloc.take_snapshot()
        try:
            stats = snap.statistics("traceback")
        except Exception:
            # tracemalloc requires tracebacks to be captured at allocation
            # time — start(25) caps the frame depth.  If we started above
            # with default depth=1, statistics() will raise.  Fall back to
            # filename grouping.
            stats = snap.statistics("filename")
        top = stats[:top_n]
        payload = {
            "ts": time.time(),
            "pid": os.getpid(),
            "rss_mb_at_dump": _get_rss_mb(),
            "peak_rss_mb_at_dump": _get_peak_rss_mb(),
            "top_allocations": [
                {
                    # traceback.statistic objects aren't trivially pickleable;
                    # serialize the fields we actually want to read offline.
                    "size_bytes": getattr(s, "size", 0),
                    "count": getattr(s, "count", 0),
                    "traceback": str(getattr(s, "traceback", "")),
                }
                for s in top
            ],
        }
        ts_label = time.strftime("%Y%m%d-%H%M%S")
        seq = next(_dump_seq)
        out_path = _heap_dump_dir / f"heap-{ts_label}-{os.getpid()}-{seq}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        return out_path
    except Exception as e:
        logger.warning("[MEMORY] HEAP_DUMP failed: %s", e)
        return None
    finally:
        if not was_tracing:
            try:
                tracemalloc.stop()
            except Exception:
                pass


def _maybe_trigger_heap_dump(current_rss_mb: int) -> None:
    """If RSS crosses threshold and trigger hasn't fired recently, dump.

    ``_heap_dump_in_flight`` guards against re-entrance: tracemalloc
    snapshots can take seconds on a process with GB of live allocations,
    and we don't want the next sample tick to start a second dump.
    Re-arming happens automatically once the dump completes and the
    RSS comes back down below threshold (next crossing fires again).
    """
    global _heap_dump_in_flight
    if _heap_dump_threshold_mb <= 0:
        return
    if _heap_dump_in_flight:
        return
    if current_rss_mb < _heap_dump_threshold_mb:
        return
    _heap_dump_in_flight = True
    try:
        logger.warning(
            "[MEMORY] HEAP_DUMP_TRIGGERED rss=%dMB threshold=%dMB",
            current_rss_mb,
            _heap_dump_threshold_mb,
        )
        out_path = _take_heap_dump(_heap_dump_top_n)
        if out_path is not None:
            try:
                size_bytes = out_path.stat().st_size
            except Exception:
                size_bytes = -1
            logger.warning(
                "[MEMORY] HEAP_DUMP path=%s size=%d top_n=%d",
                out_path,
                size_bytes,
                _heap_dump_top_n,
            )
            if _heap_dump_dir is not None and _heap_dump_max_files > 0:
                _rotate_heap_dumps(_heap_dump_dir, _heap_dump_max_files)
    finally:
        _heap_dump_in_flight = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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
    global _peak_rss_mb
    rss = _get_rss_mb()
    peak = _get_peak_rss_mb()
    if rss is not None and rss > _peak_rss_mb:
        _peak_rss_mb = rss
    # If /proc gave us current but resource failed for peak, still report
    # the in-process running peak so the operator sees trend.
    effective_peak = peak if peak is not None else _peak_rss_mb
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
            "[MEMORY] %srss=unavailable peak=%dMB gc=%s threads=%d uptime=%ds pid=%d",
            tag,
            effective_peak,
            gc_counts,
            thread_count,
            uptime,
            os.getpid(),
        )
    else:
        logger.info(
            "[MEMORY] %srss=%dMB peak=%dMB gc=%s threads=%d uptime=%ds pid=%d",
            tag,
            rss,
            effective_peak,
            gc_counts,
            thread_count,
            uptime,
            os.getpid(),
        )

    # Heap-dump trigger fires AFTER the log line so the operator sees the
    # triggering RSS sample in the same second as the dump notice.
    if rss is not None:
        _maybe_trigger_heap_dump(rss)


def _monitor_loop(stop_event: threading.Event, interval: float) -> None:
    """Background thread body — log every ``interval`` seconds until stopped."""
    while not stop_event.wait(interval):
        try:
            log_memory_usage()
        except Exception as e:
            # Never let the monitor crash the gateway; just log and carry on.
            logger.debug("Memory monitor iteration failed: %s", e)


def _apply_config(cfg: dict[str, Any]) -> None:
    """Apply resolved config block to module state.  Called under ``_lock``."""
    global _interval_seconds, _heap_dump_threshold_mb
    global _heap_dump_dir, _heap_dump_top_n, _heap_dump_max_files
    try:
        _interval_seconds = max(1.0, float(cfg.get("interval_seconds", 60)))
    except (TypeError, ValueError):
        _interval_seconds = 60.0
    try:
        _heap_dump_threshold_mb = int(cfg.get("heap_dump_threshold_mb", 4096))
    except (TypeError, ValueError):
        _heap_dump_threshold_mb = 4096
    raw_dir = cfg.get("heap_dump_dir")
    if raw_dir:
        try:
            _heap_dump_dir = Path(raw_dir).expanduser()
        except Exception:
            _heap_dump_dir = None
    else:
        _heap_dump_dir = None
    try:
        _heap_dump_top_n = max(1, int(cfg.get("heap_dump_top_n", 50)))
    except (TypeError, ValueError):
        _heap_dump_top_n = 50
    try:
        _heap_dump_max_files = max(0, int(cfg.get("heap_dump_max_files", 5)))
    except (TypeError, ValueError):
        _heap_dump_max_files = 5


def start_memory_monitoring(interval_seconds: float | None = None) -> bool:
    """Start periodic memory usage logging in a daemon thread.

    Reads ``logging.memory_monitor`` from config.yaml unless ``interval_seconds``
    is passed explicitly (useful for tests).  Logs immediately to capture
    a baseline, then every ``interval_seconds``.  Safe to call multiple
    times — subsequent calls are no-ops while the first monitor is still
    running.

    Parameters
    ----------
    interval_seconds
        Override the configured interval.  ``None`` = use config value.

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
                "[MEMORY] Memory monitoring unavailable: neither /proc/self/statm "
                "nor resource.getrusage nor psutil could read process RSS — "
                "skipping periodic logging."
            )
            return False

        cfg = _resolve_memory_monitor_config()
        if not cfg.get("enabled", True):
            logger.info("[MEMORY] Memory monitoring disabled via logging.memory_monitor.enabled")
            return False

        _apply_config(cfg)
        if interval_seconds is not None:
            try:
                # Floor at 0.01s — fast enough for tests, slow enough that a
                # misconfigured prod won't melt the CPU logging 1000×/s.
                _interval_seconds = max(0.01, float(interval_seconds))
            except (TypeError, ValueError):
                pass

        _start_time = time.monotonic()
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
            "[MEMORY] Periodic memory monitoring started (interval: %ds, "
            "heap_dump_threshold: %dMB, heap_dump_dir: %s)",
            int(_interval_seconds),
            _heap_dump_threshold_mb,
            _heap_dump_dir or "<default ~/.hermes/logs/heap-dumps>",
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

    logger.info("[MEMORY] Periodic memory monitoring stopped")


def is_running() -> bool:
    """True if the background monitor thread is alive."""
    with _lock:
        return _monitor_thread is not None and _monitor_thread.is_alive()


def configure_for_test(
    *,
    threshold_mb: int = 0,
    dump_dir: Optional[Path] = None,
    top_n: int = 50,
    max_files: int = 5,
    interval_seconds: float = 60.0,
) -> None:
    """Test-only: set module-level heap-dump config without touching the thread.

    Lets tests exercise the threshold trigger and rotation logic without
    having to write a full config.yaml or start the background thread.
    Safe to call repeatedly; resets to defaults on every call.
    """
    global _heap_dump_threshold_mb, _heap_dump_dir
    global _heap_dump_top_n, _heap_dump_max_files, _interval_seconds, _heap_dump_in_flight
    _heap_dump_threshold_mb = int(threshold_mb)
    _heap_dump_dir = Path(dump_dir).expanduser() if dump_dir is not None else None
    _heap_dump_top_n = int(top_n)
    _heap_dump_max_files = int(max_files)
    _interval_seconds = float(interval_seconds)
    _heap_dump_in_flight = False