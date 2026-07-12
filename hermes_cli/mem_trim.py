"""Rate-limited heap release for long-lived Hermes gateway processes.

On Linux/glibc, ``malloc_trim(0)`` can return pages from freed Python/C
allocations to the OS.  Other platforms and allocators are safe no-ops.
Set ``HERMES_DISABLE_MEMORY_TRIM=1`` as an emergency kill switch.
"""

from __future__ import annotations

import ctypes
import gc
import logging
import os
import platform
import sys
import threading
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

_DEFAULT_COOLDOWN_SECONDS = 60.0
_DISABLE_VALUES = frozenset({"1", "true", "yes", "on"})
_trim_lock = threading.Lock()
_last_trim_monotonic = 0.0
_probe_done = False
_malloc_trim: Callable[[int], int] | None = None


def _disabled() -> bool:
    return (
        os.environ.get("HERMES_DISABLE_MEMORY_TRIM", "").strip().lower()
        in _DISABLE_VALUES
    )


def _cooldown_seconds(value: float | None) -> float:
    if value is None:
        raw = os.environ.get("HERMES_MEMORY_TRIM_COOLDOWN_SECONDS", "")
        if raw:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = _DEFAULT_COOLDOWN_SECONDS
        else:
            value = _DEFAULT_COOLDOWN_SECONDS
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return _DEFAULT_COOLDOWN_SECONDS


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
    Unsupported allocators, the kill switch, cooldown suppression, and all
    runtime errors return ``False`` without affecting the caller.
    """
    if _disabled():
        return False

    global _last_trim_monotonic
    with _trim_lock:
        trim = _probe_glibc_malloc_trim()
        if trim is None:
            return False
        now = time.monotonic()
        cooldown = _cooldown_seconds(cooldown_seconds)
        if not force and _last_trim_monotonic and now - _last_trim_monotonic < cooldown:
            return False
        # Record the attempt before calling into libc so repeated failures do not
        # turn every turn boundary into an expensive full collection.
        _last_trim_monotonic = now
        try:
            gc.collect()
            released = bool(trim(0))
            if reason:
                logger.debug("malloc_trim(0) after %s: released=%s", reason, released)
            return released
        except Exception as exc:
            logger.debug("malloc_trim failed after %s: %s", reason or "cleanup", exc)
            return False
