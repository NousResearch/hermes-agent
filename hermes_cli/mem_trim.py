"""Optional process memory trim for dashboard backends.

On glibc/Linux, ``malloc_trim(0)`` returns freed heap pages to the OS.
On macOS, trimming is a no-op here — the desktop bundle preloads mimalloc
with ``MIMALLOC_PURGE_DELAY=0`` instead.
"""

from __future__ import annotations

import gc
import logging
import platform
import sys
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

_last_trim_monotonic: float = 0.0
_trim_lock = threading.Lock()
_glibc_malloc_trim = None
_glibc_probe_done = False


def _probe_glibc_malloc_trim():
    global _glibc_malloc_trim, _glibc_probe_done
    if _glibc_probe_done:
        return
    _glibc_probe_done = True
    if sys.platform != "linux":
        return
    try:
        libc_name = platform.libc_ver()[0]
    except Exception:
        return
    if libc_name != "glibc":
        return
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6")
        if hasattr(libc, "malloc_trim"):
            _glibc_malloc_trim = libc.malloc_trim
    except Exception as exc:
        logger.debug("malloc_trim unavailable: %s", exc)


def trim_memory(*, force: bool = False, reason: str = "") -> bool:
    """Attempt to return freed heap to the OS. Returns True if trim ran."""
    from hermes_cli.dashboard_memory import (
        memory_trim_cooldown_seconds,
        memory_trim_enabled,
    )

    if not memory_trim_enabled() and not force:
        return False

    global _last_trim_monotonic
    now = time.monotonic()
    cooldown = memory_trim_cooldown_seconds()
    with _trim_lock:
        if not force and (now - _last_trim_monotonic) < cooldown:
            return False
        _last_trim_monotonic = now

    gc.collect()
    _probe_glibc_malloc_trim()
    if _glibc_malloc_trim is None:
        return False
    try:
        _glibc_malloc_trim(0)
        if reason:
            logger.debug("malloc_trim(0) after %s", reason)
        return True
    except Exception as exc:
        logger.debug("malloc_trim failed: %s", exc)
        return False
