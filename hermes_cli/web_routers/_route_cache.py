"""Opt-in short-TTL cache for config-derived GET payloads.

These routes are polled in bursts by the mobile/desktop clients, and each call
re-reads config files and re-walks provider/profile inventories. The payloads
change on a human timescale, so a few seconds of reuse is invisible to the UI
while it removes the repeated work behind an app resume.

Opt-in via ``HERMES_ROUTE_CACHE=1`` so the default install, the test suite and
single-shot callers keep seeing live values.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Optional

_ENABLED = os.environ.get("HERMES_ROUTE_CACHE", "").strip().lower() in {"1", "true", "yes", "on"}
_MAX_ENTRIES = 128

# key -> (monotonic_at, ttl_seconds, value)
_STORE: dict[tuple, tuple[float, float, Any]] = {}
_LOCK = threading.Lock()


def enabled() -> bool:
    return _ENABLED


def get(key: tuple) -> Optional[Any]:
    if not _ENABLED:
        return None
    with _LOCK:
        entry = _STORE.get(key)
    if entry is None:
        return None
    ts, ttl, value = entry
    return value if time.monotonic() - ts < ttl else None


def put(key: tuple, value: Any, ttl: float) -> None:
    if not _ENABLED or value is None or ttl <= 0:
        return
    with _LOCK:
        # Bounded: keys carry caller-controlled profile names.
        if len(_STORE) >= _MAX_ENTRIES and key not in _STORE:
            _STORE.pop(next(iter(_STORE)), None)
        _STORE[key] = (time.monotonic(), ttl, value)


def clear() -> None:
    with _LOCK:
        _STORE.clear()
