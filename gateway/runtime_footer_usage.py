"""Non-blocking account-usage cache for the runtime footer.

This module deliberately contains no gateway-runner state.  Usage requests are
slow and optional, so callers receive the last snapshot immediately while a
single daemon thread refreshes stale entries in the background.
"""

from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path
from typing import Any

from agent.account_usage import fetch_account_usage
from hermes_constants import get_hermes_home

_CACHE: dict[tuple[str, str, str, str], tuple[float, Any]] = {}
_REFRESHING: set[tuple[str, str, str, str]] = set()
_LOCK = threading.Lock()
_TTL_SECONDS = 90.0
_MAX_ENTRIES = 64


def cache_key(provider: str | None, *, base_url: str | None = None,
              api_key: str | None = None,
              hermes_home: str | Path | None = None) -> tuple[str, str, str, str]:
    token = str(api_key or "").strip()
    digest = hashlib.sha256(token.encode()).hexdigest()[:16] if token else ""
    return (
        str(hermes_home if hermes_home is not None else get_hermes_home()),
        str(provider or "").strip().lower(),
        str(base_url or "").strip().rstrip("/").lower(),
        digest,
    )


def _refresh(key, provider, base_url, api_key) -> None:
    try:
        snapshot = fetch_account_usage(provider, base_url=base_url, api_key=api_key)
    except Exception:
        snapshot = None
    with _LOCK:
        previous = _CACHE.get(key)
        # Stale-while-revalidate: a transient provider failure must not erase a
        # previously useful quota value.
        value = snapshot if snapshot is not None else (previous[1] if previous else None)
        if key not in _CACHE and len(_CACHE) >= _MAX_ENTRIES:
            oldest = min(_CACHE, key=lambda item: _CACHE[item][0])
            _CACHE.pop(oldest, None)
        _CACHE[key] = (time.monotonic(), value)
        _REFRESHING.discard(key)


def _start_refresh(key, provider, base_url, api_key) -> None:
    try:
        threading.Thread(
            target=_refresh,
            args=(key, provider, base_url, api_key),
            name=f"runtime-footer-usage-{key[1] or 'unknown'}",
            daemon=True,
        ).start()
    except Exception:
        with _LOCK:
            _REFRESHING.discard(key)


def get_cached(provider: str | None, *, base_url: str | None = None,
               api_key: str | None = None,
               hermes_home: str | Path | None = None):
    """Return cached usage and schedule at most one non-blocking refresh."""
    normalized = str(provider or "").strip().lower()
    if normalized in {"", "auto"}:
        return None
    key = cache_key(provider, base_url=base_url, api_key=api_key, hermes_home=hermes_home)
    now = time.monotonic()
    start = False
    with _LOCK:
        cached = _CACHE.get(key)
        value = cached[1] if cached else None
        fresh = cached is not None and now - cached[0] < _TTL_SECONDS
        if not fresh and key not in _REFRESHING:
            _REFRESHING.add(key)
            start = True
    if start:
        _start_refresh(key, provider, base_url, api_key)
    return value


def clear() -> None:
    """Clear cache, primarily for tests and profile teardown."""
    with _LOCK:
        _CACHE.clear()
        _REFRESHING.clear()
