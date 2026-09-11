"""Process-wide admission control for local inference endpoints."""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Optional

from agent.model_metadata import is_local_endpoint
from utils import base_url_origin

_LOCAL_ENDPOINT_LOCKS: dict[str, threading.RLock] = {}
_LOCAL_ENDPOINT_LOCKS_GUARD = threading.Lock()
_WAIT_POLL_SECONDS = 0.1


def local_endpoint_key(base_url: str) -> Optional[str]:
    """Return a normalized ``host:port`` key for a local inference URL."""
    if not base_url or not is_local_endpoint(base_url):
        return None
    _scheme, host, port = base_url_origin(base_url)
    if not host:
        return None
    if host == "localhost":
        host = "127.0.0.1"
    rendered_host = f"[{host}]" if ":" in host else host
    return f"{rendered_host}:{port}"


@contextmanager
def local_endpoint_lock(
    base_url: str,
    *,
    on_wait: Optional[Callable[[str], None]] = None,
    cancel_requested: Optional[Callable[[], bool]] = None,
) -> Iterator[Optional[str]]:
    """Serialize one local backend while preserving same-thread nested calls.

    The non-blocking probe lets callers surface a wait notice immediately. Queued
    callers poll so an agent interrupt can cancel admission before any request is
    sent. Cloud endpoints bypass the registry entirely.
    """
    key = local_endpoint_key(base_url)
    if key is None:
        yield None
        return

    with _LOCAL_ENDPOINT_LOCKS_GUARD:
        lock = _LOCAL_ENDPOINT_LOCKS.setdefault(key, threading.RLock())

    acquired = lock.acquire(blocking=False)
    if not acquired:
        if on_wait is not None:
            with contextlib.suppress(Exception):
                on_wait(key)
        while not acquired:
            cancelled = False
            if cancel_requested is not None:
                try:
                    cancelled = bool(cancel_requested())
                except Exception:
                    pass
            if cancelled:
                raise InterruptedError(
                    f"Interrupted while waiting for local backend {key}"
                )
            acquired = lock.acquire(timeout=_WAIT_POLL_SECONDS)
    try:
        yield key
    finally:
        lock.release()
