"""Active-session lease releases off the gateway event loop.

``ActiveSessionLease.release()`` takes the registry's file lock, re-reads and prunes the JSON (a
liveness probe per entry) and atomically rewrites it. Every turn end ran it on the loop, a measured
recurring stall with many turns finishing. Releases go to ONE worker so they land in the order the
turns ended. A release that lands after the same session's next acquire is harmless: the registry
drops entries by lease id and that acquire re-keyed the entry to its own id. The lease pinned its
registry path at acquisition, so the worker needs no profile scope.
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading

logger = logging.getLogger(__name__)

_pool: concurrent.futures.ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()


def release_soon(lease) -> None:
    """Queue *lease*'s release; returns at once."""
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="active-lease-io")
        pool = _pool
    pool.submit(_release, lease)


def _release(lease) -> None:
    try:
        lease.release()
    except Exception:
        logger.debug("Failed to release active session slot", exc_info=True)
