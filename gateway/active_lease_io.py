"""Active-session lease releases off the gateway event loop.

``ActiveSessionLease.release()`` takes the registry's file lock, re-reads and prunes the JSON (a
liveness probe per entry) and atomically rewrites it. Every turn end ran it on the loop, a measured
recurring stall with many turns finishing. Only that file write is deferred: the lease leaves this
process's view of the registry before ``release_soon`` returns (``mark_release_queued``), so the
next claim on the loop -- another session at ``max_concurrent_sessions``, or a Desktop/TUI writer
taking over this session -- sees the slot free, exactly as when the release ran inline. Releases
go to ONE worker so the durable drops land in turn-end order. The lease pinned its registry path
at acquisition, so the worker needs no profile scope.
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading

logger = logging.getLogger(__name__)

_pool: concurrent.futures.ThreadPoolExecutor | None = None
_pool_lock = threading.Lock()


def release_soon(lease) -> None:
    """Retire *lease* from this process's registry view now and queue its durable release."""
    global _pool
    from hermes_cli.active_sessions import mark_release_queued

    mark_release_queued(lease)
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
