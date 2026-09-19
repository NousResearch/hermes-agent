"""Dedicated thread pool for blocking PTY I/O.

Why this exists (#95559 class of bug, dashboard variant): every keep-alive PTY
session runs a permanent drain loop whose ``bridge.read`` call is blocking on
Windows (ConPTY has no selectable fd, so the read polls and sleeps inside the
worker thread). Scheduling those loops on asyncio's DEFAULT executor parks one
default-pool thread per live terminal for the whole life of the session. The
default pool is only ``min(32, cpu_count + 4)`` threads (16 on a 12-core box),
so a handful of dashboard terminals — plus the ConPTY write workers that
``WinPtyBridge`` documents as leakable — exhaust it. Once exhausted, EVERY route
that offloads blocking work with ``asyncio.to_thread`` / ``run_in_executor(None,
...)`` queues forever: ``/api/sessions/{id}/messages`` and
``/api/sessions/{id}/latest-descendant`` hang while cheap sync routes
(``/api/status``, ``/api/sessions/stats``) still answer, so the dashboard looks
half-alive and no session can be opened.

Keeping PTY I/O on its own pool means terminal load can never starve the
control plane. Threads are created lazily, so a generous ``max_workers`` costs
nothing until terminals actually open.
"""
from __future__ import annotations

import atexit
import concurrent.futures
import threading
from typing import Optional

# Headroom for the registry's max_sessions (16) drain loops plus their write /
# close workers, so a full terminal registry still cannot queue on itself.
PTY_EXECUTOR_MAX_WORKERS = 48

_pty_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
_pty_executor_lock = threading.Lock()


def get_pty_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Lazily create the process-wide PTY worker pool."""
    global _pty_executor
    if _pty_executor is None:
        with _pty_executor_lock:
            if _pty_executor is None:
                _pty_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=PTY_EXECUTOR_MAX_WORKERS, thread_name_prefix="hermes-pty")
                # Never wait on in-flight workers at exit: a read parked inside
                # pywinpty must not block interpreter shutdown.
                atexit.register(
                    lambda: _pty_executor
                    and _pty_executor.shutdown(wait=False, cancel_futures=True))
    return _pty_executor


def reset_pty_executor_for_tests() -> None:
    """Test-only: drop the pool so a fresh one is built on next use."""
    global _pty_executor
    with _pty_executor_lock:
        pool, _pty_executor = _pty_executor, None
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)
