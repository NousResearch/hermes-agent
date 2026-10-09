"""Async/sync bridging helpers.

The codebase has ~30 sites that schedule a coroutine onto an event loop from a
worker thread via :func:`asyncio.run_coroutine_threadsafe`.  That function can
raise :class:`RuntimeError` (e.g. the loop was closed during a shutdown race),
and when it does the coroutine object is never awaited and never closed —
which triggers a ``"coroutine '<name>' was never awaited"`` RuntimeWarning and
leaks the coroutine's frame until GC.

:func:`safe_schedule_threadsafe` wraps the call, closes the coroutine on
scheduling failure, and returns ``None`` (instead of a half-formed future) so
callers can branch cleanly:

    fut = safe_schedule_threadsafe(coro, loop)
    if fut is None:
        return  # or fallback behavior
    fut.result(timeout=5)

The helper deliberately does NOT also handle ``future.result()`` failures —
that is a separate concern.  Once the loop has accepted the coroutine, its
lifecycle belongs to the loop, not the scheduling thread.
"""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import Future
from typing import Any, Coroutine, Optional


_DEFAULT_LOGGER = logging.getLogger(__name__)


def safe_schedule_threadsafe(
    coro: Coroutine[Any, Any, Any],
    loop: Optional[asyncio.AbstractEventLoop],
    *,
    logger: Optional[logging.Logger] = None,
    log_message: str = "Failed to schedule coroutine on loop",
    log_level: int = logging.DEBUG,
) -> Optional[Future]:
    """Schedule ``coro`` on ``loop`` from a sync context, leak-safe.

    Returns the :class:`concurrent.futures.Future` on success, or ``None`` if
    the loop is missing or :func:`asyncio.run_coroutine_threadsafe` raised
    (e.g. the loop was closed during a shutdown race).  In all failure paths
    the coroutine is :meth:`close`-d so it does not trigger
    ``"coroutine was never awaited"`` warnings or leak its frame.

    Callers retain full control over what to do with the returned future
    (call ``.result(timeout=...)``, attach ``add_done_callback``, ignore it
    fire-and-forget, etc.).
    """
    log = logger if logger is not None else _DEFAULT_LOGGER

    if loop is None:
        if asyncio.iscoroutine(coro):
            coro.close()
        log.log(log_level, "%s: loop is None", log_message)
        return None

    try:
        return asyncio.run_coroutine_threadsafe(coro, loop)
    except Exception as exc:
        if asyncio.iscoroutine(coro):
            coro.close()
        log.log(log_level, "%s: %s", log_message, exc)
        return None


def consume_detached_task_result(task: "asyncio.Future[Any]") -> None:
    """Retrieve a detached task's result without surfacing cancellation.

    Used as an ``add_done_callback`` on tasks that were cancelled and
    detached (e.g. an adapter close path that swallows ``CancelledError``
    past its teardown deadline). Observing ``task.exception()`` prevents
    "exception was never retrieved" noise on the event loop; cancellation
    and any terminal error are deliberately swallowed — the task's owner
    already gave up on it.
    """
    try:
        task.exception()
    except (asyncio.CancelledError, Exception):
        pass
