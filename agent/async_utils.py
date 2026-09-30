"""Async/sync bridging helpers.

``asyncio.run_coroutine_threadsafe`` can raise ``RuntimeError`` (loop closed during a
shutdown race); the coroutine is then never awaited or closed, which triggers a
"coroutine was never awaited" RuntimeWarning and leaks its frame. The helpers here
close the coroutine on scheduling failure. ``future.result()`` failures are deliberately
NOT handled: once the loop accepts the coroutine its lifecycle belongs to the loop.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import Future
from typing import Any, Coroutine, Optional


_DEFAULT_LOGGER = logging.getLogger(__name__)


def safe_schedule_threadsafe(
    coro: Coroutine[Any, Any, Any], loop: Optional[asyncio.AbstractEventLoop], *,
    logger: Optional[logging.Logger] = None,
    log_message: str = "Failed to schedule coroutine on loop", log_level: int = logging.DEBUG,
) -> Optional[Future]:
    """Schedule ``coro`` on ``loop`` from a sync context, leak-safe.

    Returns the Future on success, or ``None`` if the loop is missing or scheduling
    raised; in every failure path the coroutine is closed. Callers keep full control
    over the returned future (``.result(timeout=...)``, callbacks, fire-and-forget).
    """
    log = logger if logger is not None else _DEFAULT_LOGGER
    try:
        if loop is None:
            raise RuntimeError("loop is None")
        return asyncio.run_coroutine_threadsafe(coro, loop)
    except Exception as exc:
        if asyncio.iscoroutine(coro):
            coro.close()
        log.log(log_level, "%s: %s", log_message, exc)
        return None


class WithdrawableDispatch:
    """A coroutine scheduled on another thread's loop, which the caller can withdraw until the
    loop starts it.

    A ``run_coroutine_threadsafe`` future stays pending until its coroutine finishes, so
    ``Future.cancel()`` returns True even while the coroutine is part-way through an await, and
    then interrupts it. The return value therefore cannot tell the caller whether the coroutine
    started. Here the loop claims the dispatch before it runs the coroutine, and ``withdraw()``
    succeeds only if the loop has not made that claim. Exactly one of the two succeeds.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = "pending"
        self._future: Optional[Future] = None

    @classmethod
    def schedule(
        cls, coro: Coroutine[Any, Any, Any], loop: Optional[asyncio.AbstractEventLoop], **kwargs: Any,
    ) -> Optional["WithdrawableDispatch"]:
        """Schedule ``coro`` as ``safe_schedule_threadsafe`` does; None (``coro`` closed) on failure."""
        dispatch = cls()
        dispatch._future = safe_schedule_threadsafe(dispatch._run(coro), loop, **kwargs)
        if dispatch._future is None:
            coro.close()
            return None
        return dispatch

    def result(self, timeout: Optional[float] = None) -> Any:
        assert self._future is not None
        return self._future.result(timeout=timeout)

    def withdraw(self) -> bool:
        """Stop the coroutine from ever running. False if the loop has already started it."""
        return self._leave_pending("withdrawn")

    def _leave_pending(self, state: str) -> bool:
        with self._lock:
            if self._state != "pending":
                return False
            self._state = state
            return True

    async def _run(self, coro: Coroutine[Any, Any, Any]) -> Any:
        if not self._leave_pending("started"):
            coro.close()
            return None
        return await coro


def consume_detached_task_result(task: asyncio.Future[Any]) -> None:
    """``add_done_callback`` for cancelled-and-detached tasks: observe the exception so the
    loop does not log "exception was never retrieved"; cancellation and terminal errors
    are swallowed because the task's owner already gave up on it."""
    try:
        task.exception()
    except (asyncio.CancelledError, Exception):
        pass
