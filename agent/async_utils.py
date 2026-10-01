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
from enum import Enum
from typing import Any, Callable, Coroutine, Optional


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


class _DispatchState(Enum):
    PENDING = "pending"
    STARTED = "started"
    WITHDRAWN = "withdrawn"


class _DispatchClaim:
    def __init__(self, coro: Coroutine[Any, Any, Any]) -> None:
        self._coro = coro
        self._lock = threading.Lock()
        self._state = _DispatchState.PENDING

    def withdraw(self) -> bool:
        with self._lock:
            if self._state is not _DispatchState.PENDING:
                return False
            self._state = _DispatchState.WITHDRAWN
            self._coro.close()
            return True

    def close_if_cancelled(self, future: Future) -> None:
        if future.cancelled():
            self.withdraw()

    async def run(self) -> Any:
        with self._lock:
            if self._state is not _DispatchState.PENDING:
                return None
            self._state = _DispatchState.STARTED
        return await self._coro


class WithdrawableDispatch:
    """A cross-thread coroutine dispatch that can be withdrawn until execution starts.

    A ``run_coroutine_threadsafe`` future remains pending while its coroutine runs,
    so ``Future.cancel()`` can interrupt a send that has already started. A separate
    claim makes execution and withdrawal mutually exclusive. Withdrawal closes the
    inner coroutine immediately, even if the loop never runs the wrapper.
    """

    def __init__(self, future: Future, claim: _DispatchClaim) -> None:
        self._future = future
        self._claim = claim

    @classmethod
    def schedule(
        cls, coro: Coroutine[Any, Any, Any], loop: Optional[asyncio.AbstractEventLoop], **kwargs: Any,
    ) -> Optional["WithdrawableDispatch"]:
        """Schedule the coroutine, or close it and return None if scheduling fails."""
        claim = _DispatchClaim(coro)
        future = safe_schedule_threadsafe(claim.run(), loop, **kwargs)
        if future is None:
            claim.withdraw()
            return None
        future.add_done_callback(claim.close_if_cancelled)
        return cls(future, claim)

    def result(self, timeout: Optional[float] = None) -> Any:
        return self._future.result(timeout=timeout)

    def add_done_callback(self, callback: Callable[[Future], Any]) -> None:
        """Observe the eventual result without cancelling a started dispatch."""
        self._future.add_done_callback(callback)

    def withdraw(self) -> bool:
        """Close an unstarted coroutine; return False after execution has started."""
        return self._claim.withdraw()


def consume_detached_task_result(task: "asyncio.Future[Any]") -> None:
    """``add_done_callback`` for cancelled-and-detached tasks: observe the exception so the
    loop does not log "exception was never retrieved"; cancellation and terminal errors
    are swallowed because the task's owner already gave up on it."""
    try:
        task.exception()
    except (asyncio.CancelledError, Exception):
        pass
