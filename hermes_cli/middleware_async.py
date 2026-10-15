"""Run the synchronous execution contract around an event-loop-owned provider."""

from __future__ import annotations

import asyncio
from concurrent.futures import CancelledError, Future
from threading import Lock
from typing import Any, Awaitable, Callable

from hermes_cli.middleware import LLM_EXECUTION_MIDDLEWARE, run_llm_execution_middleware


class _LoopProviderCall:
    """Keep async clients on their owning loop while next_call returns a concrete response."""

    def __init__(self, loop: asyncio.AbstractEventLoop, call: Callable[[dict], Awaitable[Any]]):
        self.loop = loop
        self.call = call
        self.lock = Lock()
        self.cancelled = False
        self.pending: Future | None = None

    async def _invoke(self, request: dict) -> Any:
        return await self.call(request)

    def __call__(self, request: dict) -> Any:
        with self.lock:
            if self.cancelled:
                raise asyncio.CancelledError
            pending = asyncio.run_coroutine_threadsafe(self._invoke(request), self.loop)
            self.pending = pending
        try:
            return pending.result()
        except CancelledError:
            raise asyncio.CancelledError from None
        finally:
            with self.lock:
                self.pending = None

    def cancel(self) -> None:
        with self.lock:
            self.cancelled = True
            pending = self.pending
        if pending is not None:
            pending.cancel()


async def run_llm_execution_middleware_async(
    request: dict[str, Any], next_call: Callable[[dict[str, Any]], Awaitable[Any]],
    **context: Any,
) -> Any:
    """Reuse single-use/fail-open middleware without passing plugins an unawaited coroutine.

    Only the synchronous middleware chain runs in a worker. ContextVars follow it; provider
    dispatch and response handling stay on the caller's loop. Cancellation also cancels an
    outstanding provider future and prevents a still-running plugin from dispatching later.
    """
    from hermes_cli.plugins import has_middleware

    if not has_middleware(LLM_EXECUTION_MIDDLEWARE):
        return await next_call(request)
    provider = _LoopProviderCall(asyncio.get_running_loop(), next_call)
    try:
        return await asyncio.to_thread(run_llm_execution_middleware, request, provider, **context)
    except asyncio.CancelledError:
        provider.cancel()
        raise
