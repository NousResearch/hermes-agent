"""Keep native stream resources on one loop, even inside a running caller loop."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import threading
from concurrent.futures import Future
from typing import AsyncIterator, Awaitable, Callable, cast


class _SyncFromAsyncIterator:
    """One consumer pulls; its owner must close the iterator on early exit."""

    def __init__(self, source: object) -> None:
        self._source = source
        self._loop = asyncio.new_event_loop()
        self._pending: Future[object] | None = None
        self._pull_task: asyncio.Task[object] | None = None
        self._closed = False
        ready = threading.Event()

        def run() -> None:
            asyncio.set_event_loop(self._loop)
            self._loop.call_soon(ready.set)
            try:
                self._loop.run_forever()
            finally:
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                if pending:
                    self._loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
                self._loop.close()

        self._thread = threading.Thread(
            target=run, name="moa-aggregator-async-stream", daemon=True
        )
        self._thread.start()
        ready.wait()

    def resolve(self) -> object:
        """Creation must share the loop used for reads and cleanup."""

        async def open_source() -> object:
            if inspect.isawaitable(self._source):
                return await cast(Awaitable[object], self._source)
            return self._source

        try:
            self._source = asyncio.run_coroutine_threadsafe(
                open_source(), self._loop
            ).result()
            return self._source
        except BaseException:
            self.close()
            raise

    def __iter__(self) -> "_SyncFromAsyncIterator":
        return self

    def __next__(self) -> object:
        if self._closed:
            raise StopIteration

        async def pull() -> object:
            self._pull_task = asyncio.current_task()
            try:
                return await anext(cast(AsyncIterator[object], self._source))
            finally:
                self._pull_task = None

        self._pending = asyncio.run_coroutine_threadsafe(pull(), self._loop)
        try:
            return self._pending.result()
        except StopAsyncIteration:
            self.close()
            raise StopIteration from None
        except BaseException:
            with contextlib.suppress(Exception):
                self.close()
            raise
        finally:
            self._pending = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._pending is not None and not self._pending.done():
            self._pending.cancel()

        async def release() -> None:
            try:
                # Future cancellation wakes the consumer before the async task's
                # finally completes; await it before closing the source.
                task = self._pull_task
                if task is not None:
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
            finally:
                closer = getattr(self._source, "aclose", None)
                if not callable(closer):
                    closer = getattr(self._source, "close", None)
                if callable(closer):
                    result = cast(Callable[[], object], closer)()
                    if inspect.isawaitable(result):
                        await cast(Awaitable[object], result)

        try:
            asyncio.run_coroutine_threadsafe(release(), self._loop).result(timeout=5.0)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                raise RuntimeError("MoA async stream worker did not stop after close")


def coerce_sync_stream(result: object) -> object:
    """Keep one owning loop without replaying the provider request."""
    if not inspect.isawaitable(result) and not (
        hasattr(result, "__aiter__") and not hasattr(result, "__iter__")
    ):
        return result
    bridge = _SyncFromAsyncIterator(result)
    result = bridge.resolve()
    if hasattr(result, "__aiter__") and not hasattr(result, "__iter__"):
        return bridge
    bridge.close()
    return result
