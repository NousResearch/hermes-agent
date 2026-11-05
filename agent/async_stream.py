"""Keep native stream resources on one loop, even inside a running caller loop."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import threading
from collections.abc import Iterable, Iterator
from concurrent.futures import CancelledError, Future
from typing import AsyncIterable, AsyncIterator, Awaitable, Callable, cast

from agent.async_utils import safe_schedule_threadsafe
from agent.memory_provider import spawn_context_thread


async def _close_async_resource(resource: object) -> None:
    closer = getattr(resource, "aclose", None)
    if not callable(closer):
        closer = getattr(resource, "close", None)
    if callable(closer):
        result = cast(Callable[[], object], closer)()
        if inspect.isawaitable(result):
            await cast(Awaitable[object], result)


class _SyncFromAsyncIterator:
    """One consumer pulls; its owner must close the iterator on early exit."""

    def __init__(self, source: object) -> None:
        self._source: object = source
        self._iterator: AsyncIterator[object] | None = None
        self._loop = asyncio.new_event_loop()
        self._pending: Future[object] | None = None
        self._pull_task: asyncio.Task[object] | None = None
        self._lock = threading.Lock()
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

        self._thread = spawn_context_thread(
            target=run, name="moa-aggregator-async-stream", daemon=True
        )
        self._thread.start()
        ready.wait()

    def resolve(self) -> object:
        """Creation must share the loop used for reads and cleanup."""

        async def open_source() -> object:
            if inspect.isawaitable(self._source):
                self._source = await cast(Awaitable[object], self._source)
            if hasattr(self._source, "__aiter__") and not hasattr(self._source, "__iter__"):
                self._iterator = aiter(cast(AsyncIterable[object], self._source))
            return self._source

        try:
            future = cast(Future[object] | None, safe_schedule_threadsafe(open_source(), self._loop))
            if future is None:
                raise RuntimeError("MoA async stream loop is unavailable")
            self._source = future.result()
            return self._source
        except BaseException:
            self.close()
            raise

    def __iter__(self) -> _SyncFromAsyncIterator:
        return self

    def __next__(self) -> object:
        async def pull() -> object:
            self._pull_task = asyncio.current_task()
            try:
                return await anext(cast(AsyncIterator[object], self._iterator))
            finally:
                self._pull_task = None

        # Closing cannot stop the loop between the check and scheduling a read.
        with self._lock:
            if self._closed:
                raise StopIteration
            pending = cast(Future[object] | None, safe_schedule_threadsafe(pull(), self._loop))
            if pending is None:
                raise RuntimeError("MoA async stream loop is unavailable")
            self._pending = pending
        try:
            result = pending.result()
            if self._closed:
                raise StopIteration
            return result
        except CancelledError:
            if self._closed:
                raise StopIteration from None
            self.close()
            raise
        except StopAsyncIteration:
            self.close()
            raise StopIteration from None
        except BaseException:
            with contextlib.suppress(Exception):
                self.close()
            raise
        finally:
            with self._lock:
                self._pending = None

    def detach_source(self) -> object:
        """Hand a synchronous result back without closing its resources."""
        result, self._source = self._source, None
        self.close()
        return result

    def close(self) -> None:
        with self._lock:
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
                try:
                    await _close_async_resource(self._iterator)
                finally:
                    if self._source is not self._iterator:
                        await _close_async_resource(self._source)

        try:
            future = cast(Future[object] | None, safe_schedule_threadsafe(release(), self._loop))
            if future is not None:
                future.result(timeout=5.0)
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
    return bridge.detach_source()


class _SemaphoreStream(Iterator[object]):
    """Own a stream's permit, including teardown concurrent with Relay read-ahead."""

    def __init__(self, stream: Iterable[object], semaphore: threading.BoundedSemaphore) -> None:
        self._source = stream
        self._iterator: Iterator[object] | None = None
        self._semaphore: threading.BoundedSemaphore | None = semaphore
        self._lock = threading.Lock()
        self._reading = False
        self._closed = False

    def __iter__(self) -> _SemaphoreStream:
        return self

    def __next__(self) -> object:
        with self._lock:
            if self._closed:
                raise StopIteration
            self._reading = True
        try:
            if self._iterator is None:
                self._iterator = iter(self._source)
            result = next(self._iterator)
        except BaseException:
            self.close()
            raise
        finally:
            with self._lock:
                self._reading = False
            if self._closed:
                self.close()
        if self._closed:
            raise StopIteration
        return result

    def close(self) -> None:
        with self._lock:
            self._closed = True
            # The async bridge can cancel a pull. Other sync iterators (notably
            # generators) must finish their active read before they can close.
            if self._reading and not isinstance(self._source, _SyncFromAsyncIterator):
                return
            semaphore, self._semaphore = self._semaphore, None
        if semaphore is None:
            return
        try:
            resources = {id(r): r for r in (self._source, self._iterator) if r is not None}
            with contextlib.ExitStack() as cleanup:
                for resource in resources.values():
                    close = getattr(resource, "close", None)
                    if callable(close):
                        cleanup.callback(close)
        finally:
            semaphore.release()


def _release_sync_semaphore_after_stream(
    stream: Iterable[object], semaphore: threading.BoundedSemaphore,
) -> _SemaphoreStream:
    """Release a permit only after a streaming response is consumed or closed."""
    return _SemaphoreStream(stream, semaphore)
