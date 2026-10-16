"""Regression #130132: stream conversion preserves ownership during teardown."""

import asyncio
import contextvars
import threading
from collections.abc import AsyncIterator, Generator, Iterator
from types import SimpleNamespace
from typing import cast

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI

from agent.async_stream import (
    _SyncFromAsyncIterator,
    _release_sync_semaphore_after_stream,
    coerce_sync_stream,
)
from agent.auxiliary_client import _AsyncCompletionsAdapter, _relay_sync_stream


@pytest.mark.parametrize("native_async", [True, False], ids=["async", "sync"])
def test_close_during_read_releases_source_and_permit(native_async: bool) -> None:
    entered = threading.Event()
    resume = threading.Event()
    closed = threading.Event()
    owners: list[asyncio.AbstractEventLoop] = []
    received: list[object] = []
    failures: list[BaseException] = []

    async def native_source() -> AsyncIterator[str]:
        owners.append(asyncio.get_running_loop())
        try:
            entered.set()
            await asyncio.Event().wait()
            yield "chunk"
        finally:
            # Cancellation must finish before aclose and worker shutdown.
            await asyncio.sleep(0)
            closed.set()

    def sync_source() -> Generator[str, None, None]:
        try:
            entered.set()
            assert resume.wait(5)
            yield "chunk"
        finally:
            closed.set()

    source = (
        cast(_SyncFromAsyncIterator, coerce_sync_stream(native_source()))
        if native_async else sync_source()
    )
    semaphore = threading.BoundedSemaphore(1)
    semaphore.acquire()
    wrapped = _release_sync_semaphore_after_stream(source, semaphore)

    def consume() -> None:
        try:
            received.append(next(wrapped))
        except StopIteration:
            pass
        except BaseException as exc:
            failures.append(exc)

    reader = threading.Thread(target=consume, daemon=True)
    reader.start()
    try:
        assert entered.wait(5)
        wrapped.close()
        if native_async:
            assert closed.is_set()
            assert owners[0].is_closed()
        else:
            # A sync generator cannot be interrupted; its permit stays owned
            # until the in-flight read finishes and deferred cleanup completes.
            assert not semaphore.acquire(blocking=False)
        resume.set()
        reader.join(5)
        assert not reader.is_alive()
        assert closed.is_set()
        assert not failures
        assert received == []
        wrapped.close()
        assert semaphore.acquire(blocking=False)
        assert not semaphore.acquire(blocking=False)
        semaphore.release()
        with pytest.raises(ValueError):
            semaphore.release()
    finally:
        resume.set()
        if native_async:
            source.close()
        reader.join(5)
        wrapped.close()


@pytest.mark.parametrize("native_async", [False, True], ids=["sync-sdk", "async-sdk"])
@pytest.mark.parametrize("consumption", ["unread", "partial", "complete"])
def test_sdk_stream_stays_owned_by_consumer(native_async: bool, consumption: str) -> None:
    requests: list[httpx.Request] = []
    responses: list[httpx.Response] = []
    events: list[str] = []
    owners: list[asyncio.AbstractEventLoop] = []

    class Body(httpx.SyncByteStream, httpx.AsyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            events.append("read")
            yield (
                b'data: {"id":"chunk","object":"chat.completion.chunk","created":0,'
                b'"model":"offline","choices":[{"index":0,"delta":{"content":"hello"},'
                b'"finish_reason":null}]}\n\n'
            )

        def close(self) -> None:
            if native_async:
                assert asyncio.get_running_loop() is owners[0]
            events.append("close")

        async def __aiter__(self) -> AsyncIterator[bytes]:
            for chunk in self:
                yield chunk

        async def aclose(self) -> None:
            self.close()

    def transport(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if native_async:
            owners.append(asyncio.get_running_loop())
        response = httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=Body()
        )
        responses.append(response)
        return response

    if native_async:
        sdk = AsyncOpenAI(
            api_key="offline-test-key",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)),
        )
        client = sdk
    else:
        sdk = OpenAI(
            api_key="offline-test-key",
            http_client=httpx.Client(transport=httpx.MockTransport(transport)),
        )
        client = SimpleNamespace(chat=SimpleNamespace(
            completions=_AsyncCompletionsAdapter(sdk.chat.completions)
        ))
    try:
        stream = _relay_sync_stream(client, {
            "model": "offline", "messages": [{"role": "user", "content": "hi"}], "stream": True,
        })
        try:
            assert len(requests) == 1
            assert not responses[0].is_closed
            assert events == []
            if consumption == "complete":
                assert [chunk.choices[0].delta.content for chunk in stream] == ["hello"]
            elif consumption == "partial":
                assert next(iter(stream)).choices[0].delta.content == "hello"
        finally:
            stream.close()
        assert responses[0].is_closed
        assert events == (["close"] if consumption == "unread" else ["read", "close"])
        assert not sdk.is_closed()
        if native_async:
            assert owners[0].is_closed()
    finally:
        if isinstance(sdk, AsyncOpenAI):
            asyncio.run(sdk.close())
        else:
            sdk.close()


@pytest.mark.parametrize("consume", [False, True], ids=["unread", "partial"])
def test_async_iterable_closes_iterator_on_owning_loop(consume: bool) -> None:
    scope = contextvars.ContextVar("stream-test-scope", default="unbound")
    events: list[str] = []

    class Source:
        def __init__(self) -> None:
            self.owner = asyncio.get_running_loop()
            self.record("create")

        def record(self, event: str) -> None:
            assert asyncio.get_running_loop() is self.owner
            assert scope.get() == "bound"
            events.append(event)

        def __aiter__(self) -> AsyncIterator[str]:
            self.record("iterate")

            async def chunks() -> AsyncIterator[str]:
                try:
                    self.record("read")
                    yield "chunk"
                finally:
                    await asyncio.sleep(0)
                    self.record("iterator-close")

            return chunks()

        async def aclose(self) -> None:
            self.record("source-close")

    async def caller() -> None:
        scope.set("bound")
        opened: list[Source] = []

        async def create() -> Source:
            source = Source()
            opened.append(source)
            return source

        bridge = cast(_SyncFromAsyncIterator, coerce_sync_stream(create()))
        try:
            assert events == ["create", "iterate"]
            if consume:
                assert next(bridge) == "chunk"
        finally:
            bridge.close()
        assert opened[0].owner.is_closed()
        assert events == ["create", "iterate", *(
            ["read", "iterator-close"] if consume else []
        ), "source-close"]

    asyncio.run(caller())


def test_iterator_close_failure_still_closes_source_and_releases_permit() -> None:
    events: list[str] = []

    class Source:
        def __iter__(self) -> Generator[str, None, None]:
            try:
                yield "chunk"
            finally:
                events.append("iterator-close")
                raise RuntimeError("iterator cleanup failed")

        def close(self) -> None:
            events.append("source-close")

    semaphore = threading.BoundedSemaphore(1)
    semaphore.acquire()
    stream = _release_sync_semaphore_after_stream(Source(), semaphore)
    assert next(stream) == "chunk"
    with pytest.raises(RuntimeError, match="iterator cleanup failed"):
        stream.close()
    assert events == ["iterator-close", "source-close"]
    stream.close()
    assert semaphore.acquire(blocking=False)
    assert not semaphore.acquire(blocking=False)
    semaphore.release()
