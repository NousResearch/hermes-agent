"""Regression #130132: consume native MoA async results through the sync facade.

The consumer owns each returned stream and closes it on early exit. One provider
request must yield lazy, ordered chunks with caller context and loop-affine cleanup;
synchronous and nonstream results retain their existing behavior.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
from collections.abc import Iterator
from concurrent.futures import CancelledError, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDelta

from agent import moa_loop
from run_agent import AIAgent


@pytest.mark.parametrize("chunks_to_consume", [0, 1])
def test_streaming_call_closes_abandoned_aggregator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, chunks_to_consume: int
) -> None:
    """#130132: production teardown releases the stream and its concurrency permit.

    Do not close the bridge in the test: abandon consumption and let _call's
    finally reach it through the real MoA, auxiliary and Relay stream owners.
    """
    from agent import auxiliary_client, relay_runtime
    from agent import chat_completion_helpers as helpers

    def abandon(stream: Iterator[object], **_kwargs: object) -> Iterator[object]:
        for _ in range(chunks_to_consume):
            chunk = next(stream)
            delivered.append(chunk)
            yield chunk
        raise KeyboardInterrupt("abandoned aggregator")

    monkeypatch.setattr(helpers, "_iter_provider_stream_chunks", abandon)
    for managed in (False, True):
        delivered: list[object] = []
        home = tmp_path / str(managed)
        home.mkdir()
        (home / "config.yaml").write_text(
            "providers:\n  native-test:\n    base_url: http://127.0.0.1:1/v1\n"
            "    api_key: test-key\n"
            "auxiliary:\n  moa_aggregator:\n    max_concurrency: 1\n"
            "moa:\n  presets:\n    cleanup:\n      enabled: false\n"
            "      aggregator:\n        provider: native-test\n        model: native-actor\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(home))
        opened: list[_AsyncStream] = []

        async def create(**_kwargs: object) -> _AsyncStream:
            source = _AsyncStream([
                ChatCompletionChunk(
                    id="chunk",
                    object="chat.completion.chunk",
                    created=0,
                    model="native-actor",
                    choices=[
                        Choice(
                            index=0, delta=ChoiceDelta(content=text), finish_reason=None
                        )
                    ],
                )
                for text in ("first", "second", "third")
            ])
            opened.append(source)
            return source

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        monkeypatch.setattr(
            auxiliary_client,
            "_get_cached_client",
            lambda *a, **k: (client, "native-actor"),
        )
        agent = AIAgent(
            provider="moa",
            model="cleanup",
            api_key="test-key",
            base_url="moa://local",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            enabled_toolsets=[],
            max_iterations=1,
            session_id="moa-cleanup",
        )
        relay_runtime._reset_for_tests()
        lease = turn = None
        consumer = "test.moa-aggregator-cleanup"
        if managed:
            lease = relay_runtime.SESSION_COORDINATOR.acquire_conversation(
                profile_key=relay_runtime.current_profile_key(),
                session_id="moa-cleanup",
                platform="cli",
            )
            turn = relay_runtime.SESSION_COORDINATOR.begin_turn(
                lease, turn_id="turn", task_id="task"
            )
            lease.host.retain_managed_execution(consumer)
        try:
            workers_before = set(threading.enumerate())
            call = helpers._StreamingCall(
                agent,
                {"model": "cleanup", "messages": [{"role": "user", "content": "hi"}]},
                None,
            )
            with pytest.raises(KeyboardInterrupt, match="abandoned aggregator"):
                call._call()

            assert len(opened) == 1
            assert len(delivered) == chunks_to_consume
            if not managed:  # Relay may read ahead; the direct path must stay lazy.
                assert opened[0]._index == chunks_to_consume
            leaked = [
                thread
                for thread in set(threading.enumerate()) - workers_before
                if thread.name == "moa-aggregator-async-stream"
            ]
            assert (all(source.closed for source in opened), leaked) == (True, [])
            semaphore = auxiliary_client._acquire_sync_aux_semaphore("moa_aggregator")
            assert semaphore is not None
            assert semaphore.acquire(blocking=False), (
                "abandonment leaked the concurrency permit"
            )
            semaphore.release()
        finally:
            if lease is not None and turn is not None:
                lease.host.release_managed_execution(consumer)
                relay_runtime.SESSION_COORDINATOR.end_turn(turn, outcome="cancelled")
                relay_runtime.SESSION_COORDINATOR.release_conversation(lease)
            relay_runtime._reset_for_tests()


def _chunk(text):
    return SimpleNamespace(text=text)


def _completed(content="aggregator acted"):
    message = SimpleNamespace(content=content, tool_calls=[])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None, model="claude-native")


def _coro(value):
    """A fresh awaitable resolving to *value* (one per provider dispatch)."""

    async def _resolve():
        return value

    return _resolve()


def _texts(chunks):
    """Delta content from a mix of wrapped completed responses and raw token chunks."""
    out = []
    for chunk in chunks:
        choices = getattr(chunk, "choices", None)
        out.append(choices[0].delta.content if choices else chunk.text)
    return out


class _AsyncStream:
    """Async token stream double that records consumption, close, and context visibility."""

    def __init__(self, items, *, probe=None):
        self._items = list(items)
        self._index = 0
        self.closed = False
        self.seen_context = []
        self._probe = probe

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._probe is not None:
            self.seen_context.append(self._probe.get(None))
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item

    async def aclose(self):
        self.closed = True


@pytest.fixture
def facade(monkeypatch):
    """A facade whose aggregator slot resolves without touching real config."""
    monkeypatch.setattr(
        moa_loop,
        "_slot_runtime",
        lambda slot: {
            "provider": slot["provider"],
            "model": slot["model"],
            "api_mode": "chat_completions",
        },
    )
    f = moa_loop.MoAChatCompletions.__new__(moa_loop.MoAChatCompletions)
    f._pending_trace = None
    f._agent = None
    f._privacy_mode = ""
    return f


def _open(monkeypatch, on_call, facade, *, stream=True, tools=()):
    """Wire ``moa_loop.call_llm`` to *on_call* and open the prepared aggregator stream."""
    calls = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        return on_call(kwargs)

    monkeypatch.setattr(moa_loop, "call_llm", fake_call_llm)
    prepared = {
        "messages": [{"role": "user", "content": "q"}],
        "guidance": None,
        "aggregator": {"provider": "anthropic", "model": "claude-native"},
        "aggregator_temperature": None,
    }
    return facade.create(
        _moa_prepared_request=prepared, stream=stream, tools=list(tools)
    ), calls


def test_stream_create_adapts_native_async_results_and_passes_sync_shapes(
    monkeypatch, facade
):
    """Every shape ``call_llm(stream=True)`` can return reaches the synchronous consumer
    correctly, with exactly one provider dispatch (regression #130132)."""
    ordered = _AsyncStream([_chunk("a"), _chunk("b"), _chunk("c")])
    direct = _AsyncStream([_chunk("only")])
    sync_sentinel = iter([_chunk("x")])
    nonstream_response = _completed("raw")

    def lazy_ordered(stream, _calls):
        # Laziness: one pull reaches the source exactly once, in order, before the rest.
        first = next(stream)
        pulled_one = ordered._index == 1
        rest = list(stream)
        return (
            pulled_one and _texts([first, *rest]) == ["a", "b", "c"] and ordered.closed
        )

    cases = [
        (
            "awaitable resolving to a completed response",
            lambda: _coro(_completed("aggregator acted")),
            True,
            lambda stream, _calls: _texts(stream) == ["aggregator acted"],
        ),
        (
            "awaitable resolving to an async iterator (lazy, ordered, closed)",
            lambda: _coro(ordered),
            True,
            lazy_ordered,
        ),
        (
            "async iterator returned directly",
            lambda: direct,
            True,
            lambda stream, _calls: _texts(stream) == ["only"] and direct.closed,
        ),
        (
            "plain synchronous iterator is returned unchanged",
            lambda: sync_sentinel,
            True,
            lambda stream, _calls: stream is sync_sentinel,
        ),
        (
            "completed response with stream=True becomes one delta chunk",
            lambda: _completed("done"),
            True,
            lambda stream, _calls: _texts(stream) == ["done"],
        ),
        (
            "non-streaming call returns the raw response",
            lambda: nonstream_response,
            False,
            lambda stream, _calls: stream is nonstream_response,
        ),
    ]

    for name, producer, stream_flag, check in cases:
        stream, calls = _open(
            monkeypatch, lambda _kw, p=producer: p(), facade, stream=stream_flag
        )
        assert check(stream, calls), name
        assert len(calls) == 1, name
        assert bool(calls[0].get("stream")) == stream_flag, name


def test_stream_create_owns_one_loop_with_caller_context_and_deterministic_close(
    monkeypatch, facade
):
    """Creation, reads, and cleanup share one owning loop; async steps inherit the caller's
    contextvars; exhaustion and early close both release the source; and opening or consuming
    works while a caller event loop is already running (regression #130132)."""
    probe = contextvars.ContextVar("moa_aggregator_probe")
    opened = []

    class LoopBoundStream:
        """A real client can bind sockets/tasks to the loop that opened its stream."""

        def __init__(self):
            self.owner = asyncio.get_running_loop()
            self.creation_context = probe.get(None)
            self.closed = False
            self.delivered = False
            self.steps = []

        def __aiter__(self):
            return self

        async def __anext__(self):
            assert asyncio.get_running_loop() is self.owner
            self.steps.append(probe.get(None))
            if self.delivered:
                raise StopAsyncIteration
            self.delivered = True
            return _chunk("loop-bound")

        async def aclose(self):
            assert asyncio.get_running_loop() is self.owner
            self.closed = True

    async def open_stream():
        source = LoopBoundStream()
        opened.append(source)
        return source

    # Exhaustion: StopAsyncIteration closes the source and shuts the owning loop down.
    token = probe.set("turn-context")
    try:
        stream, calls = _open(monkeypatch, lambda _kw: open_stream(), facade)
        chunks = list(stream)
    finally:
        probe.reset(token)
    source = opened[-1]
    assert _texts(chunks) == ["loop-bound"]
    assert len(calls) == 1
    assert source.steps and set(source.steps) == {"turn-context"}
    assert source.creation_context == "turn-context"
    assert source.closed
    assert source.owner.is_closed()

    # Early close: deterministic cleanup without ever reaching StopAsyncIteration.
    early = _AsyncStream([_chunk("a"), _chunk("b"), _chunk("c")])
    stream, _calls = _open(monkeypatch, lambda _kw: early, facade)
    assert next(stream).text == "a"
    stream.close()
    assert early.closed is True
    assert next(stream, None) is None

    # Real OpenAI SDK streams own an HTTP response and expose async close, not aclose.
    async def close_sdk_stream():
        async with AsyncOpenAI(api_key="test-key") as client:
            for consume_first in (False, True):
                response = httpx.Response(
                    200,
                    request=httpx.Request("POST", "https://example.invalid"),
                    stream=httpx.ByteStream(
                        b'data: {"id":"chunk","object":"chat.completion.chunk",'
                        b'"created":0,"model":"test","choices":[{"index":0,'
                        b'"delta":{"content":"sdk"},"finish_reason":null}]}\n\n'
                    ),
                )
                source = AsyncStream(
                    cast_to=ChatCompletionChunk, response=response, client=client
                )
                stream, calls = _open(monkeypatch, lambda _kw: _coro(source), facade)
                try:
                    if consume_first:
                        assert next(stream).choices[0].delta.content == "sdk"
                    stream.close()
                    assert response.is_closed
                    assert not client.is_closed()
                    assert len(calls) == 1
                finally:
                    stream.close()
                    await response.aclose()

    asyncio.run(close_sdk_stream())

    # A caller event loop may already be running when the facade opens or is consumed.
    outside = _AsyncStream([_chunk("outside-loop")])
    pending, _calls = _open(monkeypatch, lambda _kw: _coro(outside), facade)

    async def consume():
        return _texts(list(pending))

    assert asyncio.run(consume()) == ["outside-loop"]
    assert outside.closed is True

    async def open_and_consume():
        live, live_calls = _open(
            monkeypatch, lambda _kw: _coro(_AsyncStream([_chunk("live")])), facade
        )
        return _texts(list(live)), len(live_calls)

    assert asyncio.run(open_and_consume()) == (["live"], 1)

    # Provider failures remain visible while releasing their owning resources.
    failed_loops = []

    async def fail_open():
        failed_loops.append(asyncio.get_running_loop())
        raise ValueError("provider open failed")

    with pytest.raises(ValueError, match="provider open failed"):
        _open(monkeypatch, lambda _kw: fail_open(), facade)
    assert failed_loops[0].is_closed()

    class FailingStream(_AsyncStream):
        async def __anext__(self):
            raise ValueError("provider read failed")

    failing = FailingStream([])
    stream, _calls = _open(monkeypatch, lambda _kw: failing, facade)
    with pytest.raises(ValueError, match="provider read failed"):
        next(stream)
    assert failing.closed

    # Close from another thread cancels a blocked pull instead of leaking a worker.
    entered = threading.Event()
    cancelled = threading.Event()

    async def blocked_generator():
        try:
            entered.set()
            await asyncio.Event().wait()
            yield _chunk("unreachable")
        finally:
            # Cancellation completes only after async generator cleanup has yielded.
            await asyncio.sleep(0)
            cancelled.set()

    blocked = blocked_generator()
    stream, _calls = _open(monkeypatch, lambda _kw: blocked, facade)
    with ThreadPoolExecutor(max_workers=1) as consumer:
        pull = consumer.submit(next, stream)
        try:
            assert entered.wait(timeout=5)
        finally:
            stream.close()
        with pytest.raises(CancelledError):
            pull.result(timeout=5)
    assert cancelled.is_set()
    with pytest.raises(StopAsyncIteration):
        asyncio.run(anext(blocked))
