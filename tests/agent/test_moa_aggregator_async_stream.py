"""MoA aggregator facade: a native async aggregator client's stream must become synchronous.

An async aggregator client (the Claude/Anthropic native async wrapper, an async
OpenAI client, or any provider-supplied async client) answers ``call_llm(stream=True)``
with an awaitable that resolves to either a completed response or an async token
stream.  ``MoAChatCompletions`` is a *synchronous* facade: its result is iterated
synchronously by Relay's managed stream and by the chat-completions loop.  Handing
that consumer a coroutine surfaces as ``TypeError: 'coroutine' object is not
iterable`` inside Relay's provider callback, which abandons the aggregator stream
and silently collapses the MoA turn onto a standalone fallback model.

The facade must therefore adapt an awaitable/async-iterator result into a sync
iterator: await the provider exactly once (never a second dispatch), keep the
stream lazy, carry the caller's contextvars into each async step (Relay runs its
in-chunk sanitization/observers under that context), and release the underlying
async stream on exhaustion or ``close()``.  A plain synchronous iterator and a
completed response keep their existing pass-through behaviour.
"""

from __future__ import annotations

import asyncio
import contextvars
from types import SimpleNamespace

import pytest

from agent import moa_loop


def _chunk(text):
    return SimpleNamespace(text=text)


def _completed(content="aggregator acted"):
    message = SimpleNamespace(content=content, tool_calls=[])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None, model="claude-native")


class _AsyncStream:
    """Async token stream double that records close and contextvar visibility."""

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


def _open(monkeypatch, on_call, facade):
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
    return facade.create(_moa_prepared_request=prepared, stream=True, tools=[]), calls


# ---------------------------------------------------------------------------
# Awaitable (coroutine) result -> synchronous iterator
# ---------------------------------------------------------------------------


def test_awaitable_completed_response_becomes_iterable_one_chunk_stream(
    monkeypatch, facade
):
    """A coroutine resolving to a completed response is awaited ONCE and wrapped
    as the one-chunk delta iterator the outer accumulator already understands."""

    async def _awaitable():
        return _completed("aggregator acted")

    stream, calls = _open(monkeypatch, lambda _kw: _awaitable(), facade)

    # The bug: this used to be the raw coroutine -> `iter(stream)` raised TypeError.
    iter(stream)
    chunks = list(stream)
    assert [c.choices[0].delta.content for c in chunks] == ["aggregator acted"]
    # Exactly one provider dispatch (awaiting the coroutine is that dispatch).
    assert len(calls) == 1
    assert calls[0]["stream"] is True


def test_awaitable_async_iterator_streams_lazily_in_order_and_closes(
    monkeypatch, facade
):
    """A coroutine resolving to an async stream is wrapped so the consumer can
    iterate it synchronously; chunky order is preserved and the source is closed."""
    source = _AsyncStream([_chunk("a"), _chunk("b"), _chunk("c")])

    async def _awaitable():
        return source

    stream, calls = _open(monkeypatch, lambda _kw: _awaitable(), facade)

    assert [c.text for c in stream] == ["a", "b", "c"]
    assert source.closed is True
    assert len(calls) == 1


def test_async_iterator_returned_directly_is_wrapped(monkeypatch, facade):
    """A client whose ``create`` returns an async iterator (not a coroutine) is
    adapted too — ``__aiter__`` without ``__iter__`` is the trigger."""
    source = _AsyncStream([_chunk("only")])

    stream, _calls = _open(monkeypatch, lambda _kw: source, facade)

    assert [c.text for c in stream] == ["only"]
    assert source.closed is True


def test_close_propagates_to_the_async_source_without_exhaustion(monkeypatch, facade):
    """Closing the facade stream early must stop the async source (resource close),
    even though iteration never reached StopAsyncIteration."""
    source = _AsyncStream([_chunk("a"), _chunk("b"), _chunk("c")])

    stream, _calls = _open(monkeypatch, lambda _kw: source, facade)
    assert next(stream).text == "a"
    stream.close()

    assert source.closed is True


def test_async_steps_run_under_the_callers_contextvars(monkeypatch, facade):
    """Each async step runs in the context captured at the facade boundary so an
    in-context event sanitizer/observer still sees the turn's contextvars."""
    probe = contextvars.ContextVar("moa_aggregator_probe")
    source = _AsyncStream([_chunk("a")], probe=probe)

    stream, _calls = _open(monkeypatch, lambda _kw: source, facade)
    token = probe.set("turn-context")
    try:
        chunks = list(stream)
    finally:
        probe.reset(token)

    assert [c.text for c in chunks] == ["a"]
    # Every async step (including the terminating __anext__) saw the turn's context.
    assert source.seen_context and set(source.seen_context) == {"turn-context"}


def test_bridging_works_while_an_event_loop_is_running(monkeypatch, facade):
    """Relay's provider callback runs on a live event loop; the bridge must not
    re-enter it (a plain ``asyncio.run`` would raise)."""

    async def _awaitable():
        return _completed("ok")

    stream, _calls = _open(monkeypatch, lambda _kw: _awaitable(), facade)

    async def _drive():
        # Executing synchronously inside a running loop must still yield chunks.
        return [c.choices[0].delta.content for c in stream]

    assert asyncio.run(_drive()) == ["ok"]


# ---------------------------------------------------------------------------
# Pass-through regressions (unchanged contracts)
# ---------------------------------------------------------------------------


def test_plain_sync_stream_is_returned_unchanged(monkeypatch, facade):
    sentinel = iter([_chunk("x")])
    stream, _calls = _open(monkeypatch, lambda _kw: sentinel, facade)
    assert stream is sentinel


def test_completed_response_is_wrapped_as_one_chunk(monkeypatch, facade):
    completed = _completed("done")
    stream, _calls = _open(monkeypatch, lambda _kw: completed, facade)
    assert [c.choices[0].delta.content for c in stream] == ["done"]


def test_non_streaming_call_still_returns_the_raw_response(monkeypatch, facade):
    completed = _completed("done")
    calls = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        return completed

    monkeypatch.setattr(moa_loop, "call_llm", fake_call_llm)
    prepared = {
        "messages": [{"role": "user", "content": "q"}],
        "guidance": None,
        "aggregator": {"provider": "anthropic", "model": "claude-native"},
        "aggregator_temperature": None,
    }
    out = facade.create(_moa_prepared_request=prepared, tools=[])
    assert out is completed


def test_stream_creation_iteration_and_close_share_one_event_loop(monkeypatch, facade):
    """A real async client can bind sockets/tasks to the loop opening its stream."""
    loops = []

    class LoopBoundStream:
        def __init__(self):
            self.owner = asyncio.get_running_loop()
            loops.append(self.owner)
            self.delivered = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            assert asyncio.get_running_loop() is self.owner
            if self.delivered:
                raise StopAsyncIteration
            self.delivered = True
            return _chunk("loop-bound")

        async def aclose(self):
            assert asyncio.get_running_loop() is self.owner
            loops.append(asyncio.get_running_loop())

    async def open_stream():
        return LoopBoundStream()

    stream, calls = _open(monkeypatch, lambda _kw: open_stream(), facade)
    assert [chunk.text for chunk in stream] == ["loop-bound"]
    assert len(calls) == 1
    assert len(loops) == 2 and loops[0] is loops[1]
    assert loops[0].is_closed()


def test_factory_is_awaited_inside_the_running_loop_bridge(monkeypatch, facade):
    """Open the facade inside a live loop, not before it as the earlier test did."""

    async def open_stream():
        return _completed("inside-loop")

    async def drive():
        stream, calls = _open(monkeypatch, lambda _kw: open_stream(), facade)
        assert len(calls) == 1
        return [chunk.choices[0].delta.content for chunk in stream]

    assert asyncio.run(drive()) == ["inside-loop"]
