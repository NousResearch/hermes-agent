"""Regression coverage for clarify-card ordering at the gateway boundary."""

import asyncio
import concurrent.futures
from types import SimpleNamespace

from gateway.run import _schedule_clarify_after_stream_drain


class _FakeStreamConsumer:
    def __init__(self, events):
        self._events = events

    def drain(self, timeout=2.0):
        self._events.append("drain")
        return True


class _FakeAdapter:
    def __init__(self, events):
        self._events = events

    async def send_clarify(self, **kwargs):
        self._events.append("send_clarify")
        return SimpleNamespace(success=True)


def test_clarify_send_is_scheduled_after_stream_drain():
    events = []
    consumer = _FakeStreamConsumer(events)
    adapter = _FakeAdapter(events)

    def schedule(coro, loop, *, logger, log_message):
        result = asyncio.run(coro)
        future = concurrent.futures.Future()
        future.set_result(result)
        return future

    future = _schedule_clarify_after_stream_drain(
        stream_consumer=consumer,
        adapter=adapter,
        chat_id="chat-1",
        question="What color?",
        choices=["red", "blue"],
        clarify_id="clarify-1",
        session_key="session-1",
        metadata=None,
        loop=None,
        schedule=schedule,
    )

    assert future.result().success is True
    assert events == ["drain", "send_clarify"]
