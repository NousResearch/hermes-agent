"""ForwardedRecordsParseMiddleware must actually send its loading heartbeat.

``_send_loading_heartbeat`` is a coroutine function. Calling it without
``await`` builds a coroutine object and drops it, so the RUNNING heartbeat is
never sent and Python raises "coroutine was never awaited" at collection time.
Forwarded-record parsing is the slow inbound path, which is exactly where the
user needs the loading bubble.
"""

import warnings

import pytest

from gateway.platforms.yuanbao import (
    ForwardedRecordsParseMiddleware,
    InboundContext,
    WS_HEARTBEAT_RUNNING,
)


class _Heartbeat:
    def __init__(self):
        self.calls = []

    async def send_heartbeat_once(self, chat_id, state):
        self.calls.append((chat_id, state))


class _Outbound:
    def __init__(self):
        self.heartbeat = _Heartbeat()


class _Adapter:
    name = "yuanbao"

    def __init__(self):
        self._outbound = _Outbound()


@pytest.mark.asyncio
async def test_forwarded_records_send_the_loading_heartbeat():
    adapter = _Adapter()
    ctx = InboundContext(adapter=adapter)
    ctx.chat_id = "chat-1"
    ctx.forwarded_records = [{"msgContent": {"text": "hello"}}]

    called = []

    async def _next():
        called.append(True)

    with warnings.catch_warnings():
        # A dropped coroutine surfaces here as a RuntimeWarning rather than a
        # failure; promote it so the regression cannot pass silently.
        warnings.simplefilter("error", RuntimeWarning)
        await ForwardedRecordsParseMiddleware().handle(ctx, _next)

    assert adapter._outbound.heartbeat.calls == [("chat-1", WS_HEARTBEAT_RUNNING)]
    assert called, "middleware must still call next_fn"


@pytest.mark.asyncio
async def test_no_forwarded_records_sends_no_heartbeat():
    adapter = _Adapter()
    ctx = InboundContext(adapter=adapter)
    ctx.chat_id = "chat-1"

    async def _next():
        pass

    await ForwardedRecordsParseMiddleware().handle(ctx, _next)

    assert adapter._outbound.heartbeat.calls == []
