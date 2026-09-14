"""The duplicate-risk diagnostic must only speak for consumers that could carry the turn final.

A stream consumer is created whenever text streaming OR interim assistant messages are on. With
text streaming off (``streaming.enabled: false``) and interim messages on (Telegram's default)
the consumer exists for commentary alone: no deltas are ever routed to it, so its
``final_response_sent`` / ``final_content_delivered`` flags are False every single turn and the
normal final send is the only delivery. Logging "possible duplicate send" there is a permanent
false positive (live: 3 warnings on 2026-09-12 with zero user-visible duplicates).

Contract: a commentary-only consumer does not claim duplicate risk AND still lets the final
through; a delta-fed consumer without a confirmed final still warns.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

from gateway.run import GatewayRunner


class _LogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _mark(response, consumer):
    runner = object.__new__(GatewayRunner)
    turn_ctx = SimpleNamespace(
        stream_consumer_holder=[consumer],
        source=SimpleNamespace(chat_id="123", platform=None),
        session_key="agent:main:telegram:dm:123",
    )
    asyncio.run(GatewayRunner._run_agent_mark_streamed_delivery(runner, response, turn_ctx))


def _consumer(*, receives_deltas: bool):
    return SimpleNamespace(
        receives_deltas=receives_deltas,
        message_id=None,
        adapter=None,
        final_response_sent=False,
        final_content_delivered=False,
    )


def _capture(fn) -> list[str]:
    handler = _LogCapture()
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        fn()
    finally:
        root.removeHandler(handler)
    return handler.messages


def test_commentary_only_consumer_is_not_reported_as_a_duplicate_risk():
    response = {"final_response": "the answer"}
    messages = _capture(lambda: _mark(response, _consumer(receives_deltas=False)))

    assert not any("NOT suppressed" in m for m in messages), messages
    assert not response.get("already_sent"), "the final must still be sent on this path"


def test_delta_fed_consumer_without_a_confirmed_final_still_warns():
    response = {"final_response": "the answer"}
    messages = _capture(lambda: _mark(response, _consumer(receives_deltas=True)))

    assert any("NOT suppressed" in m for m in messages), messages
    assert not response.get("already_sent")
