"""Telegram: a buffered event failing before acceptance is re-held, not silently lost (#160276),
and an event the gateway already accepted is never re-held (that would deliver it twice)."""

import asyncio

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SessionSource
from gateway.platforms.event import MessageEvent, MessageType


def _adapter(accepted_texts):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    held, dispatched = [], []
    adapter._hold_inbound_event = lambda ev, *, where, schedule=True: held.append((ev.text, where))

    async def handle(ev):  # fails like a thread-exhausted host: "can't start new thread"
        dispatched.append(ev.text)
        ev._gateway_accepted = ev.text in accepted_texts
        raise RuntimeError("can't start new thread")

    adapter.handle_message = handle
    return adapter, held, dispatched


def _event(text):
    return MessageEvent(
        text=text, message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm", user_id="1"))


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted, expected_held", [
    (False, [("msg", "text-flush-failed")]),
    (True, []),
])
async def test_flush_failure_reholds_only_unaccepted_events(accepted, expected_held):
    adapter, held, dispatched = _adapter({"msg"} if accepted else set())
    pending = {"k": _event("msg")}

    with pytest.raises(RuntimeError, match="can't start new thread"):
        await adapter._flush_buffered(pending, {}, "k", 0, "text", pause=lambda _: asyncio.sleep(0))

    assert dispatched == ["msg"]
    assert pending == {}
    assert held == expected_held


@pytest.mark.asyncio
async def test_failed_redispatch_does_not_rehold_an_accepted_event():
    adapter, held, dispatched = _adapter({"accepted"})
    adapter._held_inbound_events = [_event("accepted"), _event("next")]

    await adapter._redispatch_held_inbound()

    assert dispatched == ["accepted"]
    assert held == [("next", "redispatch-failed")]
