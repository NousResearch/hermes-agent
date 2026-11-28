"""Telegram: a buffered event failing before acceptance is re-held, not silently lost (#160276)."""

import asyncio

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SessionSource
from gateway.platforms.event import MessageEvent, MessageType


def _adapter():
    from plugins.platforms.telegram.adapter import TelegramAdapter

    return TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))


def _event(text):
    return MessageEvent(
        text=text, message_type=MessageType.TEXT,
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="12345", chat_type="dm", user_id="1"))


async def _flush_with_failure(adapter, event, accepted):
    """Run the real shared flush body; record holds instead of scheduling a redispatch loop."""
    held = []

    adapter._hold_inbound_event = lambda ev, *, where, schedule=True: held.append((ev, where))

    dispatched = []

    async def handle(ev):
        dispatched.append(ev)
        ev._gateway_accepted = accepted
        raise RuntimeError("can't start new thread")

    adapter.handle_message = handle
    pending, tasks = {"k": event}, {}
    with pytest.raises(RuntimeError, match="can't start new thread"):
        await adapter._flush_buffered(pending, tasks, "k", 0, "text", pause=lambda _: asyncio.sleep(0))
    return dispatched, pending, held


@pytest.mark.asyncio
async def test_flush_failure_before_accept_reholds_the_event():
    adapter = _adapter()
    event = _event("a message")

    dispatched, pending, held = await _flush_with_failure(adapter, event, accepted=False)

    assert dispatched == [event]
    assert pending == {}
    assert held == [(event, "text-flush-failed")]


@pytest.mark.asyncio
async def test_flush_failure_after_accept_does_not_rehold_the_event():
    adapter = _adapter()
    event = _event("already accepted")

    dispatched, pending, held = await _flush_with_failure(adapter, event, accepted=True)

    assert dispatched == [event]
    assert pending == {}
    assert held == []  # re-holding an accepted event would deliver it twice
