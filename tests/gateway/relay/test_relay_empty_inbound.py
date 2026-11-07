"""Relay inbound events with no user payload are dropped before dispatch.

A connector frame with no text, no media, no reply/channel context and no
structured prompt answer carries nothing for the agent to act on. It must be
consumed (and ACKed) at the relay adapter instead of reaching authorization,
pairing, plugin hooks or the agent. Media-only events, prompt answers and
context-bearing events keep flowing.
"""

from __future__ import annotations

import asyncio

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import SessionSource
from gateway.platforms.event import MessageEvent
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor
from gateway.relay.ws_transport import _event_from_wire
from tests.gateway.relay.stub_connector import StubConnector


def _adapter() -> RelayAdapter:
    desc = CapabilityDescriptor(
        contract_version=CONTRACT_VERSION,
        platform="discord",
        label="Discord",
        max_message_length=2000,
        supports_draft_streaming=False,
        supports_edit=True,
        supports_threads=True,
        markdown_dialect="discord",
        len_unit="chars",
        emoji="\U0001f4ac",
        platform_hint="",
        pii_safe=False,
        supported_ops=("send", "edit", "typing"),
    )
    return RelayAdapter(PlatformConfig(), desc, transport=StubConnector(desc))


@pytest.fixture()
def loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


def _tap(adapter, handled, consumed_prompt=False):
    async def _handle(event):
        handled.append(event)

    async def _consume(_event):
        return consumed_prompt

    async def _localize(_event):
        return None

    adapter.handle_message = _handle
    adapter._consume_prompt_response = _consume
    adapter._localize_inbound_media = _localize


def _wire(**overrides):
    raw = {
        "text": "",
        "message_type": "text",
        "message_id": "m-1",
        "source": {"platform": "discord", "chat_id": "chan1", "chat_type": "dm", "user_id": "u1"},
    }
    raw.update(overrides)
    return _event_from_wire(raw)


@pytest.mark.parametrize("text", ["", "   \n\t"])
def test_empty_relay_event_is_dropped_before_dispatch(loop, text):
    adapter = _adapter()
    handled = []
    _tap(adapter, handled)

    result = loop.run_until_complete(adapter._on_inbound(_wire(text=text)))

    assert result is None  # transport ACKs a None result: deliberately consumed
    assert handled == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"text": "hello"},
        {"media_urls": ["https://cdn.example/photo.png"], "media": [{"mime": "image/png"}]},
        {"reply_to_message_id": "m-0"},
        {"context": [{"author": "bob", "text": "anyone seen the deploy?"}]},
    ],
    ids=["text", "media-only", "reply", "channel-context"],
)
def test_relay_event_with_payload_still_dispatches(loop, overrides):
    adapter = _adapter()
    handled = []
    _tap(adapter, handled)

    loop.run_until_complete(adapter._on_inbound(_wire(**overrides)))

    assert len(handled) == 1


def test_empty_text_prompt_response_is_still_consumed(loop):
    """A structured prompt answer has no text but must reach its consumer."""
    adapter = _adapter()
    handled = []
    seen = []

    async def _consume(event):
        seen.append(event)
        return True

    _tap(adapter, handled)
    adapter._consume_prompt_response = _consume
    event = _wire(prompt_response={"prompt_id": "p-1", "option_id": "approve"})

    loop.run_until_complete(adapter._on_inbound(event))

    assert seen == [event]
    assert handled == []


def test_internal_blank_event_is_not_treated_as_empty(loop):
    adapter = _adapter()
    handled = []
    _tap(adapter, handled)
    source = SessionSource(platform=Platform.DISCORD, chat_id="chan1", chat_type="dm", user_id="u1")
    event = MessageEvent(text="", source=source, internal=True)

    loop.run_until_complete(adapter._on_inbound(event))

    assert len(handled) == 1
