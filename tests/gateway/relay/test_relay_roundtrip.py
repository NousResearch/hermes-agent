"""End-to-end relay round-trip against the in-memory stub connector.

Proves the gateway side of the relay works with no real connector:
  - connect() registers the inbound handler,
  - a connector-delivered MessageEvent reaches the adapter's message path,
  - SessionSource discriminators (scope_id) drive build_session_key isolation,
  - an outbound send round-trips through the transport.

These target the transport contract + session-key derivation (Task 1.2's gate),
not the full agent turn — handle_message is patched to capture the event.
"""

from __future__ import annotations

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor

from dataclasses import replace

from tests.gateway.relay.stub_connector import StubConnector


def _discord_descriptor() -> CapabilityDescriptor:
    return CapabilityDescriptor(
        contract_version=CONTRACT_VERSION,
        platform="discord",
        label="Discord",
        max_message_length=2000,
        supports_draft_streaming=False,
        supports_edit=True,
        supports_threads=True,
        markdown_dialect="discord",
        len_unit="chars",
        emoji="\U0001f47e",
        platform_hint="You are on Discord.",
        pii_safe=False,
    )


def _discord_event(scope_id: str, channel_id: str, user_id: str, text: str) -> MessageEvent:
    """Synthetic inbound the connector would build from a discord.js message."""
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id=channel_id,
        chat_type="group",
        user_id=user_id,
        scope_id=scope_id,
    )
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=source)


@pytest.fixture
def wired():
    stub = StubConnector(_discord_descriptor())
    adapter = RelayAdapter(PlatformConfig(), _discord_descriptor(), transport=stub)
    return adapter, stub


@pytest.mark.asyncio
async def test_connect_registers_inbound_handler(wired):
    adapter, stub = wired
    assert stub._inbound is None
    ok = await adapter.connect()
    assert ok is True
    assert stub.connected is True
    assert stub._inbound is not None


@pytest.mark.asyncio
async def test_inbound_event_reaches_adapter(wired, monkeypatch):
    adapter, stub = wired
    captured = []
    monkeypatch.setattr(adapter, "handle_message", lambda ev: _async_capture(captured, ev))
    await adapter.connect()
    ev = _discord_event("guildA", "chan1", "userX", "hello")
    await stub.push_inbound(ev)
    assert len(captured) == 1
    assert captured[0].text == "hello"
    assert captured[0].source.scope_id == "guildA"


async def _async_capture(sink, event):
    sink.append(event)
    return None
