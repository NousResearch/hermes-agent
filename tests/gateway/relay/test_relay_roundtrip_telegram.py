"""End-to-end relay round-trip for Telegram against the in-memory stub.

Companion to ``test_relay_roundtrip.py`` (Discord). Proves the relay generalizes
beyond Discord — the Phase 1 exit gate requires *both* Telegram and Discord
descriptors to round-trip and their inbound ``MessageEvent``s to drive
``build_session_key()`` correctly.

Telegram's discriminator profile differs from Discord's, which is the point:
  - No ``scope_id``; isolation between chats comes from ``chat_id`` alone.
  - Forum topics live inside ONE ``chat_id`` and isolate by ``thread_id`` (the
    Telegram analog of Discord's per-scope isolation).
  - Forum/thread sessions are shared across participants by default
    (``thread_sessions_per_user=False``) — user_id is NOT appended in a thread.
  - ``len_unit="utf16"`` (Telegram counts UTF-16 code units) and
    ``markdown_dialect="markdown_v2"`` — distinct from Discord's chars/discord.

If the descriptor or session-keying only worked for Discord, these fail.
"""

from __future__ import annotations

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key
from gateway.relay.adapter import RelayAdapter
from gateway.relay.descriptor import CONTRACT_VERSION, CapabilityDescriptor

from tests.gateway.relay.stub_connector import StubConnector


def _telegram_descriptor() -> CapabilityDescriptor:
    return CapabilityDescriptor(
        contract_version=CONTRACT_VERSION,
        platform="telegram",
        label="Telegram",
        max_message_length=4096,
        supports_draft_streaming=True,  # Telegram DMs support sendMessageDraft
        supported_ops=("send", "edit", "typing", "follow_up", "draft"),
        supports_edit=True,
        supports_threads=True,  # forum topics
        markdown_dialect="markdown_v2",
        len_unit="utf16",
        emoji="\u2708\ufe0f",
        platform_hint="You are on Telegram.",
        pii_safe=False,
    )


def _tg_group_event(chat_id: str, user_id: str, text: str, thread_id: str | None = None) -> MessageEvent:
    """Synthetic inbound the connector would build from a Telegram update.

    A plain group message has no thread_id; a forum-topic message carries the
    topic id as thread_id (no scope_id — Telegram has no scope concept).
    """
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type="forum" if thread_id else "group",
        user_id=user_id,
        thread_id=thread_id,
    )
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=source)


def _tg_dm_event(chat_id: str, user_id: str, text: str) -> MessageEvent:
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=chat_id,
        chat_type="dm",
        user_id=user_id,
    )
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=source)


@pytest.fixture
def wired():
    desc = _telegram_descriptor()
    stub = StubConnector(desc)
    adapter = RelayAdapter(PlatformConfig(), desc, transport=stub)
    return adapter, stub


@pytest.mark.asyncio
async def test_telegram_descriptor_round_trips_through_stub(wired):
    """The connector's handshake descriptor for Telegram survives JSON + the
    adapter configures itself from it (utf16 length unit, 4096 limit)."""
    adapter, stub = wired
    desc = _telegram_descriptor()
    assert CapabilityDescriptor.from_json(desc.to_json()) == desc
    # Adapter reflects the descriptor's capability profile.
    assert adapter.MAX_MESSAGE_LENGTH == 4096
    assert adapter.supports_draft_streaming() is True
    # utf16 length unit selects a non-default len fn (Telegram counts UTF-16).
    assert adapter.message_len_fn is not len


@pytest.mark.asyncio
async def test_inbound_telegram_event_reaches_adapter(wired, monkeypatch):
    adapter, stub = wired
    captured: list[MessageEvent] = []
    monkeypatch.setattr(adapter, "handle_message", lambda ev: _async_capture(captured, ev))
    await adapter.connect()
    await stub.push_inbound(_tg_group_event("chat-100", "userX", "hello"))
    assert len(captured) == 1
    assert captured[0].text == "hello"
    assert captured[0].source.platform == Platform.TELEGRAM
    assert captured[0].source.scope_id is None  # Telegram has no scope


async def _async_capture(sink, event):
    sink.append(event)
    return None
