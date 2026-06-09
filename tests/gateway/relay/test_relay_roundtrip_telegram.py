"""End-to-end relay round-trip for Telegram against the in-memory stub.

Companion to ``test_relay_roundtrip.py`` (Discord). Proves the relay generalizes
beyond Discord — the Phase 1 exit gate requires *both* Telegram and Discord
descriptors to round-trip and their inbound ``MessageEvent``s to drive
``build_session_key()`` correctly.

Telegram's discriminator profile differs from Discord's, which is the point:
  - No ``guild_id``; isolation between chats comes from ``chat_id`` alone.
  - Forum topics live inside ONE ``chat_id`` and isolate by ``thread_id`` (the
    Telegram analog of Discord's per-guild isolation).
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
    topic id as thread_id (no guild_id — Telegram has no guild concept).
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
    assert captured[0].source.guild_id is None  # Telegram has no guild


@pytest.mark.asyncio
async def test_two_telegram_chats_isolate_by_chat_id(wired):
    """No guild_id on Telegram — two distinct chats must still isolate, keyed
    on chat_id alone (the Discord-guild role is played by chat_id here)."""
    ev_a = _tg_group_event("chat-A", "userX", "hi A")
    ev_b = _tg_group_event("chat-B", "userX", "hi B")
    key_a = build_session_key(ev_a.source)
    key_b = build_session_key(ev_b.source)
    assert key_a != key_b
    # Same chat + same user collapses to one session.
    ev_a2 = _tg_group_event("chat-A", "userX", "again")
    assert build_session_key(ev_a2.source) == key_a


@pytest.mark.asyncio
async def test_forum_topics_isolate_by_thread_id_within_one_chat(wired):
    """Telegram forum topics share a single chat_id and isolate by thread_id —
    the Telegram analog of Discord per-guild isolation. Two topics in the same
    forum must NOT collide, and (threads shared across participants by default)
    a second user in the same topic shares the session."""
    topic1 = _tg_group_event("forum-1", "userX", "in topic 1", thread_id="t-1")
    topic2 = _tg_group_event("forum-1", "userX", "in topic 2", thread_id="t-2")
    k1 = build_session_key(topic1.source)
    k2 = build_session_key(topic2.source)
    assert k1 != k2, "two forum topics in one chat must not share a session"
    # Same chat, no topic → distinct from any topic session.
    plain = _tg_group_event("forum-1", "userX", "no topic")
    assert build_session_key(plain.source) not in {k1, k2}
    # Threads are shared across participants by default: a different user in the
    # same topic lands on the SAME session key (user_id not appended in threads).
    topic1_other_user = _tg_group_event("forum-1", "userY", "me too", thread_id="t-1")
    assert build_session_key(topic1_other_user.source) == k1


@pytest.mark.asyncio
async def test_telegram_dm_isolates_by_chat_id(wired):
    dm_a = _tg_dm_event("dm-111", "userX", "hey")
    dm_b = _tg_dm_event("dm-222", "userY", "yo")
    assert build_session_key(dm_a.source) != build_session_key(dm_b.source)
    assert build_session_key(dm_a.source).startswith("agent:main:telegram:dm:")


@pytest.mark.asyncio
async def test_outbound_send_round_trips_telegram(wired):
    adapter, stub = wired
    await adapter.connect()
    stub.next_send_result = {"success": True, "message_id": "tg-77"}
    result = await adapter.send("chat-100", "a reply")
    assert result.success is True
    assert result.message_id == "tg-77"
    assert stub.sent[0]["op"] == "send"
    assert stub.sent[0]["chat_id"] == "chat-100"


async def _async_capture(sink, event):
    sink.append(event)
    return None
