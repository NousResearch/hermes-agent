"""Regression tests for client-created DM topic openers (#136567).

Since 2026-10-10 Telegram delivers the opening message of a client-created DM topic WITHOUT
``message_thread_id``, while the ``forum_topic_created`` service message for the same event
still carries the new topic id. The adapter pairs the two so the opener routes to the topic
Telegram meant, instead of being pinned to the user's most-recent bound topic by the lobby
recovery.
"""

import sys
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig

# Shared telegram mock (same discipline as tests/gateway/test_dm_topics.py): install before the
# adapter import, then force reimport so the adapter binds to whatever sys.modules now holds.
from tests.gateway.conftest import _ensure_telegram_mock

_ensure_telegram_mock()
sys.modules.pop("plugins.platforms.telegram.adapter", None)

from plugins.platforms.telegram.adapter import TelegramAdapter
from telegram.error import BadRequest


def _make_adapter() -> TelegramAdapter:
    return TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={}))


def _make_message(chat_id=111, user_id=42, message_id=1002, thread_id=None, text="Prova 3",
                  chat_type="private", is_forum=None, forum_topic_created=None):
    chat = SimpleNamespace(id=chat_id, type=chat_type, title=None, full_name="Test User")
    if is_forum is not None:
        chat.is_forum = is_forum
    user = SimpleNamespace(id=user_id, full_name="Test User", is_bot=False)
    return SimpleNamespace(
        chat=chat,
        from_user=user,
        text=text,
        message_thread_id=thread_id,
        is_topic_message=bool(thread_id) if chat_type == "private" else None,
        message_id=message_id,
        reply_to_message=None,
        date=None,
        forum_topic_created=forum_topic_created,
    )


def _make_service_message(chat_id=111, user_id=42, message_id=1001, thread_id=40798,
                          chat_type="private", is_forum=None):
    # Real Bot API shape: ForumTopicCreated carries only name/icon fields — never the topic id.
    # The new topic id lives on the service Message itself (message_thread_id= below).
    return _make_message(
        chat_id=chat_id, user_id=user_id, message_id=message_id, thread_id=thread_id, text=None,
        chat_type=chat_type, is_forum=is_forum,
        forum_topic_created=SimpleNamespace(name="Prova 3", icon_color=0x6FB9F0),
    )


def _record(adapter: TelegramAdapter, service_message) -> None:
    """Drive the service message through the same path the PTB handler takes."""
    adapter._note_forum_topic_created(service_message)


def _event_thread_id(adapter: TelegramAdapter, message) -> str | None:
    from gateway.platforms.event import MessageType

    event = adapter._build_message_event(message, MessageType.TEXT)
    return event.source.thread_id


def test_thread_less_opener_routes_to_newly_created_topic():
    """Svc message 1001 (topic 40798) + thread-less opener 1002 → opener routes to 40798."""
    adapter = _make_adapter()
    _record(adapter, _make_service_message(message_id=1001, thread_id=40798))
    assert _event_thread_id(adapter, _make_message(message_id=1002, thread_id=None)) == "40798"


def test_non_adjacent_thread_less_message_keeps_legacy_routing():
    """A thread-less message NOT adjacent to the service message is left alone (None), so the
    lobby recovery keeps its previous behaviour."""
    adapter = _make_adapter()
    _record(adapter, _make_service_message(message_id=1001, thread_id=40798))
    assert _event_thread_id(adapter, _make_message(message_id=1003, thread_id=None)) is None


def test_pairing_is_consumed_once():
    """The pairing is one-shot: after the opener consumed it, a later thread-less message
    (e.g. a lobby return) is not redirected into the new topic."""
    adapter = _make_adapter()
    _record(adapter, _make_service_message(message_id=1001, thread_id=40798))
    assert _event_thread_id(adapter, _make_message(message_id=1002, thread_id=None)) == "40798"
    assert _event_thread_id(adapter, _make_message(message_id=1004, thread_id=None)) is None


def test_stale_pairing_expires():
    """Outside the 120 s window the pairing no longer applies."""
    adapter = _make_adapter()
    _record(adapter, _make_service_message(message_id=1001, thread_id=40798))
    thread_id, svc_message_id, ts = adapter._pending_topic_openers[("111", "42")]
    adapter._pending_topic_openers[("111", "42")] = (thread_id, svc_message_id, ts - 121.0)
    assert _event_thread_id(adapter, _make_message(message_id=1002, thread_id=None)) is None


def test_other_user_does_not_consume_pairing():
    """The pairing is per (chat, user): another participant's thread-less message in the same
    chat keeps its legacy routing."""
    adapter = _make_adapter()
    _record(adapter, _make_service_message(user_id=42, message_id=1001, thread_id=40798))
    assert _event_thread_id(adapter, _make_message(user_id=99, message_id=1002, thread_id=None)) is None


def test_threaded_message_is_never_rewritten():
    """A message that already carries a (non-General) thread id keeps it untouched."""
    adapter = _make_adapter()
    _record(adapter, _make_service_message(user_id=42, message_id=1001, thread_id=40798))
    assert _event_thread_id(adapter, _make_message(message_id=1002, thread_id=17281)) == "17281"


def test_forum_group_service_messages_are_not_recorded():
    """Forum groups deliver topic messages with their own thread id; only private-chat (DM
    topic) service messages need the pairing."""
    adapter = _make_adapter()
    _record(adapter, _make_service_message(message_id=1001, thread_id=40798,
                                           chat_type="supergroup", is_forum=True))
    assert adapter._pending_topic_openers == {}


def test_bot_created_topic_service_messages_are_ignored():
    """A bot-authored forum_topic_created (bot creates its own topics) pairs nothing: bot
    openers are outbound and already routed by the creating code."""
    adapter = _make_adapter()
    msg = _make_service_message(message_id=1001, thread_id=40798)
    msg.from_user = SimpleNamespace(id=777, full_name="the bot", is_bot=True)
    _record(adapter, msg)
    assert adapter._pending_topic_openers == {}


def test_plain_threaded_message_is_not_recorded():
    """A private-chat message that merely carries a thread id (no forum_topic_created service
    payload) records nothing — only real topic-creation service messages pair."""
    adapter = _make_adapter()
    _record(adapter, _make_message(message_id=1001, thread_id=40798))
    assert adapter._pending_topic_openers == {}


@pytest.mark.asyncio
async def test_dm_topic_lane_thread_not_found_fails_closed_without_prune():
    """Send-side negative for the field shape from #136567: a Hermes DM-topic reply-fallback
    lane the adapter holds a binding for (used successfully minutes earlier), where Bot API
    answers "Message thread not found" for the reply. The send must fail closed — exactly one
    Bot API attempt (no plain fallback that would render outside the topic lane) and no
    binding prune, so one rejected send cannot collapse the chat's remaining lanes."""
    adapter = _make_adapter()
    calls: list[dict] = []
    pruned: list[tuple] = []

    async def mock_send_message(**kwargs):
        calls.append(dict(kwargs))
        raise BadRequest("Message thread not found")

    adapter._bot = SimpleNamespace(send_message=mock_send_message)
    adapter._prune_stale_dm_topic_binding = lambda *args, **kwargs: pruned.append((args, kwargs))

    result = await adapter.send(
        chat_id="775566675",
        content="reply into the topic lane",
        metadata={
            "thread_id": "47321",
            "telegram_dm_topic_reply_fallback": True,
            "direct_messages_topic_id": "47321",
            "telegram_reply_to_message_id": "31890",
        },
    )

    assert result.success is False
    assert result.retryable is False
    assert "thread not found" in str(result.error).lower()
    assert len(calls) == 1
    assert calls[0]["message_thread_id"] == 47321
    assert pruned == []
