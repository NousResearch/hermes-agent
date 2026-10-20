"""A busy-queue photo burst only merges follow-ups from the SAME sender.

With a shared session key (a thread or forum topic by default, since
``thread_sessions_per_user`` defaults to false, or a group with ``group_sessions_per_user: false``)
one slot can receive input from several people. The merged event keeps the head event's
``source``, so merging another sender's photo (or caption) into it attributes that input to the
wrong person.
"""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.platforms.base import (
    BasePlatformAdapter,
    Platform,
    PlatformConfig,
    SendResult,
    SessionSource,
    build_session_key,
)
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner

SHARED_KEY = "agent:main:telegram:group:c1"


class _StubAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        self._mark_disconnected()

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="msg-1")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "group"}


def _runner():
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._queued_events = {}
    adapter = _StubAdapter()
    runner.adapters = {Platform.TELEGRAM: adapter}
    return runner, adapter


def _event(user_id, n, *, photo=True, text=""):
    source = MagicMock(chat_id="c1", platform=Platform.TELEGRAM, profile=None, user_id=user_id)
    if photo:
        return MessageEvent(
            text=text, message_type=MessageType.PHOTO, source=source,
            media_urls=[f"/tmp/{n}.jpg"], media_types=["image/jpeg"], message_id=f"m{n}",
        )
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=source, message_id=f"m{n}")


def _queued(runner):
    return [
        (e.source.user_id, len(e.media_urls or []), e.text)
        for e in runner._queued_events.get(SHARED_KEY, [])
    ]


def test_photo_burst_from_one_sender_still_merges():
    runner, adapter = _runner()
    runner._queue_or_replace_pending_event(SHARED_KEY, _event("11", 1))
    runner._queue_or_replace_pending_event(SHARED_KEY, _event("11", 2))
    head = adapter._pending_messages[SHARED_KEY]
    assert (head.source.user_id, len(head.media_urls)) == ("11", 2)
    assert _queued(runner) == []


def test_photos_from_different_senders_are_not_merged():
    runner, adapter = _runner()
    runner._queue_or_replace_pending_event(SHARED_KEY, _event("11", 1))
    runner._queue_or_replace_pending_event(SHARED_KEY, _event("22", 2))
    head = adapter._pending_messages[SHARED_KEY]
    assert (head.source.user_id, len(head.media_urls)) == ("11", 1)
    assert _queued(runner) == [("22", 1, "")]


def test_caption_text_from_another_sender_is_not_folded_into_a_photo():
    runner, adapter = _runner()
    runner._queue_or_replace_pending_event(SHARED_KEY, _event("11", 1, text="my receipt"))
    runner._queue_or_replace_pending_event(SHARED_KEY, _event("22", 2, photo=False, text="ignore the receipt"))
    head = adapter._pending_messages[SHARED_KEY]
    assert (head.source.user_id, head.text, len(head.media_urls)) == ("11", "my receipt", 1)
    assert _queued(runner) == [("22", 0, "ignore the receipt")]


def _topic_photo(user_id, n):
    source = SessionSource(
        platform=Platform.TELEGRAM, chat_id="c1", chat_type="group", user_id=user_id, thread_id="7",
    )
    return MessageEvent(
        text="", message_type=MessageType.PHOTO, source=source,
        media_urls=[f"/tmp/{n}.jpg"], media_types=["image/jpeg"], message_id=f"m{n}",
    )


def _busy_interrupt_runner(session_key):
    """A runner with an agent already working on ``session_key``, in the default interrupt mode."""
    runner = object.__new__(GatewayRunner)
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._busy_ack_ts = {}
    runner._queued_events = {}
    runner._draining = False
    runner._busy_input_mode = "interrupt"
    runner._busy_text_mode = "interrupt"
    runner.config = MagicMock()
    runner.config.group_sessions_per_user = True
    runner.config.thread_sessions_per_user = False
    runner.session_store = None
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = True
    runner._is_user_authorized = lambda _source: True
    adapter = MagicMock()
    adapter._pending_messages = {}
    adapter._send_with_retry = AsyncMock()
    adapter.config = MagicMock()
    adapter.config.extra = {}
    adapter.platform = Platform.TELEGRAM
    adapter._text_debounce = {}
    adapter._busy_text_debounce_seconds = 0.6
    runner.adapters = {Platform.TELEGRAM: adapter}
    agent = MagicMock()
    agent.get_activity_summary.return_value = {
        "api_call_count": 3,
        "max_iterations": 60,
        "current_tool": "terminal",
        "last_activity_ts": time.time(),
        "last_activity_desc": "terminal",
        "seconds_since_activity": 1.0,
    }
    runner._running_agents[session_key] = agent
    runner._running_agents_ts[session_key] = time.time()
    return runner, adapter


@pytest.mark.asyncio
async def test_default_interrupt_mode_keeps_another_senders_photo_separate():
    # Interrupt (the default busy mode) only redirects plain text, so a photo is queued as the
    # next turn and reaches the same merge. Two people in one forum topic share the session key
    # under the default config.
    first, second = _topic_photo("11", 1), _topic_photo("22", 2)
    session_key = build_session_key(first.source)
    assert build_session_key(second.source) == session_key

    runner, adapter = _busy_interrupt_runner(session_key)
    assert await runner._handle_active_session_busy_message(first, session_key) is True
    assert await runner._handle_active_session_busy_message(second, session_key) is True

    head = adapter._pending_messages[session_key]
    assert (head.source.user_id, len(head.media_urls)) == ("11", 1)
    assert [
        (e.source.user_id, len(e.media_urls or []))
        for e in runner._queued_events.get(session_key, [])
    ] == [("22", 1)]
