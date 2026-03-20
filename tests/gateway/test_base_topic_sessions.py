"""Tests for BasePlatformAdapter topic-aware session handling."""

import asyncio
from collections import deque
import sys
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.session import SessionSource, build_session_key


class DummyTelegramAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="fake-token"), Platform.TELEGRAM)
        self.sent = []
        self.typing = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id="1")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        self.typing.append({"chat_id": chat_id, "metadata": metadata})
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


def _make_event(chat_id: str, thread_id: str, message_id: str = "1") -> MessageEvent:
    return MessageEvent(
        text="hello",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=chat_id,
            chat_type="group",
            thread_id=thread_id,
        ),
        message_id=message_id,
    )


class TestBasePlatformTopicSessions:
    @pytest.mark.asyncio
    async def test_handle_message_does_not_interrupt_different_topic(self, monkeypatch):
        adapter = DummyTelegramAdapter()
        adapter.set_message_handler(lambda event: asyncio.sleep(0, result=None))

        active_event = _make_event("-1001", "10")
        adapter._active_sessions[build_session_key(active_event.source)] = asyncio.Event()

        scheduled = []

        def fake_create_task(coro):
            scheduled.append(coro)
            coro.close()
            return SimpleNamespace()

        monkeypatch.setattr(asyncio, "create_task", fake_create_task)

        await adapter.handle_message(_make_event("-1001", "11"))

        assert len(scheduled) == 1
        assert adapter._pending_messages == {}

    @pytest.mark.asyncio
    async def test_handle_message_interrupts_same_topic(self, monkeypatch):
        adapter = DummyTelegramAdapter()
        adapter.set_message_handler(lambda event: asyncio.sleep(0, result=None))

        active_event = _make_event("-1001", "10")
        adapter._active_sessions[build_session_key(active_event.source)] = asyncio.Event()

        scheduled = []

        def fake_create_task(coro):
            scheduled.append(coro)
            coro.close()
            return SimpleNamespace()

        monkeypatch.setattr(asyncio, "create_task", fake_create_task)

        pending_event = _make_event("-1001", "10", message_id="2")
        await adapter.handle_message(pending_event)

        assert scheduled == []
        assert adapter._active_sessions[build_session_key(pending_event.source)].is_set() is False
        assert adapter.get_pending_message(build_session_key(pending_event.source)) == pending_event

    @pytest.mark.asyncio
    async def test_handle_message_stop_sets_interrupt_event(self, monkeypatch):
        adapter = DummyTelegramAdapter()
        adapter.set_message_handler(lambda event: asyncio.sleep(0, result=None))

        active_event = _make_event("-1001", "10")
        session_key = build_session_key(active_event.source)
        adapter._active_sessions[session_key] = asyncio.Event()

        scheduled = []

        def fake_create_task(coro):
            scheduled.append(coro)
            coro.close()
            return SimpleNamespace()

        monkeypatch.setattr(asyncio, "create_task", fake_create_task)

        interrupt_event = MessageEvent(
            text="/stop",
            source=active_event.source,
            message_id="2",
        )
        await adapter.handle_message(interrupt_event)

        assert scheduled == []
        assert adapter.has_pending_interrupt(session_key) is True

    @pytest.mark.asyncio
    async def test_process_message_background_replies_in_same_topic(self):
        adapter = DummyTelegramAdapter()
        typing_calls = []

        async def handler(_event):
            await asyncio.sleep(0)
            return "ack"

        async def hold_typing(_chat_id, interval=2.0, metadata=None):
            typing_calls.append({"chat_id": _chat_id, "metadata": metadata})
            await asyncio.Event().wait()

        adapter.set_message_handler(handler)
        adapter._keep_typing = hold_typing

        event = _make_event("-1001", "17585")
        await adapter._process_message_background(event, build_session_key(event.source))

        assert adapter.sent == [
            {
                "chat_id": "-1001",
                "content": "ack",
                "reply_to": "1",
                "metadata": {"thread_id": "17585"},
            }
        ]
        assert typing_calls == [
            {
                "chat_id": "-1001",
                "metadata": {"thread_id": "17585"},
            }
        ]

    @pytest.mark.asyncio
    async def test_process_message_background_drains_large_pending_queue_iteratively(self):
        adapter = DummyTelegramAdapter()

        async def handler(event):
            await asyncio.sleep(0)
            return f"ack:{event.text}"

        async def hold_typing(_chat_id, interval=2.0, metadata=None):
            await asyncio.Event().wait()

        adapter.set_message_handler(handler)
        adapter._keep_typing = hold_typing

        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1001",
            chat_type="group",
            thread_id="17585",
        )
        session_key = build_session_key(source)
        first_event = MessageEvent(text="msg-0", source=source, message_id="0")
        adapter._pending_messages[session_key] = deque(
            MessageEvent(text=f"msg-{idx}", source=source, message_id=str(idx))
            for idx in range(1, 41)
        )

        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(60)
        try:
            await adapter._process_message_background(first_event, session_key)
        finally:
            sys.setrecursionlimit(old_limit)

        assert len(adapter.sent) == 41
        assert adapter.sent[0]["content"] == "ack:msg-0"
        assert adapter.sent[-1]["content"] == "ack:msg-40"

    @pytest.mark.asyncio
    async def test_pending_queue_drops_oldest_when_full(self, monkeypatch):
        """When the pending queue hits _MAX_PENDING_MESSAGES, oldest entries are dropped."""
        adapter = DummyTelegramAdapter()
        adapter.set_message_handler(lambda event: asyncio.sleep(0, result=None))

        source = SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1001",
            chat_type="group",
            thread_id="10",
        )
        session_key = build_session_key(source)
        adapter._active_sessions[session_key] = asyncio.Event()

        scheduled = []

        def fake_create_task(coro):
            scheduled.append(coro)
            coro.close()
            return SimpleNamespace()

        monkeypatch.setattr(asyncio, "create_task", fake_create_task)

        # Fill the queue to capacity + 2 extra
        cap = adapter._MAX_PENDING_MESSAGES
        for idx in range(cap + 2):
            event = MessageEvent(
                text=f"msg-{idx}",
                source=source,
                message_id=str(idx + 100),
            )
            await adapter.handle_message(event)

        queue = adapter._pending_messages[session_key]
        assert len(queue) == cap
        # The first two messages (msg-0, msg-1) should have been dropped
        assert queue[0].text == "msg-2"
        assert queue[-1].text == f"msg-{cap + 1}"
