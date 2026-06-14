import asyncio

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.runtime_guard import GuardDecision, register_runtime_guard_provider
from gateway.session import SessionSource, build_session_key


class _RecordingAdapter(BasePlatformAdapter):
    def __init__(self, config: PlatformConfig | None = None):
        super().__init__(config or PlatformConfig(enabled=True), Platform.TELEGRAM)
        self.started: list[tuple[MessageEvent, str]] = []
        self.sent: list[dict] = []

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content="", reply_to=None, metadata=None, **kwargs):
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id=f"sent-{len(self.sent)}")

    async def send_typing(self, chat_id, metadata=None):
        pass

    async def stop_typing(self, chat_id):
        pass

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "dm"}

    def _start_session_processing(
        self,
        event: MessageEvent,
        session_key: str,
        *,
        interrupt_event: asyncio.Event | None = None,
    ) -> bool:
        self.started.append((event, session_key))
        return True


def _event(text: str = "hello") -> MessageEvent:
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="chat-1",
        chat_type="dm",
        user_id="user-1",
        thread_id="thread-1",
        parent_chat_id="parent-1",
        guild_id="guild-1",
    )
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id="message-1",
    )


def _blocking_runtime_guard_extra(provider: str) -> dict:
    return {
        "runtime_guard": {
            "enabled": True,
            "provider": provider,
            "dry_run": False,
            "fail_closed": True,
            "scope": {"platforms": ["telegram"], "chat_ids": ["chat-1"]},
        }
    }


def test_runtime_guard_disabled_allows_inbound_processing():
    adapter = _RecordingAdapter()
    adapter._message_handler = lambda event: None
    event = _event()

    asyncio.run(adapter.handle_message(event))

    assert adapter.started == [
        (
            event,
            build_session_key(
                event.source,
                group_sessions_per_user=True,
                thread_sessions_per_user=False,
            ),
        )
    ]


def test_runtime_guard_enabled_blocks_inbound_before_processing():
    seen = []

    class DenyingGuard:
        def check(self, context):
            seen.append(context)
            return GuardDecision.block(reason="lease_conflict", status="denied")

    register_runtime_guard_provider("phase26_inbound_deny", DenyingGuard())
    adapter = _RecordingAdapter(
        PlatformConfig(
            enabled=True,
            extra=_blocking_runtime_guard_extra("phase26_inbound_deny"),
        )
    )
    adapter._message_handler = lambda event: None
    event = _event()

    asyncio.run(adapter.handle_message(event))

    assert adapter.started == []
    assert len(seen) == 1
    assert seen[0].surface == "inbound_message"
    assert seen[0].platform == Platform.TELEGRAM
    assert seen[0].chat_id == "chat-1"
    assert seen[0].thread_id == "thread-1"
    assert seen[0].parent_chat_id == "parent-1"
    assert seen[0].guild_id == "guild-1"
    assert seen[0].user_id == "user-1"
    assert seen[0].message_id == "message-1"
    assert seen[0].session_key == build_session_key(event.source)


@pytest.mark.asyncio
async def test_runtime_guard_disabled_allows_outbound_send():
    adapter = _RecordingAdapter()

    async def handler(event):
        return "visible response"

    adapter._message_handler = handler
    event = _event()
    session_key = build_session_key(event.source)

    await adapter._process_message_background(event, session_key)

    assert [call["content"] for call in adapter.sent] == ["visible response"]


@pytest.mark.asyncio
async def test_runtime_guard_enabled_blocks_outbound_final_send():
    seen = []

    class DenyingGuard:
        def check(self, context):
            seen.append(context)
            return GuardDecision.block(reason="lease_conflict", status="denied")

    register_runtime_guard_provider("phase26_outbound_deny", DenyingGuard())
    adapter = _RecordingAdapter(
        PlatformConfig(
            enabled=True,
            extra=_blocking_runtime_guard_extra("phase26_outbound_deny"),
        )
    )

    async def handler(event):
        return "blocked response"

    adapter._message_handler = handler
    event = _event()
    session_key = build_session_key(event.source)

    await adapter._process_message_background(event, session_key)

    assert adapter.sent == []
    assert len(seen) == 1
    assert seen[0].surface == "assistant_final"
    assert seen[0].session_key == session_key


@pytest.mark.asyncio
async def test_runtime_guard_invalid_outbound_provider_decision_blocks_without_error_send():
    class InvalidGuard:
        def check(self, context):
            return object()

    register_runtime_guard_provider("phase28_invalid_outbound", InvalidGuard())
    adapter = _RecordingAdapter(
        PlatformConfig(
            enabled=True,
            extra=_blocking_runtime_guard_extra("phase28_invalid_outbound"),
        )
    )

    async def handler(event):
        return "should not send"

    adapter._message_handler = handler
    event = _event()
    session_key = build_session_key(event.source)

    await adapter._process_message_background(event, session_key)

    assert adapter.sent == []


@pytest.mark.asyncio
async def test_runtime_guard_blocks_handler_error_fallback_send():
    class DenyingGuard:
        def check(self, context):
            return GuardDecision.block(reason="lease_conflict", status="denied")

    register_runtime_guard_provider("phase28_error_fallback_deny", DenyingGuard())
    adapter = _RecordingAdapter(
        PlatformConfig(
            enabled=True,
            extra=_blocking_runtime_guard_extra("phase28_error_fallback_deny"),
        )
    )

    async def handler(event):
        raise RuntimeError("boom")

    adapter._message_handler = handler
    event = _event()
    session_key = build_session_key(event.source)

    await adapter._process_message_background(event, session_key)

    assert adapter.sent == []
