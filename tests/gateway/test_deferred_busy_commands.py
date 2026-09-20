"""Deferred slash commands preserve command semantics across a busy turn boundary."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource, build_session_key
from hermes_cli.commands import resolve_command


class _Adapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)
        self.sent = []

    async def connect(self, *, is_reconnect: bool = False):
        pass

    async def disconnect(self):
        pass

    async def send(self, chat_id, text, **kwargs):
        self.sent.append((chat_id, text, kwargs))

    async def get_chat_info(self, chat_id):
        return {}


def _event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="chat",
            chat_type="dm",
            thread_id="thread",
        ),
    )


def test_deferred_commands_precede_prompts_and_coalesce_only_when_declared():
    adapter = _Adapter()
    key = build_session_key(_event("prompt").source)
    undo = _event("/undo")
    compress = _event("/compress here 4")
    prompt = _event("ordinary prompt")

    assert {resolve_command(name).busy_policy for name in ("compress", "undo", "retry", "save")} == {
        "defer_until_idle"
    }
    assert adapter.defer_command_until_idle(key, undo, command_name="undo") == ("queued", 1)
    assert adapter.defer_command_until_idle(
        key, compress, command_name="compress", coalesce=True
    ) == ("queued", 2)
    assert adapter.defer_command_until_idle(
        key, _event("/compact here 4"), command_name="compress", coalesce=True
    ) == ("coalesced", 2)
    adapter._pending_messages[key] = prompt

    assert [adapter._pop_next_session_event(key) for _ in range(4)] == [undo, compress, prompt, None]


@pytest.mark.asyncio
async def test_active_guard_defers_through_delivery_and_shutdown_reports_cancellation():
    adapter = _Adapter()
    event = _event("/save md")
    key = build_session_key(event.source)
    adapter._active_sessions[key] = asyncio.Event()
    adapter._message_handler = AsyncMock(return_value="must not run inline")

    async def defer_busy(incoming, session_key):
        adapter.defer_command_until_idle(session_key, incoming, command_name="save")
        return True

    adapter._busy_session_handler = AsyncMock(side_effect=defer_busy)
    await adapter.handle_message(event)

    adapter._busy_session_handler.assert_awaited_once_with(event, key)
    adapter._message_handler.assert_not_awaited()
    assert adapter._deferred_commands[key] == [event]

    await adapter.cancel_background_tasks()

    assert adapter._deferred_commands == {}
    assert len(adapter.sent) == 1
    assert "/save" in adapter.sent[0][1]
    assert "did not run" in adapter.sent[0][1]
    assert adapter.sent[0][2]["metadata"]["thread_id"] == "thread"