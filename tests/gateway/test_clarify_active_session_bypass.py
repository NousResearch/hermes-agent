"""Regression tests for clarify replies while a gateway session is busy."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource, build_session_key


class _ClarifyBypassAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)

    async def connect(self):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="text")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "private"}


def _event(text="custom answer"):
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_type="private",
            user_id="user1",
        ),
        message_id="msg1",
    )


def _clear_clarify_state():
    from tools import clarify_gateway as cm

    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()


@pytest.mark.asyncio
async def test_active_session_routes_typed_choice_clarify_reply_to_runner_not_busy_queue():
    """Typed text must resolve a pending choice clarify even while the agent is busy.

    Telegram button clarifies keep the adapter session active while the agent
    thread blocks on ``wait_for_response``.  If the adapter only bypasses for
    entries already marked ``awaiting_text``, typed replies to the visible
    multi-choice prompt are handled as busy follow-ups and the clarify wait is
    never resolved.
    """
    _clear_clarify_state()
    from tools import clarify_gateway as cm

    adapter = _ClarifyBypassAdapter()
    adapter._message_handler = AsyncMock(return_value="")
    adapter._busy_session_handler = AsyncMock(return_value=True)
    event = _event("None of those are valid options")
    session_key = build_session_key(
        event.source,
        group_sessions_per_user=adapter.config.extra.get("group_sessions_per_user", True),
        thread_sessions_per_user=adapter.config.extra.get("thread_sessions_per_user", False),
    )
    adapter._active_sessions[session_key] = asyncio.Event()
    cm.register("clarify-1", session_key, "Pick one", ["A", "B"])

    await adapter.handle_message(event)

    adapter._message_handler.assert_awaited_once_with(event)
    adapter._busy_session_handler.assert_not_awaited()
    assert adapter._pending_messages == {}


@pytest.mark.asyncio
async def test_gateway_clarify_reply_resumes_typing_before_returning_empty_ack():
    """A clarify answer must re-enable the active run's typing indicator.

    Clarify pauses typing while waiting so Slack's Assistant API does not
    disable the compose box. The typed answer is intercepted by the gateway
    and returns an empty acknowledgment instead of starting a second run; that
    interception path must therefore resume the original run's indicator.
    """
    _clear_clarify_state()
    from gateway.run import GatewayRunner
    from tools import clarify_gateway as cm

    adapter = _ClarifyBypassAdapter()
    adapter.pause_typing_for_chat("12345")
    event = _event("the missing details")

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._startup_restore_in_progress = False
    runner._scale_to_zero_note_real_inbound = lambda: None
    runner._is_user_authorized = lambda source: True
    runner._session_key_for_source = lambda source: "clarify-session"
    runner._adapter_for_source = lambda source: adapter
    runner._update_prompt_pending = {}

    cm.register("clarify-2", "clarify-session", "What is missing?", None)

    with patch("hermes_cli.plugins.invoke_hook", return_value=[]):
        result = await runner._handle_message(event)

    assert result == ""
    assert "12345" not in adapter._typing_paused
