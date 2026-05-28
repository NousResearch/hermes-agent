import asyncio
from unittest.mock import ANY, AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
)
from gateway.session import SessionSource


class _BoardAdapter(BasePlatformAdapter):
    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="sent")

    async def get_chat_info(self, chat_id):
        return {}


def _make_adapter(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_BOARD_COMMS_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("HERMES_BOARD_CHAT_IDS", "-1003817293915")
    adapter = _BoardAdapter(PlatformConfig(enabled=True, token="t"), Platform.TELEGRAM)
    adapter._keep_typing = AsyncMock()
    adapter._run_processing_hook = AsyncMock()
    adapter._send_with_retry = AsyncMock(return_value=SendResult(success=True, message_id="sent"))
    return adapter


def _board_event(text="ping", *, is_bot=False, message_type=MessageType.TEXT):
    return MessageEvent(
        text=text,
        message_type=message_type,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1003817293915",
            chat_type="group",
            thread_id="1",
            user_id="u1",
            user_name="Tester",
            is_bot=is_bot,
        ),
        message_id="m1",
    )


@pytest.mark.asyncio
async def test_board_ack_only_response_suppressed_at_send_boundary(monkeypatch, tmp_path):
    adapter = _make_adapter(tmp_path, monkeypatch)

    async def handler(event):
        return "Acknowledged."

    adapter._message_handler = handler

    await adapter._process_message_background(_board_event(), "session-key")

    adapter._send_with_retry.assert_not_called()
    adapter._run_processing_hook.assert_any_await(
        "on_processing_complete",
        ANY,
        ProcessingOutcome.SUCCESS,
    )
    assert (tmp_path / "suppression.jsonl").exists()


@pytest.mark.asyncio
async def test_board_voice_ack_response_does_not_bypass_via_tts(monkeypatch, tmp_path):
    adapter = _make_adapter(tmp_path, monkeypatch)
    adapter.play_tts = AsyncMock()
    monkeypatch.setattr(adapter, "_should_auto_tts_for_chat", lambda chat_id: True)

    async def handler(event):
        return "Acknowledged."

    adapter._message_handler = handler

    await adapter._process_message_background(
        _board_event(message_type=MessageType.VOICE),
        "session-key",
    )

    adapter.play_tts.assert_not_called()
    adapter._send_with_retry.assert_not_called()
    adapter._run_processing_hook.assert_any_await(
        "on_processing_complete",
        ANY,
        ProcessingOutcome.SUCCESS,
    )


@pytest.mark.asyncio
async def test_board_status_response_allowed_at_send_boundary(monkeypatch, tmp_path):
    adapter = _make_adapter(tmp_path, monkeypatch)

    async def handler(event):
        return "Status: Foundry Hermes is degraded; cleanup routed."

    adapter._message_handler = handler

    await adapter._process_message_background(_board_event(), "session-key")

    adapter._send_with_retry.assert_awaited_once()
