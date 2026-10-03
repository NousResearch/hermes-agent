"""Telegram ``Voice_messages_forbidden`` is an expected outcome, not a fault.

A recipient who disabled voice messages in their privacy settings makes every
``sendVoice`` fail with ``BadRequest: Voice_messages_forbidden``. ``send_voice``
already falls back to the base adapter and the reply is delivered, so the
rejection must not be logged as ERROR with a library traceback; genuine send
failures keep ERROR + traceback.
"""
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram.error import BadRequest

from gateway.config import PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from plugins.platforms.telegram import adapter as telegram_mod
from plugins.platforms.telegram.adapter import TelegramAdapter


async def _send_voice_failing_with(error, monkeypatch, tmp_path, caplog):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra={}))
    adapter._bot = MagicMock()
    adapter._send_voice_bubble = AsyncMock(side_effect=error)
    fallback = AsyncMock(return_value=SendResult(success=True, message_id="fallback"))
    monkeypatch.setattr(BasePlatformAdapter, "send_voice", fallback)
    monkeypatch.setattr(telegram_mod, "_probe_voice_duration_seconds", lambda _p: None)
    clip = tmp_path / "reply.ogg"
    clip.write_bytes(b"OggS")
    with caplog.at_level(logging.DEBUG, logger=telegram_mod.logger.name):
        result = await adapter.send_voice("123", str(clip))
    fallback.assert_awaited_once()
    assert result.success is True
    return [r for r in caplog.records if r.name == telegram_mod.logger.name]


@pytest.mark.asyncio
async def test_voice_messages_forbidden_falls_back_without_error_or_traceback(monkeypatch, tmp_path, caplog):
    records = await _send_voice_failing_with(BadRequest("Voice_messages_forbidden"), monkeypatch, tmp_path, caplog)

    assert not [r for r in records if r.levelno >= logging.ERROR]
    assert not [r for r in records if r.exc_info]
    assert any("voice messages" in r.getMessage().lower() for r in records)


@pytest.mark.asyncio
async def test_other_voice_send_failure_still_logs_error_with_traceback(monkeypatch, tmp_path, caplog):
    records = await _send_voice_failing_with(RuntimeError("boom"), monkeypatch, tmp_path, caplog)

    errors = [r for r in records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert errors[0].exc_info
