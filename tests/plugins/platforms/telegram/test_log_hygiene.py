"""Polling records stay redacted; voice privacy rejections keep the text fallback.

Combines the behavior coverage from #129996 and #129997.
"""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from plugins.platforms.telegram import adapter as telegram_mod
from plugins.platforms.telegram.adapter import TelegramAdapter

_FAKE_TOKEN = "123456789:AAFakeSecretTelegramBotTokenABCDEFGHIJ"
_FAKE_URL = f"https://api.telegram.org/bot{_FAKE_TOKEN}/getUpdates"


class BadRequest(Exception):
    """Stand-in for telegram.error.BadRequest; python-telegram-bot is an optional extra."""


@pytest.mark.asyncio
async def test_polling_error_record_has_no_raw_exception(monkeypatch, caplog):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token=_FAKE_TOKEN, extra={}))
    monkeypatch.setattr(adapter, "_delete_webhook_best_effort", AsyncMock())
    monkeypatch.setattr(adapter, "_start_polling_resilient", AsyncMock(return_value=True))
    await adapter._start_polling_mode(is_reconnect=False)

    with caplog.at_level(logging.ERROR, logger=telegram_mod.logger.name):
        # exc_info=True picks up this active exception, including its unredacted URL.
        try:
            raise RuntimeError(f"Bad Request: {_FAKE_URL}")
        except RuntimeError as error:
            adapter._polling_error_callback_ref(error)

    records = [r for r in caplog.records if "Telegram polling error" in r.getMessage()]
    assert len(records) == 1
    record = records[0]
    rendered = logging.Formatter("%(levelname)s %(message)s").format(record)
    assert _FAKE_TOKEN not in rendered
    assert _FAKE_TOKEN not in record.getMessage()
    assert _FAKE_TOKEN not in str(record.args)
    assert _FAKE_TOKEN not in (record.exc_text or "")
    assert not record.exc_info
    assert "RuntimeError" in rendered
    assert "***" in rendered


@pytest.mark.asyncio
async def test_voice_forbidden_info_preserves_fallback_and_other_errors(monkeypatch, tmp_path, caplog):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token=_FAKE_TOKEN, extra={}))
    adapter._bot = MagicMock()
    monkeypatch.setattr(telegram_mod, "_probe_voice_duration_seconds", lambda _p: None)
    monkeypatch.setattr(adapter, "warning_notifications_enabled", lambda *a, **k: True)
    receipt = SendResult(success=True, message_id="fallback")
    adapter.send = AsyncMock(return_value=receipt)
    clip = tmp_path / "reply.ogg"
    clip.write_bytes(b"OggS")
    metadata = {"thread_id": "42"}

    for error, level, traceback in (
        (BadRequest(f"Voice_messages_forbidden: {_FAKE_URL}"), logging.INFO, False),
        (RuntimeError("boom"), logging.ERROR, True),
    ):
        caplog.clear()
        adapter.send.reset_mock()
        adapter._send_voice_bubble = AsyncMock(side_effect=error)
        with caplog.at_level(logging.DEBUG, logger=telegram_mod.logger.name):
            result = await adapter.send_voice(
                "123", str(clip), caption="Caption", reply_to="7", metadata=metadata)

        assert result is receipt
        adapter.send.assert_awaited_once()
        args = adapter.send.await_args
        assert args.args[0] == "123"
        assert args.args[1].startswith("Caption\n")
        assert "audio" in args.args[1].lower()
        assert str(clip) not in args.args[1]
        assert args.kwargs == {"reply_to": "7", "metadata": metadata}
        records = [r for r in caplog.records if r.name == telegram_mod.logger.name]
        assert len(records) == 1
        record = records[0]
        assert record.levelno == level
        assert bool(record.exc_info) is traceback
        rendered = logging.Formatter("%(message)s").format(record)
        assert _FAKE_TOKEN not in rendered
        assert _FAKE_TOKEN not in str(record.args)
        assert _FAKE_TOKEN not in (record.exc_text or "")
        if level == logging.INFO:
            assert "voice_messages_forbidden" in record.getMessage().lower()
