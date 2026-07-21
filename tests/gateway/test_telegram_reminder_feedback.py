"""Telegram cron-reminder feedback buttons and durable response logging."""

import asyncio
import json
import stat
import sys
from concurrent.futures import Future
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cron.scheduler import _deliver_result
from gateway.config import Platform, PlatformConfig
from plugins.platforms.telegram.adapter import (
    TelegramAdapter,
    _append_reminder_feedback_event,
    _feedback_resolved_text,
    _validated_reminder_feedback,
)
from tools.send_message_tool import _send_telegram


def _make_adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token", extra={}))
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


def _feedback():
    return {
        "prompt": "How did it go?",
        "choices": [
            {"code": "call", "label": "📞 Called"},
            {"code": "email", "label": "✉️ Emailed"},
            {"code": "skip", "label": "❌ Didn't do it"},
        ],
    }


@pytest.mark.asyncio
async def test_cron_feedback_metadata_renders_inline_keyboard_on_final_chunk(monkeypatch):
    import plugins.platforms.telegram.adapter as telegram_adapter

    class FakeButton:
        def __init__(self, text, callback_data):
            self.text = text
            self.callback_data = callback_data

    class FakeMarkup:
        def __init__(self, rows):
            self.inline_keyboard = rows

    monkeypatch.setattr(telegram_adapter, "InlineKeyboardButton", FakeButton)
    monkeypatch.setattr(telegram_adapter, "InlineKeyboardMarkup", FakeMarkup)
    adapter = _make_adapter()
    msg = MagicMock(message_id=42)
    adapter._bot.send_message = AsyncMock(return_value=msg)

    result = await adapter.send(
        "12345",
        "Follow up with a contact.",
        metadata={"job_id": "abc123", "reminder_feedback": _feedback()},
    )

    assert result.success is True
    kwargs = adapter._bot.send_message.call_args.kwargs
    markup = kwargs["reply_markup"]
    buttons = [button for row in markup.inline_keyboard for button in row]
    assert [button.text for button in buttons] == ["📞 Called", "✉️ Emailed", "❌ Didn't do it"]
    assert [button.callback_data for button in buttons] == [
        "rf:abc123:call",
        "rf:abc123:email",
        "rf:abc123:skip",
    ]


@pytest.mark.asyncio
async def test_feedback_keyboard_is_only_on_final_chunk(monkeypatch):
    import plugins.platforms.telegram.adapter as telegram_adapter

    class FakeButton:
        def __init__(self, text, callback_data):
            self.text = text
            self.callback_data = callback_data

    class FakeMarkup:
        def __init__(self, rows):
            self.inline_keyboard = rows

    monkeypatch.setattr(telegram_adapter, "InlineKeyboardButton", FakeButton)
    monkeypatch.setattr(telegram_adapter, "InlineKeyboardMarkup", FakeMarkup)
    adapter = _make_adapter()
    adapter._bot.send_message = AsyncMock(
        side_effect=[MagicMock(message_id=1), MagicMock(message_id=2)]
    )

    result = await adapter.send(
        "12345",
        "x" * 5000,
        metadata={"job_id": "abc123", "reminder_feedback": _feedback()},
    )

    assert result.success is True
    calls = adapter._bot.send_message.await_args_list
    assert len(calls) == 2
    assert calls[0].kwargs["reply_markup"] is None
    assert calls[-1].kwargs["reply_markup"] is not None


def test_feedback_callback_payload_rejects_non_ascii_job_id():
    assert _validated_reminder_feedback(
        {"job_id": "🧠" * 32, "reminder_feedback": _feedback()}
    ) is None


def test_feedback_resolution_respects_telegram_utf16_limit():
    from gateway.platforms.base import utf16_len

    resolved = _feedback_resolved_text("😀" * 4096, "Done")
    assert utf16_len(resolved) <= 4096
    assert resolved.endswith("Feedback: Done")


@pytest.mark.asyncio
async def test_standalone_telegram_sender_renders_feedback_keyboard(monkeypatch):
    import plugins.platforms.telegram.adapter as telegram_adapter

    class FakeButton:
        def __init__(self, text, callback_data):
            self.text = text
            self.callback_data = callback_data

    class FakeMarkup:
        def __init__(self, rows):
            self.inline_keyboard = rows

    monkeypatch.setattr(telegram_adapter, "InlineKeyboardButton", FakeButton)
    monkeypatch.setattr(telegram_adapter, "InlineKeyboardMarkup", FakeMarkup)
    bot = MagicMock()
    bot.send_message = AsyncMock(return_value=MagicMock(message_id=42))
    monkeypatch.setattr(sys.modules["telegram"], "Bot", lambda **_kwargs: bot)

    result = await _send_telegram(
        "token",
        "12345",
        "Follow up with a contact.",
        metadata={"job_id": "abc123", "reminder_feedback": _feedback()},
    )

    assert result["success"] is True
    kwargs = bot.send_message.await_args.kwargs
    assert kwargs["reply_markup"].inline_keyboard
    assert "How did it go?" in kwargs["text"]


def test_feedback_messages_do_not_use_rich_send_path():
    adapter = _make_adapter()
    adapter._rich_messages_enabled = True
    adapter._rich_send_disabled = False
    adapter._bot.do_api_request = AsyncMock()
    assert adapter._should_attempt_rich(
        "**Reminder**",
        metadata={"job_id": "abc123", "reminder_feedback": _feedback()},
    ) is False


def test_scheduler_passes_job_feedback_to_live_adapter():
    adapter = AsyncMock()
    adapter.send.return_value = MagicMock(success=True)

    pconfig = MagicMock(enabled=True)
    mock_cfg = MagicMock()
    mock_cfg.platforms = {Platform.TELEGRAM: pconfig}
    loop = MagicMock()
    loop.is_running.return_value = True

    def fake_run_coro(coro, _loop):
        future = Future()
        try:
            future.set_result(asyncio.run(coro))
        except BaseException as exc:  # noqa: BLE001
            future.set_exception(exc)
        return future

    job = {
        "id": "abc123",
        "name": "Eric follow-up",
        "deliver": "origin",
        "origin": {"platform": "telegram", "chat_id": "12345"},
        "feedback": _feedback(),
    }
    with patch("gateway.config.load_gateway_config", return_value=mock_cfg), patch(
        "cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}
    ), patch("asyncio.run_coroutine_threadsafe", side_effect=fake_run_coro):
        result = _deliver_result(
            job,
            "Follow up with a contact.",
            adapters={Platform.TELEGRAM: adapter},
            loop=loop,
        )

    assert result is None
    adapter.send.assert_called_once()
    metadata = adapter.send.call_args.kwargs["metadata"]
    assert metadata["job_id"] == "abc123"
    assert metadata["reminder_feedback"] == _feedback()


def test_scheduler_passes_job_feedback_to_standalone_sender():
    pconfig = MagicMock(enabled=True)
    mock_cfg = MagicMock()
    mock_cfg.platforms = {Platform.TELEGRAM: pconfig}
    sender = AsyncMock(return_value={"success": True})
    job = {
        "id": "abc123",
        "name": "Eric follow-up",
        "deliver": "origin",
        "origin": {"platform": "telegram", "chat_id": "12345"},
        "feedback": _feedback(),
    }
    with patch("gateway.config.load_gateway_config", return_value=mock_cfg), patch(
        "cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}
    ), patch("tools.send_message_tool._send_to_platform", sender):
        result = _deliver_result(job, "Follow up with a contact.", adapters=None, loop=None)

    assert result is None
    sender.assert_awaited_once()
    metadata = sender.call_args.kwargs["metadata"]
    assert metadata["job_id"] == "abc123"
    assert metadata["reminder_feedback"] == _feedback()


@pytest.mark.asyncio
async def test_feedback_callback_logs_event_and_resolves_message(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "cron.jobs.get_job", lambda job_id: {"id": job_id, "feedback": _feedback()}
    )
    adapter = _make_adapter()
    adapter._is_callback_user_authorized = MagicMock(return_value=True)

    sent_at = datetime(2026, 7, 21, 14, 0, tzinfo=timezone.utc)
    query = AsyncMock()
    query.data = "rf:abc123:email"
    query.from_user = MagicMock(id=1001, first_name="Tester")
    query.message = MagicMock(
        chat_id=-1001234567890,
        message_id=777,
        message_thread_id=1410,
        text="Follow up with a contact.",
        date=sent_at,
    )
    query.message.chat = MagicMock(type="supergroup")

    update = MagicMock(callback_query=query)
    await adapter._handle_callback_query(update, MagicMock())

    ledger = tmp_path / "reminder-feedback" / "events.jsonl"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["job_id"] == "abc123"
    assert rows[0]["action"] == "email"
    assert rows[0]["telegram_message_id"] == 777
    assert rows[0]["telegram_user_id"] == "1001"
    assert rows[0]["thread_id"] == "1410"
    assert rows[0]["response_seconds"] >= 0
    query.answer.assert_awaited_once()
    query.edit_message_reply_markup.assert_awaited_once_with(reply_markup=None)
    query.edit_message_text.assert_awaited_once()


@pytest.mark.asyncio
async def test_unauthorized_feedback_callback_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "cron.jobs.get_job", lambda job_id: {"id": job_id, "feedback": _feedback()}
    )
    adapter = _make_adapter()
    adapter._is_callback_user_authorized = MagicMock(return_value=False)

    query = AsyncMock()
    query.data = "rf:abc123:call"
    query.from_user = MagicMock(id=999, first_name="Intruder")
    query.message = MagicMock(chat_id=-1001234567890, message_id=1, message_thread_id=1410)
    query.message.chat = MagicMock(type="supergroup")

    await adapter._handle_callback_query(MagicMock(callback_query=query), MagicMock())

    assert not (tmp_path / "reminder-feedback" / "events.jsonl").exists()
    query.answer.assert_awaited_once()
    query.edit_message_text.assert_not_awaited()


@pytest.mark.asyncio
async def test_unconfigured_feedback_action_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(
        "cron.jobs.get_job", lambda job_id: {"id": job_id, "feedback": _feedback()}
    )
    adapter = _make_adapter()
    adapter._is_callback_user_authorized = MagicMock(return_value=True)
    query = AsyncMock()
    query.data = "rf:abc123:forged"
    query.from_user = MagicMock(id=1001)
    query.message = MagicMock(chat_id=-1001234567890, message_id=2)
    query.message.chat = MagicMock(type="supergroup")

    await adapter._handle_callback_query(MagicMock(callback_query=query), MagicMock())

    assert not (tmp_path / "reminder-feedback" / "events.jsonl").exists()
    adapter._is_callback_user_authorized.assert_called_once()
    query.answer.assert_awaited_once_with(text="Invalid reminder response.")


def test_duplicate_feedback_event_is_not_appended(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    event = {
        "job_id": "abc123",
        "telegram_chat_id": "-1001",
        "telegram_message_id": 77,
        "telegram_user_id": "42",
        "action": "email",
    }
    assert _append_reminder_feedback_event(event) is True
    assert _append_reminder_feedback_event({**event, "action": "call"}) is False
    ledger = tmp_path / "reminder-feedback" / "events.jsonl"
    assert len(ledger.read_text().splitlines()) == 1
    assert stat.S_IMODE(ledger.stat().st_mode) == 0o600
