"""Keep conversation data on its delivery path without incidental log copies."""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import PlatformConfig




@pytest.mark.asyncio
@pytest.mark.parametrize("accepted", [True, False])
async def test_email_dispatch_retains_identity_and_subject_without_log_copy(monkeypatch, caplog, accepted):
    from plugins.platforms.email.adapter import EmailAdapter

    sender = "private-sender@example.com"
    monkeypatch.setenv("EMAIL_ADDRESS", "private-bot@example.com")
    monkeypatch.setenv("EMAIL_ALLOWED_USERS", sender if accepted else "other@example.com")
    monkeypatch.delenv("EMAIL_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("GATEWAY_ALLOW_ALL_USERS", raising=False)
    with caplog.at_level(logging.DEBUG, logger="plugins.platforms.email.adapter"):
        adapter = EmailAdapter(PlatformConfig(enabled=True))
        adapter.handle_message = AsyncMock()
        await adapter._dispatch_message(dict(
            sender_addr=sender, sender_name="private name", subject="private subject\nprivate continuation",
            body="private email body", message_id="private-id", in_reply_to="", attachments=[], sender_authenticated=True))

    if accepted:
        event = adapter.handle_message.call_args.args[0]
        assert event.source.user_id == sender
        assert "private subject\nprivate continuation" in event.text
        assert "private email body" in event.text
        assert adapter._thread_context[sender]["message_id"] == "private-id"
    else:
        adapter.handle_message.assert_not_awaited()
        assert sender not in adapter._thread_context
    assert "private" not in caplog.text


@pytest.mark.asyncio
async def test_update_prompt_keeps_delivery_and_restart_marker_without_log_copy(caplog):
    from gateway.run_notifications import GatewayNotificationsMixin

    state = SimpleNamespace(persistent=SimpleNamespace(update_prompt_pending=False))
    runner = SimpleNamespace(_session_state=Mock(return_value=state))
    target = SimpleNamespace(adapter=SimpleNamespace(), session_key="private-session", send=AsyncMock())
    prompt = "private update question\nprivate continuation"
    with caplog.at_level(logging.INFO, logger="gateway.run"):
        await GatewayNotificationsMixin._forward_update_prompt(runner, target, prompt, "yes")

    assert prompt in target.send.call_args.args[0]
    runner._session_state.assert_called_once_with("private-session")
    assert state.persistent.update_prompt_pending is True
    assert "private" not in caplog.text


@pytest.mark.asyncio
async def test_duplicate_voice_transcript_is_suppressed_without_log_copy(caplog):
    from gateway.run_voice import GatewayVoiceMixin

    adapter = SimpleNamespace(_voice_text_channels={1: 2})
    source = object()
    runner = SimpleNamespace(
        _voice_input_source=Mock(return_value=source), _is_user_authorized_for_source=Mock(return_value=True),
        _is_duplicate_voice_transcript=Mock(return_value=True))
    transcript = "private transcript\nprivate continuation"
    with caplog.at_level(logging.INFO, logger="gateway.run"):
        await GatewayVoiceMixin._handle_voice_channel_input(runner, 1, 3, transcript, adapter=adapter)

    runner._is_user_authorized_for_source.assert_called_once_with(source)
    runner._is_duplicate_voice_transcript.assert_called_once_with(1, 3, transcript)
    assert "Suppressing duplicate voice transcript" in caplog.text
    assert "private" not in caplog.text


@pytest.mark.asyncio
async def test_voice_transcription_callback_retains_content_without_log_copy(monkeypatch, caplog):
    from plugins.platforms.discord import adapter as discord

    transcript = "private voice message\nprivate continuation"
    monkeypatch.setattr(discord.VoiceReceiver, "pcm_to_wav", Mock())
    monkeypatch.setattr("tools.transcription_tools.transcribe_audio", lambda path: {"success": True, "transcript": transcript})
    monkeypatch.setattr("tools.voice_mode.is_whisper_hallucination", lambda text: False)
    runner = SimpleNamespace(_voice_input_callback=AsyncMock())
    with caplog.at_level(logging.INFO, logger=discord.__name__):
        await discord.DiscordAdapter._process_voice_input(runner, 1, 2, b"pcm")
    runner._voice_input_callback.assert_awaited_once_with(guild_id=1, user_id=2, transcript=transcript)
    assert "private" not in caplog.text


@pytest.mark.parametrize("level", [logging.INFO, logging.WARNING])
def test_telegram_rejection_logs_keep_event_without_identity(caplog, level):
    from plugins.platforms.telegram.adapter import TelegramAdapter

    message = SimpleNamespace(from_user=SimpleNamespace(id=987654321), chat=SimpleNamespace(id=123456789))
    with caplog.at_level(logging.INFO, logger="plugins.platforms.telegram.adapter"):
        TelegramAdapter._log_blocked_user(object(), message, level=level)
    assert "Blocked unauthorized user" in caplog.text
    assert "987654321" not in caplog.text
    assert "123456789" not in caplog.text
