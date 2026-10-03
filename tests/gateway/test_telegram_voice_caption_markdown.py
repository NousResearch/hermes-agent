"""Telegram voice-message captions must render markdown (#32029).

``TelegramAdapter.send_voice`` used to pass captions raw with no
``parse_mode``, so auto-TTS captions carrying the agent's markdown reply
showed literal *asterisks*, backticks and [links](...). It now formats the
caption to MarkdownV2 (when the formatted text fits the 1024-char caption
cap) and falls back to the plain truncated caption when the Bot API rejects
the entities or formatting overflows.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import utf16_len
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from plugins.platforms.telegram import adapter as telegram_mod  # noqa: E402
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


def _make_adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._bot = MagicMock()
    adapter._bot.send_voice = AsyncMock(return_value=MagicMock(message_id=1))
    adapter._bot.send_audio = AsyncMock(return_value=MagicMock(message_id=2))
    return adapter


def _write_ogg(tmp_path):
    audio = tmp_path / "reply.ogg"
    audio.write_bytes(b"\x00" * 16)
    return audio


@pytest.mark.asyncio
async def test_voice_caption_falls_back_to_plain_on_entity_rejection(
    monkeypatch, tmp_path
):
    """A Bot API entity-parse rejection retries with the plain caption."""
    monkeypatch.setattr(
        telegram_mod, "_probe_voice_duration_seconds", lambda _p: 3
    )
    adapter = _make_adapter()
    adapter._bot.send_voice = AsyncMock(
        side_effect=[
            Exception("Bad Request: can't parse entities"),
            MagicMock(message_id=7),
        ]
    )

    result = await adapter.send_voice(
        "123", str(_write_ogg(tmp_path)), caption="*bold* reply"
    )

    assert result.success is True
    assert adapter._bot.send_voice.await_count == 2
    retry_kwargs = adapter._bot.send_voice.await_args_list[1].kwargs
    assert retry_kwargs["parse_mode"] is None
    assert retry_kwargs["caption"] == "*bold* reply"


def _bot_enforcing_caption_cap(adapter):
    """sendVoice that refuses captions past Telegram's 1024 UTF-16-unit cap."""
    async def send_voice(**kwargs):
        if utf16_len(kwargs.get("caption") or "") > 1024:
            raise Exception("Bad Request: message caption is too long")
        return MagicMock(message_id=9)
    adapter._bot.send_voice = AsyncMock(side_effect=send_voice)


@pytest.mark.asyncio
async def test_auto_tts_reply_over_the_utf16_cap_keeps_the_voice_and_sends_text_once(
    monkeypatch, tmp_path
):
    """An emoji-heavy reply under 1024 code points but over 1024 UTF-16 units must not ride as
    the caption: the voice bubble is still delivered and the text goes out on the normal path."""
    monkeypatch.setattr(telegram_mod, "_probe_voice_duration_seconds", lambda _p: 3)
    adapter = _make_adapter()
    _bot_enforcing_caption_cap(adapter)
    adapter.send = AsyncMock()
    reply = "Plan \U0001F680 " * 120 + "\U0001F389" * 100  # 940 code points, 1180 UTF-16 units
    event = MessageEvent(
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="123", user_id="7", chat_type="dm"),
        text="(voice)", message_type=MessageType.VOICE)
    receipts = []

    caption_carried_text = await adapter._play_tts_file(
        event, reply, str(_write_ogg(tmp_path)), True, {}, receipts.append)

    assert caption_carried_text is False  # so the base adapter sends the text itself
    assert receipts and receipts[0].success is True  # the voice bubble was delivered
    adapter.send.assert_not_awaited()  # no fallback notice carrying the text


def test_media_caption_truncation_fits_the_utf16_cap():
    caption = "\U0001F389" * 700  # 700 code points, 1400 UTF-16 units
    truncated = TelegramAdapter._caption_1024(caption)
    assert utf16_len(truncated) <= 1024 and caption.startswith(truncated) and len(truncated) == 512


