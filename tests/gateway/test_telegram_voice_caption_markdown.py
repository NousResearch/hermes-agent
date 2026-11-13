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

from hermes_agent.gateway.config import Platform, PlatformConfig
from hermes_agent.gateway.platforms.event import MessageEvent, MessageType
from hermes_agent.gateway.session import SessionSource
from hermes_agent.plugins.platforms.telegram import adapter as telegram_mod
from hermes_agent.plugins.platforms.telegram.adapter import TelegramAdapter


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
    """sendVoice that refuses captions the way tdlib does: more than 1024 code points (utf8_length)."""
    async def send_voice(**kwargs):
        if len(kwargs.get("caption") or "") > 1024:
            raise Exception("Bad Request: message caption is too long")
        return MagicMock(message_id=9)
    adapter._bot.send_voice = AsyncMock(side_effect=send_voice)


@pytest.mark.asyncio
@pytest.mark.parametrize("reply, rides_as_caption", [
    ("Plan \U0001F680 " * 120 + "\U0001F389" * 100, True),  # 940 code points, 1180 UTF-16 units
    ("a" * 1023 + "\U0001F389", True),  # 1024 code points
    ("a" * 1024 + "\U0001F389", False),  # 1025 code points
], ids=["emoji_heavy_fits", "at_cap", "one_over"])
async def test_auto_tts_caption_gate_counts_code_points(monkeypatch, tmp_path, reply, rides_as_caption):
    """Telegram caps a caption at 1024 code points, so an emoji-heavy reply that fits rides as
    the voice caption, and one code point over sends the voice bare (the text goes out normally)."""
    monkeypatch.setattr(telegram_mod, "_probe_voice_duration_seconds", lambda _p: 3)
    adapter = _make_adapter()
    _bot_enforcing_caption_cap(adapter)
    adapter.send = AsyncMock()
    event = MessageEvent(
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="123", user_id="7", chat_type="dm"),
        text="(voice)", message_type=MessageType.VOICE)
    receipts = []

    caption_carried_text = await adapter._play_tts_file(
        event, reply, str(_write_ogg(tmp_path)), True, {}, receipts.append)

    assert caption_carried_text is rides_as_caption
    assert receipts and receipts[0].success is True
    assert adapter._bot.send_voice.await_args.kwargs["caption"] == (reply if rides_as_caption else None)
    adapter.send.assert_not_awaited()


def test_media_caption_truncation_keeps_1024_code_points():
    caption = "\U0001F389" * 1100  # 1100 code points, 2200 UTF-16 units
    assert TelegramAdapter._caption_1024(caption) == "\U0001F389" * 1024
