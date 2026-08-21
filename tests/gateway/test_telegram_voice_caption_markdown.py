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

from gateway.config import PlatformConfig
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


