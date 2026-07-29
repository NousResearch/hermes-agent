"""Telegram voice-message captions must render markdown (#32029).

``TelegramAdapter.send_voice`` used to pass captions raw with no
``parse_mode``, so auto-TTS captions carrying the agent's markdown reply
showed literal *asterisks*, backticks and [links](...). It now formats the
caption to MarkdownV2 (when the formatted text fits the 1024-char caption
cap) and falls back to the plain truncated caption when the Bot API rejects
the entities or formatting overflows.
"""
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    mod = MagicMock()
    mod.error.NetworkError = type("NetworkError", (OSError,), {})
    mod.error.TimedOut = type("TimedOut", (OSError,), {})
    mod.error.BadRequest = type("BadRequest", (Exception,), {})
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("telegram.error", mod.error)


_ensure_telegram_mock()

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
async def test_voice_caption_gets_markdown_parse_mode(monkeypatch, tmp_path):
    """A markdown caption is MarkdownV2-formatted and sent with parse_mode."""
    monkeypatch.setattr(
        telegram_mod, "_probe_voice_duration_seconds", lambda _p: 3
    )
    adapter = _make_adapter()

    result = await adapter.send_voice(
        "123", str(_write_ogg(tmp_path)), caption="*bold* reply"
    )

    assert result.success is True
    adapter._bot.send_voice.assert_awaited_once()
    kwargs = adapter._bot.send_voice.await_args.kwargs
    assert kwargs["parse_mode"] is not None
    # format_message converts *bold* to MarkdownV2 bold; the raw caption must
    # have gone through formatting rather than being passed verbatim.
    assert kwargs["caption"] == adapter.format_message("*bold* reply")


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


@pytest.mark.asyncio
async def test_voice_caption_overflow_skips_formatting(monkeypatch, tmp_path):
    """When the formatted caption exceeds 1024 UTF-16 units, send plain
    (truncated) — MarkdownV2 escaping inflates length, and a truncated
    formatted caption could cut an entity in half."""
    monkeypatch.setattr(
        telegram_mod, "_probe_voice_duration_seconds", lambda _p: 3
    )
    adapter = _make_adapter()
    # Enough special chars that escaping pushes the formatted text over 1024.
    caption = ("hello. " * 160)[:1020]

    result = await adapter.send_voice("123", str(_write_ogg(tmp_path)), caption=caption)

    assert result.success is True
    kwargs = adapter._bot.send_voice.await_args.kwargs
    assert kwargs["parse_mode"] is None
    assert kwargs["caption"] == caption[:1024]


@pytest.mark.asyncio
async def test_voice_without_caption_unchanged(monkeypatch, tmp_path):
    monkeypatch.setattr(
        telegram_mod, "_probe_voice_duration_seconds", lambda _p: 3
    )
    adapter = _make_adapter()

    result = await adapter.send_voice("123", str(_write_ogg(tmp_path)))

    assert result.success is True
    kwargs = adapter._bot.send_voice.await_args.kwargs
    assert kwargs["caption"] is None
    assert kwargs["parse_mode"] is None
