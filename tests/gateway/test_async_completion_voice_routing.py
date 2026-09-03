"""Regression coverage for auto-TTS delivery after synthetic completions."""

import json
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _runner(adapter):
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.DISCORD: adapter}
    return runner


def _synthetic_event():
    return MessageEvent(
        text="background completion",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="123",
            user_id="user-1",
            chat_type="channel",
        ),
        message_id="completion-1",
        raw_message=None,
    )


def _fake_tts(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

    def fake_text_to_speech(*, text, output_path, **_kwargs):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as audio:
            audio.write(b"audio")
        return json.dumps({"success": True, "file_path": output_path})

    monkeypatch.setattr("tools.tts_tool.text_to_speech_tool", fake_text_to_speech)
    monkeypatch.setattr("tools.tts_tool._strip_markdown_for_tts", lambda text: text)


@pytest.mark.asyncio
async def test_synthetic_completion_routes_tts_through_adapter_play_tts(monkeypatch, tmp_path):
    """A synthetic event has no guild raw_message but still uses adapter routing."""
    play_tts = AsyncMock()
    send_voice = AsyncMock()
    adapter = SimpleNamespace(
        play_tts=play_tts,
        send_voice=send_voice,
        is_in_voice_channel=lambda _guild_id: False,
    )
    _fake_tts(monkeypatch, tmp_path)

    await _runner(adapter)._send_voice_reply(_synthetic_event(), "Spoken result")

    play_tts.assert_awaited_once()
    kwargs = play_tts.await_args.kwargs
    assert kwargs["chat_id"] == "123"
    assert kwargs["reply_to"] == "completion-1"
    assert kwargs["metadata"] == {"notify": True}
    send_voice.assert_not_awaited()


@pytest.mark.asyncio
async def test_synthetic_completion_reaches_real_discord_bound_voice_channel(
    monkeypatch, tmp_path
):
    """Discord resolves a synthetic completion's text chat to its bound VC."""
    from gateway.config import PlatformConfig
    from plugins.platforms.discord.adapter import DiscordAdapter

    adapter = object.__new__(DiscordAdapter)
    adapter.platform = Platform.DISCORD
    adapter.config = PlatformConfig(enabled=True, extra={})
    adapter._voice_text_channels = {456: 123}
    voice_client = MagicMock()
    voice_client.is_connected.return_value = True
    adapter._voice_clients = {456: voice_client}
    adapter.play_in_voice_channel = AsyncMock(return_value=True)
    adapter.send_voice = AsyncMock()
    _fake_tts(monkeypatch, tmp_path)

    await _runner(adapter)._send_voice_reply(_synthetic_event(), "Spoken result")

    adapter.play_in_voice_channel.assert_awaited_once()
    guild_id, audio_path = adapter.play_in_voice_channel.await_args.args
    assert guild_id == 456
    assert audio_path.endswith(".mp3")
    adapter.send_voice.assert_not_awaited()


@pytest.mark.asyncio
async def test_raw_guild_completion_keeps_direct_voice_channel_playback(monkeypatch, tmp_path):
    """Direct raw-guild delivery remains ahead of adapter auto-TTS routing."""
    play_in_voice_channel = AsyncMock()
    play_tts = AsyncMock()
    send_voice = AsyncMock()
    adapter = SimpleNamespace(
        play_in_voice_channel=play_in_voice_channel,
        play_tts=play_tts,
        send_voice=send_voice,
        is_in_voice_channel=lambda guild_id: guild_id == 456,
    )
    event = _synthetic_event()
    event.raw_message = SimpleNamespace(guild_id=456, guild=None)
    _fake_tts(monkeypatch, tmp_path)

    await _runner(adapter)._send_voice_reply(event, "Spoken result")

    play_in_voice_channel.assert_awaited_once()
    guild_id, audio_path = play_in_voice_channel.await_args.args
    assert guild_id == 456
    assert audio_path.endswith(".mp3")
    play_tts.assert_not_awaited()
    send_voice.assert_not_awaited()


@pytest.mark.asyncio
async def test_synthetic_completion_uses_play_tts_when_adapter_must_fallback(monkeypatch, tmp_path):
    """The runner delegates no-binding fallback to the adapter play_tts seam."""
    send_voice = AsyncMock()
    play_in_voice_channel = AsyncMock()

    async def _fallback_to_attachment(**kwargs):
        await send_voice(**kwargs)

    play_tts = AsyncMock(side_effect=_fallback_to_attachment)
    adapter = SimpleNamespace(
        play_in_voice_channel=play_in_voice_channel,
        play_tts=play_tts,
        send_voice=send_voice,
        is_in_voice_channel=lambda _guild_id: False,
    )
    _fake_tts(monkeypatch, tmp_path)

    await _runner(adapter)._send_voice_reply(_synthetic_event(), "Spoken result")

    play_tts.assert_awaited_once()
    play_in_voice_channel.assert_not_awaited()
    send_voice.assert_awaited_once()
    assert send_voice.await_args.kwargs["metadata"] == {"notify": True}


@pytest.mark.asyncio
async def test_adapter_without_play_tts_keeps_send_voice_compatibility(monkeypatch, tmp_path):
    """Unusual legacy adapters retain their attachment fallback and notify metadata."""
    send_voice = AsyncMock()
    adapter = SimpleNamespace(
        send_voice=send_voice,
        is_in_voice_channel=lambda _guild_id: False,
    )
    _fake_tts(monkeypatch, tmp_path)

    await _runner(adapter)._send_voice_reply(_synthetic_event(), "Spoken result")

    send_voice.assert_awaited_once()
    assert send_voice.await_args.kwargs["chat_id"] == "123"
    assert send_voice.await_args.kwargs["metadata"] == {"notify": True}
