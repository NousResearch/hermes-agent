"""Tests for Discord adapter realtime voice wiring."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_receiver():
    from gateway.platforms.discord import VoiceReceiver

    mock_vc = MagicMock()
    mock_vc._connection.secret_key = [0] * 32
    mock_vc._connection.dave_session = None
    mock_vc._connection.ssrc = 9999
    mock_vc._connection.add_socket_listener = MagicMock()
    mock_vc._connection.remove_socket_listener = MagicMock()
    mock_vc._connection.hook = None
    mock_vc.channel.members = []
    return VoiceReceiver(mock_vc)


def _make_adapter():
    from gateway.config import Platform, PlatformConfig
    from gateway.platforms.discord import DiscordAdapter

    adapter = object.__new__(DiscordAdapter)
    adapter.platform = Platform.DISCORD
    adapter.config = PlatformConfig(enabled=True, extra={})
    adapter._client = MagicMock()
    adapter._voice_clients = {}
    adapter._voice_locks = {}
    adapter._voice_text_channels = {}
    adapter._voice_sources = {}
    adapter._voice_timeout_tasks = {}
    adapter._voice_receivers = {}
    adapter._voice_listen_tasks = {}
    adapter._realtime_voice_sessions = {}
    adapter._realtime_audio_sources = {}
    adapter._voice_input_callback = None
    adapter._on_voice_disconnect = None
    adapter._allowed_user_ids = set()
    return adapter


@pytest.mark.asyncio
async def test_voice_receiver_realtime_callback_receives_pcm():
    receiver = _make_receiver()
    received: list[tuple[int, bytes]] = []
    loop = asyncio.get_running_loop()

    receiver.map_ssrc(100, 42)
    receiver.set_realtime_callback(loop, lambda user_id, pcm: received.append((user_id, pcm)))
    receiver._handle_decoded_pcm(100, b"pcm")

    await asyncio.sleep(0)

    assert received == [(42, b"pcm")]


@pytest.mark.asyncio
async def test_legacy_voice_receiver_buffering_still_works():
    receiver = _make_receiver()
    loop = asyncio.get_running_loop()

    receiver.map_ssrc(100, 42)
    receiver.set_realtime_callback(loop, lambda _user_id, _pcm: None)
    receiver._handle_decoded_pcm(100, b"pcm")

    assert receiver._buffers[100] == bytearray(b"pcm")
    assert 100 in receiver._last_packet_time


@pytest.mark.asyncio
async def test_leave_voice_stops_realtime_session():
    adapter = _make_adapter()
    mock_session = AsyncMock()
    mock_source = MagicMock()
    mock_vc = MagicMock()
    mock_vc.is_connected.return_value = True
    mock_vc.disconnect = AsyncMock()
    adapter._realtime_voice_sessions[111] = mock_session
    adapter._realtime_audio_sources[111] = mock_source
    adapter._voice_clients[111] = mock_vc

    await adapter.leave_voice_channel(111)

    mock_session.stop.assert_awaited_once()
    mock_source.close.assert_called_once()
    mock_vc.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_join_realtime_voice_channel_starts_session_and_receiver(monkeypatch):
    adapter = _make_adapter()
    mock_vc = MagicMock()
    mock_vc.is_connected.return_value = True
    mock_vc.is_playing.return_value = False
    mock_vc.play = MagicMock()
    channel = MagicMock()
    channel.guild.id = 111
    channel.connect = AsyncMock(return_value=mock_vc)
    context = SimpleNamespace(config=MagicMock())

    mock_session = AsyncMock()
    session_cls = MagicMock(return_value=mock_session)
    monkeypatch.setattr("gateway.platforms.discord.RealtimeVoiceSession", session_cls)

    assert await adapter.join_realtime_voice_channel(channel, context) is True

    mock_vc.play.assert_called_once()
    mock_session.start.assert_awaited_once()
    assert 111 in adapter._realtime_voice_sessions
    assert 111 in adapter._realtime_audio_sources
    assert 111 in adapter._voice_receivers
    assert 111 not in adapter._voice_listen_tasks
