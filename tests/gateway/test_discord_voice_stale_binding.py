"""An utterance is dispatched only into the conversation it was captured for.

``_process_voice_input`` awaits WAV conversion and STT, then the gateway callback reads the guild's
binding as it is at that moment. A ``/voice join`` from another text channel during transcription
rewrote the binding, and the old utterance became a turn in the new conversation, with its prompt,
skills and replies. The listen loop transcribes a poll batch serially, so a later utterance of the
same batch must keep the binding of the batch, not one set during an earlier utterance's STT.
Speech the receiver still buffers when the binding moves is dropped too: a poll or leave flush
after the move must not stamp it with the new binding.
"""
from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SessionSource
from gateway.platforms.event import MessageEvent, MessageType
from plugins.platforms.discord.adapter import DiscordAdapter, VoiceReceiver

_GUILD, _USER = 1, 42


class _OneBatchReceiver:
    """One check_silence() batch of two utterances, both completed while bound to channel 700."""

    def __init__(self) -> None:
        self._running = True

    def check_silence(self) -> list[tuple[int, bytes]]:
        self._running = False
        return [(_USER, b"\x00" * 9600), (_USER + 1, b"\x00" * 9600)]


@pytest.mark.asyncio
@pytest.mark.parametrize("bound_after,dispatched", [
    (700, 2),    # unchanged binding (also leave + rejoin from the same channel)
    (800, 0),    # /voice join from another text channel during the first utterance's STT
])
async def test_transcribed_utterance_keeps_its_captured_binding(bound_after: int, dispatched: int) -> None:
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake"))
    adapter._voice_text_channels = {_GUILD: 700}
    adapter._voice_receivers[_GUILD] = _OneBatchReceiver()
    adapter._voice_input_callback = callback = AsyncMock()
    adapter._is_allowed_user = lambda *a, **k: True
    adapter._reset_voice_timeout = lambda *a: None
    started, release = threading.Event(), threading.Event()

    def transcribe(_path: str) -> dict[str, object]:
        started.set()
        release.wait(5)
        return {"success": True, "transcript": "what broke on the ingest box"}

    with patch("plugins.platforms.discord.adapter.VoiceReceiver.pcm_to_wav"), \
         patch("tools.transcription_tools.transcribe_audio", side_effect=transcribe):
        task = asyncio.ensure_future(adapter._voice_listen_loop(_GUILD))
        while not started.is_set():
            await asyncio.sleep(0.01)
        adapter._voice_text_channels[_GUILD] = bound_after
        release.set()
        await task

    assert callback.await_count == dispatched


_SSRC, _BYTES_PER_SECOND = 7, 48000 * 2 * 2


def _speak(receiver: VoiceReceiver, seconds: float) -> None:
    """Decoded PCM arriving for one speaker, as the packet hook appends it."""
    with receiver._lock:
        receiver._buffers[_SSRC].extend(b"\x01\x00" * int(seconds * _BYTES_PER_SECOND / 2))
        receiver._last_packet_time[_SSRC] = time.monotonic()


def _fall_silent(receiver: VoiceReceiver) -> None:
    receiver._last_packet_time[_SSRC] = time.monotonic() - 2 * receiver.SILENCE_THRESHOLD


async def _voice_join_from(adapter: DiscordAdapter, chat_id: str, tmp_path: Path) -> None:
    """``/voice join`` through the gateway handler while the bot is already in the voice channel."""
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.adapters, runner._voice_mode = {}, {}
    runner._VOICE_MODE_PATH = tmp_path / "gateway_voice_mode.json"
    runner._session_db, runner.session_store = None, MagicMock()
    runner._is_user_authorized = MagicMock(return_value=True)
    platform = MagicMock()
    platform.value = "discord"
    source = SessionSource(chat_id=chat_id, user_id="owner", platform=platform)
    source.thread_id = None
    event = MessageEvent(text="/voice join", message_type=MessageType.TEXT, source=source)
    event.raw_message = SimpleNamespace(guild_id=_GUILD, guild=None)
    runner.adapters[platform] = adapter
    adapter.get_user_voice_channel = AsyncMock(return_value=SimpleNamespace(name="General"))
    adapter.join_voice_channel = AsyncMock(return_value=True)
    await runner._handle_voice_channel_join(event)


def _listening_adapter() -> tuple[DiscordAdapter, VoiceReceiver]:
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake"))
    receiver = VoiceReceiver(SimpleNamespace())
    receiver._ssrc_to_user[_SSRC] = _USER
    adapter._voice_receivers[_GUILD] = receiver
    adapter._voice_text_channels = {_GUILD: 700}
    adapter._is_allowed_user = MagicMock(return_value=True)
    adapter._reset_voice_timeout = MagicMock()
    return adapter, receiver


async def _transcribed_pcm(adapter: DiscordAdapter, drain: Callable[[], Awaitable[None]]) -> tuple[int, list[int]]:
    """Run *drain* with STT stubbed; return (callback dispatches, PCM length of each converted utterance)."""
    converted: list[int] = []
    adapter._voice_input_callback = callback = AsyncMock()
    with patch("plugins.platforms.discord.adapter.VoiceReceiver.pcm_to_wav",
               side_effect=lambda pcm, path: converted.append(len(pcm))), \
         patch("tools.transcription_tools.transcribe_audio",
               return_value={"success": True, "transcript": "what broke on the ingest box"}):
        await drain()
    return callback.await_count, converted


async def _one_poll(adapter: DiscordAdapter, receiver: VoiceReceiver) -> None:
    receiver._running = True
    check = receiver.check_silence

    def once() -> list[tuple[int, bytes]]:
        receiver._running = False
        return check()
    receiver.check_silence = MagicMock(side_effect=once)
    await adapter._voice_listen_loop(_GUILD)


@pytest.mark.asyncio
@pytest.mark.parametrize("join_from,reaching", [
    ("800", []),                                  # another text channel: the buffered speech is dropped
    ("700", [int(0.6 * _BYTES_PER_SECOND)]),     # same channel rejoin keeps it
])
async def test_speech_buffered_across_a_rebind_is_not_stamped_with_the_new_binding(
    join_from: str, reaching: list[int], tmp_path: Path,
) -> None:
    adapter, receiver = _listening_adapter()
    _speak(receiver, 0.6)
    await _voice_join_from(adapter, join_from, tmp_path)
    _fall_silent(receiver)
    assert await _transcribed_pcm(adapter, lambda: _one_poll(adapter, receiver)) == (len(reaching), reaching)


@pytest.mark.asyncio
async def test_speech_spanning_a_rebind_reaches_the_new_channel_without_the_earlier_samples(
    tmp_path: Path,
) -> None:
    adapter, receiver = _listening_adapter()
    _speak(receiver, 0.3)
    await _voice_join_from(adapter, "800", tmp_path)
    _speak(receiver, 0.6)
    _fall_silent(receiver)
    assert await _transcribed_pcm(adapter, lambda: _one_poll(adapter, receiver)) == (1, [int(0.6 * _BYTES_PER_SECOND)])


@pytest.mark.asyncio
@pytest.mark.parametrize("join_from,reaching", [
    ("800", []),
    ("700", [int(0.6 * _BYTES_PER_SECOND)]),
])
async def test_leave_after_a_rebind_does_not_flush_old_speech_into_the_new_channel(
    join_from: str, reaching: list[int], tmp_path: Path,
) -> None:
    adapter, receiver = _listening_adapter()
    _speak(receiver, 0.6)  # still speaking: the leave flush, not silence, emits it
    await _voice_join_from(adapter, join_from, tmp_path)
    assert await _transcribed_pcm(adapter, lambda: adapter.leave_voice_channel(_GUILD)) == (len(reaching), reaching)
