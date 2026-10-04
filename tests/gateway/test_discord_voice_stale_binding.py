"""An utterance is dispatched only into the conversation it was captured for.

``_process_voice_input`` awaits WAV conversion and STT, then the gateway callback reads the guild's
binding as it is at that moment. A ``/voice join`` from another text channel during transcription
rewrote the binding, and the old utterance became a turn in the new conversation, with its prompt,
skills and replies. The listen loop transcribes a poll batch serially, so a later utterance of the
same batch must keep the binding of the batch, not one set during an earlier utterance's STT.
"""

import asyncio
import threading
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter

_GUILD, _USER = 1, 42


class _OneBatchReceiver:
    """One check_silence() batch of two utterances, both completed while bound to channel 700."""

    def __init__(self):
        self._running = True

    def check_silence(self):
        self._running = False
        return [(_USER, b"\x00" * 9600), (_USER + 1, b"\x00" * 9600)]


@pytest.mark.asyncio
@pytest.mark.parametrize("bound_after,dispatched", [
    (700, 2),    # unchanged binding (also leave + rejoin from the same channel)
    (800, 0),    # /voice join from another text channel during the first utterance's STT
])
async def test_transcribed_utterance_keeps_its_captured_binding(bound_after, dispatched):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake"))
    adapter._voice_text_channels = {_GUILD: 700}
    adapter._voice_receivers[_GUILD] = _OneBatchReceiver()
    adapter._voice_input_callback = callback = AsyncMock()
    adapter._is_allowed_user = lambda *a, **k: True
    adapter._reset_voice_timeout = lambda *a: None
    started, release = threading.Event(), threading.Event()

    def transcribe(_path):
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
