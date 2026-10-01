"""An utterance is dispatched only into the conversation it was captured for.

``_process_voice_input`` awaits WAV conversion and STT, then the gateway callback reads the guild's
binding as it is at that moment. A ``/voice join`` from another text channel during transcription
rewrote the binding, and the old utterance became a turn in the new conversation, with its prompt,
skills and replies.
"""

import asyncio
import threading
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter

_GUILD, _USER = 1, 42


@pytest.mark.asyncio
@pytest.mark.parametrize("bound_after,dispatched", [
    (700, 1),    # unchanged binding (also leave + rejoin from the same channel)
    (800, 0),    # /voice join from another text channel while transcribing
])
async def test_transcribed_utterance_keeps_its_captured_binding(bound_after, dispatched):
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fake"))
    adapter._voice_text_channels = {_GUILD: 700}
    adapter._voice_input_callback = callback = AsyncMock()
    started, release = threading.Event(), threading.Event()

    def transcribe(_path):
        started.set()
        release.wait(5)
        return {"success": True, "transcript": "what broke on the ingest box"}

    with patch("plugins.platforms.discord.adapter.VoiceReceiver.pcm_to_wav"), \
         patch("tools.transcription_tools.transcribe_audio", side_effect=transcribe):
        task = asyncio.ensure_future(adapter._process_voice_input(_GUILD, _USER, b"\x00" * 9600))
        while not started.is_set():
            await asyncio.sleep(0.01)
        adapter._voice_text_channels[_GUILD] = bound_after
        release.set()
        await task

    assert callback.await_count == dispatched
