"""Regression tests for the Discord voice inactivity timer re-arm on user input.

Refs #105974: ``_process_voice_input`` (the STT path that hears the user) never
called ``_reset_voice_timeout()``, so an active two-way voice conversation
dropped exactly ``voice_channel_inactivity_timeout_seconds`` after the bot
joined. The bot's own TTS playback already re-armed via
``play_in_voice_channel``'s ``finally``; the listening path did not.
"""

from unittest.mock import patch

import pytest

from plugins.platforms.discord.adapter import DiscordAdapter

pytestmark = pytest.mark.asyncio


def _make_adapter():
    """Build a DiscordAdapter without the heavy __init__ (see existing fixtures)."""
    adapter = object.__new__(DiscordAdapter)
    adapter._voice_timeout_tasks = {}
    adapter._voice_input_callback = None
    return adapter


async def test_process_voice_input_re_arms_inactivity_timeout(monkeypatch):
    """When the user is heard (valid transcript), the STT path must re-arm the
    auto-disconnect inactivity timer so an active conversation does not drop
    (refs #105974)."""
    adapter = _make_adapter()
    reset_calls = []
    adapter._reset_voice_timeout = lambda guild_id: reset_calls.append(guild_id)

    # Stub the STT pipeline so _process_voice_input sees a real, non-hallucinated
    # transcript without touching ffmpeg / whisper.
    monkeypatch.setattr(
        "plugins.platforms.discord.adapter.VoiceReceiver.pcm_to_wav",
        lambda pcm, wav_path: None,
    )
    monkeypatch.setattr(
        "tools.transcription_tools.transcribe_audio",
        lambda wav_path: {"success": True, "transcript": "hello there"},
    )
    monkeypatch.setattr(
        "tools.voice_mode.is_whisper_hallucination", lambda t: False)

    await adapter._process_voice_input(guild_id=42, user_id=7, pcm_data=b"\x00\x00")

    assert reset_calls == [42], (
        "_process_voice_input must re-arm the voice inactivity timer when the "
        "user is heard (refs #105974); got calls={}".format(reset_calls)
    )


async def test_process_voice_input_does_not_rearm_on_failed_stt(monkeypatch):
    """The re-arm must be gated on a valid transcript — a failed/empty STT must
    NOT reset the timer (that would keep a silent bot connected indefinitely)."""
    adapter = _make_adapter()
    reset_calls = []
    adapter._reset_voice_timeout = lambda guild_id: reset_calls.append(guild_id)

    monkeypatch.setattr(
        "plugins.platforms.discord.adapter.VoiceReceiver.pcm_to_wav",
        lambda pcm, wav_path: None,
    )
    monkeypatch.setattr(
        "tools.transcription_tools.transcribe_audio",
        lambda wav_path: {"success": False, "transcript": ""},
    )
    monkeypatch.setattr(
        "tools.voice_mode.is_whisper_hallucination", lambda t: False)

    await adapter._process_voice_input(guild_id=99, user_id=7, pcm_data=b"\x00\x00")

    assert reset_calls == [], (
        "failed STT must not re-arm the inactivity timer; got calls={}".format(
            reset_calls)
    )
