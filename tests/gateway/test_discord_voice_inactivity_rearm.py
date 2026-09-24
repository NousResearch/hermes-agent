"""Discord voice: hearing the user must re-arm the inactivity auto-leave timer.

``_reset_voice_timeout`` is armed on join and re-armed when the bot itself plays
audio (TTS / ack).  Listening to the user is activity too: any user audio that
reaches ``_process_voice_input`` must re-arm the timer — including audio whose
STT result is missing/failed/hallucinated, because the user was talking either
way.  Without it an auto-joined bot drops mid-conversation exactly
``voice_channel_inactivity_timeout_seconds`` after its own last playback.

Silence must stay non-activity: with nobody talking the timer still fires and
the bot leaves (that is the protection the timer exists for), and join / play /
ack keep re-arming as they always did.
"""

import asyncio
import contextlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_adapter():
    from plugins.platforms.discord.adapter import DiscordAdapter
    from gateway.config import Platform, PlatformConfig

    config = PlatformConfig(enabled=True, extra={})
    config.token = "fake-token"
    adapter = object.__new__(DiscordAdapter)
    adapter.platform = Platform.DISCORD
    adapter.config = config
    adapter._client = MagicMock()
    adapter._voice_clients = {}
    adapter._voice_locks = {}
    adapter._voice_text_channels = {}
    adapter._voice_sources = {}
    adapter._voice_timeout_tasks = {}
    adapter._voice_receivers = {}
    adapter._voice_listen_tasks = {}
    adapter._voice_mixers = {}
    adapter._voice_input_callback = None
    adapter._on_voice_disconnect = None
    adapter._voice_mode_getter = None
    adapter._allowed_user_ids = set()
    adapter._voice_timeout_seconds = 300
    adapter._running = True
    return adapter


def _spy_rearm(adapter):
    """Record ``_reset_voice_timeout`` calls while keeping the real behaviour."""
    calls = []
    real = adapter._reset_voice_timeout

    def _spy(guild_id):
        calls.append(guild_id)
        return real(guild_id)

    adapter._reset_voice_timeout = _spy
    return calls


async def _drain(task):
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


def _join_timer(adapter, guild_id=111):
    """Stand-in for the timer task armed when the bot joined the channel."""
    timer = MagicMock()
    adapter._voice_timeout_tasks[guild_id] = timer
    return timer


# 0.5s of 48kHz stereo s16 — one real utterance (== VoiceReceiver.MIN_SPEECH_DURATION).
PCM_UTTERANCE = b"\x00" * 96000


class TestUserVoiceInputCountsAsActivity:
    """① User audio processed as voice input re-arms the inactivity timer."""

    @pytest.mark.asyncio
    async def test_voice_input_rearms_inactivity_timer(self):
        adapter = _make_adapter()
        adapter._voice_input_callback = AsyncMock()
        join_timer = _join_timer(adapter)

        with patch("plugins.platforms.discord.adapter.VoiceReceiver.pcm_to_wav"), \
             patch("tools.transcription_tools.transcribe_audio",
                   return_value={"success": True, "transcript": "Can you hear me now?"}), \
             patch("tools.voice_mode.is_whisper_hallucination", return_value=False):
            await adapter._process_voice_input(111, 42, PCM_UTTERANCE)

        # RED on the buggy code: the join-time timer is never touched by voice input.
        assert join_timer.cancel.call_count == 1, (
            "user voice input must re-arm the inactivity timer: the join-time "
            "timer task was never cancelled (bot drops mid-conversation)"
        )
        armed = adapter._voice_timeout_tasks.get(111)
        assert armed is not None and armed is not join_timer, (
            "user voice input must install a fresh inactivity timer"
        )
        assert not armed.done()
        adapter._voice_input_callback.assert_awaited_once_with(
            guild_id=111, user_id=42, transcript="Can you hear me now?"
        )
        await _drain(armed)

    @pytest.mark.parametrize(
        "stt_result,hallucination",
        [
            ({"success": False}, False),                           # STT backend failed
            ({"success": True, "transcript": "Thank you."}, True),  # whisper hallucination
            ({"success": True, "transcript": "   "}, False),        # empty transcript
        ],
        ids=["stt_failure", "hallucination", "empty_transcript"],
    )
    @pytest.mark.asyncio
    async def test_audio_stt_discards_still_rearms(self, stt_result, hallucination):
        """The sibling exits inside the same path: the user was talking anyway."""
        adapter = _make_adapter()
        adapter._voice_input_callback = AsyncMock()
        join_timer = _join_timer(adapter)

        with patch("plugins.platforms.discord.adapter.VoiceReceiver.pcm_to_wav"), \
             patch("tools.transcription_tools.transcribe_audio", return_value=stt_result), \
             patch("tools.voice_mode.is_whisper_hallucination", return_value=hallucination):
            await adapter._process_voice_input(111, 42, PCM_UTTERANCE)

        assert join_timer.cancel.call_count == 1, (
            "audio that STT drops (failure/hallucination/empty) is still the user "
            "talking and must re-arm the inactivity timer"
        )
        armed = adapter._voice_timeout_tasks.get(111)
        assert armed is not None and armed is not join_timer
        adapter._voice_input_callback.assert_not_awaited()
        await _drain(armed)


class TestSilenceStillLeaves:
    """② Silence is not activity: no re-arm, and the timer still disconnects."""

    @pytest.mark.asyncio
    async def test_silence_does_not_rearm_timer(self):
        adapter = _make_adapter()
        receiver = MagicMock()
        receiver._running = True
        receiver.check_silence = MagicMock(return_value=[])  # nobody speaks
        adapter._voice_receivers[111] = receiver
        join_timer = _join_timer(adapter)
        calls = _spy_rearm(adapter)

        loop = asyncio.create_task(adapter._voice_listen_loop(111))
        await asyncio.sleep(0.5)  # several 0.2s poll iterations
        await _drain(loop)

        assert calls == [], "silence must not re-arm the inactivity timer"
        assert adapter._voice_timeout_tasks[111] is join_timer
        join_timer.cancel.assert_not_called()

    @pytest.mark.asyncio
    async def test_armed_timer_still_leaves_on_silence(self):
        adapter = _make_adapter()
        adapter._voice_timeout_seconds = 1
        adapter._voice_text_channels = {}  # no "Left voice channel" notice to deliver
        adapter.leave_voice_channel = AsyncMock()

        adapter._reset_voice_timeout(111)
        await asyncio.sleep(2.0)

        adapter.leave_voice_channel.assert_awaited_once_with(111)


class TestExistingRearmCallSitesUnchanged:
    """③ Join / play / ack keep re-arming exactly as before."""

    @pytest.mark.asyncio
    async def test_join_voice_channel_rearms(self):
        from plugins.platforms.discord import adapter as discord_mod

        adapter = _make_adapter()
        join_timer = _join_timer(adapter, guild_id=42)
        calls = _spy_rearm(adapter)

        class _FakeVC:
            def __init__(self, channel):
                self.channel = channel

            def is_connected(self):
                return True

        channel = MagicMock()
        channel.id = 111
        channel.guild.id = 42

        async def _connect():
            return _FakeVC(channel)

        channel.connect = _connect

        with patch.object(discord_mod, "VoiceReceiver",
                          MagicMock(return_value=MagicMock(start=MagicMock()))):
            assert await adapter.join_voice_channel(channel) is True

        assert calls == [42]
        assert join_timer.cancel.call_count == 1
        armed = adapter._voice_timeout_tasks[42]
        assert armed is not join_timer
        await _drain(armed)
        await _drain(adapter._voice_listen_tasks.get(42))

    @pytest.mark.asyncio
    async def test_play_in_voice_channel_rearms(self):
        adapter = _make_adapter()
        adapter._voice_timeout_seconds = 0  # re-arm installs no task; the call itself is recorded
        calls = _spy_rearm(adapter)
        vc = MagicMock()
        vc.is_connected.return_value = True
        vc.is_playing.return_value = False
        adapter._voice_clients[111] = vc
        adapter._playback_timeout_for_audio = AsyncMock(return_value=120.0)

        def _play(_source, after):
            after(None)

        vc.play.side_effect = _play

        with patch("plugins.platforms.discord.adapter.discord") as mock_discord:
            mock_discord.FFmpegPCMAudio.return_value = MagicMock()
            mock_discord.PCMVolumeTransformer.return_value = MagicMock()
            assert await adapter.play_in_voice_channel(111, "/tmp/x.mp3") is True

        assert calls == [111]

    @pytest.mark.asyncio
    async def test_play_ack_in_voice_rearms(self):
        from plugins.platforms.discord import adapter as discord_mod

        adapter = _make_adapter()
        adapter._voice_timeout_seconds = 0
        adapter._voice_fx_cfg = {
            "ack_enabled": True, "ack_phrases": ["One moment."], "speech_gain": 1.0,
        }
        mixer = MagicMock()
        adapter._voice_mixers[111] = mixer
        calls = _spy_rearm(adapter)

        def _fake_tts(text=None, output_path=None, **kwargs):
            with open(output_path, "wb") as fh:
                fh.write(b"fake-mp3")
            return json.dumps({"success": True, "file_path": output_path})

        with patch("tools.tts_tool.text_to_speech_tool", side_effect=_fake_tts), \
             patch.object(discord_mod, "_voice_mixer_module",
                          lambda: MagicMock(decode_to_pcm=MagicMock(return_value=b"\x00" * 384))):
            assert await adapter.play_ack_in_voice(111) is True

        assert calls == [111]
        mixer.play_speech.assert_called_once()
