"""Tests for the Discord continuous voice mixer (ambient + ducked speech)
and the verbal-ack-before-tool-calls hook.

The mixer (plugins/platforms/discord/voice_mixer.py) is pure-PCM and has no
discord.py dependency, so its core is tested directly.  The adapter
integration (install on join, play routing, ack) is tested with the standard
``object.__new__(DiscordAdapter)`` helper used elsewhere in the voice suite.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# numpy ships only in the optional "voice" extra (not [all,dev]); the mixer
# math needs it, so skip this whole module when it isn't installed.
np = pytest.importorskip("numpy")

# voice_mixer lives inside the discord plugin package dir; import by path the
# same way the adapter does.
_DISCORD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "plugins", "platforms", "discord",
)
if _DISCORD_DIR not in sys.path:
    sys.path.insert(0, _DISCORD_DIR)

import voice_mixer as vm  # noqa: E402


# =====================================================================
# Pure mixer unit tests
# =====================================================================

class TestVoiceMixerCore:
    def test_frame_geometry_matches_discord(self):
        # 20ms @ 48kHz stereo s16 == 3840 bytes (discord.opus.Encoder.FRAME_SIZE)
        assert vm.FRAME_SIZE == 3840
        assert vm.SAMPLES_PER_FRAME == 960
        assert len(vm.SILENCE_FRAME) == vm.FRAME_SIZE

    def test_empty_mixer_returns_silence_frames(self):
        mx = vm.VoiceMixer()
        for _ in range(5):
            frame = mx.read()
            assert len(frame) == vm.FRAME_SIZE
            assert frame == vm.SILENCE_FRAME

    def test_is_opus_false(self):
        # discord.py sends raw PCM when is_opus() is False.
        assert vm.VoiceMixer().is_opus() is False

    def test_ambient_loops_and_is_quiet(self):
        mx = vm.VoiceMixer(ambient_gain=0.2)
        amb = vm.synth_ambient_pcm(seconds=0.5)
        assert len(amb) % vm.FRAME_SIZE == 0  # frame-aligned for seamless loop
        mx.set_ambient(amb)
        peaks = [int(np.max(np.abs(np.frombuffer(mx.read(), dtype=np.int16))))
                 for _ in range(100)]  # 2s >> 0.5s loop
        # Produces audio after the fade-in and stays under the configured gain.
        assert any(p > 0 for p in peaks[10:])
        assert max(peaks) < int(32767 * 0.5)


# =====================================================================
# Adapter integration
# =====================================================================

def _make_adapter(fx_cfg=None):
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
    adapter._ambient_pcm_cache = None
    adapter._voice_fx_cfg = fx_cfg if fx_cfg is not None else {
        "enabled": True, "ambient_enabled": True, "ambient_path": "",
        "ambient_gain": 0.18, "duck_gain": 0.06, "speech_gain": 1.0,
        "ack_enabled": True, "ack_phrases": ["One moment."],
    }
    return adapter


class TestVoiceMixerActive:


    def test_false_when_attr_missing(self):
        # Defensive getattr path (object.__new__ helper that forgot the attr).
        from plugins.platforms.discord.adapter import DiscordAdapter
        from gateway.config import Platform
        bare = object.__new__(DiscordAdapter)
        bare.platform = Platform.DISCORD
        assert bare.voice_mixer_active(111) is False


class TestPlayInVoiceChannelMixerPath:
    @pytest.mark.asyncio
    async def test_routes_through_mixer_when_present(self):
        adapter = _make_adapter()
        vc = MagicMock()
        vc.is_connected.return_value = True
        adapter._voice_clients[111] = vc

        # speech_active returns True once (so play_speech is observed) then
        # False so the wait loop exits promptly.
        class _Mixer:
            def __init__(self):
                self._polls = 0
                self.play_speech = MagicMock()

            @property
            def speech_active(self):
                self._polls += 1
                return self._polls <= 1

        mixer = _Mixer()
        adapter._voice_mixers[111] = mixer
        adapter._reset_voice_timeout = MagicMock()

        fake_pcm = b"\x00" * vm.FRAME_SIZE
        with patch.object(vm, "decode_to_pcm", return_value=fake_pcm):
            ok = await adapter.play_in_voice_channel(111, "/tmp/x.mp3")
        assert ok is True
        mixer.play_speech.assert_called_once()
        adapter._reset_voice_timeout.assert_called_once_with(111)
        # Legacy path must NOT have been used.
        vc.play.assert_not_called()


class TestLeadSilence:
    """Warm-up lead silence prepended to speech so the first word isn't clipped
    (issue #66827)."""

    def test_bytes_empty_when_unset(self):
        adapter = _make_adapter()  # default cfg has no lead_silence_ms
        assert adapter._lead_silence_bytes() == b""


    def test_bytes_length_matches_ms(self):
        adapter = _make_adapter({"lead_silence_ms": 200})
        lead = adapter._lead_silence_bytes()
        assert lead == b"\x00" * (vm.BYTES_PER_MS * 200)
        assert len(lead) == 200 * 192  # 48kHz stereo s16 -> 192 bytes/ms


class TestPlayAckInVoice:
    @pytest.mark.asyncio
    async def test_noop_when_ack_disabled(self):
        adapter = _make_adapter({"ack_enabled": False})
        adapter._voice_mixers[111] = MagicMock()
        assert await adapter.play_ack_in_voice(111) is False


# =====================================================================
# Streaming TTS: mixer child + resampler (#60671)
# =====================================================================

class TestStreamingMixerChild:
    def _full_frame(self, value=0x10):
        # One 20ms frame of a constant loud-ish sample.
        return (value.to_bytes(2, "little", signed=False)) * (
            vm.SAMPLES_PER_FRAME * vm.CHANNELS
        )

    def test_delivers_buffered_frames_then_silence_on_underrun(self):
        child = vm.StreamingMixerChild("s", gain=1.0, fade_in_ms=0)
        child.feed(self._full_frame() * 3)
        for _ in range(3):
            frame = child.read_frame()
            assert frame is not None and frame.shape[0] == vm.SAMPLES_PER_FRAME * vm.CHANNELS
            assert frame.any()  # real audio
        # Buffer drained but not closed → silence frame keeps the child alive.
        under = child.read_frame()
        assert under is not None and not under.any()
        assert child.finished is False

    def test_finishes_after_close_and_drain(self):
        child = vm.StreamingMixerChild("s", fade_in_ms=0)
        child.feed(self._full_frame())
        assert child.read_frame() is not None       # the one real frame
        child.close()
        assert child.read_frame() is None            # drained + closed → done
        assert child.finished is True

    def test_close_pads_final_partial_frame(self):
        child = vm.StreamingMixerChild("s", fade_in_ms=0)
        child.feed(b"\x01\x02" * 10)  # less than a full frame
        child.close()
        frame = child.read_frame()
        assert frame is not None and frame.shape[0] == vm.SAMPLES_PER_FRAME * vm.CHANNELS
        assert child.read_frame() is None
        assert child.finished is True

    def test_clear_and_close_drops_buffer_immediately(self):
        child = vm.StreamingMixerChild("s", fade_in_ms=0)
        child.feed(self._full_frame() * 5)
        child.clear_and_close()
        assert child.read_frame() is None
        assert child.finished is True

    def test_play_stream_ducks_ambient_and_returns_child(self):
        mx = vm.VoiceMixer()
        mx.set_ambient(vm.synth_ambient_pcm(1.0))
        child = mx.play_stream(gain=1.0)
        assert isinstance(child, vm.StreamingMixerChild)
        assert mx.speech_active is True
        # The mixer read loop drops the child and releases the duck once it
        # finishes (empty + closed).
        child.close()
        for _ in range(3):
            mx.read()
        assert mx.speech_active is False


class TestPcmResampler:
    def test_24k_mono_to_48k_stereo_doubles_and_widens(self):
        # 0.5s of 24k mono -> ~0.5s of 48k stereo.
        n = 12000
        mono = (np.sin(np.arange(n) / 5.0) * 8000).astype(np.int16).tobytes()
        rs = vm.PcmResamplerTo48Stereo(24000, 1)
        out = rs.feed(mono)
        out_frames = len(out) // 2 // vm.CHANNELS
        assert abs(out_frames - 24000) < 50  # ~2x samples, stereo

    def test_carries_odd_byte_remainder_across_chunks(self):
        n = 4000
        mono = (np.arange(n) % 100 - 50).astype(np.int16).tobytes()
        rs_split = vm.PcmResamplerTo48Stereo(24000, 1)
        # Feed in deliberately sample-splitting sizes (odd byte counts).
        out_split = b""
        for i in range(0, len(mono), 777):
            out_split += rs_split.feed(mono[i:i + 777])
        rs_whole = vm.PcmResamplerTo48Stereo(24000, 1)
        out_whole = rs_whole.feed(mono)
        # Same total audio regardless of how it was chunked (within rounding).
        assert abs(len(out_split) - len(out_whole)) <= 8

    def test_48k_stereo_passthrough_is_lossless_length(self):
        frames = 960 * 3
        stereo = (np.arange(frames * 2) % 200 - 100).astype(np.int16).tobytes()
        rs = vm.PcmResamplerTo48Stereo(48000, 2)
        out = rs.feed(stereo)
        # Passthrough consumes one carried sample of latency; near-identical.
        assert abs(len(out) - len(stereo)) <= 8


# =====================================================================
# Streaming TTS: adapter contract (#60671)
# =====================================================================

def _voice_adapter_with_mixer(guild_id=111, chat_id="900"):
    """An adapter wired to a live mixer + connected VC for streaming tests."""
    adapter = _make_adapter()
    vc = MagicMock()
    vc.is_connected.return_value = True
    adapter._voice_clients[guild_id] = vc
    adapter._voice_mixers[guild_id] = vm.VoiceMixer()
    adapter._voice_text_channels[guild_id] = int(chat_id)
    adapter._cancel_voice_timeout = MagicMock()
    adapter._reset_voice_timeout = MagicMock()
    return adapter, guild_id, chat_id


class TestAdapterStreamingTTS:
    def test_supports_only_when_mixer_bound_to_chat(self):
        from gateway.platforms.base import AudioFormat
        adapter, gid, chat_id = _voice_adapter_with_mixer()
        assert adapter.supports_streaming_tts(chat_id, AudioFormat()) is True
        # Unknown chat, or a chat with no mixer, is declined.
        assert adapter.supports_streaming_tts("does-not-exist", AudioFormat()) is False

    @pytest.mark.asyncio
    async def test_begin_write_finish_flows_pcm_into_mixer(self):
        from gateway.platforms.base import AudioFormat
        adapter, gid, chat_id = _voice_adapter_with_mixer()
        fmt = AudioFormat(sample_rate=24000, channels=1)

        handle = await adapter.begin_streaming_tts(chat_id, fmt)
        assert handle is not None
        adapter._cancel_voice_timeout.assert_called_once_with(gid)
        # No child yet — ambient must not duck until real audio arrives.
        assert handle.child is None
        assert adapter._voice_mixers[gid].speech_active is False

        # Feed ~0.2s of 24k mono; a child appears and the ambient ducks.
        mono = (np.sin(np.arange(4800) / 4.0) * 8000).astype(np.int16).tobytes()
        await adapter.write_streaming_tts(handle, mono)
        assert handle.child is not None
        assert adapter._voice_mixers[gid].speech_active is True

        # Drain frames off the mixer — we should hear real (non-silent) audio.
        speechy = 0
        for _ in range(20):
            frame = np.frombuffer(adapter._voice_mixers[gid].read(), dtype=np.int16)
            if np.abs(frame).mean() > 100:
                speechy += 1
        assert speechy > 0

        await adapter.finish_streaming_tts(handle)
        adapter._reset_voice_timeout.assert_called_with(gid)
        # close() marks end-of-turn but the child only finishes once its buffer
        # drains through read_frame() — it must not have finished mid-buffer.
        assert handle.child.finished is False
        while handle.child.read_frame() is not None:
            pass
        assert handle.child.finished is True

    @pytest.mark.asyncio
    async def test_new_reply_supersedes_prior_stream(self):
        # A long reply plays out in the background after its turn; when the next
        # reply begins it must stop the old one so they don't overlap.
        from gateway.platforms.base import AudioFormat
        adapter, gid, chat_id = _voice_adapter_with_mixer()
        fmt = AudioFormat(sample_rate=24000, channels=1)
        h1 = await adapter.begin_streaming_tts(chat_id, fmt)
        mono = (np.ones(4800) * 5000).astype(np.int16).tobytes()
        await adapter.write_streaming_tts(h1, mono)
        assert h1.child is not None and not h1.child.finished

        h2 = await adapter.begin_streaming_tts(chat_id, fmt)  # next reply
        # Prior reply is superseded: flagged aborted and its audio cut.
        assert h1.aborted is True
        assert h1.child.finished is True
        # The new reply is now the tracked stream for the guild.
        assert adapter._voice_stream_handles[gid] is h2

    @pytest.mark.asyncio
    async def test_stop_streaming_speech_cuts_current(self):
        from gateway.platforms.base import AudioFormat
        adapter, gid, chat_id = _voice_adapter_with_mixer()
        h = await adapter.begin_streaming_tts(chat_id, AudioFormat(sample_rate=24000, channels=1))
        mono = (np.ones(4800) * 5000).astype(np.int16).tobytes()
        await adapter.write_streaming_tts(h, mono)
        adapter.stop_streaming_speech(chat_id)   # barge-in
        assert h.aborted is True
        assert h.child.finished is True
        assert gid not in adapter._voice_stream_handles

    def test_stop_streaming_speech_unknown_chat_is_noop(self):
        adapter, gid, chat_id = _voice_adapter_with_mixer()
        adapter.stop_streaming_speech("no-such-chat")  # must not raise

    @pytest.mark.asyncio
    async def test_begin_declines_without_mixer(self):
        from gateway.platforms.base import AudioFormat
        adapter = _make_adapter()
        adapter._voice_text_channels = {}
        assert await adapter.begin_streaming_tts("nope", AudioFormat()) is None

    @pytest.mark.asyncio
    async def test_bargein_abort_clears_child_immediately(self):
        from gateway.platforms.base import AudioFormat
        adapter, gid, chat_id = _voice_adapter_with_mixer()
        handle = await adapter.begin_streaming_tts(chat_id, AudioFormat(sample_rate=24000, channels=1))
        mono = (np.ones(4800) * 5000).astype(np.int16).tobytes()
        await adapter.write_streaming_tts(handle, mono)
        # Barge-in: cut immediately, dropping buffered audio.
        await adapter.abort_streaming_tts(handle, error="barge-in")
        assert handle.aborted is True
        assert handle.child.finished is True
        # A late write after abort is dropped, not raised.
        await adapter.write_streaming_tts(handle, mono)

    @pytest.mark.asyncio
    async def test_graceful_abort_drains_buffered_audio(self):
        # A finalisation-timeout / cleanup abort must NOT discard already-
        # synthesised audio — it should play out (child.close(), not clear).
        from gateway.platforms.base import AudioFormat
        adapter, gid, chat_id = _voice_adapter_with_mixer()
        handle = await adapter.begin_streaming_tts(chat_id, AudioFormat(sample_rate=24000, channels=1))
        # Buffer ~0.3s of audio (several frames).
        mono = (np.ones(7200) * 5000).astype(np.int16).tobytes()
        await adapter.write_streaming_tts(handle, mono)
        await adapter.abort_streaming_tts(handle, error="streaming TTS finalisation timeout")
        assert handle.aborted is True
        # Not finished yet — buffered audio still there to play out.
        assert handle.child.finished is False
        drained = 0
        for _ in range(200):
            frame = handle.child.read_frame()
            if frame is None:
                break
            drained += 1
        assert drained > 0            # buffered audio played out
        assert handle.child.finished is True


