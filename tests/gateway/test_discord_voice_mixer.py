"""Tests for the Discord continuous voice mixer (ambient + ducked speech)
and the verbal-ack-before-tool-calls hook.

The mixer (plugins/platforms/discord/voice_mixer.py) is pure-PCM and has no
discord.py dependency, so its core is tested directly.  The adapter
integration (install on join, play routing, ack) is tested with the standard
``object.__new__(DiscordAdapter)`` helper used elsewhere in the voice suite.
"""

import asyncio
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

class TestStreamingMixerChild:
    def test_split_24k_mono_input_converts_to_discord_frame(self):
        child = vm.StreamingMixerChild("stream", max_buffer_bytes=vm.FRAME_SIZE * 2)
        mono = np.arange(480, dtype=np.int16).tobytes()
        child.write(mono[:101])
        child.write(mono[101:])
        child.finish()

        frame = child.read_frame()
        assert frame is not None
        expected = np.repeat(np.repeat(np.arange(480, dtype=np.int16), 2), 2)
        np.testing.assert_array_equal(frame.astype(np.int16), expected)
        assert child.read_frame() is None

    def test_underrun_returns_silence_until_finished(self):
        child = vm.StreamingMixerChild("stream")
        frame = child.read_frame()
        assert frame is not None
        assert np.count_nonzero(frame) == 0
        assert not child.finished

        child.finish()
        assert child.read_frame() is None
        assert child.finished

    def test_finish_drains_partial_frame_with_padding(self):
        child = vm.StreamingMixerChild("stream")
        child.write(np.array([123], dtype=np.int16).tobytes())
        child.finish()
        frame = child.read_frame()
        assert frame is not None
        assert frame[0] == frame[1] == frame[2] == frame[3] == 123
        assert np.count_nonzero(frame[4:]) == 0
        assert child.read_frame() is None

    def test_abort_drops_buffered_audio(self):
        child = vm.StreamingMixerChild("stream")
        child.write(np.arange(480, dtype=np.int16).tobytes())
        child.abort()
        assert child.read_frame() is None
        assert child.finished

    def test_buffer_is_bounded(self):
        child = vm.StreamingMixerChild("stream", max_buffer_bytes=vm.FRAME_SIZE)
        child.write(np.arange(480, dtype=np.int16).tobytes())
        with pytest.raises(BufferError):
            child.write(np.array([1], dtype=np.int16).tobytes())


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
    adapter._voice_text_channels = {111: 111}
    adapter._voice_sources = {}
    adapter._voice_timeout_tasks = {}
    adapter._voice_receivers = {}
    adapter._voice_listen_tasks = {}
    adapter._voice_mixers = {}
    adapter._streaming_tts_handles = {}
    adapter._cancel_voice_timeout = MagicMock()
    adapter._reset_voice_timeout = MagicMock()
    adapter._ambient_pcm_cache = None
    adapter._voice_fx_cfg = fx_cfg if fx_cfg is not None else {
        "enabled": True, "ambient_enabled": True, "ambient_path": "",
        "ambient_gain": 0.18, "duck_gain": 0.06, "speech_gain": 1.0,
        "ack_enabled": True, "ack_phrases": ["One moment."],
    }
    return adapter


class TestStreamingMixerLifecycle:
    def test_drained_callback_fires_once_on_natural_drain(self):
        events = []
        mx = vm.VoiceMixer()
        child = mx.begin_streaming_speech(on_drained=lambda c: events.append("drained"))
        child.write(np.arange(480, dtype=np.int16).tobytes())
        child.finish()
        for _ in range(100):
            mx.read()
            if not mx.speech_active:
                break
        assert events == ["drained"]

    def test_drained_callback_fires_on_abort(self):
        events = []
        mx = vm.VoiceMixer()
        child = mx.begin_streaming_speech(on_drained=lambda c: events.append("drained"))
        child.write(np.arange(480, dtype=np.int16).tobytes())
        child.abort()
        assert events == ["drained"]

    def test_abort_one_streaming_child_keeps_sibling_speech_active(self):
        mx = vm.VoiceMixer()
        a = mx.begin_streaming_speech()
        b = mx.begin_streaming_speech()
        b.write(np.arange(960, dtype=np.int16).tobytes())
        a.abort()
        assert mx.speech_active is True, "sibling speech must stay active"
        assert mx.read() != vm.SILENCE_FRAME, "sibling audio must still play"
        b.abort()
        assert mx.speech_active is False

    def test_stop_speech_aborts_streaming_children_and_notifies_owner(self):
        events = []
        mx = vm.VoiceMixer()
        child = mx.begin_streaming_speech(
            on_drained=lambda _child: events.append("drained")
        )
        child.write(np.arange(960, dtype=np.int16).tobytes())
        mx.stop_speech()
        assert mx.speech_active is False
        assert events == ["drained"]
        # A late producer write must not resurrect audio into the mixer.
        assert child.write(np.arange(480, dtype=np.int16).tobytes()) is False
        assert mx.read() == vm.SILENCE_FRAME

    def test_stopping_one_shot_clip_does_not_abort_streaming_sibling(self):
        mx = vm.VoiceMixer()
        stream = mx.begin_streaming_speech()
        stream.write(np.arange(960, dtype=np.int16).tobytes())
        clip = mx.play_speech(b"\x01\x00" * (vm.FRAME_SIZE // 2))

        mx.stop_speech_child(clip)

        assert clip.finished is True
        assert stream.write(np.arange(480, dtype=np.int16).tobytes()) is True
        assert mx.speech_active is True
        assert mx.read() != vm.SILENCE_FRAME
        stream.abort()

    def test_audible_callback_runs_outside_mixer_lock(self):
        callback_ran = asyncio.Event()
        mx = vm.VoiceMixer()

        def on_audible(_child):
            assert mx.speech_active is True
            callback_ran.set()

        child = mx.begin_streaming_speech(on_audible=on_audible)
        child.write(np.arange(480, dtype=np.int16).tobytes())
        mx.read()  # first pull; callback is deferred until post-send acknowledgement
        thread = __import__("threading").Thread(target=mx.read, daemon=True)
        thread.start()
        thread.join(timeout=1.0)

        assert not thread.is_alive(), "on_audible re-entered the mixer lock"
        assert callback_ran.is_set()

    def test_cleanup_notifies_stream_owner_and_rejects_new_children(self):
        events = []
        mx = vm.VoiceMixer()
        child = mx.begin_streaming_speech(on_drained=lambda _c: events.append("drained"))
        child.write(np.arange(480, dtype=np.int16).tobytes())
        mx.cleanup()
        assert events == ["drained"]
        assert child.write(b"\x00\x00") is False
        with pytest.raises(RuntimeError, match="closed"):
            mx.begin_streaming_speech()


class TestVoiceMixerActive:
    def test_streaming_child_plays_before_finish(self):
        mx = vm.VoiceMixer()
        child = mx.begin_streaming_speech()
        child.write(np.arange(480, dtype=np.int16).tobytes())
        frame = mx.read()
        assert frame != vm.SILENCE_FRAME
        assert mx.speech_active

    def test_streaming_child_underrun_keeps_mixer_alive(self):
        mx = vm.VoiceMixer()
        child = mx.begin_streaming_speech()
        # Empty but open: silence frames, mixer must not stop the stream.
        assert mx.read() == vm.SILENCE_FRAME
        # One full 20ms-worth of input (480 mono samples) converts to a full
        # Discord frame, so the next read emits audio.
        child.write(np.arange(480, dtype=np.int16).tobytes())
        assert mx.read() != vm.SILENCE_FRAME

    def test_streaming_child_finish_releases_duck(self):
        mx = vm.VoiceMixer()
        child = mx.begin_streaming_speech()
        child.write(np.arange(480, dtype=np.int16).tobytes())
        child.finish()
        drained = 0
        while mx.speech_active and drained < 100:
            mx.read()
            drained += 1
        assert not mx.speech_active

    def test_abort_streaming_child_stops_speech_immediately(self):
        mx = vm.VoiceMixer()
        child = mx.begin_streaming_speech()
        child.write(np.arange(960, dtype=np.int16).tobytes())
        child.abort()
        assert not mx.speech_active

    def test_false_when_attr_missing(self):
        # Defensive getattr path (object.__new__ helper that forgot the attr).
        from plugins.platforms.discord.adapter import DiscordAdapter
        from gateway.config import Platform
        bare = object.__new__(DiscordAdapter)
        bare.platform = Platform.DISCORD
        assert bare.voice_mixer_active(111) is False


class TestVoiceMixerActivation:
    def test_streaming_requests_mixer_without_effects(self):
        adapter = _make_adapter({"enabled": False, "streaming_tts": True})
        assert adapter._voice_mixer_requested() is True

    def test_no_mixer_when_effects_and_streaming_are_off(self):
        adapter = _make_adapter({"enabled": False, "streaming_tts": False})
        assert adapter._voice_mixer_requested() is False

    def test_streaming_only_does_not_create_ambient_audio(self):
        adapter = _make_adapter({
            "enabled": False,
            "streaming_tts": True,
            "ambient_enabled": True,
        })
        assert adapter._get_ambient_pcm() is None


class TestPlayInVoiceChannelMixerPath:
    @pytest.mark.asyncio
    async def test_routes_through_mixer_when_present(self):
        adapter = _make_adapter()
        vc = MagicMock()
        vc.is_connected.return_value = True
        adapter._voice_clients[111] = vc

        # The returned one-shot child is already finished, so the wait loop
        # exits without consulting unrelated streaming siblings.
        class _Mixer:
            def __init__(self):
                child = MagicMock()
                child.finished = True
                self.play_speech = MagicMock(return_value=child)
                self.stop_speech_child = MagicMock()

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

    @pytest.mark.asyncio
    async def test_clip_timeout_stops_only_its_returned_child(self):
        adapter = _make_adapter()
        vc = MagicMock()
        vc.is_connected.return_value = True
        adapter._voice_clients[111] = vc
        adapter._playback_timeout_for_audio = AsyncMock(return_value=-1.0)
        clip = MagicMock()
        clip.finished = False
        mixer = MagicMock()
        mixer.play_speech.return_value = clip
        adapter._voice_mixers[111] = mixer

        fake_pcm = b"\x00" * vm.FRAME_SIZE
        with patch.object(vm, "decode_to_pcm", return_value=fake_pcm):
            assert await adapter.play_in_voice_channel(111, "/tmp/x.mp3") is True

        mixer.stop_speech_child.assert_called_once_with(clip)
        mixer.stop_speech.assert_not_called()


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
        adapter = _make_adapter({"enabled": True, "ack_enabled": False})
        adapter._voice_mixers[111] = MagicMock()
        assert await adapter.play_ack_in_voice(111) is False

    @pytest.mark.asyncio
    async def test_noop_when_effects_disabled_but_streaming_enabled(self):
        adapter = _make_adapter({
            "enabled": False,
            "ack_enabled": True,
            "streaming_tts": True,
        })
        adapter._voice_mixers[111] = MagicMock()
        assert await adapter.play_ack_in_voice(111) is False


class TestVoiceReceiverPauseState:
    def test_pause_state_is_observable(self):
        from plugins.platforms.discord.adapter import VoiceReceiver

        receiver = object.__new__(VoiceReceiver)
        receiver._paused = False
        assert receiver.paused is False
        receiver.pause()
        assert receiver.paused is True
        receiver.resume()
        assert receiver.paused is False


class TestStreamingTTSContract:
    @pytest.mark.asyncio
    async def test_supports_false_without_mixer(self):
        from gateway.platforms.base import AudioFormat
        adapter = _make_adapter()
        assert adapter.supports_streaming_tts("111", AudioFormat()) is False

    @pytest.mark.asyncio
    async def test_supports_true_when_connected_flag_on(self):
        from gateway.platforms.base import AudioFormat
        adapter = _make_adapter({"streaming_tts": True})
        vc = MagicMock()
        vc.is_connected.return_value = True
        adapter._voice_clients[111] = vc
        adapter._voice_mixers[111] = MagicMock()
        assert adapter.supports_streaming_tts("111", AudioFormat()) is True

    @pytest.mark.asyncio
    async def test_supports_false_on_wrong_format(self):
        from gateway.platforms.base import AudioFormat
        adapter = _make_adapter({"streaming_tts": True})
        vc = MagicMock()
        vc.is_connected.return_value = True
        adapter._voice_clients[111] = vc
        adapter._voice_mixers[111] = MagicMock()
        assert adapter.supports_streaming_tts(
            "111", AudioFormat(sample_rate=44100, channels=2, sample_width=2)
        ) is False

    @pytest.mark.asyncio
    async def test_begin_returns_handle_and_warms_child_with_lead_silence(self):
        from gateway.platforms.base import AudioFormat
        adapter = _make_adapter({"lead_silence_ms": 200})
        mixer = MagicMock()
        child = MagicMock()
        mixer.begin_streaming_speech.return_value = child
        adapter._voice_mixers[111] = mixer
        receiver = MagicMock()
        adapter._voice_receivers[111] = receiver
        fmt = AudioFormat(sample_rate=24000, channels=1, sample_width=2)
        handle = await adapter.begin_streaming_tts("111", fmt)
        assert handle is not None
        assert handle.audio_format == fmt
        mixer.begin_streaming_speech.assert_called_once()
        # 200 ms of 24 kHz mono s16le lead silence == 200 * 48 bytes.
        written = child.write.call_args[0][0]
        assert len(written) == 200 * 48
        assert written == b"\x00" * (200 * 48)
        receiver.pause.assert_not_called()
        adapter._cancel_voice_timeout.assert_called_once_with(111)

    @pytest.mark.asyncio
    async def test_write_becomes_audible_only_after_next_sender_pull_acknowledges_send(self):
        from gateway.platforms.base import AudioFormat
        adapter = _make_adapter()
        mixer = vm.VoiceMixer()
        adapter._voice_mixers[111] = mixer
        handle = await adapter.begin_streaming_tts("111", AudioFormat())
        assert handle is not None
        await adapter.write_streaming_tts(
            handle, np.arange(480, dtype=np.int16).tobytes()
        )
        assert handle.audible is False
        assert mixer.read() != vm.SILENCE_FRAME
        assert handle.audible is False  # frame was pulled but not yet send-acked
        mixer.read()  # AudioPlayer only pulls again after send_audio_packet succeeds
        await asyncio.sleep(0)
        assert handle.audible is True

    @pytest.mark.asyncio
    async def test_first_frame_send_failure_cleanup_never_marks_audible(self):
        from gateway.platforms.base import AudioFormat

        adapter = _make_adapter()
        mixer = vm.VoiceMixer()
        adapter._voice_mixers[111] = mixer
        handle = await adapter.begin_streaming_tts("111", AudioFormat())
        await adapter.write_streaming_tts(
            handle, np.arange(480, dtype=np.int16).tobytes()
        )

        assert mixer.read() != vm.SILENCE_FRAME
        # Model discord.py AudioPlayer failing in send_audio_packet(): cleanup is
        # called and there is no subsequent source.read() acknowledgement.
        mixer.cleanup()
        await asyncio.sleep(0)

        assert handle.audible is False
        assert handle.platform_failed is True

    @pytest.mark.asyncio
    async def test_finish_waits_for_final_frame_send_ack_before_fallback_decision(self):
        from gateway.platforms.base import AudioFormat

        adapter = _make_adapter()
        mixer = vm.VoiceMixer()
        adapter._voice_mixers[111] = mixer
        handle = await adapter.begin_streaming_tts("111", AudioFormat())
        await adapter.write_streaming_tts(
            handle, np.arange(480, dtype=np.int16).tobytes()
        )

        finishing = asyncio.create_task(adapter.finish_streaming_tts(handle))
        await asyncio.sleep(0)
        assert finishing.done() is False
        assert mixer.read() != vm.SILENCE_FRAME
        await asyncio.sleep(0)
        assert finishing.done() is False

        mixer.read()  # acknowledges the preceding non-silent sender frame
        await finishing

        assert handle.audible is True
        assert handle.aborted is False

    @pytest.mark.asyncio
    async def test_finish_sender_ack_timeout_aborts_buffer_before_fallback(self):
        from gateway.platforms.base import AudioFormat

        adapter = _make_adapter()
        adapter.STREAMING_TTS_TERMINAL_TIMEOUT = 0.01
        mixer = vm.VoiceMixer()
        adapter._voice_mixers[111] = mixer
        handle = await adapter.begin_streaming_tts("111", AudioFormat())
        await adapter.write_streaming_tts(
            handle, np.arange(480, dtype=np.int16).tobytes()
        )

        with pytest.raises(RuntimeError, match="did not acknowledge"):
            await adapter.finish_streaming_tts(handle)

        assert handle.audible is False
        assert handle.aborted is True
        assert handle.child.aborted is True
        assert mixer.read() == vm.SILENCE_FRAME

    @pytest.mark.asyncio
    async def test_finish_without_audio_keeps_handle_not_audible(self):
        from gateway.platforms.base import AudioFormat
        adapter = _make_adapter()
        mixer = MagicMock()
        child = MagicMock()
        child.drained = True
        child.aborted = False
        mixer.begin_streaming_speech.return_value = child
        adapter._voice_mixers[111] = mixer
        handle = await adapter.begin_streaming_tts("111", AudioFormat())
        assert handle is not None
        await adapter.finish_streaming_tts(handle)
        assert handle.audible is False, (
            "never-audible stream must stay not-audible so whole-file fallback "
            "is not suppressed"
        )

    @pytest.mark.asyncio
    async def test_finish_rearms_timeout_only_after_drain(self):
        from gateway.platforms.base import AudioFormat

        adapter = _make_adapter()
        mixer = vm.VoiceMixer()
        adapter._voice_mixers[111] = mixer
        handle = await adapter.begin_streaming_tts("111", AudioFormat())
        assert handle is not None
        await adapter.write_streaming_tts(
            handle, np.arange(1440, dtype=np.int16).tobytes()
        )
        finishing = asyncio.create_task(adapter.finish_streaming_tts(handle))
        await asyncio.sleep(0)
        mixer.read()
        mixer.read()  # post-send acknowledgement while more PCM is buffered
        await finishing
        adapter._reset_voice_timeout.assert_not_called()
        for _ in range(200):
            mixer.read()
            if not mixer.speech_active:
                break
        await asyncio.sleep(0)
        adapter._reset_voice_timeout.assert_called_once_with(111)

    @pytest.mark.asyncio
    async def test_streaming_never_touches_receiver_pause_state(self):
        from gateway.platforms.base import AudioFormat
        adapter = _make_adapter()
        mixer = MagicMock()
        child = MagicMock()
        mixer.begin_streaming_speech.return_value = child
        receiver = MagicMock()
        receiver.paused = True
        adapter._voice_mixers[111] = mixer
        adapter._voice_receivers[111] = receiver
        handle = await adapter.begin_streaming_tts("111", AudioFormat())
        assert handle is not None
        await adapter.abort_streaming_tts(handle)
        receiver.pause.assert_not_called()
        receiver.resume.assert_not_called()

    @pytest.mark.asyncio
    async def test_begin_rolls_back_child_and_timeout_on_failure(self):
        from gateway.platforms.base import AudioFormat
        adapter = _make_adapter({"lead_silence_ms": 10})
        mixer = MagicMock()
        child = MagicMock()
        child.write.side_effect = RuntimeError("boom")
        mixer.begin_streaming_speech.return_value = child
        adapter._voice_mixers[111] = mixer
        handle = await adapter.begin_streaming_tts("111", AudioFormat())
        assert handle is None
        child.abort.assert_called_once()
        adapter._reset_voice_timeout.assert_called_once_with(111)

    @pytest.mark.asyncio
    async def test_finish_after_abort_is_noop(self):
        from gateway.platforms.base import AudioFormat
        adapter = _make_adapter()
        mixer = MagicMock()
        child = MagicMock()
        mixer.begin_streaming_speech.return_value = child
        adapter._voice_mixers[111] = mixer
        handle = await adapter.begin_streaming_tts("111", AudioFormat())
        assert handle is not None
        await adapter.abort_streaming_tts(handle)
        await adapter.finish_streaming_tts(handle)
        assert child.finish.call_count == 0

    @pytest.mark.asyncio
    async def test_abort_aborts_child_and_rearms_timeout(self):
        from gateway.platforms.base import AudioFormat
        adapter = _make_adapter()
        mixer = MagicMock()
        child = MagicMock()
        mixer.begin_streaming_speech.return_value = child
        adapter._voice_mixers[111] = mixer
        handle = await adapter.begin_streaming_tts("111", AudioFormat())
        assert handle is not None
        await adapter.abort_streaming_tts(handle, "boom")
        child.abort.assert_called_once()
        adapter._reset_voice_timeout.assert_called_once_with(111)
        assert handle.aborted is True

    @pytest.mark.asyncio
    async def test_new_stream_supersedes_old_without_rearming_timeout(self):
        from gateway.platforms.base import AudioFormat
        adapter = _make_adapter()
        mixer = MagicMock()
        first_child = MagicMock()
        second_child = MagicMock()
        mixer.begin_streaming_speech.side_effect = [first_child, second_child]
        adapter._voice_mixers[111] = mixer
        first = await adapter.begin_streaming_tts("111", AudioFormat())
        on_interrupt = MagicMock()
        first.on_interrupt = on_interrupt
        second = await adapter.begin_streaming_tts("111", AudioFormat())
        assert first.interrupted is True
        assert first.aborted is True
        on_interrupt.assert_called_once_with("stream replaced")
        first_child.abort.assert_called_once()
        assert adapter._streaming_tts_handles[111] is second
        adapter._reset_voice_timeout.assert_not_called()

    @pytest.mark.asyncio
    async def test_leave_interrupts_stream_and_cancels_producer(self):
        from gateway.platforms.base import AudioFormat

        adapter = _make_adapter()
        mixer = MagicMock()
        child = MagicMock()
        mixer.begin_streaming_speech.return_value = child
        adapter._voice_mixers[111] = mixer
        voice_client = MagicMock()
        voice_client.disconnect = AsyncMock()
        adapter._voice_clients[111] = voice_client
        handle = await adapter.begin_streaming_tts("111", AudioFormat())
        on_interrupt = MagicMock()
        handle.on_interrupt = on_interrupt

        await adapter.leave_voice_channel(111)

        assert handle.interrupted is True
        assert handle.aborted is True
        on_interrupt.assert_called_once_with("voice channel left")
        child.abort.assert_called_once()

    @pytest.mark.asyncio
    async def test_unexpected_mixer_cleanup_aborts_provider_owner(self):
        from gateway.platforms.base import AudioFormat

        adapter = _make_adapter()
        mixer = vm.VoiceMixer()
        adapter._voice_mixers[111] = mixer
        handle = await adapter.begin_streaming_tts("111", AudioFormat())
        on_abort = MagicMock()
        handle.on_abort = on_abort

        mixer.cleanup()
        await asyncio.sleep(0)

        assert handle.child.aborted is True
        assert handle.aborted is True
        on_abort.assert_called_once_with("Discord voice mixer stopped unexpectedly")
        with pytest.raises(RuntimeError, match="mixer stopped unexpectedly"):
            await adapter.finish_streaming_tts(handle)
        assert 111 not in adapter._streaming_tts_handles

    @pytest.mark.asyncio
    async def test_closed_mixer_rejects_stream(self):
        from gateway.platforms.base import AudioFormat
        adapter = _make_adapter()
        mixer = vm.VoiceMixer()
        mixer.cleanup()
        adapter._voice_mixers[111] = mixer
        assert await adapter.begin_streaming_tts("111", AudioFormat()) is None
        assert adapter.voice_mixer_active(111) is False

    @pytest.mark.asyncio
    async def test_authorized_voice_barge_in_aborts_current_stream(self):
        from gateway.platforms.base import AudioFormat
        adapter = _make_adapter()
        adapter._is_allowed_user = MagicMock(return_value=True)
        mixer = MagicMock()
        child = MagicMock()
        mixer.begin_streaming_speech.return_value = child
        adapter._voice_mixers[111] = mixer
        handle = await adapter.begin_streaming_tts("111", AudioFormat())
        on_interrupt = MagicMock()
        handle.on_interrupt = on_interrupt
        adapter._handle_voice_barge_in(111, 42)
        assert handle.interrupted is True
        assert handle.aborted is True
        on_interrupt.assert_called_once_with("voice barge-in")
        child.abort.assert_called_once()
        assert 111 not in adapter._streaming_tts_handles

    @pytest.mark.asyncio
    async def test_unauthorized_voice_does_not_interrupt_stream(self):
        from gateway.platforms.base import AudioFormat
        adapter = _make_adapter()
        adapter._is_allowed_user = MagicMock(return_value=False)
        mixer = MagicMock()
        child = MagicMock()
        mixer.begin_streaming_speech.return_value = child
        adapter._voice_mixers[111] = mixer
        handle = await adapter.begin_streaming_tts("111", AudioFormat())
        adapter._handle_voice_barge_in(111, 99)
        assert handle.aborted is False
        child.abort.assert_not_called()

    @pytest.mark.asyncio
    async def test_abort_is_idempotent(self):
        from gateway.platforms.base import AudioFormat
        adapter = _make_adapter()
        mixer = MagicMock()
        child = MagicMock()
        mixer.begin_streaming_speech.return_value = child
        adapter._voice_mixers[111] = mixer
        handle = await adapter.begin_streaming_tts("111", AudioFormat())
        assert handle is not None
        await adapter.abort_streaming_tts(handle)
        await adapter.abort_streaming_tts(handle)
        assert child.abort.call_count == 1
        assert handle.aborted is True

    @pytest.mark.asyncio
    async def test_no_streaming_when_mixer_absent(self):
        from gateway.platforms.base import AudioFormat
        adapter = _make_adapter()
        handle = await adapter.begin_streaming_tts("111", AudioFormat())
        assert handle is None


