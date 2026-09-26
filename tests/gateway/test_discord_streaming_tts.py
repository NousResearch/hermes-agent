"""Tests for streaming TTS into Discord voice channels (#60671).

Covers the ``voice_mixer.PCMStream`` buffer/resampler and its standalone and mixer consumers, the
adapter's streaming contract (supports/begin/write/finish/abort), and one end-to-end run through the
real ``StreamingTTSConsumer``. No Discord connection, network or audio device: the VoiceClient and
receiver are faked, and discord.py's sender thread is simulated by calling ``read()`` directly.
"""

import asyncio
import os
import sys
from unittest.mock import MagicMock

import pytest

# Mixer children do float math in numpy (optional "voice" extra), as in test_discord_voice_mixer.
np = pytest.importorskip("numpy")

_DISCORD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "plugins", "platforms", "discord",
)
if _DISCORD_DIR not in sys.path:
    sys.path.insert(0, _DISCORD_DIR)

import voice_mixer as vm  # noqa: E402

from gateway.platforms.base import AudioFormat  # noqa: E402


def _tone(n_samples: int, *, rate: int = 24000, channels: int = 1, amp: int = 8000) -> bytes:
    """s16le 440 Hz sine, interleaved when stereo."""
    mono = (amp * np.sin(2 * np.pi * 440 * np.arange(n_samples) / rate)).astype(np.int16)
    return (np.repeat(mono[:, None], channels, axis=1).reshape(-1) if channels == 2 else mono).tobytes()


def _drain(stream: "vm.PCMStream", limit: int = 100_000) -> list:
    frames = []
    for _ in range(limit):
        frame = stream.next_frame()
        if frame is None:
            return frames
        frames.append(frame)
    raise AssertionError("stream never ended")


# =====================================================================
# PCMStream
# =====================================================================

class TestPCMStream:

    def test_converts_24k_mono_to_48k_stereo(self):
        stream = vm.PCMStream(24000, 1)
        stream.feed(_tone(24000))  # 1 s
        stream.close()
        frames = _drain(stream)
        assert all(len(f) == vm.FRAME_SIZE for f in frames)
        assert abs(len(frames) - 50) <= 1  # 1 s of 20 ms frames at 48 kHz
        samples = np.frombuffer(frames[25], dtype=np.int16).reshape(-1, 2)
        assert np.array_equal(samples[:, 0], samples[:, 1])  # mono duplicated to both channels
        assert np.abs(samples).max() > 4000

    def test_48k_stereo_passes_through_unchanged(self):
        pcm = np.random.default_rng(1).integers(-20000, 20000, vm.FRAME_SIZE * 3 // 2, dtype=np.int16).tobytes()
        stream = vm.PCMStream(48000, 2)
        stream.feed(pcm)
        stream.close()
        assert b"".join(_drain(stream)) == pcm

    @pytest.mark.parametrize("rate,channels", [(24000, 1), (48000, 2), (22050, 1)])
    def test_chunks_that_split_samples_match_one_feed(self, rate, channels):
        pcm = _tone(rate // 5, rate=rate, channels=channels)
        whole, pieces = vm.PCMStream(rate, channels), vm.PCMStream(rate, channels)
        whole.feed(pcm)
        pos, sizes = 0, [1, 3, 7, 1000, 5, 4097]
        while pos < len(pcm):
            size = sizes[pos % len(sizes)]
            pieces.feed(pcm[pos:pos + size])
            pos += size
        for s in (whole, pieces):
            s.close()
        assert _drain(pieces) == _drain(whole)

    def test_lead_silence_plays_first(self):
        stream = vm.PCMStream(48000, 2, lead=b"\x00" * vm.FRAME_SIZE * 2)
        stream.feed(_tone(960 * 3, rate=48000, channels=2))
        assert stream.next_frame() == vm.SILENCE_FRAME
        assert stream.next_frame() == vm.SILENCE_FRAME
        assert stream.next_frame() != vm.SILENCE_FRAME

    def test_underrun_holds_silence_until_rebuffered(self):
        frame = _tone(960, rate=48000, channels=2)
        resume_frames = vm._RESUME_AFTER_UNDERRUN_BYTES // vm.FRAME_SIZE
        stream = vm.PCMStream(48000, 2)
        stream.feed(frame)
        assert stream.next_frame() == frame
        # Producer behind: silence, not end of stream (playback must survive between sentences).
        assert stream.next_frame() == vm.SILENCE_FRAME
        stream.feed(frame * (resume_frames - 1))  # just under the re-buffer threshold
        assert stream.next_frame() == vm.SILENCE_FRAME
        stream.feed(frame)
        assert stream.next_frame() == frame
        stream.close()
        assert len(_drain(stream)) == resume_frames - 1

    def test_close_pads_partial_tail_then_ends(self):
        stream = vm.PCMStream(48000, 2)
        stream.feed(b"\x01\x00" * 100)
        stream.close()
        frame = stream.next_frame()
        assert len(frame) == vm.FRAME_SIZE and frame.startswith(b"\x01\x00" * 100)
        assert stream.next_frame() is None

    def test_on_finished_fires_once_on_natural_end(self):
        calls = []
        stream = vm.PCMStream(48000, 2, on_finished=lambda: calls.append(1))
        stream.feed(_tone(960, rate=48000, channels=2))
        stream.close()
        _drain(stream)
        assert stream.next_frame() is None
        stream.cancel()
        assert calls == [1]

    def test_cancel_drops_audio_and_ignores_late_feeds(self):
        calls = []
        stream = vm.PCMStream(48000, 2, on_finished=lambda: calls.append(1))
        stream.feed(_tone(960 * 10, rate=48000, channels=2))
        stream.cancel()
        stream.cancel()
        stream.feed(_tone(960, rate=48000, channels=2))
        assert stream.cancelled
        assert stream.next_frame() is None
        assert calls == [1]

    def test_on_speaking_tracks_audible_speech_with_hangover(self):
        events = []
        frame = _tone(960, rate=48000, channels=2)
        stream = vm.PCMStream(48000, 2, on_speaking=events.append)
        stream.feed(frame * 2)
        stream.next_frame()
        assert events == [True]
        stream.next_frame()
        for _ in range(vm._QUIET_AFTER_FRAMES - 1):  # brief underrun: still "speaking"
            assert stream.next_frame() == vm.SILENCE_FRAME
        assert events == [True]
        stream.next_frame()  # a real gap (tool call, slow clause): quiet
        assert events == [True, False]
        stream.feed(frame * (vm._RESUME_AFTER_UNDERRUN_BYTES // vm.FRAME_SIZE))
        stream.next_frame()
        assert events == [True, False, True]
        stream.close()
        _drain(stream)
        assert events == [True, False, True, False]  # always ends quiet

    def test_cancel_while_speaking_reports_quiet(self):
        events = []
        stream = vm.PCMStream(48000, 2, on_speaking=events.append)
        stream.feed(_tone(960 * 4, rate=48000, channels=2))
        stream.next_frame()
        stream.cancel()
        stream.cancel()
        assert events == [True, False]

    @pytest.mark.parametrize("rate,channels", [(24000, 3), (0, 1)])
    def test_rejects_unsupported_format(self, rate, channels):
        with pytest.raises(ValueError):
            vm.PCMStream(rate, channels)


class TestPCMStreamConsumers:

    def test_standalone_source_ends_with_empty_read(self):
        stream = vm.PCMStream(48000, 2)
        source = vm.PCMStreamSource(stream)
        assert source.is_opus() is False
        stream.feed(_tone(960 * 2, rate=48000, channels=2))
        stream.close()
        assert len(source.read()) == vm.FRAME_SIZE
        assert len(source.read()) == vm.FRAME_SIZE
        assert source.read() == b""

    def test_source_cleanup_cancels_stream(self):
        stream = vm.PCMStream(48000, 2)
        vm.PCMStreamSource(stream).cleanup()  # discord.py: vc.stop() / playback error
        assert stream.cancelled

    def test_mixer_keeps_ambient_ducked_until_stream_ends(self):
        mixer = vm.VoiceMixer(ambient_gain=0.5, duck_gain=0.05)
        mixer.set_ambient(vm.synth_ambient_pcm(seconds=0.5))
        stream = vm.PCMStream(48000, 2)
        mixer.play_speech_stream(stream)
        stream.feed(_tone(960 * 5, rate=48000, channels=2))
        for _ in range(20):  # past the buffered speech: the producer is "behind"
            assert len(mixer.read()) == vm.FRAME_SIZE
        assert mixer.speech_active  # still ducked between sentences
        stream.close()
        mixer.read()
        assert not mixer.speech_active

    def test_mixer_stop_speech_cancels_stream(self):
        calls = []
        mixer = vm.VoiceMixer()
        stream = vm.PCMStream(48000, 2, on_finished=lambda: calls.append(1))
        mixer.play_speech_stream(stream)
        mixer.stop_speech()
        assert stream.cancelled and calls == [1]


# =====================================================================
# Adapter streaming contract
# =====================================================================

class FakeVoiceClient:
    """VoiceClient stand-in: ``play`` records the source; ``pump`` runs it to the end the way
    discord.py's AudioPlayer does (read until b"", then ``after`` and ``cleanup``)."""

    def __init__(self, *, playing: bool = False, play_error: Exception = None):
        self.source, self.after, self._playing, self._play_error = None, None, playing, play_error
        self.connected = True

    def is_connected(self):
        return self.connected

    def is_playing(self):
        return self._playing

    def play(self, source, *, after=None):
        if self._play_error:
            raise self._play_error
        self.source, self.after, self._playing = source, after, True

    def stop(self):
        self._playing = False

    async def disconnect(self):
        self.connected = False

    def pump(self, limit: int = 100_000) -> list:
        frames = []
        for _ in range(limit):
            frame = self.source.read()
            if not frame:
                break
            frames.append(frame)
        self._playing = False
        if self.after:
            self.after(None)
        self.source.cleanup()
        return frames


GUILD, TEXT_CH = 111, 222


def _make_adapter(fx_cfg=None, vc=None):
    from gateway.config import Platform, PlatformConfig
    from plugins.platforms.discord.adapter import DiscordAdapter
    config = PlatformConfig(enabled=True, extra={})
    config.token = "fake-token"
    adapter = object.__new__(DiscordAdapter)
    adapter.platform = Platform.DISCORD
    adapter.config = config
    adapter._client = MagicMock()
    adapter._voice_clients = {GUILD: vc or FakeVoiceClient()}
    adapter._voice_locks = {}
    adapter._voice_text_channels = {GUILD: TEXT_CH}
    adapter._voice_sources = {}
    adapter._voice_timeout_tasks = {}
    adapter._voice_receivers = {}
    adapter._voice_listen_tasks = {}
    adapter._voice_mixers = {}
    adapter._voice_streams = {}
    adapter._ambient_pcm_cache = None
    adapter._voice_fx_cfg = fx_cfg if fx_cfg is not None else {"enabled": False, "lead_silence_ms": 0, "speech_gain": 1.0}
    adapter._reset_voice_timeout = MagicMock()
    return adapter


async def _settle_callbacks():
    """Let ``loop.call_soon_threadsafe`` callbacks (stream on_finished) run."""
    for _ in range(3):
        await asyncio.sleep(0)


class TestSupportsStreamingTTS:

    def test_only_for_linked_connected_voice_channel(self):
        adapter = _make_adapter()
        assert adapter.supports_streaming_tts(str(TEXT_CH), AudioFormat()) is True
        assert adapter.supports_streaming_tts("999", AudioFormat()) is False
        adapter._voice_clients[GUILD].connected = False
        assert adapter.supports_streaming_tts(str(TEXT_CH), AudioFormat()) is False

    @pytest.mark.parametrize("fmt", [AudioFormat(sample_width=1), AudioFormat(channels=6), AudioFormat(sample_rate=0)])
    def test_rejects_unsupported_pcm(self, fmt):
        assert _make_adapter().supports_streaming_tts(str(TEXT_CH), fmt) is False


class TestStreamingPlayback:

    @pytest.mark.asyncio
    async def test_plays_standalone_and_restores_state_when_drained(self):
        adapter = _make_adapter()
        vc, receiver = adapter._voice_clients[GUILD], MagicMock()
        adapter._voice_receivers[GUILD] = receiver
        handle = await adapter.begin_streaming_tts(str(TEXT_CH), AudioFormat())
        assert isinstance(vc.source, vm.PCMStreamSource)
        assert adapter._voice_streams[GUILD] is handle
        await adapter.write_streaming_tts(handle, _tone(24000 // 2))  # 0.5 s
        receiver.pause.assert_not_called()  # muted by audible frames, not by opening the stream
        vc.source.read()
        receiver.pause.assert_called_once()
        await adapter.finish_streaming_tts(handle)
        # finish returned with the audio still buffered: playback drains in the background.
        assert vc.is_playing() and not handle.stream.cancelled
        frames = vc.pump()
        assert abs(len(frames) - 24) <= 1
        await _settle_callbacks()
        receiver.resume.assert_called_once()
        assert GUILD not in adapter._voice_streams
        adapter._reset_voice_timeout.assert_called_with(GUILD)

    @pytest.mark.asyncio
    async def test_capture_unmuted_during_gap_between_clauses(self):
        adapter = _make_adapter()
        vc, receiver = adapter._voice_clients[GUILD], MagicMock()
        adapter._voice_receivers[GUILD] = receiver
        handle = await adapter.begin_streaming_tts(str(TEXT_CH), AudioFormat())
        await adapter.write_streaming_tts(handle, _tone(2400))  # one short clause, then a tool call
        for _ in range(5 + vm._QUIET_AFTER_FRAMES):
            vc.source.read()
        # The user can talk (and barge in) while Iris runs the tool; the stream is still open.
        receiver.resume.assert_called_once()
        assert not handle.stream.cancelled and vc.is_playing()
        await adapter.write_streaming_tts(handle, _tone(24000 * 6 // 10))  # next clause, past the re-buffer
        vc.source.read()
        assert receiver.pause.call_count == 2

    @pytest.mark.asyncio
    async def test_resamples_to_discord_format(self):
        adapter = _make_adapter()
        handle = await adapter.begin_streaming_tts(str(TEXT_CH), AudioFormat(sample_rate=22050))
        await adapter.write_streaming_tts(handle, _tone(22050, rate=22050))
        await adapter.finish_streaming_tts(handle)
        assert abs(len(adapter._voice_clients[GUILD].pump()) - 50) <= 1

    @pytest.mark.asyncio
    async def test_lead_silence_from_voice_fx_config(self):
        adapter = _make_adapter({"enabled": False, "lead_silence_ms": 100})
        handle = await adapter.begin_streaming_tts(str(TEXT_CH), AudioFormat())
        await adapter.write_streaming_tts(handle, _tone(2400))
        await adapter.finish_streaming_tts(handle)
        frames = adapter._voice_clients[GUILD].pump()
        assert frames[:5] == [vm.SILENCE_FRAME] * 5 and frames[5] != vm.SILENCE_FRAME

    @pytest.mark.asyncio
    async def test_queues_behind_previous_reply_without_blocking(self):
        vc = FakeVoiceClient(playing=True)
        adapter = _make_adapter(vc=vc)
        handle = await adapter.begin_streaming_tts(str(TEXT_CH), AudioFormat())
        assert handle is not None and vc.source is None  # returned at once; audio buffers meanwhile
        await adapter.write_streaming_tts(handle, _tone(2400))
        vc.stop()
        await asyncio.sleep(0.15)
        assert isinstance(vc.source, vm.PCMStreamSource)
        await adapter.finish_streaming_tts(handle)
        assert abs(len(vc.pump()) - 5) <= 1

    @pytest.mark.asyncio
    async def test_queued_reply_aborted_before_start_never_plays(self):
        vc = FakeVoiceClient(playing=True)
        adapter = _make_adapter(vc=vc)
        handle = await adapter.begin_streaming_tts(str(TEXT_CH), AudioFormat())
        await adapter.abort_streaming_tts(handle, "barge-in")
        vc.stop()
        await asyncio.sleep(0.15)
        assert vc.source is None and handle.start_task.done()

    @pytest.mark.asyncio
    async def test_abort_stops_playback_and_is_idempotent(self):
        adapter = _make_adapter()
        vc, receiver = adapter._voice_clients[GUILD], MagicMock()
        adapter._voice_receivers[GUILD] = receiver
        handle = await adapter.begin_streaming_tts(str(TEXT_CH), AudioFormat())
        await adapter.write_streaming_tts(handle, _tone(24000 * 5))
        vc.source.read()
        await adapter.abort_streaming_tts(handle, "barge-in")
        await adapter.abort_streaming_tts(handle, "barge-in")
        await adapter.write_streaming_tts(handle, _tone(2400))  # late producer chunk: dropped
        assert handle.aborted
        assert vc.source.read() == b""  # discord.py stops within one frame
        await _settle_callbacks()
        receiver.resume.assert_called_once()
        assert GUILD not in adapter._voice_streams

    @pytest.mark.asyncio
    async def test_interrupted_finish_aborts(self):
        adapter = _make_adapter()
        handle = await adapter.begin_streaming_tts(str(TEXT_CH), AudioFormat())
        await adapter.write_streaming_tts(handle, _tone(24000))
        await adapter.finish_streaming_tts(handle, interrupted=True)
        assert handle.stream.cancelled

    @pytest.mark.asyncio
    async def test_routes_through_mixer_when_installed(self):
        adapter = _make_adapter({"enabled": True, "lead_silence_ms": 0, "speech_gain": 1.0})
        vc, receiver, mixer = adapter._voice_clients[GUILD], MagicMock(), vm.VoiceMixer()
        adapter._voice_receivers[GUILD] = receiver
        adapter._voice_mixers[GUILD] = mixer
        handle = await adapter.begin_streaming_tts(str(TEXT_CH), AudioFormat())
        assert vc.source is None  # the mixer already owns the VoiceClient
        assert mixer.speech_active
        receiver.pause.assert_not_called()  # matches the mixer's whole-file path
        await adapter.write_streaming_tts(handle, _tone(2400))
        await adapter.finish_streaming_tts(handle)
        for _ in range(10):
            mixer.read()
        await _settle_callbacks()
        assert not mixer.speech_active and GUILD not in adapter._voice_streams

    @pytest.mark.asyncio
    async def test_declines_and_restores_state_when_play_fails(self):
        adapter = _make_adapter(vc=FakeVoiceClient(play_error=RuntimeError("Not connected to voice.")))
        receiver = MagicMock()
        adapter._voice_receivers[GUILD] = receiver
        assert await adapter.begin_streaming_tts(str(TEXT_CH), AudioFormat()) is None
        await _settle_callbacks()
        receiver.pause.assert_not_called()
        assert adapter._voice_streams == {}

    @pytest.mark.asyncio
    async def test_leave_voice_channel_cancels_stream(self):
        adapter = _make_adapter()
        handle = await adapter.begin_streaming_tts(str(TEXT_CH), AudioFormat())
        await adapter.write_streaming_tts(handle, _tone(24000))
        await adapter.leave_voice_channel(GUILD)
        await _settle_callbacks()
        assert handle.stream.cancelled and handle.aborted
        assert adapter._voice_streams == {}


class TestVoiceReceiverPauseDepth:

    def test_nested_pause_needs_matching_resumes(self):
        from plugins.platforms.discord.adapter import VoiceReceiver
        receiver = VoiceReceiver(MagicMock())
        receiver.pause()
        receiver.pause()  # streaming reply begins before the previous reply's resume()
        receiver.resume()
        assert receiver._paused
        receiver.resume()
        assert not receiver._paused
        receiver.resume()  # unbalanced resume never goes negative
        receiver.pause()
        assert receiver._paused


# =====================================================================
# End to end through the gateway consumer
# =====================================================================

class _ToneStreamer:
    """Streaming provider stand-in: each clause yields 0.2 s of 24 kHz mono PCM in odd-sized chunks."""
    sample_rate, channels, sample_width = 24000, 1, 2

    def __init__(self):
        self.clauses = []

    def stream(self, text):
        self.clauses.append(text)
        pcm = _tone(4800)
        for i in range(0, len(pcm), 1001):
            yield pcm[i:i + 1001]


@pytest.mark.asyncio
async def test_consumer_streams_reply_into_voice_channel(monkeypatch):
    import tools.tts_streaming as tts_streaming
    from gateway.streaming_tts_consumer import StreamingTTSConsumer

    streamer = _ToneStreamer()
    monkeypatch.setattr(tts_streaming, "resolve_streaming_provider", lambda cfg, preferred=None: streamer)
    adapter = _make_adapter()
    vc = adapter._voice_clients[GUILD]
    consumer = StreamingTTSConsumer(adapter, str(TEXT_CH), {}, asyncio.get_running_loop())
    assert consumer.active
    consumer.start()
    consumer.on_delta("The parity check finished with zero errors. ")
    consumer.on_delta("Your array is healthy and nothing needs attention.")
    consumer.finish()
    assert await consumer.wait_complete(timeout=5.0) is True
    assert consumer.suppress_whole_file  # the gateway must not replay it as a file
    assert len(streamer.clauses) == 2
    assert abs(len(vc.pump()) - 20) <= 2  # 2 clauses x 0.2 s


class TestFlushAtEndOfModelResponse:
    """A reply's last sentence has no trailing whitespace, so the chunker held it until turn end,
    after Hermes' post-reply work (~0.5 s+). The agent now flushes TTS when the model stream ends."""

    def test_stream_end_flushes_tts_only_on_success(self):
        from agent.stream_delivery import StreamDeliveryMixin
        agent = StreamDeliveryMixin()
        agent.stream_flush_callback = MagicMock()
        agent._emit_stream_end(final_text="", finished=False, error="boom")  # a retry would repeat it
        agent.stream_flush_callback.assert_not_called()
        agent._emit_stream_end(final_text="All good.", finished=True, error=None)
        agent.stream_flush_callback.assert_called_once_with()

    def test_no_callback_is_a_noop(self):
        from agent.stream_delivery import StreamDeliveryMixin
        StreamDeliveryMixin()._emit_stream_end(final_text="All good.", finished=True, error=None)

    @pytest.mark.asyncio
    async def test_single_sentence_reply_speaks_on_flush_not_turn_end(self, monkeypatch):
        import tools.tts_streaming as tts_streaming
        from gateway.streaming_tts_consumer import StreamingTTSConsumer

        streamer = _ToneStreamer()
        monkeypatch.setattr(tts_streaming, "resolve_streaming_provider", lambda cfg, preferred=None: streamer)
        adapter = _make_adapter()
        consumer = StreamingTTSConsumer(adapter, str(TEXT_CH), {}, asyncio.get_running_loop())
        consumer.start()
        consumer.on_delta("The array is healthy and nothing needs attention.")
        await asyncio.sleep(0.3)
        assert streamer.clauses == []  # held: no whitespace after the final period
        consumer.on_delta(None)  # what stream_flush_callback sends when the model stream ends
        await asyncio.sleep(0.3)
        assert streamer.clauses == ["The array is healthy and nothing needs attention."]
        assert isinstance(adapter._voice_clients[GUILD].source, vm.PCMStreamSource)  # playing before finish()
        consumer.finish()
        assert await consumer.wait_complete(timeout=5.0) is True
        assert len(streamer.clauses) == 1  # the turn-end flush doesn't speak it twice


class _PacedStreamer(_ToneStreamer):
    """Slow provider: one chunk per ``delay`` seconds (``block`` stalls before the first one)."""

    def __init__(self, chunks=6, delay=0.1, block=None):
        super().__init__()
        self.chunks, self.delay, self.block = chunks, delay, block

    def stream(self, text):
        import time
        self.clauses.append(text)
        if self.block is not None:
            self.block.wait(5.0)
        for _ in range(self.chunks):
            time.sleep(self.delay)
            yield _tone(2400)


async def _finalize(monkeypatch, streamer, *, stall=0.3):
    """Run the real runner finalisation over a real consumer streaming into the Discord adapter."""
    import gateway.run_turn as run_turn
    import tools.tts_streaming as tts_streaming
    from types import SimpleNamespace
    from gateway.run_turn import GatewayTurnMixin
    from gateway.streaming_tts_consumer import StreamingTTSConsumer

    monkeypatch.setattr(run_turn, "STREAMING_TTS_STALL_SECONDS", stall)
    monkeypatch.setattr(tts_streaming, "resolve_streaming_provider", lambda cfg, preferred=None: streamer)
    adapter = _make_adapter()
    consumer = StreamingTTSConsumer(adapter, str(TEXT_CH), {}, asyncio.get_running_loop())
    consumer.start()
    for sentence in ("First sentence of a long reply. ", "Second sentence keeps going. ",
                     "Third sentence is still arriving."):
        consumer.on_delta(sentence)
    turn_ctx = SimpleNamespace(streaming_tts_consumer_holder=[consumer], session_key="s", run_generation=1)
    await GatewayTurnMixin._run_agent_finalize_streaming_tts(None, turn_ctx, adapter)
    return consumer


@pytest.mark.asyncio
async def test_finalize_keeps_draining_while_tts_makes_progress(monkeypatch):
    # 3 clauses x 0.6 s of synthesis = 1.8 s, far past the 0.3 s stall budget, but never stalled.
    consumer = await _finalize(monkeypatch, _PacedStreamer())
    assert consumer.done and consumer.completed


@pytest.mark.asyncio
async def test_finalize_aborts_stalled_tts(monkeypatch):
    import threading
    release = threading.Event()
    try:
        consumer = await _finalize(monkeypatch, _PacedStreamer(block=release))
        assert not consumer.completed and consumer._aborted
    finally:
        release.set()
