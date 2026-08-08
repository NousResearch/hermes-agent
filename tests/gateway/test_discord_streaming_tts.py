"""Streaming TTS on Discord voice channels.

Hermes already ships the whole producer side of streaming TTS —
``gateway/streaming_tts_consumer.py`` chunks the model's deltas into clauses,
synthesises each one, and writes PCM to the adapter, and ``gateway/run.py``
instantiates it for every voice-input turn. The consumer's first act is to ask
``adapter.supports_streaming_tts(...)``, which ``BasePlatformAdapter`` answers
``False`` ("Override to opt in") — and no adapter overrode it, so the path was
unreachable and every voice reply waited for the full file.

These tests drive the real mixer, the real resampler, and the real adapter
methods; only discord.py's socket is absent.
"""
import asyncio

import pytest

np = pytest.importorskip("numpy", reason="voice extra (numpy) not installed")

from gateway.platforms.base import AudioFormat  # noqa: E402
from plugins.platforms.discord.voice_mixer import (  # noqa: E402
    CHANNELS,
    FRAME_SIZE,
    SAMPLE_RATE,
    PCMResampler,
    StreamingSpeechChild,
    VoiceMixer,
)


def _tone(n, rate=24000):
    """n samples of s16 mono."""
    return (np.sin(np.arange(n) * (2 * np.pi * 220 / rate)) * 18000).astype(np.int16).tobytes()


# ── the resampler ──────────────────────────────────────────────────────────

class TestPCMResampler:
    """Chunked conversion must equal whole-buffer conversion.

    A naive per-chunk resample clicks twice at every boundary: a chunk can
    split a sample across its final byte, and linear interpolation needs the
    previous chunk's last sample. Both are carried, so this equality is the
    property worth asserting.
    """

    @pytest.mark.parametrize("src_rate", [24000, 16000, 22050, 48000])
    @pytest.mark.parametrize("chunk", [97, 237, 4096])
    def test_chunked_matches_whole(self, src_rate, chunk):
        sig = _tone(2400, src_rate)
        whole = PCMResampler(src_rate, 1).convert(sig)
        r = PCMResampler(src_rate, 1)
        parts = b"".join(r.convert(sig[i:i + chunk]) for i in range(0, len(sig), chunk))
        assert parts == whole, (
            f"chunked conversion diverged from whole-buffer at {src_rate}Hz "
            f"with {chunk}-byte chunks — audible as a click per boundary"
        )

    def test_output_is_48k_stereo(self):
        out = PCMResampler(24000, 1).convert(_tone(480))
        n_out = len(out) // (CHANNELS * 2)
        # 2x upsample; the final fractional sample waits for the next chunk.
        assert 959 <= n_out <= 960
        assert len(out) % (CHANNELS * 2) == 0

    def test_odd_byte_chunk_does_not_corrupt(self):
        """A chunk ending mid-sample must carry the stray byte, not drop it."""
        sig = _tone(1200)
        whole = PCMResampler(24000, 1).convert(sig)
        r = PCMResampler(24000, 1)
        parts = b"".join(r.convert(sig[i:i + 101]) for i in range(0, len(sig), 101))
        assert parts == whole

    def test_stereo_input_is_downmixed(self):
        mono = np.frombuffer(_tone(480), dtype=np.int16)
        stereo = np.repeat(mono[:, None], 2, axis=1).reshape(-1).tobytes()
        out = PCMResampler(24000, 2).convert(stereo)
        assert len(out) // (CHANNELS * 2) >= 959

    def test_rejects_nonsense_format(self):
        with pytest.raises(ValueError):
            PCMResampler(0, 1)
        with pytest.raises(ValueError):
            PCMResampler(24000, 0)


# ── the mixer child ────────────────────────────────────────────────────────

class TestStreamingSpeechChild:
    """Running dry must mean "wait", not "done"."""

    def test_starvation_emits_silence_and_keeps_the_child_alive(self):
        child = StreamingSpeechChild("s")
        frame = child.read_frame()
        assert frame is not None, (
            "an open but starved child reported finished — the mixer would "
            "drop it mid-sentence and release the ambient duck"
        )
        assert not child.finished
        assert child.starved_frames == 1
        assert not np.any(frame)  # silence

    def test_fed_audio_is_played_back(self):
        child = StreamingSpeechChild("s")
        pcm = b"\x11\x22" * (FRAME_SIZE // 2)
        child.feed(pcm)
        frame = child.read_frame()
        assert np.any(frame)
        assert child.starved_frames == 0

    def test_finishes_only_after_close_and_drain(self):
        child = StreamingSpeechChild("s")
        child.feed(b"\x01\x02" * (FRAME_SIZE // 2))
        child.close()
        assert child.read_frame() is not None   # the buffered frame
        assert child.read_frame() is None       # now drained
        assert child.finished

    def test_partial_tail_is_padded_not_clipped(self):
        """The last clause must not lose its final syllable."""
        child = StreamingSpeechChild("s")
        child.feed(b"\x7f\x00" * 100)  # far less than one frame
        child.close()
        frame = child.read_frame()
        assert frame is not None and len(frame) == FRAME_SIZE // 2
        assert np.any(frame)
        assert child.read_frame() is None

    def test_feed_after_close_is_ignored(self):
        child = StreamingSpeechChild("s")
        child.close()
        child.feed(b"\x01\x02" * (FRAME_SIZE // 2))
        assert child.read_frame() is None


class TestMixerIntegration:
    def test_begin_speech_stream_ducks_ambient_and_mixes(self):
        mixer = VoiceMixer()
        mixer.set_ambient(b"\x05\x00" * (FRAME_SIZE // 2 * 4), gain=1.0)
        child = mixer.begin_speech_stream(gain=1.0, fade_in_ms=0)

        assert mixer.speech_active, "the duck was never engaged for the stream"
        child.feed(b"\x10\x27" * (FRAME_SIZE // 2))       # 10000 per sample
        out = mixer.read()
        assert len(out) == FRAME_SIZE
        assert np.any(np.frombuffer(out, dtype=np.int16))

        child.close()
        for _ in range(4):
            mixer.read()
        assert not mixer.speech_active, "the duck was never released after the stream ended"

    def test_stream_survives_a_starved_gap(self):
        """The gap between two clauses must not end the reply."""
        mixer = VoiceMixer()
        child = mixer.begin_speech_stream(gain=1.0, fade_in_ms=0)
        child.feed(b"\x10\x27" * (FRAME_SIZE // 2))
        mixer.read()
        for _ in range(5):                                # producer stalls
            mixer.read()
        assert mixer.speech_active, "a synthesis gap ended the reply early"
        child.feed(b"\x10\x27" * (FRAME_SIZE // 2))       # next clause arrives
        assert np.any(np.frombuffer(mixer.read(), dtype=np.int16))


# ── the adapter hooks ──────────────────────────────────────────────────────

class _Receiver:
    def __init__(self):
        self.paused = False

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False


def _adapter(*, in_vc=True, with_mixer=True):
    from plugins.platforms.discord.adapter import DiscordAdapter

    from gateway.config import Platform

    a = object.__new__(DiscordAdapter)
    a.platform = Platform.DISCORD          # ``name`` is a property over this
    a._voice_text_channels = {77: 99}
    a._voice_mixers = {77: VoiceMixer()} if with_mixer else {77: None}
    a._voice_receivers = {77: _Receiver()}
    a._voice_fx_cfg = {"speech_gain": 1.0, "lead_silence_ms": 0}
    a._voice_timeout_tasks = {}
    a.is_in_voice_channel = lambda gid: in_vc
    a._cancel_voice_timeout = lambda gid: None
    a._reset_voice_timeout = lambda gid: None
    a._lead_silence_bytes = lambda: b""
    a._playback_timeout_limit = lambda: 2   # keep a stalled drain from hanging tests
    return a


async def _drain(mixer, *, frames=2000):
    """Stand in for discord.py's sender thread, which reads continuously."""
    for _ in range(frames):
        if not mixer.speech_active:
            return
        mixer.read()
        await asyncio.sleep(0)


def _run_with_reader(coro_fn, mixer):
    """Run *coro_fn* while the mixer is being drained, as in production."""
    async def _main():
        task = asyncio.ensure_future(_drain(mixer))
        try:
            return await coro_fn()
        finally:
            task.cancel()
    return asyncio.run(_main())


class TestAdapterHooks:
    FMT = AudioFormat(sample_rate=24000, channels=1, sample_width=2)

    def test_opts_in_when_a_mixer_backed_voice_channel_is_live(self):
        assert _adapter().supports_streaming_tts("99", self.FMT) is True

    def test_declines_when_not_in_a_voice_channel(self):
        assert _adapter(in_vc=False).supports_streaming_tts("99", self.FMT) is False

    def test_declines_without_the_mixer(self):
        """The legacy one-shot path takes a finished file; it cannot stream."""
        assert _adapter(with_mixer=False).supports_streaming_tts("99", self.FMT) is False

    def test_declines_an_unrelated_chat(self):
        assert _adapter().supports_streaming_tts("12345", self.FMT) is False

    def test_declines_a_non_s16_format(self):
        fmt = AudioFormat(sample_rate=24000, channels=1, sample_width=4)
        assert _adapter().supports_streaming_tts("99", fmt) is False

    def test_begin_write_finish_reaches_the_mixer(self):
        a = _adapter()
        handle = asyncio.run(a.begin_streaming_tts("99", self.FMT))
        assert handle is not None, "the adapter declined a chat it had accepted"
        assert a._voice_mixers[77].speech_active
        assert a._voice_receivers[77].paused, "the receiver was left open (echo)"

        asyncio.run(a.write_streaming_tts(handle, _tone(2400)))
        assert handle.audible, (
            "audible was never set — the consumer would replay the whole reply "
            "from the start on any later failure"
        )
        assert np.any(np.frombuffer(a._voice_mixers[77].read(), dtype=np.int16))

        _run_with_reader(lambda: a.finish_streaming_tts(handle), a._voice_mixers[77])
        assert not a._voice_mixers[77].speech_active
        assert not a._voice_receivers[77].paused, "the receiver stayed paused after the reply"

    def test_begin_declines_when_the_chat_has_no_voice_channel(self):
        a = _adapter(in_vc=False)
        assert asyncio.run(a.begin_streaming_tts("99", self.FMT)) is None

    def test_abort_is_idempotent_and_drops_late_chunks(self):
        a = _adapter()
        handle = asyncio.run(a.begin_streaming_tts("99", self.FMT))
        asyncio.run(a.abort_streaming_tts(handle, "cancelled"))
        asyncio.run(a.abort_streaming_tts(handle, "cancelled again"))  # must not raise
        assert handle.aborted
        # A late chunk from the producer is silently dropped.
        asyncio.run(a.write_streaming_tts(handle, _tone(480)))
        assert not a._voice_receivers[77].paused

    def test_write_before_begin_is_a_noop(self):
        a = _adapter()
        asyncio.run(a.write_streaming_tts(None, _tone(480)))  # must not raise

    def test_finish_on_interrupt_stops_playback(self):
        a = _adapter()
        handle = asyncio.run(a.begin_streaming_tts("99", self.FMT))
        asyncio.run(a.write_streaming_tts(handle, _tone(4800)))
        asyncio.run(a.finish_streaming_tts(handle, interrupted=True))  # must not block
        assert not a._voice_mixers[77].speech_active, (
            "an interrupted reply kept speaking"
        )


class TestFailureAndDrainContract:
    """Two ways the adapter can quietly break the consumer's contract."""

    FMT = AudioFormat(sample_rate=24000, channels=1, sample_width=2)

    def test_conversion_failure_propagates_so_fallback_survives(self):
        """A swallowed failure costs the user the whole reply.

        gateway/streaming_tts_consumer.py awaits write_streaming_tts and then
        sets ``handle.audible = True`` and ``_suppress_whole_file = True``
        unconditionally. Returning quietly after a failed conversion therefore
        claims audio played when none did — the gateway suppresses whole-file
        TTS and the user hears nothing at all.
        """
        a = _adapter()
        handle = asyncio.run(a.begin_streaming_tts("99", self.FMT))

        class _Broken:
            def convert(self, _chunk):
                raise RuntimeError("resampler exploded")

        handle.resampler = _Broken()

        with pytest.raises(RuntimeError):
            asyncio.run(a.write_streaming_tts(handle, _tone(480)))

        assert not handle.audible, (
            "the handle was marked audible despite no PCM reaching the mixer"
        )

    def test_normal_finish_waits_for_buffered_audio_to_play_out(self):
        """close() only stops input; the tail is still queued.

        Un-pausing the mic and re-arming the idle timer at close() would open
        the receiver while the bot is still speaking, and start counting idle
        time against a reply in progress.
        """
        a = _adapter()
        handle = asyncio.run(a.begin_streaming_tts("99", self.FMT))
        # ~1s of audio: far more than one 20 ms frame.
        asyncio.run(a.write_streaming_tts(handle, _tone(24000)))

        mixer = a._voice_mixers[77]
        frames_read = {"n": 0}

        async def _reader():
            for _ in range(5000):
                if not mixer.speech_active:
                    return
                mixer.read()
                frames_read["n"] += 1
                await asyncio.sleep(0)

        async def _main():
            task = asyncio.ensure_future(_reader())
            await a.finish_streaming_tts(handle)
            resumed_after = frames_read["n"]
            task.cancel()
            return resumed_after

        read_before_resume = asyncio.run(_main())

        assert read_before_resume > 40, (
            "finish() returned after only "
            f"{read_before_resume} frames — it did not wait for the buffered "
            "tail, so the receiver reopens while the bot is still speaking"
        )
        assert not a._voice_receivers[77].paused
        assert not mixer.speech_active

    def test_interrupt_does_not_wait(self):
        """interrupted=True means stop now — waiting would defeat the point."""
        a = _adapter()
        handle = asyncio.run(a.begin_streaming_tts("99", self.FMT))
        asyncio.run(a.write_streaming_tts(handle, _tone(24000)))

        import time as _t
        started = _t.monotonic()
        asyncio.run(a.finish_streaming_tts(handle, interrupted=True))
        elapsed = _t.monotonic() - started

        assert elapsed < 0.5, f"interrupt blocked for {elapsed:.2f}s"
        assert not a._voice_mixers[77].speech_active

    def test_a_stalled_playout_gives_up_at_the_configured_timeout(self):
        """Nothing draining the mixer must not hang the turn forever."""
        a = _adapter()
        a._playback_timeout_limit = lambda: 0.2
        handle = asyncio.run(a.begin_streaming_tts("99", self.FMT))
        asyncio.run(a.write_streaming_tts(handle, _tone(24000)))

        import time as _t
        started = _t.monotonic()
        asyncio.run(a.finish_streaming_tts(handle))   # nobody is reading
        elapsed = _t.monotonic() - started

        assert 0.15 < elapsed < 3.0, f"drain wait ran for {elapsed:.2f}s"
        assert not a._voice_mixers[77].speech_active, "the stalled stream was not stopped"


class TestBeginFailureLeavesNothingBehind:
    """A failure partway through begin() must not strand the mixer.

    `begin_speech_stream` registers the child before the rest of the setup
    runs. If something after it raises, an un-closed child sits starved and
    open forever: `speech_active` never clears, the ambient bed stays ducked
    for the rest of the session, and every later clip queues underneath a
    stream that will never produce audio.
    """

    FMT = AudioFormat(sample_rate=24000, channels=1, sample_width=2)

    def _failing_adapter(self):
        """An adapter whose begin() raises *after* the child is registered.

        The failure is injected on the adapter instance rather than on the
        voice_mixer module: adapter.py imports PCMResampler as either
        ``voice_mixer`` or ``.voice_mixer`` depending on sys.path, so patching
        one module object is not reliably the one it uses.
        """
        a = _adapter()

        def _boom():
            raise RuntimeError("setup failed after the child was registered")

        a._lead_silence_bytes = _boom
        return a

    def test_a_failure_after_registration_does_not_strand_a_speech_child(self):
        a = self._failing_adapter()

        handle = asyncio.run(a.begin_streaming_tts("99", self.FMT))

        assert handle is None, "begin() reported success despite failing"
        assert not a._voice_mixers[77].speech_active, (
            "a speech child was left registered — the ambient bed stays ducked "
            "and later clips queue behind a stream that never plays"
        )
        assert not a._voice_receivers[77].paused, (
            "the receiver was left paused after a failed start"
        )

    def test_a_later_reply_still_works_after_a_failed_start(self):
        """The session must not be poisoned by one bad start."""
        a = self._failing_adapter()
        assert asyncio.run(a.begin_streaming_tts("99", self.FMT)) is None

        a._lead_silence_bytes = lambda: b""      # the next reply is fine
        handle = asyncio.run(a.begin_streaming_tts("99", self.FMT))
        assert handle is not None
        asyncio.run(a.write_streaming_tts(handle, _tone(2400)))
        assert np.any(np.frombuffer(a._voice_mixers[77].read(), dtype=np.int16))


class TestFormatValidation:
    """A nonsense format is declined, not coerced.

    Coercing a declared rate of 0 to the 24 kHz default would play the whole
    reply at the wrong pitch; declining lets the gateway fall back to
    whole-file TTS, which is the honest outcome.
    """

    @pytest.mark.parametrize("fmt", [
        AudioFormat(sample_rate=0, channels=1, sample_width=2),
        AudioFormat(sample_rate=-1, channels=1, sample_width=2),
        AudioFormat(sample_rate=24000, channels=0, sample_width=2),
        AudioFormat(sample_rate=24000, channels=1, sample_width=4),
    ])
    def test_declines_an_unusable_format(self, fmt):
        assert _adapter().supports_streaming_tts("99", fmt) is False

    def test_accepts_the_ordinary_provider_format(self):
        assert _adapter().supports_streaming_tts(
            "99", AudioFormat(sample_rate=24000, channels=1, sample_width=2)
        ) is True

    def test_a_junk_format_object_is_declined_not_raised(self):
        class _Junk:
            sample_rate = "not a number"
            channels = 1
            sample_width = 2

        assert _adapter().supports_streaming_tts("99", _Junk()) is False
