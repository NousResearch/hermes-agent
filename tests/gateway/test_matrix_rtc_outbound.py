"""Phase 3 behaviour contracts: the bot speaks into a MatrixRTC call.

Fake LiveKit objects only — nothing here imports the SDK, opens a socket, or shells out
to ffmpeg. The fakes are deliberately strict about the two things the real API is strict
about (``AudioResampler.push`` wants a ``bytearray``; ``AudioFrame`` wants s16), because a
lenient fake would let a real-world failure pass green.
"""

import asyncio
import sys
import types

import pytest

from gateway.platforms.base import AudioFormat, SendResult, StreamingTTSHandle
from plugins.platforms.matrix.rtc import outbound as ob
from plugins.platforms.matrix.rtc import publisher as pub

ROOM = "!voice:hs.tld"
CONTRACT_RATE = AudioFormat.sample_rate  # 24000: what the gateway writes
WIRE_RATE = pub.LIVEKIT_SAMPLE_RATE  # 48000: what LiveKit carries
FRAME_BYTES = int(WIRE_RATE * pub.FRAME_DURATION) * pub.SAMPLE_WIDTH


def pcm(seconds: float, rate: int = CONTRACT_RATE) -> bytes:
    return b"\x00\x01" * int(rate * seconds)


# --------------------------------------------------------------------------- fake LiveKit


class _FakeFrame:
    def __init__(self, data, sample_rate, num_channels, samples_per_channel):
        if len(data) % pub.SAMPLE_WIDTH:
            raise ValueError("data length must be a multiple of sizeof(int16)")
        self.data, self.sample_rate = bytes(data), sample_rate
        self.num_channels, self.samples_per_channel = num_channels, samples_per_channel


class _FakeAudioSource:
    def __init__(self, sample_rate, num_channels):
        self.sample_rate, self.num_channels = sample_rate, num_channels
        self.captured, self.events, self.closed = [], [], False

    async def capture_frame(self, frame):
        self.captured.append(frame)
        self.events.append("capture")

    async def wait_for_playout(self):
        self.events.append("playout")

    def clear_queue(self):
        self.events.append("clear")

    async def aclose(self):
        self.closed = True


class _FakeResampler:
    """Models the real one's two load-bearing behaviours: bytearray-only input, and a
    tail that only comes out on flush()."""

    def __init__(self, input_rate, output_rate, num_channels=1, **kw):
        self.input_rate, self.output_rate, self.num_channels = (
            input_rate, output_rate, num_channels)
        self.pushed, self.flushed = [], False

    def push(self, data):
        if not isinstance(data, bytearray):
            raise TypeError(f"AudioResampler.push needs a bytearray, not {type(data).__name__}")
        self.pushed.append(bytes(data))
        ratio = self.output_rate // self.input_rate
        return [_FakeFrame(bytes(data) * ratio, self.output_rate, self.num_channels,
                           len(data) * ratio // (self.num_channels * pub.SAMPLE_WIDTH))]

    def flush(self):
        self.flushed = True
        return [_FakeFrame(b"\x00\x00", self.output_rate, self.num_channels, 1)]


class _FakeTrack:
    def __init__(self, name, source):
        self.name, self.source = name, source


class _FakePublication:
    sid = "TR_abc123"


class _FakeLocalParticipant:
    def __init__(self):
        self.published, self.unpublished = [], []

    async def publish_track(self, track, options):
        self.published.append((track, options))
        return _FakePublication()

    async def unpublish_track(self, sid):
        self.unpublished.append(sid)


class _FakeRoom:
    def __init__(self):
        self.local_participant = _FakeLocalParticipant()


def _fake_rtc_module():
    return types.SimpleNamespace(
        AudioSource=_FakeAudioSource,
        AudioResampler=_FakeResampler,
        AudioFrame=_FakeFrame,
        LocalAudioTrack=types.SimpleNamespace(create_audio_track=_FakeTrack),
        TrackPublishOptions=lambda source=None: types.SimpleNamespace(source=source),
        TrackSource=types.SimpleNamespace(SOURCE_MICROPHONE="mic"))


@pytest.fixture
def livekit(monkeypatch):
    """Make ``from livekit import rtc`` resolve to the fakes for the duration of a test."""
    rtc = _fake_rtc_module()
    monkeypatch.setitem(sys.modules, "livekit", types.SimpleNamespace(rtc=rtc))
    monkeypatch.setitem(sys.modules, "livekit.rtc", rtc)
    return rtc


@pytest.fixture
def no_drain(monkeypatch):
    """Record the FFI drain sleep instead of actually waiting half a second."""
    slept = []
    real_sleep = asyncio.sleep

    async def tracking_sleep(delay):
        slept.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(pub.asyncio, "sleep", tracking_sleep)
    return slept


async def live_publisher(room=None, **kw) -> pub.MatrixRTCPublisher:
    publisher = pub.MatrixRTCPublisher(room or _FakeRoom(), **kw)
    await publisher.start()
    return publisher


# --------------------------------------------------------------------------- publisher


class TestPublishing:
    @pytest.mark.asyncio
    async def test_one_microphone_track_is_published_at_the_wire_rate(self, livekit):
        room = _FakeRoom()
        publisher = await live_publisher(room)

        (track, options), = room.local_participant.published
        assert track.name == pub.TRACK_NAME
        assert options.source == "mic", "must publish as a microphone, not screen share"
        assert track.source.sample_rate == WIRE_RATE, "the source carries 48 kHz, not 24 kHz"
        assert publisher.live

    @pytest.mark.asyncio
    async def test_starting_twice_does_not_stack_a_second_microphone(self, livekit):
        room = _FakeRoom()
        publisher = await live_publisher(room)
        await publisher.start()
        assert len(room.local_participant.published) == 1

    @pytest.mark.asyncio
    async def test_a_contract_rate_publisher_builds_a_24k_to_48k_resampler(self, livekit):
        publisher = await live_publisher()
        assert (publisher._resampler.input_rate, publisher._resampler.output_rate) == (
            CONTRACT_RATE, WIRE_RATE)

    @pytest.mark.asyncio
    async def test_no_resampler_when_the_caller_already_writes_at_the_wire_rate(self, livekit):
        publisher = await live_publisher(sample_rate=WIRE_RATE)
        assert publisher._resampler is None


class TestWriting:
    @pytest.mark.asyncio
    async def test_contract_pcm_is_resampled_before_it_reaches_the_source(self, livekit):
        publisher = await live_publisher()
        await publisher.write(pcm(0.1))

        assert publisher._resampler.pushed == [pcm(0.1)]
        assert [f.sample_rate for f in publisher._source.captured] == [WIRE_RATE]

    @pytest.mark.asyncio
    async def test_the_resampler_is_handed_a_bytearray(self, livekit):
        """The real push() dereferences anything else as an AudioFrame."""
        publisher = await live_publisher()
        await publisher.write(b"\x00\x01" * 100)  # immutable bytes from the TTS provider
        assert publisher._resampler.pushed, "a TypeError here means bytes reached push()"

    @pytest.mark.asyncio
    async def test_native_pcm_is_split_into_frames_no_larger_than_the_frame_duration(self, livekit):
        publisher = await live_publisher(sample_rate=WIRE_RATE)
        await publisher.write_native(b"\x00\x01" * (FRAME_BYTES // 2 * 3))  # 3 frames' worth

        captured = publisher._source.captured
        assert len(captured) == 3
        assert {len(f.data) for f in captured} == {FRAME_BYTES}

    @pytest.mark.asyncio
    async def test_a_native_tail_shorter_than_a_frame_is_still_played(self, livekit):
        publisher = await live_publisher(sample_rate=WIRE_RATE)
        await publisher.write_native(b"\x00\x01" * (FRAME_BYTES // 2 + 10))

        sizes = [len(f.data) for f in publisher._source.captured]
        assert sizes == [FRAME_BYTES, 20], "the last partial frame must not be dropped"

    @pytest.mark.asyncio
    async def test_writing_before_start_is_a_no_op_not_a_crash(self):
        publisher = pub.MatrixRTCPublisher(_FakeRoom())
        await publisher.write(pcm(0.1))
        await publisher.write_native(pcm(0.1, WIRE_RATE))
        assert not publisher.live

    @pytest.mark.asyncio
    async def test_writing_after_close_is_dropped(self, livekit, no_drain):
        publisher = await live_publisher()
        source = publisher._source
        await publisher.close()
        await publisher.write(pcm(0.1))
        assert source.captured == [], "a closed publisher must not keep feeding the SFU"


class TestFlushAndClear:
    @pytest.mark.asyncio
    async def test_flush_drains_the_resampler_tail_before_waiting_for_playout(self, livekit):
        publisher = await live_publisher()
        await publisher.write(pcm(0.1))
        await publisher.flush()

        assert publisher._resampler.flushed
        assert publisher._source.events[-2:] == ["capture", "playout"], (
            "the tail must be queued before we wait for the queue to empty")

    @pytest.mark.asyncio
    async def test_clear_drops_audio_that_has_not_been_heard_yet(self, livekit):
        publisher = await live_publisher()
        publisher.clear()
        assert publisher._source.events == ["clear"]


class TestTeardown:
    """Same tokio trap as receiver.close(): the FFI worker outlives the handle."""

    @pytest.mark.asyncio
    async def test_close_unpublishes_then_lets_the_ffi_drain(self, livekit, no_drain):
        room = _FakeRoom()
        publisher = await live_publisher(room)
        source = publisher._source

        await publisher.close()

        assert room.local_participant.unpublished == [_FakePublication.sid]
        assert source.closed
        assert pub.FFI_DRAIN_DELAY in no_drain, "closing the source must be followed by a drain"
        assert not publisher.live

    @pytest.mark.asyncio
    async def test_closing_twice_neither_raises_nor_unpublishes_twice(self, livekit, no_drain):
        room = _FakeRoom()
        publisher = await live_publisher(room)
        await publisher.close()
        await publisher.close()
        assert room.local_participant.unpublished == [_FakePublication.sid]

    @pytest.mark.asyncio
    async def test_close_is_safe_before_anything_was_published(self, no_drain):
        await pub.MatrixRTCPublisher(_FakeRoom()).close()
        assert no_drain == [], "nothing was started, so there is no FFI worker to drain"


# --------------------------------------------------------------------------- adapter mixin


class _BaseStub:
    """Stands in for BasePlatformAdapter: only what the mixin calls through to."""

    def __init__(self):
        self.voice_messages = []

    async def play_tts(self, chat_id, audio_path, **kwargs) -> SendResult:
        self.voice_messages.append((chat_id, audio_path))
        return SendResult(success=True)


class _Adapter(ob.MatrixRTCOutboundMixin, _BaseStub):
    """Same MRO shape as the real MatrixAdapter."""


async def joined(**kw) -> tuple:
    """An adapter with a live call in ROOM; returns ``(adapter, publisher)``."""
    adapter = _Adapter()
    publisher = await adapter.start_rtc_audio(ROOM, _FakeRoom(), **kw)
    return adapter, publisher


class TestCallRegistry:
    def test_a_room_without_a_call_is_not_speakable(self):
        assert _Adapter().is_in_voice_channel(ROOM) is False

    @pytest.mark.asyncio
    async def test_joining_the_same_call_twice_reuses_one_publication(self, livekit):
        adapter, publisher = await joined()
        again = await adapter.start_rtc_audio(ROOM, _FakeRoom())
        assert again is publisher

    @pytest.mark.asyncio
    async def test_leaving_closes_the_publisher_and_ends_speakability(self, livekit, no_drain):
        adapter, publisher = await joined()
        await adapter.stop_rtc_audio(ROOM)

        assert not publisher.live
        assert adapter.is_in_voice_channel(ROOM) is False

    @pytest.mark.asyncio
    async def test_leaving_a_room_that_never_joined_is_a_no_op(self):
        await _Adapter().stop_rtc_audio(ROOM)

    def test_state_works_on_an_adapter_that_never_ran_init(self):
        """The house pattern builds adapters with object.__new__."""
        adapter = object.__new__(_Adapter)
        assert adapter.rtc_publishers == {}
        assert adapter.is_in_voice_channel(ROOM) is False


class TestStreamingSupport:
    @pytest.mark.asyncio
    async def test_streaming_is_declined_without_a_live_call(self):
        adapter = _Adapter()
        assert adapter.supports_streaming_tts(ROOM, AudioFormat()) is False
        assert await adapter.begin_streaming_tts(ROOM, AudioFormat()) is None

    @pytest.mark.asyncio
    async def test_the_contract_default_format_is_supported_during_a_call(self, livekit):
        adapter, _ = await joined()
        assert adapter.supports_streaming_tts(ROOM, AudioFormat()) is True

    @pytest.mark.asyncio
    async def test_a_non_s16_provider_is_declined(self, livekit):
        adapter, _ = await joined()
        assert adapter.supports_streaming_tts(ROOM, AudioFormat(sample_width=4)) is False

    @pytest.mark.asyncio
    async def test_a_channel_count_the_track_was_not_created_with_is_declined(self, livekit):
        adapter, _ = await joined()
        assert adapter.supports_streaming_tts(ROOM, AudioFormat(channels=2)) is False

    @pytest.mark.asyncio
    async def test_a_rate_the_resampler_was_not_built_for_falls_back_to_whole_file(self, livekit):
        """The resampler is built at start(); a provider at another rate must decline rather
        than silently play back at the wrong speed."""
        adapter, _ = await joined()
        assert await adapter.begin_streaming_tts(ROOM, AudioFormat(sample_rate=16000)) is None


class TestStreamingTurn:
    @pytest.mark.asyncio
    async def test_a_whole_turn_reaches_the_track_and_plays_out(self, livekit):
        adapter, publisher = await joined()
        handle = await adapter.begin_streaming_tts(ROOM, AudioFormat())

        await adapter.write_streaming_tts(handle, pcm(0.1))
        await adapter.write_streaming_tts(handle, pcm(0.1))
        await adapter.finish_streaming_tts(handle)

        assert publisher._resampler.pushed == [pcm(0.1), pcm(0.1)]
        assert publisher._source.events[-1] == "playout"

    @pytest.mark.asyncio
    async def test_an_interrupted_turn_drops_the_tail_instead_of_playing_it(self, livekit):
        adapter, publisher = await joined()
        handle = await adapter.begin_streaming_tts(ROOM, AudioFormat())
        await adapter.write_streaming_tts(handle, pcm(0.1))

        await adapter.finish_streaming_tts(handle, interrupted=True)
        assert publisher._source.events[-1] == "clear"
        assert "playout" not in publisher._source.events

    @pytest.mark.asyncio
    async def test_a_superseded_turn_can_no_longer_write(self, livekit):
        """One track per room: two overlapping turns would interleave clauses and share
        the resampler's state."""
        adapter, publisher = await joined()
        first = await adapter.begin_streaming_tts(ROOM, AudioFormat())
        second = await adapter.begin_streaming_tts(ROOM, AudioFormat())

        await adapter.write_streaming_tts(first, pcm(0.1))
        await adapter.write_streaming_tts(second, pcm(0.2))

        assert publisher._resampler.pushed == [pcm(0.2)], "the newest turn holds the floor"
        assert first is not second

    @pytest.mark.asyncio
    async def test_abort_clears_the_queue_and_silences_late_chunks(self, livekit):
        adapter, publisher = await joined()
        handle = await adapter.begin_streaming_tts(ROOM, AudioFormat())
        await adapter.write_streaming_tts(handle, pcm(0.1))

        await adapter.abort_streaming_tts(handle, error="cancelled")
        await adapter.write_streaming_tts(handle, pcm(0.1))

        assert handle.aborted
        assert publisher._source.events[-1] == "clear"
        assert publisher._resampler.pushed == [pcm(0.1)], "post-abort chunks must be dropped"

    @pytest.mark.asyncio
    async def test_abort_is_idempotent(self, livekit):
        adapter, _ = await joined()
        handle = await adapter.begin_streaming_tts(ROOM, AudioFormat())
        await adapter.abort_streaming_tts(handle)
        await adapter.abort_streaming_tts(handle, error="again")
        await adapter.finish_streaming_tts(handle)

    @pytest.mark.asyncio
    async def test_writing_after_the_call_ended_does_not_raise(self, livekit, no_drain):
        adapter, _ = await joined()
        handle = await adapter.begin_streaming_tts(ROOM, AudioFormat())
        await adapter.stop_rtc_audio(ROOM)

        await adapter.write_streaming_tts(handle, pcm(0.1))
        await adapter.finish_streaming_tts(handle)


# --------------------------------------------------------------------------- whole-file path


@pytest.fixture
def decoded(monkeypatch):
    """Replace the ffmpeg decode with a recorder returning one frame of 48 kHz PCM."""
    calls = {}

    def fake_decode(path, channels=1):
        calls["path"], calls["channels"] = path, channels
        return calls.get("pcm", b"\x00\x01" * (FRAME_BYTES // 2))

    monkeypatch.setattr(ob, "decode_to_livekit_pcm", fake_decode)
    return calls


class TestWholeFileFallback:
    @pytest.mark.asyncio
    async def test_playback_without_a_call_reports_failure_rather_than_swallowing_it(self):
        assert await _Adapter().play_in_voice_channel(ROOM, "/tmp/reply.mp3") is False

    @pytest.mark.asyncio
    async def test_a_file_is_decoded_to_the_wire_rate_and_played_out(self, livekit, decoded):
        adapter, publisher = await joined()

        assert await adapter.play_in_voice_channel(ROOM, "/tmp/reply.mp3") is True
        assert decoded["path"] == "/tmp/reply.mp3"
        assert decoded["channels"] == publisher.channels
        assert [f.sample_rate for f in publisher._source.captured] == [WIRE_RATE]
        assert publisher._source.events[-1] == "playout"

    @pytest.mark.asyncio
    async def test_the_whole_file_path_never_touches_the_streaming_resampler(self, livekit, decoded):
        """Its state belongs to whichever streamed turn is mid-sentence: flushing its tail
        here would splice a fragment of that turn onto the end of the file."""
        adapter, publisher = await joined()
        await adapter.play_in_voice_channel(ROOM, "/tmp/reply.mp3")
        assert publisher._resampler.pushed == []
        assert publisher._resampler.flushed is False

    @pytest.mark.asyncio
    async def test_a_file_that_would_not_decode_reports_failure(self, livekit, decoded):
        adapter, _ = await joined()
        decoded["pcm"] = b""
        assert await adapter.play_in_voice_channel(ROOM, "/tmp/broken.mp3") is False

    @pytest.mark.asyncio
    async def test_auto_tts_speaks_into_the_call_when_one_is_live(self, livekit, decoded):
        adapter, _ = await joined()
        result = await adapter.play_tts(ROOM, "/tmp/reply.mp3")

        assert result.success is True
        assert adapter.voice_messages == [], "a live call must not also get a voice bubble"

    @pytest.mark.asyncio
    async def test_auto_tts_falls_back_to_a_voice_message_without_a_call(self):
        adapter = _Adapter()
        await adapter.play_tts(ROOM, "/tmp/reply.mp3")
        assert adapter.voice_messages == [(ROOM, "/tmp/reply.mp3")]


class TestDecode:
    def test_no_ffmpeg_means_no_audio_rather_than_an_exception(self, monkeypatch, tmp_path):
        monkeypatch.setattr(ob.shutil, "which", lambda _name: None)
        path = tmp_path / "reply.mp3"
        path.write_bytes(b"not really an mp3")
        assert ob.decode_to_livekit_pcm(str(path)) == b""

    def test_a_missing_file_is_not_handed_to_ffmpeg(self, monkeypatch):
        monkeypatch.setattr(ob.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
        monkeypatch.setattr(ob.subprocess, "run", lambda *a, **kw: pytest.fail("ffmpeg ran"))
        assert ob.decode_to_livekit_pcm("/nope/missing.mp3") == b""

    def test_ffmpeg_is_asked_for_the_wire_rate_and_raw_s16le(self, monkeypatch, tmp_path):
        path = tmp_path / "reply.mp3"
        path.write_bytes(b"x")
        seen = {}

        def fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            return types.SimpleNamespace(returncode=0, stdout=b"\x00\x01")

        monkeypatch.setattr(ob.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
        monkeypatch.setattr(ob.subprocess, "run", fake_run)

        assert ob.decode_to_livekit_pcm(str(path), channels=1) == b"\x00\x01"
        cmd = seen["cmd"]
        assert cmd[cmd.index("-ar") + 1] == str(WIRE_RATE)
        assert cmd[cmd.index("-f") + 1] == "s16le"
        assert cmd[cmd.index("-ac") + 1] == "1"

    def test_an_ffmpeg_failure_yields_no_audio(self, monkeypatch, tmp_path):
        path = tmp_path / "reply.mp3"
        path.write_bytes(b"x")
        monkeypatch.setattr(ob.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
        monkeypatch.setattr(ob.subprocess, "run",
                            lambda *a, **kw: types.SimpleNamespace(returncode=1, stdout=b""))
        assert ob.decode_to_livekit_pcm(str(path)) == b""


# --------------------------------------------------------------------------- wiring


class TestAdapterCarriesTheContract:
    def test_the_real_matrix_adapter_implements_the_streaming_hooks(self):
        """Dead code otherwise: the gateway only ever calls these on the adapter."""
        from gateway.platforms.base import BasePlatformAdapter
        from plugins.platforms.matrix.adapter import MatrixAdapter

        for name in ("supports_streaming_tts", "begin_streaming_tts", "write_streaming_tts",
                     "finish_streaming_tts", "abort_streaming_tts", "play_tts"):
            assert getattr(MatrixAdapter, name) is not getattr(BasePlatformAdapter, name), name
        assert hasattr(MatrixAdapter, "play_in_voice_channel")

    def test_an_uninitialised_matrix_adapter_declines_streaming(self):
        from plugins.platforms.matrix.adapter import MatrixAdapter

        adapter = object.__new__(MatrixAdapter)
        assert adapter.supports_streaming_tts(ROOM, AudioFormat()) is False

    def test_the_handle_is_the_gateways_own_type(self):
        """StreamingTTSConsumer reads .audible/.aborted off whatever begin() returns."""
        assert set(StreamingTTSHandle.__dataclass_fields__) >= {
            "chat_id", "audio_format", "audible", "aborted"}
