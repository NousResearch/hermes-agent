"""Slice 6b: live aiortc PCM streaming track + direct-feed + start() wiring.

The track scenarios require aiortc + av (``importorskip`` — skip in CI, run
locally). The start()-wiring scenario is CI-runnable: it drives the streaming
branch of ``AiortcAudioPeer.start()`` with a fully faked peer connection (no
aiortc) and a real ``StreamingPipeline``, asserting the transport sink is
swapped + the direct-feed accumulator is selected + the pipeline is retained.
"""
from __future__ import annotations

import asyncio

import pytest

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Scenarios 1-5: live PCM track + direct-feed (require aiortc/av)
# ---------------------------------------------------------------------------


def _make_16k_frame(samples: int = 320):
    """Build a 16k mono s16 av.AudioFrame with non-zero samples."""
    import fractions

    from av import AudioFrame

    frame = AudioFrame(format="s16", layout="mono", samples=samples)
    frame.sample_rate = 16000
    frame.pts = 0
    frame.time_base = fractions.Fraction(1, 16000)
    pcm = (b"\x10\x10") * samples
    for plane in frame.planes:
        plane.update(pcm)
    return frame


async def test_track_plays_enqueued_audio_resampled_to_target_rate():
    pytest.importorskip("aiortc")
    pytest.importorskip("av")
    from gateway.calls.native.aiortc_engine import _create_pcm_streaming_track

    track = _create_pcm_streaming_track(48000)
    await track.enqueue(_make_16k_frame(320))  # 20ms @ 16k
    out = await track.recv()
    assert out.sample_rate == 48000
    assert int(getattr(out, "samples", 0) or 0) > 0


async def test_track_silence_when_empty():
    pytest.importorskip("aiortc")
    pytest.importorskip("av")
    from gateway.calls.native.aiortc_engine import _create_pcm_streaming_track

    track = _create_pcm_streaming_track(48000)
    out = await asyncio.wait_for(track.recv(), timeout=1.0)
    assert out.sample_rate == 48000
    assert int(getattr(out, "samples", 0) or 0) > 0


async def test_track_drop_pending_empties_and_reports_count():
    pytest.importorskip("aiortc")
    pytest.importorskip("av")
    from gateway.calls.native.aiortc_engine import _create_pcm_streaming_track

    track = _create_pcm_streaming_track(48000)
    for _ in range(3):
        await track.enqueue(_make_16k_frame(320))
    dropped = track.drop_pending()
    assert dropped == 3
    assert track.drop_pending() == 0


async def test_track_resampler_continuity_across_frames():
    pytest.importorskip("aiortc")
    pytest.importorskip("av")
    from gateway.calls.native.aiortc_engine import _create_pcm_streaming_track

    track = _create_pcm_streaming_track(48000)
    # Two consecutive 16k 20ms frames (320 samples each) -> ~960 samples each at
    # 48k; total output across the pair ~= 3x the input (state preserved).
    total_in = 0
    total_out = 0
    for _ in range(2):
        await track.enqueue(_make_16k_frame(320))
        total_in += 320
        out = await track.recv()
        total_out += int(getattr(out, "samples", 0) or 0)
    assert abs(total_out - 3 * total_in) <= 96  # within ~one 48k frame


async def test_recv_drains_multiple_resampler_frames():
    pytest.importorskip("aiortc")
    pytest.importorskip("av")
    from gateway.calls.native.aiortc_engine import _create_pcm_streaming_track

    track = _create_pcm_streaming_track(48000)
    # A single LARGE 16k input frame (4000 samples, 8000 bytes int16) resamples
    # to ~12000 samples at 48k, which spans MORE than one output AudioFrame.
    # av.AudioResampler.resample() returns >1 frame; the pre-fix recv() returned
    # only the first and discarded the rest, falling short of the ~3x count.
    input_samples = 4000
    await track.enqueue(_make_16k_frame(input_samples))

    real_frames = []
    last_pts = -1
    for _ in range(40):
        out = await asyncio.wait_for(track.recv(), timeout=1.0)
        samples = int(getattr(out, "samples", 0) or 0)
        pts = int(getattr(out, "pts", 0) or 0)
        assert pts > last_pts  # pts strictly monotonically increasing
        last_pts = pts
        # Silence fallback frames carry exactly target_rate//50 (= 960) samples
        # of zeros once the resampled audio is exhausted; the real resampled
        # frames are the ones we accumulate until we hit that fallback.
        if not out.to_ndarray().any():
            break
        real_frames.append(out)

    # (a) More than one real resampled frame returned (nothing discarded).
    assert len(real_frames) > 1
    # (b) Total samples across drained real frames ~= 3x input (16k -> 48k).
    # The resampler holds back a partial trailing frame (< one frame_size of
    # 960 samples) until the next input/flush, so allow up to one frame of slack.
    total_out = sum(int(getattr(f, "samples", 0) or 0) for f in real_frames)
    assert abs(total_out - 3 * input_samples) <= 960  # within ~one 48k frame


async def test_direct_feed_accumulator_calls_process_pcm16_and_ignores_ack():
    pytest.importorskip("aiortc")
    pytest.importorskip("av")
    from gateway.calls.native.aiortc_engine import _DirectFeedAccumulator

    class SpyPipeline:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def process_pcm16(self, *, call_id, pcm16, sample_rate):
            self.calls.append(
                {"call_id": call_id, "pcm16": pcm16, "sample_rate": sample_rate}
            )
            return object()  # ack ignored

    pipeline = SpyPipeline()
    acc = _DirectFeedAccumulator(pipeline, "call-1", 48000)
    result = await acc.accept_pcm16(b"\x01\x02\x03\x04")
    assert result is None
    assert len(pipeline.calls) == 1
    assert pipeline.calls[0] == {
        "call_id": "call-1",
        "pcm16": b"\x01\x02\x03\x04",
        "sample_rate": 48000,
    }


# ---------------------------------------------------------------------------
# Scenario 6 (CI-runnable): start() streaming branch wiring (no aiortc)
# ---------------------------------------------------------------------------


class _FakePeerConnection:
    def __init__(self, *args, **kwargs) -> None:
        self.added_tracks: list = []
        self._handlers: dict = {}

    def on(self, event):
        def _register(fn):
            self._handlers[event] = fn
            return fn

        return _register

    def addTrack(self, track) -> None:  # noqa: N802 - aiortc API name
        self.added_tracks.append(track)


async def test_start_streaming_branch_wires_sink_and_direct_feed(monkeypatch):
    """CI-runnable: the streaming branch swaps the transport sink + uses the
    direct-feed accumulator + retains the pipeline for teardown."""
    from gateway.calls.native import aiortc_engine
    from gateway.calls.native.aiortc_engine import (
        AiortcAudioPeer,
        SimplexAiortcConfig,
        _DirectFeedAccumulator,
    )
    from gateway.calls.native.streaming.aiortc_transport import build_streaming_pipeline

    # Stub aiortc peer-type loading + transport policy (no aiortc on CI).
    monkeypatch.setattr(
        aiortc_engine,
        "_load_aiortc_peer_types",
        lambda: (
            lambda config: _FakePeerConnection(),
            lambda **kwargs: None,
            lambda **kwargs: None,
        ),
    )
    monkeypatch.setattr(aiortc_engine, "apply_aiortc_transport_policy", lambda policy: None)

    # Stub the PCM track builder so we need no aiortc/av; expose enqueue/drop.
    class FakeStreamingTrack:
        def __init__(self) -> None:
            self.dropped = 0

        async def enqueue(self, frame) -> None:  # pragma: no cover - not called here
            return None

        def drop_pending(self) -> int:  # pragma: no cover - not called here
            self.dropped += 1
            return 0

    fake_track = FakeStreamingTrack()
    monkeypatch.setattr(
        aiortc_engine,
        "_create_pcm_streaming_track",
        lambda target_rate: fake_track,
    )

    # Spy on the transport's set_outbound_sink.
    pipeline = build_streaming_pipeline({}, cognitive="fake", sink=None)
    captured: dict = {}
    real_set = pipeline.transport.set_outbound_sink

    def _spy(sink, *, drop=None):
        captured["sink"] = sink
        captured["drop"] = drop
        return real_set(sink, drop=drop)

    monkeypatch.setattr(pipeline.transport, "set_outbound_sink", _spy)

    config = SimplexAiortcConfig()
    peer = AiortcAudioPeer(config)
    await peer.start("call-xyz", pipeline)

    # Sink swapped to the track's enqueue + drop hook to drop_pending.
    assert captured["sink"] == fake_track.enqueue
    assert captured["drop"] == fake_track.drop_pending
    # Direct-feed accumulator selected (not the turn-based one).
    assert isinstance(peer._accumulator, _DirectFeedAccumulator)
    # Pipeline retained for teardown.
    assert peer._pipeline is pipeline
    # Track added to the peer connection.
    assert peer._pc.added_tracks == [fake_track]


async def test_start_non_streaming_path_unchanged(monkeypatch):
    """A non-streaming pipeline keeps the turn-based accumulator + queued track."""
    from gateway.calls.native import aiortc_engine
    from gateway.calls.native.aiortc_engine import AiortcAudioPeer, SimplexAiortcConfig
    from gateway.calls.native.webrtc_media import AudioUtteranceAccumulator

    monkeypatch.setattr(
        aiortc_engine,
        "_load_aiortc_peer_types",
        lambda: (
            lambda config: _FakePeerConnection(),
            lambda **kwargs: None,
            lambda **kwargs: None,
        ),
    )
    monkeypatch.setattr(aiortc_engine, "apply_aiortc_transport_policy", lambda policy: None)

    sentinel_track = object()
    monkeypatch.setattr(
        aiortc_engine,
        "_create_queued_audio_track",
        lambda sample_rate, event_sink: sentinel_track,
    )

    class TurnPipeline:
        is_streaming = False

        async def process_pcm16(self, **kwargs):  # pragma: no cover
            return None

    peer = AiortcAudioPeer(SimplexAiortcConfig())
    await peer.start("call-abc", TurnPipeline())

    assert isinstance(peer._accumulator, AudioUtteranceAccumulator)
    assert peer._pc.added_tracks == [sentinel_track]
    assert getattr(peer, "_pipeline", None) is None or not getattr(
        peer._pipeline, "is_streaming", False
    )
