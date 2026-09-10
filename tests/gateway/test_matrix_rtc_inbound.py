"""Phase 1 behaviour contracts for MatrixRTC inbound audio.

Synthetic PCM only — nothing here opens a socket, joins an SFU, or needs the LiveKit
SDK installed. The receiver test drives a fake room so the teardown contract (the FFI
drain delay) is asserted without native code.
"""

import asyncio
import logging
import sys
import types
import wave
from unittest.mock import patch

import pytest

from plugins.platforms.matrix.rtc import segmenter as seg
from plugins.platforms.matrix.rtc.focus import (
    MatrixRTCError, discover_livekit_focus, fetch_livekit_credentials, request_openid_token)

RATE = seg.SAMPLE_RATE
CHANNELS = seg.CHANNELS
ALICE = "@alice:hs:AAA"


def pcm(seconds: float) -> bytes:
    """`seconds` of speech-level audio at the segmenter's native format.

    The level is what matters, not the waveform: anything below ``seg.SPEECH_RMS`` is
    comfort noise to the segmenter and is never buffered, so audio a test means as speech
    has to clear that floor. RMS 1024 here, five times over it.
    """
    return b"\x00\x04" * int(RATE * CHANNELS * seconds)


def quiet(seconds: float) -> bytes:
    """`seconds` of the comfort noise a decoded WebRTC sink delivers between utterances.

    RMS 2. LiveKit hands us these continuously for the whole call — which is why frame
    arrival cannot be what tells the segmenter someone is talking.
    """
    return b"\x02\x00" * int(RATE * CHANNELS * seconds)


# --------------------------------------------------------------------------- segmenter


class TestTurnSegmentation:
    def test_utterance_released_only_after_the_silence_threshold(self):
        s = seg.TurnSegmenter()
        s.feed("@alice:hs:AAA", pcm(1.0), now=100.0)

        # Still mid-turn: a pause shorter than the threshold is part of the sentence.
        assert s.check_silence(now=100.0 + s.silence_threshold - 0.01) == []

        released = s.check_silence(now=100.0 + s.silence_threshold)
        assert [identity for identity, _ in released] == ["@alice:hs:AAA"]
        assert len(released[0][1]) == len(pcm(1.0))

    def test_released_buffer_does_not_replay_on_the_next_poll(self):
        s = seg.TurnSegmenter()
        s.feed("@alice:hs:AAA", pcm(1.0), now=0.0)
        assert len(s.check_silence(now=5.0)) == 1
        assert s.check_silence(now=10.0) == []

    def test_speech_shorter_than_the_minimum_is_never_delivered(self):
        s = seg.TurnSegmenter()
        s.feed("@alice:hs:AAA", pcm(0.2), now=0.0)
        assert s.check_silence(now=0.0 + s.silence_threshold) == []

    def test_abandoned_sub_minimum_noise_is_discarded_not_accumulated(self):
        """Every cough would otherwise leave a buffer entry alive for the whole call."""
        s = seg.TurnSegmenter()
        for i in range(3):
            s.feed(f"@noise{i}:hs:AAA", pcm(0.1), now=0.0)
        assert s.check_silence(now=s.silence_threshold) == []
        assert s._buffers  # not yet: only past 2x the threshold is a buffer abandoned

        assert s.check_silence(now=s.silence_threshold * 2) == []
        assert not s._buffers, "stale sub-minimum buffers must be dropped"

    def test_speakers_are_segmented_independently(self):
        s = seg.TurnSegmenter()
        s.feed("@alice:hs:AAA", pcm(1.0), now=0.0)
        s.feed("@bob:hs:BBB", pcm(1.0), now=0.0)
        # Alice keeps talking; Bob has stopped.
        s.feed("@alice:hs:AAA", pcm(1.0), now=2.0)

        released = s.check_silence(now=2.0)
        assert [identity for identity, _ in released] == ["@bob:hs:BBB"]
        assert len(s.check_silence(now=4.0)) == 1, "Alice releases once she stops too"

    def test_a_continuing_speaker_accumulates_rather_than_splitting(self):
        s = seg.TurnSegmenter()
        s.feed("@alice:hs:AAA", pcm(0.6), now=0.0)
        s.feed("@alice:hs:AAA", pcm(0.6), now=0.5)
        released = s.check_silence(now=0.5 + s.silence_threshold)
        assert len(released) == 1
        assert len(released[0][1]) == len(pcm(1.2))

    def test_flush_pending_drains_speech_and_drops_noise(self):
        s = seg.TurnSegmenter()
        s.feed("@alice:hs:AAA", pcm(1.0), now=0.0)
        s.feed("@bob:hs:BBB", pcm(0.1), now=0.0)

        flushed = s.flush_pending()
        assert [identity for identity, _ in flushed] == ["@alice:hs:AAA"]
        assert not s._buffers, "flush leaves nothing behind"
        assert s.flush_pending() == []

    def test_empty_feed_does_not_create_a_speaker(self):
        s = seg.TurnSegmenter()
        s.feed("@alice:hs:AAA", b"", now=0.0)
        assert not s._buffers

    def test_thresholds_come_from_config_yaml_not_the_environment(self):
        with patch.object(seg, "_rtc_config",
                          return_value={"silence_threshold": 3.0, "min_speech_duration": 0.25}):
            s = seg.TurnSegmenter()
        assert (s.silence_threshold, s.min_speech_duration) == (3.0, 0.25)

        s.feed("@alice:hs:AAA", pcm(0.3), now=0.0)
        assert s.check_silence(now=2.9) == [], "configured threshold is the one in force"
        assert len(s.check_silence(now=3.0)) == 1

    @pytest.mark.parametrize("bad", [0, -1, "loud", None])
    def test_unusable_config_falls_back_to_the_discord_defaults(self, bad):
        with patch.object(seg, "_rtc_config",
                          return_value={"silence_threshold": bad, "min_speech_duration": bad}):
            s = seg.TurnSegmenter()
        assert s.silence_threshold == seg.SILENCE_THRESHOLD
        assert s.min_speech_duration == seg.MIN_SPEECH_DURATION

    def test_duration_math_tracks_the_declared_rate(self):
        """A rate change must move the speech/noise boundary, not silently mis-measure it."""
        fast = seg.TurnSegmenter(sample_rate=48000, channels=1)
        # 0.5s at 16 kHz is only ~0.167s at 48 kHz — below the minimum, so not speech.
        fast.feed("@alice:hs:AAA", pcm(0.5), now=0.0)
        assert fast.check_silence(now=2.0) == []


class TestVoiceActivityDetection:
    """A LiveKit stream never stops, so a frame arriving cannot mean someone is talking.

    Discord's receiver gets RTP only while a user speaks, which is why the ported timers
    could take frame arrival as speech. A decoded WebRTC sink delivers comfort noise for
    the whole call instead, so ``check_silence`` never saw a gap and no turn ever ended:
    the bot heard everything and answered nothing.
    """

    def test_a_turn_ends_while_the_stream_keeps_delivering_frames(self):
        """The production bug, in one test: silence is a level, not an absence of frames."""
        s = seg.TurnSegmenter()
        s.feed(ALICE, pcm(1.0), now=0.0)
        for tick in range(1, 21):  # 2 s of comfort noise, in the 100 ms the SDK ships
            s.feed(ALICE, quiet(0.1), now=tick * 0.1)

        released = s.check_silence(now=2.0)
        assert [identity for identity, _ in released] == [ALICE]
        assert len(released[0][1]) == len(pcm(1.0)), "the speech, not the silence after it"

    def test_comfort_noise_alone_never_becomes_an_utterance(self):
        s = seg.TurnSegmenter()
        for tick in range(200):
            s.feed(ALICE, quiet(0.1), now=tick * 0.1)

        assert s.check_silence(now=1e6) == []
        assert not s._buffers, "a silent stream must not even create a speaker"

    def test_a_pause_inside_a_sentence_does_not_split_the_utterance(self):
        s = seg.TurnSegmenter()
        s.feed(ALICE, pcm(0.6), now=0.0)
        s.feed(ALICE, quiet(0.5), now=0.6)
        s.feed(ALICE, pcm(0.6), now=1.1)

        released = s.check_silence(now=1.1 + s.silence_threshold)
        assert len(released) == 1
        assert len(released[0][1]) == len(pcm(1.2)), "one turn, and none of the pause in it"

    def test_quiet_frames_do_not_hold_a_sub_minimum_buffer_open(self):
        """Otherwise every cough leaves an entry alive for the length of the call."""
        s = seg.TurnSegmenter()
        s.feed(ALICE, pcm(0.1), now=0.0)
        for tick in range(1, 61):
            s.feed(ALICE, quiet(0.1), now=tick * 0.1)

        assert s.check_silence(now=6.0) == []
        assert not s._buffers

    def test_the_floor_comes_from_config_yaml_not_the_environment(self):
        with patch.object(seg, "_rtc_config", return_value={"speech_rms": 10_000}):
            s = seg.TurnSegmenter()
        assert s.speech_rms == 10_000

        s.feed(ALICE, pcm(1.0), now=0.0)
        assert s.check_silence(now=1e9) == [], "a floor that high hears nothing as speech"

    @pytest.mark.parametrize("bad", [0, -1, "loud", None])
    def test_an_unusable_floor_falls_back_to_the_default(self, bad):
        """A floor of zero is the bug this whole class is about — never accept one."""
        with patch.object(seg, "_rtc_config", return_value={"speech_rms": bad}):
            assert seg.TurnSegmenter().speech_rms == seg.SPEECH_RMS

    def test_a_released_utterance_is_logged_with_its_duration_and_level(self, caplog):
        """The next live call has to be diagnosable from gateway.log alone."""
        s = seg.TurnSegmenter()
        s.feed(ALICE, pcm(1.0), now=0.0)
        with caplog.at_level(logging.INFO, logger=seg.__name__):
            s.check_silence(now=1e9)

        line, = [r.getMessage() for r in caplog.records if "utterance" in r.getMessage()]
        assert "1.00s" in line and "1024" in line and ALICE in line


# ------------------------------------------------------------------------ pcm -> wav


class TestPcmToWav:
    def test_writes_a_whisper_ready_container_without_ffmpeg(self, tmp_path):
        raw = pcm(0.5)
        out = str(tmp_path / "utterance.wav")

        with patch("subprocess.run", side_effect=AssertionError("no ffmpeg on this path")):
            seg.pcm_to_wav(raw, out)

        with wave.open(out, "rb") as w:
            assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (RATE, CHANNELS, 2)
            assert w.readframes(w.getnframes()) == raw


# ----------------------------------------------------------------------- transcription


def _stub_transcription(monkeypatch, result, hallucination=False):
    calls = {}

    def fake_transcribe(path, source=None, **kwargs):
        calls["path"] = path
        calls["source"] = source
        with wave.open(path, "rb") as w:
            calls["rate"] = w.getframerate()
        return result

    monkeypatch.setitem(
        sys.modules, "tools.transcription_tools",
        types.SimpleNamespace(transcribe_audio=fake_transcribe))
    monkeypatch.setitem(
        sys.modules, "tools.voice_mode_transcript",
        types.SimpleNamespace(is_whisper_hallucination=lambda _t: hallucination))
    return calls


class TestTranscribePcm:
    def test_successful_transcript_is_returned_and_the_temp_wav_is_removed(self, monkeypatch):
        calls = _stub_transcription(
            monkeypatch, {"success": True, "transcript": "  turn the lights on  "})

        assert seg.transcribe_pcm(pcm(1.0)) == "turn the lights on"
        assert calls["rate"] == RATE, "Whisper must receive 16 kHz"
        assert calls["source"] == "voice_mode"
        import os
        assert not os.path.exists(calls["path"]), "temp wav must not leak"

    def test_hallucinated_transcript_is_suppressed(self, monkeypatch):
        _stub_transcription(
            monkeypatch, {"success": True, "transcript": "Thank you."}, hallucination=True)
        assert seg.transcribe_pcm(pcm(1.0)) is None

    def test_failed_transcription_returns_none_rather_than_raising(self, monkeypatch):
        _stub_transcription(monkeypatch, {"success": False, "error": "no STT provider"})
        assert seg.transcribe_pcm(pcm(1.0)) is None

    def test_empty_transcript_is_suppressed(self, monkeypatch):
        _stub_transcription(monkeypatch, {"success": True, "transcript": "   "})
        assert seg.transcribe_pcm(pcm(1.0)) is None


# ------------------------------------------------------------------------- focus/JWT


class _FakeResponse:
    def __init__(self, status, body):
        self.status, self._body = status, body

    async def json(self, content_type=None):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    """Replays queued responses and records what was sent."""

    def __init__(self, get=None, post=None):
        self._get, self._post = list(get or []), list(post or [])
        self.posts = []

    def get(self, url, **kwargs):
        return _FakeResponse(*self._get.pop(0))

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        return _FakeResponse(*self._post.pop(0))


WELL_KNOWN_OK = (200, {"org.matrix.msc4143.rtc_foci": [
    {"type": "livekit", "livekit_service_url": "https://call.hs/livekit/jwt/"}]})


class TestFocusDiscovery:
    @pytest.mark.asyncio
    async def test_livekit_focus_url_is_normalised(self):
        session = _FakeSession(get=[WELL_KNOWN_OK])
        assert await discover_livekit_focus(session, "https://hs/") == "https://call.hs/livekit/jwt"

    @pytest.mark.asyncio
    async def test_a_non_livekit_focus_is_not_mistaken_for_one(self):
        session = _FakeSession(get=[(200, {"org.matrix.msc4143.rtc_foci": [
            {"type": "jitsi", "preferredDomain": "meet.hs"}]})])
        with pytest.raises(MatrixRTCError, match="no livekit focus"):
            await discover_livekit_focus(session, "https://hs")

    @pytest.mark.asyncio
    async def test_missing_well_known_raises(self):
        with pytest.raises(MatrixRTCError, match="HTTP 404"):
            await discover_livekit_focus(_FakeSession(get=[(404, {})]), "https://hs")

    @pytest.mark.asyncio
    async def test_openid_failure_never_echoes_the_access_token(self):
        session = _FakeSession(post=[(403, {"errcode": "M_FORBIDDEN"})])
        with pytest.raises(MatrixRTCError) as excinfo:
            await request_openid_token(session, "https://hs", "@bot:hs", "super-secret-token")
        assert "super-secret-token" not in str(excinfo.value)


class TestJwtExchange:
    @pytest.mark.asyncio
    async def test_full_chain_returns_the_sfu_url_the_jwt_and_the_focus(self):
        session = _FakeSession(
            get=[WELL_KNOWN_OK],
            post=[(200, {"access_token": "oid", "matrix_server_name": "hs"}),
                  (200, {"url": "wss://call.hs/livekit/sfu", "jwt": "j.w.t"})])

        url, jwt, focus = await fetch_livekit_credentials(
            "https://hs", "@bot:hs", "tok", "!room:hs", "DEVICE1", session=session)

        assert (url, jwt) == ("wss://call.hs/livekit/sfu", "j.w.t")
        assert focus == "https://call.hs/livekit/jwt", \
            "the JWT service, not the SFU: it is what our own call membership advertises"
        sfu_url, sfu_kwargs = session.posts[-1]
        assert sfu_url == "https://call.hs/livekit/jwt/sfu/get"
        assert sfu_kwargs["json"] == {
            "room": "!room:hs",
            "openid_token": {"access_token": "oid", "matrix_server_name": "hs"},
            "device_id": "DEVICE1",
        }

    @pytest.mark.asyncio
    async def test_a_fresh_openid_token_is_minted_for_the_exchange(self):
        """The homeserver treats the OpenID token as single use — never cache one."""
        session = _FakeSession(
            get=[WELL_KNOWN_OK],
            post=[(200, {"access_token": "oid"}), (200, {"url": "wss://s", "jwt": "j"})])
        await fetch_livekit_credentials(
            "https://hs", "@bot:hs", "tok", "!room:hs", "D1", session=session)
        assert sum("openid/request_token" in url for url, _ in session.posts) == 1

    @pytest.mark.asyncio
    async def test_sfu_get_without_a_jwt_is_an_error_not_a_silent_none(self):
        session = _FakeSession(get=[WELL_KNOWN_OK],
                               post=[(200, {"access_token": "oid"}), (200, {"url": "wss://s"})])
        with pytest.raises(MatrixRTCError, match="no url/jwt"):
            await fetch_livekit_credentials(
                "https://hs", "@bot:hs", "tok", "!room:hs", "D1", session=session)


# --------------------------------------------------------------------------- receiver


class _FakeRoom:
    def __init__(self):
        self.disconnected = False

    async def disconnect(self):
        self.disconnected = True


class TestReceiverTeardown:
    """The LiveKit FFI worker outlives disconnect(); skipping the drain aborts the process."""

    def _receiver(self, seen):
        from plugins.platforms.matrix.rtc.receiver import MatrixRTCReceiver

        async def on_transcript(identity, transcript):
            seen.append((identity, transcript))

        return MatrixRTCReceiver(on_transcript)

    @pytest.mark.asyncio
    async def test_close_disconnects_then_waits_for_the_ffi_to_drain(self, monkeypatch):
        from plugins.platforms.matrix.rtc import receiver as rcv

        rx = self._receiver([])
        room = _FakeRoom()
        rx._room = room

        order = []
        real_sleep = asyncio.sleep

        async def tracking_sleep(delay):
            order.append(delay)
            await real_sleep(0)

        monkeypatch.setattr(rcv.asyncio, "sleep", tracking_sleep)
        await rx.close()

        assert room.disconnected
        assert rcv.FFI_DRAIN_DELAY in order, "disconnect() must be followed by the drain delay"
        assert rx._room is None

    @pytest.mark.asyncio
    async def test_close_emits_the_utterance_still_in_the_buffer(self, monkeypatch):
        from plugins.platforms.matrix.rtc import receiver as rcv

        seen = []
        rx = self._receiver(seen)
        rx._room = _FakeRoom()
        rx.segmenter.feed("@alice:hs:AAA", pcm(1.0), now=0.0)
        monkeypatch.setattr(rcv, "transcribe_pcm", lambda *a, **kw: "goodbye")

        await rx.close()
        assert seen == [("@alice:hs:AAA", "goodbye")]

    @pytest.mark.asyncio
    async def test_close_is_safe_before_any_room_was_joined(self):
        await self._receiver([]).close()

    @pytest.mark.asyncio
    async def test_a_failing_callback_does_not_break_the_call(self, monkeypatch):
        from plugins.platforms.matrix.rtc.receiver import MatrixRTCReceiver
        from plugins.platforms.matrix.rtc import receiver as rcv

        async def boom(identity, transcript):
            raise RuntimeError("session dispatch failed")

        rx = MatrixRTCReceiver(boom)
        rx._room = _FakeRoom()
        rx.segmenter.feed("@alice:hs:AAA", pcm(1.0), now=0.0)
        monkeypatch.setattr(rcv, "transcribe_pcm", lambda *a, **kw: "hello")

        await rx.close()  # must not raise


# ----------------------------------------------------------------- track subscription


class _FakeTrack:
    def __init__(self, kind, frames=()):
        self.kind = kind
        self.frames = list(frames)


def _fake_livekit(record):
    """Stand-in for ``livekit.rtc`` — just enough surface for connect()/_drain_track().

    ``AudioStream``'s defaults mirror the real SDK's (48 kHz), so a receiver that
    forgets to ask for 16 kHz shows up here instead of silently handing Whisper
    triple-speed audio.
    """
    class FakeAudioStream:
        def __init__(self, track, sample_rate=48000, num_channels=1):
            record["asked_for"] = (sample_rate, num_channels)
            self._frames = list(track.frames)

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._frames:
                raise StopAsyncIteration
            return types.SimpleNamespace(
                frame=types.SimpleNamespace(data=self._frames.pop(0)))

        async def aclose(self):
            record["stream_closed"] = True

    class FakeRoom:
        def __init__(self):
            self._handlers = {}
            self.local_participant = types.SimpleNamespace(identity="@bot:hs:BOTDEV")

        def on(self, event):
            def register(fn):
                self._handlers[event] = fn
                return fn
            return register

        async def connect(self, url, token, options=None):
            record["connect"] = (url, token, options.auto_subscribe)

        async def disconnect(self):
            record["disconnected"] = True

        def fire(self, event, *args):
            self._handlers[event](*args)

    return types.SimpleNamespace(
        Room=FakeRoom,
        AudioStream=FakeAudioStream,
        TrackKind=types.SimpleNamespace(KIND_AUDIO=1, KIND_VIDEO=2),
        RoomOptions=lambda auto_subscribe=True: types.SimpleNamespace(
            auto_subscribe=auto_subscribe),
    )


class TestTrackSubscription:
    async def _connected(self, monkeypatch, record):
        import tools.lazy_deps
        from plugins.platforms.matrix.rtc.receiver import MatrixRTCReceiver

        monkeypatch.setattr(tools.lazy_deps, "ensure", lambda feature, prompt=True: None)
        monkeypatch.setitem(
            sys.modules, "livekit", types.SimpleNamespace(rtc=_fake_livekit(record)))

        async def on_transcript(identity, transcript):
            pass

        rx = MatrixRTCReceiver(on_transcript)
        await rx.connect("wss://sfu", "j.w.t")
        return rx

    async def _fire_and_drain(self, rx, track, identity):
        rx._room.fire("track_subscribed", track, None,
                      types.SimpleNamespace(identity=identity))
        await asyncio.gather(*list(rx._tasks), return_exceptions=True)

    @pytest.mark.asyncio
    async def test_audio_is_requested_at_whisper_rate_not_the_sdk_default(self, monkeypatch):
        """The wire is 48 kHz; asking the FFI for 16 kHz is what keeps ffmpeg out."""
        record = {}
        rx = await self._connected(monkeypatch, record)
        try:
            await self._fire_and_drain(rx, _FakeTrack(kind=1, frames=[pcm(0.1)]), "@a:hs:A")
            assert record["asked_for"] == (RATE, CHANNELS)
        finally:
            await rx.close()

    @pytest.mark.asyncio
    async def test_subscribed_pcm_lands_in_the_segmenter_under_the_speaker(self, monkeypatch):
        record = {}
        rx = await self._connected(monkeypatch, record)
        try:
            await self._fire_and_drain(
                rx, _FakeTrack(kind=1, frames=[pcm(0.4), pcm(0.4)]), "@alice:hs:AAA")
            released = rx.segmenter.check_silence(now=1e9)
            assert [identity for identity, _ in released] == ["@alice:hs:AAA"]
            assert len(released[0][1]) == len(pcm(0.8)), "both frames, in order"
            assert record["stream_closed"], "the stream must close when the track ends"
        finally:
            await rx.close()

    @pytest.mark.asyncio
    async def test_a_video_track_never_opens_an_audio_stream(self, monkeypatch):
        record = {}
        rx = await self._connected(monkeypatch, record)
        try:
            await self._fire_and_drain(rx, _FakeTrack(kind=2, frames=[pcm(1.0)]), "@a:hs:A")
            assert "asked_for" not in record
            assert not rx.segmenter.check_silence(now=1e9)
        finally:
            await rx.close()

    @pytest.mark.asyncio
    async def test_connect_subscribes_automatically(self, monkeypatch):
        """Nothing calls set_subscribed(), so auto_subscribe is the only path to audio."""
        record = {}
        rx = await self._connected(monkeypatch, record)
        try:
            assert record["connect"] == ("wss://sfu", "j.w.t", True)
        finally:
            await rx.close()
