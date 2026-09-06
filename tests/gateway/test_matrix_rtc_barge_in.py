"""Phase 5 behaviour contracts: the bot stops hearing itself, and stops talking when told.

Synthetic PCM only — nothing here joins an SFU, opens a socket or needs the LiveKit SDK.
Two behaviours earn the file:

* **Echo.** Audio arriving while the publisher is playing is the bot's own reply coming
  back. Buffering it means transcribing the reply as if the user had said it, and the bot
  answers itself for the rest of the call.
* **Barge-in.** Speech that keeps arriving *through* that gate is the user cutting the
  reply off, and has to reach the gateway's own barge-in seam — not just mute the track,
  which would leave the model generating a reply nobody will hear.

The level floor is what separates the two, so the quiet-frames case is tested as hard as
the loud one: a stream that delivers frames continuously (silence included, which is what a
decoded WebRTC sink does) must not read as a permanent interruption.
"""

import asyncio
import types

import pytest

from gateway.config import Platform
from gateway.platforms.base import AudioFormat, StreamingTTSHandle
from gateway.run_inbound import GatewayInboundMixin
from gateway.run_voice import GatewayVoiceMixin
from gateway.session import SessionSource
from plugins.platforms.matrix.rtc import join as jn
from plugins.platforms.matrix.rtc import outbound as ob
from plugins.platforms.matrix.rtc import receiver as rcv
from plugins.platforms.matrix.rtc import segmenter as seg
from plugins.platforms.matrix.rtc.join import MatrixCall, MatrixRTCVoiceMixin
from plugins.platforms.matrix.rtc.receiver import MatrixRTCReceiver
from plugins.platforms.matrix.rtc.session import MatrixRTCSessions

ROOM = "!voice:hs.tld"
OWNER, ALICE, MALLORY = "@owner:hs.tld", "@alice:hs.tld", "@mallory:hs.tld"
ALICE_ID, MALLORY_ID = f"{ALICE}:DEVICEAAA", f"{MALLORY}:DEVICEZZZ"
RATE = seg.SAMPLE_RATE


def loud(seconds: float) -> bytes:
    """*seconds* of s16 audio well above the barge-in floor (RMS 4096)."""
    return b"\x00\x10" * int(RATE * seconds)


def quiet(seconds: float) -> bytes:
    """*seconds* of digital silence — what a continuously-delivered stream sends between
    utterances, and what must never be mistaken for someone talking over the bot."""
    return b"\x00\x00" * int(RATE * seconds)


# --------------------------------------------------------------------------- level


class TestPcmRms:
    def test_silence_reads_as_zero(self):
        assert seg.pcm_rms(quiet(0.1)) == 0.0

    def test_speech_reads_far_above_the_barge_in_floor(self):
        assert seg.pcm_rms(loud(0.1)) > rcv.BARGE_IN_RMS

    def test_an_empty_frame_is_not_a_division_by_zero(self):
        assert seg.pcm_rms(b"") == 0.0

    def test_a_trailing_half_sample_is_dropped_not_fatal(self):
        """s16 is two bytes; an odd-length buffer must not raise on the audio path."""
        assert seg.pcm_rms(loud(0.01) + b"\x7f") > 0

    def test_the_scale_is_the_sample_value_not_the_byte_count(self):
        assert seg.pcm_rms(b"\x00\x10" * 50) == pytest.approx(4096.0)


class TestPcmDuration:
    def test_duration_is_measured_at_the_declared_rate(self):
        assert seg.pcm_duration(loud(0.5)) == pytest.approx(0.5)
        assert seg.pcm_duration(loud(0.5), sample_rate=48000) == pytest.approx(0.5 / 3)


# --------------------------------------------------------------------------- publisher


class _FakeSource:
    def __init__(self, sample_rate, num_channels):
        self.sample_rate, self.num_channels = sample_rate, num_channels
        self.cleared = 0

    async def capture_frame(self, frame):
        pass

    async def wait_for_playout(self):
        pass

    def clear_queue(self):
        self.cleared += 1

    async def aclose(self):
        pass


def _fake_rtc():
    return types.SimpleNamespace(
        AudioSource=_FakeSource,
        AudioResampler=lambda *a, **kw: None,
        AudioFrame=lambda data, rate, ch, samples: types.SimpleNamespace(data=data),
        LocalAudioTrack=types.SimpleNamespace(
            create_audio_track=lambda name, source: types.SimpleNamespace(name=name)),
        TrackPublishOptions=lambda source=None: types.SimpleNamespace(source=source),
        TrackSource=types.SimpleNamespace(SOURCE_MICROPHONE="mic"))


class _FakeParticipant:
    async def publish_track(self, track, options):
        return types.SimpleNamespace(sid="TR_abc")

    async def unpublish_track(self, sid):
        pass


class _FakeLiveKitRoom:
    def __init__(self, participants=None):
        self.local_participant = _FakeParticipant()
        self.remote_participants = participants or {}


@pytest.fixture
def livekit(monkeypatch):
    """Make ``from livekit import rtc`` resolve to the fakes, and skip the FFI drain sleep."""
    import sys

    rtc = _fake_rtc()
    monkeypatch.setitem(sys.modules, "livekit", types.SimpleNamespace(rtc=rtc))
    monkeypatch.setitem(sys.modules, "livekit.rtc", rtc)

    real_sleep = asyncio.sleep
    monkeypatch.setattr(
        "plugins.platforms.matrix.rtc.publisher.asyncio.sleep", lambda d: real_sleep(0))
    return rtc


async def speaking_publisher(**kw):
    """A publisher at the wire rate (no resampler) that has already written a frame."""
    from plugins.platforms.matrix.rtc.publisher import LIVEKIT_SAMPLE_RATE, MatrixRTCPublisher

    publisher = MatrixRTCPublisher(_FakeLiveKitRoom(), sample_rate=LIVEKIT_SAMPLE_RATE, **kw)
    await publisher.start()
    await publisher.write(loud(0.1))
    return publisher


class TestSpeakingWindow:
    @pytest.mark.asyncio
    async def test_a_publisher_that_has_said_nothing_is_not_speaking(self, livekit):
        from plugins.platforms.matrix.rtc.publisher import MatrixRTCPublisher

        publisher = MatrixRTCPublisher(_FakeLiveKitRoom())
        assert publisher.speaking is False
        await publisher.start()
        assert publisher.speaking is False, "publishing a track is not the same as talking"

    @pytest.mark.asyncio
    async def test_writing_audio_opens_the_window(self, livekit):
        assert (await speaking_publisher()).speaking is True

    @pytest.mark.asyncio
    async def test_playing_the_queue_out_closes_it(self, livekit):
        publisher = await speaking_publisher()
        await publisher.drain()
        assert publisher.speaking is False, "the reply finished; the room is ours to listen to"

    @pytest.mark.asyncio
    async def test_dropping_the_queue_closes_it_immediately(self, livekit):
        """Barge-in's own path: after clear() the user is talking, not us."""
        publisher = await speaking_publisher()
        publisher.clear()
        assert publisher.speaking is False

    @pytest.mark.asyncio
    async def test_a_closed_publisher_is_never_speaking(self, livekit):
        publisher = await speaking_publisher()
        await publisher.close()
        assert publisher.speaking is False


class TestAdapterSpeakingProbe:
    class _Adapter(ob.MatrixRTCOutboundMixin):
        pass

    def test_a_room_without_a_call_is_never_speaking(self):
        assert self._Adapter().is_speaking_in(ROOM) is False

    @pytest.mark.asyncio
    async def test_the_probe_reports_that_rooms_own_publisher(self, livekit):
        adapter = self._Adapter()
        publisher = await adapter.start_rtc_audio(
            ROOM, _FakeLiveKitRoom(), sample_rate=48000)
        assert adapter.is_speaking_in(ROOM) is False

        await publisher.write(loud(0.1))
        assert adapter.is_speaking_in(ROOM) is True
        assert adapter.is_speaking_in("!other:hs.tld") is False

    @pytest.mark.asyncio
    async def test_an_interrupted_stream_hands_the_room_back(self, livekit):
        """``finish_streaming_tts(interrupted=True)`` is what the aborting turn calls."""
        adapter = self._Adapter()
        await adapter.start_rtc_audio(ROOM, _FakeLiveKitRoom(), sample_rate=48000)
        handle = await adapter.begin_streaming_tts(
            ROOM, AudioFormat(sample_rate=48000))
        await adapter.write_streaming_tts(handle, loud(0.1))
        assert adapter.is_speaking_in(ROOM) is True

        await adapter.finish_streaming_tts(handle, interrupted=True)
        assert adapter.is_speaking_in(ROOM) is False

    @pytest.mark.asyncio
    async def test_aborting_the_stream_hands_the_room_back(self, livekit):
        adapter = self._Adapter()
        await adapter.start_rtc_audio(ROOM, _FakeLiveKitRoom(), sample_rate=48000)
        handle = await adapter.begin_streaming_tts(ROOM, AudioFormat(sample_rate=48000))
        await adapter.write_streaming_tts(handle, loud(0.1))

        await adapter.abort_streaming_tts(handle, error="barge-in")
        assert adapter.is_speaking_in(ROOM) is False


# --------------------------------------------------------------------------- receiver


class _FakeTrack:
    def __init__(self, *frames):
        self.frames = list(frames)


class _FakeAudioStream:
    def __init__(self, track, sample_rate=48000, num_channels=1):
        self._frames = list(track.frames)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._frames:
            raise StopAsyncIteration
        return types.SimpleNamespace(frame=types.SimpleNamespace(data=self._frames.pop(0)))

    async def aclose(self):
        pass


_FAKE_RTC = types.SimpleNamespace(AudioStream=_FakeAudioStream)


async def drain(receiver, *frames, identity: str = ALICE_ID) -> None:
    """Push *frames* through the receiver's real track loop.

    Straight at ``_drain_track`` rather than through ``connect()``: the gate under test is
    per-frame, and a fake room adds nothing but ceremony to it.
    """
    await receiver._drain_track(_FAKE_RTC, _FakeTrack(*frames), identity)


def receiver(speaking=False, barge_ins=None, **kw) -> MatrixRTCReceiver:
    """A receiver whose speaking state is a plain flag the test flips."""
    async def on_transcript(identity, transcript):
        pass

    async def on_barge_in(identity):
        if barge_ins is not None:
            barge_ins.append(identity)

    state = {"speaking": speaking}
    rx = MatrixRTCReceiver(
        on_transcript, is_speaking=lambda: state["speaking"],
        on_barge_in=on_barge_in if barge_ins is not None else None, **kw)
    rx.state = state
    return rx


class TestEchoSuppression:
    @pytest.mark.asyncio
    async def test_audio_heard_while_we_speak_is_never_buffered(self):
        """Self-playback in the segmenter becomes a transcript, and the bot answers itself."""
        rx = receiver(speaking=True)
        await drain(rx, loud(0.6), loud(0.6))
        assert rx.segmenter.check_silence(now=1e9) == []

    @pytest.mark.asyncio
    async def test_audio_heard_while_we_are_quiet_still_reaches_the_segmenter(self):
        rx = receiver(speaking=False)
        await drain(rx, loud(0.6), loud(0.6))
        released = rx.segmenter.check_silence(now=1e9)
        assert [identity for identity, _ in released] == [ALICE_ID]

    @pytest.mark.asyncio
    async def test_the_gate_opens_again_the_moment_the_reply_ends(self):
        rx = receiver(speaking=True)
        await drain(rx, loud(0.6))
        rx.state["speaking"] = False
        await drain(rx, loud(0.6))

        released = rx.segmenter.check_silence(now=1e9)
        assert len(released) == 1
        assert len(released[0][1]) == len(loud(0.6)), "only what was heard after the reply"

    @pytest.mark.asyncio
    async def test_a_receiver_with_no_gate_transcribes_everything(self):
        """Phases 1-4 built the receiver without the callbacks; that must still work."""
        async def on_transcript(identity, transcript):
            pass

        rx = MatrixRTCReceiver(on_transcript)
        await drain(rx, loud(0.6), loud(0.6))
        assert rx.segmenter.check_silence(now=1e9) != []


class TestBargeIn:
    @pytest.mark.asyncio
    async def test_sustained_speech_over_the_reply_interrupts(self):
        heard = []
        await drain(receiver(speaking=True, barge_ins=heard), loud(0.4))
        assert heard == [ALICE_ID]

    @pytest.mark.asyncio
    async def test_a_burst_shorter_than_the_threshold_does_not(self):
        """A door, a cough, one keystroke: not worth cutting a reply in half."""
        heard = []
        await drain(receiver(speaking=True, barge_ins=heard), loud(0.2))
        assert heard == []

    @pytest.mark.asyncio
    async def test_it_fires_once_however_long_the_speech_runs(self):
        heard = []
        await drain(receiver(speaking=True, barge_ins=heard), *([loud(0.4)] * 5))
        assert heard == [ALICE_ID], "one interruption per reply, not one per frame"

    @pytest.mark.asyncio
    async def test_silence_delivered_through_the_gate_never_interrupts(self):
        """A decoded WebRTC sink keeps producing frames between utterances. Counting those
        as speech would cut every reply short 300 ms in, forever."""
        heard = []
        await drain(receiver(speaking=True, barge_ins=heard), *([quiet(1.0)] * 5))
        assert heard == []

    @pytest.mark.asyncio
    async def test_a_quiet_frame_restarts_the_count(self):
        """Two sub-threshold bursts either side of a gap are not one interruption."""
        heard = []
        await drain(receiver(speaking=True, barge_ins=heard),
                    loud(0.2), quiet(0.1), loud(0.2))
        assert heard == []

    @pytest.mark.asyncio
    async def test_the_next_reply_can_be_interrupted_too(self):
        heard = []
        rx = receiver(speaking=True, barge_ins=heard)
        await drain(rx, loud(0.4))
        rx.state["speaking"] = False
        await drain(rx, loud(0.4))  # heard normally: this one is transcribed, not an interrupt
        rx.state["speaking"] = True
        await drain(rx, loud(0.4))
        assert heard == [ALICE_ID, ALICE_ID]

    @pytest.mark.asyncio
    async def test_nothing_is_transcribed_on_the_way_to_an_interruption(self):
        heard = []
        rx = receiver(speaking=True, barge_ins=heard)
        await drain(rx, loud(1.0))
        assert heard == [ALICE_ID]
        assert rx.segmenter.check_silence(now=1e9) == [], "the interrupting frames are ours"

    @pytest.mark.asyncio
    async def test_a_failing_callback_does_not_end_the_track(self):
        async def boom(identity):
            raise RuntimeError("runner is gone")

        async def on_transcript(identity, transcript):
            pass

        rx = MatrixRTCReceiver(on_transcript, is_speaking=lambda: True, on_barge_in=boom)
        await drain(rx, loud(0.4))  # must not raise

    @pytest.mark.asyncio
    async def test_without_a_barge_in_callback_the_frames_are_still_dropped(self):
        rx = receiver(speaking=True)
        await drain(rx, loud(1.0))
        assert rx.segmenter.check_silence(now=1e9) == []


class TestBargeInTuning:
    def test_the_knobs_come_from_config_yaml(self, monkeypatch):
        monkeypatch.setattr(
            rcv, "_rtc_config", lambda: {"barge_in_duration": 1.0, "barge_in_rms": 50})
        rx = receiver()
        assert (rx.barge_in_duration, rx.barge_in_rms) == (1.0, 50)

    @pytest.mark.asyncio
    async def test_a_configured_threshold_is_the_one_in_force(self, monkeypatch):
        monkeypatch.setattr(rcv, "_rtc_config", lambda: {"barge_in_duration": 1.0})
        heard = []
        rx = receiver(speaking=True, barge_ins=heard)
        await drain(rx, loud(0.5))
        assert heard == []
        await drain(rx, loud(0.6))
        assert heard == [ALICE_ID]

    @pytest.mark.parametrize("bad", [0, -1, "loud", None])
    def test_unusable_values_fall_back_to_the_defaults(self, bad, monkeypatch):
        monkeypatch.setattr(
            rcv, "_rtc_config",
            lambda: {"barge_in_duration": bad, "barge_in_rms": bad})
        rx = receiver()
        assert rx.barge_in_duration == rcv.BARGE_IN_DURATION
        assert rx.barge_in_rms == rcv.BARGE_IN_RMS


# --------------------------------------------------------------------------- session


class _Runner(GatewayInboundMixin, GatewayVoiceMixin):
    """The production mixins, subclassed only to pin config and the authorization verdict."""

    def __init__(self, echo: bool = True, authorized: bool = True):
        self.config = types.SimpleNamespace(stt_echo_transcripts=echo)
        self.authorized = authorized

    def _is_user_authorized(self, source, **_kw) -> bool:
        return self.authorized


class _FakeAdapter:
    """Only the adapter surface ``session.py`` reaches for."""

    def __init__(self, runner=None, allowed_users=(ALICE,)):
        self.gateway_runner = runner
        self._allowed_rooms = set()
        self._allowed_user_ids = set(allowed_users)
        self.interrupted, self.sent, self.handled = [], [], []

    def _is_authorized_user(self, user_id: str) -> bool:
        return user_id in self._allowed_user_ids

    def _event_session_key(self, event) -> str:
        """Shaped like the real one in the only way that matters here: it is derived from
        ``event.source``, and a group session is per speaker."""
        return f"matrix:{event.source.chat_id}:{event.source.user_id}"

    async def interrupt_session_activity(self, session_key, chat_id, metadata=None) -> None:
        self.interrupted.append((session_key, chat_id))

    async def send(self, chat_id, text, metadata=None) -> None:
        self.sent.append((chat_id, text))

    async def _get_display_name(self, room_id, user_id) -> str:
        return user_id

    async def handle_message(self, event) -> None:
        self.handled.append(event)


def room_source(**kw) -> SessionSource:
    """The room's own session source — owned by someone other than the speaker."""
    return SessionSource(
        platform=Platform.MATRIX, chat_id=ROOM, chat_name="Voice Room", chat_type="group",
        user_id=OWNER, user_name="Owner", **kw)


def bound(adapter=None, **runner_kw):
    adapter = adapter or _FakeAdapter(runner=_Runner(**runner_kw))
    sessions = MatrixRTCSessions(adapter)
    sessions.bind(ROOM, room_source())
    return sessions, adapter


class TestSessionBargeIn:
    @pytest.mark.asyncio
    async def test_it_interrupts_the_session_the_speaker_is_talking_into(self):
        sessions, adapter = bound()
        await sessions.barge_in(ROOM, ALICE_ID)
        assert adapter.interrupted == [(f"matrix:{ROOM}:{ALICE}", ROOM)], (
            "the key must carry the speaker, not the source the room was bound with")

    @pytest.mark.asyncio
    async def test_an_unauthorized_speaker_cannot_cut_off_someone_elses_turn(self):
        sessions, adapter = bound(_FakeAdapter(runner=_Runner(authorized=False)))
        await sessions.barge_in(ROOM, MALLORY_ID)
        assert adapter.interrupted == []

    @pytest.mark.asyncio
    async def test_a_room_with_no_bound_session_has_no_turn_to_interrupt(self):
        adapter = _FakeAdapter(runner=_Runner())
        await MatrixRTCSessions(adapter).barge_in(ROOM, ALICE_ID)
        assert adapter.interrupted == []

    @pytest.mark.asyncio
    async def test_interrupting_twice_is_harmless(self):
        """The guard is an Event: the second set() is a no-op, so this must not raise."""
        sessions, adapter = bound()
        await sessions.barge_in(ROOM, ALICE_ID)
        await sessions.barge_in(ROOM, ALICE_ID)
        assert len(adapter.interrupted) == 2


class TestTranscriptEcho:
    @pytest.mark.asyncio
    async def test_what_we_heard_is_posted_back_into_the_room(self):
        sessions, adapter = bound()
        await sessions.on_transcript(ROOM, ALICE_ID, "turn the lights on")
        assert adapter.sent == [(ROOM, '🎙️ "turn the lights on"')]
        assert len(adapter.handled) == 1, "the echo does not replace the turn"

    @pytest.mark.asyncio
    async def test_nothing_is_echoed_when_the_operator_turned_it_off(self):
        sessions, adapter = bound(echo=False)
        await sessions.on_transcript(ROOM, ALICE_ID, "turn the lights on")
        assert adapter.sent == []
        assert len(adapter.handled) == 1

    @pytest.mark.asyncio
    async def test_a_dropped_transcript_is_never_echoed(self):
        sessions, adapter = bound(_FakeAdapter(runner=_Runner(authorized=False)))
        await sessions.on_transcript(ROOM, MALLORY_ID, "delete everything")
        assert (adapter.sent, adapter.handled) == ([], [])

    @pytest.mark.asyncio
    async def test_a_runner_without_the_helper_simply_does_not_echo(self):
        """The adapter can be driven standalone; that must not lose the turn."""
        sessions, adapter = bound(_FakeAdapter(runner=None))
        await sessions.on_transcript(ROOM, ALICE_ID, "still here")
        assert len(adapter.handled) == 1


# --------------------------------------------------------------------------- join wiring


class _FakeReceiver:
    """Records what ``join_voice_channel`` wired it with."""

    instances: list = []

    def __init__(self, on_transcript, is_authorized=None, is_speaking=None, on_barge_in=None):
        self.on_transcript, self.is_authorized = on_transcript, is_authorized
        self.is_speaking, self.on_barge_in = is_speaking, on_barge_in
        self.room = _FakeLiveKitRoom()
        _FakeReceiver.instances.append(self)

    async def connect(self, sfu_url, jwt):
        pass

    async def close(self):
        pass


class _FakePublisher:
    def __init__(self, room, sample_rate=None, channels=1):
        self.room, self.sample_rate, self.channels = room, sample_rate, channels
        self.live = self.speaking = False

    async def start(self):
        self.live = True

    async def close(self):
        self.live = self.speaking = False


class _JoinAdapter(MatrixRTCVoiceMixin, ob.MatrixRTCOutboundMixin, _FakeAdapter):
    """The real mixins over the little of ``MatrixAdapter`` they reach for."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self._homeserver, self._user_id = "https://hs.tld", "@hermes:hs.tld"
        self._access_token, self._device_id = "not-a-real-token", "CONFIGURED"
        self._client = self._room_identities = None


@pytest.fixture
def rtc(monkeypatch):
    _FakeReceiver.instances = []

    async def fake_credentials(*a, **kw):
        return "wss://sfu.hs.tld", "jwt-token", "https://call.hs.tld/livekit/jwt"

    monkeypatch.setattr(jn, "MatrixRTCReceiver", _FakeReceiver)
    monkeypatch.setattr(jn, "fetch_livekit_credentials", fake_credentials)
    monkeypatch.setattr(ob, "MatrixRTCPublisher", _FakePublisher)


async def joined() -> tuple:
    adapter = _JoinAdapter(runner=_Runner())
    adapter.bind_voice_session(ROOM, room_source())
    await adapter.join_voice_channel(MatrixCall(room_id=ROOM, name="Voice Room"))
    return adapter, _FakeReceiver.instances[-1]


class TestJoinWiring:
    @pytest.mark.asyncio
    async def test_the_receiver_watches_this_rooms_publisher(self, rtc):
        adapter, rx = await joined()
        assert rx.is_speaking() is False

        adapter.rtc_publishers[ROOM].speaking = True
        assert rx.is_speaking() is True, "the ears must see the mouth of the same room"

    @pytest.mark.asyncio
    async def test_barge_in_from_the_receiver_reaches_the_rooms_session(self, rtc):
        adapter, rx = await joined()
        await rx.on_barge_in(ALICE_ID)
        assert adapter.interrupted == [(f"matrix:{ROOM}:{ALICE}", ROOM)]

    @pytest.mark.asyncio
    async def test_leaving_leaves_nothing_speaking_behind(self, rtc):
        adapter, _ = await joined()
        adapter.rtc_publishers[ROOM].speaking = True
        await adapter.leave_voice_channel(ROOM)
        assert adapter.is_speaking_in(ROOM) is False


# --------------------------------------------------------------------- streaming handle


class TestAbortIsIdempotent:
    """The gateway aborts on barge-in and may abort again as the turn unwinds."""

    class _Adapter(ob.MatrixRTCOutboundMixin):
        pass

    @pytest.mark.asyncio
    async def test_a_second_abort_after_barge_in_is_harmless(self, livekit):
        adapter = self._Adapter()
        publisher = await adapter.start_rtc_audio(
            ROOM, _FakeLiveKitRoom(), sample_rate=48000)
        handle = await adapter.begin_streaming_tts(ROOM, AudioFormat(sample_rate=48000))
        await adapter.write_streaming_tts(handle, loud(0.1))

        await adapter.abort_streaming_tts(handle, error="barge-in")
        await adapter.abort_streaming_tts(handle, error="barge-in")
        await adapter.finish_streaming_tts(handle, interrupted=True)

        assert isinstance(handle, StreamingTTSHandle) and handle.aborted
        assert publisher._source.cleared == 1, "only the first abort had a queue to drop"
        assert adapter.is_speaking_in(ROOM) is False
