"""Phase 2 behaviour contracts: a MatrixRTC transcript becomes a session message.

Fake transcripts only — nothing here joins an SFU, opens a socket, or needs the LiveKit
SDK. The dedupe tests run the gateway's *real* ``GatewayVoiceMixin`` method with Matrix
ids, which is the point: it is reused, not reimplemented.
"""

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageType
from gateway.run_voice import GatewayVoiceMixin
from gateway.session import SessionSource
from plugins.platforms.matrix.rtc.session import MatrixRTCSessions, split_identity

ROOM = "!voice:hs.tld"
ALICE, ALICE_ID = "@alice:hs.tld", "@alice:hs.tld:DEVICEAAA"
MALLORY_ID = "@mallory:hs.tld:DEVICEZZZ"


class _Runner(GatewayVoiceMixin):
    """The production mixin, subclassed only to pin the authorization verdict."""

    def __init__(self, authorized: bool = True):
        self.authorized = authorized

    def _is_user_authorized(self, source, **_kw) -> bool:
        return self.authorized


class _FakeAdapter:
    """Only the adapter surface ``session.py`` actually reaches for."""

    def __init__(self, runner=None, allowed_rooms=(), allowed_users=(ALICE,)):
        self.gateway_runner = runner
        self._allowed_rooms = set(allowed_rooms)
        self._allowed_user_ids = set(allowed_users)
        self.display_names: dict[str, str] = {}
        self.handled = []

    def _is_authorized_user(self, user_id: str) -> bool:
        return user_id in self._allowed_user_ids

    async def _get_display_name(self, room_id: str, user_id: str) -> str:
        return self.display_names[user_id]  # a miss raises, exercising the fallback

    async def handle_message(self, event) -> None:
        self.handled.append(event)


def room_source(chat_type: str = "group", **kw) -> SessionSource:
    """The source the room's typed messages already use."""
    return SessionSource(
        platform=Platform.MATRIX, chat_id=ROOM, chat_name="Voice Room", chat_type=chat_type,
        user_id="@owner:hs.tld", user_name="Owner", **kw)


def bound(adapter=None, source=None, **adapter_kw):
    """A ``MatrixRTCSessions`` with ROOM already bound; returns ``(sessions, adapter)``."""
    adapter = adapter or _FakeAdapter(runner=_Runner(), **adapter_kw)
    sessions = MatrixRTCSessions(adapter)
    sessions.bind(ROOM, source or room_source())
    return sessions, adapter


# --------------------------------------------------------------------------- identity


class TestSplitIdentity:
    def test_the_device_suffix_is_stripped_off_the_matrix_user_id(self):
        """The gateway session is keyed on the user; two devices are one speaker."""
        assert split_identity(ALICE_ID) == (ALICE, "DEVICEAAA")

    def test_a_user_id_carrying_no_device_survives_whole(self):
        assert split_identity(ALICE) == (ALICE, "")

    def test_a_homeserver_port_stays_on_the_user_id(self):
        assert split_identity("@alice:hs.tld:8448:DEVICEAAA") == ("@alice:hs.tld:8448", "DEVICEAAA")

    def test_a_non_matrix_identity_is_not_torn_apart(self):
        assert split_identity("some-sfu-bot") == ("some-sfu-bot", "")


# --------------------------------------------------------------------------- binding


class TestRoomBinding:
    @pytest.mark.asyncio
    async def test_audio_in_an_unbound_room_reaches_nobody(self):
        adapter = _FakeAdapter(runner=_Runner())
        sessions = MatrixRTCSessions(adapter)
        await sessions.on_transcript(ROOM, ALICE_ID, "hello?")
        assert adapter.handled == []

    @pytest.mark.asyncio
    async def test_a_transcript_becomes_a_voice_message_on_the_bound_session(self):
        sessions, adapter = bound()
        await sessions.on_transcript(ROOM, ALICE_ID, "what time is it")

        assert len(adapter.handled) == 1
        event = adapter.handled[0]
        assert event.text == "what time is it"
        assert event.message_type == MessageType.VOICE
        assert event.source.chat_id == ROOM
        # Top-level sender fields mirror source.*; downstream prompt code reads them.
        assert event.user_id == event.source.user_id == ALICE

    @pytest.mark.asyncio
    async def test_the_voice_turn_lands_in_the_room_session_not_a_sibling(self):
        """The whole reason to bind the room's own source: thread and profile carry over,
        so speaking continues the conversation the room was already having."""
        sessions, adapter = bound(source=room_source(thread_id="$root", profile="work"))
        await sessions.on_transcript(ROOM, ALICE_ID, "carry on")

        source = adapter.handled[0].source
        assert (source.thread_id, source.profile) == ("$root", "work")

    @pytest.mark.asyncio
    async def test_the_speaker_is_stamped_without_mutating_the_bind(self):
        """Two people in one call must not overwrite each other's identity."""
        original = room_source()
        sessions, adapter = bound(source=original)
        await sessions.on_transcript(ROOM, ALICE_ID, "first")
        await sessions.on_transcript(ROOM, "@bob:hs.tld:DEVICEBBB", "second")

        assert [e.source.user_id for e in adapter.handled] == [ALICE, "@bob:hs.tld"]
        assert original.user_id == "@owner:hs.tld", "the bound source must stay untouched"

    @pytest.mark.asyncio
    async def test_audio_after_unbind_is_dropped_not_guessed_at(self):
        sessions, adapter = bound()
        sessions.unbind(ROOM)
        await sessions.on_transcript(ROOM, ALICE_ID, "still there?")
        assert adapter.handled == []

    def test_unbinding_a_room_that_was_never_bound_is_not_an_error(self):
        MatrixRTCSessions(_FakeAdapter()).unbind(ROOM)


# --------------------------------------------------------------------------- authorization


class TestAuthorization:
    @pytest.mark.asyncio
    async def test_an_unauthorized_speaker_never_reaches_handle_message(self):
        sessions, adapter = bound(adapter=_FakeAdapter(runner=_Runner(authorized=False)))
        await sessions.on_transcript(ROOM, MALLORY_ID, "delete everything")
        assert adapter.handled == []

    def test_the_allowlist_verdict_is_available_before_transcription(self):
        """Done-when #2: the check must be answerable from the identity alone, so an
        unauthorized participant's audio never reaches Whisper."""
        sessions, _ = bound()
        assert sessions.is_authorized(ROOM, ALICE_ID) is True

        denied, _ = bound(adapter=_FakeAdapter(runner=_Runner(authorized=False)))
        assert denied.is_authorized(ROOM, MALLORY_ID) is False

    def test_an_unbound_room_is_never_authorized(self):
        sessions = MatrixRTCSessions(_FakeAdapter(runner=_Runner()))
        assert sessions.is_authorized(ROOM, ALICE_ID) is False

    @pytest.mark.asyncio
    async def test_a_room_outside_the_room_allowlist_is_silenced(self):
        sessions, adapter = bound(allowed_rooms={"!other:hs.tld"})
        await sessions.on_transcript(ROOM, ALICE_ID, "hello")
        assert adapter.handled == []

    @pytest.mark.asyncio
    async def test_a_dm_is_exempt_from_the_room_allowlist(self):
        """Matches the text path: a project-scoped MATRIX_ALLOWED_ROOMS must not silence
        the operator's own DM."""
        sessions, adapter = bound(source=room_source(chat_type="dm"),
                                  allowed_rooms={"!other:hs.tld"})
        await sessions.on_transcript(ROOM, ALICE_ID, "hello")
        assert len(adapter.handled) == 1

    @pytest.mark.asyncio
    async def test_without_a_runner_the_adapter_allowlist_still_gates(self):
        """No runner is no excuse for an open door."""
        sessions, adapter = bound(adapter=_FakeAdapter(runner=None, allowed_users={ALICE}))
        await sessions.on_transcript(ROOM, MALLORY_ID, "let me in")
        assert adapter.handled == []

        await sessions.on_transcript(ROOM, ALICE_ID, "and me?")
        assert len(adapter.handled) == 1


# --------------------------------------------------------------------------- dedupe


class TestDuplicateSuppression:
    @pytest.mark.asyncio
    async def test_the_same_utterance_delivered_twice_runs_once(self):
        """STT can emit one utterance twice seconds apart -> two queued runs and two
        spoken replies. The gateway's suppressor takes Matrix ids unchanged."""
        sessions, adapter = bound()
        await sessions.on_transcript(ROOM, ALICE_ID, "what is on my calendar today")
        await sessions.on_transcript(ROOM, ALICE_ID, "what is on my calendar today")
        assert len(adapter.handled) == 1

    @pytest.mark.asyncio
    async def test_two_different_utterances_both_get_through(self):
        sessions, adapter = bound()
        await sessions.on_transcript(ROOM, ALICE_ID, "what is on my calendar today")
        await sessions.on_transcript(ROOM, ALICE_ID, "cancel the first meeting")
        assert len(adapter.handled) == 2

    @pytest.mark.asyncio
    async def test_two_speakers_saying_the_same_thing_are_not_confused(self):
        """The suppressor keys on (room, speaker), so agreeing is not deduplication."""
        sessions, adapter = bound()
        await sessions.on_transcript(ROOM, ALICE_ID, "yes please do that")
        await sessions.on_transcript(ROOM, "@bob:hs.tld:DEVICEBBB", "yes please do that")
        assert len(adapter.handled) == 2


# --------------------------------------------------------------------------- display name


class TestSpeakerName:
    @pytest.mark.asyncio
    async def test_the_room_display_name_labels_the_speaker(self):
        sessions, adapter = bound()
        adapter.display_names[ALICE] = "Alice"
        await sessions.on_transcript(ROOM, ALICE_ID, "hello")
        assert adapter.handled[0].user_name == adapter.handled[0].source.user_name == "Alice"

    @pytest.mark.asyncio
    async def test_an_unresolvable_name_costs_the_label_not_the_turn(self):
        sessions, adapter = bound()  # display_names empty -> _get_display_name raises
        await sessions.on_transcript(ROOM, ALICE_ID, "hello")
        assert len(adapter.handled) == 1
        assert adapter.handled[0].user_name == ALICE


# --------------------------------------------------------------------------- pre-STT gate


class TestReceiverAuthorizationHook:
    @staticmethod
    def _receiver(monkeypatch, **kw):
        """A receiver whose transcription is a spy, so "was Whisper reached" is assertable."""
        from plugins.platforms.matrix.rtc import receiver as rcv
        transcribed, delivered = [], []
        monkeypatch.setattr(
            rcv, "transcribe_pcm", lambda pcm, *a, **k: transcribed.append(pcm) or "heard")

        async def on_transcript(identity, transcript):
            delivered.append((identity, transcript))

        return rcv.MatrixRTCReceiver(on_transcript, **kw), transcribed, delivered

    @pytest.mark.asyncio
    async def test_rejected_audio_is_discarded_before_it_reaches_whisper(self, monkeypatch):
        receiver, transcribed, delivered = self._receiver(
            monkeypatch, is_authorized=lambda identity: identity == ALICE_ID)
        pcm = b"\x00\x01" * 8000
        await receiver._emit([(MALLORY_ID, pcm), (ALICE_ID, pcm)])

        assert transcribed == [pcm], "only the allowed speaker is transcribed"
        assert [identity for identity, _ in delivered] == [ALICE_ID]

    @pytest.mark.asyncio
    async def test_the_predicate_sees_the_raw_identity_with_its_device_suffix(self, monkeypatch):
        seen = []
        receiver, _, _ = self._receiver(
            monkeypatch, is_authorized=lambda i: bool(seen.append(i)) or True)
        await receiver._emit([(ALICE_ID, b"\x00\x01" * 8000)])
        assert seen == [ALICE_ID]

    @pytest.mark.asyncio
    async def test_without_a_predicate_every_speaker_is_still_transcribed(self, monkeypatch):
        """Phase 1's contract is unchanged for callers that do their own gating."""
        receiver, transcribed, delivered = self._receiver(monkeypatch)
        await receiver._emit([(MALLORY_ID, b"\x00\x01" * 8000)])
        assert len(transcribed) == 1 and len(delivered) == 1
