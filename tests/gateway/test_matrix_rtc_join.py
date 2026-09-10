"""Phase 4 behaviour contracts: ``/voice join`` puts the bot in a MatrixRTC call.

Fake state events, a fake receiver and a fake publisher — nothing here imports the LiveKit
SDK, opens a socket or talks to a homeserver. The gateway half runs the *real*
``GatewayVoiceMixin`` methods against a chat-scoped adapter, because the point of the phase
is that those methods stopped being Discord-only.
"""

from urllib.parse import quote

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run_voice import GatewayVoiceMixin
from gateway.session import SessionSource
from plugins.platforms.matrix.rtc import join as jn
from plugins.platforms.matrix.rtc import outbound as ob
from plugins.platforms.matrix.rtc.join import (
    CALL_MEMBER_TYPE, MatrixCall, MatrixRTCVoiceMixin, call_membership_content,
    live_call_members, membership_user_id)

ROOM = "!voice:hs.tld"
ALICE, BOT = "@alice:hs.tld", "@hermes:hs.tld"
ALICE_ID, MALLORY_ID = f"{ALICE}:DEVICEAAA", "@mallory:hs.tld:DEVICEZZZ"
NOW_MS = 1_757_000_000_000
FOCUS_URL = "https://call.hs.tld/livekit/jwt"

# Verbatim off a live Element Desktop 1.12.27 call, 2026-09-06. The state key is the
# percent-decoded path segment Synapse logged; the content is the MSC3401 membership that
# came with it, with the focus URL blanked (no tokens, no internal hostnames beyond the
# user id the bug is about).
ADMIN = "@admin:matrix.myhome.internal"
ELEMENT_KEY = f"_{ADMIN}_EMNHZTXVIO_m.call"
ELEMENT_CONTENT = {
    "application": "m.call",
    "call_id": "",
    "device_id": "EMNHZTXVIO",
    "expires": 14_400_000,
    "focus_active": {"type": "livekit", "focus_selection": "oldest_membership"},
    "foci_preferred": [{"type": "livekit", "livekit_alias": ROOM,
                        "livekit_service_url": "https://livekit.example"}],
    "scope": "m.room",
}


def rtc_member(user_id: str = ALICE, device: str = "DEVICEAAA", *,
               event_type: str = "m.rtc.member", content=None, state_key=None) -> dict:
    """One RTC membership state event in the flattened per-device shape."""
    return {
        "type": event_type,
        "state_key": f"_{user_id}_{device}" if state_key is None else state_key,
        "content": {"application": "m.call", "call_id": "", "device_id": device}
        if content is None else content,
    }


# --------------------------------------------------------------------------- fakes


class _FakeReceiver:
    """``MatrixRTCReceiver``'s surface, minus the SDK. Records what it was wired with."""

    instances: list = []

    def __init__(self, on_transcript, is_authorized=None, **kw):
        self.on_transcript, self.is_authorized = on_transcript, is_authorized
        self.connected, self.closed, self.flush_on_close = None, False, None
        self.room = _FakeLiveKitRoom()
        _FakeReceiver.instances.append(self)

    async def connect(self, sfu_url, jwt):
        self.connected = (sfu_url, jwt)

    async def close(self):
        self.closed = True
        if self.flush_on_close is not None:  # the utterance still in the segmenter
            await self.on_transcript(*self.flush_on_close)


class _FakeApi:
    """``client.api``: records raw requests. RTC membership is an unmodelled event type, so
    the mixin has to reach for the raw API and so does its double."""

    def __init__(self, fail: bool = False):
        self.calls, self.fail = [], fail

    async def request(self, method, path, content=None, **kw):
        self.calls.append((str(method), path, content))
        if self.fail:
            raise RuntimeError("M_FORBIDDEN: you don't have permission to post that event")
        return {"event_id": "$evt"}


class _FakeClient:
    def __init__(self, device_id="DEVICEBOT", api=None):
        self.device_id, self.api = device_id, api or _FakeApi()


class _FakeParticipant:
    def __init__(self, name):
        self.name = name


class _FakeLiveKitRoom:
    def __init__(self, participants=None):
        self.remote_participants = participants or {}


class _FakePublisher:
    """``MatrixRTCPublisher``'s lifecycle only; Phase 3 owns the audio contracts."""

    fail_on_start = False

    def __init__(self, room, sample_rate=None, channels=1):
        self.room, self.sample_rate, self.channels = room, sample_rate, channels
        self.live, self.closed = False, False

    async def start(self):
        if _FakePublisher.fail_on_start:
            raise RuntimeError("SFU refused the track")
        self.live = True

    async def close(self):
        self.live, self.closed = False, True


class _Adapter(MatrixRTCVoiceMixin, ob.MatrixRTCOutboundMixin):
    """The real mixins over the little of ``MatrixAdapter`` they reach for."""

    def __init__(self, state=(), allowed_users=(ALICE,)):
        self._homeserver, self._user_id = "https://hs.tld", BOT
        self._access_token, self._device_id = "not-a-real-token", "CONFIGURED"
        self._client = None
        self._allowed_rooms, self._allowed_user_ids = set(), set(allowed_users)
        self._room_identities = {}
        self.gateway_runner = None
        self.state, self.handled = list(state), []

    async def _fetch_room_state(self, room_id):
        return self.state

    async def _resolve_room_identity(self, room_id):
        return type("_Identity", (), {"display_name": "Voice Room"})()

    def _is_authorized_user(self, user_id):
        return user_id in self._allowed_user_ids

    async def _get_display_name(self, room_id, user_id):
        return user_id

    async def handle_message(self, event):
        self.handled.append(event)


def room_source(**kw) -> SessionSource:
    return SessionSource(
        platform=Platform.MATRIX, chat_id=ROOM, chat_name="Voice Room", chat_type="group",
        user_id=ALICE, user_name="Alice", **kw)


def voice_event(text: str = "/voice join", source=None) -> MessageEvent:
    return MessageEvent(text=text, message_type=MessageType.TEXT, source=source or room_source())


@pytest.fixture
def rtc(monkeypatch):
    """Swap the receiver, the publisher and the JWT exchange for fakes."""
    _FakeReceiver.instances = []
    _FakePublisher.fail_on_start = False
    credentials = {"calls": []}

    async def fake_credentials(homeserver, user_id, access_token, room_id, device_id, **kw):
        credentials["calls"].append(
            {"homeserver": homeserver, "user_id": user_id, "room_id": room_id,
             "device_id": device_id, "session": kw.get("session")})
        return "wss://sfu.hs.tld", "jwt-token", FOCUS_URL

    monkeypatch.setattr(jn, "MatrixRTCReceiver", _FakeReceiver)
    monkeypatch.setattr(jn, "fetch_livekit_credentials", fake_credentials)
    monkeypatch.setattr(ob, "MatrixRTCPublisher", _FakePublisher)
    return credentials


async def joined(adapter=None, source=None, **kw) -> _Adapter:
    """An adapter already in ROOM's call, bound to the room's session."""
    adapter = adapter or _Adapter(**kw)
    adapter.bind_voice_session(ROOM, source or room_source())
    await adapter.join_voice_channel(MatrixCall(room_id=ROOM, name="Voice Room"))
    return adapter


# --------------------------------------------------------------------------- state keys


class TestMembershipUserId:
    def test_the_per_device_state_key_yields_the_bare_user_id(self):
        assert membership_user_id(f"_{ALICE}_DEVICEAAA") == ALICE

    def test_the_same_key_without_the_leading_underscore_also_works(self):
        assert membership_user_id(f"{ALICE}_DEVICEAAA") == ALICE

    def test_a_plain_user_id_state_key_survives_whole(self):
        assert membership_user_id(ALICE) == ALICE

    def test_an_underscore_in_the_localpart_is_not_mistaken_for_a_device_suffix(self):
        assert membership_user_id("@my_bot:hs.tld") == "@my_bot:hs.tld"

    def test_that_localpart_still_gives_up_a_real_device_suffix(self):
        assert membership_user_id("@my_bot:hs.tld_DEVICEAAA") == "@my_bot:hs.tld"

    def test_element_appends_the_application_after_the_device(self):
        """The key Element Desktop 1.12.27 actually writes, verbatim off a live call:
        ``PUT .../state/org.matrix.msc3401.call.member/_%40admin%3A..._EMNHZTXVIO_m.call``.
        Two suffixes, not one — cutting only the last leaves the device on the user id."""
        assert membership_user_id(ELEMENT_KEY) == ADMIN

    def test_the_application_suffix_survives_an_underscore_in_the_localpart(self):
        assert membership_user_id("@my_bot:hs.tld_DEVICEAAA_m.call") == "@my_bot:hs.tld"


# --------------------------------------------------------------------------- memberships


class TestLiveCallMembers:
    def test_a_membership_with_no_expiry_stated_counts_as_live(self):
        """Refusing a call that is plainly running, over a field we guessed at, is worse."""
        assert live_call_members([rtc_member()], NOW_MS) == {ALICE}

    def test_leaving_is_published_as_empty_content_not_a_redaction(self):
        assert live_call_members([rtc_member(content={})], NOW_MS) == set()

    def test_the_msc3401_event_type_counts_too(self):
        events = [rtc_member(event_type="org.matrix.msc3401.call.member")]
        assert live_call_members(events, NOW_MS) == {ALICE}

    def test_a_legacy_memberships_list_is_read_entry_by_entry(self):
        content = {"memberships": [{"call_id": "", "expires_ts": NOW_MS + 60_000}]}
        assert live_call_members([rtc_member(content=content)], NOW_MS) == {ALICE}

    def test_an_empty_memberships_list_is_nobody(self):
        assert live_call_members([rtc_member(content={"memberships": []})], NOW_MS) == set()

    def test_an_absolute_expiry_in_the_past_is_not_live(self):
        content = {"call_id": "", "expires_ts": NOW_MS - 1}
        assert live_call_members([rtc_member(content=content)], NOW_MS) == set()

    def test_a_relative_expiry_is_measured_from_created_ts(self):
        stale = {"call_id": "", "created_ts": NOW_MS - 20_000, "expires": 10_000}
        fresh = {"call_id": "", "created_ts": NOW_MS - 5_000, "expires": 10_000}
        assert live_call_members([rtc_member(content=stale)], NOW_MS) == set()
        assert live_call_members([rtc_member(content=fresh)], NOW_MS) == {ALICE}

    def test_ordinary_room_state_is_not_mistaken_for_a_call(self):
        join_event = {"type": "m.room.member", "state_key": ALICE,
                      "content": {"membership": "join"}}
        assert live_call_members([join_event], NOW_MS) == set()

    def test_every_participant_is_reported_once_across_their_devices(self):
        events = [rtc_member(device="DEVICEAAA"), rtc_member(device="DEVICEBBB"),
                  rtc_member(user_id=BOT)]
        assert live_call_members(events, NOW_MS) == {ALICE, BOT}

    def test_a_live_element_call_membership_is_read_off_the_wire_shape(self):
        """Both halves of the production report at once: Element's three-part state key and
        the MSC3401 content it ships with it. This is the call ``/voice join`` said nobody
        was in."""
        event = {"type": "org.matrix.msc3401.call.member", "state_key": ELEMENT_KEY,
                 "content": ELEMENT_CONTENT}
        assert live_call_members([event], NOW_MS) == {ADMIN}


# --------------------------------------------------------------------------- join / leave


class TestGetUserVoiceChannel:
    @pytest.mark.asyncio
    async def test_a_user_in_the_rooms_call_gets_that_call_back(self):
        call = await _Adapter([rtc_member()]).get_user_voice_channel(ROOM, ALICE)
        assert (call.room_id, call.name) == (ROOM, "Voice Room")

    @pytest.mark.asyncio
    async def test_a_user_who_has_not_started_a_call_gets_nothing(self):
        adapter = _Adapter([rtc_member(user_id=BOT)])
        assert await adapter.get_user_voice_channel(ROOM, ALICE) is None

    @pytest.mark.asyncio
    async def test_a_room_with_no_rtc_state_at_all_is_not_a_call(self):
        assert await _Adapter([]).get_user_voice_channel(ROOM, ALICE) is None


class TestJoin:
    @pytest.mark.asyncio
    async def test_joining_hears_the_room_and_speaks_into_the_same_connection(self, rtc):
        adapter = await joined()

        receiver, = _FakeReceiver.instances
        assert receiver.connected == ("wss://sfu.hs.tld", "jwt-token")
        assert adapter.rtc_publishers[ROOM].room is receiver.room, "one call, one membership"
        assert adapter.is_in_voice_channel(ROOM)

    @pytest.mark.asyncio
    async def test_the_jwt_is_minted_for_the_room_and_the_clients_own_device(self, rtc):
        adapter = _Adapter()
        adapter._client = type("_C", (), {"device_id": "RESOLVED", "api": None})()
        await joined(adapter)

        call, = rtc["calls"]
        assert (call["room_id"], call["user_id"]) == (ROOM, BOT)
        assert call["device_id"] == "RESOLVED", "the token's real device, not the configured one"

    @pytest.mark.asyncio
    async def test_transcripts_from_the_call_land_on_the_rooms_own_session(self, rtc):
        adapter = await joined()

        receiver, = _FakeReceiver.instances
        await receiver.on_transcript(ALICE_ID, "hello there")

        event, = adapter.handled
        assert (event.text, event.message_type) == ("hello there", MessageType.VOICE)
        assert (event.source.chat_id, event.source.user_id) == (ROOM, ALICE)

    @pytest.mark.asyncio
    async def test_the_receiver_can_check_the_speaker_before_transcribing(self, rtc):
        await joined(allowed_users=(ALICE,))

        receiver, = _FakeReceiver.instances
        assert receiver.is_authorized(ALICE_ID) is True
        assert receiver.is_authorized(MALLORY_ID) is False

    @pytest.mark.asyncio
    async def test_joining_a_call_we_are_already_in_reuses_the_connection(self, rtc):
        adapter = await joined()

        assert await adapter.join_voice_channel(MatrixCall(ROOM, "Voice Room")) is True
        assert len(_FakeReceiver.instances) == 1
        assert len(rtc["calls"]) == 1

    @pytest.mark.asyncio
    async def test_a_failed_join_does_not_leave_the_room_bound(self, rtc, monkeypatch):
        async def boom(*a, **kw):
            raise RuntimeError("/sfu/get returned HTTP 403")

        monkeypatch.setattr(jn, "fetch_livekit_credentials", boom)
        adapter = _Adapter()
        adapter.bind_voice_session(ROOM, room_source())

        with pytest.raises(RuntimeError):
            await adapter.join_voice_channel(MatrixCall(ROOM, "Voice Room"))
        assert adapter.rtc_sessions.source_for(ROOM, ALICE) is None
        assert ROOM not in adapter.rtc_receivers

    @pytest.mark.asyncio
    async def test_a_call_we_can_hear_but_not_speak_into_is_still_a_call(self, rtc):
        """The outbound half is the one with a fallback: play_tts sends a voice message."""
        _FakePublisher.fail_on_start = True
        adapter = await joined()

        assert ROOM in adapter.rtc_receivers
        assert adapter.is_in_voice_channel(ROOM) is False


class TestLeave:
    @pytest.mark.asyncio
    async def test_leaving_stops_the_track_the_receiver_and_the_binding(self, rtc):
        adapter = await joined()
        publisher, receiver = adapter.rtc_publishers[ROOM], _FakeReceiver.instances[0]

        await adapter.leave_voice_channel(ROOM)

        assert (publisher.closed, receiver.closed) == (True, True)
        assert adapter.rtc_receivers == {} and adapter.rtc_publishers == {}
        assert adapter.rtc_sessions.source_for(ROOM, ALICE) is None

    @pytest.mark.asyncio
    async def test_the_utterance_flushed_on_close_still_reaches_a_session(self, rtc):
        """Unbinding before close() would throw away the last thing the user said."""
        adapter = await joined()
        _FakeReceiver.instances[0].flush_on_close = (ALICE_ID, "one last thing")

        await adapter.leave_voice_channel(ROOM)

        assert [e.text for e in adapter.handled] == ["one last thing"]

    @pytest.mark.asyncio
    async def test_leaving_a_room_we_never_joined_is_not_an_error(self, rtc):
        await _Adapter().leave_voice_channel(ROOM)

    @pytest.mark.asyncio
    async def test_leaving_after_a_restart_still_clears_the_membership(self, rtc):
        """A restart takes the publisher, the receiver and the LiveKit session with it and
        leaves the membership state event behind, so leave has to clear that event from an
        adapter holding none of the in-memory half. Skip it and Element keeps a muted ghost
        of the bot in the call that nothing can ever evict."""
        adapter = await joined(with_api())
        adapter.rtc_publishers.clear()  # what the new process wakes up holding
        adapter.rtc_receivers.clear()
        adapter.rtc_sessions.unbind(ROOM)
        assert adapter.is_in_voice_channel(ROOM) is False

        await adapter.leave_voice_channel(ROOM)

        join_call, leave_call = adapter._client.api.calls
        assert leave_call == ("PUT", join_call[1], {})


class TestVoiceChannelInfo:
    @pytest.mark.asyncio
    async def test_the_call_reports_its_participants_without_their_devices(self, rtc):
        adapter = await joined()
        _FakeReceiver.instances[0].room.remote_participants = {
            ALICE_ID: _FakeParticipant("Alice")}

        info = adapter.get_voice_channel_info(ROOM)

        assert info["member_count"] == 1
        assert info["members"][0]["user_id"] == ALICE
        assert info["members"][0]["display_name"] == "Alice"

    def test_a_room_with_no_call_reports_nothing(self):
        assert _Adapter().get_voice_channel_info(ROOM) is None


# ------------------------------------------------------------------ publishing our own


def with_api(adapter=None, fail=False, device="DEVICEBOT") -> _Adapter:
    """An adapter whose homeserver requests can be inspected."""
    adapter = adapter or _Adapter()
    adapter._client = _FakeClient(device, _FakeApi(fail=fail))
    return adapter


def state_path(state_key: str) -> str:
    return (f"/_matrix/client/v3/rooms/{quote(ROOM, safe='')}"
            f"/state/{CALL_MEMBER_TYPE}/{quote(state_key, safe='')}")


class TestCallMembershipContent:
    """The SFU is not the call UI. ``/sfu/get`` checks neither membership nor the room, so
    the bot can be audible to everyone and still absent from Element's widget — which is
    exactly what happened live. This state event is the only thing that closes that gap."""

    def test_the_content_carries_the_keys_element_actually_publishes(self):
        """Against the captured event, not a guess: matrix-js-sdk drops a membership that
        is missing any of application / call_id / device_id / focus_active / foci_preferred,
        and a dropped membership is an invisible bot."""
        content = call_membership_content(ROOM, "DEVICEBOT", FOCUS_URL)

        assert set(content) == set(ELEMENT_CONTENT)
        assert (content["application"], content["scope"]) == ("m.call", "m.room")
        assert content["call_id"] == "", "the room's own call, not a named one"
        assert content["device_id"] == "DEVICEBOT"
        assert content["focus_active"] == ELEMENT_CONTENT["focus_active"]
        assert content["foci_preferred"] == [
            {"type": "livekit", "livekit_alias": ROOM, "livekit_service_url": FOCUS_URL}]

    def test_the_membership_we_publish_reads_back_as_live(self):
        """The round trip that matters: another Hermes asking who is on this call has to
        see us, so the writer and ``live_call_members`` cannot drift apart."""
        event = {"type": CALL_MEMBER_TYPE, "state_key": f"_{BOT}_DEVICEBOT_m.call",
                 "content": call_membership_content(ROOM, "DEVICEBOT", FOCUS_URL)}
        assert live_call_members([event], NOW_MS) == {BOT}


class TestCallMembershipPublishing:
    @pytest.mark.asyncio
    async def test_joining_publishes_the_membership_under_our_own_device(self, rtc):
        adapter = await joined(with_api())

        (method, path, content), = adapter._client.api.calls
        assert method == "PUT"
        assert path == state_path(f"_{BOT}_DEVICEBOT_m.call")
        assert content == call_membership_content(ROOM, "DEVICEBOT", FOCUS_URL)

    @pytest.mark.asyncio
    async def test_the_state_key_is_the_shape_element_writes(self, rtc):
        """``_@user:hs_DEVICE_m.call`` — and our own parser has to survive the round trip."""
        adapter = await joined(with_api())

        state_key = quote(f"_{BOT}_DEVICEBOT_m.call", safe="")
        assert adapter._client.api.calls[0][1].endswith(state_key)
        assert membership_user_id(f"_{BOT}_DEVICEBOT_m.call") == BOT

    @pytest.mark.asyncio
    async def test_leaving_clears_the_membership_with_empty_content(self, rtc):
        """Leaving a call is published as empty content, never a redaction."""
        adapter = await joined(with_api())
        await adapter.leave_voice_channel(ROOM)

        join_call, leave_call = adapter._client.api.calls
        assert leave_call[2] == {}
        assert leave_call[1] == join_call[1], "a different state key leaves a ghost behind"

    @pytest.mark.asyncio
    async def test_a_rejected_membership_does_not_take_the_call_down(self, rtc):
        """The audio is already up. A widget listing is not worth hanging up over."""
        adapter = with_api(fail=True)
        assert await adapter.join_voice_channel(MatrixCall(ROOM, "Voice Room")) is True
        assert ROOM in adapter.rtc_receivers and adapter.is_in_voice_channel(ROOM)

        await adapter.leave_voice_channel(ROOM)
        assert adapter.rtc_receivers == {}

    @pytest.mark.asyncio
    async def test_an_adapter_with_no_client_still_joins(self, rtc):
        """``_client`` is None on object.__new__ instances and in most of this file."""
        adapter = await joined()
        assert adapter.is_in_voice_channel(ROOM)


# --------------------------------------------------------------------------- gateway


class _Runner(GatewayVoiceMixin):
    """The production mixin over the two lookups the runner would provide."""

    def __init__(self, adapter, tmp_path):
        self.adapter, self.adapters = adapter, {Platform.MATRIX: adapter}
        self._voice_mode = {}
        self._VOICE_MODE_PATH = tmp_path / "voice_mode.json"

    def _adapter_for_source(self, source):
        return self.adapter

    def _adapter_profile_for_source(self, source):
        return None


class TestGatewayScope:
    def test_a_chat_scoped_adapter_is_keyed_by_its_chat_not_a_guild(self, tmp_path):
        runner = _Runner(_Adapter(), tmp_path)
        assert runner._voice_scope_id(runner.adapter, voice_event()) == ROOM

    def test_a_guild_scoped_adapter_still_resolves_the_guild_off_the_raw_message(self, tmp_path):
        from types import SimpleNamespace
        runner = _Runner(object(), tmp_path)
        event = voice_event()
        event.raw_message = SimpleNamespace(guild_id=111, guild=None)
        assert runner._voice_scope_id(runner.adapter, event) == 111


class TestGatewayJoin:
    @pytest.mark.asyncio
    async def test_voice_join_puts_a_matrix_adapter_in_the_rooms_call(self, rtc, tmp_path):
        adapter = _Adapter([rtc_member()])
        runner = _Runner(adapter, tmp_path)

        reply = await runner._handle_voice_channel_join(voice_event())

        assert "Voice Room" in reply
        assert adapter.is_in_voice_channel(ROOM)
        assert runner._voice_mode[f"matrix:{ROOM}"] == "all", "replies are spoken after a join"

    @pytest.mark.asyncio
    async def test_the_call_is_bound_to_the_live_session_source(self, rtc, tmp_path):
        """Not a to_dict() round-trip: that drops the transport ref authorization reads."""
        adapter = _Adapter([rtc_member()])
        runner = _Runner(adapter, tmp_path)
        event = voice_event()

        await runner._handle_voice_channel_join(event)

        assert adapter.rtc_sessions._sources[ROOM] is event.source

    @pytest.mark.asyncio
    async def test_a_user_not_in_the_call_is_told_to_start_one(self, rtc, tmp_path):
        runner = _Runner(_Adapter([]), tmp_path)

        reply = await runner._handle_voice_channel_join(voice_event())

        assert "voice channel first" in reply.lower()
        assert not _FakeReceiver.instances

    @pytest.mark.asyncio
    async def test_voice_leave_hangs_up(self, rtc, tmp_path):
        adapter = await joined(_Adapter([rtc_member()]))
        runner = _Runner(adapter, tmp_path)

        reply = await runner._handle_voice_channel_leave(voice_event("/voice leave"))

        assert "left" in reply.lower()
        assert adapter.rtc_receivers == {}
        assert runner._voice_mode[f"matrix:{ROOM}"] == "off"

    @pytest.mark.asyncio
    async def test_voice_leave_clears_the_call_ui_after_a_gateway_restart(self, rtc, tmp_path):
        """The live bug. ``is_in_voice_channel`` is False on a fresh process, so the guard
        answered "Not in a voice channel." and the one thing a restart *cannot* clean up by
        itself — the membership the room is still advertising — was never cleared."""
        adapter = await joined(with_api(_Adapter([rtc_member()])))
        adapter.rtc_publishers.clear()
        adapter.rtc_receivers.clear()
        runner = _Runner(adapter, tmp_path)

        reply = await runner._handle_voice_channel_leave(voice_event("/voice leave"))

        assert "left" in reply.lower()
        assert adapter._client.api.calls[-1] == (
            "PUT", state_path(f"_{BOT}_DEVICEBOT_m.call"), {})


class TestGatewayPlayback:
    @pytest.mark.asyncio
    async def test_a_spoken_reply_goes_into_the_call_not_out_as_a_file(self, rtc, tmp_path):
        adapter = await joined(_Adapter([rtc_member()]))
        runner = _Runner(adapter, tmp_path)
        played = []

        async def play(room_id, path):
            played.append((room_id, path))
            return True

        adapter.play_in_voice_channel = play
        adapter.send_voice = None  # falling back here would be a bug, not a degradation

        await runner._deliver_voice_reply(voice_event(), ["/tmp/reply.ogg"])

        assert played == [(ROOM, "/tmp/reply.ogg")]
