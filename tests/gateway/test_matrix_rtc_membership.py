"""MatrixRTC call membership: which state counts as being in the call, and how the bot
keeps its own membership valid.

The membership rules are matrix-js-sdk's ``CallMembership`` and
``checkSessionsMembershipData``. The lease tests drive the renewal loop through an
injected sleep and wall clock, so no test waits for real time to pass.
"""

import asyncio
import functools
from urllib.parse import quote

import pytest

from gateway.config import Platform
from gateway.session import SessionSource
from plugins.platforms.matrix.rtc import join as jn
from plugins.platforms.matrix.rtc import outbound as ob
from plugins.platforms.matrix.rtc.join import CALL_MEMBER_TYPE, MatrixCall, MatrixRTCVoiceMixin, live_call_members
from plugins.platforms.matrix.rtc.membership import CallMembershipLease
from tests.gateway.matrix_rtc_helpers import FOCUS_URL, SESSION, call_member_event, room_member_event

ROOM = "!voice:hs.tld"
ALICE, BOB, BOT = "@alice:hs.tld", "@bob:hs.tld", "@hermes:hs.tld"
NOW_MS = 1_757_000_000_000
HOUR_MS = 60 * 60 * 1000


def member_event(user=ALICE, device="DEVICEAAA", **kwargs):
    """A call membership stamped relative to the fixed test clock."""
    return call_member_event(user, device, now_ms=NOW_MS, **kwargs)


# --------------------------------------------------------------------------- reading


@pytest.mark.parametrize("event, expected", [
    pytest.param(member_event(), {ALICE}, id="session-membership"),
    pytest.param(member_event(age_ms=4 * HOUR_MS - 1), {ALICE}, id="default-expiry-not-reached"),
    pytest.param(member_event(age_ms=4 * HOUR_MS), set(), id="default-expiry-is-four-hours"),
    pytest.param(member_event(age_ms=20_000, expires=10_000), set(), id="expires-from-origin-ts"),
    pytest.param(member_event(age_ms=5 * HOUR_MS, created_ts=NOW_MS - 5_000, expires=10_000), {ALICE},
                 id="expires-from-created-ts"),
    pytest.param({**member_event(), "content": {}}, set(), id="left-with-empty-content"),
    pytest.param(member_event(focus_active=None), set(), id="no-focus-active"),
    pytest.param(member_event(call_id="breakout"), set(), id="not-the-room-call"),
    pytest.param(member_event(application="io.element.other"), set(), id="other-application"),
    pytest.param(member_event(sender=BOB), set(), id="sender-does-not-own-state-key"),
    pytest.param(member_event(event_type="m.rtc.member"), set(), id="rtc-member-is-not-room-state"),
    pytest.param({**member_event(), "content": {"memberships": [{**SESSION, "expires_ts": NOW_MS + 60_000}]}},
                 set(), id="pre-2024-memberships-list"),
])
def test_live_call_members_follow_the_matrix_js_sdk_session_rules(event, expected):
    assert live_call_members([event], NOW_MS) == expected


# --------------------------------------------------------------------------- lease


class _FakeClock:
    """A sleep that waits for the test to release it, and a wall clock that it advances."""

    def __init__(self):
        self.now_ms = NOW_MS
        self.sleeping: list[tuple[float, asyncio.Future]] = []

    def wall_ms(self) -> float:
        return self.now_ms

    async def sleep(self, seconds: float) -> None:
        future = asyncio.get_running_loop().create_future()
        self.sleeping.append((seconds, future))
        await future

    async def tick(self) -> float:
        """Let the next pending sleep finish, advancing the clock by its length."""
        for _ in range(100):
            if self.sleeping:
                break
            await asyncio.sleep(0)
        seconds, future = self.sleeping.pop(0)
        self.now_ms += seconds * 1000
        future.set_result(None)
        for _ in range(20):
            await asyncio.sleep(0)
        return seconds


class _Api:
    """``client.api``: records every raw request after the versions probe."""

    def __init__(self, delayed_events=True, fail_on=None):
        self.calls, self.delayed_events, self.fail_on = [], delayed_events, fail_on

    async def request(self, method, path, content=None, query_params=None, **kwargs):
        if str(method) == "GET":
            return {"unstable_features": {"org.matrix.msc4140": self.delayed_events}}
        self.calls.append((str(method), path, content, query_params))
        if self.fail_on is not None and self.fail_on in path:
            raise RuntimeError("M_NOT_FOUND")
        return {"delay_id": "delay1"} if query_params else {"event_id": "$event"}


class _Client:
    def __init__(self, api):
        self.device_id, self.api = "DEVICEBOT", api


class _Receiver:
    def __init__(self, on_transcript, **kwargs):
        self.room, self.closed = object(), False

    async def connect(self, sfu_url, jwt):
        pass

    async def close(self):
        self.closed = True


class _Publisher:
    def __init__(self, room, sample_rate=None, channels=1):
        self.live, self.channels = False, channels

    async def start(self):
        self.live = True

    async def close(self):
        self.live = False


class _Adapter(MatrixRTCVoiceMixin, ob.MatrixRTCOutboundMixin):
    def __init__(self, api, state):
        self._homeserver, self._user_id = "https://hs.tld", BOT
        self._access_token, self._device_id = "not-a-real-token", "CONFIGURED"
        self._client = _Client(api)
        self._allowed_rooms, self._allowed_user_ids = set(), {ALICE, BOB}
        self._room_identities = {}
        self.gateway_runner = None
        self.state = state

    async def _fetch_room_state(self, room_id):
        return self.state

    def _is_authorized_user(self, user_id):
        return user_id in self._allowed_user_ids


def source(user=ALICE):
    return SessionSource(platform=Platform.MATRIX, chat_id=ROOM, chat_name="Voice Room",
                         chat_type="group", user_id=user, user_name=user)


def state_path():
    key = quote(f"_{BOT}_DEVICEBOT_m.call", safe="")
    return f"/_matrix/client/v3/rooms/{quote(ROOM, safe='')}/state/{CALL_MEMBER_TYPE}/{key}"


def delayed_path(action):
    return f"/_matrix/client/unstable/org.matrix.msc4140/delayed_events/delay1/{action}"


@pytest.fixture
def call(monkeypatch):
    """A joined call whose lease runs on a fake clock. Returns ``(adapter, api, clock)``."""
    clock = _FakeClock()
    monkeypatch.setattr(jn, "MatrixRTCReceiver", _Receiver)
    monkeypatch.setattr(ob, "MatrixRTCPublisher", _Publisher)
    monkeypatch.setattr(jn, "CallMembershipLease",
                        functools.partial(CallMembershipLease, sleep=clock.sleep, wall_ms=clock.wall_ms))
    monkeypatch.setattr(jn, "leave_delay_ms", lambda: 8_000)

    async def credentials(*args, **kwargs):
        return "wss://sfu.hs.tld", "jwt-token", FOCUS_URL

    monkeypatch.setattr(jn, "fetch_livekit_credentials", credentials)

    async def join(api=None, state=None):
        api = api or _Api()
        state = state if state is not None else [call_member_event(ALICE), room_member_event(ALICE),
                                                 call_member_event(BOB, "DEVICEBBB"), room_member_event(BOB)]
        adapter = _Adapter(api, state)
        adapter.bind_voice_session(ROOM, source())
        await adapter.join_voice_channel(MatrixCall(ROOM, "Voice Room"))
        return adapter, api, clock

    return join


@pytest.mark.asyncio
async def test_the_delayed_leave_is_scheduled_before_the_membership_is_published(call):
    adapter, api, _ = await call()
    (schedule_method, schedule_path, schedule_content, schedule_query), (_, join_path, join_content, _) = api.calls
    assert ((schedule_method, schedule_path, schedule_content, schedule_query), join_path) == (
        ("PUT", state_path(), {}, {"org.matrix.msc4140.delay": 8_000}), state_path())
    assert join_content["device_id"] == "DEVICEBOT" and "created_ts" not in join_content
    await adapter.leave_voice_channel(ROOM)


@pytest.mark.asyncio
async def test_the_lease_restarts_the_delayed_leave_and_extends_expiry_from_the_first_join(call):
    adapter, api, clock = await call()
    del api.calls[:]
    await clock.tick()
    assert api.calls == [("POST", delayed_path("restart"), {}, None)]

    del api.calls[:]
    clock.now_ms = NOW_MS + 2 * HOUR_MS
    await clock.tick()
    restart, renewal = api.calls
    assert (restart, renewal[1], renewal[2]["created_ts"], renewal[2]["expires"]) == (
        ("POST", delayed_path("restart"), {}, None), state_path(), NOW_MS, 8 * HOUR_MS)
    await adapter.leave_voice_channel(ROOM)


@pytest.mark.asyncio
async def test_a_failed_restart_leaves_the_call_and_clears_the_membership(call):
    adapter, api, clock = await call(_Api(fail_on="/restart"))
    del api.calls[:]
    await clock.tick()
    assert (adapter.rtc_receivers, adapter.rtc_publishers, api.calls[-1][1:3]) == ({}, {}, (state_path(), {}))


@pytest.mark.asyncio
async def test_leaving_sends_the_delayed_leave_and_stops_renewing(call):
    adapter, api, clock = await call()
    del api.calls[:]
    await adapter.leave_voice_channel(ROOM)
    assert [path for _, path, _, _ in api.calls] == [delayed_path("send"), state_path()]
    assert all(future.cancelled() for _, future in clock.sleeping)


@pytest.mark.asyncio
async def test_the_bot_leaves_when_the_requester_hangs_up(call):
    adapter, api, clock = await call()
    adapter.update_rtc_call_state({"rooms": {"join": {ROOM: {"timeline": {"events": [
        {**call_member_event(ALICE), "content": {}}]}}}}})
    await clock.tick()
    assert (adapter.rtc_receivers, api.calls[-1][1:3]) == ({}, (state_path(), {}))


@pytest.mark.asyncio
async def test_a_second_voice_join_moves_the_live_call_to_the_new_requester(call):
    adapter, api, clock = await call()
    adapter.bind_voice_session(ROOM, source(BOB))
    assert await adapter.join_voice_channel(MatrixCall(ROOM, "Voice Room")) is True
    await clock.tick()
    assert (list(adapter.rtc_receivers), adapter.rtc_sessions.binding_for(ROOM).source.user_id) == ([ROOM], BOB)
    await adapter.leave_voice_channel(ROOM)


@pytest.mark.asyncio
async def test_a_call_in_an_encrypted_room_is_refused_before_any_media_connects(call):
    state = [call_member_event(ALICE), room_member_event(ALICE),
             {"type": "m.room.encryption", "state_key": "", "content": {"algorithm": "m.megolm.v1.aes-sha2"}}]
    with pytest.raises(Exception, match="encrypted rooms"):
        await call(state=state)
