"""A Group Chat the gateway took over keeps answering, and a stored Send is never reported as failed.

#145960: rooms handed from the standalone backend to the gateway (``hosted_rooms_gateway_import``)
refused every Send with ``invalid_params`` after storing the message, and no member was woken.
#143239: any failure after the user event committed came back to the client as a failed Send.
"""

from __future__ import annotations

import sqlite3
import threading

import pytest
from pathlib import Path
from types import SimpleNamespace

from gateway import hosted_room_driver, hosted_rooms
from gateway.hosted_rooms_gateway_import import handoff_gateway_id, import_standalone_rooms
from tests.waiting import wait_until
from gateway.session_hosted_controls import HostedControls
from tui_gateway.hosted_room_service import HostedRoomService

MEMBERS = [{"member_id": "default", "profile": "default", "handle": "hermes"},
           {"member_id": "ops", "profile": "ops", "handle": "ops"}]


class _RPC:
    def __init__(self) -> None:
        self.sessions, self.submits = {}, []

    def resolve_exact(self, *, profile, title, source):
        return self.sessions.get((profile, title))

    def create(self, *, profile, title, source):
        self.sessions[(profile, title)] = session = {"session_id": f"{profile}-session", "title": title}
        return session

    def resume(self, *, profile, session_id, source):
        return {"session_id": session_id}

    def submit(self, **kwargs):
        self.submits.append(kwargs["profile"])
        kwargs["on_terminal"]({"status": "settled", "text": f"reply from {kwargs['profile']}"})
        return {"accepted": True}

    def history(self, **kwargs):
        return []

    def info(self, **kwargs):
        return {"active": False, "task_id": None}


class _GatewayRouteService(HostedControls, HostedRoomService):
    """The gateway route's control overrides (CanonicalHostedRoomService) over the shared service."""


def _service(db_path: Path, cls: type[HostedRoomService] = HostedRoomService) -> tuple[HostedRoomService, _RPC]:
    service = cls(
        SimpleNamespace(_methods={}, _sessions={}, _sessions_lock=threading.Lock()), db_path=db_path)
    service.rpc = service.runtime.rpc = rpc = _RPC()
    service.local_profiles = lambda: ("default", "ops")
    return service, rpc


def _kinds(db: Path, room_id: str) -> list[str]:
    return [e["kind"] for e in hosted_rooms.read_events(db, room_id=room_id, since_seq=0)["events"]]


def _authority(db: Path, room_id: str) -> tuple[str, int]:
    with sqlite3.connect(db) as conn:
        return conn.execute("SELECT authority_gateway_id, authority_epoch FROM hosted_rooms WHERE room_id=?",
                            (room_id,)).fetchone()


def test_a_room_handed_to_the_gateway_takes_sends_and_wakes_its_members(tmp_path):
    source, target = tmp_path / "shared-state.db", tmp_path / "state.db"
    with sqlite3.connect(target) as conn:  # the gateway's SessionDB table the owner claim lives in
        conn.execute("CREATE TABLE state_meta (key TEXT PRIMARY KEY, value TEXT)")
    local = hosted_rooms.local_authority_gateway_id()
    hosted_rooms.create_room(source, room_id="room-desk", name="Desk", members=MEMBERS,
                             authority_gateway_id=local)
    assert import_standalone_rooms(target) == 1
    # The stores every room the shipped import handed over is in (#145960's report): nothing to migrate.
    assert _authority(source, "room-desk") == (handoff_gateway_id(), 2)
    assert _authority(target, "room-desk") == (local, 3)

    service, rpc = _service(target)
    sent = service.send(room_id="room-desk", event_id="user-1",
                        payload={"text": "@ops inspect", "thread_id": "thread-1"})
    assert sent["kind"] == "message.user" and sent["authority_epoch"] == 3
    # #145977: a Desktop that journaled the refused Send retries the exact same intent; it is
    # acknowledged as the stored message (so the client retires it), never stored twice.
    again = service.send(room_id="room-desk", event_id="user-1",
                         payload={"text": "@ops inspect", "thread_id": "thread-1"})
    assert again["seq"] == sent["seq"] and _kinds(target, "room-desk").count("message.user") == 1
    assert len(hosted_room_driver.list_tasks(target, room_id="room-desk", status="queued")) == 1

    service.start()
    try:
        wait_until(lambda: "message.member" in _kinds(target, "room-desk"), timeout=5, what="member reply")
    finally:
        assert service.stop(timeout=5.0)
    assert rpc.submits == ["ops"]
    assert _kinds(target, "room-desk").count("message.member") == 1


def test_a_send_already_stored_is_answered_as_sent_when_preparing_the_room_fails(tmp_path):
    db = tmp_path / "state.db"
    service, _ = _service(db)
    service.create_room(room_id="room-1", name="Room", members=MEMBERS)

    def storage_full(binding):
        raise hosted_rooms.HostedRoomError("Group Chat storage is full on this host.")

    service.prepare_room = storage_full
    sent = service.send(room_id="room-1", event_id="user-1", payload={"text": "hello", "thread_id": "t"})
    service.bindings = lambda: ()  # the binding list has not caught up with the room yet
    again = service.send(room_id="room-1", event_id="user-2", payload={"text": "again", "thread_id": "t"})

    assert [sent["kind"], again["kind"]] == ["message.user", "message.user"]
    assert _kinds(db, "room-1").count("message.user") == 2


@pytest.mark.parametrize("route", [HostedRoomService, _GatewayRouteService], ids=["tui_gateway", "gateway"])
def test_a_room_whose_members_cannot_be_woken_says_so_once_and_stops_retrying(tmp_path, route):
    db = tmp_path / "state.db"
    service, _ = _service(db, route)
    service.create_room(room_id="room-1", name="Room", members=MEMBERS)
    service.local_profiles = lambda: ("default", "dev", "ops")
    service.create_room(room_id="room-2", name="Other", members=[
        MEMBERS[0], {"member_id": "dev", "profile": "dev", "handle": "dev"}])
    service.local_profiles = lambda: ("default", "dev")  # the ops member's profile was deleted
    attempts: list[str] = []
    real_prepare = service.prepare_room

    def counted(binding):
        attempts.append(binding.room_id)
        return real_prepare(binding)

    service.prepare_room = counted
    sent = service.send(room_id="room-1", event_id="user-1", payload={"text": "@ops hi", "thread_id": "t"})
    assert sent["kind"] == "message.user"

    service.runtime.poll_interval_seconds = 0.02
    service.start()
    try:
        wait_until(lambda: "member.unavailable" in _kinds(db, "room-1"), timeout=5, what="stall notice")
        settled = attempts.count("room-1")
        wait_until(lambda: service.runtime.status()["cycles"] > 30, timeout=5, what="more driver passes")
    finally:
        assert service.stop(timeout=5.0)
    assert attempts.count("room-1") == settled == 3  # bounded: no retries after the stall
    events = hosted_rooms.read_events(db, room_id="room-1", since_seq=0)["events"]
    notices = [e for e in events if e["kind"] == "member.unavailable"]
    assert len(notices) == 1 and "ops" in notices[0]["payload"]["error"]
    status = service.status("room-1")
    assert "ops" in status["last_error"] and service.status("room-2")["last_error"] is None
    [retry] = [a for a in status["pending_actions"] if a["task_id"].startswith("prepare:")]

    # Retry while the profile is still missing fails loudly and adds no second notice.
    try:
        service.retry_room_task("room-1", task_id=retry["task_id"], member_id=retry["member_id"],
                                execution_generation=retry["execution_generation"])
        raise AssertionError("retry of a still-broken room must fail")
    except Exception as exc:
        assert "ops" in str(exc)
    assert _kinds(db, "room-1").count("member.unavailable") == 1

    service.local_profiles = lambda: ("default", "dev", "ops")  # restored
    receipt = service.retry_room_task("room-1", task_id=retry["task_id"], member_id=retry["member_id"],
                                      execution_generation=1)
    assert receipt["identity"].task_id == retry["task_id"]
    assert service.status("room-1")["last_error"] is None
    assert not [a for a in service.status("room-1")["pending_actions"] if a["task_id"].startswith("prepare:")]
    assert len(hosted_room_driver.list_tasks(db, room_id="room-1", status="queued")) == 1
