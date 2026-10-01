"""A quarantined Group Chat can be ended on this gateway, after explicit confirmation, without
touching its history, its authority or any other gateway."""

import asyncio
import sqlite3
from types import SimpleNamespace

import pytest

from gateway import hosted_rooms as rooms
from hermes_state import SessionDB

USER = {"kind": "user", "id": "tek"}
MEMBERS = [{"kind": "bot", "id": "planner"}]
LOCAL = "install:" + "a" * 32
REMOTE = "install:" + "b" * 32


def _quarantined_room(db, *, gateway=LOCAL, room_id="room-1"):
    rooms.create_room(db, room_id=room_id, name="Field Room", members=MEMBERS, authority_gateway_id=gateway)
    for index in range(2):
        rooms.append_event(
            db, room_id=room_id, event_id=f"e{index}", kind="message.user", actor=USER,
            payload={"text": f"msg {index}"}, authority_gateway_id=gateway, authority_epoch=1)
    with sqlite3.connect(db) as conn:
        conn.execute("INSERT INTO hosted_room_quarantine VALUES (?, 'unsafe_authority_demotion', 1)", (room_id,))


def _snapshot(db, room_id="room-1"):
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        return {
            "room": dict(conn.execute("SELECT * FROM hosted_rooms WHERE room_id=?", (room_id,)).fetchone()),
            "events": [tuple(row) for row in conn.execute(
                "SELECT * FROM hosted_room_events WHERE room_id=? ORDER BY seq", (room_id,))],
            "quarantine": tuple(conn.execute(
                "SELECT * FROM hosted_room_quarantine WHERE room_id=?", (room_id,)).fetchone()),
            "reservation": tuple(conn.execute(
                "SELECT * FROM hosted_room_id_reservations WHERE room_id=?", (room_id,)).fetchone()),
            "budget": conn.execute("SELECT event_bytes FROM hosted_room_event_budget").fetchone()[0],
        }


def test_disband_of_a_quarantined_room_needs_confirmation(tmp_path):
    db = tmp_path / "state.db"
    _quarantined_room(db)
    before = _snapshot(db)
    for disband in (
        lambda: rooms.disband_room(db, room_id="room-1", expected_gateway_id=LOCAL, expected_epoch=1),
        lambda: rooms.disband_quarantined_room(db, room_id="room-1", confirmed=False),
    ):
        with pytest.raises(rooms.RoomQuarantinedError, match="confirm_quarantined set to true") as refused:
            disband()
        assert refused.value.reason == "room_authority_quarantined"
    assert _snapshot(db) == before


@pytest.mark.parametrize("authority", [LOCAL, REMOTE], ids=["promoted-here", "demoted-here"])
def test_confirmed_disband_only_tombstones_the_room_here(tmp_path, authority):
    db = tmp_path / "state.db"
    _quarantined_room(db, gateway=authority)
    before = _snapshot(db)
    assert rooms.disband_quarantined_room(db, room_id="room-1", confirmed=True, now=50) == {
        "room_id": "room-1", "disbanded_at": 50.0, "idempotent": False}
    after = _snapshot(db)
    # Same authority, history, quarantine evidence, reservation and byte budget: only the tombstone.
    changed = {"disbanded_at", "updated_at", "revision"}
    assert {k: v for k, v in after["room"].items() if k not in changed} == {
        k: v for k, v in before["room"].items() if k not in changed}
    assert after["room"]["disbanded_at"] == 50.0
    assert {k: v for k, v in after.items() if k != "room"} == {k: v for k, v in before.items() if k != "room"}
    assert rooms.list_rooms(db) == []
    listed, = rooms.list_rooms(db, include_disbanded=True)
    assert (listed["disbanded_at"], listed["safety_status"]) == (50.0, "authority_quarantined")
    replay = rooms.read_events(db, room_id="room-1", include_disbanded=True)
    assert [event["event_id"] for event in replay["events"]] == ["e0", "e1"]
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT confirmed_at FROM hosted_room_quarantine_disbands").fetchall() == [(50.0,)]
        assert conn.execute("SELECT retired_at FROM hosted_room_retired_ids").fetchall() == [(50.0,)]
    with pytest.raises(rooms.HostedRoomError):
        rooms.create_room(db, room_id="room-1", name="Reuse", members=MEMBERS, authority_gateway_id=LOCAL)


def test_confirmed_disband_is_idempotent(tmp_path):
    db = tmp_path / "state.db"
    _quarantined_room(db)
    first = rooms.disband_quarantined_room(db, room_id="room-1", confirmed=True, now=5)
    ended = _snapshot(db)
    assert rooms.disband_quarantined_room(db, room_id="room-1", confirmed=True, now=9) == {
        **first, "idempotent": True}
    assert _snapshot(db) == ended


def test_confirmed_disband_keeps_the_history_from_any_pruning(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    _quarantined_room(db)
    rooms.disband_quarantined_room(db, room_id="room-1", confirmed=True, now=1)
    ended = _snapshot(db)
    monkeypatch.setattr(rooms, "MAX_DISBANDED_ROOM_TOMBSTONES", 0)
    assert rooms.prune_disbanded_rooms(db, now=rooms.DISBANDED_ROOM_RETENTION_SECONDS + 10) == 0
    with rooms._transaction(db, immediate=True) as conn:
        assert rooms._prune_disbanded_rooms_locked(conn, now=None, max_gateway_event_bytes=0) == 0
    # An older process prunes disbanded rooms without knowing about quarantine: the store refuses it.
    for statement in ("DELETE FROM hosted_room_events WHERE room_id='room-1'",
                      "DELETE FROM hosted_rooms WHERE room_id='room-1'"):
        with sqlite3.connect(db) as conn, pytest.raises(sqlite3.IntegrityError, match="history is kept"):
            conn.execute(statement)
    assert _snapshot(db) == ended


def test_confirmation_does_not_bypass_the_normal_disband(tmp_path):
    db = tmp_path / "state.db"
    rooms.create_room(db, room_id="room-1", name="Field Room", members=MEMBERS, authority_gateway_id=REMOTE)
    with pytest.raises(rooms.RoomConflictError, match="not quarantined"):
        rooms.disband_quarantined_room(db, room_id="room-1", confirmed=True)
    assert "disbanded_at" not in rooms.room_state(db, room_id="room-1")
    # The normal Disband still requires this gateway to hold the room's authority.
    with pytest.raises(rooms.AuthorityConflictError):
        rooms.disband_room(db, room_id="room-1", expected_gateway_id=LOCAL, expected_epoch=1)


def test_a_quarantined_room_takes_only_the_confirmed_tombstone(tmp_path):
    db = tmp_path / "state.db"
    _quarantined_room(db)
    with sqlite3.connect(db) as conn:
        # No recorded confirmation: an older process or a stray write cannot end the room.
        with pytest.raises(sqlite3.IntegrityError, match="quarantined"):
            conn.execute("UPDATE hosted_rooms SET disbanded_at=1 WHERE room_id='room-1'")
        conn.execute("INSERT INTO hosted_room_quarantine_disbands VALUES ('room-1', 1)")
        # Even confirmed, the tombstone cannot carry an authority change...
        with pytest.raises(sqlite3.IntegrityError, match="quarantined"):
            conn.execute("UPDATE hosted_rooms SET disbanded_at=1, authority_gateway_id=?, authority_epoch=2 "
                          "WHERE room_id='room-1'", (REMOTE,))
        conn.execute("UPDATE hosted_rooms SET disbanded_at=1 WHERE room_id='room-1'")
        # ...and once ended, the room stays ended.
        with pytest.raises(sqlite3.IntegrityError, match="quarantined"):
            conn.execute("UPDATE hosted_rooms SET disbanded_at=NULL WHERE room_id='room-1'")


def test_canonical_disband_of_a_quarantined_room_stops_and_sends_nothing(tmp_path, monkeypatch):
    from gateway.session_controls import AuthorityConnection

    home = tmp_path / "state"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HOME", str(tmp_path / "user"))
    (home / "config.yaml").write_text("model:\n  default: fixture-model\n  provider: custom\n")

    def forbidden(*args, **kwargs):
        raise AssertionError("a quarantined Disband must not stop work or contact other gateways")

    with SessionDB(home / "state.db") as db:
        _quarantined_room(db.db_path, gateway=rooms.local_authority_gateway_id())
        history = _snapshot(db.db_path)["events"]
        service = SimpleNamespace(
            db_path=db.db_path, runtime=SimpleNamespace(status=lambda: {"running": True}),
            authorize_room=lambda *args, **kwargs: True, stop_room=forbidden, revoke_room_routes=forbidden)
        authority = SimpleNamespace(profile_id=str(home), instance_id="owner", db=db, events={}, sessions={},
                                    hosted_room_service=service)

        async def call(method, **params):
            connection = AuthorityConnection(authority, object(), {"user_id": "owner"})
            return await connection.dispatch({"id": "request", "method": method, "params": params})

        async def probe():
            refused = await call("groups.disband", room_id="room-1")
            assert refused["error"]["data"] == {"reason": "room_authority_quarantined"}, refused
            confirmed = await call("groups.disband", room_id="room-1", confirm_quarantined=True)
            tombstone = confirmed["result"]["tombstone"]
            assert tombstone["idempotent"] is False and "event" not in tombstone
            again = await call("groups.disband", room_id="room-1", confirm_quarantined=True)
            assert again["result"]["tombstone"] == {**tombstone, "idempotent": True}
            assert (await call("groups.list"))["result"]["rooms"] == []
            log = await call("groups.log", room_id="room-1", include_disbanded=True)
            assert [event["event_id"] for event in log["result"]["events"]] == ["e0", "e1"]
        asyncio.run(probe())
        assert _snapshot(db.db_path)["events"] == history
