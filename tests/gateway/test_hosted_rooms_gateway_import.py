"""The gateway takes over the standalone backend's Group Chats once, and each room keeps one driver (WA-19)."""

import sqlite3
from pathlib import Path

from gateway import hosted_rooms
from gateway.hosted_rooms_gateway_import import handoff_gateway_id, import_standalone_rooms
from gateway.owner_identity import host_owner_subject

MEMBERS = [{"member_id": "a", "profile": "default", "handle": "a"},
           {"member_id": "b", "profile": "default", "handle": "b"}]


def _stores(tmp_path: Path) -> tuple[Path, Path]:
    source, target = tmp_path / "shared-state.db", tmp_path / "state.db"
    with sqlite3.connect(target) as conn:  # the gateway's SessionDB table the owner claim lives in
        conn.execute("CREATE TABLE state_meta (key TEXT PRIMARY KEY, value TEXT)")
    return source, target


def _standalone_room(source: Path, room_id: str) -> None:
    hosted_rooms.create_room(source, room_id=room_id, name=room_id, members=MEMBERS,
                             authority_gateway_id=hosted_rooms.local_authority_gateway_id())
    hosted_rooms.append_event(
        source, room_id=room_id, event_id=f"user:{room_id}", kind="message.user",
        actor={"kind": "user", "id": "desktop"}, payload={"text": "hello"},
        authority_gateway_id=hosted_rooms.local_authority_gateway_id(), authority_epoch=1)


def test_standalone_rooms_move_once_with_history_owner_and_one_driver(tmp_path):
    source, target = _stores(tmp_path)
    _standalone_room(source, "room-desk")

    assert import_standalone_rooms(target) == 1
    assert import_standalone_rooms(target) == 0  # restart: no duplicate, no second claim

    local = hosted_rooms.local_authority_gateway_id()
    [room] = hosted_rooms.list_rooms(target)
    assert (room["room_id"], room["authority_gateway_id"]) == ("room-desk", local)
    kinds = [e["kind"] for e in hosted_rooms.read_events(target, room_id="room-desk")["events"]]
    assert kinds.count("message.user") == 1 and kinds[-2:] == ["authority.lost", "authority.claimed"]
    with sqlite3.connect(target) as conn:
        assert conn.execute("SELECT value FROM state_meta WHERE key=?",
                            ("gateway.hosted.owner.v1:room-desk",)).fetchone() == (host_owner_subject(),)
    # The source is kept, fenced away from the standalone driver (it binds only its own authority).
    [kept] = hosted_rooms.list_rooms(source)
    assert kept["authority_gateway_id"] == handoff_gateway_id()


def test_a_room_made_standalone_after_the_import_moves_on_the_next_start(tmp_path):
    source, target = _stores(tmp_path)
    _standalone_room(source, "room-one")
    import_standalone_rooms(target)
    _standalone_room(source, "room-two")  # gateway was down: Desktop kept working standalone

    assert import_standalone_rooms(target) == 1
    assert sorted(r["room_id"] for r in hosted_rooms.list_rooms(target)) == ["room-one", "room-two"]


def test_a_named_profile_store_imports_nothing(tmp_path):
    source, _ = _stores(tmp_path)
    _standalone_room(source, "room-desk")
    named = tmp_path / "profiles" / "work" / "state.db"
    named.parent.mkdir(parents=True)

    assert import_standalone_rooms(named) == 0
    assert hosted_rooms.list_rooms(source)[0]["authority_gateway_id"] == hosted_rooms.local_authority_gateway_id()
