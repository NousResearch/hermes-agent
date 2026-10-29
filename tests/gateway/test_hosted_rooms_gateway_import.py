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


def test_a_room_disbanded_in_the_gateway_never_comes_back(tmp_path):
    import time
    source, target = _stores(tmp_path)
    _standalone_room(source, "room-desk")
    import_standalone_rooms(target)
    [room] = hosted_rooms.list_rooms(target)
    hosted_rooms.disband_room(target, room_id="room-desk", expected_gateway_id=room["authority_gateway_id"],
                              expected_epoch=room["authority_epoch"])
    # The tombstone is pruned later (retention, count or byte cap); only the reserved id remains.
    later = time.time() + hosted_rooms.DISBANDED_ROOM_RETENTION_SECONDS + 10
    assert hosted_rooms.prune_disbanded_rooms(target, now=later) == 1

    assert import_standalone_rooms(target) == 0  # next gateway start
    assert hosted_rooms.list_rooms(target, include_disbanded=True) == []


def test_the_gateway_imports_again_once_ready(tmp_path):
    import threading
    from types import SimpleNamespace
    from gateway.hosted_rooms_gateway_import import import_after_ready
    source, target = _stores(tmp_path)
    woke = threading.Event()
    authority = SimpleNamespace(db=SimpleNamespace(db_path=str(target)), epoch=None,
                                hosted_room_service=SimpleNamespace(wakeup=woke.set))
    _standalone_room(source, "room-window")  # written by a standalone backend before `ready`

    import_after_ready([authority])

    assert woke.wait(10)
    assert [r["room_id"] for r in hosted_rooms.list_rooms(target)] == ["room-window"]


def test_gateway_startup_takes_over_the_standalone_rooms(tmp_path, monkeypatch):
    """The import is wired into the gateway's own hosted-service start, not only callable: a
    gateway starting beside a standalone store serves and drives the rooms that store held."""
    import asyncio
    from types import SimpleNamespace
    from gateway import session_hosted_service, session_hosted_transport
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch

    source = tmp_path / "shared-state.db"
    _standalone_room(source, "old-room")
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        authority = SimpleNamespace(db=db, epoch=begin_runtime_epoch(db, instance_id="first"),
                                    profile_id=str(tmp_path), runner=None)
        runner = SimpleNamespace(session_control_server=None)
        monkeypatch.setattr(session_hosted_transport, "install_hosted_transport", lambda *args, **kwargs: None)

        asyncio.run(session_hosted_service._ensure_hosted_service(runner, authority))

        assert [r["room_id"] for r in hosted_rooms.list_rooms(db.db_path)] == ["old-room"]
        assert [b.room_id for b in authority.hosted_room_service.bindings()] == ["old-room"]
    finally:
        db.close()
