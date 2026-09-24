"""Same-writer NEW-event fences and canonical receipt replay."""
from pathlib import Path
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from gateway import hosted_room_driver as driver
from gateway import hosted_rooms as rooms
from hermes_state_runtime import RuntimeStoreError
from tui_gateway.hosted_room_service import HostedRoomService


def _room(db, room_id="room-1"):
    return rooms.create_room(
        db,
        room_id=room_id,
        name=room_id,
        members=[
            {"member_id": "writer", "profile": "default", "handle": "writer"},
            {"member_id": "reviewer", "profile": "reviewer", "handle": "reviewer"},
        ],
        authority_gateway_id="gateway-a",
    )


def _append(db, *, room_id="room-1", event_id="event-1", text="hello", authorize_new=None):
    return rooms.append_event(
        db,
        room_id=room_id,
        event_id=event_id,
        kind="message.user",
        actor={"kind": "user", "id": "desktop"},
        payload={"text": text, "thread_id": "thread-1"},
        authority_gateway_id="gateway-a",
        authority_epoch=1,
        authorize_new=authorize_new,
    )


def _server():
    return SimpleNamespace(_methods={}, _sessions={}, _sessions_lock=threading.Lock())


def test_new_authorizer_runs_on_held_writer_only_and_replay_precedes_it(tmp_path):
    db = tmp_path / "state.db"
    _room(db)
    calls = []

    def allow(conn):
        assert conn.in_transaction
        assert Path(conn.execute("PRAGMA database_list").fetchone()[2]).resolve() == db.resolve()
        calls.append("new")

    first = _append(db, authorize_new=allow)
    assert calls == ["new"] and first["idempotent"] is False

    def deny(_conn):
        raise RuntimeStoreError("permission_denied")

    replay = _append(db, authorize_new=deny)
    assert replay["idempotent"] is True and replay["seq"] == first["seq"]
    assert calls == ["new"]

    with pytest.raises(rooms.EventConflictError):
        _append(db, text="changed", authorize_new=lambda _conn: pytest.fail("conflict is first"))
    with pytest.raises(RuntimeStoreError, match="permission_denied"):
        _append(db, event_id="event-denied", authorize_new=deny)
    assert [row["event_id"] for row in rooms.read_events(db, room_id="room-1")["events"]] == ["event-1"]


def test_disbanded_room_refuses_new_user_event_after_authorizer(tmp_path):
    db = tmp_path / "state.db"
    _room(db)
    rooms.disband_room(
        db,
        room_id="room-1",
        expected_gateway_id="gateway-a",
        expected_epoch=1,
    )
    calls = []
    with pytest.raises(rooms.RoomNotFoundError):
        _append(db, event_id="after-disband", authorize_new=lambda _conn: calls.append(True))
    assert calls == []
    with sqlite3.connect(db) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM hosted_room_events WHERE kind='message.user'"
        ).fetchone()[0]
    assert count == 0


def test_messaging_source_receipt_lookup_is_index_bounded(tmp_path):
    db = tmp_path / "state.db"
    _room(db)
    with sqlite3.connect(db) as conn:
        indexes = {row[1] for row in conn.execute("PRAGMA index_list(hosted_room_events)")}
        assert "idx_hosted_room_events_message_thread" in indexes
        plan = conn.execute(
            "EXPLAIN QUERY PLAN SELECT room_id,event_id FROM hosted_room_events "
            "WHERE kind='message.user' "
            "AND json_extract(payload_json,'$.thread_id')>=? "
            "AND json_extract(payload_json,'$.thread_id')<? LIMIT 1",
            ("msg:" + "0" * 64 + ":", "msg:" + "0" * 64 + ":g"),
        ).fetchall()
    assert any("idx_hosted_room_events_message_thread" in str(row) for row in plan)


def test_delegated_exact_replay_skips_policy_prepare_wakeup_and_second_task(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    monkeypatch.setattr(rooms, "local_authority_gateway_id", lambda: "gateway-a")
    service = HostedRoomService(_server(), db_path=db)
    service.local_profiles = lambda: ("default", "reviewer")
    service.create_room(
        room_id="room-1",
        name="Room",
        members=[
            {"member_id": "writer", "profile": "default", "handle": "writer"},
            {"member_id": "reviewer", "profile": "reviewer", "handle": "reviewer"},
        ],
    )
    prepared = []
    wakeups = []
    original_prepare = service.prepare_room

    def prepare(binding):
        prepared.append(binding)
        return original_prepare(binding)

    monkeypatch.setattr(service, "prepare_room", prepare)
    monkeypatch.setattr(service.runtime, "wakeup", lambda: wakeups.append(True))
    calls = []

    first = service.send(
        room_id="room-1",
        event_id=rooms.user_event_id("stable-source"),
        payload={"text": "@writer review this", "thread_id": "stable-source"},
        new_event_authorizer=lambda conn: calls.append(conn.in_transaction),
    )
    tasks = driver.list_tasks(db, room_id="room-1")
    assert first["idempotent"] is False
    assert calls == [True] and len(tasks) == 1
    assert len(prepared) == len(wakeups) == 1

    replay = service.send(
        room_id="room-1",
        event_id=rooms.user_event_id("stable-source"),
        payload={"text": "@writer review this", "thread_id": "stable-source"},
        new_event_authorizer=lambda _conn: pytest.fail("replay must not reacquire work consent"),
    )
    assert replay["idempotent"] is True and replay["seq"] == first["seq"]
    assert driver.list_tasks(db, room_id="room-1") == tasks
    assert len(prepared) == len(wakeups) == 1


def test_ordinary_service_send_keeps_existing_replay_prepare_and_wakeup_behavior(
    tmp_path, monkeypatch
):
    db = tmp_path / "state.db"
    monkeypatch.setattr(rooms, "local_authority_gateway_id", lambda: "gateway-a")
    service = HostedRoomService(_server(), db_path=db)
    service.local_profiles = lambda: ("default", "reviewer")
    service.create_room(
        room_id="room-1",
        name="Room",
        members=[
            {"member_id": "writer", "profile": "default", "handle": "writer"},
            {"member_id": "reviewer", "profile": "reviewer", "handle": "reviewer"},
        ],
    )
    prepared = []
    wakeups = []
    original_prepare = service.prepare_room

    def prepare(binding):
        prepared.append(binding)
        return original_prepare(binding)

    monkeypatch.setattr(service, "prepare_room", prepare)
    monkeypatch.setattr(service.runtime, "wakeup", lambda: wakeups.append(True))
    params = {
        "room_id": "room-1",
        "event_id": rooms.user_event_id("ordinary-source"),
        "payload": {"text": "@writer ordinary", "thread_id": "ordinary-source"},
    }
    first = service.send(**params)
    replay = service.send(**params)
    assert first["idempotent"] is False and replay["idempotent"] is True
    assert replay["seq"] == first["seq"]
    assert len(driver.list_tasks(db, room_id="room-1")) == 1
    assert len(prepared) == len(wakeups) == 2


def test_denied_new_service_send_has_no_event_task_prepare_or_wakeup(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    monkeypatch.setattr(rooms, "local_authority_gateway_id", lambda: "gateway-a")
    service = HostedRoomService(_server(), db_path=db)
    service.local_profiles = lambda: ("default", "reviewer")
    service.create_room(
        room_id="room-1",
        name="Room",
        members=[
            {"member_id": "writer", "profile": "default", "handle": "writer"},
            {"member_id": "reviewer", "profile": "reviewer", "handle": "reviewer"},
        ],
    )
    prepared = []
    wakeups = []
    monkeypatch.setattr(service, "prepare_room", lambda binding: prepared.append(binding))
    monkeypatch.setattr(service.runtime, "wakeup", lambda: wakeups.append(True))

    with pytest.raises(RuntimeStoreError, match="permission_denied"):
        service.send(
            room_id="room-1",
            event_id=rooms.user_event_id("denied-source"),
            payload={"text": "review this", "thread_id": "denied-source"},
            new_event_authorizer=lambda _conn: (_ for _ in ()).throw(
                RuntimeStoreError("permission_denied")
            ),
        )
    assert rooms.read_events(db, room_id="room-1")["events"] == []
    assert driver.list_tasks(db, room_id="room-1") == []
    assert prepared == wakeups == []
