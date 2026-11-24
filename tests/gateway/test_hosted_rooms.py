"""Behavior tests for the gateway-hosted room event log."""

from __future__ import annotations

import json
import logging
import sqlite3
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import pytest

from gateway import hosted_room_driver as driver
from gateway import hosted_rooms as rooms
import sqlite_support
from gateway.hosted_room_policy_checkpoint import HostedRoomPolicyCheckpoint
from persistence.core.errors import PersistenceError
from persistence.sqlite.gateway import _hosted_rooms_schema as backend_schema
from persistence.sqlite.gateway import hosted_rooms as backend_rooms
from persistence_assembly import open_persistence, profile_scope
from persistence.gateway.hosted_rooms import (
    AuthorityConflictError, AuthoritySupersededError, EventConflictError, HostedRoomError, HostedTaskIdentity, RoomConflictError, RoomHistoryExpiredError, RoomNotFoundError)

USER = {"kind": "user", "id": "desktop-user", "display_name": "User"}
GATEWAY_A = {"kind": "gateway", "id": "gateway-a"}
GATEWAY_B = {"kind": "gateway", "id": "gateway-b"}


@pytest.fixture(name="handle")
def handle_fixture(tmp_path):
    """A store handle on the test home, closed at teardown."""
    handle = open_persistence(profile_scope(home=tmp_path))
    yield handle
    handle.close()


def _create_pre_actor_database(path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """CREATE TABLE hosted_rooms (
                room_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                members_json TEXT NOT NULL,
                next_seq INTEGER NOT NULL,
                revision INTEGER NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                disbanded_at REAL
            )"""
        )
        conn.execute(
            """CREATE TABLE hosted_room_events (
                room_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                event_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (room_id, seq),
                UNIQUE (room_id, event_id)
            )"""
        )
        conn.execute(
            """INSERT INTO hosted_rooms
               VALUES ('room-1', 'Legacy', '[]', 2, 1, 1, 1, NULL)"""
        )
        conn.execute(
            """INSERT INTO hosted_room_events
               VALUES ('room-1', 1, 'legacy-event', 'message.created', '{}', 1)"""
        )
        conn.commit()
    finally:
        conn.close()


def _read_legacy_state(path: str) -> tuple[str, int]:
    placement = "shared" if Path(path).name == "shared-state.db" else "profile"
    with open_persistence(profile_scope(home=Path(path).parent)) as store:
        state = rooms.room_state(store, placement=placement, room_id="room-1")
    return state["authority_gateway_id"], state["latest_seq"]


def _create(handle, room_id="room-1", placement="profile"):
    return rooms.create_room(
        handle, placement=placement,
        room_id=room_id,
        name="Release room",
        members=[{"profile": "ops", "handle": "ops"}],
        authority_gateway_id="gateway-a",
        now=10,
    )


def _append(handle, placement="profile", **kwargs):
    if kwargs.get("kind") == "message.user":
        kwargs.setdefault("authority_gateway_id", "gateway-a")
        kwargs.setdefault("authority_epoch", 1)
    return rooms.append_event(handle, placement=placement, **kwargs)


def _disband(handle, **kwargs):
    kwargs.setdefault("expected_gateway_id", "gateway-a")
    kwargs.setdefault("expected_epoch", 1)
    return rooms.disband_room(handle, placement="profile", **kwargs)


def _assert_retired_identity_stays_reserved(handle, db, room_id, *, fresh_id):
    # Reopen the database to prove the reservation is durable rather than a
    # process-local cache, while the heavier room/event payload is gone.
    with sqlite3.connect(db) as conn:
        assert (
            conn.execute(
                "SELECT 1 FROM hosted_rooms WHERE room_id=?", (room_id,)
            ).fetchone()
            is None
        )
        assert (
            conn.execute(
                "SELECT retired_at FROM hosted_room_retired_ids WHERE room_id=?",
                (room_id,),
            ).fetchone()
            is not None
        )

    with pytest.raises(RoomConflictError, match="disbanded"):
        _create(handle, room_id)
    with pytest.raises(RoomHistoryExpiredError) as send_error:
        _append(
            handle,
            room_id=room_id,
            event_id="stale-send",
            kind="message.user",
            actor=USER,
            payload={"text": "stale"},
        )
    assert send_error.value.reason == "room_history_expired"
    with pytest.raises(RoomHistoryExpiredError) as log_error:
        rooms.read_events(handle, placement="profile", room_id=room_id, include_disbanded=True)
    assert log_error.value.reason == "room_history_expired"
    repeated = _disband(handle, room_id=room_id, now=999)
    assert repeated == {
        "room_id": room_id,
        "disbanded_at": 20.0,
        "idempotent": True,
        "history_expired": True,
    }

    assert _create(handle, fresh_id)["room_id"] == fresh_id


def test_create_room_is_idempotent_but_conflicts_fail_closed(tmp_path, handle):
    first = _create(handle)
    second = _create(handle)

    assert first["idempotent"] is False
    assert second["idempotent"] is True
    assert first["room_id"] == second["room_id"] == "room-1"

    with pytest.raises(RoomConflictError):
        rooms.create_room(
            handle, placement="profile",
            room_id="room-1",
            name="A different room",
            members=[],
            authority_gateway_id="gateway-a",
        )


def test_concurrent_first_database_open_keeps_every_room(tmp_path, handle):

    with ThreadPoolExecutor(max_workers=8) as pool:
        created = list(pool.map(lambda index: _create(handle, f"room-{index}"), range(16)))

    assert {room["room_id"] for room in created} == {
        f"room-{index}" for index in range(16)
    }
    assert len(rooms.list_rooms(handle, placement="profile")) == 16


def test_first_database_open_retries_only_transient_journal_lock(
    tmp_path,
    monkeypatch, handle,
):
    original = sqlite_support.apply_wal_with_fallback
    attempts = 0

    def transient_lock(conn, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise sqlite3.OperationalError("database is locked")
        return original(conn, **kwargs)

    monkeypatch.setattr(sqlite_support, "apply_wal_with_fallback", transient_lock)

    assert _create(handle, placement="shared")["room_id"] == "room-1"
    assert attempts == 3


def test_first_database_open_does_not_retry_other_journal_errors(
    tmp_path,
    monkeypatch, handle,
):
    attempts = 0

    def configured_delete_refusal(conn, **kwargs):
        nonlocal attempts
        attempts += 1
        raise sqlite3.OperationalError(
            "could not verify configured journal_mode=delete (database is locked)"
        )

    monkeypatch.setattr(
        sqlite_support, "apply_wal_with_fallback",
        configured_delete_refusal,
    )

    with pytest.raises(PersistenceError, match="configured") as refused:
        _create(handle, placement="shared")
    assert isinstance(refused.value.__cause__, sqlite3.OperationalError)
    assert attempts == 1




def test_authority_claim_fences_stale_gateway_events(tmp_path, handle):
    db = tmp_path / "state.db"
    _create(handle)

    first = _append(
        handle,
        room_id="room-1",
        event_id="turn-1",
        kind="turn.started",
        actor=GATEWAY_A,
        authority_gateway_id="gateway-a",
        authority_epoch=1,
        payload={"member": "ops"},
    )
    claimed = rooms.claim_authority(
        db,
        room_id="room-1",
        expected_gateway_id="gateway-a",
        expected_epoch=1,
        new_gateway_id="gateway-b",
        event_id="claim-gateway-b",
        now=30,
    )
    retried = rooms.claim_authority(
        db,
        room_id="room-1",
        expected_gateway_id="gateway-a",
        expected_epoch=1,
        new_gateway_id="gateway-b",
        event_id="claim-gateway-b",
        now=40,
    )

    assert claimed["authority_gateway_id"] == "gateway-b"
    assert claimed["authority_epoch"] == 2
    assert claimed["idempotent"] is False
    assert claimed["claim_event"]["kind"] == "authority.claimed"
    assert claimed["claim_event"]["authority_epoch"] == 2
    assert claimed["claim_event"]["payload"] == {
        "previous_gateway_id": "gateway-a",
        "authority_gateway_id": "gateway-b",
        "authority_epoch": 2,
    }
    assert retried["authority_epoch"] == 2
    assert retried["idempotent"] is True
    assert retried["claim_event"]["seq"] == claimed["claim_event"]["seq"]
    assert retried["claim_event"]["idempotent"] is True
    state = rooms.room_state(handle, placement="profile", room_id="room-1")
    assert state["authority_claim"]["event_id"] == "claim-gateway-b"
    assert state["authority_claim"]["payload"]["previous_gateway_id"] == "gateway-a"

    # An exact retry admitted before takeover stays idempotent and cannot
    # produce a second side effect, even though its epoch is now stale.
    repeated = _append(
        handle,
        room_id="room-1",
        event_id="turn-1",
        kind="turn.started",
        actor=GATEWAY_A,
        authority_gateway_id="gateway-a",
        authority_epoch=1,
        payload={"member": "ops"},
    )
    assert repeated["seq"] == first["seq"]
    assert repeated["idempotent"] is True

    with pytest.raises(AuthorityConflictError, match="stale"):
        _append(
            handle,
            room_id="room-1",
            event_id="turn-2-stale",
            kind="turn.started",
            actor=GATEWAY_A,
            authority_gateway_id="gateway-a",
            authority_epoch=1,
            payload={"member": "ops"},
        )

    with pytest.raises(AuthorityConflictError, match="stale"):
        _append(
            handle,
            room_id="room-1",
            event_id="member-2-stale",
            kind="message.member",
            actor={"kind": "member", "id": "ops"},
            authority_gateway_id="gateway-a",
            authority_epoch=1,
            payload={"text": "stale result"},
        )

    current = _append(
        handle,
        room_id="room-1",
        event_id="turn-2-current",
        kind="turn.started",
        actor=GATEWAY_B,
        authority_gateway_id="gateway-b",
        authority_epoch=2,
        payload={"member": "ops"},
    )
    assert current["authority_epoch"] == 2
    current_member = _append(
        handle,
        room_id="room-1",
        event_id="member-2-current",
        kind="message.member",
        actor={"kind": "member", "id": "ops"},
        authority_gateway_id="gateway-b",
        authority_epoch=2,
        payload={"text": "current result"},
    )
    assert current_member["authority_epoch"] == 2


def test_concurrent_authority_claim_has_one_winner(tmp_path, handle):
    db = tmp_path / "state.db"
    _create(handle)

    def claim(gateway_id):
        try:
            return rooms.claim_authority(
                db,
                room_id="room-1",
                expected_gateway_id="gateway-a",
                expected_epoch=1,
                new_gateway_id=gateway_id,
                event_id=f"claim-{gateway_id}",
            )
        except AuthorityConflictError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, ["gateway-b", "gateway-c"]))

    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    assert winners[0]["authority_epoch"] == 2
    assert rooms.room_state(handle, placement="profile", room_id="room-1")["authority_gateway_id"] in {
        "gateway-b",
        "gateway-c",
    }


def test_retry_of_successful_but_superseded_claim_is_distinct(tmp_path, handle):
    db = tmp_path / "state.db"
    _create(handle)
    rooms.claim_authority(
        db,
        room_id="room-1",
        expected_gateway_id="gateway-a",
        expected_epoch=1,
        new_gateway_id="gateway-b",
        event_id="claim-b",
    )
    rooms.claim_authority(
        db,
        room_id="room-1",
        expected_gateway_id="gateway-b",
        expected_epoch=2,
        new_gateway_id="gateway-c",
        event_id="claim-c",
    )

    with pytest.raises(AuthoritySupersededError, match="later superseded"):
        rooms.claim_authority(
            db,
            room_id="room-1",
            expected_gateway_id="gateway-a",
            expected_epoch=1,
            new_gateway_id="gateway-b",
            event_id="claim-b",
        )


def test_authority_scoped_events_require_gateway_and_epoch(tmp_path, handle):
    _create(handle)

    with pytest.raises(HostedRoomError, match="authority_gateway_id"):
        rooms.append_event(
            handle, placement="profile",
            room_id="room-1",
            event_id="turn-1",
            kind="member.unavailable",
            actor=GATEWAY_A,
            payload={"member": "ops"},
        )

    with pytest.raises(HostedRoomError, match="actor.id"):
        _append(
            handle,
            room_id="room-1",
            event_id="turn-2",
            kind="turn.started",
            actor=GATEWAY_B,
            authority_gateway_id="gateway-a",
            authority_epoch=1,
            payload={"member": "ops"},
        )

    with pytest.raises(HostedRoomError, match="authority_gateway_id"):
        rooms.append_event(
            handle, placement="profile",
            room_id="room-1",
            event_id="message-1",
            kind="message.user",
            actor=USER,
            payload={"text": "hello"},
        )

    appended = _append(
        handle,
        room_id="room-1",
        event_id="message-1",
        kind="message.user",
        actor=USER,
        payload={"text": "hello"},
    )
    assert appended["authority_epoch"] == 1


def test_append_is_idempotent_and_conflicting_event_id_is_rejected(tmp_path, handle):
    _create(handle)

    first = _append(
        handle,
        room_id="room-1",
        event_id="event-1",
        kind="message.user",
        actor=USER,
        payload={"text": "hello"},
        now=20,
    )
    repeated = _append(
        handle,
        room_id="room-1",
        event_id="event-1",
        kind="message.user",
        actor=USER,
        payload={"text": "hello"},
        now=30,
    )

    assert first["seq"] == repeated["seq"] == 1
    assert repeated["idempotent"] is True
    assert rooms.read_events(handle, placement="profile", room_id="room-1")["latest_seq"] == 1

    with pytest.raises(EventConflictError):
        _append(
            handle,
            room_id="room-1",
            event_id="event-1",
            kind="message.user",
            actor=USER,
            payload={"text": "changed"},
        )


def test_since_seq_returns_ordered_deltas_and_stable_cursor(tmp_path, handle):
    _create(handle)
    for index in range(1, 5):
        _append(
            handle,
            room_id="room-1",
            event_id=f"event-{index}",
            kind="message.user",
            actor=USER,
            payload={"index": index},
            now=20 + index,
        )

    first = rooms.read_events(handle, placement="profile", room_id="room-1", since_seq=0, limit=2)
    assert [event["seq"] for event in first["events"]] == [1, 2]
    assert first == {
        "events": first["events"],
        "cursor": 2,
        "latest_seq": 4,
        "has_more": True,
        "authority": {"gateway_id": "gateway-a", "epoch": 1},
    }

    second = rooms.read_events(
        handle, placement="profile",
        room_id="room-1",
        since_seq=first["cursor"],
        limit=2,
    )
    assert [event["seq"] for event in second["events"]] == [3, 4]
    assert second["cursor"] == 4
    assert second["has_more"] is False

    settled = rooms.read_events(handle, placement="profile", room_id="room-1", since_seq=4)
    assert settled["events"] == []
    assert settled["cursor"] == settled["latest_seq"] == 4


def test_room_log_survives_store_reopen(tmp_path, handle):
    _create(handle)
    _append(
        handle,
        room_id="room-1",
        event_id="event-1",
        kind="message.user",
        actor=USER,
        payload={"text": "persist me"},
    )

    assert rooms.list_rooms(handle, placement="profile")[0]["name"] == "Release room"
    replay = rooms.read_events(handle, placement="profile", room_id="room-1")
    assert replay["events"][0]["payload"] == {"text": "persist me"}


@pytest.mark.parametrize("session_first", [True, False])
def test_room_tables_coexist_with_session_db_schema(tmp_path, session_first, handle):
    if session_first:
        with open_persistence(profile_scope(home=tmp_path)) as store:
            store.ready()

    _create(handle)
    _append(
        handle,
        room_id="room-1",
        event_id="event-1",
        kind="message.user",
        actor=USER,
        payload={"text": "shared database"},
    )

    with open_persistence(profile_scope(home=tmp_path)) as store:
        store.ready()
    replay = rooms.read_events(handle, placement="profile", room_id="room-1")
    assert replay["latest_seq"] == 1
    assert replay["events"][0]["payload"]["text"] == "shared database"


def test_concurrent_appends_allocate_one_monotonic_sequence(tmp_path, handle):
    _create(handle)

    def append(index):
        return _append(
            handle,
            room_id="room-1",
            event_id=f"event-{index}",
            kind="message.user",
            actor=USER,
            payload={"index": index},
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(append, range(24)))

    assert sorted(result["seq"] for result in results) == list(range(1, 25))
    replay = rooms.read_events(handle, placement="profile", room_id="room-1", limit=100)
    assert [event["seq"] for event in replay["events"]] == list(range(1, 25))


def test_rolled_back_append_does_not_consume_sequence(tmp_path, handle):
    db = tmp_path / "state.db"
    _create(handle)

    conn = sqlite3.connect(db)
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """INSERT INTO hosted_room_events
               (room_id, seq, event_id, kind, actor_json, payload_json, created_at)
               VALUES ('room-1', 1, 'crash-event', 'message.user',
                       '{"id":"desktop-user","kind":"user"}', '{}', 20)"""
        )
        conn.execute("UPDATE hosted_rooms SET next_seq=2 WHERE room_id='room-1'")
        conn.rollback()
    finally:
        conn.close()

    event = _append(
        handle,
        room_id="room-1",
        event_id="after-restart",
        kind="message.user",
        actor=USER,
        payload={"text": "safe"},
    )
    assert event["seq"] == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"room_id": "../escape"}, "invalid room_id"),
        ({"event_id": "event id"}, "invalid event_id"),
        ({"kind": "Message Created"}, "invalid event kind"),
        ({"kind": "directive.run"}, "cannot append"),
        ({"actor": {"kind": "member", "id": "ops"}}, "cannot append"),
        ({"payload": []}, "payload must be an object"),
    ],
)
def test_invalid_event_contract_is_rejected(tmp_path, kwargs, message, handle):
    _create(handle)
    params = {
        "room_id": "room-1",
        "event_id": "event-1",
        "kind": "message.user",
        "actor": USER,
        "payload": {},
    }
    params.update(kwargs)

    with pytest.raises(HostedRoomError, match=message):
        _append(handle, **params)


def test_unknown_room_and_invalid_cursor_fail_closed(tmp_path, handle):
    _create(handle)

    with pytest.raises(RoomNotFoundError):
        rooms.read_events(handle, placement="profile", room_id="missing")
    with pytest.raises(HostedRoomError, match="since_seq"):
        rooms.read_events(handle, placement="profile", room_id="room-1", since_seq=-1)
    with pytest.raises(HostedRoomError, match="ahead"):
        rooms.read_events(handle, placement="profile", room_id="room-1", since_seq=1)
    with pytest.raises(HostedRoomError, match="limit"):
        rooms.read_events(handle, placement="profile", room_id="room-1", limit=0)


def test_actor_is_part_of_event_idempotency_and_replay(tmp_path, handle):
    _create(handle)
    event = _append(
        handle,
        room_id="room-1",
        event_id="event-1",
        kind="message.user",
        actor=USER,
        payload={"text": "hello"},
    )

    assert event["actor"] == USER
    assert rooms.read_events(handle, placement="profile", room_id="room-1")["events"][0]["actor"] == USER

    with pytest.raises(EventConflictError):
        _append(
            handle,
            room_id="room-1",
            event_id="event-1",
            kind="message.user",
            actor={"kind": "user", "id": "another-user"},
            payload={"text": "hello"},
        )


def test_disband_is_idempotent_and_room_id_cannot_be_reused(tmp_path, handle):
    _create(handle)

    first = _disband(handle, room_id="room-1", now=50)
    repeated = _disband(handle, room_id="room-1", now=60)

    assert first["room_id"] == "room-1"
    assert first["disbanded_at"] == 50.0
    assert first["idempotent"] is False
    assert first["event"]["kind"] == "room.disbanded"
    assert first["event"]["seq"] == 1
    assert first["event"]["payload"] == {"room_id": "room-1"}
    assert repeated["idempotent"] is True
    assert repeated["event"]["seq"] == first["event"]["seq"]
    assert rooms.list_rooms(handle, placement="profile") == []
    deleted = rooms.list_rooms(handle, placement="profile", include_disbanded=True)
    assert deleted[0]["disbanded_at"] == 50.0
    assert deleted[0]["latest_seq"] == 1
    state = rooms.room_state(handle, placement="profile", room_id="room-1", include_disbanded=True)
    assert state["disbanded_at"] == 50.0
    assert state["latest_seq"] == 1
    replay = rooms.read_events(handle, placement="profile", room_id="room-1", include_disbanded=True)
    assert [event["kind"] for event in replay["events"]] == ["room.disbanded"]
    with pytest.raises(RoomConflictError):
        _create(handle)
    with pytest.raises(RoomNotFoundError) as missing:
        rooms.read_events(handle, placement="profile", room_id="room-1")
    assert not isinstance(missing.value, RoomHistoryExpiredError)


def test_disband_rejects_stale_authority(tmp_path, handle):
    db = tmp_path / "state.db"
    _create(handle)
    rooms.claim_authority(
        db,
        room_id="room-1",
        expected_gateway_id="gateway-a",
        expected_epoch=1,
        new_gateway_id="gateway-b",
        event_id="claim-b",
    )

    with pytest.raises(AuthorityConflictError, match="stale"):
        _disband(handle, room_id="room-1")

    tombstone = _disband(
        handle,
        room_id="room-1",
        expected_gateway_id="gateway-b",
        expected_epoch=2,
    )
    assert tombstone["event"]["authority_epoch"] == 2


def test_room_log_pages_are_bounded_by_serialized_event_bytes(tmp_path, monkeypatch, handle):
    _create(handle)
    for index in range(4):
        _append(
            handle,
            room_id="room-1",
            event_id=f"message-{index}",
            kind="message.user",
            actor=USER,
            payload={"text": "x" * 180, "index": index},
            # A fixed clock: the budget below is one event's serialized size, and a wall-clock
            # created_at's float repr varies by a few bytes from event to event.
            now=1_700_000_000.5,
        )

    one_event = rooms.read_events(handle, placement="profile", room_id="room-1", limit=1)
    budget = len(
        json.dumps(one_event, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ) + 1
    monkeypatch.setattr(backend_rooms, "MAX_LOG_PAGE_BYTES", budget)

    first = rooms.read_events(handle, placement="profile", room_id="room-1", limit=4)
    assert len(first["events"]) == 1
    assert first["has_more"] is True
    second = rooms.read_events(
        handle, placement="profile",
        room_id="room-1",
        since_seq=first["cursor"],
        limit=4,
    )
    assert second["events"][0]["seq"] == first["cursor"] + 1




def test_room_log_page_bound_includes_json_structure_overhead(tmp_path, monkeypatch, handle):
    _create(handle)
    for index in range(20):
        _append(
            handle,
            room_id="room-1",
            event_id=f"small-{index:02d}",
            kind="message.user",
            actor=USER,
            payload={"text": "ok", "index": index},
        )

    budget = 1_000
    monkeypatch.setattr(backend_rooms, "MAX_LOG_PAGE_BYTES", budget)
    cursor = 0
    replayed = []
    while True:
        page = rooms.read_events(handle, placement="profile", room_id="room-1", since_seq=cursor, limit=20)
        assert page["events"]
        assert len(
            json.dumps(page, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        ) <= budget
        replayed.extend(page["events"])
        assert page["cursor"] > cursor
        cursor = page["cursor"]
        if not page["has_more"]:
            break

    assert [event["seq"] for event in replayed] == list(range(1, 21))


def test_active_rooms_and_event_storage_are_bounded(tmp_path, monkeypatch, handle):
    db = tmp_path / "state.db"
    _create(handle)
    monkeypatch.setattr(backend_rooms, "MAX_ACTIVE_ROOMS", 1)

    with pytest.raises(HostedRoomError, match="too many active"):
        _create(handle, "room-2")

    _disband(handle, room_id="room-1", now=20)
    _create(handle, "room-2")
    first = _append(
        handle,
        room_id="room-2",
        event_id="message-1",
        kind="message.user",
        actor=USER,
        payload={"text": "first"},
    )
    with sqlite3.connect(db) as conn:
        current_bytes = conn.execute(
            "SELECT event_bytes FROM hosted_rooms WHERE room_id='room-2'"
        ).fetchone()[0]
    monkeypatch.setattr(backend_rooms, "MAX_ROOM_EVENT_BYTES", current_bytes)

    with pytest.raises(HostedRoomError, match="storage limit"):
        _append(
            handle,
            room_id="room-2",
            event_id="message-2",
            kind="message.user",
            actor=USER,
            payload={"text": "second"},
        )
    assert rooms.read_events(handle, placement="profile", room_id="room-2")["events"] == [first]
    assert _disband(handle, room_id="room-2")["event"]["kind"] == "room.disbanded"


def test_room_listing_is_paged_and_old_tombstones_are_pruned(tmp_path, monkeypatch, handle):
    db = tmp_path / "state.db"
    monkeypatch.setattr(backend_rooms, "MAX_DISBANDED_ROOM_TOMBSTONES", 1)
    for index in range(3):
        room_id = f"room-{index}"
        _create(handle, room_id)
        if index < 2:
            _disband(handle, room_id=room_id, now=20 + index)

    first_page = rooms.list_rooms(handle, placement="profile", include_disbanded=True, limit=1)
    second_page = rooms.list_rooms(
        handle, placement="profile",
        include_disbanded=True,
        limit=1,
        offset=1,
    )

    assert [room["room_id"] for room in first_page + second_page] == [
        "room-1",
        "room-2",
    ]
    with pytest.raises(RoomNotFoundError):
        rooms.room_state(handle, placement="profile", room_id="room-0", include_disbanded=True)
    _assert_retired_identity_stays_reserved(handle, db, "room-0", fresh_id="room-new")


def test_retention_pruning_keeps_retired_room_id_reserved(tmp_path, handle):
    db = tmp_path / "state.db"
    _create(handle, "room-old")
    _disband(handle, room_id="room-old", now=20)

    assert (
        rooms.prune_disbanded_rooms(
            handle, placement="profile",
            now=20 + rooms.DISBANDED_ROOM_RETENTION_SECONDS + 1,
        )
        == 1
    )

    _assert_retired_identity_stays_reserved(handle, db, "room-old", fresh_id="room-new")


def test_byte_pressure_pruning_keeps_retired_room_id_reserved(
    tmp_path,
    monkeypatch, handle,
):
    db = tmp_path / "state.db"
    _create(handle, "room-full")
    monkeypatch.setattr(backend_rooms, "MAX_GATEWAY_EVENT_BYTES", 0)

    _disband(handle, room_id="room-full", now=20)

    _assert_retired_identity_stays_reserved(handle, db, "room-full", fresh_id="room-new")


def test_tombstone_pruning_owns_only_room_log_driver_and_policy_tables(tmp_path, handle):
    db = tmp_path / "state.db"
    _create(handle)
    identity = HostedTaskIdentity(
        room_id="room-1",
        task_id="task-1",
        thread_id="thread-1",
        turn_id="turn-1",
    )
    driver.admit_task(
        handle,
        identity, placement="profile",
        payload={
            "target_profile": "ops",
            "prompt": "Inspect.",
            "source_event_seq": 1,
        },
        clock=lambda: 20,
    )
    driver.acquire_lease(
        handle, placement="profile",
        room_id="room-1",
        gateway_id="gateway-a",
        authority_epoch=1,
        process_generation="process-a",
        ttl_seconds=30,
        clock=lambda: 20,
    )
    HostedRoomPolicyCheckpoint(handle, placement="profile").compact_completed(room_id="room-1")
    _disband(handle, room_id="room-1", now=50)
    with sqlite3.connect(db) as conn:
        conn.execute(
            """INSERT INTO hosted_room_policy_cursors(
                   room_id, through_seq, stopped_through_seq, updated_at
               ) VALUES ('room-1', 1, 0, 50)"""
        )
        conn.execute(
            """INSERT INTO hosted_room_policy_threads
               VALUES ('room-1', 'thread-1', 'user-1', 1, 0)"""
        )
        conn.execute(
            """INSERT INTO hosted_room_policy_events
               VALUES ('room-1', 'thread-1', 'user-1', 1, '{}')"""
        )
        conn.execute(
            """INSERT INTO hosted_room_policy_watermarks
               VALUES ('room-1', 'thread-1', 'ops', 1)"""
        )
        conn.execute(
            """INSERT INTO hosted_room_policy_publications
               VALUES ('room-1', 'task-1', 'turn.settled', 0, 1)"""
        )
        conn.execute(
            """INSERT INTO hosted_room_policy_transcript
               VALUES (
                   'room-1', 'thread-1', 1, 'message.user', NULL
               )"""
        )
        conn.execute(
            """CREATE TABLE hosted_room_messaging_refs (
                room_id TEXT NOT NULL,
                marker TEXT NOT NULL
            )"""
        )
        conn.execute(
            """INSERT INTO hosted_room_policy_transcript_state
               VALUES ('room-1', 1)"""
        )
        conn.execute(
            """INSERT INTO hosted_room_messaging_refs
               VALUES ('room-1', 'outside-pr-b')"""
        )

    assert (
        rooms.prune_disbanded_rooms(
            handle, placement="profile",
            now=50 + rooms.DISBANDED_ROOM_RETENTION_SECONDS + 1,
        )
        == 1
    )
    with sqlite3.connect(db) as conn:
        for table in (
            "hosted_rooms",
            "hosted_room_events",
            "hosted_room_driver_tasks",
            "hosted_room_driver_leases",
            "hosted_room_policy_cursors",
            "hosted_room_policy_threads",
            "hosted_room_policy_events",
            "hosted_room_policy_watermarks",
            "hosted_room_policy_publications",
            "hosted_room_policy_transcript",
            "hosted_room_policy_transcript_state",
        ):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
        assert (
            conn.execute("SELECT marker FROM hosted_room_messaging_refs").fetchone()[0]
            == "outside-pr-b"
        )


def test_peer_reservation_rejects_stale_or_conflicting_authority(tmp_path, handle):
    db = tmp_path / "state.db"
    current = {
        "room_id": "room-peer",
        "member_id": "member-peer",
        "target_profile": "reviewer",
        "authority_gateway_id": "gateway-current",
        "authority_epoch": 2,
    }
    rooms.reserve_peer_room(handle, placement="profile", claims=current, expires_at=300, now=100)

    with pytest.raises(AuthorityConflictError, match="authority changed"):
        rooms.reserve_peer_room(
            handle, placement="profile",
            claims={
                **current,
                "authority_gateway_id": "gateway-stale",
                "authority_epoch": 1,
            },
            expires_at=300,
            now=100,
        )
    with pytest.raises(AuthorityConflictError, match="authority changed"):
        rooms.reserve_peer_room(
            handle, placement="profile",
            claims={**current, "authority_gateway_id": "gateway-conflict"},
            expires_at=300,
            now=100,
        )
    assert rooms.peer_room_is_reserved(
        db,
        room_id="room-peer",
        target_profile="reviewer",
        now=200,
    )


def test_policy_sync_cannot_recreate_projection_after_room_pruning(
    tmp_path,
    monkeypatch, handle,
):
    db = tmp_path / "state.db"
    _create(handle)
    _append(
        handle,
        room_id="room-1",
        event_id="user-1",
        kind="message.user",
        actor=USER,
        payload={"text": "hello", "thread_id": "thread-1"},
        now=11,
    )
    checkpoint = HostedRoomPolicyCheckpoint(handle, placement="profile")
    original_read = backend_rooms.read_events_page

    def read_then_prune(*args, **kwargs):
        page = original_read(*args, **kwargs)
        _disband(handle, room_id="room-1", now=20)
        rooms.prune_disbanded_rooms(
            handle, placement="profile",
            now=20 + rooms.DISBANDED_ROOM_RETENTION_SECONDS + 1,
        )
        return page

    monkeypatch.setattr(backend_rooms, "read_events_page", read_then_prune)
    with pytest.raises(RoomNotFoundError, match="not found"):
        checkpoint.sync(room_id="room-1", latest_seq=1)
    with sqlite3.connect(db) as conn:
        for table in (
            "hosted_room_policy_cursors",
            "hosted_room_policy_events",
            "hosted_room_policy_transcript",
            "hosted_room_policy_transcript_state",
        ):
            assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_pre_actor_draft_database_migrates_with_explicit_legacy_identity(tmp_path, handle):
    db = tmp_path / "state.db"
    _create_pre_actor_database(db)

    replay = rooms.read_events(handle, placement="profile", room_id="room-1")
    assert replay["events"][0]["actor"] == {"kind": "system", "id": "legacy"}
    state = rooms.room_state(handle, placement="profile", room_id="room-1")
    assert state["authority_gateway_id"] == "legacy"
    assert state["authority_epoch"] == 1
    adopted = rooms.create_room(
        handle, placement="profile",
        room_id="room-1",
        name="Legacy",
        members=[],
        authority_gateway_id="gateway-a",
        now=2,
    )
    assert adopted["authority_gateway_id"] == "gateway-a"
    assert adopted["authority_epoch"] == 2
    assert adopted["adopted"] is True
    assert adopted["claim_event"]["seq"] == 2
    assert adopted["claim_event"]["payload"]["previous_gateway_id"] == "legacy"


def test_migration_reserves_existing_disbanded_room_id_before_pruning(tmp_path, handle):
    db = tmp_path / "state.db"
    _create_pre_actor_database(db)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE hosted_rooms SET disbanded_at=20 WHERE room_id='room-1'")
        conn.commit()

    assert (
        rooms.prune_disbanded_rooms(
            handle, placement="profile",
            now=20 + rooms.DISBANDED_ROOM_RETENTION_SECONDS + 1,
        )
        == 1
    )

    _assert_retired_identity_stays_reserved(handle, db, "room-1", fresh_id="room-new")


def test_legacy_adoption_fills_missing_targets_but_rejects_target_changes(tmp_path, handle):
    legacy_members = [
        {"member_id": "ops", "profile": "ops", "handle": "ops"},
    ]
    targeted_members = [
        {
            **legacy_members[0],
            "target": {"kind": "local", "profile": "ops"},
        },
    ]
    rooms.create_room(
        handle, placement="profile",
        room_id="legacy-targets",
        name="Legacy targets",
        members=legacy_members,
        authority_gateway_id="legacy",
        now=1,
    )

    adopted = rooms.create_room(
        handle, placement="profile",
        room_id="legacy-targets",
        name="Legacy targets",
        members=targeted_members,
        authority_gateway_id="gateway-a",
        now=2,
    )

    assert adopted["members"] == targeted_members
    with pytest.raises(RoomConflictError, match="different state"):
        rooms.create_room(
            handle, placement="profile",
            room_id="legacy-targets",
            name="Legacy targets",
            members=[
                {
                    **legacy_members[0],
                    "target": {
                        "kind": "peer",
                        "installation_id": "install-b",
                        "profile": "ops",
                    },
                },
            ],
            authority_gateway_id="gateway-a",
            now=3,
        )


def test_draft_schema_migration_is_safe_across_processes(tmp_path, handle):
    db = tmp_path / "state.db"
    _create_pre_actor_database(db)

    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_read_legacy_state, [str(db)] * 4))

    assert results == [("legacy", 1)] * 4
    replay = rooms.read_events(handle, placement="profile", room_id="room-1")
    assert replay["events"][0]["actor"] == {"kind": "system", "id": "legacy"}


def test_legacy_remote_run_receipt_migrates_without_current_lineage_access(
    tmp_path, handle,
):
    db = tmp_path / "state.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            """CREATE TABLE hosted_room_remote_runs (
                room_id TEXT NOT NULL,
                member_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                execution_generation INTEGER NOT NULL,
                target_install_id TEXT NOT NULL,
                target_profile TEXT NOT NULL,
                run_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (task_id, execution_generation)
            )"""
        )
        conn.execute(
            """INSERT INTO hosted_room_remote_runs VALUES(
                'room-1', 'member-reviewer', 'task-1', 1, 'install-peer',
                'reviewer', 'run-legacy', 'session-legacy', 1, 1
            )"""
        )

    current = {
        "room_id": "room-1",
        "home_install_id": "install-home",
        "authority_gateway_id": "gateway-home",
        "authority_epoch": 2,
        "member_id": "member-reviewer",
        "target_install_id": "install-peer",
        "target_profile": "reviewer",
        "task_id": "task-1",
        "execution_generation": 1,
    }
    assert rooms.remote_run_receipt(handle, placement="profile", record=current) is None
    legacy = rooms.list_remote_run_receipts(db)
    assert legacy[0]["home_install_id"] == "legacy"
    assert legacy[0]["authority_gateway_id"] == "legacy"

    rooms.upsert_remote_run_receipt(
        handle, placement="profile",
        record={**current, "run_id": "run-current", "session_id": "session-current"},
    )
    assert rooms.remote_run_receipt(handle, placement="profile", record=current)["run_id"] == "run-current"


def test_interrupted_draft_schema_migration_rolls_back_atomically(
    tmp_path,
    monkeypatch, handle,
):
    db = tmp_path / "state.db"
    _create_pre_actor_database(db)
    original = backend_schema._initialize_schema

    def interrupt_after_first_alter(conn):
        conn.execute(
            "ALTER TABLE hosted_rooms "
            "ADD COLUMN authority_gateway_id TEXT NOT NULL DEFAULT 'legacy'"
        )
        raise RuntimeError("simulated migration interruption")

    monkeypatch.setattr(backend_schema, "_initialize_schema", interrupt_after_first_alter)
    with pytest.raises(RuntimeError, match="simulated migration interruption"):
        rooms.list_rooms(handle, placement="profile")
    with sqlite3.connect(db) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(hosted_rooms)")}
    assert "authority_gateway_id" not in columns

    monkeypatch.setattr(backend_schema, "_initialize_schema", original)
    assert rooms.room_state(handle, placement="profile", room_id="room-1")["authority_gateway_id"] == "legacy"


def test_gateway_event_budget_leaves_pre_update_snapshot_headroom():
    from hermes_cli import update_cmd

    # SQLite stores room ids and index entries beyond the logical payload
    # accounting, while session data shares the same file. Keep at least an
    # order-of-magnitude safety margin below the physical snapshot cutoff.
    assert (
        rooms.MAX_GATEWAY_EVENT_BYTES * 32
        <= update_cmd._PRE_UPDATE_SNAPSHOT_MAX_FILE_SIZE
    )


def test_room_log_page_bound_counts_bytes_not_characters(tmp_path, monkeypatch, handle):
    """Char-based accounting (SQLite LENGTH(TEXT) / len(str)) must not let a
    multibyte page exceed the byte ceiling.

    Budget sits between the character length and the byte length of a
    two-event page: character accounting would admit both events, byte
    accounting admits exactly one.
    """
    _create(handle)
    for index in range(2):
        _append(
            handle,
            room_id="room-1",
            event_id=f"bytes-{index}",
            kind="message.user",
            actor=USER,
            payload={"text": "😀" * 200, "index": index},
        )

    def page_json(page):
        return json.dumps(page, ensure_ascii=False, separators=(",", ":"))

    def page_bytes(page):
        return len(page_json(page).encode("utf-8"))

    one_event = rooms.read_events(handle, placement="profile", room_id="room-1", since_seq=0, limit=1)
    unbounded_two = rooms.read_events(handle, placement="profile", room_id="room-1", since_seq=0, limit=2)
    budget = page_bytes(one_event) + 64

    # Precondition: the budget discriminates — char accounting would accept
    # the two-event page, byte accounting must reject it.
    assert len(page_json(unbounded_two)) <= budget
    assert page_bytes(unbounded_two) > budget

    monkeypatch.setattr(backend_rooms, "MAX_LOG_PAGE_BYTES", budget)
    page = rooms.read_events(handle, placement="profile", room_id="room-1", since_seq=0, limit=2)
    assert [event["seq"] for event in page["events"]] == [1]
    assert page_bytes(page) <= budget
    assert page["has_more"] is True


def test_default_db_path_never_names_the_master_session_store(tmp_path, monkeypatch):
    """Hosted-room coordination lives beside, never inside, the master ``state.db``.

    Every profile gateway starts the hosted-room worker, so a store resolved to the
    root session DB made every profile process a long-lived writer on state.db —
    the multi-profile restart corruption in #102120 / #103339 / #103490. The store
    is shared across profiles (one file at the install root) but is not the
    session store file the root gateway owns.
    """
    root = tmp_path / ".hermes"
    profile_home = root / "profiles" / "bot1"
    profile_home.mkdir(parents=True)

    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    from_profile = rooms.default_db_path()
    monkeypatch.setenv("HERMES_HOME", str(root))
    from_root = rooms.default_db_path()

    assert from_profile == from_root, "one coordination file per install"
    assert from_root.parent == root
    assert from_root.name != "state.db"


def test_upgrade_keeps_rooms_from_before_the_shared_state_db_split(tmp_path, handle):
    """Rooms an install already had stay reachable after the move to ``shared-state.db``.

    ``0e422e0ece`` repointed the store at ``shared-state.db`` but left the hosted_room* rows in
    the root ``state.db``, so every pre-existing room resolved to "hosted room not found" (#109775).
    This is that upgrade: rooms and their events already in ``state.db``, nothing in the new file.
    """
    _create(handle)
    _append(
        handle,
        room_id="room-1",
        event_id="event-1",
        kind="message.user",
        actor=USER,
        payload={"text": "before the upgrade"},
        now=11,
    )


    assert [room["room_id"] for room in rooms.list_rooms(handle, placement="shared")] == ["room-1"]
    assert rooms.room_state(handle, placement="shared", room_id="room-1")["latest_seq"] == 1
    assert [
        event["event_id"] for event in rooms.read_events(handle, placement="shared", room_id="room-1")["events"]
    ] == ["event-1"]


def test_legacy_import_is_a_one_shot_and_skips_driver_liveness_state(tmp_path, handle):
    """The copy runs once, never overwrites, and leaves the driver's lease behind (#109775)."""
    _create(handle)

    def now() -> float:
        return 100.0

    driver.admit_task(
        handle,
        HostedTaskIdentity(room_id="room-1", task_id="task-1", thread_id="thread-1", turn_id="turn-1"), placement="profile",
        payload={"target_profile": "ops", "prompt": "ping", "source_event_seq": 1},
        clock=now,
    )
    driver.acquire_lease(
        handle, placement="profile",
        room_id="room-1",
        gateway_id="gateway-a",
        authority_epoch=1,
        process_generation="process-a",
        ttl_seconds=30,
        clock=now,
    )

    store = tmp_path / "shared-state.db"
    assert [room["room_id"] for room in rooms.list_rooms(handle, placement="shared")] == ["room-1"]
    with sqlite3.connect(store) as conn:
        # Durable work follows the room across; the lease is liveness state and stays behind.
        assert conn.execute("SELECT COUNT(*) FROM hosted_room_driver_tasks").fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='hosted_room_driver_leases'"
        ).fetchone()[0] == 0

    # Forcing a second import (marker cleared) must not replace what this store already owns.
    rooms.rename_room(handle, placement="shared", room_id="room-1", event_id="rename-1", name="Renamed", now=12)
    with sqlite3.connect(store) as conn:
        conn.execute("DELETE FROM hosted_room_legacy_imports")
    assert rooms.room_state(handle, placement="shared", room_id="room-1")["name"] == "Renamed"

    # And the record of the import is what keeps a purge from being undone by a later open.
    with sqlite3.connect(store) as conn:
        conn.execute("DELETE FROM hosted_rooms")
    assert rooms.list_rooms(handle, placement="shared") == []


def test_legacy_import_skips_a_room_this_store_already_owns_as_a_unit(tmp_path, handle):
    """A room id present in both stores keeps THIS store's history intact and appendable.

    Grafting only the non-colliding legacy events under the store's own room left ``next_seq``
    behind ``MAX(seq)``, so every later append collided on (room_id, seq).
    """
    _create(handle, room_id="same")
    for index in range(5):
        _append(handle, room_id="same", event_id=f"legacy-{index}", kind="message.user", actor=USER,
                payload={"text": str(index)}, now=11 + index)
    # The store already has its own "same" before the import runs (a room re-created after "not found").
    with open_persistence(profile_scope(home=tmp_path / "scratch")) as own:
        _create(own, room_id="same")
        _append(own, room_id="same", event_id="own-1", kind="message.user", actor=USER, payload={"text": "s"}, now=11)
    (tmp_path / "scratch" / "state.db").rename(tmp_path / "shared-state.db")

    assert [event["event_id"] for event in rooms.read_events(handle, placement="shared", room_id="same")["events"]] == ["own-1"]
    _append(handle, placement="shared", room_id="same", event_id="own-2", kind="message.user", actor=USER, payload={"text": "t"}, now=30)
    assert rooms.room_state(handle, placement="shared", room_id="same")["latest_seq"] == 2


def test_legacy_import_reads_layouts_from_before_the_actor_and_authority_columns(tmp_path, handle):
    """A legacy store without authority_gateway_id/actor_json imports with the migration's defaults.

    ``INSERT OR IGNORE`` used to swallow the NOT NULL violations, drop every row and still record
    the marker with rooms=0, losing the rooms permanently.
    """
    _create_pre_actor_database(str(tmp_path / "state.db"))
    store = tmp_path / "shared-state.db"

    assert _read_legacy_state(str(store)) == ("legacy", 1)
    assert [event["event_id"] for event in rooms.read_events(handle, placement="shared", room_id="room-1")["events"]] == ["legacy-event"]
    with sqlite3.connect(store) as conn:
        assert conn.execute("SELECT rooms FROM hosted_room_legacy_imports").fetchone() == (1,)
        assert conn.execute("SELECT event_bytes FROM hosted_rooms").fetchone()[0] > 0


def test_unreadable_legacy_store_is_reported_once_per_process(tmp_path, caplog, handle):
    """A corrupt legacy file leaves the marker unset but does not re-warn on every poll."""
    (tmp_path / "state.db").write_bytes(b"not a sqlite file" * 100)
    store = tmp_path / "shared-state.db"

    with caplog.at_level(logging.WARNING, logger="gateway.hosted_rooms_legacy_import"):
        for _ in range(4):
            assert rooms.list_rooms(handle, placement="shared") == []
    assert len([record for record in caplog.records if "could not import" in record.message]) == 1
    with sqlite3.connect(store) as conn:
        assert not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='hosted_room_legacy_imports'").fetchone()
