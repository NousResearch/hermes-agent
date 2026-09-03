"""Tests for passive hosted Group Chat room replicas."""

import json
import sqlite3

import pytest

import gateway.hosted_room_replicas as replicas
import gateway.hosted_rooms as rooms

USER = {"kind": "user", "id": "tek"}
MEMBERS = [{"kind": "bot", "id": "planner"}, {"kind": "bot", "id": "coder"}]

AUTH_A = "install:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
AUTH_B = "install:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


def _authority_db(tmp_path, name="authority.db"):
    return tmp_path / name


def _replica_db(tmp_path, name="replica.db"):
    return tmp_path / name


def _seed_room(db, *, gateway_id=AUTH_A, n_events=3, room_id="room-1"):
    rooms.create_room(
        db,
        room_id=room_id,
        name="Field Room",
        members=MEMBERS,
        authority_gateway_id=gateway_id,
    )
    for index in range(n_events):
        rooms.append_event(
            db,
            room_id=room_id,
            event_id=f"e{index}",
            kind="message.user",
            actor=USER,
            payload={"text": f"msg {index} 😀"},
            authority_gateway_id=gateway_id,
            authority_epoch=1,
        )
    return rooms.read_events(db, room_id=room_id, since_seq=0, limit=100)


def test_ingest_page_persists_events_and_lineage(tmp_path):
    page = _seed_room(_authority_db(tmp_path))
    rdb = _replica_db(tmp_path)
    result = replicas.ingest_page(
        rdb, room_id="room-1", room_name="Field Room", members=MEMBERS, page=page
    )
    assert result["ingested"] == 3
    assert result["stored_seq"] == 3
    assert result["caught_up"] is True
    state = replicas.replica_state(rdb, room_id="room-1")
    assert state["last_seq"] == 3
    assert state["authority"] == page["authority"]
    assert state["members"] == MEMBERS


def test_ingest_page_is_idempotent(tmp_path):
    page = _seed_room(_authority_db(tmp_path))
    rdb = _replica_db(tmp_path)
    replicas.ingest_page(
        rdb, room_id="room-1", room_name="Field Room", members=MEMBERS, page=page
    )
    again = replicas.ingest_page(
        rdb, room_id="room-1", room_name="Field Room", members=MEMBERS, page=page
    )
    assert again["ingested"] == 0
    assert again["stored_seq"] == 3


def test_passive_replica_reserves_room_id_against_local_create(tmp_path):
    page = _seed_room(_authority_db(tmp_path))
    db = _replica_db(tmp_path)
    replicas.ingest_page(
        db, room_id="room-1", room_name="Field Room", members=MEMBERS, page=page
    )
    with pytest.raises(rooms.RoomConflictError, match="passive replica"):
        rooms.create_room(
            db,
            room_id="room-1",
            name="Field Room",
            members=MEMBERS,
            authority_gateway_id=AUTH_B,
        )


def test_database_guard_blocks_an_old_process_promoting_a_replica(tmp_path):
    page = _seed_room(_authority_db(tmp_path))
    db = _replica_db(tmp_path)
    replicas.ingest_page(
        db, room_id="room-1", room_name="Field Room", members=MEMBERS, page=page
    )
    with sqlite3.connect(db) as conn, pytest.raises(
        sqlite3.IntegrityError, match="already reserved"
    ):
        conn.execute(
            """INSERT INTO hosted_rooms
               (room_id, name, members_json, authority_gateway_id,
                authority_epoch, next_seq, event_bytes, revision,
                created_at, updated_at, disbanded_at)
               VALUES ('room-1', 'Field Room', ?, ?, 2, 4, 0, 1, 2, 2, NULL)""",
            (json.dumps(MEMBERS, separators=(",", ":")), AUTH_B),
        )


def test_disbanded_replica_room_id_cannot_be_recreated(tmp_path):
    authority_db = _authority_db(tmp_path)
    _seed_room(authority_db, n_events=1)
    rooms.disband_room(
        authority_db,
        room_id="room-1",
        expected_gateway_id=AUTH_A,
        expected_epoch=1,
    )
    page = rooms.read_events(
        authority_db, room_id="room-1", include_disbanded=True
    )
    db = _replica_db(tmp_path)
    replicas.ingest_page(
        db, room_id="room-1", room_name="Field Room", members=MEMBERS, page=page
    )
    with pytest.raises(rooms.RoomConflictError, match="passive replica"):
        rooms.create_room(
            db,
            room_id="room-1",
            name="Field Room",
            members=MEMBERS,
            authority_gateway_id=AUTH_B,
        )


def test_replica_ingest_rejects_existing_authoritative_room(tmp_path):
    page = _seed_room(_authority_db(tmp_path))
    db = _replica_db(tmp_path)
    rooms.create_room(
        db,
        room_id="room-1",
        name="Field Room",
        members=MEMBERS,
        authority_gateway_id=AUTH_B,
    )
    with pytest.raises(replicas.ReplicaError, match="locally authoritative"):
        replicas.ingest_page(
            db,
            room_id="room-1",
            room_name="Field Room",
            members=MEMBERS,
            page=page,
        )


@pytest.mark.parametrize(
    ("kind", "payload", "reason"),
    [
        (
            "authority.claimed",
            {
                "authority_gateway_id": AUTH_B,
                "authority_epoch": 1,
                "previous_gateway_id": AUTH_A,
                "promoted_from_replica": True,
            },
            "unsafe_replica_promotion",
        ),
        (
            "authority.lost",
            {
                "authority_gateway_id": AUTH_B,
                "authority_epoch": 1,
                "previous_gateway_id": AUTH_A,
            },
            "unsafe_authority_demotion",
        ),
    ],
)
def test_migration_quarantines_unsafe_takeover_lineage(
    tmp_path, kind, payload, reason
):
    db = _replica_db(tmp_path)
    rooms.create_room(
        db,
        room_id="room-1",
        name="Field Room",
        members=MEMBERS,
        authority_gateway_id=AUTH_B,
    )
    rooms.append_event(
        db,
        room_id="room-1",
        event_id=f"unsafe-{kind}",
        kind=kind,
        actor={"kind": "system", "id": "authority-control"},
        payload=payload,
        authority_gateway_id=AUTH_B,
        authority_epoch=1,
    )
    with sqlite3.connect(db) as conn:
        conn.execute("DROP TABLE hosted_room_quarantine")

    with pytest.raises(rooms.RoomQuarantinedError, match="read-only"):
        rooms.room_state(db, room_id="room-1")
    listed = rooms.list_rooms(db)
    assert listed[0]["safety_status"] == "authority_quarantined"
    assert listed[0]["safety_reason"] == reason
    with pytest.raises(rooms.RoomQuarantinedError):
        rooms.append_event(
            db,
            room_id="room-1",
            event_id="blocked",
            kind="message.user",
            actor=USER,
            payload={"text": "must not commit"},
            authority_gateway_id=AUTH_B,
            authority_epoch=1,
        )


def test_database_guard_quarantines_a_late_old_process_demotion(tmp_path):
    db = _replica_db(tmp_path)
    rooms.create_room(
        db,
        room_id="room-1",
        name="Field Room",
        members=MEMBERS,
        authority_gateway_id=AUTH_A,
    )
    actor = json.dumps(
        {"kind": "system", "id": "authority-control"},
        separators=(",", ":"),
        sort_keys=True,
    )
    payload = json.dumps(
        {
            "previous_gateway_id": AUTH_A,
            "authority_gateway_id": AUTH_B,
            "authority_epoch": 2,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    with sqlite3.connect(db) as conn:
        conn.execute(
            """INSERT INTO hosted_room_events
               (room_id, seq, event_id, kind, actor_json, authority_epoch,
                payload_json, created_at)
               VALUES ('room-1', 1, 'old-demotion', 'authority.lost', ?, 2, ?, 2)""",
            (actor, payload),
        )
        conn.execute(
            """UPDATE hosted_rooms
                  SET authority_gateway_id=?, authority_epoch=2, next_seq=2
                WHERE room_id='room-1'""",
            (AUTH_B,),
        )
        with pytest.raises(sqlite3.IntegrityError, match="quarantined"):
            conn.execute(
                """INSERT INTO hosted_room_events
                   (room_id, seq, event_id, kind, actor_json, authority_epoch,
                    payload_json, created_at)
                   VALUES ('room-1', 2, 'late-write', 'message.user', ?, 2,
                           '{"text":"unsafe"}', 3)""",
                (json.dumps(USER, separators=(",", ":"), sort_keys=True),),
            )
    with pytest.raises(rooms.RoomQuarantinedError):
        rooms.room_state(db, room_id="room-1")


def test_ingest_rejects_sequence_gap(tmp_path):
    adb = _authority_db(tmp_path)
    _seed_room(adb, n_events=5)
    later = rooms.read_events(adb, room_id="room-1", since_seq=2, limit=100)
    rdb = _replica_db(tmp_path)
    with pytest.raises(replicas.ReplicaGapError):
        replicas.ingest_page(
            rdb,
            room_id="room-1",
            room_name="Field Room",
            members=MEMBERS,
            page=later,
        )


def test_ingest_rejects_conflicting_overlap(tmp_path):
    page = _seed_room(_authority_db(tmp_path))
    rdb = _replica_db(tmp_path)
    replicas.ingest_page(
        rdb, room_id="room-1", room_name="Field Room", members=MEMBERS, page=page
    )
    conflicting = json.loads(json.dumps(page))
    conflicting["events"][0]["payload"]["text"] = "rewritten"
    with pytest.raises(replicas.ReplicaError, match="conflicts"):
        replicas.ingest_page(
            rdb,
            room_id="room-1",
            room_name="Field Room",
            members=MEMBERS,
            page=conflicting,
        )


def test_ingest_rejects_duplicate_event_ids_in_one_page(tmp_path):
    page = _seed_room(_authority_db(tmp_path))
    page["events"][1]["event_id"] = page["events"][0]["event_id"]
    with pytest.raises(replicas.ReplicaError, match="repeats an event_id"):
        replicas.ingest_page(
            _replica_db(tmp_path),
            room_id="room-1",
            room_name="Field Room",
            members=MEMBERS,
            page=page,
        )


def test_ingest_rejects_same_epoch_gateway_substitution(tmp_path):
    page = _seed_room(_authority_db(tmp_path))
    rdb = _replica_db(tmp_path)
    replicas.ingest_page(
        rdb, room_id="room-1", room_name="Field Room", members=MEMBERS, page=page
    )
    substituted = json.loads(json.dumps(page))
    substituted["authority"]["gateway_id"] = AUTH_B
    with pytest.raises(
        replicas.ReplicaLineageUnverifiedError, match="authority"
    ) as raised:
        replicas.ingest_page(
            rdb,
            room_id="room-1",
            room_name="Field Room",
            members=MEMBERS,
            page=substituted,
        )
    assert raised.value.reason == "replica_lineage_unverified"


def test_ingest_rejects_epoch_jump_without_claim(tmp_path):
    page = _seed_room(_authority_db(tmp_path))
    jumped = json.loads(json.dumps(page))
    jumped["authority"] = {"gateway_id": AUTH_B, "epoch": 3}
    with pytest.raises(replicas.ReplicaError, match="lineage"):
        replicas.ingest_page(
            _replica_db(tmp_path),
            room_id="room-1",
            room_name="Field Room",
            members=MEMBERS,
            page=jumped,
        )


def test_fresh_replica_reports_unverified_later_epoch_lineage(tmp_path):
    authority_db = _authority_db(tmp_path)
    _seed_room(authority_db, n_events=1)
    rooms.claim_authority(
        authority_db,
        room_id="room-1",
        expected_gateway_id=AUTH_A,
        expected_epoch=1,
        new_gateway_id=AUTH_B,
        event_id="claim-b",
    )
    page = rooms.read_events(authority_db, room_id="room-1", limit=100)

    with pytest.raises(
        replicas.ReplicaLineageUnverifiedError, match="first authority epoch"
    ) as raised:
        replicas.ingest_page(
            _replica_db(tmp_path),
            room_id="room-1",
            room_name="Field Room",
            members=MEMBERS,
            page=page,
        )
    assert raised.value.reason == "replica_lineage_unverified"


def test_ingest_rejects_latest_seq_regression(tmp_path):
    page = _seed_room(_authority_db(tmp_path), n_events=4)
    rdb = _replica_db(tmp_path)
    replicas.ingest_page(
        rdb, room_id="room-1", room_name="Field Room", members=MEMBERS, page=page
    )
    stale = json.loads(json.dumps(page))
    stale["latest_seq"] = 2
    stale["cursor"] = 2
    stale["events"] = stale["events"][:2]
    stale["has_more"] = False
    with pytest.raises(replicas.ReplicaError, match="regress"):
        replicas.ingest_page(
            rdb,
            room_id="room-1",
            room_name="Field Room",
            members=MEMBERS,
            page=stale,
        )


def test_ingest_rejects_inconsistent_page_cursor(tmp_path):
    page = _seed_room(_authority_db(tmp_path))
    page["cursor"] -= 1
    with pytest.raises(replicas.ReplicaError, match="cursor"):
        replicas.ingest_page(
            _replica_db(tmp_path),
            room_id="room-1",
            room_name="Field Room",
            members=MEMBERS,
            page=page,
        )


def test_ingest_rejects_non_verbatim_oversized_page(tmp_path):
    page = _seed_room(_authority_db(tmp_path), n_events=1)
    template = page["events"][0]
    page["events"] = [
        {
            **template,
            "seq": index,
            "event_id": f"event-{index}",
        }
        for index in range(1, rooms.MAX_LOG_LIMIT + 2)
    ]
    page["cursor"] = len(page["events"])
    page["latest_seq"] = len(page["events"])
    with pytest.raises(replicas.ReplicaError, match="cannot exceed"):
        replicas.ingest_page(
            _replica_db(tmp_path),
            room_id="room-1",
            room_name="Field Room",
            members=MEMBERS,
            page=page,
        )


def test_ingest_requires_authority_stamp(tmp_path):
    page = _seed_room(_authority_db(tmp_path))
    page.pop("authority")
    with pytest.raises(replicas.ReplicaError):
        replicas.ingest_page(
            _replica_db(tmp_path),
            room_id="room-1",
            room_name="Field Room",
            members=MEMBERS,
            page=page,
        )


def test_partial_replica_remains_passive_and_reports_coverage(tmp_path):
    adb = _authority_db(tmp_path)
    _seed_room(adb, n_events=5)
    partial = rooms.read_events(adb, room_id="room-1", since_seq=0, limit=2)
    rdb = _replica_db(tmp_path)
    result = replicas.ingest_page(
        rdb, room_id="room-1", room_name="Field Room", members=MEMBERS, page=partial
    )
    assert result["caught_up"] is False
    state = replicas.replica_state(rdb, room_id="room-1")
    assert state["last_seq"] == 2
    assert state["latest_seq"] == 5
    assert not hasattr(replicas, "promote_replica")
    assert not hasattr(replicas, "demote_room")


def test_disbanded_room_replica_keeps_terminal_state(tmp_path):
    adb = _authority_db(tmp_path)
    _seed_room(adb, n_events=1)
    rooms.disband_room(
        adb,
        room_id="room-1",
        expected_gateway_id=AUTH_A,
        expected_epoch=1,
    )
    page = rooms.read_events(
        adb,
        room_id="room-1",
        since_seq=0,
        limit=100,
        include_disbanded=True,
    )
    rdb = _replica_db(tmp_path)
    replicas.ingest_page(
        rdb, room_id="room-1", room_name="Field Room", members=MEMBERS, page=page
    )
    assert replicas.replica_state(rdb, room_id="room-1")["disbanded_at"] is not None


def test_ingest_rejects_terminal_event_before_source_history_is_complete(tmp_path):
    page = _terminal_page(tmp_path, room_id="room-1")
    page["latest_seq"] += 1
    page["has_more"] = True
    with pytest.raises(replicas.ReplicaError, match="complete the source history"):
        replicas.ingest_page(
            _replica_db(tmp_path),
            room_id="room-1",
            room_name="Field Room",
            members=MEMBERS,
            page=page,
        )


def test_replica_storage_shares_the_gateway_event_budget(tmp_path, monkeypatch):
    page = _seed_room(_authority_db(tmp_path))
    monkeypatch.setattr(replicas, "MAX_REPLICA_EVENT_BYTES", 0)
    with pytest.raises(replicas.ReplicaError, match="storage exhausted"):
        replicas.ingest_page(
            _replica_db(tmp_path),
            room_id="room-1",
            room_name="Field Room",
            members=MEMBERS,
            page=page,
        )


def test_authoritative_append_also_counts_replica_bytes(tmp_path, monkeypatch):
    page = _seed_room(_authority_db(tmp_path))
    db = _replica_db(tmp_path)
    replicas.ingest_page(
        db, room_id="room-1", room_name="Field Room", members=MEMBERS, page=page
    )
    rooms.create_room(
        db,
        room_id="local-room",
        name="Local",
        members=MEMBERS,
        authority_gateway_id=AUTH_B,
    )
    with sqlite3.connect(db) as conn:
        replica_bytes = int(
            conn.execute(
                "SELECT SUM(event_bytes) FROM hosted_room_replicas"
            ).fetchone()[0]
        )
    monkeypatch.setattr(rooms, "MAX_GATEWAY_EVENT_BYTES", replica_bytes)
    with pytest.raises(rooms.HostedRoomError, match="storage is full"):
        rooms.append_event(
            db,
            room_id="local-room",
            event_id="would-overflow",
            kind="message.user",
            actor=USER,
            payload={"text": "one byte too many"},
            authority_gateway_id=AUTH_B,
            authority_epoch=1,
        )


def test_replica_budget_matches_the_authoritative_store():
    assert replicas.MAX_REPLICA_EVENT_BYTES == rooms.MAX_GATEWAY_EVENT_BYTES


def _terminal_page(tmp_path, *, room_id: str, now: float = 0):
    authority_db = tmp_path / f"authority-{room_id}.db"
    _seed_room(authority_db, n_events=1, room_id=room_id)
    rooms.disband_room(
        authority_db,
        room_id=room_id,
        expected_gateway_id=AUTH_A,
        expected_epoch=1,
        now=now,
    )
    return rooms.read_events(
        authority_db,
        room_id=room_id,
        include_disbanded=True,
    )


def test_aged_terminal_replicas_free_room_slots_but_keep_ids(
    tmp_path, monkeypatch
):
    db = _replica_db(tmp_path)
    monkeypatch.setattr(replicas, "MAX_REPLICA_ROOMS", 2)
    monkeypatch.setattr(rooms, "DISBANDED_REPLICA_RETENTION_SECONDS", 1)
    for room_id in ("old-1", "old-2"):
        replicas.ingest_page(
            db,
            room_id=room_id,
            room_name="Field Room",
            members=MEMBERS,
            page=_terminal_page(tmp_path, room_id=room_id),
            now=0,
        )

    fresh_page = _seed_room(
        tmp_path / "authority-fresh.db", room_id="fresh", n_events=1
    )
    replicas.ingest_page(
        db,
        room_id="fresh",
        room_name="Field Room",
        members=MEMBERS,
        page=fresh_page,
        now=2,
    )
    with pytest.raises(
        replicas.ReplicaHistoryExpiredError, match="history expired"
    ):
        replicas.replica_state(db, room_id="old-1")
    with pytest.raises(rooms.RoomConflictError, match="retired passive replica"):
        rooms.create_room(
            db,
            room_id="old-1",
            name="Field Room",
            members=MEMBERS,
            authority_gateway_id=AUTH_B,
        )


def test_fresh_terminal_replicas_yield_slots_under_count_pressure(
    tmp_path, monkeypatch
):
    db = _replica_db(tmp_path)
    monkeypatch.setattr(replicas, "MAX_REPLICA_ROOMS", 2)
    for room_id in ("recent-1", "recent-2"):
        replicas.ingest_page(
            db,
            room_id=room_id,
            room_name="Field Room",
            members=MEMBERS,
            page=_terminal_page(tmp_path, room_id=room_id, now=10),
            now=10,
        )
    fresh_page = _seed_room(
        tmp_path / "authority-current.db", room_id="current", n_events=1
    )
    replicas.ingest_page(
        db,
        room_id="current",
        room_name="Field Room",
        members=MEMBERS,
        page=fresh_page,
        now=10,
    )
    with pytest.raises(replicas.ReplicaHistoryExpiredError):
        replicas.replica_state(db, room_id="recent-1")
    with sqlite3.connect(db) as conn, pytest.raises(
        sqlite3.IntegrityError, match="already reserved"
    ):
        conn.execute(
            """INSERT INTO hosted_rooms
               (room_id, name, members_json, authority_gateway_id,
                authority_epoch, next_seq, event_bytes, revision,
                created_at, updated_at, disbanded_at)
               VALUES ('recent-1', 'Field Room', ?, ?, 2, 1, 0, 1, 3, 3, NULL)""",
            (json.dumps(MEMBERS, separators=(",", ":")), AUTH_B),
        )


def test_authoritative_append_reclaims_terminal_replica_bytes(
    tmp_path, monkeypatch
):
    db = _replica_db(tmp_path)
    replicas.ingest_page(
        db,
        room_id="old",
        room_name="Field Room",
        members=MEMBERS,
        page=_terminal_page(tmp_path, room_id="old"),
        now=0,
    )
    with sqlite3.connect(db) as conn:
        replica_bytes = int(
            conn.execute(
                "SELECT SUM(event_bytes) FROM hosted_room_replicas"
            ).fetchone()[0]
        )
    rooms.create_room(
        db,
        room_id="local-room",
        name="Local",
        members=MEMBERS,
        authority_gateway_id=AUTH_B,
    )
    monkeypatch.setattr(rooms, "MAX_GATEWAY_EVENT_BYTES", replica_bytes)
    event = rooms.append_event(
        db,
        room_id="local-room",
        event_id="after-pressure",
        kind="message.user",
        actor=USER,
        payload={"text": "still writable"},
        authority_gateway_id=AUTH_B,
        authority_epoch=1,
    )
    assert event["seq"] == 1
    with pytest.raises(replicas.ReplicaHistoryExpiredError):
        replicas.replica_state(db, room_id="old")
    with pytest.raises(rooms.RoomConflictError, match="retired passive replica"):
        rooms.create_room(
            db,
            room_id="old",
            name="Field Room",
            members=MEMBERS,
            authority_gateway_id=AUTH_B,
        )


def test_pre_terminal_state_replica_schema_migrates_in_place(tmp_path):
    db = _replica_db(tmp_path)
    with sqlite3.connect(db) as conn:
        conn.execute(
            """CREATE TABLE hosted_room_replicas (
                room_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                members_json TEXT NOT NULL,
                authority_gateway_id TEXT NOT NULL,
                authority_epoch INTEGER NOT NULL,
                last_seq INTEGER NOT NULL,
                latest_seq INTEGER NOT NULL,
                event_bytes INTEGER NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )"""
        )
        conn.execute(
            """INSERT INTO hosted_room_replicas VALUES
               ('room-1', 'Field Room', ?, ?, 1, 0, 0, 0, 1.0, 1.0)""",
            (json.dumps(MEMBERS, separators=(",", ":")), AUTH_A),
        )
    state = replicas.replica_state(db, room_id="room-1")
    assert state["disbanded_at"] is None


def test_schema_migration_recovers_existing_disband_tombstone(tmp_path):
    authority_db = _authority_db(tmp_path)
    _seed_room(authority_db, n_events=1)
    rooms.disband_room(
        authority_db,
        room_id="room-1",
        expected_gateway_id=AUTH_A,
        expected_epoch=1,
        now=42,
    )
    page = rooms.read_events(
        authority_db,
        room_id="room-1",
        include_disbanded=True,
    )
    db = _replica_db(tmp_path)
    replicas.ingest_page(
        db, room_id="room-1", room_name="Field Room", members=MEMBERS, page=page
    )
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE hosted_room_replicas SET disbanded_at=NULL")
    assert replicas.replica_state(db, room_id="room-1")["disbanded_at"] == 42


def test_schema_migration_quarantines_post_tombstone_history(tmp_path):
    db = _replica_db(tmp_path)
    actor = json.dumps(USER, separators=(",", ":"), sort_keys=True)
    system_actor = json.dumps(
        {"kind": "system", "id": "room-control"},
        separators=(",", ":"),
        sort_keys=True,
    )
    with sqlite3.connect(db) as conn:
        conn.execute(
            """CREATE TABLE hosted_room_replicas (
                room_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                members_json TEXT NOT NULL,
                authority_gateway_id TEXT NOT NULL,
                authority_epoch INTEGER NOT NULL,
                last_seq INTEGER NOT NULL,
                latest_seq INTEGER NOT NULL,
                event_bytes INTEGER NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )"""
        )
        conn.execute(
            """CREATE TABLE hosted_room_replica_events (
                room_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                event_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                actor_json TEXT NOT NULL,
                authority_epoch INTEGER,
                payload_json TEXT NOT NULL,
                created_at REAL NOT NULL,
                PRIMARY KEY (room_id, seq)
            )"""
        )
        conn.execute(
            """INSERT INTO hosted_room_replicas VALUES
               ('room-1', 'Field Room', ?, ?, 1, 2, 2, 0, 1.0, 2.0)""",
            (json.dumps(MEMBERS, separators=(",", ":")), AUTH_A),
        )
        conn.executemany(
            """INSERT INTO hosted_room_replica_events VALUES
               ('room-1', ?, ?, ?, ?, 1, ?, ?)""",
            [
                (1, "disband", "room.disbanded", system_actor, "{}", 1.0),
                (2, "later", "message.user", actor, '{"text":"later"}', 2.0),
            ],
        )
    state = replicas.replica_state(db, room_id="room-1")
    assert state["safety_status"] == "quarantined"
    assert state["safety_reason"] == "events_after_disband"
    assert state["event_bytes"] > 0
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        assert rooms._prune_disbanded_replicas_locked(  # noqa: SLF001
            conn,
            now=None,
            max_replica_event_bytes=0,
            max_replica_rooms=0,
        ) == 0
    assert replicas.replica_state(db, room_id="room-1")["safety_status"] == (
        "quarantined"
    )


def test_each_replica_transaction_audits_late_old_process_writes(tmp_path):
    db = _replica_db(tmp_path)
    page = _seed_room(_authority_db(tmp_path), n_events=2)
    replicas.ingest_page(
        db, room_id="room-1", room_name="Field Room", members=MEMBERS, page=page
    )

    with sqlite3.connect(db) as conn:
        original = conn.execute(
            """SELECT event_id, kind, actor_json, authority_epoch,
                      payload_json, created_at
                 FROM hosted_room_replica_events
                WHERE room_id='room-1' AND seq=1"""
        ).fetchone()
        conn.execute(
            """INSERT INTO hosted_room_replica_events(
                   room_id, seq, event_id, kind, actor_json, authority_epoch,
                   payload_json, created_at
               ) VALUES ('room-1', 3, ?, ?, ?, ?, ?, ?)""",
            original,
        )
        conn.execute(
            """UPDATE hosted_room_replicas
                  SET last_seq=3, latest_seq=3
                WHERE room_id='room-1'"""
        )

    state = replicas.replica_state(db, room_id="room-1")
    assert state["safety_status"] == "quarantined"
    assert state["safety_reason"] == "duplicate_event_id"


def test_late_old_process_gateway_actor_must_match_replica_authority(tmp_path):
    db = _replica_db(tmp_path)
    page = _seed_room(_authority_db(tmp_path), n_events=2)
    replicas.ingest_page(
        db, room_id="room-1", room_name="Field Room", members=MEMBERS, page=page
    )

    with sqlite3.connect(db) as conn:
        conn.execute(
            """INSERT INTO hosted_room_replica_events(
                   room_id, seq, event_id, kind, actor_json, authority_epoch,
                   payload_json, created_at
               ) VALUES ('room-1', 3, 'old-gateway-write', 'room.activity',
                         ?, 1, '{}', 3.0)""",
            (
                json.dumps(
                    {"kind": "gateway", "id": AUTH_B},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ),
        )
        conn.execute(
            """UPDATE hosted_room_replicas
                  SET last_seq=3, latest_seq=3
                WHERE room_id='room-1'"""
        )

    state = replicas.replica_state(db, room_id="room-1")
    assert state["safety_status"] == "quarantined"
    assert state["safety_reason"] == "gateway_actor_authority_mismatch"


def test_late_old_process_cannot_assert_a_later_epoch_without_proof(tmp_path):
    db = _replica_db(tmp_path)
    page = _seed_room(_authority_db(tmp_path), n_events=2)
    replicas.ingest_page(
        db, room_id="room-1", room_name="Field Room", members=MEMBERS, page=page
    )

    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE hosted_room_replicas SET authority_epoch=2 WHERE room_id='room-1'"
        )
        conn.execute(
            """UPDATE hosted_room_replica_events
                  SET authority_epoch=2 WHERE room_id='room-1'"""
        )

    state = replicas.replica_state(db, room_id="room-1")
    assert state["safety_status"] == "quarantined"
    assert state["safety_reason"] == "unverified_authority_epoch"


def test_sharded_store_preserves_public_limit_overrides(tmp_path, monkeypatch):
    db = _replica_db(tmp_path)
    rooms.create_room(
        db,
        room_id="room-1",
        name="Room",
        members=MEMBERS,
        authority_gateway_id=AUTH_A,
    )

    monkeypatch.setattr(rooms, "MAX_ROOM_NAME_CHARS", 3)
    with pytest.raises(rooms.HostedRoomError, match="invalid room name"):
        rooms.create_room(
            db,
            room_id="room-2",
            name="Long name",
            members=MEMBERS,
            authority_gateway_id=AUTH_A,
        )

    monkeypatch.setattr(rooms, "MAX_EVENTS_PER_ROOM", 0)
    monkeypatch.setattr(rooms, "CONTROL_EVENT_COUNT_RESERVE", 0)
    monkeypatch.setattr(rooms, "CONTROL_EVENT_BYTE_RESERVE", 0)
    with pytest.raises(rooms.HostedRoomError, match="history limit"):
        rooms.disband_room(
            db,
            room_id="room-1",
            expected_gateway_id=AUTH_A,
            expected_epoch=1,
        )
