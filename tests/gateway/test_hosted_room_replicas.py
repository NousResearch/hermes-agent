"""Tests for gateway/hosted_room_replicas.py — replica ingest, promotion, and
stale-authority demotion for hosted Group Chat rooms."""

import json
import sqlite3

import pytest

import gateway.hosted_room_driver as driver
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


def test_ingest_preserves_monotonic_latest_seq_from_delayed_page(
    tmp_path, monkeypatch
):
    adb = _authority_db(tmp_path)
    _seed_room(adb, n_events=2)
    first_page = rooms.read_events(adb, room_id="room-1", since_seq=0, limit=1)
    assert first_page["latest_seq"] == 2

    rdb = _replica_db(tmp_path)
    replicas.ingest_page(
        rdb,
        room_id="room-1",
        room_name="Field Room",
        members=MEMBERS,
        page=first_page,
    )
    delayed_page = json.loads(json.dumps(first_page))
    delayed_page["latest_seq"] = 1
    replicas.ingest_page(
        rdb,
        room_id="room-1",
        room_name="Field Room",
        members=MEMBERS,
        page=delayed_page,
    )

    state = replicas.replica_state(rdb, room_id="room-1")
    assert state["last_seq"] == 1
    assert state["latest_seq"] == 2
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_B)
    with pytest.raises(replicas.ReplicaError, match="caught up"):
        replicas.promote_replica(rdb, room_id="room-1")


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


def test_ingest_rejects_epoch_regression(tmp_path):
    page = _seed_room(_authority_db(tmp_path))
    rdb = _replica_db(tmp_path)
    newer = json.loads(json.dumps(page))
    newer["authority"]["epoch"] = 3
    replicas.ingest_page(
        rdb, room_id="room-1", room_name="Field Room", members=MEMBERS, page=newer
    )
    stale = json.loads(json.dumps(page))
    stale["authority"]["epoch"] = 2
    with pytest.raises(replicas.ReplicaEpochRegressionError):
        replicas.ingest_page(
            rdb,
            room_id="room-1",
            room_name="Field Room",
            members=MEMBERS,
            page=stale,
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


def test_promote_replica_continues_room_at_next_epoch(tmp_path, monkeypatch):
    page = _seed_room(_authority_db(tmp_path))
    rdb = _replica_db(tmp_path)
    replicas.ingest_page(
        rdb, room_id="room-1", room_name="Field Room", members=MEMBERS, page=page
    )
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_B)

    promoted = replicas.promote_replica(rdb, room_id="room-1")
    assert promoted["authority_gateway_id"] == AUTH_B
    assert promoted["authority_epoch"] == 2
    assert promoted["previous_gateway_id"] == AUTH_A
    assert promoted["claim_seq"] == 4

    # The room is now locally authoritative with the full history + claim.
    replay = rooms.read_events(rdb, room_id="room-1", since_seq=0, limit=100)
    assert [e["seq"] for e in replay["events"]] == [1, 2, 3, 4]
    claim = replay["events"][-1]
    assert claim["kind"] == "authority.claimed"
    assert claim["payload"]["previous_gateway_id"] == AUTH_A
    assert claim["payload"]["authority_epoch"] == 2
    assert replay["authority"] == {"gateway_id": AUTH_B, "epoch": 2}

    # New work continues under the new epoch.
    rooms.append_event(
        rdb,
        room_id="room-1",
        event_id="post-takeover",
        kind="message.user",
        actor=USER,
        payload={"text": "continuing"},
        authority_gateway_id=AUTH_B,
        authority_epoch=2,
    )

    # The old authority's identity/epoch is fenced out.
    with pytest.raises(rooms.HostedRoomError):
        rooms.append_event(
            rdb,
            room_id="room-1",
            event_id="stale-write",
            kind="message.user",
            actor=USER,
            payload={"text": "stale"},
            authority_gateway_id=AUTH_A,
            authority_epoch=1,
        )

    # Replica bookkeeping is consumed by promotion.
    with pytest.raises(replicas.ReplicaError):
        replicas.replica_state(rdb, room_id="room-1")


def test_promote_refuses_replica_that_has_not_caught_up(tmp_path, monkeypatch):
    adb = _authority_db(tmp_path)
    _seed_room(adb, n_events=200)
    first_page = rooms.read_events(adb, room_id="room-1", since_seq=0, limit=100)
    assert first_page["latest_seq"] == 200
    assert first_page["has_more"] is True
    rdb = _replica_db(tmp_path)
    replicas.ingest_page(
        rdb,
        room_id="room-1",
        room_name="Field Room",
        members=MEMBERS,
        page=first_page,
    )
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_B)

    with pytest.raises(replicas.ReplicaError, match="caught up"):
        replicas.promote_replica(rdb, room_id="room-1")

    assert replicas.replica_state(rdb, room_id="room-1")["last_seq"] == 100


def test_promote_preflights_authoritative_event_count(tmp_path, monkeypatch):
    page = _seed_room(_authority_db(tmp_path), n_events=2)
    rdb = _replica_db(tmp_path)
    replicas.ingest_page(
        rdb, room_id="room-1", room_name="Field Room", members=MEMBERS, page=page
    )
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_B)
    monkeypatch.setattr(rooms, "MAX_EVENTS_PER_ROOM", 1)

    with pytest.raises(rooms.HostedRoomError, match="history limit"):
        replicas.promote_replica(rdb, room_id="room-1")

    assert replicas.replica_state(rdb, room_id="room-1")["last_seq"] == 2


def test_promote_preflights_authoritative_gateway_bytes(tmp_path, monkeypatch):
    page = _seed_room(_authority_db(tmp_path), n_events=1)
    rdb = _replica_db(tmp_path)
    replicas.ingest_page(
        rdb, room_id="room-1", room_name="Field Room", members=MEMBERS, page=page
    )
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_B)
    monkeypatch.setattr(rooms, "MAX_GATEWAY_EVENT_BYTES", 1)

    with pytest.raises(rooms.HostedRoomError, match="storage is full"):
        replicas.promote_replica(rdb, room_id="room-1")

    assert replicas.replica_state(rdb, room_id="room-1")["last_seq"] == 1


def test_promote_preflights_authoritative_room_bytes(tmp_path, monkeypatch):
    page = _seed_room(_authority_db(tmp_path), n_events=1)
    rdb = _replica_db(tmp_path)
    replicas.ingest_page(
        rdb, room_id="room-1", room_name="Field Room", members=MEMBERS, page=page
    )
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_B)
    monkeypatch.setattr(rooms, "MAX_ROOM_EVENT_BYTES", 1)

    with pytest.raises(rooms.HostedRoomError, match="storage limit"):
        replicas.promote_replica(rdb, room_id="room-1")

    assert replicas.replica_state(rdb, room_id="room-1")["last_seq"] == 1


def test_promote_replays_history_that_validly_used_stop_reserve(
    tmp_path, monkeypatch
):
    adb = _authority_db(tmp_path)
    rooms.create_room(
        adb,
        room_id="room-1",
        name="Field Room",
        members=MEMBERS,
        authority_gateway_id=AUTH_A,
    )
    monkeypatch.setattr(rooms, "MAX_EVENTS_PER_ROOM", 1)
    rooms.append_event(
        adb,
        room_id="room-1",
        event_id="ordinary-1",
        kind="message.user",
        actor=USER,
        payload={"text": "at the ordinary limit"},
        authority_gateway_id=AUTH_A,
        authority_epoch=1,
    )
    rooms.request_room_stop(
        adb,
        room_id="room-1",
        cancel_id="stop-1",
        expected_gateway_id=AUTH_A,
        expected_epoch=1,
    )
    page = rooms.read_events(adb, room_id="room-1", since_seq=0, limit=100)
    rdb = _replica_db(tmp_path)
    replicas.ingest_page(
        rdb, room_id="room-1", room_name="Field Room", members=MEMBERS, page=page
    )
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_B)

    replicas.promote_replica(rdb, room_id="room-1")

    replay = rooms.read_events(rdb, room_id="room-1", since_seq=0, limit=100)
    assert [event["kind"] for event in replay["events"]] == [
        "message.user",
        "room.stop_requested",
        "authority.claimed",
    ]


def test_promote_replays_correlated_terminal_pair_with_its_reserve(
    tmp_path, monkeypatch
):
    adb = _authority_db(tmp_path)
    rooms.create_room(
        adb,
        room_id="room-1",
        name="Field Room",
        members=MEMBERS,
        authority_gateway_id=AUTH_A,
    )
    rooms.append_event(
        adb,
        room_id="room-1",
        event_id="ordinary-1",
        kind="message.user",
        actor=USER,
        payload={"text": "at the ordinary limit"},
        authority_gateway_id=AUTH_A,
        authority_epoch=1,
    )
    monkeypatch.setattr(rooms, "MAX_EVENTS_PER_ROOM", 1)
    monkeypatch.setattr(rooms, "STOP_EVENT_COUNT_RESERVE", 0)
    monkeypatch.setattr(rooms, "TERMINAL_RECOVERY_COUNT_RESERVE", 2)
    correlation = {
        "task_id": "task-1",
        "discussion_event_id": "discussion-1",
        "member_id": "planner",
        "thread_id": "thread-1",
        "turn_id": "turn-1",
    }
    rooms.append_events(
        adb,
        events=[
            {
                "room_id": "room-1",
                "event_id": "member-terminal-1",
                "kind": "message.member",
                "actor": {"kind": "member", "id": "planner"},
                "payload": correlation,
                "authority_gateway_id": AUTH_A,
                "authority_epoch": 1,
            },
            {
                "room_id": "room-1",
                "event_id": "settled-terminal-1",
                "kind": "turn.settled",
                "actor": {"kind": "gateway", "id": AUTH_A},
                "payload": {
                    **correlation,
                    "message_event_id": "member-terminal-1",
                    "passed": False,
                },
                "authority_gateway_id": AUTH_A,
                "authority_epoch": 1,
            },
        ],
        allow_terminal_recovery=True,
    )
    page = rooms.read_events(adb, room_id="room-1", since_seq=0, limit=100)
    rdb = _replica_db(tmp_path)
    replicas.ingest_page(
        rdb, room_id="room-1", room_name="Field Room", members=MEMBERS, page=page
    )
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_B)

    replicas.promote_replica(rdb, room_id="room-1")

    replay = rooms.read_events(rdb, room_id="room-1", since_seq=0, limit=100)
    assert [event["kind"] for event in replay["events"]] == [
        "message.user",
        "message.member",
        "turn.settled",
        "authority.claimed",
    ]


def test_promote_replays_closing_room_activity_with_discussion_reserve(
    tmp_path, monkeypatch
):
    adb = _authority_db(tmp_path)
    rooms.create_room(
        adb,
        room_id="room-1",
        name="Field Room",
        members=MEMBERS,
        authority_gateway_id=AUTH_A,
    )
    rooms.append_event(
        adb,
        room_id="room-1",
        event_id="discussion-1",
        kind="message.user",
        actor=USER,
        payload={"text": "at the ordinary limit", "thread_id": "thread-1"},
        authority_gateway_id=AUTH_A,
        authority_epoch=1,
    )
    monkeypatch.setattr(rooms, "MAX_EVENTS_PER_ROOM", 1)
    monkeypatch.setattr(rooms, "STOP_EVENT_COUNT_RESERVE", 0)
    monkeypatch.setattr(rooms, "TERMINAL_RECOVERY_COUNT_RESERVE", 3)
    correlation = {
        "task_id": "task-1",
        "discussion_event_id": "discussion-1",
        "member_id": "planner",
        "thread_id": "thread-1",
        "turn_id": "turn-1",
    }
    rooms.append_events(
        adb,
        events=[
            {
                "room_id": "room-1",
                "event_id": "member-terminal-1",
                "kind": "message.member",
                "actor": {"kind": "member", "id": "planner"},
                "payload": correlation,
                "authority_gateway_id": AUTH_A,
                "authority_epoch": 1,
            },
            {
                "room_id": "room-1",
                "event_id": "settled-terminal-1",
                "kind": "turn.settled",
                "actor": {"kind": "gateway", "id": AUTH_A},
                "payload": {
                    **correlation,
                    "message_event_id": "member-terminal-1",
                    "passed": False,
                },
                "authority_gateway_id": AUTH_A,
                "authority_epoch": 1,
            },
        ],
        allow_terminal_recovery=True,
    )
    rooms.append_event(
        adb,
        room_id="room-1",
        event_id="activity-settled-1",
        kind="room.activity",
        actor={"kind": "gateway", "id": AUTH_A},
        payload={
            "status": "settled",
            "reason_code": "complete",
            "thread_id": "thread-1",
            "discussion_event_id": "discussion-1",
        },
        authority_gateway_id=AUTH_A,
        authority_epoch=1,
    )
    page = rooms.read_events(adb, room_id="room-1", since_seq=0, limit=100)
    rdb = _replica_db(tmp_path)
    replicas.ingest_page(
        rdb, room_id="room-1", room_name="Field Room", members=MEMBERS, page=page
    )
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_B)

    replicas.promote_replica(rdb, room_id="room-1")

    replay = rooms.read_events(rdb, room_id="room-1", since_seq=0, limit=100)
    assert [event["kind"] for event in replay["events"]] == [
        "message.user",
        "message.member",
        "turn.settled",
        "room.activity",
        "authority.claimed",
    ]


def test_promote_refuses_when_active_room_capacity_is_full(tmp_path, monkeypatch):
    page = _seed_room(_authority_db(tmp_path), n_events=1)
    rdb = _replica_db(tmp_path)
    replicas.ingest_page(
        rdb, room_id="room-1", room_name="Field Room", members=MEMBERS, page=page
    )
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_B)
    monkeypatch.setattr(replicas, "MAX_ACTIVE_ROOMS", 0)

    with pytest.raises(rooms.HostedRoomError, match="too many active"):
        replicas.promote_replica(rdb, room_id="room-1")

    assert replicas.replica_state(rdb, room_id="room-1")["last_seq"] == 1


def test_promote_refuses_when_room_exists_locally(tmp_path, monkeypatch):
    db = _authority_db(tmp_path)
    page = _seed_room(db)
    # Same DB also holds a replica row for the same id — conflict must win.
    replicas.ingest_page(
        db, room_id="room-1", room_name="Field Room", members=MEMBERS, page=page
    )
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_B)
    with pytest.raises(rooms.RoomConflictError):
        replicas.promote_replica(db, room_id="room-1")


def test_promote_refuses_when_already_authority(tmp_path, monkeypatch):
    page = _seed_room(_authority_db(tmp_path))
    rdb = _replica_db(tmp_path)
    replicas.ingest_page(
        rdb, room_id="room-1", room_name="Field Room", members=MEMBERS, page=page
    )
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_A)
    with pytest.raises(replicas.ReplicaError):
        replicas.promote_replica(rdb, room_id="room-1")


def test_demote_fences_stale_local_authority(tmp_path, monkeypatch):
    adb = _authority_db(tmp_path)
    _seed_room(adb)
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_A)
    with sqlite3.connect(adb) as conn:
        bytes_before = conn.execute(
            "SELECT event_bytes FROM hosted_rooms WHERE room_id='room-1'"
        ).fetchone()[0]

    result = replicas.demote_room(
        adb, room_id="room-1", observed_gateway_id=AUTH_B, observed_epoch=2
    )
    assert result["idempotent"] is False
    assert result["authority_gateway_id"] == AUTH_B
    assert result["authority_epoch"] == 2

    replay = rooms.read_events(adb, room_id="room-1", since_seq=0, limit=100)
    lost = replay["events"][-1]
    assert lost["kind"] == "authority.lost"
    assert lost["payload"]["authority_gateway_id"] == AUTH_B
    assert replay["authority"] == {"gateway_id": AUTH_B, "epoch": 2}
    lost_bytes = rooms._event_storage_bytes(
        event_id=lost["event_id"],
        kind=lost["kind"],
        actor_json=rooms._canonical_json(
            lost["actor"], label="actor", max_bytes=4 * 1024
        ),
        payload_json=rooms._canonical_json(
            lost["payload"], label="payload", max_bytes=rooms.MAX_EVENT_JSON_BYTES
        ),
    )
    with sqlite3.connect(adb) as conn:
        bytes_after = conn.execute(
            "SELECT event_bytes FROM hosted_rooms WHERE room_id='room-1'"
        ).fetchone()[0]
    assert bytes_after == bytes_before + lost_bytes

    # Local sends at the stale identity/epoch are now rejected.
    with pytest.raises(rooms.HostedRoomError):
        rooms.append_event(
            adb,
            room_id="room-1",
            event_id="after-demote",
            kind="message.user",
            actor=USER,
            payload={"text": "stale"},
            authority_gateway_id=AUTH_A,
            authority_epoch=1,
        )

    # Repeating the same observation is idempotent.
    again = replicas.demote_room(
        adb, room_id="room-1", observed_gateway_id=AUTH_B, observed_epoch=2
    )
    assert again["idempotent"] is True


def test_demote_rejects_unpublished_terminal_liability_without_mutation(
    tmp_path, monkeypatch
):
    adb = _authority_db(tmp_path)
    _seed_room(adb, n_events=1)
    driver.admit_task(
        adb,
        driver.TaskIdentity(
            room_id="room-1",
            task_id="task-unpublished",
            thread_id="thread-1",
            turn_id="turn-1",
        ),
        payload={
            "target_profile": "planner",
            "prompt": "Publish this task's terminal result before demotion.",
            "source_event_seq": 1,
        },
        clock=lambda: 100.0,
    )
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_A)
    before = rooms.read_events(adb, room_id="room-1", since_seq=0, limit=100)

    with pytest.raises(replicas.ReplicaError, match="unpublished terminal"):
        replicas.demote_room(
            adb, room_id="room-1", observed_gateway_id=AUTH_B, observed_epoch=2
        )

    after = rooms.read_events(adb, room_id="room-1", since_seq=0, limit=100)
    assert after["authority"] == before["authority"]
    assert after["events"] == before["events"]


@pytest.mark.parametrize(
    ("mismatched_field", "mismatched_value"),
    [
        ("discussion_event_id", "different-discussion"),
        ("member_id", "coder"),
        ("thread_id", "different-thread"),
        ("turn_id", "different-turn"),
    ],
)
def test_demote_rejects_terminal_event_with_mismatched_task_correlation(
    tmp_path,
    monkeypatch,
    mismatched_field,
    mismatched_value,
):
    adb = _authority_db(tmp_path)
    rooms.create_room(
        adb,
        room_id="room-1",
        name="Field Room",
        members=[
            {"member_id": "planner", "profile": "planner", "handle": "planner"},
            {"member_id": "coder", "profile": "coder", "handle": "coder"},
        ],
        authority_gateway_id=AUTH_A,
    )
    source = rooms.append_event(
        adb,
        room_id="room-1",
        event_id="discussion-1",
        kind="message.user",
        actor=USER,
        payload={"text": "inspect", "thread_id": "thread-1"},
        authority_gateway_id=AUTH_A,
        authority_epoch=1,
        require_open_admissions=True,
    )
    identity = driver.TaskIdentity(
        room_id="room-1",
        task_id="task-correlated",
        thread_id="thread-1",
        turn_id="turn-1",
    )
    driver.admit_task(
        adb,
        identity,
        payload={
            "target_profile": "planner",
            "prompt": "Publish this task's exact terminal result before demotion.",
            "source_event_seq": source["seq"],
        },
        clock=lambda: 100.0,
    )
    driver.cancel_task(
        adb,
        identity,
        cancel_id="cancel-correlated",
        expected_cancel_generation=0,
        clock=lambda: 100.0,
    )
    terminal_payload = {
        "discussion_event_id": "discussion-1",
        "member_id": "planner",
        "member_index": 0,
        "round_index": 0,
        "seen_through_seq": source["seq"],
        "task_id": "task-correlated",
        "thread_id": "thread-1",
        "turn_id": "turn-1",
        "reason": "forged correlation",
    }
    terminal_payload[mismatched_field] = mismatched_value
    rooms.append_event(
        adb,
        room_id="room-1",
        event_id="forged-terminal",
        kind="turn.cancelled",
        actor={"kind": "gateway", "id": AUTH_A},
        payload=terminal_payload,
        authority_gateway_id=AUTH_A,
        authority_epoch=1,
    )
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_A)
    before = rooms.read_events(adb, room_id="room-1", since_seq=0, limit=100)

    with pytest.raises(replicas.ReplicaError, match="unpublished terminal"):
        replicas.demote_room(
            adb, room_id="room-1", observed_gateway_id=AUTH_B, observed_epoch=2
        )

    after = rooms.read_events(adb, room_id="room-1", since_seq=0, limit=100)
    assert after["authority"] == before["authority"]
    assert after["events"] == before["events"]


def test_demote_preflights_authority_lost_control_capacity(tmp_path, monkeypatch):
    adb = _authority_db(tmp_path)
    _seed_room(adb, n_events=0)
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_A)
    monkeypatch.setattr(rooms, "MAX_GATEWAY_EVENT_BYTES", 0)
    monkeypatch.setattr(rooms, "CONTROL_EVENT_BYTE_RESERVE", 0)

    with pytest.raises(rooms.HostedRoomError, match="storage is full"):
        replicas.demote_room(
            adb, room_id="room-1", observed_gateway_id=AUTH_B, observed_epoch=2
        )

    state = rooms.room_state(adb, room_id="room-1")
    assert state["authority_gateway_id"] == AUTH_A
    assert state["authority_epoch"] == 1
    assert rooms.read_events(adb, room_id="room-1", since_seq=0, limit=100)[
        "events"
    ] == []


def test_demote_rejects_non_superseding_epoch(tmp_path, monkeypatch):
    adb = _authority_db(tmp_path)
    _seed_room(adb)
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_A)
    with pytest.raises(replicas.ReplicaEpochRegressionError):
        replicas.demote_room(
            adb, room_id="room-1", observed_gateway_id=AUTH_B, observed_epoch=1
        )


def test_full_failover_round_trip(tmp_path, monkeypatch):
    """Authority A hosts, replica B follows, A dies, B promotes, A returns
    and is fenced + demoted; the room's history survives intact throughout."""
    adb = _authority_db(tmp_path)
    rdb = _replica_db(tmp_path)
    page = _seed_room(adb, n_events=4)
    replicas.ingest_page(
        rdb, room_id="room-1", room_name="Field Room", members=MEMBERS, page=page
    )

    # A "dies"; B takes over.
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_B)
    promoted = replicas.promote_replica(rdb, room_id="room-1")
    rooms.append_event(
        rdb,
        room_id="room-1",
        event_id="b-work",
        kind="message.user",
        actor=USER,
        payload={"text": "work continues on B"},
        authority_gateway_id=AUTH_B,
        authority_epoch=promoted["authority_epoch"],
    )

    # A comes back, observes B's claim, and fences itself.
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_A)
    replicas.demote_room(
        adb,
        room_id="room-1",
        observed_gateway_id=AUTH_B,
        observed_epoch=promoted["authority_epoch"],
    )
    with pytest.raises(rooms.HostedRoomError):
        rooms.append_event(
            adb,
            room_id="room-1",
            event_id="a-stale",
            kind="message.user",
            actor=USER,
            payload={"text": "split brain attempt"},
            authority_gateway_id=AUTH_A,
            authority_epoch=1,
        )

    # B's room holds the complete history: 4 original + claim + new work.
    replay = rooms.read_events(rdb, room_id="room-1", since_seq=0, limit=100)
    kinds = [e["kind"] for e in replay["events"]]
    assert kinds == ["message.user"] * 4 + ["authority.claimed", "message.user"]
    assert replay["authority"]["gateway_id"] == AUTH_B
