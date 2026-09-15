"""Focused replica-correctness regressions preserved from PR #19."""

import json
import sqlite3
import pytest
import gateway.hosted_room_driver as driver
import gateway.hosted_room_replicas as replicas
import gateway.hosted_rooms as rooms
from gateway.hosted_rooms_common import utf8_len


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


def test_ingest_preserves_monotonic_latest_seq_from_delayed_page(tmp_path, monkeypatch):
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


def test_replica_rejects_different_owner_at_same_epoch_without_mutation(tmp_path):
    page = _seed_room(_authority_db(tmp_path), n_events=2)
    rdb = _replica_db(tmp_path)
    replicas.ingest_page(
        rdb, room_id="room-1", room_name="Field Room", members=MEMBERS, page=page)
    before = replicas.replica_state(rdb, room_id="room-1")
    conflicting_page = {
        **page, "events": [], "authority": {"gateway_id": AUTH_B, "epoch": page["authority"]["epoch"]}}
    with pytest.raises(replicas.ReplicaError, match="owner conflicts"):
        replicas.ingest_page(
            rdb, room_id="room-1", room_name="Changed", members=MEMBERS, page=conflicting_page)
    assert replicas.replica_state(rdb, room_id="room-1") == before


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


def test_promote_replays_history_that_validly_used_stop_reserve(tmp_path, monkeypatch):
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
    lost_bytes = utf8_len(
        lost["event_id"],
        lost["kind"],
        rooms._canonical_json(
            lost["actor"], label="actor", max_bytes=4 * 1024
        ),
        rooms._canonical_json(
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


@pytest.mark.parametrize("target_member_id", ["member-a", "member-b", None])
def test_same_profile_terminal_requires_exact_member_before_demotion(tmp_path, monkeypatch, target_member_id):
    db = _authority_db(tmp_path)
    rooms.create_room(
        db, room_id="room-1", name="Same profile, distinct targets",
        members=[
            {"member_id": "member-a", "profile": "ops", "handle": "local",
             "target": {"kind": "local", "profile": "ops"}},
            {"member_id": "member-b", "profile": "ops", "handle": "remote",
             "target": {"kind": "peer", "profile": "ops", "peer_id": "peer-b",
                        "installation_id": "install-b", "capability_digest": "a" * 64}}],
        authority_gateway_id=AUTH_A)
    source = rooms.append_event(
        db, room_id="room-1", event_id="source", kind="message.user", actor=USER,
        payload={"text": "@local inspect", "thread_id": "thread-1"},
        authority_gateway_id=AUTH_A, authority_epoch=1, require_open_admissions=True)
    identity = driver.TaskIdentity("room-1", "task-exact-member", "thread-1", "turn-1")
    task_payload = {"target_profile": "ops", "prompt": "inspect", "source_event_seq": source["seq"]}
    if target_member_id is not None:
        task_payload["target_member_id"] = target_member_id
    driver.admit_task(db, identity, payload=task_payload, clock=lambda: 100)
    driver.cancel_task(db, identity, cancel_id="cancel", expected_cancel_generation=0, clock=lambda: 100)
    wrong_member = "member-a" if target_member_id == "member-b" else "member-b"
    terminal = {
        "discussion_event_id": source["event_id"], "member_id": wrong_member,
        "member_index": 0, "round_index": 0, "seen_through_seq": source["seq"],
        "task_id": identity.task_id, "thread_id": identity.thread_id, "turn_id": identity.turn_id,
        "reason": "cancelled"}
    rooms.append_event(
        db, room_id="room-1", event_id="wrong-member", kind="turn.cancelled",
        actor={"kind": "gateway", "id": AUTH_A}, payload=terminal,
        authority_gateway_id=AUTH_A, authority_epoch=1)
    rooms.append_event(
        db, room_id="room-1", event_id="closed", kind="room.activity",
        actor={"kind": "gateway", "id": AUTH_A},
        payload={"status": "settled", "reason_code": "complete", "thread_id": "thread-1",
                 "discussion_event_id": source["event_id"]},
        authority_gateway_id=AUTH_A, authority_epoch=1)
    monkeypatch.setattr(replicas, "local_authority_gateway_id", lambda: AUTH_A)
    before = rooms.read_events(db, room_id="room-1")
    with pytest.raises(replicas.ReplicaError, match="unpublished terminal"):
        replicas.demote_room(db, room_id="room-1", observed_gateway_id=AUTH_B, observed_epoch=2)
    assert rooms.read_events(db, room_id="room-1") == before
    with rooms._connect(db) as conn:
        assert rooms._terminal_publication_liabilities(conn) == {("room-1", identity.task_id)}
    if target_member_id is not None:
        rooms.append_event(
            db, room_id="room-1", event_id="exact-member", kind="turn.cancelled",
            actor={"kind": "gateway", "id": AUTH_A}, payload={**terminal, "member_id": target_member_id},
            authority_gateway_id=AUTH_A, authority_epoch=1)
        assert replicas.demote_room(
            db, room_id="room-1", observed_gateway_id=AUTH_B, observed_epoch=2)["authority_epoch"] == 2


@pytest.mark.parametrize(
    ("members", "target_member_id", "published"),
    [
        ([{"id": "ops"}], None, True),
        ([{"member_id": "ops", "profile": "ops"}], None, True),
        ([{"member_id": "ops", "profile": "ops", "target": {"kind": "local", "profile": "other"}}],
         "ops", False),
        ([{"member_id": "ops", "profile": "ops", "target": {"kind": [], "profile": "ops"}}],
         "ops", False),
        ([{"member_id": "ops", "profile": "ops"}, {"member_id": "ops", "profile": "ops"}], "ops", False),
    ],
    ids=["legacy-id", "legacy-profile", "mismatched-target", "malformed-target", "ambiguous-member-id"],
)
def test_terminal_identity_preserves_unambiguous_legacy_rows_only(tmp_path, members, target_member_id, published):
    db = _authority_db(tmp_path)
    rooms.create_room(db, room_id="room-1", name="Legacy identity", members=members, authority_gateway_id=AUTH_A)
    source = rooms.append_event(
        db, room_id="room-1", event_id="source", kind="message.user", actor=USER,
        payload={"text": "inspect", "thread_id": "thread-1"}, authority_gateway_id=AUTH_A, authority_epoch=1)
    identity = driver.TaskIdentity("room-1", "task", "thread-1", "turn-1")
    payload = {"target_profile": "ops", "prompt": "inspect", "source_event_seq": source["seq"]}
    if target_member_id is not None:
        payload["target_member_id"] = target_member_id
    driver.admit_task(db, identity, payload=payload, clock=lambda: 100)
    driver.cancel_task(db, identity, cancel_id="cancel", expected_cancel_generation=0, clock=lambda: 100)
    rooms.append_event(
        db, room_id="room-1", event_id="terminal", kind="turn.cancelled",
        actor={"kind": "gateway", "id": AUTH_A}, authority_gateway_id=AUTH_A, authority_epoch=1,
        payload={"discussion_event_id": source["event_id"], "member_id": "ops", "member_index": 0,
                 "round_index": 0, "seen_through_seq": source["seq"], "task_id": identity.task_id,
                 "thread_id": identity.thread_id, "turn_id": identity.turn_id, "reason": "cancelled"})
    rooms.append_event(
        db, room_id="room-1", event_id="close", kind="room.activity",
        actor={"kind": "gateway", "id": AUTH_A}, authority_gateway_id=AUTH_A, authority_epoch=1,
        payload={"status": "settled", "reason_code": "complete", "thread_id": identity.thread_id,
                 "discussion_event_id": source["event_id"]})
    with rooms._connect(db) as conn:
        liabilities = rooms._terminal_publication_liabilities(conn)
    assert liabilities == (set() if published else {("room-1", identity.task_id)})
    assert rooms.room_state(db, room_id="room-1")["members"] == members
    assert driver.get_task(db, identity)["payload"] == payload


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
    assert (
        rooms.read_events(adb, room_id="room-1", since_seq=0, limit=100)["events"] == []
    )
