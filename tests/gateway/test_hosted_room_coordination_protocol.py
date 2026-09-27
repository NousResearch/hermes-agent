"""SQLite extraction contract for the backend-neutral hosted-room protocol.

This is deliberately an adapter-level conformance harness over current public
SQLite APIs. It records behavior a future PostgreSQL implementation must match;
it neither selects a backend nor proves external delivery.
"""

from __future__ import annotations

import sqlite3

import pytest

from gateway import hosted_room_driver as driver
from gateway import hosted_rooms as rooms
from gateway.hosted_room_coordination import HostedRoomCoordination, sqlite_hosted_room_coordination



class FakeClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def create_room(db, *, room_id: str = "room-1"):
    return sqlite_hosted_room_coordination(db).create_room(
        room_id=room_id,
        name="Protocol room",
        members=[{"profile": "ops", "handle": "ops"}],
        authority_gateway_id="gateway-a",
        now=10,
    )


def append_user_event(db, *, event_id: str, text: str, now: float):
    return sqlite_hosted_room_coordination(db).append_event(
        room_id="room-1",
        event_id=event_id,
        kind="message.user",
        actor={"kind": "user", "id": "user-1"},
        payload={"text": text, "thread_id": "thread-1"},
        authority_gateway_id="gateway-a",
        authority_epoch=1,
        now=now,
    )


def test_room_event_contract_is_ordered_idempotent_and_fenced(tmp_path):
    db = tmp_path / "state.db"
    created = create_room(db)
    first = append_user_event(db, event_id="event-1", text="first", now=11)
    repeated = append_user_event(db, event_id="event-1", text="first", now=12)
    second = append_user_event(db, event_id="event-2", text="second", now=13)

    assert created["authority_epoch"] == 1
    assert (first["seq"], repeated["seq"], second["seq"]) == (1, 1, 2)
    assert repeated["idempotent"] is True
    store = sqlite_hosted_room_coordination(db)
    assert [event["event_id"] for event in store.read_events(room_id="room-1")["events"]] == [
        "event-1",
        "event-2",
    ]

    store.claim_authority(
        room_id="room-1",
        expected_gateway_id="gateway-a",
        expected_epoch=1,
        new_gateway_id="gateway-b",
        event_id="claim-2",
        now=14,
    )
    with pytest.raises(rooms.AuthorityConflictError, match="stale"):
        append_user_event(db, event_id="stale-event", text="must not append", now=15)


def test_driver_lease_contract_reclaims_after_expiry_and_fences_old_owner(tmp_path):
    db = tmp_path / "state.db"
    create_room(db)
    clock = FakeClock()
    store = sqlite_hosted_room_coordination(db)
    old = store.acquire_lease(
        room_id="room-1",
        gateway_id="gateway-a",
        authority_epoch=1,
        process_generation="process-a",
        ttl_seconds=5,
        clock=clock,
    )

    clock.advance(5)
    new = store.acquire_lease(
        room_id="room-1",
        gateway_id="gateway-a",
        authority_epoch=1,
        process_generation="process-b",
        ttl_seconds=30,
        clock=clock,
    )

    assert new.reclaimed is True
    assert new.lease_generation == old.lease_generation + 1
    with pytest.raises(driver.StaleLeaseError):
        store.renew_lease(old, ttl_seconds=30, clock=clock)
    with pytest.raises(driver.StaleLeaseError):
        store.release_lease(old, clock=clock)


def test_remote_receipt_is_exactly_idempotent_not_delivery_proof(tmp_path):
    db = tmp_path / "state.db"
    record = {
        "room_id": "room-1",
        "home_install_id": "install-home",
        "authority_gateway_id": "gateway-a",
        "authority_epoch": 1,
        "member_id": "member-1",
        "target_install_id": "install-peer",
        "target_profile": "reviewer",
        "task_id": "task-1",
        "execution_generation": 1,
        "run_id": "run-1",
        "session_id": "session-1",
    }

    store = sqlite_hosted_room_coordination(db)
    store.upsert_remote_run_receipt(record=record, now=20)
    store.upsert_remote_run_receipt(record=record, now=21)
    stored = store.remote_run_receipt(record=record)

    assert stored is not None
    assert (stored["run_id"], stored["session_id"]) == ("run-1", "session-1")
    with pytest.raises(rooms.HostedRoomError, match="conflicts"):
        store.upsert_remote_run_receipt(record={**record, "run_id": "other-run"}, now=22)


def test_peer_reservation_revocation_and_expiry_are_fenced(tmp_path):
    db = tmp_path / "state.db"
    claims = {
        "room_id": "room-peer",
        "home_install_id": "install-home",
        "member_id": "member-1",
        "target_install_id": "install-peer",
        "target_profile": "reviewer",
        "authority_gateway_id": "gateway-a",
        "authority_epoch": 1,
        "issued_at": 100,
    }

    store = sqlite_hosted_room_coordination(db)
    store.reserve_peer_room(claims=claims, expires_at=200, now=100)
    assert store.peer_room_grant_is_current(claims=claims, now=101)
    assert store.room_grant_is_revoked(claims=claims, now=101) is False

    store.revoke_room_grant_scope(claims=claims, expires_at=300, now=110)
    assert store.room_grant_is_revoked(claims=claims, now=111) is True
    assert store.peer_room_grant_is_current(claims=claims, now=111) is False
    assert store.room_grant_is_revoked(claims={**claims, "issued_at": 111}, now=111) is False
    assert store.room_grant_is_revoked(claims=claims, now=301) is False


def test_policy_cursor_and_watermark_only_advance_from_committed_log_order(tmp_path):
    db = tmp_path / "state.db"
    create_room(db)
    store: HostedRoomCoordination = sqlite_hosted_room_coordination(db)
    user = append_user_event(db, event_id="user-1", text="question", now=11)
    member = store.append_event(
        room_id="room-1",
        event_id="member-1",
        kind="message.member",
        actor={"kind": "member", "id": "ops"},
        payload={"text": "answer", "thread_id": "thread-1", "discussion_event_id": "user-1"},
        authority_gateway_id="gateway-a",
        authority_epoch=1,
        now=12,
    )
    terminal = store.append_event(
        room_id="room-1",
        event_id="settled-1",
        kind="turn.settled",
        actor={"kind": "gateway", "id": "gateway-a"},
        payload={
            "task_id": "task-1",
            "thread_id": "thread-1",
            "discussion_event_id": "user-1",
            "message_event_id": "member-1",
            "member_id": "ops",
            "seen_through_seq": user["seq"],
        },
        authority_gateway_id="gateway-a",
        authority_epoch=1,
        now=13,
    )

    checkpoint = store.policy_checkpoint()
    snapshot = checkpoint.snapshot(room_id="room-1", latest_seq=terminal["seq"])
    repeated = checkpoint.snapshot(room_id="room-1", latest_seq=terminal["seq"])

    assert (user["seq"], member["seq"], terminal["seq"]) == (1, 2, 3)
    assert snapshot.through_seq == repeated.through_seq == terminal["seq"]
    assert snapshot.watermarks[("thread-1", "ops")] == member["seq"]
    with sqlite3.connect(db) as conn:
        cursor = conn.execute(
            "SELECT through_seq FROM hosted_room_policy_cursors WHERE room_id=?", ("room-1",)
        ).fetchone()[0]
    assert cursor == terminal["seq"]
