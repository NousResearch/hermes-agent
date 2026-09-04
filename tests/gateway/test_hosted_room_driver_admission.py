"""Focused schema and admission regressions preserved from PR #19."""

from __future__ import annotations

import sqlite3

import pytest

from gateway import hosted_room_driver as driver
from gateway import hosted_rooms as rooms


class FakeClock:
    def __init__(self, value: float = 100.0):
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _identity(task_id: str = "task-1", *, turn_id: str = "turn-1"):
    return driver.TaskIdentity(
        room_id="room-1",
        task_id=task_id,
        thread_id="thread-1",
        turn_id=turn_id,
    )


def _payload(
    *,
    target_profile: str = "ops",
    prompt: str = "Inspect the release candidate.",
    source_event_seq: int = 1,
):
    return {
        "target_profile": target_profile,
        "prompt": prompt,
        "source_event_seq": source_event_seq,
    }


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "state.db"
    rooms.create_room(
        path,
        room_id="room-1",
        name="Release room",
        members=[{"member_id": "member-ops", "profile": "ops", "handle": "ops"}],
        authority_gateway_id="gateway-a",
        now=90,
    )
    return path


def _lease(
    db,
    clock,
    *,
    gateway="gateway-a",
    authority_epoch=1,
    process="process-a",
    process_pid=1001,
    process_start_time=5001,
    ttl=30,
):
    return driver.acquire_lease(
        db,
        room_id="room-1",
        gateway_id=gateway,
        authority_epoch=authority_epoch,
        process_generation=process,
        process_pid=process_pid,
        process_start_time=process_start_time,
        ttl_seconds=ttl,
        clock=clock,
    )


def _admit(db, identity, clock, *, payload=None):
    return driver.admit_task(
        db,
        identity,
        payload=_payload() if payload is None else payload,
        clock=clock,
    )


def test_task_admission_is_atomically_fenced_by_latest_stop(db):
    clock = FakeClock()
    room = rooms.room_state(db, room_id="room-1")
    first = rooms.append_event(
        db,
        room_id="room-1",
        event_id="user-before-stop",
        kind="message.user",
        actor={"kind": "user", "id": "desktop"},
        payload={"text": "inspect"},
        authority_gateway_id=room["authority_gateway_id"],
        authority_epoch=room["authority_epoch"],
    )
    stop = rooms.request_room_stop(
        db,
        room_id="room-1",
        cancel_id="stop-1",
        expected_gateway_id=room["authority_gateway_id"],
        expected_epoch=room["authority_epoch"],
    )

    with pytest.raises(driver.TaskAdmissionBlockedError, match="Stop fence"):
        _admit(
            db,
            _identity(),
            clock,
            payload=_payload(source_event_seq=first["seq"]),
        )

    with pytest.raises(driver.TaskAdmissionBlockedError, match="Stop fence"):
        _admit(
            db,
            _identity("task-at-stop", turn_id="turn-at-stop"),
            clock,
            payload=_payload(source_event_seq=stop["seq"]),
        )

    newer = rooms.append_event(
        db,
        room_id="room-1",
        event_id="user-after-stop",
        kind="message.user",
        actor={"kind": "user", "id": "desktop"},
        payload={"text": "inspect again"},
        authority_gateway_id=room["authority_gateway_id"],
        authority_epoch=room["authority_epoch"],
    )
    admitted = _admit(
        db,
        _identity("task-2", turn_id="turn-2"),
        clock,
        payload=_payload(source_event_seq=newer["seq"]),
    )
    assert admitted["status"] == "queued"


def test_durable_admission_barrier_blocks_new_tasks_idempotently(db):
    clock = FakeClock()
    existing_identity = _identity()
    _admit(db, existing_identity, clock)
    lease = _lease(db, clock)

    first = driver.block_room_admissions(
        db,
        room_id="room-1",
        reason="authority-demotion",
        expected_gateway_id="gateway-a",
        expected_epoch=1,
        clock=clock,
    )
    repeated = driver.block_room_admissions(
        db,
        room_id="room-1",
        reason="authority-demotion",
        expected_gateway_id="gateway-a",
        expected_epoch=1,
        clock=clock,
    )

    assert first["idempotent"] is False
    assert repeated["idempotent"] is True
    with pytest.raises(driver.TaskAdmissionBlockedError, match="admission barrier"):
        _admit(db, _identity("task-2", turn_id="turn-2"), clock)
    with pytest.raises(driver.TaskAdmissionBlockedError, match="admission barrier"):
        driver.start_task(
            db,
            existing_identity,
            lease,
            expected_cancel_generation=0,
            clock=clock,
        )


def test_room_demotion_intent_is_atomic_idempotent_and_conflict_checked(db):
    clock = FakeClock()
    first = driver.begin_room_demotion(
        db,
        room_id="room-1",
        expected_gateway_id="gateway-a",
        expected_epoch=1,
        observed_gateway_id="gateway-b",
        observed_epoch=2,
        cancel_id="demotion-1",
        clock=clock,
    )
    repeated = driver.begin_room_demotion(
        db,
        room_id="room-1",
        expected_gateway_id="gateway-a",
        expected_epoch=1,
        observed_gateway_id="gateway-b",
        observed_epoch=2,
        cancel_id="demotion-racing-process",
        clock=clock,
    )

    assert first["idempotent"] is False
    assert repeated["idempotent"] is True
    assert repeated["cancel_id"] == "demotion-1"
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        stop = conn.execute(
            """SELECT kind, actor_json, authority_epoch, payload_json
                 FROM hosted_room_events
                WHERE room_id='room-1' AND event_id=?""",
            (rooms._stop_event_id("demotion-1"),),
        ).fetchone()
    assert stop is not None
    assert stop["kind"] == "room.stop_requested"
    assert stop["authority_epoch"] == 1
    assert stop["actor_json"] == '{"id":"gateway-a","kind":"gateway"}'
    assert stop["payload_json"] == '{"cancel_id":"demotion-1"}'
    assert driver.pending_room_demotion(db, room_id="room-1") == {
        key: first[key] for key in first if key != "idempotent"
    }
    with pytest.raises(driver.TaskConflictError, match="different demotion intent"):
        driver.begin_room_demotion(
            db,
            room_id="room-1",
            expected_gateway_id="gateway-a",
            expected_epoch=1,
            observed_gateway_id="gateway-c",
            observed_epoch=3,
            cancel_id="demotion-2",
            clock=clock,
        )


def test_demotion_intent_and_barrier_roll_back_when_atomic_stop_fails(
    db,
    monkeypatch,
):
    def fail_stop(*args, **kwargs):
        raise rooms.HostedRoomError("no reserved Stop capacity")

    monkeypatch.setattr(rooms, "_request_room_stop_locked", fail_stop)

    with pytest.raises(rooms.HostedRoomError, match="reserved Stop capacity"):
        driver.begin_room_demotion(
            db,
            room_id="room-1",
            expected_gateway_id="gateway-a",
            expected_epoch=1,
            observed_gateway_id="gateway-b",
            observed_epoch=2,
            cancel_id="demotion-without-capacity",
            clock=FakeClock(),
        )

    with sqlite3.connect(db) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM hosted_room_driver_admission_barriers"
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM hosted_room_driver_demotion_intents"
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                """SELECT COUNT(*) FROM hosted_room_events
               WHERE kind='room.stop_requested'"""
            ).fetchone()[0]
            == 0
        )


def test_admission_barrier_is_scoped_to_superseded_authority_epoch(db):
    clock = FakeClock()
    driver.block_room_admissions(
        db,
        room_id="room-1",
        reason="authority-demotion",
        expected_gateway_id="gateway-a",
        expected_epoch=1,
        clock=clock,
    )
    rooms.claim_authority(
        db,
        room_id="room-1",
        expected_gateway_id="gateway-a",
        expected_epoch=1,
        new_gateway_id="gateway-b",
        event_id="claim-gateway-b",
        now=clock(),
    )
    current = rooms.claim_authority(
        db,
        room_id="room-1",
        expected_gateway_id="gateway-b",
        expected_epoch=2,
        new_gateway_id="gateway-a",
        event_id="claim-gateway-a",
        now=clock(),
    )
    source = rooms.append_event(
        db,
        room_id="room-1",
        event_id="user-after-repromotion",
        kind="message.user",
        actor={"kind": "user", "id": "desktop"},
        payload={"text": "inspect after repromotion", "thread_id": "thread-1"},
        authority_gateway_id=current["authority_gateway_id"],
        authority_epoch=current["authority_epoch"],
        require_open_admissions=True,
    )
    identity = _identity()
    admitted = _admit(
        db,
        identity,
        clock,
        payload=_payload(source_event_seq=source["seq"]),
    )
    lease = _lease(db, clock, authority_epoch=3, process="process-epoch-3")

    assert admitted["status"] == "queued"
    assert (
        driver.start_task(
            db,
            identity,
            lease,
            expected_cancel_generation=0,
            clock=clock,
        ).identity
        == identity
    )
    next_barrier = driver.block_room_admissions(
        db,
        room_id="room-1",
        reason="authority-demotion",
        expected_gateway_id="gateway-a",
        expected_epoch=3,
        clock=clock,
    )
    assert next_barrier["idempotent"] is False
    assert next_barrier["authority_epoch"] == 3


def test_admission_is_bounded_by_terminal_recovery_liability(db, monkeypatch):
    monkeypatch.setattr(rooms, "TERMINAL_RECOVERY_COUNT_RESERVE", 2)
    clock = FakeClock()
    _admit(db, _identity("task-1", turn_id="turn-1"), clock)

    with pytest.raises(driver.TaskAdmissionBlockedError, match="recovery headroom"):
        _admit(db, _identity("task-2", turn_id="turn-2"), clock)

    assert len(driver.list_tasks(db, room_id="room-1")) == 1


def test_terminal_recovery_audit_runs_once_per_process_schema(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    rooms.create_room(
        db,
        room_id="room-1",
        name="Release room",
        members=[{"profile": "ops", "handle": "ops"}],
        authority_gateway_id="gateway-a",
        now=90,
    )
    original = driver._raise_if_terminal_recovery_headroom_is_unrecoverable
    calls = 0

    def counted(conn):
        nonlocal calls
        calls += 1
        return original(conn)

    monkeypatch.setattr(
        driver,
        "_raise_if_terminal_recovery_headroom_is_unrecoverable",
        counted,
    )
    identity = _identity()
    _admit(db, identity, FakeClock())

    for _ in range(5):
        assert driver.get_task(db, identity)["status"] == "queued"
        assert len(driver.list_tasks(db, room_id="room-1")) == 1

    assert calls == 1


def test_pre_admission_barrier_schema_is_migrated_without_losing_work(db):
    identity = _identity()
    _admit(db, identity, FakeClock())
    with sqlite3.connect(db) as conn:
        conn.execute("DROP TABLE hosted_room_driver_admission_barriers")

    assert driver.get_task(db, identity)["status"] == "queued"
    with sqlite3.connect(db) as conn:
        table = conn.execute(
            """SELECT name FROM sqlite_master
               WHERE type='table'
                 AND name='hosted_room_driver_admission_barriers'"""
        ).fetchone()
        task_count = conn.execute(
            "SELECT COUNT(*) FROM hosted_room_driver_tasks"
        ).fetchone()[0]

    assert table == ("hosted_room_driver_admission_barriers",)
    assert task_count == 1


def test_pre_demotion_intent_schema_is_migrated_without_losing_work(db):
    identity = _identity()
    _admit(db, identity, FakeClock())
    with sqlite3.connect(db) as conn:
        conn.execute("DROP TABLE hosted_room_driver_demotion_intents")

    assert driver.get_task(db, identity)["status"] == "queued"
    with sqlite3.connect(db) as conn:
        table = conn.execute(
            """SELECT name FROM sqlite_master
               WHERE type='table'
                 AND name='hosted_room_driver_demotion_intents'"""
        ).fetchone()
        task_count = conn.execute(
            "SELECT COUNT(*) FROM hosted_room_driver_tasks"
        ).fetchone()[0]

    assert table == ("hosted_room_driver_demotion_intents",)
    assert task_count == 1


def test_barrier_only_demotion_schema_fails_loudly_instead_of_wedging(db):
    driver.block_room_admissions(
        db,
        room_id="room-1",
        reason="authority-demotion",
        expected_gateway_id="gateway-a",
        expected_epoch=1,
        clock=FakeClock(),
    )
    with sqlite3.connect(db) as conn:
        conn.execute("DROP TABLE hosted_room_driver_demotion_intents")

    with pytest.raises(driver.DriverStateError, match="lacks resumable target"):
        driver.pending_room_demotion(db, room_id="room-1")


def test_completed_legacy_demotion_barrier_does_not_block_schema_migration(db):
    driver.block_room_admissions(
        db,
        room_id="room-1",
        reason="authority-demotion",
        expected_gateway_id="gateway-a",
        expected_epoch=1,
        clock=FakeClock(),
    )
    rooms.claim_authority(
        db,
        room_id="room-1",
        expected_gateway_id="gateway-a",
        expected_epoch=1,
        new_gateway_id="gateway-b",
        event_id="authority-claimed-by-gateway-b",
        now=101.0,
    )
    with sqlite3.connect(db) as conn:
        conn.execute("DROP TABLE hosted_room_driver_demotion_intents")

    assert driver.pending_room_demotion(db, room_id="room-1") is None
    with sqlite3.connect(db) as conn:
        table = conn.execute(
            """SELECT name FROM sqlite_master
               WHERE type='table'
                 AND name='hosted_room_driver_demotion_intents'"""
        ).fetchone()
    assert table == ("hosted_room_driver_demotion_intents",)


def test_current_demotion_intent_without_atomic_stop_fails_loudly(db):
    driver.begin_room_demotion(
        db,
        room_id="room-1",
        expected_gateway_id="gateway-a",
        expected_epoch=1,
        observed_gateway_id="gateway-b",
        observed_epoch=2,
        cancel_id="demotion-lost-stop",
        clock=FakeClock(),
    )
    with sqlite3.connect(db) as conn:
        conn.execute(
            "DELETE FROM hosted_room_events WHERE room_id='room-1' AND event_id=?",
            (rooms._stop_event_id("demotion-lost-stop"),),
        )

    with pytest.raises(driver.DriverStateError, match="lacks its atomic Stop"):
        driver.pending_room_demotion(db, room_id="room-1")


def test_pre_owner_identity_schema_is_migrated_without_losing_work(db):
    clock = FakeClock()
    identity = _identity()
    _admit(db, identity, clock)
    _lease(db, clock)

    with sqlite3.connect(db) as conn:
        conn.execute("ALTER TABLE hosted_room_driver_leases DROP COLUMN process_pid")
        conn.execute(
            "ALTER TABLE hosted_room_driver_leases DROP COLUMN process_start_time"
        )
        conn.execute("ALTER TABLE hosted_room_driver_tasks DROP COLUMN run_process_pid")
        conn.execute(
            "ALTER TABLE hosted_room_driver_tasks DROP COLUMN run_process_start_time"
        )

    assert driver.get_task(db, identity)["status"] == "queued"
    with sqlite3.connect(db) as conn:
        lease_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(hosted_room_driver_leases)")
        }
        task_columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(hosted_room_driver_tasks)")
        }
        legacy_lease = conn.execute(
            """SELECT process_generation, process_pid, process_start_time,
                      lease_generation
               FROM hosted_room_driver_leases WHERE room_id='room-1'"""
        ).fetchone()

    assert {"process_pid", "process_start_time"} <= lease_columns
    assert {"run_process_pid", "run_process_start_time"} <= task_columns
    assert legacy_lease == ("process-a", None, None, 1)
    with pytest.raises(driver.LeaseHeldError, match="another generation"):
        _lease(db, clock)

    clock.advance(31)
    successor = _lease(
        db,
        clock,
        process="process-b",
        process_pid=2002,
        process_start_time=6002,
    )
    assert successor.lease_generation == 2
    assert successor.reclaimed is True
    driver.start_task(
        db,
        identity,
        successor,
        expected_cancel_generation=0,
        clock=clock,
    )
    running = driver.get_task(db, identity)
    assert running["run_process_generation"] == "process-b"
    assert running["run_process_pid"] == 2002
    assert running["run_process_start_time"] == 6002
