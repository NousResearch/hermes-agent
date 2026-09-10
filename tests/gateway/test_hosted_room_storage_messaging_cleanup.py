"""Messaging cleanup ownership after the hosted-room storage split."""

import sqlite3
import time

import pytest

from gateway import hosted_room_controls as controls
from gateway import hosted_room_messaging_approvals as approvals
from gateway import hosted_rooms as storage
from gateway import hosted_rooms as rooms


CONTROL_TABLES = (
    "hosted_room_control_commands",
    "hosted_room_control_tokens",
    "hosted_room_peer_controls",
)
APPROVAL_TABLES = (
    "hosted_room_pending_approvals",
    "hosted_room_messaging_approval_commands",
)


def _create(db, room_id, *, now):
    rooms.create_room(
        db,
        room_id=room_id,
        name="Cleanup scope",
        members=[{"member_id": "member-1", "profile": "reviewer"}],
        authority_gateway_id="gateway-1",
        now=now,
    )


def _seed_controls(db, room_id, *, now):
    issued = controls.issue_home_control_token(
        db,
        room_id=room_id,
        member_id="member-1",
        authority_gateway_id="gateway-1",
        authority_epoch=1,
        expires_at=now + 600,
        now=now,
    )
    controls.save_peer_control_link(
        db,
        room_id=room_id,
        member_id="member-1",
        home_url="https://home.example.test",
        authority_gateway_id="gateway-1",
        authority_epoch=1,
        room_name="Cleanup scope",
        member_count=1,
        control_token=issued.control_token,
        expires_at=now + 600,
        now=now,
    )
    controls.begin_control_retry(
        db,
        command_id=f"retry-{room_id}",
        room_id=room_id,
        member_id="member-1",
        task_ids=[f"task-{room_id}"],
        now=now,
    )


def _disband(db, room_id, *, now):
    rooms.disband_room(
        db,
        room_id=room_id,
        expected_gateway_id="gateway-1",
        expected_epoch=1,
        now=now,
    )


def _rows(db, tables, room_id):
    with sqlite3.connect(db) as conn:
        return {
            table: conn.execute(
                f"SELECT * FROM {table} WHERE room_id=? ORDER BY rowid",
                (room_id,),
            ).fetchall()
            for table in tables
        }


@pytest.mark.parametrize("pressure", ["age", "count", "bytes"])
def test_pruning_removes_only_retired_room_controls(tmp_path, monkeypatch, pressure):
    db = tmp_path / "state.db"
    now = time.time()
    for room_id in ("room-old", "room-young", "room-active"):
        _create(db, room_id, now=now - 100)
        _seed_controls(db, room_id, now=now - 100)
    _disband(db, "room-old", now=now - 20)
    _disband(db, "room-young", now=now - 10)
    before = {
        room_id: _rows(db, CONTROL_TABLES, room_id)
        for room_id in ("room-old", "room-young", "room-active")
    }
    assert all(len(rows) == 1 for tables in before.values() for rows in tables.values())

    if pressure == "age":
        monkeypatch.setattr(rooms, "DISBANDED_ROOM_RETENTION_SECONDS", 15)
        assert rooms.prune_disbanded_rooms(db, now=now) == 1
    elif pressure == "count":
        monkeypatch.setattr(rooms, "MAX_DISBANDED_ROOM_TOMBSTONES", 1)
        assert rooms.prune_disbanded_rooms(db, now=now) == 1
    else:
        # Capacity callers supply a byte budget to this same transaction owner.
        with storage._transaction(db, immediate=True) as conn:
            budget = conn.execute(
                "SELECT SUM(event_bytes) FROM hosted_rooms WHERE room_id != 'room-old'"
            ).fetchone()[0]
            assert (
                storage._prune_disbanded_rooms_locked(
                    conn, now=now, max_gateway_event_bytes=budget
                )
                == 1
            )

    assert _rows(db, CONTROL_TABLES, "room-old") == dict.fromkeys(CONTROL_TABLES, [])
    for room_id in ("room-young", "room-active"):
        assert _rows(db, CONTROL_TABLES, room_id) == before[room_id]
    assert rooms.prune_disbanded_rooms(db, now=now) == 0
    with pytest.raises(rooms.RoomConflictError, match="disbanded"):
        _create(db, "room-old", now=now)


@pytest.mark.parametrize("present_mask", range(8))
def test_optional_control_tables_are_not_required_or_created(tmp_path, present_mask):
    db = tmp_path / "state.db"
    now = time.time()
    _create(db, "room-old", now=now)
    present = {
        table
        for index, table in enumerate(CONTROL_TABLES)
        if present_mask & (1 << index)
    }
    if present:
        _seed_controls(db, "room-old", now=now)
        with sqlite3.connect(db) as conn:
            for table in set(CONTROL_TABLES) - present:
                conn.execute(f"DROP TABLE {table}")
    _disband(db, "room-old", now=now)

    assert (
        rooms.prune_disbanded_rooms(
            db, now=now + rooms.DISBANDED_ROOM_RETENTION_SECONDS + 1
        )
        == 1
    )

    with sqlite3.connect(db) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert tables & set(CONTROL_TABLES) == present
    assert _rows(db, present, "room-old") == dict.fromkeys(present, [])
    assert not tables & set(APPROVAL_TABLES)


def _pending(db, suffix):
    return approvals.persist_pending_approval(
        db,
        room_id="room-old",
        member_id="member-1",
        action={
            "kind": "approval",
            "authority_gateway_id": "gateway-1",
            "authority_epoch": 1,
            "task_id": f"task-{suffix}",
            "execution_generation": 1,
            "request_id": f"request-{suffix}",
            "approval": {
                "description": "Run focused tests",
                "command": "pytest tests/focused",
                "choices": ["once", "deny"],
            },
        },
    )


def test_pruning_preserves_completed_uncertain_and_unstarted_approval_receipts(
    tmp_path,
):
    db = tmp_path / "state.db"
    now = time.time()
    _create(db, "room-old", now=now)
    _seed_controls(db, "room-old", now=now)
    completed = _pending(db, "completed")
    assert approvals.apply_pending_decision(
        db, pending=completed, choice="once", apply=lambda: {"resolved": 1}
    ) == {"resolved": 1}

    def uncertain_delivery():
        raise RuntimeError("delivery outcome unknown")

    with pytest.raises(RuntimeError, match="delivery outcome unknown"):
        approvals.apply_pending_decision(
            db,
            pending=_pending(db, "uncertain"),
            choice="once",
            apply=uncertain_delivery,
        )
    approvals.begin_approval_command(
        db, command_id="not-started", pending=_pending(db, "unstarted"), choice="deny"
    )
    before = _rows(db, APPROVAL_TABLES, "room-old")
    assert len(before[APPROVAL_TABLES[0]]) == 1
    assert len(before[APPROVAL_TABLES[1]]) == 3
    _disband(db, "room-old", now=now)

    assert (
        rooms.prune_disbanded_rooms(
            db, now=now + rooms.DISBANDED_ROOM_RETENTION_SECONDS + 1
        )
        == 1
    )
    assert _rows(db, APPROVAL_TABLES, "room-old") == before
    assert _rows(db, CONTROL_TABLES, "room-old") == dict.fromkeys(CONTROL_TABLES, [])

    def unexpected_reapplication():
        pytest.fail("completed approval must replay its receipt without delivery")

    assert approvals.apply_pending_decision(
        db, pending=completed, choice="once", apply=unexpected_reapplication
    ) == {"resolved": 1}
    assert _rows(db, APPROVAL_TABLES, "room-old") == before


def test_cleanup_failure_rolls_back_control_deletions_and_retired_identity(tmp_path):
    db = tmp_path / "state.db"
    now = time.time()
    _create(db, "room-old", now=now)
    _seed_controls(db, "room-old", now=now)
    _disband(db, "room-old", now=now)
    tables = (
        *CONTROL_TABLES,
        "hosted_rooms",
        "hosted_room_events",
        "hosted_room_retired_ids",
    )
    before = _rows(db, tables, "room-old")
    with sqlite3.connect(db) as conn:
        conn.execute(
            """CREATE TRIGGER refuse_control_cleanup
               BEFORE DELETE ON hosted_room_control_tokens
               BEGIN SELECT RAISE(ABORT, 'cleanup refused'); END"""
        )

    with pytest.raises(sqlite3.IntegrityError, match="cleanup refused"):
        rooms.prune_disbanded_rooms(
            db, now=now + rooms.DISBANDED_ROOM_RETENTION_SECONDS + 1
        )

    assert _rows(db, tables, "room-old") == before
    with sqlite3.connect(db) as conn:
        conn.execute("DROP TRIGGER refuse_control_cleanup")
    assert (
        rooms.prune_disbanded_rooms(
            db, now=now + rooms.DISBANDED_ROOM_RETENTION_SECONDS + 1
        )
        == 1
    )
