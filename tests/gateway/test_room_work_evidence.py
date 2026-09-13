"""Passive receipts survive owner changes without granting replay rights."""
import copy
import json
import sqlite3
from contextlib import closing

import pytest

from gateway import hosted_rooms as rooms


def connect(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def create(path, room="room"):
    rooms.create_room(path, room_id=room, name="Evidence", members=[{"profile": "ops", "handle": "ops"}],
                      authority_gateway_id="owner-a", now=10)


def test_pending_ack_survives_restart_and_fences_authority(tmp_path):
    from gateway import hosted_room_work_records as work
    path = tmp_path / "state.db"
    create(path)
    rooms.append_event(path, room_id="room", event_id="input", kind="message.user",
        actor={"kind": "user", "id": "alice"}, payload={"text": "hello"},
        authority_gateway_id="owner-a", authority_epoch=1)
    from gateway import hosted_room_driver as driver
    identity = driver.TaskIdentity("room", "task", "thread", "turn")
    driver.admit_task(path, identity, payload={"target_profile": "ops", "prompt": "private prompt",
        "source_event_seq": 1}, clock=lambda: 10)
    first = work.capture(path, room_id="room", local_gateway_id="owner-a")
    assert first["tasks"][0]["task_id"] == "task"
    assert "private prompt" not in work.encode(first)
    assert first["version"] == 1
    with closing(connect(path)) as conn, conn:
        pending = work.prepare_delivery_locked(conn, room_id="room", local_gateway_id="owner-a",
            target_install_id="peer", route_generation="route-1", through_seq=100)
        raw = conn.execute(f"SELECT record_json FROM {work.PENDING_TABLE}").fetchone()[0]
    with closing(connect(path)) as conn, conn:
        for key, value in (("revision", first["revision"] + 1), ("digest", "0" * 64),
                           ("authority", {"gateway_id": "owner-b", "epoch": 2})):
            ack = work.acknowledgement(pending)
            ack[key] = value
            assert not work.acknowledge_locked(conn, room_id="room", target_install_id="peer",
                route_generation="route-1", record=pending, ack=ack)
        assert not work.acknowledge_locked(conn, room_id="room", target_install_id="peer",
            route_generation="route-2", record=pending, ack=work.acknowledgement(pending))
        conn.execute("UPDATE hosted_rooms SET authority_gateway_id='owner-b',authority_epoch=2 WHERE room_id='room'")
        conn.execute("INSERT INTO hosted_room_events VALUES ('room',2,'claim','authority.claimed',?,2,?,11)",
            (json.dumps({"kind": "system", "id": "authority-control"}), json.dumps({
                "previous_gateway_id": "owner-a", "authority_gateway_id": "owner-b", "authority_epoch": 2})))
        conn.execute("UPDATE hosted_rooms SET next_seq=3 WHERE room_id='room'")
    successor = work.capture(path, room_id="room", local_gateway_id="owner-b")
    assert successor["version"] == 2
    assert successor["incompleteness"] == ["prior_authority_work_unknown"]
    with closing(connect(path)) as conn, conn:
        assert not work.acknowledge_locked(conn, room_id="room", target_install_id="peer",
            route_generation="route-1", record=pending, ack=work.acknowledgement(pending))
        old = conn.execute(f"SELECT * FROM {work.PENDING_TABLE}").fetchone()
        assert old["record_json"] == raw and old["status"] == "pending"
        assert old["disposition"] == "superseded_authority"
        current = work.prepare_delivery_locked(conn, room_id="room", local_gateway_id="owner-b",
            target_install_id="peer", route_generation="route-3", through_seq=100)
        ack = work.acknowledgement(current)
        wrong = copy.deepcopy(ack)
        wrong["lineage_sha256"] = "0" * 64
        assert not work.acknowledge_locked(conn, room_id="room", target_install_id="peer",
            route_generation="route-3", record=current, ack=wrong)
        assert work.acknowledge_locked(conn, room_id="room", target_install_id="peer",
            route_generation="route-3", record=current, ack=ack)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(f"UPDATE {work.PENDING_TABLE} SET room_id='other' WHERE producer_epoch=1")
    with pytest.raises(work.WorkRecordError):
        work.capture(path, room_id="room", local_gateway_id="owner-a")


def test_migration_preserves_bytes_invalid_rows_and_caller_transaction(tmp_path, monkeypatch):
    from gateway import hosted_room_work_records as work
    from gateway import hosted_room_work_storage as storage
    path = tmp_path / "state.db"
    create(path)
    create(path, "healthy")
    record = work.capture(path, room_id="room", local_gateway_id="owner-a")
    raw = json.dumps(record, indent=3)
    with closing(connect(path)) as conn, conn:
        conn.execute(f"DROP TABLE {work.SOURCE_TABLE}")
        conn.execute("DROP TRIGGER IF EXISTS trg_work_invalid_insert")
        conn.execute(f"CREATE TABLE {work.SOURCE_TABLE} (room_id TEXT PRIMARY KEY, revision INTEGER, digest TEXT, record_json TEXT)")
        conn.execute(f"INSERT INTO {work.SOURCE_TABLE} VALUES (?,?,?,?)", ("room", record["revision"], record["digest"], raw))
        conn.execute(f"INSERT INTO {work.SOURCE_TABLE} VALUES ('orphan',4,'broken','not-json')")
    with closing(connect(path)) as conn:
        conn.execute("BEGIN IMMEDIATE")
        work.initialize(conn)
        assert conn.in_transaction
        conn.rollback()
        assert "producer_epoch" not in {r[1] for r in conn.execute(f"PRAGMA table_info({work.SOURCE_TABLE})")}
        work.initialize(conn)
        assert conn.execute(f"SELECT record_json FROM {work.SOURCE_TABLE} WHERE room_id='room'").fetchone()[0] == raw
        assert conn.execute(f"SELECT record_json FROM {storage.INVALID_TABLE}").fetchone()[0] == "not-json"
        conn.execute(f"UPDATE {work.SOURCE_TABLE} SET record_json='bad' WHERE room_id='room'")
        conn.commit()
    # Ordinary init/capture must never scan and mutate unrelated damaged evidence.
    work.capture(path, room_id="healthy", local_gateway_id="owner-a")
    with closing(connect(path)) as conn:
        assert conn.execute(f"SELECT disposition FROM {work.SOURCE_TABLE} WHERE room_id='room'").fetchone()[0] == "current"
    with pytest.raises(work.InvalidStoredWorkRecord):
        work.capture(path, room_id="room", local_gateway_id="owner-a")
    with closing(connect(path)) as conn:
        assert conn.execute(f"SELECT disposition,record_json FROM {work.SOURCE_TABLE} WHERE room_id='room'").fetchone()[:] == ("invalid", "bad")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(f"DELETE FROM {storage.INVALID_TABLE}")
        monkeypatch.setattr(work, "MAX_STORE_BYTES", 1)
        work.initialize(conn)
        with pytest.raises(work.WorkRecordCapacityError):
            work.prepare_delivery_locked(conn, room_id="healthy", local_gateway_id="owner-a",
                target_install_id="peer", route_generation="route", through_seq=100)
