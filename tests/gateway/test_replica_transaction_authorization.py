"""The replica writer admits requests before replica maintenance (#97681)."""

import sqlite3

import pytest

from gateway import hosted_room_replicas as replicas
from gateway import hosted_rooms as rooms


def _seed_unaudited_replica(db):
    rooms.create_room(
        db,
        room_id="authority-room",
        name="Authority",
        members=[{"kind": "bot", "id": "worker"}],
        authority_gateway_id="install:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE admission_marker (value TEXT NOT NULL)")
        conn.execute(
            """INSERT INTO hosted_room_replicas
               (room_id, name, members_json, authority_gateway_id, authority_epoch,
                last_seq, latest_seq, event_bytes, created_at, updated_at)
               VALUES ('replica-room', 'Replica', '[]', ?, 1, 0, 0, 17, 1, 1)""",
            ("install:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",),
        )


def _event_bytes(db):
    with sqlite3.connect(db) as conn:
        return conn.execute(
            "SELECT event_bytes FROM hosted_room_replicas WHERE room_id='replica-room'"
        ).fetchone()[0]


def test_refusal_is_inside_writer_before_replica_maintenance_and_rolls_back(
    tmp_path, monkeypatch
):
    db = tmp_path / "shared-state.db"
    _seed_unaudited_replica(db)

    def forbidden(_conn):
        pytest.fail("replica schema/audit ran before authorization")

    monkeypatch.setattr(replicas, "_initialize_replica_schema", forbidden)
    monkeypatch.setattr(replicas, "_audit_existing_replicas_locked", forbidden)
    calls = []

    def refuse(conn):
        calls.append("refused")
        assert conn.in_transaction
        assert conn.execute(
            "SELECT event_bytes FROM hosted_room_replicas WHERE room_id='replica-room'"
        ).fetchone()[0] == 17
        with sqlite3.connect(db, timeout=0) as contender:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                contender.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT INTO admission_marker VALUES ('transient')")
        raise PermissionError("grant withdrawn")

    with pytest.raises(PermissionError, match="grant withdrawn"):
        with replicas._replica_transaction(db, _authorize=refuse):
            pytest.fail("refused writer reached its body")

    assert calls == ["refused"]
    assert _event_bytes(db) == 17
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM admission_marker").fetchone()[0] == 0


def test_allowed_callback_precedes_schema_and_audit_and_default_is_unchanged(
    tmp_path, monkeypatch
):
    db = tmp_path / "shared-state.db"
    _seed_unaudited_replica(db)
    original_schema = replicas._initialize_replica_schema
    original_audit = replicas._audit_existing_replicas_locked
    order = []

    def schema(conn):
        order.append("schema")
        original_schema(conn)

    def audit(conn):
        order.append("audit")
        original_audit(conn)

    monkeypatch.setattr(replicas, "_initialize_replica_schema", schema)
    monkeypatch.setattr(replicas, "_audit_existing_replicas_locked", audit)

    def allow(conn):
        order.append("authorized")
        assert conn.in_transaction
        assert conn.execute(
            "SELECT event_bytes FROM hosted_room_replicas WHERE room_id='replica-room'"
        ).fetchone()[0] == 17

    with replicas._replica_transaction(db, _authorize=allow) as conn:
        order.append("body")
        assert conn.execute(
            "SELECT event_bytes FROM hosted_room_replicas WHERE room_id='replica-room'"
        ).fetchone()[0] == 0
        conn.execute("INSERT INTO admission_marker VALUES ('allowed')")

    assert order == ["authorized", "schema", "audit", "body"]
    assert _event_bytes(db) == 0
    with replicas._replica_transaction(db) as conn:
        assert conn.execute("SELECT value FROM admission_marker").fetchone()[0] == "allowed"
    assert order == ["authorized", "schema", "audit", "body", "schema", "audit"]
