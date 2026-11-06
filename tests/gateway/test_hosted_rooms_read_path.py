"""Focused lock-behavior coverage for hosted Group Chat reads."""

import sqlite3

from gateway import hosted_rooms


def test_list_rooms_does_not_enter_a_write_transaction_or_prune(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    hosted_rooms.create_room(
        db,
        room_id="room-1",
        name="Release room",
        members=[{"profile": "default", "handle": "hermes"}],
        authority_gateway_id="gateway-a",
    )

    def reject_write_transaction(*_args, **_kwargs):
        raise AssertionError("list_rooms must not enter a write transaction")

    def reject_prune(*_args, **_kwargs):
        raise AssertionError("list_rooms must not prune retention state")

    monkeypatch.setattr(hosted_rooms, "_transaction", reject_write_transaction)
    monkeypatch.setattr(
        hosted_rooms,
        "_prune_disbanded_rooms_locked",
        reject_prune,
    )

    rows = hosted_rooms.list_rooms(db)

    assert [row["room_id"] for row in rows] == ["room-1"]


def test_repeated_probes_reuse_a_positive_readiness_verdict(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    hosted_rooms.create_room(
        db,
        room_id="room-1",
        name="Release room",
        members=[{"profile": "default", "handle": "hermes"}],
        authority_gateway_id="gateway-a",
    )
    real_schema_is_current = hosted_rooms._schema_is_current
    probes = []

    def counting_schema_is_current(conn: sqlite3.Connection) -> bool:
        probes.append(conn)
        return real_schema_is_current(conn)

    monkeypatch.setattr(hosted_rooms, "_schema_is_current", counting_schema_is_current)
    conn = hosted_rooms.open_sqlite(db)
    try:
        assert hosted_rooms._store_ready(conn, db) is True
        assert hosted_rooms._store_ready(conn, db) is True
        assert hosted_rooms._store_ready(conn, db) is True
    finally:
        conn.close()

    assert len(probes) == 1


def test_a_schema_change_invalidates_the_cached_readiness_verdict(tmp_path):
    db = tmp_path / "state.db"
    hosted_rooms.create_room(
        db,
        room_id="room-1",
        name="Release room",
        members=[{"profile": "default", "handle": "hermes"}],
        authority_gateway_id="gateway-a",
    )
    conn = hosted_rooms.open_sqlite(db)
    breaker = hosted_rooms.open_sqlite(db)
    try:
        assert hosted_rooms._store_ready(conn, db) is True
        breaker.execute("DROP TABLE hosted_rooms")
        breaker.commit()
        assert hosted_rooms._store_ready(conn, db) is False
    finally:
        conn.close()
        breaker.close()


def test_a_not_ready_store_stays_on_the_full_probe_path(tmp_path, monkeypatch):
    db = tmp_path / "state.db"
    hosted_rooms.create_room(
        db,
        room_id="room-1",
        name="Release room",
        members=[{"profile": "default", "handle": "hermes"}],
        authority_gateway_id="gateway-a",
    )
    breaker = hosted_rooms.open_sqlite(db)
    breaker.execute("DROP TABLE hosted_rooms")
    breaker.commit()
    breaker.close()

    real_schema_is_current = hosted_rooms._schema_is_current
    probes = []

    def counting_schema_is_current(conn: sqlite3.Connection) -> bool:
        probes.append(conn)
        return real_schema_is_current(conn)

    monkeypatch.setattr(hosted_rooms, "_schema_is_current", counting_schema_is_current)
    conn = hosted_rooms.open_sqlite(db)
    try:
        assert hosted_rooms._store_ready(conn, db) is False
        assert hosted_rooms._store_ready(conn, db) is False
    finally:
        conn.close()

    assert len(probes) == 2
    hosted_rooms.create_room(
        db,
        room_id="room-2",
        name="Rebuilt room",
        members=[{"profile": "default", "handle": "hermes"}],
        authority_gateway_id="gateway-a",
    )
    assert [row["room_id"] for row in hosted_rooms.list_rooms(db)] == ["room-2"]
