"""Atomic, incremental persistence contracts for gateway routing."""

import sqlite3

import pytest

from hermes_state import SessionDB


def _row(db, scope, key):
    return db._conn.execute(
        "SELECT entry_json, updated_at FROM gateway_routing "
        "WHERE scope = ? AND session_key = ?",
        (scope, key),
    ).fetchone()


def test_replace_routing_snapshot_only_writes_changed_rows(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        assert db.replace_gateway_routing_entries(
            {"a": '{"v":1}', "b": '{"v":1}', "deleted": '{"v":1}'},
            scope="one",
            generation=10,
        ) is True
        untouched_before = tuple(_row(db, "one", "b"))

        assert db.replace_gateway_routing_entries(
            {"a": '{"v":2}', "b": '{"v":1}', "c": '{"v":1}'},
            scope="one",
            generation=11,
        ) is True

        state, generation = db.load_gateway_routing_state(scope="one")
        assert state == {"a": '{"v":2}', "b": '{"v":1}', "c": '{"v":1}'}
        assert generation == 11
        assert tuple(_row(db, "one", "b")) == untouched_before
        assert _row(db, "one", "deleted") is None
    finally:
        db.close()


def test_routing_snapshot_is_scope_isolated_and_rejects_stale_generation(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.replace_gateway_routing_entries({"key": "new"}, scope="one", generation=20)
        db.replace_gateway_routing_entries({"key": "other"}, scope="two", generation=3)

        assert db.replace_gateway_routing_entries(
            {"key": "stale"}, scope="one", generation=19
        ) is False
        assert db.load_gateway_routing_state(scope="one") == ({"key": "new"}, 20)
        assert db.load_gateway_routing_state(scope="two") == ({"key": "other"}, 3)
    finally:
        db.close()


def test_routing_rows_and_generation_roll_back_together(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.replace_gateway_routing_entries({"a": "before"}, scope="one", generation=1)
        with pytest.raises(sqlite3.Error):
            db.replace_gateway_routing_entries(
                {"a": object()}, scope="one", generation=2
            )
        assert db.load_gateway_routing_state(scope="one") == ({"a": "before"}, 1)
    finally:
        db.close()
