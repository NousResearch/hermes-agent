"""The SQLite backend serves local-policy target lookups (live_listing_facts,
session_current_segments, retire_local_receipts) from the partial covering expression
index idx_state_meta_local_policy_target instead of reading and JSON-parsing full
gateway.local_policy.v1:* receipt blobs, while remaining compatible with read-only
legacy databases that lack the index."""

from __future__ import annotations

import json
import sqlite3

import pytest

from persistence import SessionId
from types import SimpleNamespace
from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.local import (
    LOCAL_POLICY_GLOB_LITERAL,
    POLICY_PREFIX,
    local_policy_target_expr,
    retire_local_receipts,
)
from persistence.sqlite.conversation._store.sessions import create_session
from persistence.sqlite.conversation._store.stored_reads import session_current_segments
from persistence.sqlite.conversation.sessions import SQLiteSessionRepository


@pytest.fixture
def holder(tmp_path):
    db = StoreHolder(tmp_path / "state.db")
    yield db
    db.close()


def _seed_receipts(db: StoreHolder, count: int = 50) -> None:
    assert db._conn is not None
    pad = "x" * 1024
    for i in range(count):
        sid = f"local-{i:03d}"
        create_session(db, sid, source="cli")
        db._conn.execute(
            "INSERT INTO state_meta(key, value) VALUES(?, ?)",
            (
                POLICY_PREFIX + sid,
                json.dumps({"entry": {"session_id": sid}, "policy": {"config_json": pad}}),
            ),
        )
    db._conn.commit()


def test_covering_index_created_and_used_by_all_three_queries(holder):
    _seed_receipts(holder, count=50)
    assert holder._conn is not None

    indexes = {row[1] for row in holder._conn.execute("PRAGMA index_list('state_meta')")}
    assert "idx_state_meta_local_policy_target" in indexes

    target_expr = local_policy_target_expr("m.value")
    ids = json.dumps(["local-001", "local-002"])

    live_plan = [
        row[3]
        for row in holder._conn.execute(
            "EXPLAIN QUERY PLAN "
            f"SELECT s.id, (SELECT t.ended_at FROM state_meta m JOIN sessions t ON t.id = {target_expr} "
            f"WHERE m.key GLOB {LOCAL_POLICY_GLOB_LITERAL} "
            "AND m.key >= ? || s.id AND m.key <= ? || s.id) "
            "FROM sessions s WHERE s.id IN (SELECT value FROM json_each(?))",
            (POLICY_PREFIX, POLICY_PREFIX, ids),
        )
    ]
    assert any("COVERING INDEX idx_state_meta_local_policy_target" in detail for detail in live_plan), live_plan

    seg_plan = [
        row[3]
        for row in holder._conn.execute(
            "EXPLAIN QUERY PLAN "
            f"SELECT s.id, (SELECT {target_expr} FROM state_meta m "
            f"WHERE m.key GLOB {LOCAL_POLICY_GLOB_LITERAL} "
            "AND m.key >= ? || s.id AND m.key <= ? || s.id) "
            "FROM sessions s WHERE s.id IN (SELECT value FROM json_each(?))",
            (POLICY_PREFIX, POLICY_PREFIX, ids),
        )
    ]
    assert any("COVERING INDEX idx_state_meta_local_policy_target" in detail for detail in seg_plan), seg_plan

    retire_expr = local_policy_target_expr("value")
    retire_plan = [
        row[3]
        for row in holder._conn.execute(
            "EXPLAIN QUERY PLAN "
            f"SELECT key FROM state_meta WHERE key GLOB {LOCAL_POLICY_GLOB_LITERAL} "
            f"AND {retire_expr} IN (SELECT value FROM json_each(?))",
            (ids,),
        )
    ]
    assert any("COVERING INDEX idx_state_meta_local_policy_target" in detail for detail in retire_plan), retire_plan


def test_behavioral_parity_including_retargeted_and_malformed_receipts(holder):
    repo = SQLiteSessionRepository(SimpleNamespace(holder=lambda: holder))
    try:
        create_session(holder, "owner-1", source="cli")
        create_session(holder, "child-1", source="cli")
        create_session(holder, "malformed-1", source="cli")
        create_session(holder, "plain-1", source="cli")
        assert holder._conn is not None
        holder._conn.execute("UPDATE sessions SET ended_at = 1700000123.0 WHERE id = 'child-1'")
        holder._conn.executemany(
            "INSERT INTO state_meta(key, value) VALUES(?, ?)",
            [
                (POLICY_PREFIX + "owner-1", json.dumps({"entry": {"session_id": "child-1"}})),
                (POLICY_PREFIX + "malformed-1", "{not valid json"),
                (POLICY_PREFIX + "plain-1", json.dumps({"entry": {"session_id": "plain-1"}})),
                ("gateway.other.v1:unrelated", json.dumps({"entry": {"session_id": "child-1"}})),
            ],
        )
        holder._conn.commit()

        facts = repo.live_listing_facts([
            SessionId("owner-1"),
            SessionId("malformed-1"),
            SessionId("plain-1"),
        ])
        assert facts.rows[SessionId("owner-1")].target_ended_at == 1700000123.0
        assert facts.rows[SessionId("malformed-1")].target_ended_at is None
        assert facts.rows[SessionId("plain-1")].target_ended_at is None

        segs = session_current_segments(holder, ["owner-1", "malformed-1", "plain-1"])
        assert segs == {"owner-1": "child-1", "malformed-1": "malformed-1", "plain-1": "plain-1"}

        retire_local_receipts(holder._conn, ["child-1"])
        holder._conn.commit()
        remaining = {
            row[0] for row in holder._conn.execute("SELECT key FROM state_meta")
        }
        assert POLICY_PREFIX + "owner-1" not in remaining
        assert POLICY_PREFIX + "malformed-1" in remaining
        assert POLICY_PREFIX + "plain-1" in remaining
        assert "gateway.other.v1:unrelated" in remaining
    finally:
        pass


def test_read_only_legacy_db_without_index_works_without_schema_change(tmp_path):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE sessions(id TEXT PRIMARY KEY, end_reason TEXT, parent_session_id TEXT);"
        "CREATE TABLE state_meta(key TEXT PRIMARY KEY, value TEXT);"
    )
    conn.executemany(
        "INSERT INTO sessions VALUES(?, ?, ?)",
        [("owner-1", None, None), ("child-1", None, None)],
    )
    conn.execute(
        "INSERT INTO state_meta VALUES(?, ?)",
        (POLICY_PREFIX + "owner-1", json.dumps({"entry": {"session_id": "child-1"}})),
    )
    conn.commit()
    conn.close()

    before = path.read_bytes()
    db = StoreHolder(path, read_only=True)
    try:
        assert session_current_segments(db, ["owner-1"]) == {"owner-1": "child-1"}
    finally:
        db.close()
    assert path.read_bytes() == before
