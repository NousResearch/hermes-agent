"""Regression tests for the session.list effective_last_active denormalization."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path

import pytest

import hermes_state
from hermes_state import SessionDB


def _db(tmp_path) -> SessionDB:
    return SessionDB(db_path=tmp_path / "state.db")


def _set_session_times(db: SessionDB, session_id: str, *, started_at: float | None = None,
                       ended_at: float | None = None, end_reason: str | None = None) -> None:
    sets: list[str] = []
    params: list[object] = []
    if started_at is not None:
        sets.append("started_at = ?")
        params.append(started_at)
    if ended_at is not None or end_reason is not None:
        sets.extend(["ended_at = ?", "end_reason = ?"])
        params.extend([ended_at, end_reason])
    if not sets:
        return
    with db._lock:
        db._conn.execute(f"UPDATE sessions SET {', '.join(sets)} WHERE id = ?", (*params, session_id))
        db._conn.commit()
    db.recompute_effective_last_active(session_id)


def _stored(db: SessionDB, session_id: str):
    with db._lock:
        row = db._conn.execute(
            "SELECT effective_last_active FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
    return None if row is None else row["effective_last_active"]


def _assert_matches_oracle(db: SessionDB, session_id: str) -> None:
    assert _stored(db, session_id) == db.expected_effective_last_active(session_id)


def _create_session(db: SessionDB, session_id: str, source: str = "cli", *, started_at: float,
                    parent_session_id: str | None = None, model_config: dict | None = None) -> None:
    db.create_session(
        session_id,
        source,
        parent_session_id=parent_session_id,
        model_config=model_config,
    )
    _set_session_times(db, session_id, started_at=started_at)


def _append(db: SessionDB, session_id: str, content: str, ts: float, *, role: str = "user") -> None:
    db.append_message(session_id, role, content, timestamp=ts)


def _ordered_ids(db: SessionDB, **kwargs) -> list[str]:
    return [row["id"] for row in db.list_sessions_rich(order_by_last_active=True, **kwargs)]


def _rows_bytes(rows: list[dict]) -> bytes:
    return json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()


class TestEffectiveLastActiveSchema:
    def test_schema_adds_index_and_inner_limit_is_covering_index(self, tmp_path):
        db = _db(tmp_path)
        try:
            columns = {row["name"] for row in db._conn.execute("PRAGMA table_info(sessions)")}
            assert "effective_last_active" in columns

            indexes = {row["name"] for row in db._conn.execute("PRAGMA index_list(sessions)")}
            assert "idx_sessions_effective_last_active" in indexes

            queries = [
                (
                    "default",
                    "SELECT id FROM sessions "
                    "WHERE effective_last_active IS NOT NULL AND archived = ? "
                    "ORDER BY effective_last_active DESC, started_at DESC, id DESC "
                    "LIMIT ?",
                    (0, 400),
                    "idx_sessions_effective_last_active",
                ),
                (
                    "source-filter",
                    "SELECT id FROM sessions "
                    "WHERE effective_last_active IS NOT NULL AND archived = ? AND source = ? "
                    "ORDER BY effective_last_active DESC, started_at DESC, id DESC "
                    "LIMIT ?",
                    (0, "cli", 400),
                    "idx_sessions_source_effective_last_active",
                ),
                (
                    "id-search",
                    "SELECT id FROM sessions "
                    "WHERE effective_last_active IS NOT NULL AND archived = ? AND id LIKE ? ESCAPE '\\' "
                    "ORDER BY effective_last_active DESC, started_at DESC, id DESC "
                    "LIMIT ?",
                    (0, "%root%", 400),
                    "idx_sessions_effective_last_active",
                ),
            ]
            for label, sql, params, index_name in queries:
                plan = "\n".join(
                    row["detail"]
                    for row in db._conn.execute(f"EXPLAIN QUERY PLAN {sql}", params)
                )
                assert f"USING COVERING INDEX {index_name}" in plan, (label, plan)
                assert "USE TEMP B-TREE" not in plan, (label, plan)
        finally:
            db.close()

    def test_effective_last_active_schema_rollback_order(self, tmp_path):
        if sqlite3.sqlite_version_info < (3, 35, 0):
            pytest.skip("DROP COLUMN requires SQLite >= 3.35")
        db = _db(tmp_path)
        db.close()

        conn = sqlite3.connect(tmp_path / "state.db")
        try:
            conn.execute("DROP INDEX IF EXISTS idx_sessions_source_effective_last_active")
            conn.execute("DROP INDEX IF EXISTS idx_sessions_effective_last_active")
            conn.execute("ALTER TABLE sessions DROP COLUMN effective_last_active")
            columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
            indexes = {row[1] for row in conn.execute("PRAGMA index_list(sessions)")}
            assert "effective_last_active" not in columns
            assert "idx_sessions_effective_last_active" not in indexes
        finally:
            conn.close()

    def test_migration_backfill_is_idempotent_and_visibility_encoded_by_null(self, tmp_path):
        db_path = tmp_path / "state.db"
        db = SessionDB(db_path=db_path)
        try:
            _create_session(db, "visible", started_at=10.0)
            _append(db, "visible", "visible", 30.0)
            _create_session(db, "tool-root", "tool", started_at=50.0)
            _append(db, "tool-root", "hidden", 60.0)
            _create_session(db, "root", started_at=100.0)
            db.end_session("root", "compression")
            _create_session(db, "tip", started_at=110.0, parent_session_id="root")
            _append(db, "tip", "tip", 140.0)

            with db._lock:
                db._conn.execute("UPDATE sessions SET effective_last_active = NULL")
                db._conn.execute("UPDATE schema_version SET version = ?", (hermes_state.SCHEMA_VERSION - 1,))
                db._conn.commit()
        finally:
            db.close()

        db = SessionDB(db_path=db_path)
        try:
            first = {sid: _stored(db, sid) for sid in ("visible", "tool-root", "root", "tip")}
            db.backfill_effective_last_active()
            second = {sid: _stored(db, sid) for sid in ("visible", "tool-root", "root", "tip")}

            assert first == second
            assert first["visible"] == 30.0
            assert first["tool-root"] is None
            assert first["root"] == 140.0
            assert first["tip"] is None
        finally:
            db.close()

    def test_v18_v2_stale_backfill_marker_repairs_on_open_without_manual_backfill(self, tmp_path):
        db_path = tmp_path / "state.db"
        db = SessionDB(db_path=db_path)
        try:
            _create_session(db, "stale-root", started_at=100.0)
            _append(db, "stale-root", "stale root", 120.0)
            db.end_session("stale-root", "compression")
            _create_session(db, "stale-tip", started_at=130.0, parent_session_id="stale-root")
            _append(db, "stale-tip", "fresh continuation", 200.0)
            _create_session(db, "stale-other", started_at=140.0)
            _append(db, "stale-other", "middle standalone", 150.0)
            _create_session(db, "archived-stale", started_at=300.0)
            _append(db, "archived-stale", "archived newest", 400.0)
            db.set_session_archived("archived-stale", True)

            with db._lock:
                # Simulate the live failure shape: schema_version is already current
                # and the already-deployed v2 backfill marker is present, but the
                # denormalized root value is stale behind its continuation tip.
                conn = db._conn
                assert conn is not None
                conn.execute(
                    "UPDATE sessions SET effective_last_active = ? WHERE id = ?",
                    (120.0, "stale-root"),
                )
                conn.execute(
                    "UPDATE schema_version SET version = ?",
                    (hermes_state.SCHEMA_VERSION,),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO state_meta (key, value) VALUES (?, ?)",
                    (hermes_state._EFFECTIVE_LAST_ACTIVE_BACKFILL_META_KEY, "2"),
                )
                conn.commit()
        finally:
            db.close()

        db = SessionDB(db_path=db_path)
        try:
            assert _stored(db, "stale-root") == db.expected_effective_last_active("stale-root") == 200.0
            with db._lock:
                conn = db._conn
                assert conn is not None
                marker = conn.execute(
                    "SELECT value FROM state_meta WHERE key = ?",
                    (hermes_state._EFFECTIVE_LAST_ACTIVE_BACKFILL_META_KEY,),
                ).fetchone()
            assert marker is not None
            assert marker["value"] == hermes_state._EFFECTIVE_LAST_ACTIVE_BACKFILL_VERSION

            cases = [
                ("default", {}),
                ("include_archived", {"include_archived": True}),
                ("id_query", {"id_query": "stale"}),
            ]
            for label, kwargs in cases:
                actual = db.list_sessions_rich(
                    order_by_last_active=True,
                    limit=10,
                    **kwargs,
                )
                expected = db.list_sessions_rich_cte_oracle(limit=10, **kwargs)
                assert _rows_bytes(actual) == _rows_bytes(expected), label
        finally:
            db.close()

    def test_legacy_database_without_column_adds_column_before_index(self, tmp_path):
        if sqlite3.sqlite_version_info < (3, 35, 0):
            pytest.skip("DROP COLUMN requires SQLite >= 3.35")
        db_path = tmp_path / "state.db"
        db = SessionDB(db_path=db_path)
        try:
            _create_session(db, "visible", started_at=10.0)
            _append(db, "visible", "visible", 30.0)
        finally:
            db.close()

        conn = sqlite3.connect(db_path)
        try:
            conn.execute("DROP INDEX IF EXISTS idx_sessions_source_effective_last_active")
            conn.execute("DROP INDEX IF EXISTS idx_sessions_effective_last_active")
            conn.execute("ALTER TABLE sessions DROP COLUMN effective_last_active")
            conn.execute("UPDATE schema_version SET version = ?", (hermes_state.SCHEMA_VERSION - 1,))
            conn.commit()
        finally:
            conn.close()

        db = SessionDB(db_path=db_path)
        try:
            columns = {row["name"] for row in db._conn.execute("PRAGMA table_info(sessions)")}
            indexes = {row["name"] for row in db._conn.execute("PRAGMA index_list(sessions)")}
            assert "effective_last_active" in columns
            assert "idx_sessions_effective_last_active" in indexes
            assert _stored(db, "visible") == 30.0
        finally:
            db.close()


class TestEffectiveLastActiveQueryAndMaintenance:
    def test_two_stage_query_matches_cte_oracle_with_deny_chain_ties_and_projection(self, tmp_path):
        db = _db(tmp_path)
        try:
            _create_session(db, "root", started_at=10.0)
            _append(db, "root", "root", 11.0)
            db.end_session("root", "compression")
            _create_session(db, "mid", started_at=20.0, parent_session_id="root")
            _append(db, "mid", "mid", 90.0)
            db.end_session("mid", "compression")
            _create_session(db, "tip", started_at=30.0, parent_session_id="mid")
            _append(db, "tip", "tip", 95.0)

            _create_session(db, "tool-member-root", started_at=40.0)
            db.end_session("tool-member-root", "compression")
            _create_session(db, "tool-member", "tool", started_at=45.0, parent_session_id="tool-member-root")
            _append(db, "tool-member", "tool member wins max", 120.0)

            _create_session(db, "tool-root", "tool", started_at=200.0)
            _append(db, "tool-root", "hidden noisy root", 300.0)

            _create_session(db, "tie-a", started_at=70.0)
            _append(db, "tie-a", "same timestamp", 500.0)
            _create_session(db, "tie-b", started_at=80.0)
            _append(db, "tie-b", "same timestamp", 500.0)
            _create_session(db, "archived", started_at=600.0)
            _append(db, "archived", "archived still has recency", 700.0)
            before_archive = _stored(db, "archived")
            db.set_session_archived("archived", True)
            assert _stored(db, "archived") == before_archive

            expected = db.list_sessions_rich_cte_oracle(limit=10)
            actual = db.list_sessions_rich(order_by_last_active=True, limit=10)
            expected_archived = db.list_sessions_rich_cte_oracle(limit=10, archived_only=True)
            actual_archived = db.list_sessions_rich(
                order_by_last_active=True,
                limit=10,
                archived_only=True,
            )

            assert [row["id"] for row in actual] == [row["id"] for row in expected]
            assert [row.get("_lineage_root_id") for row in actual] == [row.get("_lineage_root_id") for row in expected]
            assert [row["id"] for row in actual_archived] == [row["id"] for row in expected_archived]
            assert [row.get("_lineage_root_id") for row in actual_archived] == [
                row.get("_lineage_root_id") for row in expected_archived
            ]
            assert "tool-root" not in [row["id"] for row in actual]
            assert [row["id"] for row in actual_archived] == ["archived"]
            assert _stored(db, "tool-member-root") == 120.0
        finally:
            db.close()

    def test_insert_monotonic_bump_does_not_lower_and_empty_root_falls_back_to_started_at(self, tmp_path):
        db = _db(tmp_path)
        try:
            _create_session(db, "root", started_at=10.0)
            _append(db, "root", "newer", 100.0)
            _append(db, "root", "stale", 50.0)
            assert _stored(db, "root") == 100.0

            db.clear_messages("root")
            assert _stored(db, "root") == 10.0
            assert _ordered_ids(db) == ["root"]
        finally:
            db.close()

    def test_source_deny_list_change_recomputes_visibility(self, tmp_path):
        db = _db(tmp_path)
        try:
            _create_session(db, "root", "tool", started_at=10.0)
            _append(db, "root", "initially hidden", 100.0)
            assert _stored(db, "root") is None

            db.record_gateway_session_peer("root", source="cli", session_key="key-1")
            assert _stored(db, "root") == 100.0
            assert _ordered_ids(db) == ["root"]

            db.record_gateway_session_peer("root", source="tool", session_key="key-1")
            assert _stored(db, "root") is None
            assert _ordered_ids(db) == []
        finally:
            db.close()

    def test_parent_changes_recompute_both_roots_for_detach_and_coalesce_merge(self, tmp_path):
        db = _db(tmp_path)
        try:
            _create_session(db, "root", started_at=10.0)
            _append(db, "root", "root", 20.0)
            db.end_session("root", "compression")
            _create_session(db, "child", started_at=30.0, parent_session_id="root")
            _append(db, "child", "child max", 400.0)
            assert _stored(db, "root") == 400.0
            assert _stored(db, "child") is None

            db._set_parent_session_id("child", None)
            assert _stored(db, "root") == 20.0
            assert _stored(db, "child") == 400.0
            assert _ordered_ids(db, limit=2) == ["child", "root"]

            _create_session(db, "parent", started_at=300.0)
            _append(db, "parent", "parent", 310.0)
            db.end_session("parent", "compression")
            db.create_session("child", "cli", parent_session_id="parent")  # COALESCE fills parent
            assert _stored(db, "child") is None
            assert _stored(db, "parent") == 400.0
        finally:
            db.close()

    def test_reopen_and_downward_recompute_drop_stale_high_values(self, tmp_path):
        db = _db(tmp_path)
        try:
            _create_session(db, "root", started_at=10.0)
            _append(db, "root", "root", 20.0)
            db.end_session("root", "compression")
            _create_session(db, "tip", started_at=30.0, parent_session_id="root")
            _append(db, "tip", "tip max", 200.0)
            assert _stored(db, "root") == 200.0

            db.reopen_session("root")
            assert _stored(db, "root") == 20.0
            assert _stored(db, "tip") is None
            _assert_matches_oracle(db, "root")
        finally:
            db.close()

    @pytest.mark.parametrize("delete_mode", [
        "delegate_cascade",
        "delete_session",
        "delete_sessions",
        "delete_empty_sessions",
        "prune_sessions",
    ])
    def test_delete_orphan_sites_promote_surviving_continuation_to_visible_root(self, tmp_path, delete_mode):
        db = _db(tmp_path)
        try:
            parent_id = "parent"
            if delete_mode == "delegate_cascade":
                _create_session(db, "grand", started_at=1.0)
                db.create_session(
                    parent_id,
                    "cli",
                    parent_session_id="grand",
                    model_config={"_delegate_from": "grand"},
                )
                _set_session_times(db, parent_id, started_at=10.0)
            else:
                _create_session(db, parent_id, started_at=10.0)
            db.end_session(parent_id, "compression")
            _create_session(db, "child", started_at=20.0, parent_session_id=parent_id)
            _append(db, "child", "survivor", 50.0)
            assert _stored(db, "child") is None

            if delete_mode == "delegate_cascade":
                db.delete_session("grand")
            elif delete_mode == "delete_session":
                db.delete_session(parent_id)
            elif delete_mode == "delete_sessions":
                db.delete_sessions([parent_id])
            elif delete_mode == "delete_empty_sessions":
                db.clear_messages(parent_id)
                db.delete_empty_sessions()
            elif delete_mode == "prune_sessions":
                _set_session_times(db, parent_id, started_at=1.0, ended_at=2.0, end_reason="compression")
                db.prune_sessions(older_than_days=0)

            assert db.get_session(parent_id) is None
            child = db.get_session("child")
            assert child is not None
            assert child["parent_session_id"] is None
            assert _stored(db, "child") == 50.0
            assert "child" in _ordered_ids(db)
        finally:
            db.close()

    def test_mid_chain_delete_recomputes_surviving_root_and_orphaned_child(self, tmp_path):
        db = _db(tmp_path)
        try:
            _create_session(db, "root", started_at=10.0)
            _append(db, "root", "root", 20.0)
            db.end_session("root", "compression")
            _create_session(db, "mid", started_at=30.0, parent_session_id="root")
            _append(db, "mid", "mid", 200.0)
            db.end_session("mid", "compression")
            _create_session(db, "tip", started_at=40.0, parent_session_id="mid")
            _append(db, "tip", "tip", 300.0)
            assert _stored(db, "root") == 300.0

            db.delete_session("mid")

            assert _stored(db, "root") == 20.0
            assert db.get_session("tip")["parent_session_id"] is None
            assert _stored(db, "tip") == 300.0
        finally:
            db.close()

    def test_reconcile_audit_logs_drift(self, tmp_path, caplog):
        db = _db(tmp_path)
        try:
            _create_session(db, "root", started_at=10.0)
            _append(db, "root", "root", 20.0)
            with db._lock:
                db._conn.execute(
                    "UPDATE sessions SET effective_last_active = ? WHERE id = ?",
                    (999.0, "root"),
                )
                db._conn.commit()

            caplog.set_level(logging.WARNING, logger="hermes_state")
            drift = db.audit_effective_last_active(limit=10)
            assert drift == [{"id": "root", "stored": 999.0, "expected": 20.0}]
            assert "effective_last_active drift" in caplog.text
        finally:
            db.close()


def test_parent_session_id_writers_are_maintenance_adjacent():
    source = (Path(__file__).parents[2] / "hermes_state.py").read_text()
    writers = list(
        re.finditer(
            r"UPDATE\s+sessions\s+SET[^\n]*parent_session_id\s*=|"
            r"parent_session_id\s*=\s*COALESCE|"
            r"INSERT\s+INTO\s+sessions\s*\([^)]*parent_session_id",
            source,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    assert len(writers) == 8
    for match in writers:
        window = source[match.start(): match.start() + 900]
        assert "effective_last_active" in window or "_recompute" in window or "INSERT INTO sessions" in window


def test_list_deny_sources_single_sourced():
    source = (Path(__file__).parents[2] / "hermes_state.py").read_text()
    assert getattr(hermes_state, "_LIST_DENY_SOURCES", None) == ("tool",)
    assert source.count("_LIST_DENY_SOURCES") >= 5
    list_path_region = source[source.index("def list_sessions_rich"): source.index("def list_cron_job_runs")]
    assert "'tool'" not in list_path_region
    assert '"tool"' not in list_path_region
