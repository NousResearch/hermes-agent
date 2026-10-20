"""Durable activity and bounded session-list hydration regressions for #119408."""
import sqlite3

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(tmp_path / "test_state.db")
    yield session_db
    session_db.close()


class TestSessionListPreviewHydration:
    def test_last_active_uses_newer_message_over_stale_heartbeat(self, db):
        """Rate-limited heartbeats can lag message writes; last_active must take max."""
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "hello")
        with db._lock:
            db._conn.execute(
                "UPDATE messages SET timestamp=? WHERE session_id=?",
                (1_700_000_800.0, "s1"),
            )
            db._conn.execute("UPDATE sessions SET last_activity_at = NULL WHERE id = ?", ("s1",))
            db._conn.commit()
        db.touch_session_activity("s1", 1_700_000_500.0, description="api")  # older than message
        assert db.list_sessions_rich()[0]["last_active"] == 1_700_000_800.0

    def test_last_active_order_backfills_legacy_message_before_bounded_selection(self, tmp_path):
        """A pre-stamp store must hydrate its newest transcript before the candidate limit applies."""
        db_path = tmp_path / "state.db"
        state = SessionDB(db_path=db_path)
        try:
            state.create_session("newest", "cli")
            state.create_session("older", "cli")
            state.append_message("newest", "user", "new transcript", timestamp=300.0)
            state.append_message("older", "user", "old transcript", timestamp=200.0)
        finally:
            state.close()

        conn = sqlite3.connect(db_path)
        conn.execute("UPDATE schema_version SET version = 30")
        conn.execute("DELETE FROM state_meta WHERE key = 'session_activity_message_backfill_version'")
        conn.executemany(
            "UPDATE sessions SET started_at = ?, last_activity_at = ? WHERE id = ?",
            [(100.0, None, "newest"), (200.0, 200.0, "older")],
        )
        conn.commit()
        conn.close()

        migrated = SessionDB(db_path=db_path)
        try:
            rows = migrated.list_sessions_rich(limit=1, order_by_last_active=True,
                                               project_compression_tips=False)
            assert [row["id"] for row in rows] == ["newest"]
        finally:
            migrated.close()

    def test_last_active_order_maps_recent_continuation_to_root(self, db):
        """A recent compression child must admit its logical root before projecting the tip."""
        db.create_session("root", "cli")
        db._conn.execute("UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = 'compression' WHERE id = ?",
                         (100.0, 150.0, "root"))
        db.create_session("tip", "cli", parent_session_id="root")
        db.append_message("tip", "user", "latest continuation", timestamp=300.0)
        db.create_session("competing", "cli")
        db.append_message("competing", "user", "middle activity", timestamp=250.0)
        db._conn.commit()

        rows = db.list_sessions_rich(limit=2, order_by_last_active=True)

        assert [row["id"] for row in rows] == ["tip", "competing"]
        assert rows[0]["_lineage_root_id"] == "root"

    def test_last_active_candidate_window_deduplicates_compression_chain(self, db):
        """Several recent physical continuations must not consume one logical page twice."""
        db.create_session("root-a", "cli")
        db.append_message("root-a", "user", "root-a", timestamp=100.0)
        db.end_session("root-a", "compression")
        db.create_session("child-a1", "cli", parent_session_id="root-a")
        db.append_message("child-a1", "user", "child-a1", timestamp=400.0)
        db.end_session("child-a1", "compression")
        db.create_session("child-a2", "cli", parent_session_id="child-a1")
        db.append_message("child-a2", "user", "child-a2", timestamp=500.0)
        db.create_session("logical-b", "cli")
        db.append_message("logical-b", "user", "logical-b", timestamp=300.0)

        rows = db.list_sessions_rich(limit=2, order_by_last_active=True)

        assert [row["id"] for row in rows] == ["child-a2", "logical-b"]

    def test_hidden_delegate_does_not_consume_bounded_candidate_slot(self, db):
        """A recent delegate child stays hidden before the logical candidate limit applies."""
        db.create_session("root-old", "cli")
        db.append_message("root-old", "user", "root-old", timestamp=100.0)
        db.create_session("root-new", "cli")
        db.append_message("root-new", "user", "root-new", timestamp=200.0)
        db.create_session(
            "delegate", "subagent", parent_session_id="root-old",
            model_config={"_delegate_from": "root-old"},
        )
        db.append_message("delegate", "user", "hidden", timestamp=300.0)

        rows = db.list_sessions_rich(
            limit=2, order_by_last_active=True, project_compression_tips=False,
        )

        assert [row["id"] for row in rows] == ["root-new", "root-old"]

    def test_append_message_advances_activity_without_regressing_heartbeat(self, db):
        db.create_session("session", "cli")
        db.append_message("session", "user", "known timestamp", timestamp=200.0)
        assert db.get_session("session")["last_activity_at"] == 200.0

        db.touch_session_activity("session", 300.0)
        db.append_message("session", "user", "historical timestamp", timestamp=100.0)
        assert db.get_session("session")["last_activity_at"] == 300.0

    def test_batch_append_advances_activity_to_latest_persisted_timestamp(self, db):
        db.create_session("session", "cli")

        db.append_messages_batch("session", [
            {"role": "user", "content": "older", "timestamp": 100.0},
            {"role": "assistant", "content": "newer", "timestamp": 200.0},
        ])

        assert db.get_session("session")["last_activity_at"] == 200.0

    def test_delegation_delivery_advances_activity(self, db):
        db.create_session("session", "cli")

        db.append_delegation_delivery("session", "completed", {"delegation_id": "delivery-1"})

        persisted = db._conn.execute(
            "SELECT timestamp FROM messages WHERE session_id = ?", ("session",)
        ).fetchone()[0]
        assert db.get_session("session")["last_activity_at"] == persisted

    def test_preview_subquery_only_runs_for_limited_rows(self, db, monkeypatch):
        """Recent sidebar hydration must not inspect every session's messages."""
        import hermes_state_sessions

        for i in range(12):
            session_id = f"session-{i}"
            db.create_session(session_id, "cli")
            db.append_message(session_id, "user", f"preview-{i}")
            db._conn.execute(
                "UPDATE sessions SET last_activity_at = ? WHERE id = ?", (i, session_id),
            )
        db.set_session_pinned("session-0", True)
        db._conn.commit()

        preview_calls: list[str] = []
        def read_all(sql, params=()):
            with db._read_ctx() as conn:
                conn.create_function(
                    "preview_probe", 1, lambda content: preview_calls.append(content) or content,
                )
                return conn.execute(sql, params).fetchall()

        monkeypatch.setattr(db, "_read_all", read_all)
        monkeypatch.setattr(
            hermes_state_sessions,
            "_sql_session_last_active_by_id",
            lambda session_id: f"""(
                (SELECT last_activity_at FROM sessions _act_s WHERE _act_s.id = {session_id})
                + 0 * COALESCE((SELECT preview_probe(m.content) IS NOT NULL FROM messages m
                                WHERE m.session_id = {session_id}
                                ORDER BY m.timestamp, m.id LIMIT 1), 0)
            )""",
        )
        monkeypatch.setattr(
            hermes_state_sessions,
            "_sql_preview_raw",
            lambda: """COALESCE(
                        (SELECT preview_probe(m.content)
                         FROM messages m
                         WHERE m.session_id = s.id AND m.role = 'user'
                         ORDER BY m.timestamp, m.id LIMIT 1),
                        ''
                    ) AS _preview_raw""",
        )

        rows = db.list_sessions_rich(
            limit=2, offset=1, order_by_last_active=True, include_pinned=True,
            project_compression_tips=False,
        )

        assert {row["id"] for row in rows} == {"session-10", "session-9", "session-0"}
        # Three candidates pay activity lookup; only the two page rows and
        # the pinned back-fill pay preview lookup.
        assert len(preview_calls) == 6


class TestSessionActivityBackfill:
    def test_migration_backfills_message_activity_without_clobbering_heartbeat(self, tmp_path):
        db_path = tmp_path / "state.db"
        state = SessionDB(db_path=db_path)
        try:
            for session_id in ("null", "stale", "heartbeat"):
                state.create_session(session_id, "cli")
            state.append_message("null", "user", "null", timestamp=100.0)
            state.append_message("stale", "user", "stale", timestamp=200.0)
            state.append_message("heartbeat", "user", "heartbeat", timestamp=250.0)
            state.touch_session_activity("heartbeat", 300.0, description="still working")
        finally:
            state.close()

        conn = sqlite3.connect(db_path)
        conn.execute("UPDATE schema_version SET version = 30")
        conn.execute("DELETE FROM state_meta WHERE key = 'session_activity_message_backfill_version'")
        conn.executemany(
            "UPDATE sessions SET last_activity_at = ? WHERE id = ?",
            [(None, "null"), (150.0, "stale"), (300.0, "heartbeat")],
        )
        conn.commit()
        conn.close()

        migrated = SessionDB(db_path=db_path)
        try:
            assert migrated.get_session("null")["last_activity_at"] == 100.0
            assert migrated.get_session("stale")["last_activity_at"] == 200.0
            assert migrated.get_session("heartbeat")["last_activity_at"] == 300.0
            assert migrated.get_session("heartbeat")["last_activity_description"] == "still working"
            assert migrated.get_session("heartbeat")["last_activity_provenance"] == "unknown"
        finally:
            migrated.close()
