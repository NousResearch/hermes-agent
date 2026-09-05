"""Regression tests for #103840: `.recover` resurrects orphan FTS5 shadow tables.

A sqlite3 `.recover`-recovered state.db re-emits FTS5 shadow tables as ordinary
tables but loses the virtual parents, so the gateway fails startup with
`fts5: error creating shadow table messages_fts_data: table 'messages_fts_data'
already exists` and SessionDB is unavailable until the orphans are dropped.

Fix under test: `_ensure_fts_schema` detects the shadow-name collision, verifies
the virtual parent is absent, drops only the orphaned objects (never shadows owned
by a healthy virtual table), retries the DDL, and leaves the store fully usable
with FTS rebuilt from canonical `messages`.
"""

import sqlite3

import pytest

from hermes_state import SessionDB


def _orphan_the_base_fts(db_path):
    """Turn a healthy state.db into the exact post-`.recover` state: drop the
    `messages_fts` virtual parent while keeping its shadow tables (renamed as
    ordinary tables in sqlite_master) and triggers."""
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    # Drop the virtual parent only; FTS5 keeps shadow tables when the parent is
    # dropped via writable_schema surgery (a real DROP TABLE would cascade).
    conn.execute("PRAGMA writable_schema=ON")
    conn.execute("DELETE FROM sqlite_master WHERE name='messages_fts'")
    conn.execute("PRAGMA writable_schema=OFF")
    conn.close()


class TestOrphanFtsShadowRepair:
    def test_orphan_base_fts_shadows_self_heal_on_open(self, tmp_path):
        """Post-recover DB (shadows without parent) opens cleanly and ends with a
        working virtual table + populated index."""
        db_path = tmp_path / "state.db"
        db = SessionDB(db_path=db_path)
        db.create_session("s1", source="cli")
        db.append_message("s1", role="user", content="recoverable needle content")
        db.close()

        _orphan_the_base_fts(db_path)

        # Pre-fix this open raised OperationalError("fts5: error creating shadow
        # table messages_fts_data: table 'messages_fts_data' already exists").
        db2 = SessionDB(db_path=db_path)
        try:
            rows = db2.search_messages("recoverable")
            assert any("recoverable" in (r.get("snippet") or "") for r in rows)
            with db2._lock:
                cur = db2._conn.execute(
                    "SELECT count(*) FROM sqlite_master WHERE type='table' "
                    "AND name='messages_fts' AND sql LIKE 'CREATE VIRTUAL TABLE%'"
                )
                assert cur.fetchone()[0] == 1
        finally:
            db2.close()

    def test_healthy_trigram_survives_base_family_cleanup(self, tmp_path):
        """When only the base family is orphaned, shadows owned by the healthy
        `messages_fts_trigram` virtual table must NOT be dropped."""
        db_path = tmp_path / "state.db"
        db = SessionDB(db_path=db_path)
        db.create_session("s1", source="cli")
        db.append_message("s1", role="user", content="trigram survival probe")
        db.close()

        _orphan_the_base_fts(db_path)
        # Confirm trigram parent is alive before the heal.
        conn = sqlite3.connect(str(db_path))
        before = conn.execute(
            "SELECT count(*) FROM sqlite_master WHERE name LIKE 'messages_fts_trigram%'"
        ).fetchone()[0]
        conn.close()
        assert before > 0

        db2 = SessionDB(db_path=db_path)
        try:
            conn = sqlite3.connect(str(db_path))
            after = conn.execute(
                "SELECT count(*) FROM sqlite_master WHERE name LIKE 'messages_fts_trigram%'"
            ).fetchone()[0]
            alive = conn.execute(
                "SELECT count(*) FROM sqlite_master WHERE type='table' "
                "AND name='messages_fts_trigram' AND sql LIKE 'CREATE VIRTUAL TABLE%'"
            ).fetchone()[0]
            conn.close()
            assert after == before  # nothing owned by trigram was dropped
            assert alive == 1
        finally:
            db2.close()

    def test_full_orphan_rebuild_restores_search_and_triggers(self, tmp_path):
        """The VPS case end-to-end: both virtual parents gone, all shadows + view +
        triggers orphaned. Open must clean everything orphaned, recreate both FTS
        families from canonical messages, and message writes must work again."""
        db_path = tmp_path / "state.db"
        db = SessionDB(db_path=db_path)
        db.create_session("s1", source="cli")
        db.append_message("s1", role="user", content="full rebuild probe")
        db.close()

        conn = sqlite3.connect(str(db_path), isolation_level=None)
        conn.execute("PRAGMA writable_schema=ON")
        conn.execute("DELETE FROM sqlite_master WHERE name IN ('messages_fts','messages_fts_trigram')")
        conn.execute("PRAGMA writable_schema=OFF")
        conn.close()

        db2 = SessionDB(db_path=db_path)
        try:
            with db2._lock:
                cur = db2._conn.execute(
                    "SELECT count(*) FROM sqlite_master WHERE type='table' "
                    "AND sql LIKE 'CREATE VIRTUAL TABLE%' AND name LIKE 'messages_fts%'"
                )
                assert cur.fetchone()[0] == 2
            # Writes flow again through the recreated triggers.
            db2.append_message("s1", role="user", content="post-heal write")
        finally:
            db2.close()

    def test_legitimate_shadows_never_touched_when_parent_present(self, tmp_path):
        """Healthy DB: an unrelated OperationalError mentioning 'already exists'
        must not trigger orphan cleanup, and a normal open leaves shadows intact."""
        db_path = tmp_path / "state.db"
        db = SessionDB(db_path=db_path)
        db.create_session("s1", source="cli")
        db.append_message("s1", role="user", content="do not touch my shadows")
        db.close()

        db2 = SessionDB(db_path=db_path)
        try:
            conn = sqlite3.connect(str(db_path))
            shadows = conn.execute(
                "SELECT count(*) FROM sqlite_master WHERE name LIKE 'messages_fts\\_%' ESCAPE '\\'"
            ).fetchone()[0]
            conn.close()
            assert shadows >= 4
        finally:
            db2.close()

    def test_persistent_collision_falls_back_to_deferred_rebuild_contract(self, tmp_path, monkeypatch):
        """#93200 contract: when the orphan collision survives one cleanup retry
        (a concurrent opener keeps re-creating shadows), this open must NOT walk
        away with live triggers over an unrebuilt index. It must persist the
        fts_stale breadcrumb, drop the just-created triggers, and degrade."""
        db_path = tmp_path / "state.db"
        db = SessionDB(db_path=db_path)
        db.create_session("s1", source="cli")
        db.append_message("s1", role="user", content="deferred contract probe")
        db.close()

        _orphan_the_base_fts(db_path)

        # Simulate a concurrent opener that keeps re-creating the orphans: sabotage
        # the cleanup to a no-op. Sweep drops nothing, the DDL collides for real
        # (orphans are actually on disk), the retry cleanup again drops nothing,
        # and the deferred fallback (#93200) must engage.
        from hermes_state_fts import SessionFtsSetupMixin

        def noop_drop(self, cursor, table_name):
            return False

        monkeypatch.setattr(
            SessionFtsSetupMixin, "_drop_orphan_fts_shadow_objects", noop_drop
        )

        db2 = SessionDB(db_path=db_path)
        try:
            conn = sqlite3.connect(str(db_path))
            stale = conn.execute(
                "SELECT 1 FROM state_meta WHERE key='fts_stale'"
            ).fetchone()
            triggers = conn.execute(
                "SELECT count(*) FROM sqlite_master WHERE type='trigger' "
                "AND name LIKE 'messages_fts_%'"
            ).fetchone()[0]
            conn.close()
            assert stale is not None          # breadcrumb persisted
            assert triggers == 0              # no live triggers over the unrebuilt index
            assert db2._fts_stale is True
            assert db2._fts_enabled is False
        finally:
            db2.close()
