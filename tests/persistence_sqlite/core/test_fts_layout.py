"""The FTS layout of the SQLite backend on a real StoreHolder: the open-time FTS repair converging on SQLite builds without the trigram tokenizer, bounded incremental merges on the write path, tool fields indexed after the v23 transition, and the opt-in storage rewrite (``admin._maintenance._FtsStorageOptimizer``: demote to trash, deferred rebuild, chunked teardown, resume after a crash window)."""

import sqlite3
import time

import pytest

from persistence.sqlite import _schema
from persistence.sqlite._holder import StoreHolder
from persistence.sqlite._schema import (
    _fts_external_index_empty_with_messages,
    _fts_table_exists,
    _has_fts_trash,
    _merge_fts_incrementally,
)
from persistence.sqlite._schema_ddl import FTS_SQL, FTS_STORAGE_VERSION, SCHEMA_SQL, SCHEMA_VERSION
from persistence.sqlite.admin import _maintenance
from persistence.sqlite.admin._maintenance import _FtsStorageOptimizer
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.search import fts_rebuild_status, search_messages
from persistence.sqlite.conversation._store.sessions import create_session


class _NoTrigramCursor(sqlite3.Cursor):
    """Simulate a SQLite build with FTS5 but without the trigram tokenizer."""

    def executescript(self, sql_script):
        if "tokenize='trigram'" in sql_script:
            raise sqlite3.OperationalError("no such tokenizer: trigram")
        return super().executescript(sql_script)


class _NoTrigramConnection(sqlite3.Connection):
    def cursor(self, factory=None):
        return super().cursor(factory or _NoTrigramCursor)


@pytest.fixture()
def db(tmp_path):
    """Create a StoreHolder with a temp database file."""
    db_path = tmp_path / "test_state.db"
    holder = StoreHolder(db_path=db_path)
    yield holder
    holder.close()


class TestFtsRebuildLoopWithoutTrigram:
    """A trigram-less SQLite build must not re-index the store on every open.

    The three ``messages_fts_trigram_*`` triggers are declared only by the
    trigram DDL, whose ``CREATE VIRTUAL TABLE ... tokenize='trigram'`` needs a
    tokenizer SQLite only gained in 3.34 — Ubuntu 20.04 (3.31), RHEL/CentOS 8
    (3.26) and Amazon Linux 2 all ship older. ``_ensure_fts_schema``
    soft-fails that DDL there by design, so those three triggers can never
    exist, and startup's "are all six canonical triggers present?" check was
    therefore permanently unsatisfiable: the full FTS repair ran on every
    single ``StoreHolder`` open, holding the write lock, and never converged.

    The v23 repair also clears the deferred-rebuild resume markers, so an
    interrupted ``hermes sessions optimize-storage`` silently lost its place
    every time the store was reopened.
    """

    @staticmethod
    def _trace(monkeypatch, statements, *, trigram):
        """Record every statement a StoreHolder executes during an open.

        ``trigram=False`` additionally routes connections through the
        module's existing ``_NoTrigramConnection``, which raises
        ``no such tokenizer: trigram`` for the trigram DDL exactly as an
        older SQLite does.
        """
        real_connect = sqlite3.connect

        def connect(*args, **kwargs):
            if not trigram:
                kwargs["factory"] = _NoTrigramConnection
            conn = real_connect(*args, **kwargs)
            conn.set_trace_callback(statements.append)
            return conn

        monkeypatch.setattr("persistence.sqlite._holder.sqlite3.connect", connect)

    @staticmethod
    def _rebuilds(statements):
        """External-content 'rebuild' commands seen in *statements*."""
        return [sql for sql in statements if "VALUES('rebuild')" in "".join(sql.split())]

    @staticmethod
    def _legacy_wipes(statements):
        """Legacy inline repair wipes the index before reinserting every row."""
        return [
            sql for sql in statements
            if "".join(sql.split()).upper().startswith("DELETEFROMMESSAGES_FTS")
        ]

    @staticmethod
    def _seed(db_path):
        db = StoreHolder(db_path=db_path)
        try:
            create_session(db, session_id="s1", source="cli")
            for i in range(5):
                append_message(db, "s1", role="user", content=f"payload {i} zebra")
        finally:
            db.close()

    @staticmethod
    def _build_legacy_inline_db(db_path):
        """A v22 store as it exists on a host that never had the tokenizer.

        Only the three base inline triggers were ever creatable there, so —
        unlike the migration fixtures elsewhere in this file — this build
        deliberately has no trigram table and no trigram triggers.
        """
        conn = sqlite3.connect(str(db_path))
        try:
            conn.executescript(SCHEMA_SQL)
            conn.executescript("""
                DROP TABLE IF EXISTS messages_fts;
                DROP TABLE IF EXISTS messages_fts_trigram;
                DROP VIEW IF EXISTS messages_fts_trigram_src;

                CREATE VIRTUAL TABLE messages_fts USING fts5(content);

                CREATE TRIGGER messages_fts_insert AFTER INSERT ON messages BEGIN
                    INSERT INTO messages_fts(rowid, content) VALUES (
                        new.id,
                        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '')
                        || ' ' || COALESCE(new.tool_calls, '')
                    );
                END;

                CREATE TRIGGER messages_fts_delete AFTER DELETE ON messages BEGIN
                    DELETE FROM messages_fts WHERE rowid = old.id;
                END;

                CREATE TRIGGER messages_fts_update
                AFTER UPDATE OF content, tool_name, tool_calls ON messages BEGIN
                    DELETE FROM messages_fts WHERE rowid = old.id;
                    INSERT INTO messages_fts(rowid, content) VALUES (
                        new.id,
                        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '')
                        || ' ' || COALESCE(new.tool_calls, '')
                    );
                END;
            """)
            conn.execute("DELETE FROM schema_version")
            conn.execute("INSERT INTO schema_version (version) VALUES (22)")
            conn.execute(
                "INSERT INTO sessions (id, source, started_at) VALUES ('s1', 'cli', ?)",
                (time.time(),),
            )
            for i in range(5):
                conn.execute(
                    "INSERT INTO messages (session_id, timestamp, role, content) "
                    "VALUES ('s1', ?, 'user', ?)",
                    (time.time(), f"legacy payload {i} zebra"),
                )
            conn.commit()
        finally:
            conn.close()

    def test_fts_trigger_subsets_match_the_ddl(self):
        """The split must track the DDL each trigger actually comes from.

        The gate is only correct while every trigger classified as "trigram"
        is one the trigram DDL creates, and every other one is created by DDL
        that always works. Renaming a trigger without updating its DDL would
        otherwise silently reintroduce an unsatisfiable check.
        """
        from persistence.sqlite._schema import _FTS_BASE_TRIGGERS, _FTS_TRIGRAM_TRIGGERS
        from persistence.sqlite._schema_ddl import (
            FTS_SQL,
            FTS_TRIGRAM_SQL,
            LEGACY_FTS_SQL,
            LEGACY_FTS_TRIGRAM_SQL,
            _FTS_TRIGGERS,
        )

        # Exhaustive and disjoint: nothing may fall out of the classification.
        assert set(_FTS_BASE_TRIGGERS) | set(_FTS_TRIGRAM_TRIGGERS) == set(_FTS_TRIGGERS)
        assert not set(_FTS_BASE_TRIGGERS) & set(_FTS_TRIGRAM_TRIGGERS)

        for name in _FTS_TRIGRAM_TRIGGERS:
            assert name in FTS_TRIGRAM_SQL and name in LEGACY_FTS_TRIGRAM_SQL, (
                f"{name} is classified as trigram-only but the trigram DDL "
                f"does not create it"
            )
        for name in _FTS_BASE_TRIGGERS:
            assert name in FTS_SQL and name in LEGACY_FTS_SQL, (
                f"{name} is classified as always-creatable but the base DDL "
                f"does not create it"
            )
            assert name not in FTS_TRIGRAM_SQL

    def test_missing_trigram_tokenizer_does_not_rebuild_fts_on_every_open(
        self, tmp_path, monkeypatch
    ):
        """v23 branch: the repair must converge instead of firing forever."""
        db_path = tmp_path / "state.db"
        statements = []
        self._trace(monkeypatch, statements, trigram=False)

        self._seed(db_path)

        # Second and third opens of an already-initialised store. The trigram
        # triggers are still absent and always will be, but nothing is
        # actually broken, so there is nothing to repair.
        for _ in range(2):
            statements.clear()
            db = StoreHolder(db_path=db_path)
            try:
                assert db._trigram_available is False
                assert self._rebuilds(statements) == []
                # The narrowed gate must not have cost us a working index.
                assert len(search_messages(db, "zebra")) == 5
            finally:
                db.close()

    def test_legacy_inline_fts_without_trigram_does_not_rebuild_on_every_open(
        self, tmp_path, monkeypatch
    ):
        """Legacy (pre-v23) branch: same gate, same permanent repair loop.

        This path is the more expensive of the two — inline tables have no
        external-content 'rebuild' source, so the repair deletes the index and
        reinserts a concatenation of every row in ``messages``.
        """
        db_path = tmp_path / "legacy.db"
        self._build_legacy_inline_db(db_path)

        statements = []
        self._trace(monkeypatch, statements, trigram=False)

        for _ in range(2):
            statements.clear()
            db = StoreHolder(db_path=db_path)
            try:
                assert _schema._db_has_legacy_inline_fts(db._conn.cursor()) is True
                assert db._trigram_available is False
                assert self._legacy_wipes(statements) == []
                assert len(search_messages(db, "zebra")) == 5
            finally:
                db.close()

    def test_pending_fts_rebuild_markers_survive_a_trigramless_open(
        self, tmp_path, monkeypatch
    ):
        """An interrupted optimize-storage must keep its resume point.

        ``_rebuild_fts_indexes`` clears both markers because a full rebuild
        genuinely does cover every row. Running it unconditionally on a
        trigram-less host therefore threw away the progress of a chunked,
        throttled backfill on the very next open.
        """
        db_path = tmp_path / "state.db"
        statements = []
        self._trace(monkeypatch, statements, trigram=False)

        self._seed(db_path)

        db = StoreHolder(db_path=db_path)
        try:
            for key, value in (
                ("fts_rebuild_high_water", "30"),
                ("fts_rebuild_progress", "10"),
            ):
                db._conn.execute(
                    "INSERT INTO state_meta (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (key, value),
                )
            db._conn.commit()
        finally:
            db.close()

        db = StoreHolder(db_path=db_path)
        try:
            assert db.get_meta("fts_rebuild_high_water") == "30"
            assert db.get_meta("fts_rebuild_progress") == "10"
        finally:
            db.close()

    def test_missing_base_trigger_still_repairs_once(self, tmp_path, monkeypatch):
        """Control: narrowing the gate must not disable genuine repair.

        A base trigger really can go missing (an earlier no-FTS5 runtime drops
        them to keep writes alive), and rows written meanwhile are absent from
        the index. That still has to be repaired — once, and then converge.
        """
        db_path = tmp_path / "state.db"
        statements = []
        self._trace(monkeypatch, statements, trigram=False)

        self._seed(db_path)

        db = StoreHolder(db_path=db_path)
        try:
            db._conn.execute("DROP TRIGGER messages_fts_insert")
            db._conn.commit()
        finally:
            db.close()

        statements.clear()
        db = StoreHolder(db_path=db_path)
        try:
            assert len(self._rebuilds(statements)) == 1
            assert db._conn.execute(
                "SELECT COUNT(*) FROM sqlite_master "
                "WHERE type = 'trigger' AND name = 'messages_fts_insert'"
            ).fetchone()[0] == 1
        finally:
            db.close()

        # …and having repaired it, the next open is quiet again.
        statements.clear()
        db = StoreHolder(db_path=db_path)
        try:
            assert self._rebuilds(statements) == []
        finally:
            db.close()

    def test_missing_trigram_trigger_still_repairs_where_the_tokenizer_exists(
        self, tmp_path, monkeypatch
    ):
        """Control: on a capable host a missing trigram trigger is real damage.

        Only the permanently-unsatisfiable case changes. Where the trigram DDL
        can run, a gap in those triggers means the index missed rows and must
        still be rebuilt.
        """
        db_path = tmp_path / "state.db"
        statements = []
        self._trace(monkeypatch, statements, trigram=True)

        self._seed(db_path)

        db = StoreHolder(db_path=db_path)
        trigram_available = db._trigram_available
        try:
            if not trigram_available:
                pytest.skip("this SQLite build has no trigram tokenizer")
            db._conn.execute("DROP TRIGGER messages_fts_trigram_insert")
            db._conn.commit()
        finally:
            db.close()

        statements.clear()
        db = StoreHolder(db_path=db_path)
        try:
            assert db._trigram_available is True
            assert len(self._rebuilds(statements)) > 0
        finally:
            db.close()


class TestOptimizeFts:

    def test_incremental_merge_bounded_commands_per_present_index(self, db):
        """Each pass issues bounded 'merge' commands, never 'optimize'."""
        create_session(db, session_id="s1", source="cli")
        append_message(db, session_id="s1", role="user", content="bounded merge")
        statements = []
        db._conn.set_trace_callback(statements.append)
        try:
            executed = _merge_fts_incrementally(db, max_pages=37)
        finally:
            db._conn.set_trace_callback(None)

        # At least one merge command per present FTS index, and never more
        # than the per-pass command cap per index.
        present = [t for t in _schema._FTS_TABLES if _fts_table_exists(db, t)]
        assert len(present) >= 2  # messages_fts + trigram on a fresh DB
        merge_sql = [sql for sql in statements if "VALUES('merge', 37)" in sql]
        assert len(merge_sql) == executed
        assert len(present) <= executed <= (
            len(present) * db._FTS_MERGE_COMMANDS_PER_PASS
        )
        for tbl in present:
            n = sum(f"{tbl}({tbl}, rank)" in sql for sql in merge_sql)
            assert 1 <= n <= db._FTS_MERGE_COMMANDS_PER_PASS
        # The usermerge floor is applied so positive merges can make
        # progress on levels with >= 2 segments (SQLite FTS5 §6.8).
        assert any("VALUES('usermerge', 2)" in sql for sql in statements)
        assert not any("'optimize'" in sql for sql in statements)

    def test_write_path_merges_fts_only_at_cadence_boundary(self, db, monkeypatch):
        """Routine writes use bounded merge and never full optimize."""
        db._FTS_MERGE_EVERY_N_WRITES = 5
        calls = []

        def _counting_merge(holder, *, max_pages):
            calls.append(max_pages)
            return 0

        def _unexpected_optimize(holder):
            raise AssertionError("routine cadence must not call optimize")

        monkeypatch.setattr(_schema, "_merge_fts_incrementally", _counting_merge)
        monkeypatch.setattr(_maintenance, "_optimize_fts", _unexpected_optimize)
        create_session(db, session_id="s1", source="cli")
        for i in range(3):
            append_message(db, session_id="s1", role="user", content=f"needle {i}")
        assert calls == []  # Four successful writes are below the boundary.
        append_message(db, session_id="s1", role="user", content="needle 3")
        assert calls == [500]  # The fifth write gets the production page budget.
        for i in range(4, 8):
            append_message(db, session_id="s1", role="user", content=f"needle {i}")
        assert calls == [500]
        append_message(db, session_id="s1", role="user", content="needle 8")
        assert calls == [500, 500]  # The tenth write is the next boundary.
        assert len(search_messages(db, "needle")) == 9


class TestFTS5ToolCallMigration:
    """v11 migration: pre-existing state.db with old external-content FTS tables
    must be re-indexed so tool_name / tool_calls become searchable after upgrade."""

    def test_v10_to_v11_upgrade_backfills_tool_fields(self, tmp_path):
        """Simulate an existing user: build a v10-shaped DB by hand, insert a
        row with tool_calls, then open via StoreHolder (which runs migrations).
        After upgrade, the tool_calls token must be searchable."""
        import sqlite3

        db_path = tmp_path / "legacy.db"

        # Build the pre-v11 schema by hand: external-content FTS tables +
        # old triggers that only reference new.content.
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            INSERT INTO schema_version (version) VALUES (10);

            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT,
                started_at REAL,
                ended_at REAL,
                title TEXT,
                parent_session_id TEXT,
                message_count INTEGER DEFAULT 0,
                tool_call_count INTEGER DEFAULT 0,
                api_call_count INTEGER DEFAULT 0
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_name TEXT,
                tool_calls TEXT,
                tool_call_id TEXT,
                token_count INTEGER,
                finish_reason TEXT,
                reasoning TEXT,
                reasoning_content TEXT,
                reasoning_details TEXT,
                codex_reasoning_items TEXT,
                codex_message_items TEXT
            );

            CREATE VIRTUAL TABLE messages_fts USING fts5(
                content, content=messages, content_rowid=id
            );
            CREATE TRIGGER messages_fts_insert AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
            END;

            CREATE VIRTUAL TABLE messages_fts_trigram USING fts5(
                content, content=messages, content_rowid=id, tokenize='trigram'
            );
            CREATE TRIGGER messages_fts_trigram_insert AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts_trigram(rowid, content) VALUES (new.id, new.content);
            END;
        """)
        conn.execute(
            "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
            ("s1", "cli", time.time()),
        )
        conn.execute(
            "INSERT INTO messages (session_id, timestamp, role, content, tool_name, tool_calls) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("s1", time.time(), "assistant", "", "LEGACYTOOL",
             '{"function":{"name":"web_search","arguments":"{\\"q\\":\\"LEGACYARG\\"}"}}'),
        )
        conn.commit()

        # Verify the legacy FTS rows don't contain the tool tokens yet.
        legacy_hits = conn.execute(
            "SELECT rowid FROM messages_fts WHERE messages_fts MATCH 'LEGACYTOOL'"
        ).fetchall()
        assert legacy_hits == [], "sanity: legacy FTS must NOT contain tool_name"
        conn.close()

        # Open via StoreHolder — the legacy DB is detected as optimizable but
        # NOT auto-migrated (opt-in). Its old content-only index still works
        # for content, but doesn't yet cover tool_name/tool_calls (#16751).
        session_db = StoreHolder(db_path=db_path)
        try:
            assert _FtsStorageOptimizer(session_db).fts_optimize_available() is True

            # `hermes db optimize` performs the v23 transition; afterwards the
            # tool fields are searchable.
            result = _FtsStorageOptimizer(session_db).optimize_fts_storage(vacuum=False)
            assert result["ok"] is True
            assert len(search_messages(session_db, "LEGACYTOOL")) == 1, \
                "v23 optimize must index tool_name into FTS"
            assert len(search_messages(session_db, "LEGACYARG")) == 1, \
                "v23 optimize must index tool_calls JSON into FTS"
            # schema_version bumped once the FTS layer is v23
            row = session_db._conn.execute(
                "SELECT version FROM schema_version LIMIT 1"
            ).fetchone()
            version = row["version"] if hasattr(row, "keys") else row[0]
            assert version == SCHEMA_VERSION
        finally:
            session_db.close()


class TestFTSExternalContentMigration:
    """v23 migration: inline-mode FTS tables (v11-v22) are rebuilt as
    external-content tables, and role='tool' rows are excluded from the
    trigram index while remaining searchable via the standard index."""

    @staticmethod
    def _build_v22_db(db_path):
        """Build a v22-shaped DB by hand: inline FTS tables + concat triggers."""
        conn = sqlite3.connect(str(db_path))
        conn.executescript(SCHEMA_SQL)
        # Replace the current (v23) FTS objects with the v22 inline shape.
        conn.executescript("""
            DROP TABLE IF EXISTS messages_fts;
            DROP TABLE IF EXISTS messages_fts_trigram;
            DROP VIEW IF EXISTS messages_fts_trigram_src;

            CREATE VIRTUAL TABLE messages_fts USING fts5(content);
            CREATE TRIGGER messages_fts_insert AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts(rowid, content) VALUES (
                    new.id,
                    COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
                );
            END;

            CREATE VIRTUAL TABLE messages_fts_trigram USING fts5(content, tokenize='trigram');
            CREATE TRIGGER messages_fts_trigram_insert AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts_trigram(rowid, content) VALUES (
                    new.id,
                    COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
                );
            END;
        """)
        conn.execute("DELETE FROM schema_version")
        conn.execute("INSERT INTO schema_version (version) VALUES (22)")
        conn.execute(
            "INSERT INTO sessions (id, source, started_at) VALUES ('s1', 'cli', ?)",
            (time.time(),),
        )
        rows = [
            ("user", "find the 大别山项目 deployment notes", None, None),
            ("assistant", "关于大别山项目的总结在这里", None,
             '{"function":{"name":"send_message","arguments":"{}"}}'),
            ("tool", "TOOLBLOB " + "x" * 5000 + " 项目文件内容测试", "read_file", None),
        ]
        for role, content, tool_name, tool_calls in rows:
            conn.execute(
                "INSERT INTO messages (session_id, timestamp, role, content, tool_name, tool_calls) "
                "VALUES ('s1', ?, ?, ?, ?, ?)",
                (time.time(), role, content, tool_name, tool_calls),
            )
        conn.commit()
        # Sanity: v22 inline tables have their own content shadow tables.
        shadow = conn.execute(
            "SELECT name FROM sqlite_master WHERE name = 'messages_fts_content'"
        ).fetchall()
        assert shadow, "sanity: v22 inline FTS must have a content shadow table"
        conn.close()

    def test_v22_open_leaves_legacy_untouched_and_advertises(self, tmp_path):
        """Opening a legacy v22 DB must NOT auto-migrate the FTS layout, but
        the main schema_version DOES advance (decoupled) so future non-FTS
        migrations aren't blocked. The inline index keeps working and the
        opt-in flag is set."""
        db_path = tmp_path / "v22.db"
        self._build_v22_db(db_path)

        db = StoreHolder(db_path=db_path)
        try:
            # DECOUPLED: the main schema_version advances to current even though
            # the FTS layout stays legacy — future migrations must not be gated
            # behind the FTS opt-in.
            version = db._conn.execute(
                "SELECT version FROM schema_version"
            ).fetchone()[0]
            assert version == SCHEMA_VERSION, "main schema version must advance"
            # But the FTS storage layout is NOT stamped current — it's legacy.
            assert db.get_meta("fts_storage_version") is None
            assert _FtsStorageOptimizer(db).fts_optimize_available() is True
            assert db.get_meta("fts_optimize_available") == "1"

            # Legacy inline shape is intact (content shadow table still there).
            assert db._conn.execute(
                "SELECT name FROM sqlite_master WHERE name = 'messages_fts_content'"
            ).fetchone() is not None

            # Search still works on the legacy index (no deferred rebuild).
            assert fts_rebuild_status(db) is None
            assert len(search_messages(db, "deployment")) == 1
            assert len(search_messages(db, "send_message")) == 1  # #16751 held

            # A new write is indexed live by the legacy triggers.
            append_message(db, "s1", role="user", content="AFTEROPEN token")
            assert len(search_messages(db, "AFTEROPEN")) == 1
        finally:
            db.close()

    @pytest.mark.parametrize("with_message", [False, True])
    def test_v23_rebuild_from_trigram_tool_calls_projection(
        self, tmp_path, monkeypatch, with_message
    ):
        """v23 installs built with historical trigram projection should be
        repaired via optimize-storage: trigram must drop tool_calls while
        standard messages_fts keeps indexing them."""
        db_path = tmp_path / "v23-toolcalls.db"

        # Build an external-content DB that is already at schema version 23,
        # but with the old tool_calls-inclusive trigram projection.
        conn = sqlite3.connect(str(db_path))
        conn.executescript(SCHEMA_SQL)
        conn.executescript(FTS_SQL)
        conn.executescript(
            """
            DROP TRIGGER IF EXISTS messages_fts_trigram_insert;
            DROP TRIGGER IF EXISTS messages_fts_trigram_delete;
            DROP TRIGGER IF EXISTS messages_fts_trigram_update;
            DROP TABLE IF EXISTS messages_fts_trigram;
            DROP VIEW IF EXISTS messages_fts_trigram_src;

            CREATE VIEW IF NOT EXISTS messages_fts_trigram_src AS
                SELECT id, role, content, tool_name, tool_calls
                FROM messages
                WHERE role <> 'tool';

            CREATE VIRTUAL TABLE messages_fts_trigram USING fts5(
                content,
                tool_name,
                tool_calls,
                content='messages_fts_trigram_src',
                content_rowid='id',
                tokenize='trigram'
            );

            CREATE TRIGGER messages_fts_trigram_insert AFTER INSERT ON messages
            WHEN new.role <> 'tool'
            BEGIN
                INSERT INTO messages_fts_trigram(rowid, content, tool_name, tool_calls)
                VALUES (new.id, new.content, new.tool_name, new.tool_calls);
            END;

            CREATE TRIGGER messages_fts_trigram_delete AFTER DELETE ON messages
            WHEN old.role <> 'tool'
            BEGIN
                INSERT INTO messages_fts_trigram(messages_fts_trigram, rowid, content, tool_name, tool_calls)
                VALUES ('delete', old.id, old.content, old.tool_name, old.tool_calls);
            END;

            CREATE TRIGGER messages_fts_trigram_update
            AFTER UPDATE OF content, tool_name, tool_calls, role ON messages
            WHEN (old.content IS NOT new.content
                OR old.tool_name IS NOT new.tool_name
                OR old.tool_calls IS NOT new.tool_calls
                OR old.role IS NOT new.role)
            BEGIN
                INSERT INTO messages_fts_trigram(messages_fts_trigram, rowid, content, tool_name, tool_calls)
                SELECT 'delete', old.id, old.content, old.tool_name, old.tool_calls
                WHERE old.role <> 'tool';
                INSERT INTO messages_fts_trigram(rowid, content, tool_name, tool_calls)
                SELECT new.id, new.content, new.tool_name, new.tool_calls
                WHERE new.role <> 'tool';
            END;
            """
        )
        # Simulate the historical v23 projection shipped before this fix.
        conn.execute(
            "INSERT OR REPLACE INTO state_meta (key, value) VALUES ('fts_storage_version', '1')"
        )
        conn.execute(
            "INSERT OR REPLACE INTO state_meta (key, value) VALUES ('fts_optimize_available', '1')"
        )
        conn.commit()
        conn.close()

        if with_message:
            conn = sqlite3.connect(str(db_path))
            conn.execute(
                "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
                ("s1", "cli", time.time()),
            )
            conn.execute(
                "INSERT INTO messages (session_id, timestamp, role, content, tool_name, tool_calls) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    "s1",
                    time.time(),
                    "assistant",
                    "部署完成 assistant content",
                    "legacyTool",
                    '{"name": "legacy", "arguments": "UNIQUE_TOOLCALL_TOKEN_43701"}',
                ),
            )
            conn.commit()
            assert conn.execute(
                "SELECT rowid FROM messages_fts_trigram WHERE messages_fts_trigram MATCH 'UNIQUE_TOOLCALL_TOKEN_43701'"
            ).fetchall()
            conn.close()

        db = StoreHolder(db_path=db_path)
        try:
            assert db._conn is not None
            assert _FtsStorageOptimizer(db).fts_optimize_available() is True
            assert db.get_meta("fts_storage_version") == "1"

            original_ensure = _maintenance._ensure_fts_schema

            def interrupt_after_demote(holder, cursor, table_name, ddl):
                if table_name == "messages_fts_trigram":
                    raise RuntimeError("injected trigram rebuild interruption")
                return original_ensure(holder, cursor, table_name, ddl)

            monkeypatch.setattr(_maintenance, "_ensure_fts_schema", interrupt_after_demote)
            with pytest.raises(RuntimeError, match="injected trigram"):
                _FtsStorageOptimizer(db).optimize_fts_storage(vacuum=False)

            db.close()
            db = StoreHolder(db_path=db_path)
            assert db._conn is not None
            assert db.get_meta("fts_rebuild_high_water") is None
            assert db.get_meta("fts_rebuild_progress") is None
            assert _has_fts_trash(db, db._conn) is True
            assert _FtsStorageOptimizer(db).fts_optimize_available() is True
            assert db.get_meta("fts_storage_version") == "1"
            if with_message:
                assert db._conn.execute(
                    "SELECT 1 FROM messages_fts_trigram "
                    "WHERE messages_fts_trigram MATCH '部署完成' LIMIT 1"
                ).fetchone()

            result = _FtsStorageOptimizer(db).optimize_fts_storage(vacuum=False)
            assert result["ok"] is True

            if with_message:
                # messages_fts stays in the tool-calls search path.
                assert len(search_messages(db, "UNIQUE_TOOLCALL_TOKEN_43701")) == 1
                # New trigram schema excludes tool_calls from trigram projection.
                assert not db._conn.execute(
                    "SELECT 1 FROM messages_fts_trigram WHERE messages_fts_trigram MATCH 'UNIQUE_TOOLCALL_TOKEN_43701' LIMIT 1"
                ).fetchone()
                assert db._conn.execute(
                    "SELECT 1 FROM messages_fts_trigram "
                    "WHERE messages_fts_trigram MATCH '部署完成' LIMIT 1"
                ).fetchone()
            trigger_sql = db._conn.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE type = 'trigger' AND name = 'messages_fts_trigram_update'"
            ).fetchone()[0]
            assert "tool_calls" not in trigger_sql
            assert db.get_meta("fts_storage_version") == str(FTS_STORAGE_VERSION)
        finally:
            db.close()

    def _simulate_pre_fix_demote_crash_window(self, db):
        """Replay the pre-fix demote crash window: trash + empty v23 schema,
        no rebuild markers (executescript committed mid-demote before markers).

        Mirrors what happened when ``_ensure_fts_schema`` ran inside
        ``_execute_write`` and the process died before the marker writes.
        """
        from persistence.sqlite._schema_ddl import FTS_TRIGRAM_SQL

        conn = db._conn
        _schema._drop_fts_triggers(conn)
        conn.execute("DROP VIEW IF EXISTS messages_fts_trigram_src")
        had = bool(conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name IN ('messages_fts', 'messages_fts_trigram') "
            "AND sql LIKE 'CREATE VIRTUAL TABLE%' LIMIT 1"
        ).fetchone())
        assert had, "sanity: expected legacy/virtual FTS tables to demote"
        conn.execute("PRAGMA writable_schema=ON")
        conn.execute(
            "DELETE FROM sqlite_master WHERE type = 'table' "
            "AND name IN ('messages_fts', 'messages_fts_trigram') "
            "AND sql LIKE 'CREATE VIRTUAL TABLE%'"
        )
        conn.execute("PRAGMA writable_schema=RESET")
        shadows = [
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND (name LIKE 'messages_fts_%' ESCAPE '\\' "
                "OR name LIKE 'messages_fts_trigram_%' ESCAPE '\\')"
            ).fetchall()
        ]
        for sh in shadows:
            conn.execute(f"ALTER TABLE {sh} RENAME TO fts_v22_trash_{sh}")
        # executescript commits — empty v23 tables without markers.
        conn.executescript(FTS_SQL)
        try:
            conn.executescript(FTS_TRIGRAM_SQL)
        except sqlite3.OperationalError:
            pass
        # Intentionally leave fts_rebuild_* unset (the crash window).

    def test_optimize_resume_after_demote_crash_window_restores_search(
        self, tmp_path
    ):
        """Pre-fix: demote crash left trash + empty v23 index, no markers.
        Re-run tore down trash and stamped optimized with docsize=0 — permanent
        search loss for historical rows. Re-run must backfill and restore."""
        db_path = tmp_path / "v22.db"
        self._build_v22_db(db_path)

        db = StoreHolder(db_path=db_path)
        try:
            assert len(search_messages(db, "deployment")) == 1
            self._simulate_pre_fix_demote_crash_window(db)
            # Crash window shape: no markers, trash present, empty index.
            assert db.get_meta("fts_rebuild_high_water") is None
            assert db.get_meta("fts_rebuild_progress") is None
            assert _has_fts_trash(db, db._conn) is True
            assert db._conn.execute(
                "SELECT COUNT(*) FROM messages_fts_docsize"
            ).fetchone()[0] == 0
            assert len(search_messages(db, "deployment")) == 0

            # Still offered (trash and/or empty-index heal).
            assert _FtsStorageOptimizer(db).fts_optimize_available() is True

            result = _FtsStorageOptimizer(db).optimize_fts_storage(vacuum=False)
            assert result["ok"] is True
            assert fts_rebuild_status(db) is None
            assert _FtsStorageOptimizer(db).fts_optimize_available() is False
            assert db.get_meta("fts_storage_version") == str(
                FTS_STORAGE_VERSION
            )
            assert db._conn.execute(
                "SELECT name FROM sqlite_master WHERE name LIKE '%_v22_trash%'"
            ).fetchall() == []
            # Historical rows searchable again; index fully populated.
            assert len(search_messages(db, "deployment")) == 1
            assert len(search_messages(db, "TOOLBLOB")) == 1
            n_msg = db._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            n_fts = db._conn.execute(
                "SELECT COUNT(*) FROM messages_fts_docsize"
            ).fetchone()[0]
            assert n_fts == n_msg
            db._conn.execute(
                "INSERT INTO messages_fts(messages_fts, rank) VALUES('integrity-check', 1)"
            )
        finally:
            db.close()

    def test_optimize_heals_premature_stamp_with_empty_index(self, tmp_path):
        """Pre-fix settle could stamp fts_storage_version after tearing down
        trash with an empty index and no markers. Re-run must clear the stamp,
        backfill, and re-earn the layout version."""
        db_path = tmp_path / "v22.db"
        self._build_v22_db(db_path)

        db = StoreHolder(db_path=db_path)
        try:
            self._simulate_pre_fix_demote_crash_window(db)
            # Simulate the bad resume: trash already gone, empty index stamped.
            trash = [
                r[0] for r in db._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' "
                    "AND name LIKE 'fts\\_v22\\_trash\\_%' ESCAPE '\\'"
                ).fetchall()
            ]
            for tbl in trash:
                db._conn.execute(f"DROP TABLE IF EXISTS {tbl}")
            db._conn.execute(
                "INSERT INTO state_meta (key, value) VALUES "
                "('fts_storage_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(FTS_STORAGE_VERSION),),
            )
            db._conn.commit()

            assert db.get_meta("fts_rebuild_high_water") is None
            assert _has_fts_trash(db, db._conn) is False
            assert _fts_external_index_empty_with_messages(db, db._conn) is True
            # Must still be offered despite the premature stamp.
            assert _FtsStorageOptimizer(db).fts_optimize_available() is True
            assert len(search_messages(db, "deployment")) == 0

            result = _FtsStorageOptimizer(db).optimize_fts_storage(vacuum=False)
            assert result["ok"] is True
            assert len(search_messages(db, "deployment")) == 1
            assert db.get_meta("fts_storage_version") == str(
                FTS_STORAGE_VERSION
            )
            assert _FtsStorageOptimizer(db).fts_optimize_available() is False
        finally:
            db.close()

    def test_optimize_heals_high_water_without_progress(self, tmp_path):
        """high_water without progress used to make fts_rebuild_step return
        False immediately (treated as finished by another process), then
        settle stamped success while the marker remained. Re-seed progress
        and complete the empty-index backfill."""
        db_path = tmp_path / "v22.db"
        self._build_v22_db(db_path)
        db = StoreHolder(db_path=db_path)
        try:
            self._simulate_pre_fix_demote_crash_window(db)
            hw = db._conn.execute(
                "SELECT COALESCE(MAX(id), 0) FROM messages"
            ).fetchone()[0]
            # Orphan shape: high_water alone on an empty external index.
            db.set_meta("fts_rebuild_high_water", str(hw))
            db._conn.execute(
                "DELETE FROM state_meta WHERE key = ?", ("fts_rebuild_progress",)
            )
            db._conn.commit()
            assert db.get_meta("fts_rebuild_progress") is None
            assert _FtsStorageOptimizer(db).fts_optimize_available() is True
            # Empty index: base FTS MATCH finds nothing (gap LIKE may still
            # supplement when high_water is set — that is intentional).
            assert db._conn.execute(
                "SELECT COUNT(*) FROM messages_fts_docsize"
            ).fetchone()[0] == 0

            result = _FtsStorageOptimizer(db).optimize_fts_storage(vacuum=False)
            assert result["ok"] is True
            assert db.get_meta("fts_rebuild_high_water") is None
            assert db.get_meta("fts_rebuild_progress") is None
            n_msg = db._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            n_fts = db._conn.execute(
                "SELECT COUNT(*) FROM messages_fts_docsize"
            ).fetchone()[0]
            assert n_fts == n_msg
            assert len(search_messages(db, "deployment")) == 1
            assert _FtsStorageOptimizer(db).fts_optimize_available() is False
        finally:
            db.close()

    def test_repair_rebuilds_partial_index_without_duplicates(self, tmp_path):
        """high_water without progress on a PARTIALLY indexed DB must not
        replay the backfill from zero on top of surviving rows: the chunk
        worker inserts its whole id range with no anti-join, so replay
        duplicates every already-indexed row. Recovery must reset the index
        to a known-empty surface first, then rebuild."""
        db_path = tmp_path / "v22.db"
        self._build_v22_db(db_path)
        db = StoreHolder(db_path=db_path)
        try:
            self._simulate_pre_fix_demote_crash_window(db)
            hw = db._conn.execute(
                "SELECT COALESCE(MAX(id), 0) FROM messages"
            ).fetchone()[0]
            db.set_meta("fts_rebuild_high_water", str(hw))
            db._conn.execute(
                "DELETE FROM state_meta WHERE key = ?", ("fts_rebuild_progress",)
            )
            # Partial index: one row survived from an interrupted backfill.
            db._conn.execute(
                "INSERT INTO messages_fts(rowid, content, tool_name, tool_calls) "
                "SELECT id, content, tool_name, tool_calls FROM messages "
                "WHERE id = 1"
            )
            db._conn.commit()
            assert db._conn.execute(
                "SELECT COUNT(*) FROM messages_fts_docsize"
            ).fetchone()[0] == 1

            result = _FtsStorageOptimizer(db).optimize_fts_storage(vacuum=False)
            assert result["ok"] is True
            n_msg = db._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            n_fts = db._conn.execute(
                "SELECT COUNT(*) FROM messages_fts_docsize"
            ).fetchone()[0]
            # Exactly one index entry per message: no replay duplicates.
            assert n_fts == n_msg
            assert len(search_messages(db, "deployment")) == 1
            db._conn.execute(
                "INSERT INTO messages_fts(messages_fts, rank) VALUES('integrity-check', 1)"
            )
        finally:
            db.close()

    def test_repair_bookkeeping_reseeds_missing_progress(self, tmp_path):
        """Unit: high_water without progress gets progress='0' without
        forcing a full marker reset when a real backfill is already claimed."""
        db = StoreHolder(db_path=tmp_path / "fresh.db")
        try:
            create_session(db, session_id="s1", source="cli")
            append_message(db, "s1", role="user", content="bookkeeping needle")
            db.set_meta("fts_rebuild_high_water", "42")
            db._conn.execute(
                "DELETE FROM state_meta WHERE key = ?", ("fts_rebuild_progress",)
            )
            db._conn.commit()
            _FtsStorageOptimizer(db)._repair_optimize_bookkeeping()
            assert db.get_meta("fts_rebuild_high_water") == "42"
            assert db.get_meta("fts_rebuild_progress") == "0"
        finally:
            db.close()

    def test_demote_writes_markers_before_empty_schema(self, tmp_path, monkeypatch):
        """Demote must commit rebuild markers before createscript builds the
        empty v23 tables — so a crash between stage and ensure still leaves
        a resumable claim rather than an unmarked empty index."""
        db_path = tmp_path / "v22.db"
        self._build_v22_db(db_path)
        db = StoreHolder(db_path=db_path)
        try:
            # Patch ensure to fail *after* the staged write commits, simulating
            # death mid schema-create. Markers must already be durable.
            orig_ensure = _maintenance._ensure_fts_schema
            calls = {"n": 0}

            def boom(holder, cursor, table_name, ddl):
                calls["n"] += 1
                if table_name == "messages_fts":
                    # Markers must already be on disk from the staged write.
                    row = db._conn.execute(
                        "SELECT value FROM state_meta "
                        "WHERE key = 'fts_rebuild_high_water'"
                    ).fetchone()
                    assert row is not None, (
                        "markers must be committed before empty v23 schema create"
                    )
                    progress = db._conn.execute(
                        "SELECT value FROM state_meta "
                        "WHERE key = 'fts_rebuild_progress'"
                    ).fetchone()
                    assert progress is not None and progress[0] == "0"
                    raise sqlite3.OperationalError("simulated crash mid-ensure")
                return orig_ensure(holder, cursor, table_name, ddl)

            monkeypatch.setattr(_maintenance, "_ensure_fts_schema", boom)
            try:
                _FtsStorageOptimizer(db)._demote_legacy_fts_to_trash()
                raise AssertionError("demote should have raised")
            except sqlite3.OperationalError as exc:
                assert "simulated crash" in str(exc)

            # Staged demote survived: markers + trash, no successful stamp.
            assert db.get_meta("fts_rebuild_high_water") is not None
            assert db.get_meta("fts_rebuild_progress") == "0"
            assert _has_fts_trash(db, db._conn) is True
            assert db.get_meta("fts_storage_version") is None

            # Restore ensure and resume — full optimize completes.
            monkeypatch.setattr(_maintenance, "_ensure_fts_schema", orig_ensure)
            result = _FtsStorageOptimizer(db).optimize_fts_storage(vacuum=False)
            assert result["ok"] is True
            assert len(search_messages(db, "deployment")) == 1
            assert _FtsStorageOptimizer(db).fts_optimize_available() is False
        finally:
            db.close()

    def test_v23_fresh_db_born_optimized(self, tmp_path):
        """A brand-new DB is born on v23 — no legacy layout, no opt-in flag,
        no pending rebuild."""
        db = StoreHolder(db_path=tmp_path / "fresh.db")
        try:
            assert _FtsStorageOptimizer(db).fts_optimize_available() is False
            assert fts_rebuild_status(db) is None
            assert db.get_meta("fts_optimize_available") is None
            # Already external-content: no shadow copy tables.
            assert db._conn.execute(
                "SELECT name FROM sqlite_master WHERE name = 'messages_fts_content'"
            ).fetchone() is None
            create_session(db, session_id="s1", source="cli")
            append_message(db, "s1", role="user", content="hello fresh world")
            assert len(search_messages(db, "fresh")) == 1
        finally:
            db.close()

    def test_v23_cjk_tool_role_filter_uses_like_fallback(self, tmp_path):
        """A CJK query with role_filter=['tool'] must bypass the trigram index
        (tool rows aren't in it) and still find matches via LIKE."""
        db = StoreHolder(db_path=tmp_path / "fresh.db")
        try:
            create_session(db, session_id="s1", source="cli")
            append_message(db, "s1", role="tool", content="错误日志：数据库连接超时",
                              tool_name="terminal")
            hits = search_messages(db, "数据库连接", role_filter=["tool"])
            assert len(hits) == 1
            assert hits[0]["role"] == "tool"
        finally:
            db.close()

    def test_fts_teardown_single_key_high_water_drains_and_drops(self, tmp_path):
        """#79324: single-column-key trash tables drain via a high-water
        marker so each chunk only scans rows after the previous chunk.

        Builds a large trash table with a rowid-like integer PK, then drives
        ``_fts_teardown_trash_step`` to completion. Verifies every row is
        removed, the marker advances monotonically, the marker is cleared,
        and the table is dropped at the end.
        """
        db = StoreHolder(db_path=tmp_path / "trash.db")
        try:
            conn = db._conn
            # A plain trash table shaped like a demoted FTS shadow table
            # (single integer PK — the common, large-table shape).
            conn.execute(
                "CREATE TABLE fts_v22_trash_messages_fts_data "
                "(docid INTEGER PRIMARY KEY, block BLOB)"
            )
            conn.executemany(
                "INSERT INTO fts_v22_trash_messages_fts_data "
                "(docid, block) VALUES (?, ?)",
                [(i, b"x" * 64) for i in range(1, 2501)],
            )
            conn.commit()

            assert _has_fts_trash(db, conn) is True

            steps = 0
            while _FtsStorageOptimizer(db)._fts_teardown_trash_step():
                steps += 1
                assert steps < 100, "teardown never finished"

            # All rows gone, table dropped, marker cleaned up.
            assert _has_fts_trash(db, conn) is False
            assert conn.execute(
                "SELECT name FROM sqlite_master WHERE name = "
                "'fts_v22_trash_messages_fts_data'"
            ).fetchone() is None
            assert db.get_meta("fts_teardown_fts_v22_trash_messages_fts_data_progress") is None
            # Multiple chunks were needed (2500 rows / 500 chunk).
            assert steps >= 5
        finally:
            db.close()

    def test_fts_teardown_high_water_resumes_after_interruption(self, tmp_path):
        """#79324: the high-water marker survives an interrupted teardown,
        so the next call resumes from the marker instead of the table start."""
        db = StoreHolder(db_path=tmp_path / "trash.db")
        try:
            conn = db._conn
            conn.execute(
                "CREATE TABLE fts_v22_trash_messages_fts_data "
                "(docid INTEGER PRIMARY KEY, block BLOB)"
            )
            conn.executemany(
                "INSERT INTO fts_v22_trash_messages_fts_data "
                "(docid, block) VALUES (?, ?)",
                [(i, b"x" * 64) for i in range(1, 1201)],
            )
            conn.commit()

            # Drain two chunks, then simulate a crash: the marker stays at
            # the last drained key and the remaining rows are intact.
            assert _FtsStorageOptimizer(db)._fts_teardown_trash_step() is True
            assert _FtsStorageOptimizer(db)._fts_teardown_trash_step() is True
            marker = db.get_meta("fts_teardown_fts_v22_trash_messages_fts_data_progress")
            assert marker is not None
            assert int(marker) == 1000  # 2 chunks x 500 rows

            remaining = conn.execute(
                "SELECT COUNT(*) FROM fts_v22_trash_messages_fts_data"
            ).fetchone()[0]
            assert remaining == 200

            # Resume: drains the rest, drops the table.
            while _FtsStorageOptimizer(db)._fts_teardown_trash_step():
                pass
            assert _has_fts_trash(db, conn) is False
            assert db.get_meta("fts_teardown_fts_v22_trash_messages_fts_data_progress") is None
        finally:
            db.close()

    def test_fts_teardown_compound_key_keeps_legacy_path(self, tmp_path):
        """#79324: multi-column-PK trash tables (small by construction) keep
        the legacy chunked delete — the high-water path only applies to
        single-column keys."""
        db = StoreHolder(db_path=tmp_path / "trash.db")
        try:
            conn = db._conn
            conn.execute(
                "CREATE TABLE fts_v22_trash_messages_fts_idx "
                "(segid INTEGER, term TEXT, pgno INTEGER, "
                "PRIMARY KEY (segid, term, pgno)) WITHOUT ROWID"
            )
            conn.executemany(
                "INSERT INTO fts_v22_trash_messages_fts_idx "
                "(segid, term, pgno) VALUES (?, ?, ?)",
                [(i % 3, f"term-{i}", i) for i in range(20)],
            )
            conn.commit()

            steps = 0
            while _FtsStorageOptimizer(db)._fts_teardown_trash_step():
                steps += 1
                assert steps < 10

            assert _has_fts_trash(db, conn) is False
            assert conn.execute(
                "SELECT name FROM sqlite_master WHERE name = "
                "'fts_v22_trash_messages_fts_idx'"
            ).fetchone() is None
        finally:
            db.close()


class TestFtsRebuildFinishWithoutTrigram:
    """An FTS index that the runtime cannot maintain must not wedge the store.

    Two independent failure sites shared one root shape: code that writes to
    ``messages_fts_trigram`` without first checking the table is actually
    present. It is legitimately absent whenever the trigram index is
    unavailable (SQLite build without the tokenizer), and it can also be left
    absent by an interrupted migration or a partially-applied schema change.
    """

    @staticmethod
    def _seed(db_path, n=60):
        seeded = StoreHolder(db_path=db_path)
        try:
            create_session(seeded, session_id="s1", source="cli")
            for i in range(n):
                append_message(seeded,
                    "s1",
                    role=("user" if i % 3 == 0
                          else "assistant" if i % 3 == 1 else "tool"),
                    content=f"sentinel payload {i} zebra",
                )
            high_water = seeded._conn.execute(
                "SELECT COALESCE(MAX(id), 0) FROM messages"
            ).fetchone()[0]
        finally:
            seeded.close()
        return high_water

    def test_rebuild_finish_skips_trigram_when_unavailable(
        self, tmp_path, monkeypatch
    ):
        """optimize_fts_storage() completes when the trigram index is absent.

        ``fts_rebuild_step()`` already guards its backfill INSERT on
        ``_trigram_available``; ``_fts_rebuild_finish()``'s boundary sweep did
        not, so finishing a deferred rebuild on a trigram-less runtime raised
        ``no such table: messages_fts_trigram`` and aborted the whole
        optimization. The base index must still be swept and the markers
        cleared.
        """
        db_path = tmp_path / "state.db"
        high_water = self._seed(db_path)

        real_connect = sqlite3.connect

        def connect_without_trigram(*args, **kwargs):
            kwargs["factory"] = _NoTrigramConnection
            return real_connect(*args, **kwargs)

        monkeypatch.setattr(
            "persistence.sqlite._holder.sqlite3.connect", connect_without_trigram
        )
        db = StoreHolder(db_path=db_path)
        try:
            assert db._trigram_available is False
            # A trigram-less runtime leaves no trigram index on disk.
            db._conn.execute("DROP TABLE IF EXISTS messages_fts_trigram")
            db._conn.commit()
            assert _fts_table_exists(db, "messages_fts_trigram") is False

            # Put the DB in the pending-deferred-rebuild state.
            for key, value in (
                ("fts_rebuild_high_water", str(high_water)),
                ("fts_rebuild_progress", str(high_water)),
            ):
                db._conn.execute(
                    "INSERT INTO state_meta (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (key, value),
                )
            db._conn.commit()

            # Pre-fix this raised OperationalError("no such table: ...").
            _FtsStorageOptimizer(db)._fts_rebuild_finish()

            # The sweep ran to completion: markers cleared…
            assert db.get_meta("fts_rebuild_high_water") is None
            assert db.get_meta("fts_rebuild_progress") is None
            # …and the base index is still usable (the fix must not disable
            # real search to dodge the error).
            assert search_messages(db, "zebra")
        finally:
            db.close()

    def test_optimize_fts_storage_succeeds_without_trigram(
        self, tmp_path, monkeypatch
    ):
        """End-to-end: the public optimize entry point returns ok=True."""
        db_path = tmp_path / "state.db"
        high_water = self._seed(db_path)

        real_connect = sqlite3.connect

        def connect_without_trigram(*args, **kwargs):
            kwargs["factory"] = _NoTrigramConnection
            return real_connect(*args, **kwargs)

        monkeypatch.setattr(
            "persistence.sqlite._holder.sqlite3.connect", connect_without_trigram
        )
        db = StoreHolder(db_path=db_path)
        try:
            db._conn.execute("DROP TABLE IF EXISTS messages_fts_trigram")
            db._conn.commit()
            assert db._trigram_available is False
            for key, value in (
                ("fts_rebuild_high_water", str(high_water)),
                ("fts_rebuild_progress", "0"),
            ):
                db._conn.execute(
                    "INSERT INTO state_meta (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (key, value),
                )
            db._conn.commit()

            result = _FtsStorageOptimizer(db).optimize_fts_storage(vacuum=False)
            assert result["ok"] is True
            assert db.get_meta("fts_rebuild_high_water") is None
            assert search_messages(db, "zebra")
        finally:
            db.close()
