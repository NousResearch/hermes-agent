"""Tests for hermes_state.py — SessionDB SQLite CRUD, FTS5 search, export."""

import sqlite3
import time
import json
from unittest import mock

import pytest

import hermes_state
from hermes_state import SCHEMA_SQL, SCHEMA_VERSION, SessionDB


class _NoFtsCursor(sqlite3.Cursor):
    """Simulate a SQLite build without the fts5 module."""

    def execute(self, sql, parameters=()):
        probe = sql.strip()
        if "USING fts5" in probe:
            raise sqlite3.OperationalError("no such module: fts5")
        if probe in (
            "SELECT * FROM messages_fts LIMIT 0",
            "SELECT * FROM messages_fts_trigram LIMIT 0",
        ):
            raise sqlite3.OperationalError("no such table: " + probe.split()[-3])
        return super().execute(sql, parameters)

    def executescript(self, sql_script):
        if "USING fts5" in sql_script:
            raise sqlite3.OperationalError("no such module: fts5")
        return super().executescript(sql_script)


class _NoFtsConnection(sqlite3.Connection):
    def cursor(self, factory=None):
        return super().cursor(factory or _NoFtsCursor)


class _NoFtsExistingTableCursor(_NoFtsCursor):
    """Simulate existing FTS virtual tables under a runtime without FTS5."""

    def execute(self, sql, parameters=()):
        probe = sql.strip()
        if probe in (
            "SELECT * FROM messages_fts LIMIT 0",
            "SELECT * FROM messages_fts_trigram LIMIT 0",
        ):
            raise sqlite3.OperationalError("no such module: fts5")
        return super().execute(sql, parameters)


class _NoFtsExistingTableConnection(sqlite3.Connection):
    def cursor(self, factory=None):
        return super().cursor(factory or _NoFtsExistingTableCursor)


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
    """Create a SessionDB with a temp database file."""
    db_path = tmp_path / "test_state.db"
    session_db = SessionDB(db_path=db_path)
    yield session_db
    session_db.close()


# =========================================================================
# Connection lifecycle
# =========================================================================


class TestConnectionLifecycle:
    def test_read_only_close_never_requests_wal_checkpoint(self, tmp_path):
        db_path = tmp_path / "state.db"
        writable = SessionDB(db_path=db_path)
        writable.create_session("s1", source="cli")
        writable.close()

        executed = []
        read_only = SessionDB(db_path=db_path, read_only=True)
        read_only._conn.set_trace_callback(executed.append)
        read_only.close()

        assert not any("wal_checkpoint" in sql.lower() for sql in executed)

    def test_writable_close_retains_truncate_checkpoint(self, tmp_path):
        db_path = tmp_path / "state.db"
        writable = SessionDB(db_path=db_path)
        executed = []
        writable._conn.set_trace_callback(executed.append)

        writable.close()

        assert any(
            "pragma wal_checkpoint(truncate)" == " ".join(sql.lower().split())
            for sql in executed
        )

    def test_read_only_connection_keeps_fts_search_available(self, tmp_path):
        db_path = tmp_path / "state.db"
        writable = SessionDB(db_path=db_path)
        writable.create_session("fts-read-only", source="cli")
        writable.append_message(
            "fts-read-only",
            role="user",
            content="readonlywoodpecker 大别山项目",
        )
        writable.close()

        read_only = SessionDB(db_path=db_path, read_only=True)
        try:
            base_matches = read_only.search_messages("readonlywoodpecker")
            trigram_matches = read_only.search_messages("大别山")
        finally:
            read_only.close()

        assert [match["session_id"] for match in base_matches] == [
            "fts-read-only"
        ]
        assert [match["session_id"] for match in trigram_matches] == [
            "fts-read-only"
        ]

    def test_failed_read_only_open_does_not_leak_tracked_connection(
        self, tmp_path
    ):
        """A malformed store makes the RO FTS probe raise DatabaseError.
        The connection must be closed on that failure path: a leaked tracked
        connection blocks _backup_db_file's raw-copy for the process
        lifetime, so the writable heal that follows would repair WITHOUT its
        forensic backup."""
        import sqlite3

        from hermes_cli.sqlite_safe_read import has_live_connection

        db_path = tmp_path / "state.db"
        writable = SessionDB(db_path=db_path)
        writable.create_session("s1", source="cli")
        writable.append_message("s1", role="user", content="leak probe")
        writable.close()

        # Corrupt sqlite_master: duplicate messages_fts definition. Any
        # statement on a fresh connection then raises "malformed database
        # schema" (DatabaseError, not the OperationalError the probe eats).
        conn = sqlite3.connect(str(db_path), isolation_level=None)
        conn.execute("PRAGMA writable_schema=ON")
        row = conn.execute(
            "SELECT type,name,tbl_name,rootpage,sql FROM sqlite_master "
            "WHERE name='messages_fts'"
        ).fetchone()
        assert row is not None
        conn.execute(
            "INSERT INTO sqlite_master (type,name,tbl_name,rootpage,sql) "
            "VALUES (?,?,?,?,?)",
            row,
        )
        conn.execute("PRAGMA writable_schema=OFF")
        conn.close()

        with pytest.raises(sqlite3.DatabaseError):
            SessionDB(db_path=db_path, read_only=True)

        assert has_live_connection(db_path) is False

        # The writable heal must still take its forensic backup.
        healed = SessionDB(db_path=db_path, read_only=False)
        healed.close()
        assert list(tmp_path.glob("*malformed-backup*"))


# =========================================================================
# Session lifecycle
# =========================================================================

class TestSessionLifecycle:
    def test_create_and_get_session(self, db):
        sid = db.create_session(
            session_id="s1",
            source="cli",
            model="test-model",
        )
        assert sid == "s1"

        session = db.get_session("s1")
        assert session is not None
        assert session["source"] == "cli"
        assert session["model"] == "test-model"
        assert session["ended_at"] is None





    def test_update_session_cwd_persists_git_branch(self, db):
        db.create_session(session_id="s1", source="cli")
        db.update_session_cwd("s1", "/work/repo", git_branch="pets-feature")

        session = db.get_session("s1")
        assert session["cwd"] == "/work/repo"
        assert session["git_branch"] == "pets-feature"


















    def test_end_session_first_reason_wins_across_concurrent_connections(
        self, db
    ):
        """Concurrent finalizers perform one transition, not last-write-wins."""
        import threading

        db.create_session(session_id="s1", source="cron")
        db._conn.execute(
            "CREATE TABLE session_end_audit (reason TEXT NOT NULL)"
        )
        db._conn.execute(
            """
            CREATE TRIGGER audit_session_end
            AFTER UPDATE OF ended_at ON sessions
            WHEN OLD.ended_at IS NULL AND NEW.ended_at IS NOT NULL
            BEGIN
                INSERT INTO session_end_audit(reason) VALUES (NEW.end_reason);
            END
            """
        )

        peer = SessionDB(db_path=db.db_path)
        barrier = threading.Barrier(2)
        errors = []

        def _end(session_db, reason):
            try:
                barrier.wait(timeout=5)
                session_db.end_session("s1", reason)
            except BaseException as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=_end, args=(db, "compression")),
            threading.Thread(target=_end, args=(peer, "cron_complete")),
        ]
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)

            assert all(not thread.is_alive() for thread in threads)
            assert errors == []
            audit_rows = db._conn.execute(
                "SELECT reason FROM session_end_audit"
            ).fetchall()
            assert len(audit_rows) == 1
            assert db.get_session("s1")["end_reason"] == audit_rows[0]["reason"]
        finally:
            peer.close()











    def test_update_session_model_clears_browser_lock_and_preserves_lineage(self, db):
        """A later /model switch must replace, not compete with, a Browser lock."""
        db.create_session(
            session_id="s1",
            source="hermes_browser",
            model="x-ai/grok-4.5",
            model_config={
                "_branched_from": "parent-session",
                "browser_model_lock": {
                    "provider": "nous",
                    "model": "x-ai/grok-4.5",
                    "confirmed": True,
                },
            },
        )

        db.update_session_model("s1", "anthropic/claude-opus-4.8")

        session = db.get_session("s1")
        model_config = json.loads(session["model_config"])
        assert session["model"] == "anthropic/claude-opus-4.8"
        assert "browser_model_lock" not in model_config
        assert model_config["_branched_from"] == "parent-session"








    def test_first_accounted_route_replaces_all_route_fields_atomically(self, db):
        db.create_session(session_id="route", source="cli", model="primary")
        db.update_session_billing_route(
            "route", provider="primary-provider",
            base_url="https://primary.example/v1", billing_mode="api_key",
        )
        db.update_token_counts(
            "route", model="fallback", billing_provider="fallback-provider",
            billing_base_url=None, billing_mode=None, api_call_count=1,
        )
        row = db.get_session("route")
        assert row["model"] == "fallback"
        assert row["billing_provider"] == "fallback-provider"
        assert row["billing_base_url"] is None
        assert row["billing_mode"] is None












    def test_cjk_search_falls_back_to_like_when_trigram_unavailable(
        self, tmp_path, monkeypatch
    ):
        """Regression: long CJK queries must fall back to LIKE when trigram is missing."""
        real_connect = sqlite3.connect
        db_path = tmp_path / "state.db"

        def connect_without_trigram(*args, **kwargs):
            kwargs["factory"] = _NoTrigramConnection
            return real_connect(*args, **kwargs)

        monkeypatch.setattr("hermes_state.sqlite3.connect", connect_without_trigram)
        db = SessionDB(db_path=db_path)
        try:
            db.create_session(session_id="s1", source="cli")
            db.append_message("s1", role="user", content="大别山项目计划书")
            db.append_message("s1", role="user", content="长江大桥设计方案")

            # 3+ CJK chars would normally use trigram, but it's unavailable.
            # Must fall back to LIKE and still return results.
            results = db.search_messages("大别山")
            assert len(results) == 1
            # Note: search_messages strips 'content' from results; use 'snippet'.
            assert "大别山" in results[0]["snippet"]
        finally:
            db.close()


# =========================================================================
# Message storage
# =========================================================================

class TestMessageStorage:
    def test_append_and_get_messages(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="Hello")
        db.append_message("s1", role="assistant", content="Hi there!")

        messages = db.get_messages("s1")
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"



    def test_startup_heals_null_active_rows(self, tmp_path):
        """Rows written as active=NULL before the fix are un-hidden on startup.

        The repair UPDATE used to be gated at schema_version < 12, so
        already-v12+ databases (the exact population hit by #51646) never
        healed their historical NULL rows. It now runs on every startup.
        """
        db_path = tmp_path / "legacy_state.db"
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE schema_version (version INTEGER);
            INSERT INTO schema_version VALUES (12);
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY, source TEXT, started_at REAL, ended_at REAL,
                message_count INTEGER DEFAULT 0, tool_call_count INTEGER DEFAULT 0,
                title TEXT, parent_session_id TEXT, model_config TEXT
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT,
                tool_call_id TEXT, tool_calls TEXT, tool_name TEXT,
                timestamp REAL NOT NULL, token_count INTEGER, finish_reason TEXT,
                reasoning TEXT, reasoning_content TEXT, reasoning_details TEXT,
                codex_reasoning_items TEXT, codex_message_items TEXT,
                platform_message_id TEXT, observed INTEGER DEFAULT 0
            );
            CREATE TABLE state_meta (key TEXT PRIMARY KEY, value TEXT);
            """
        )
        # Default-less active column, as seen in the wild (#51646 PRAGMA).
        conn.execute("ALTER TABLE messages ADD COLUMN active INTEGER")
        conn.execute("ALTER TABLE messages ADD COLUMN compacted INTEGER DEFAULT 0")
        conn.execute(
            "INSERT INTO sessions (id, source, started_at) VALUES ('s1', 'discord', 1.0)"
        )
        # A row written by the pre-fix INSERT: active is NULL.
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) "
            "VALUES ('s1', 'user', 'old hidden turn', 1.0)"
        )
        conn.commit()
        conn.close()

        session_db = SessionDB(db_path=db_path)
        try:
            active = session_db._conn.execute(
                "SELECT active FROM messages WHERE content = 'old hidden turn'"
            ).fetchone()[0]
            assert active == 1
            assert len(session_db.get_messages_as_conversation("s1")) == 1
        finally:
            session_db.close()


























    def test_get_messages_as_conversation_strips_leaked_memory_context(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message(
            "s1",
            role="assistant",
            content=(
                "<memory-context>\n"
                "[System note: The following is recalled memory context, NOT new user input. Treat as informational background data.]\n\n"
                "## Honcho Context\n"
                "stale memory\n"
                "</memory-context>\n\n"
                "Visible answer"
            ),
        )

        conv = db.get_messages_as_conversation("s1")
        assert len(conv) == 1
        assert conv[0]["role"] == "assistant"
        assert conv[0]["content"] == "Visible answer"
        assert isinstance(conv[0].get("timestamp"), float)

    def test_reasoning_persisted_and_restored(self, db):
        """Reasoning text is stored for assistant messages and restored by
        get_messages_as_conversation() so providers receive coherent multi-turn
        reasoning context."""
        db.create_session(session_id="s1", source="telegram")
        db.append_message("s1", role="user", content="create a cron job")
        db.append_message(
            "s1",
            role="assistant",
            content=None,
            tool_calls=[{"function": {"name": "cronjob", "arguments": "{}"}, "id": "c1", "type": "function"}],
            reasoning="I should call the cronjob tool to schedule this.",
        )
        db.append_message("s1", role="tool", content='{"job_id": "abc"}', tool_call_id="c1")

        conv = db.get_messages_as_conversation("s1")
        assert len(conv) == 3
        # reasoning must be present on the assistant message
        assistant = conv[1]
        assert assistant["role"] == "assistant"
        assert assistant.get("reasoning") == "I should call the cronjob tool to schedule this."
        # user and tool messages must NOT carry reasoning
        assert "reasoning" not in conv[0]
        assert "reasoning" not in conv[2]










# =========================================================================
# Timestamp preservation
# =========================================================================


class TestTimestampPreservation:
    """Tests for the timestamp preservation feature.

    ``append_message()`` and ``replace_messages()`` now accept/forward an
    optional ``timestamp`` parameter.  These tests verify custom timestamps
    survive the round trip through the DB and fall back to ``time.time()``
    when omitted.
    """

    @staticmethod
    def _build_messages(ts_list, contents=None, roles=None):
        """Build message dicts with explicit timestamps for testing."""
        if contents is None:
            contents = [f"msg-{i}" for i in range(len(ts_list))]
        if roles is None:
            roles = ["user", "assistant"] * (len(ts_list) // 2 + 1)
        return [
            {"role": roles[i], "content": contents[i], "timestamp": ts}
            for i, ts in enumerate(ts_list)
        ]

    def _raw_timestamps(self, db, session_id):
        """Query timestamp column directly from SQLite for verification."""
        rows = db._conn.execute(
            "SELECT timestamp FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def test_append_message_with_explicit_timestamp(self, db):
        """A caller-supplied timestamp is stored and round-tripped."""
        db.create_session(session_id="s1", source="cli")
        ts = 1_234_567.0
        mid = db.append_message("s1", role="user", content="hello",
                                timestamp=ts)
        msgs = db.get_messages("s1")
        assert len(msgs) == 1
        assert msgs[0]["timestamp"] == ts
        assert msgs[0]["id"] == mid
        raw = self._raw_timestamps(db, "s1")
        assert raw == [ts]




    def test_replace_messages_preserves_timestamps(self, db):
        """Message dicts with ``timestamp`` passed to ``replace_messages``
        retain those timestamps after the rewrite."""
        db.create_session(session_id="s1", source="cli")
        msgs_in = [
            {"role": "user", "content": "first", "timestamp": 100.0},
            {"role": "assistant", "content": "second", "timestamp": 200.0},
            {"role": "user", "content": "third", "timestamp": 300.0},
        ]
        db.replace_messages("s1", msgs_in)
        msgs_out = db.get_messages("s1")
        assert [m["timestamp"] for m in msgs_out] == [100.0, 200.0, 300.0]
        assert self._raw_timestamps(db, "s1") == [100.0, 200.0, 300.0]





    def test_compression_replace_roundtrip_preserves_timestamps(self, db):
        """Compression-style rewrite: replace_messages with dicts loaded from
        get_messages_as_conversation must keep the surviving messages'
        original timestamps (#28841)."""
        timestamps = [1_500_000_000.0, 1_500_000_100.0, 1_500_000_200.0]
        db.create_session(session_id="s1", source="cli")
        for i, ts in enumerate(timestamps):
            db.append_message(
                "s1",
                role="user" if i % 2 == 0 else "assistant",
                content=f"msg-{i}",
                timestamp=ts,
            )

        history = db.get_messages_as_conversation("s1")
        # Simulate a compression that keeps the last two turns verbatim and
        # prepends a fresh summary message (no timestamp — falls back to now).
        compressed = [{"role": "user", "content": "[summary]"}] + history[-2:]
        db.replace_messages("s1", compressed)

        raw = self._raw_timestamps(db, "s1")
        assert len(raw) == 3
        assert raw[1:] == timestamps[-2:]
        assert raw[0] > timestamps[-1]  # summary stamped with a current time


# =========================================================================
# FTS5 search
# =========================================================================

class TestFTS5Search:
    def test_search_finds_content(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="How do I deploy with Docker?")
        db.append_message("s1", role="assistant", content="Use docker compose up.")

        results = db.search_messages("docker")
        assert len(results) == 2
        # At least one result should mention docker
        snippets = [r.get("snippet", "") for r in results]
        assert any("docker" in s.lower() or "Docker" in s for s in snippets)






    def test_search_returns_context(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="Tell me about Kubernetes")
        db.append_message("s1", role="assistant", content="Kubernetes is an orchestrator.")

        results = db.search_messages("Kubernetes")
        assert len(results) == 2
        assert "context" in results[0]
        assert isinstance(results[0]["context"], list)
        assert len(results[0]["context"]) > 0








    def test_sanitize_fts5_query_strips_dangerous_chars(self):
        """Unit test for _sanitize_fts5_query static method."""
        from hermes_state import SessionDB
        s = SessionDB._sanitize_fts5_query
        assert s('hello world') == 'hello world'
        assert '+' not in s('C++')
        assert '"' not in s('"unterminated')
        assert '(' not in s('(problem')
        assert '{' not in s('{test}')
        # Dangling operators removed
        assert s('hello AND') == 'hello'
        assert s('OR world') == 'world'
        # Leading bare * removed
        assert s('***') == ''
        # Valid prefix kept
        assert s('deploy*') == 'deploy*'
        # Colon (FTS5 column-filter operator) stripped, both terms preserved
        assert ':' not in s('TODO: fix')
        assert s('TODO: fix').split() == ['TODO', 'fix']
        assert ':' not in s('error:timeout')






    def test_long_search_query_is_capped_and_does_not_crash(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="bounded sanitizer target")

        query = ('"' * 50_000) + (" bounded" * 10_000)
        start = time.perf_counter()
        results = db.search_messages(query)
        elapsed = time.perf_counter() - start

        assert isinstance(results, list)
        assert elapsed < 1.0


# =========================================================================
# CJK (Chinese/Japanese/Korean) LIKE fallback
# =========================================================================

class TestCJKSearchFallback:
    """Regression tests for CJK search (see #11511).

    SQLite FTS5's default tokenizer treats contiguous CJK runs as a single
    token ("和其他agent的聊天记录" → one token), so substring queries like
    "记忆断裂" return 0 rows despite the data being present. SessionDB falls
    back to LIKE substring matching whenever FTS5 returns no results and
    the query contains CJK characters.
    """

    def test_cjk_detection_covers_all_ranges(self):
        from hermes_state import SessionDB
        f = SessionDB._contains_cjk
        # Chinese (CJK Unified Ideographs)
        assert f("记忆断裂") is True
        # Japanese Hiragana + Katakana
        assert f("こんにちは") is True
        assert f("カタカナ") is True
        # Korean Hangul syllables (both early and late — guards against
        # the \ud7a0-\ud7af typo seen in one of the duplicate PRs)
        assert f("안녕하세요") is True
        assert f("기억") is True
        # Non-CJK
        assert f("hello world") is False
        assert f("日本語mixedwithenglish") is True
        assert f("") is False









        # No CJK in query → LIKE fallback must not run. We don't assert this
        # directly (no instrumentation), but the FTS5 path produces an
        # FTS5-style snippet with highlight markers when the term is short.
        # At minimum: english queries must still match.


    def test_mixed_cjk_english_query(self, db):
        """Mixed queries should still fall back to LIKE when FTS5 misses."""
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="讨论Agent通信协议")
        # "Agent通信" is CJK+English — FTS5 default tokenizer indexes the
        # whole CJK run with embedded "agent" as separate tokens; the LIKE
        # fallback handles the substring correctly.
        results = db.search_messages("Agent通信")
        assert len(results) == 1



    def test_cjk_like_escapes_wildcards(self, db):
        """Special characters (%, _) in CJK queries are treated as literals."""
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="cli")
        db.append_message("s1", role="user", content="达成100%完成率")
        db.append_message("s2", role="user", content="达成100完成率是目标")
        # The % in the query must be literal — should only match s1
        results = db.search_messages("100%完成")
        assert len(results) == 1
        assert results[0]["session_id"] == "s1"





# =========================================================================
# Session search and listing
# =========================================================================

class TestSearchSessions:
    def test_list_all_sessions(self, db):
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="telegram")

        sessions = db.search_sessions()
        assert len(sessions) == 2


    def test_pagination(self, db):
        for i in range(5):
            db.create_session(session_id=f"s{i}", source="cli")

        page1 = db.search_sessions(limit=2)
        page2 = db.search_sessions(limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 2
        assert page1[0]["id"] != page2[0]["id"]


# =========================================================================
# Counts
# =========================================================================

class TestCounts:

    def test_session_count_by_source(self, db):
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="telegram")
        db.create_session(session_id="s3", source="cli")
        assert db.session_count(source="cli") == 2
        assert db.session_count(source="telegram") == 1






    def test_message_count_total(self, db):
        assert db.message_count() == 0
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="Hello")
        db.append_message("s1", role="assistant", content="Hi")
        assert db.message_count() == 2



# =========================================================================
# Delete and export
# =========================================================================

class TestDeleteAndExport:
    def test_delete_session(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="Hello")

        assert db.delete_session("s1") is True
        assert db.get_session("s1") is None
        assert db.message_count(session_id="s1") == 0





    def test_resolve_session_id_ambiguous_prefix_returns_none(self, db):
        db.create_session(session_id="20260315_092437_c9a6aa", source="cli")
        db.create_session(session_id="20260315_092437_c9a6bb", source="cli")
        assert db.resolve_session_id("20260315_092437_c9a6") is None













    def test_import_sessions_rejects_oversized_payloads_atomically(self, db):
        oversized = "x" * (SessionDB._IMPORT_MAX_SESSION_BYTES + 1)
        result = db.import_sessions(
            [{"id": "oversized", "messages": [{"role": "user", "content": oversized}]}]
        )

        assert result["ok"] is False
        assert result["errors"][0]["error"] == "session exceeds the import size limit"
        assert db.get_session("oversized") is None

        result = db.import_sessions(
            [
                {
                    "id": "too-many-messages",
                    "messages": [
                        {"role": "user", "content": "x"}
                    ]
                    * (SessionDB._IMPORT_MAX_MESSAGES_PER_SESSION + 1),
                }
            ]
        )

        assert result["ok"] is False
        assert result["errors"][0]["error"] == "messages exceeds the per-session import limit"
        assert db.get_session("too-many-messages") is None


# =========================================================================
# Prune
# =========================================================================

class TestPruneSessions:
    def test_prune_old_ended_sessions(self, db):
        # Create and end an "old" session
        db.create_session(session_id="old", source="cli")
        db.end_session("old", end_reason="done")
        # Manually backdate started_at
        db._conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = ?",
            (time.time() - 100 * 86400, "old"),
        )
        db._conn.commit()

        # Create a recent session
        db.create_session(session_id="new", source="cli")

        pruned = db.prune_sessions(older_than_days=90)
        assert pruned == 1
        assert db.get_session("old") is None
        session = db.get_session("new")
        assert session is not None
        assert session["id"] == "new"


    def test_prune_skips_active_sessions(self, db):
        db.create_session(session_id="active", source="cli")
        # Backdate but don't end
        db._conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = ?",
            (time.time() - 200 * 86400, "active"),
        )
        db._conn.commit()

        pruned = db.prune_sessions(older_than_days=90)
        assert pruned == 0
        assert db.get_session("active") is not None





class TestPruneSessionFilters:
    """Extended filter surface shared by prune/archive/list_prune_candidates."""

    @staticmethod
    def _mk(db, sid, *, source="cli", age_seconds=0, title=None,
            end_reason="done", message_count=0, cwd=None):
        db.create_session(session_id=sid, source=source, cwd=cwd)
        db.end_session(sid, end_reason=end_reason)
        db._conn.execute(
            "UPDATE sessions SET started_at = ?, message_count = ?, title = ? "
            "WHERE id = ?",
            (time.time() - age_seconds, message_count, title, sid),
        )
        db._conn.commit()

    def test_started_after_window_prunes_only_recent(self, db):
        self._mk(db, "recent1", age_seconds=3600)       # 1h ago
        self._mk(db, "recent2", age_seconds=2 * 3600)   # 2h ago
        self._mk(db, "old", age_seconds=10 * 3600)      # 10h ago

        cutoff = time.time() - 5 * 3600
        pruned = db.prune_sessions(older_than_days=None, started_after=cutoff)
        assert pruned == 2
        assert db.get_session("old") is not None
        assert db.get_session("recent1") is None


    def test_title_and_message_count_filters(self, db):
        self._mk(db, "smoke1", age_seconds=60, title="Codex Smoke Test 1",
                 message_count=2)
        self._mk(db, "smoke2", age_seconds=60, title="codex smoke test 2",
                 message_count=8)
        self._mk(db, "real", age_seconds=60, title="Debugging auth",
                 message_count=8)

        rows = db.list_prune_candidates(title_like="smoke")
        assert {r["id"] for r in rows} == {"smoke1", "smoke2"}

        pruned = db.prune_sessions(
            older_than_days=None, title_like="Smoke", max_messages=3
        )
        assert pruned == 1
        assert db.get_session("smoke1") is None
        assert db.get_session("smoke2") is not None
        assert db.get_session("real") is not None






    @staticmethod
    def _mk_rich(db, sid, **cols):
        """Create an ended session then set arbitrary sessions columns."""
        db.create_session(session_id=sid, source=cols.pop("source", "cli"))
        db.end_session(sid, end_reason=cols.pop("end_reason", "done"))
        cols.setdefault("started_at", time.time() - 60)
        sets = ", ".join(f"{k} = ?" for k in cols)
        db._conn.execute(
            f"UPDATE sessions SET {sets} WHERE id = ?", (*cols.values(), sid)
        )
        db._conn.commit()






    def test_unknown_filter_rejected(self, db):
        import pytest as _pytest
        with _pytest.raises(TypeError):
            db.prune_sessions(older_than_days=None, bogus_filter="x")


class TestDeleteSessionOrphansChildren:
    def test_delete_orphans_children(self, db):
        """Deleting a parent session orphans its children."""
        db.create_session(session_id="parent", source="cli")
        db.create_session(session_id="child", source="cli", parent_session_id="parent")
        db.create_session(session_id="grandchild", source="cli", parent_session_id="child")

        # Should not raise IntegrityError
        result = db.delete_session("parent")
        assert result is True
        assert db.get_session("parent") is None
        # Child is orphaned, not deleted
        child = db.get_session("child")
        assert child is not None
        assert child["parent_session_id"] is None
        # Grandchild is untouched
        grandchild = db.get_session("grandchild")
        assert grandchild is not None
        assert grandchild["parent_session_id"] == "child"


class TestBulkDeleteSessions:
    """``delete_sessions(ids)`` — the bulk-delete primitive backing the
    sessions-page "Delete N selected" button. Per-row contract matches
    :meth:`SessionDB.delete_session` (children orphaned, not cascade-
    deleted), but applied across the whole list in one transaction.

    Invariants this class locks in:

    1. Returns the real deleted count (existing intersection), not
       just ``len(session_ids)`` — selection state in the UI can race
       against another tab's delete.
    2. Unknown IDs are silently skipped, never raise.
    3. ``message_count > 0`` sessions are deleted too — unlike
       ``delete_empty_sessions``, the user explicitly picked them, so
       we trust the selection.
    4. Live (un-ended) and archived sessions ARE deleted on explicit
       selection (no bulk-sweep safety guards apply when the user
       hand-picks the row).
    5. Children of any deleted parent are orphaned, even when the
       parent is mid-list.
    6. ``[]`` / ``None``-laden lists are safe no-ops.
    """

    def test_deletes_listed_sessions(self, db):
        db.create_session(session_id="a", source="cli")
        db.append_message("a", role="user", content="hi")
        db.create_session(session_id="b", source="cli")
        db.create_session(session_id="c", source="cli")

        deleted = db.delete_sessions(["a", "b"])
        assert deleted == 2
        assert db.get_session("a") is None
        assert db.get_session("b") is None
        # Unlisted survives.
        assert db.get_session("c") is not None





    def test_orphans_children_of_deleted_parents(self, db):
        """Bulk-deleting a parent leaves its children alive but
        re-parented to NULL. Same contract as the single-session
        :meth:`delete_session` path."""
        db.create_session(session_id="parent", source="cli")
        db.create_session(
            session_id="child", source="cli", parent_session_id="parent"
        )

        deleted = db.delete_sessions(["parent"])
        assert deleted == 1
        child = db.get_session("child")
        assert child is not None
        assert child["parent_session_id"] is None


    def test_cleans_up_transcript_files(self, db, tmp_path):
        """When ``sessions_dir`` is provided, on-disk transcripts are
        swept as part of the bulk operation — mirrors the per-row
        :meth:`delete_session(sessions_dir=...)` behaviour so the
        bulk-delete CLI / web flows don't leak files."""
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="cli")
        (tmp_path / "s1.jsonl").write_text("")
        (tmp_path / "s2.json").write_text("{}")

        deleted = db.delete_sessions(["s1", "s2"], sessions_dir=tmp_path)
        assert deleted == 2
        assert not (tmp_path / "s1.jsonl").exists()
        assert not (tmp_path / "s2.json").exists()


class TestDeleteEmptySessions:
    """``delete_empty_sessions`` sweeps every ended, non-archived session
    whose ``message_count`` is 0. Backs the dashboard's "Delete empty"
    button — see ``SessionsPage.tsx`` + ``DELETE /api/sessions/empty``
    in ``hermes_cli/web_server.py``.

    Invariants this class locks in:

    1. Only ``message_count = 0`` rows are touched.
    2. Active (un-ended) sessions are skipped even if they're empty —
       the agent might be mid-handshake, and yanking the row would
       race the live runtime.
    3. Archived sessions are skipped — the user already filed them away.
    4. Children of a deleted parent are orphaned (parent_session_id →
       NULL) rather than cascade-deleted, matching the
       ``delete_session`` / ``prune_sessions`` contract.
    5. The pre-DB count matches the post-DB delete return value.
    """

    def test_count_and_delete_empties_only(self, db):
        # Two empty + ended sessions → both should be in the kill list.
        db.create_session(session_id="empty1", source="cli")
        db.end_session("empty1", end_reason="done")
        db.create_session(session_id="empty2", source="cli")
        db.end_session("empty2", end_reason="done")

        # One non-empty + ended session → must survive.
        db.create_session(session_id="hasmsg", source="cli")
        db.append_message("hasmsg", role="user", content="Hello")
        db.end_session("hasmsg", end_reason="done")

        assert db.count_empty_sessions() == 2

        deleted = db.delete_empty_sessions()
        assert deleted == 2
        assert db.get_session("empty1") is None
        assert db.get_session("empty2") is None
        assert db.get_session("hasmsg") is not None
        assert db.count_empty_sessions() == 0





    def test_cleans_up_on_disk_transcript_files(self, db, tmp_path):
        """When ``sessions_dir`` is provided, transcript files left
        behind by a crashed gateway (``request_dump_*.json``) are swept
        too. Empty sessions rarely have ``{id}.json`` / ``.jsonl``
        transcripts, but the request-dump path is real — the gateway
        writes one before the first reply lands, so a crash mid-reply
        produces an empty session with a non-empty dump file."""
        db.create_session(session_id="empty_with_dump", source="cli")
        db.end_session("empty_with_dump", end_reason="done")

        dump = tmp_path / "request_dump_empty_with_dump_0.json"
        dump.write_text("{}")
        transcript = tmp_path / "empty_with_dump.jsonl"
        transcript.write_text("")

        deleted = db.delete_empty_sessions(sessions_dir=tmp_path)
        assert deleted == 1
        assert not dump.exists()
        assert not transcript.exists()


# =========================================================================
# Schema and WAL mode
# =========================================================================

# =========================================================================
# Session title
# =========================================================================

class TestSessionTitle:
    def test_set_and_get_title(self, db):
        db.create_session(session_id="s1", source="cli")
        assert db.set_session_title("s1", "My Session") is True

        session = db.get_session("s1")
        assert session["title"] == "My Session"








    def test_title_empty_string_normalized_to_none(self, db):
        """Empty strings are normalized to None (clearing the title)."""
        db.create_session(session_id="s1", source="cli")
        db.set_session_title("s1", "My Title")
        # Setting to empty string should clear the title (normalize to None)
        db.set_session_title("s1", "")

        session = db.get_session("s1")
        assert session["title"] is None




class TestSessionTitleIndexRepair:
    @staticmethod
    def _seed_legacy_database(tmp_path, *, duplicate_titles):
        db_path = tmp_path / "legacy_titles.db"
        session_db = SessionDB(db_path=db_path)
        session_db.create_session("older", "cli")
        session_db.append_message("older", role="user", content="keep older message")
        session_db.create_session("newer", "cli")
        session_db.append_message(
            "newer", role="assistant", content="keep newer message"
        )
        session_db.create_session("unique", "cli")
        session_db.set_session_title("unique", "unique-title")
        session_db.close()

        with sqlite3.connect(db_path) as conn:
            conn.execute("DROP INDEX idx_sessions_title_unique")
            if duplicate_titles:
                conn.execute(
                    "UPDATE sessions SET title = 'shared-title' "
                    "WHERE id IN ('older', 'newer')"
                )

        return db_path

    def test_duplicate_titles_are_repaired_without_deleting_sessions(self, tmp_path):
        db_path = self._seed_legacy_database(tmp_path, duplicate_titles=True)

        reopened = SessionDB(db_path=db_path)
        try:
            conn = reopened._conn
            assert conn is not None
            rows = {
                row["id"]: row
                for row in conn.execute(
                    "SELECT id, title FROM sessions ORDER BY rowid"
                ).fetchall()
            }
            assert set(rows) == {"older", "newer", "unique"}
            assert rows["older"]["title"] is None
            assert rows["newer"]["title"] == "shared-title"
            assert rows["unique"]["title"] == "unique-title"
            assert reopened.get_messages("older")[0]["content"] == "keep older message"
            assert reopened.get_messages("newer")[0]["content"] == "keep newer message"
            index = conn.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE type = 'index' AND name = 'idx_sessions_title_unique'"
            ).fetchone()
            assert index is not None
        finally:
            reopened.close()




class TestSessionTitleLineage:
    """Renaming a compression continuation back to its base title must succeed
    by transferring the title off the ended, hidden predecessor.

    After a context compaction the original session is ended and projected
    behind its live tip in the session list (list_sessions_rich), so the user
    cannot see or free it. Without lineage-aware handling, renaming the visible
    tip back to the base name dead-ends with "already in use by <session they
    can't find>".
    """

    def _make_compression_chain(self, db, t0, *, root="root", tip="tip"):
        db.create_session(root, "cli")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0, root))
        db._conn.execute(
            "UPDATE sessions SET ended_at=?, end_reason='compression' WHERE id=?",
            (t0 + 100, root),
        )
        db.create_session(tip, "cli", parent_session_id=root)
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0 + 200, tip))
        db._conn.commit()

    def test_rename_continuation_back_to_base_transfers_title(self, db):
        import time as _time
        self._make_compression_chain(db, _time.time() - 3600)
        db.set_session_title("root", "fingerprint-scanner")
        db.set_session_title("tip", "fingerprint-scanner #2")

        # User renames the visible tip back to the base name — must succeed.
        assert db.set_session_title("tip", "fingerprint-scanner") is True
        assert db.get_session("tip")["title"] == "fingerprint-scanner"
        # Title transferred off the hidden ancestor — no duplicate titles.
        assert db.get_session("root")["title"] is None


    def test_unrelated_session_still_conflicts(self, db):
        db.create_session("a", "cli")
        db.create_session("b", "cli")
        db.set_session_title("a", "shared")
        with pytest.raises(ValueError, match="already in use"):
            db.set_session_title("b", "shared")
        # The unrelated holder keeps its title.
        assert db.get_session("a")["title"] == "shared"



class TestSanitizeTitle:
    """Tests for SessionDB.sanitize_title() validation and cleaning."""

    def test_normal_title_unchanged(self):
        assert SessionDB.sanitize_title("My Project") == "My Project"







    def test_control_chars_stripped(self):
        # Null byte, bell, backspace, etc.
        assert SessionDB.sanitize_title("hello\x00world") == "helloworld"
        assert SessionDB.sanitize_title("\x07\x08test\x1b") == "test"







    def test_exceeds_max_length_raises(self):
        title = "A" * 101
        with pytest.raises(ValueError, match="too long"):
            SessionDB.sanitize_title(title)








class TestSchemaInit:
    def test_wal_mode(self, db):
        """Prefer WAL on fixed SQLite; DELETE on WAL-reset-vulnerable builds (#69784)."""
        from hermes_state import is_sqlite_wal_reset_vulnerable

        cursor = db._conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0].lower()
        if is_sqlite_wal_reset_vulnerable():
            assert mode == "delete"
        else:
            assert mode == "wal"







    def test_telegram_topic_binding_roundtrip_requires_explicit_schema(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session(
            session_id="topic-session",
            source="telegram",
            user_id="208214988",
        )

        assert db.get_telegram_topic_binding(chat_id="208214988", thread_id="17585") is None

        db.bind_telegram_topic(
            chat_id="208214988",
            thread_id="17585",
            user_id="208214988",
            session_key="telegram:dm:208214988:thread:17585",
            session_id="topic-session",
        )

        binding = db.get_telegram_topic_binding(chat_id="208214988", thread_id="17585")
        assert binding is not None
        assert binding["chat_id"] == "208214988"
        assert binding["thread_id"] == "17585"
        assert binding["user_id"] == "208214988"
        assert binding["session_key"] == "telegram:dm:208214988:thread:17585"
        assert binding["session_id"] == "topic-session"
        assert db.get_meta("telegram_dm_topic_schema_version") == "2"
        db.close()







    def test_schema_sql_is_source_of_truth(self, db):
        """Every column in SCHEMA_SQL exists in the live database.

        This is the architectural invariant: SCHEMA_SQL declares the
        desired schema, _reconcile_columns ensures it matches reality.
        """
        from hermes_state import SCHEMA_SQL

        expected = SessionDB._parse_schema_columns(SCHEMA_SQL)
        for table_name, declared_cols in expected.items():
            live_cols = {
                r[1]
                for r in db._conn.execute(
                    f'PRAGMA table_info("{table_name}")'
                ).fetchall()
            }
            for col_name in declared_cols:
                assert col_name in live_cols, (
                    f"Column {col_name} declared in SCHEMA_SQL for {table_name} "
                    f"but missing from live DB. Live columns: {live_cols}"
                )


class TestTitleUniqueness:
    """Tests for unique title enforcement and title-based lookups."""

    def test_duplicate_title_raises(self, db):
        """Setting a title already used by another session raises ValueError."""
        db.create_session("s1", "cli")
        db.create_session("s2", "cli")
        db.set_session_title("s1", "my project")
        with pytest.raises(ValueError, match="already in use"):
            db.set_session_title("s2", "my project")


    def test_null_titles_not_unique(self, db):
        """Multiple sessions can have NULL titles (no constraint violation)."""
        db.create_session("s1", "cli")
        db.create_session("s2", "cli")
        # Both have NULL titles — no error
        assert db.get_session("s1")["title"] is None
        assert db.get_session("s2")["title"] is None






class TestTitleLineage:
    """Tests for title lineage resolution and auto-numbering."""

    def test_resolve_exact_title(self, db):
        db.create_session("s1", "cli")
        db.set_session_title("s1", "my project")
        assert db.resolve_session_by_title("my project") == "s1"



    def test_resolve_nonexistent_title(self, db):
        assert db.resolve_session_by_title("nonexistent") is None

    def test_next_title_no_existing(self, db):
        """With no existing sessions, base title is returned as-is."""
        assert db.get_next_title_in_lineage("my project") == "my project"





class TestTitleSqlWildcards:
    """Titles containing SQL LIKE wildcards (%, _) must not cause false matches."""

    def test_resolve_title_with_underscore(self, db):
        """A title like 'test_project' should not match 'testXproject #2'."""
        db.create_session("s1", "cli")
        db.set_session_title("s1", "test_project")
        db.create_session("s2", "cli")
        db.set_session_title("s2", "testXproject #2")
        # Resolving "test_project" should return s1 (exact), not s2
        assert db.resolve_session_by_title("test_project") == "s1"




class TestListSessionsRich:
    """Tests for enhanced session listing with preview and last_active."""

    def test_preview_from_first_user_message(self, db):
        db.create_session("s1", "cli")
        db.append_message("s1", "system", "You are a helpful assistant.")
        db.append_message("s1", "user", "Help me refactor the auth module please")
        db.append_message("s1", "assistant", "Sure, let me look at it.")
        sessions = db.list_sessions_rich()
        assert len(sessions) == 1
        assert "Help me refactor the auth module" in sessions[0]["preview"]












    def test_rich_list_session_key_filter_precedes_limit(self, db):
        lane_key = "agent:main:telegram:dm:lane"
        db.create_session(
            "lane_oldest", "telegram", session_key=lane_key,
            user_id="lane-user", chat_id="lane",
        )
        db.create_session(
            "lane_newest", "telegram", session_key=lane_key,
            user_id="lane-user", chat_id="lane",
        )
        for i in range(60):
            db.create_session(
                f"foreign_{i}", "telegram",
                session_key=f"agent:main:telegram:dm:foreign-{i}",
                user_id=f"foreign-user-{i}", chat_id=f"foreign-{i}",
            )
        db.create_session(
            "legacy_null_key", "telegram", user_id="lane-user", chat_id="lane"
        )

        sessions = db.list_sessions_rich(
            source="telegram", session_key=lane_key, limit=2
        )

        assert [session["id"] for session in sessions] == [
            "lane_newest", "lane_oldest",
        ]

    def test_rich_list_session_key_scopes_search_and_projects_compression(self, db):
        lane_key = "agent:main:telegram:dm:lane"
        db.create_session(
            "lane_root", "telegram", session_key=lane_key,
            user_id="lane-user", chat_id="lane",
        )
        db.set_session_title("lane_root", "Needle root")
        db.end_session("lane_root", "compression")
        db.create_session(
            "lane_tip", "telegram", session_key=lane_key,
            user_id="lane-user", chat_id="lane", parent_session_id="lane_root",
        )
        db.set_session_title("lane_tip", "Needle continuation")
        db.append_message("lane_tip", "user", "latest lane activity")
        db.create_session(
            "foreign_match", "telegram",
            session_key="agent:main:telegram:dm:foreign",
            user_id="foreign-user", chat_id="foreign",
        )
        db.set_session_title("foreign_match", "Needle foreign")

        sessions = db.list_sessions_rich(
            source="telegram",
            session_key=lane_key,
            search_query="needle",
            order_by_last_active=True,
            limit=1,
        )

        assert [session["id"] for session in sessions] == ["lane_tip"]
        assert sessions[0]["_lineage_root_id"] == "lane_root"

    def test_session_key_predicate_can_use_session_key_index(self, db):
        plan = db._conn.execute(
            "EXPLAIN QUERY PLAN "
            "SELECT s.id FROM sessions s WHERE s.session_key = ? "
            "ORDER BY s.started_at DESC LIMIT 10",
            ("agent:main:telegram:dm:lane",),
        ).fetchall()

        detail = " ".join(row[-1] for row in plan)
        assert "idx_sessions_session_key" in detail, detail

    def test_delegate_subagent_marker_hides_orphaned_row(self, db):
        """``_delegate_from`` keeps delegate rows out of pickers after orphaning."""
        db.create_session("parent", "cli")
        db.create_session(
            "delegate",
            "cli",
            parent_session_id="parent",
            model_config={"_delegate_from": "parent"},
        )
        db.append_message("delegate", "user", "scan the repo")

        assert "delegate" not in [s["id"] for s in db.list_sessions_rich()]

        db._conn.execute(
            "UPDATE sessions SET parent_session_id = NULL WHERE id = ?", ("delegate",)
        )
        db._conn.commit()

        assert "delegate" not in [s["id"] for s in db.list_sessions_rich()]


    def test_delete_session_expected_targets_fail_closed_on_new_delegate(self, db):
        db.create_session("parent", "cli")
        db.create_session(
            "delegate",
            "cli",
            parent_session_id="parent",
            model_config={"_delegate_from": "parent"},
        )
        db.create_session(
            "branch",
            "cli",
            parent_session_id="parent",
            model_config={"_branched_from": "parent"},
        )

        expected_ids = db.get_session_delete_targets("parent")
        assert expected_ids == ["parent", "delegate"]

        db.create_session(
            "late-delegate",
            "cli",
            parent_session_id="parent",
            model_config={"_delegate_from": "parent"},
        )

        assert (
            db.delete_session("parent", expected_delete_ids=expected_ids) is False
        )
        assert db.get_session("parent") is not None
        assert db.get_session("delegate") is not None
        assert db.get_session("late-delegate") is not None
        assert db.get_session("branch") is not None




    def test_subagent_session_still_hidden(self, db):
        """Sub-agent children (parent NOT ended with 'branched') remain hidden."""
        db.create_session("root", "cli")
        db.create_session("delegate", "cli", parent_session_id="root")

        sessions = db.list_sessions_rich()
        ids = [s["id"] for s in sessions]
        assert "delegate" not in ids, "Delegate sub-agent should not appear in default list"
        assert "root" in ids



class TestCompressionChainProjection:
    """Tests for lineage-aware list_sessions_rich — compressed conversations
    surface as their live continuation tip, not the dead parent root.
    """

    def _build_compression_chain(self, db, t0: float):
        """Helper: builds root -> delegate -> compression-child -> tip chain.

        Returns (root_id, delegate_id, mid_id, tip_id).
        """
        # Root that gets compressed
        db.create_session("root1", "cli")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0, "root1"))
        db.append_message("root1", "user", "help me refactor auth")

        # Delegate subagent spawned while root1 was live (before it ended)
        db.create_session("delegate1", "cli", parent_session_id="root1")
        db._conn.execute(
            "UPDATE sessions SET started_at=?, ended_at=? WHERE id=?",
            (t0 + 600, t0 + 650, "delegate1"),
        )
        db.append_message("delegate1", "user", "delegate task")

        # root1 compressed at t0+1800
        t_compress_root = t0 + 1800
        db._conn.execute(
            "UPDATE sessions SET ended_at=?, end_reason=? WHERE id=?",
            (t_compress_root, "compression", "root1"),
        )

        # Continuation mid created 1s after parent ended
        db.create_session("mid1", "cli", parent_session_id="root1")
        db._conn.execute(
            "UPDATE sessions SET started_at=? WHERE id=?",
            (t_compress_root + 1, "mid1"),
        )
        db.append_message("mid1", "user", "continuing")

        # mid1 also compressed
        t_compress_mid = t_compress_root + 1800
        db._conn.execute(
            "UPDATE sessions SET ended_at=?, end_reason=? WHERE id=?",
            (t_compress_mid, "compression", "mid1"),
        )

        # Tip — latest continuation
        db.create_session("tip1", "cli", parent_session_id="mid1")
        db._conn.execute(
            "UPDATE sessions SET started_at=? WHERE id=?",
            (t_compress_mid + 1, "tip1"),
        )
        db.append_message("tip1", "user", "latest message")

        db._conn.commit()
        return ("root1", "delegate1", "mid1", "tip1")

    def test_get_compression_tip_walks_full_chain(self, db):
        import time as _time
        self._build_compression_chain(db, _time.time() - 3600)
        assert db.get_compression_tip("root1") == "tip1"
        assert db.get_compression_tip("mid1") == "tip1"
        assert db.get_compression_tip("tip1") == "tip1"



    def test_list_surfaces_tip_for_compressed_root(self, db):
        """The list must show the tip's id/message_count/preview in place of
        the root row, so users can see and resume the live conversation.
        """
        import time as _time
        self._build_compression_chain(db, _time.time() - 3600)
        # Add an uncompressed root for comparison.
        db.create_session("solo", "cli")
        db.append_message("solo", "user", "standalone")
        db._conn.commit()

        sessions = db.list_sessions_rich(source="cli", limit=20)
        ids = [s["id"] for s in sessions]
        # Only top-level conversations appear: tip1 (projected from root1) + solo.
        # Delegate children, mid1, and the dead root1 must NOT be in the list.
        assert "tip1" in ids
        assert "solo" in ids
        assert "root1" not in ids
        assert "mid1" not in ids
        assert "delegate1" not in ids

        tip_row = next(s for s in sessions if s["id"] == "tip1")
        # The row surfaces the tip's identity but preserves the root's start
        # timestamp for stable ordering and lineage tracking.
        assert tip_row["_lineage_root_id"] == "root1"
        assert tip_row["preview"].startswith("latest message")
        assert tip_row["ended_at"] is None  # tip is still live
        assert tip_row["end_reason"] is None




    def test_list_handles_broken_chain_gracefully(self, db):
        """A compression root with no child (e.g. DB corruption or a partial
        end_session call that didn't finish creating the child) must not
        crash the list — it should fall back to surfacing the root as-is.
        """
        import time as _time
        t0 = _time.time() - 100
        db.create_session("orphan", "cli")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0, "orphan"))
        db._conn.execute(
            "UPDATE sessions SET ended_at=?, end_reason=? WHERE id=?",
            (t0 + 10, "compression", "orphan"),
        )
        db._conn.commit()

        sessions = db.list_sessions_rich(source="cli", limit=10)
        ids = [s["id"] for s in sessions]
        assert "orphan" in ids
        row = next(s for s in sessions if s["id"] == "orphan")
        # No tip means no projection — row stays raw.
        assert "_lineage_root_id" not in row
        assert row["end_reason"] == "compression"


# =========================================================================
# Session source exclusion (--source flag for third-party isolation)
# =========================================================================

class TestExcludeSources:
    """Tests for exclude_sources on list_sessions_rich and search_messages."""

    def test_list_sessions_rich_excludes_tool_source(self, db):
        db.create_session("s1", "cli")
        db.create_session("s2", "tool")
        db.create_session("s3", "telegram")
        sessions = db.list_sessions_rich(exclude_sources=["tool"])
        ids = [s["id"] for s in sessions]
        assert "s1" in ids
        assert "s3" in ids
        assert "s2" not in ids





    def test_search_messages_excludes_tool_source(self, db):
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "Python deployment question")
        db.create_session("s2", "tool")
        db.append_message("s2", "user", "Python automated question")
        results = db.search_messages("Python", exclude_sources=["tool"])
        sources = [r["source"] for r in results]
        assert "cli" in sources
        assert "tool" not in sources




class TestResolveSessionByNameOrId:
    """Tests for the main.py helper that resolves names or IDs."""

    def test_resolve_by_id(self, db):
        db.create_session("test-id-123", "cli")
        session = db.get_session("test-id-123")
        assert session is not None
        assert session["id"] == "test-id-123"



# =========================================================================
# Concurrent write safety / lock contention fixes (#3139)
# =========================================================================

class TestConcurrentWriteSafety:
    def test_create_session_insert_or_ignore_is_idempotent(self, db):
        """create_session with the same ID twice must not raise (INSERT OR IGNORE)."""
        db.create_session(session_id="dup-1", source="cli", model="m")
        # Second call should be silent — no IntegrityError
        db.create_session(session_id="dup-1", source="gateway", model="m2")
        session = db.get_session("dup-1")
        # Row should exist (first write wins with OR IGNORE)
        assert session is not None
        assert session["source"] == "cli"

    def test_ensure_session_creates_missing_row(self, db):
        """ensure_session must create a minimal row when the session doesn't exist."""
        assert db.get_session("orphan-session") is None
        db.ensure_session("orphan-session", source="gateway", model="test-model")
        row = db.get_session("orphan-session")
        assert row is not None
        assert row["source"] == "gateway"
        assert row["model"] == "test-model"





# =========================================================================
# Auto-maintenance: state_meta + vacuum + maybe_auto_prune_and_vacuum
# =========================================================================

class TestStateMeta:
    def test_get_meta_missing_returns_none(self, db):
        assert db.get_meta("nonexistent") is None

    def test_set_then_get_meta(self, db):
        db.set_meta("foo", "bar")
        assert db.get_meta("foo") == "bar"



class TestVacuum:
    def test_vacuum_runs_without_error(self, db):
        """VACUUM must succeed on a fresh DB (no rows to reclaim)."""
        db.create_session(session_id="s1", source="cli")
        db.append_message(session_id="s1", role="user", content="hi")
        # Should not raise, even though there's nothing significant to reclaim.
        db.vacuum()

    def test_auto_maintenance_records_successful_vacuum(self, db, monkeypatch):
        monkeypatch.setattr(db, "prune_sessions", lambda **_kwargs: 3)
        vacuum_calls = []
        monkeypatch.setattr(db, "vacuum", lambda: vacuum_calls.append(True))

        result = db.maybe_auto_prune_and_vacuum(min_interval_hours=0)

        assert result["vacuumed"] is True
        assert vacuum_calls == [True]
        assert db.get_meta("last_vacuum") is not None

    def test_auto_maintenance_skips_recent_vacuum(self, db, monkeypatch):
        monkeypatch.setattr(db, "prune_sessions", lambda **_kwargs: 3)
        db.set_meta("last_vacuum", str(time.time()))
        vacuum_calls = []
        monkeypatch.setattr(db, "vacuum", lambda: vacuum_calls.append(True))

        result = db.maybe_auto_prune_and_vacuum(
            min_interval_hours=0,
            min_vacuum_interval_days=30,
        )

        assert result["vacuumed"] is False
        assert vacuum_calls == []

    def test_auto_maintenance_retries_after_vacuum_interval(self, db, monkeypatch):
        monkeypatch.setattr(db, "prune_sessions", lambda **_kwargs: 3)
        db.set_meta("last_vacuum", str(time.time() - 31 * 86400))
        vacuum_calls = []
        monkeypatch.setattr(db, "vacuum", lambda: vacuum_calls.append(True))

        result = db.maybe_auto_prune_and_vacuum(
            min_interval_hours=0,
            min_vacuum_interval_days=30,
        )

        assert result["vacuumed"] is True
        assert vacuum_calls == [True]

    def test_auto_maintenance_retries_after_failed_vacuum(self, db, monkeypatch):
        monkeypatch.setattr(db, "prune_sessions", lambda **_kwargs: 3)
        vacuum_calls = []

        def fail_first_vacuum():
            vacuum_calls.append(True)
            if len(vacuum_calls) == 1:
                raise RuntimeError("vacuum failed")

        monkeypatch.setattr(db, "vacuum", fail_first_vacuum)

        first = db.maybe_auto_prune_and_vacuum(min_interval_hours=0)

        assert first["vacuumed"] is False
        assert db.get_meta("last_vacuum") is None

        second = db.maybe_auto_prune_and_vacuum(min_interval_hours=0)

        assert second["vacuumed"] is True
        assert vacuum_calls == [True, True]
        assert db.get_meta("last_vacuum") is not None


class TestOptimizeFts:
    def test_optimize_returns_index_count(self, db):
        """A fresh DB has both FTS indexes; optimize merges both."""
        db.create_session(session_id="s1", source="cli")
        db.append_message(session_id="s1", role="user", content="hello world")
        statements = []
        db._conn.set_trace_callback(statements.append)
        try:
            assert db.optimize_fts() == 2
        finally:
            db._conn.set_trace_callback(None)
        optimize_sql = [sql for sql in statements if "'optimize'" in sql]
        assert len(optimize_sql) == 2
        assert not any("'merge'" in sql for sql in optimize_sql)




    def test_incremental_merge_bounded_commands_per_present_index(self, db):
        """Each pass issues bounded 'merge' commands, never 'optimize'."""
        db.create_session(session_id="s1", source="cli")
        db.append_message(session_id="s1", role="user", content="bounded merge")
        statements = []
        db._conn.set_trace_callback(statements.append)
        try:
            executed = db._merge_fts_incrementally(max_pages=37)
        finally:
            db._conn.set_trace_callback(None)

        # At least one merge command per present FTS index, and never more
        # than the per-pass command cap per index.
        present = [t for t in db._FTS_TABLES if db._fts_table_exists(t)]
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

        def _counting_merge(*, max_pages):
            calls.append(max_pages)
            return 0

        def _unexpected_optimize():
            raise AssertionError("routine cadence must not call optimize")

        monkeypatch.setattr(db, "_merge_fts_incrementally", _counting_merge)
        monkeypatch.setattr(db, "optimize_fts", _unexpected_optimize)
        db.create_session(session_id="s1", source="cli")
        for i in range(3):
            db.append_message(session_id="s1", role="user", content=f"needle {i}")
        assert calls == []  # Four successful writes are below the boundary.
        db.append_message(session_id="s1", role="user", content="needle 3")
        assert calls == [500]  # The fifth write gets the production page budget.
        for i in range(4, 8):
            db.append_message(session_id="s1", role="user", content=f"needle {i}")
        assert calls == [500]
        db.append_message(session_id="s1", role="user", content="needle 8")
        assert calls == [500, 500]  # The tenth write is the next boundary.
        assert len(db.search_messages("needle")) == 9



class TestAutoMaintenance:
    def _make_old_ended(self, db, sid: str, days_old: int = 100):
        """Create a session that is ended and was started `days_old` days ago."""
        db.create_session(session_id=sid, source="cli")
        db.end_session(sid, end_reason="done")
        db._conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = ?",
            (time.time() - days_old * 86400, sid),
        )
        db._conn.commit()

    def test_first_run_prunes_and_vacuums(self, db):
        self._make_old_ended(db, "old1", days_old=100)
        self._make_old_ended(db, "old2", days_old=100)
        db.create_session(session_id="new", source="cli")  # active, must survive

        result = db.maybe_auto_prune_and_vacuum(retention_days=90)
        assert result["skipped"] is False
        assert result["pruned"] == 2
        assert result["vacuumed"] is True
        assert result.get("error") is None
        assert db.get_session("old1") is None
        assert db.get_session("old2") is None
        assert db.get_session("new") is not None

    def test_second_call_within_interval_skips(self, db):
        self._make_old_ended(db, "old", days_old=100)
        first = db.maybe_auto_prune_and_vacuum(
            retention_days=90, min_interval_hours=24
        )
        assert first["skipped"] is False
        assert first["pruned"] == 1

        # Create another prunable session; a second call within
        # min_interval_hours should still skip without touching it.
        self._make_old_ended(db, "old2", days_old=100)
        second = db.maybe_auto_prune_and_vacuum(
            retention_days=90, min_interval_hours=24
        )
        assert second["skipped"] is True
        assert second["pruned"] == 0
        assert db.get_session("old2") is not None  # untouched






    def test_auto_prune_deletes_transcript_files(self, db, tmp_path):
        """Issue #3015: auto-prune must also delete on-disk transcript files."""
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()

        self._make_old_ended(db, "old1", days_old=100)
        self._make_old_ended(db, "old2", days_old=100)
        db.create_session(session_id="new", source="cli")  # active

        # Transcript files mimicking real gateway/CLI layout
        (sessions_dir / "old1.json").write_text("{}")
        (sessions_dir / "old1.jsonl").write_text("{}\n")
        (sessions_dir / "old2.jsonl").write_text("{}\n")
        (sessions_dir / "request_dump_old1_001.json").write_text("{}")
        (sessions_dir / "new.jsonl").write_text("{}\n")  # active, must survive

        result = db.maybe_auto_prune_and_vacuum(
            retention_days=90, sessions_dir=sessions_dir
        )
        assert result["pruned"] == 2

        # Pruned transcript files are gone
        assert not (sessions_dir / "old1.json").exists()
        assert not (sessions_dir / "old1.jsonl").exists()
        assert not (sessions_dir / "old2.jsonl").exists()
        assert not (sessions_dir / "request_dump_old1_001.json").exists()
        # Active session's transcript is untouched
        assert (sessions_dir / "new.jsonl").exists()





# =========================================================================
# FTS5 indexing of tool_calls / tool_name (#16751)
# =========================================================================

class TestFTS5ToolCallIndexing:
    """Regression tests: search_messages must see tool_name and tool_calls.

    Before #16751's fix, `messages_fts` only indexed `messages.content`, so
    tokens that only appeared in `tool_name` or the serialized `tool_calls`
    JSON were invisible to session_search even though the row was in the DB.
    """

    def test_tool_name_is_searchable(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message(
            "s1", role="assistant", content="",
            tool_name="UNIQUETOOLNAME",
        )
        results = db.search_messages("UNIQUETOOLNAME")
        assert len(results) == 1

    def test_tool_calls_args_are_searchable(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message(
            "s1", role="assistant", content="",
            tool_calls=[{
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "web_search",
                    "arguments": '{"query": "UNIQUESEARCHTOKEN"}',
                },
            }],
        )
        results = db.search_messages("UNIQUESEARCHTOKEN")
        assert len(results) == 1





class TestFTS5ToolCallMigration:
    """v11 migration: pre-existing state.db with old external-content FTS tables
    must be re-indexed so tool_name / tool_calls become searchable after upgrade."""

    def test_v10_to_v11_upgrade_backfills_tool_fields(self, tmp_path):
        """Simulate an existing user: build a v10-shaped DB by hand, insert a
        row with tool_calls, then open via SessionDB (which runs migrations).
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

        # Open via SessionDB — the legacy DB is detected as optimizable but
        # NOT auto-migrated (opt-in). Its old content-only index still works
        # for content, but doesn't yet cover tool_name/tool_calls (#16751).
        session_db = SessionDB(db_path=db_path)
        try:
            assert session_db.fts_optimize_available() is True

            # `hermes db optimize` performs the v23 transition; afterwards the
            # tool fields are searchable.
            result = session_db.optimize_fts_storage(vacuum=False)
            assert result["ok"] is True
            assert len(session_db.search_messages("LEGACYTOOL")) == 1, \
                "v23 optimize must index tool_name into FTS"
            assert len(session_db.search_messages("LEGACYARG")) == 1, \
                "v23 optimize must index tool_calls JSON into FTS"
            # schema_version bumped once the FTS layer is v23
            from hermes_state import SCHEMA_VERSION
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

        db = SessionDB(db_path=db_path)
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
            assert db.fts_optimize_available() is True
            assert db.get_meta("fts_optimize_available") == "1"

            # Legacy inline shape is intact (content shadow table still there).
            assert db._conn.execute(
                "SELECT name FROM sqlite_master WHERE name = 'messages_fts_content'"
            ).fetchone() is not None

            # Search still works on the legacy index (no deferred rebuild).
            assert db.fts_rebuild_status() is None
            assert len(db.search_messages("deployment")) == 1
            assert len(db.search_messages("send_message")) == 1  # #16751 held

            # A new write is indexed live by the legacy triggers.
            db.append_message("s1", role="user", content="AFTEROPEN token")
            assert len(db.search_messages("AFTEROPEN")) == 1
        finally:
            db.close()






    def test_v23_fresh_db_born_optimized(self, tmp_path):
        """A brand-new DB is born on v23 — no legacy layout, no opt-in flag,
        no pending rebuild."""
        db = SessionDB(db_path=tmp_path / "fresh.db")
        try:
            assert db.fts_optimize_available() is False
            assert db.fts_rebuild_status() is None
            assert db.get_meta("fts_optimize_available") is None
            # Already external-content: no shadow copy tables.
            assert db._conn.execute(
                "SELECT name FROM sqlite_master WHERE name = 'messages_fts_content'"
            ).fetchone() is None
            db.create_session(session_id="s1", source="cli")
            db.append_message("s1", role="user", content="hello fresh world")
            assert len(db.search_messages("fresh")) == 1
        finally:
            db.close()


    def test_v23_cjk_tool_role_filter_uses_like_fallback(self, tmp_path):
        """A CJK query with role_filter=['tool'] must bypass the trigram index
        (tool rows aren't in it) and still find matches via LIKE."""
        db = SessionDB(db_path=tmp_path / "fresh.db")
        try:
            db.create_session(session_id="s1", source="cli")
            db.append_message("s1", role="tool", content="错误日志：数据库连接超时",
                              tool_name="terminal")
            hits = db.search_messages("数据库连接", role_filter=["tool"])
            assert len(hits) == 1
            assert hits[0]["role"] == "tool"
        finally:
            db.close()



# ---------------------------------------------------------------------------
# apply_wal_with_fallback — read-only probe tests
# ---------------------------------------------------------------------------


class TestApplyWalProbe:
    """Unit tests for the journal_mode probe in apply_wal_with_fallback."""

    @pytest.fixture(autouse=True)
    def _assume_fixed_sqlite(self, monkeypatch):
        """These cases cover the fixed-SQLite WAL path (not the #69784 gate)."""
        import hermes_state

        monkeypatch.setattr(
            hermes_state, "is_sqlite_wal_reset_vulnerable", lambda version_info=None: False
        )


    def test_sets_wal_on_fresh_connection(self, tmp_path):
        """Probe sees 'delete', then set-pragma runs and returns 'wal'."""
        import sqlite3
        from hermes_state import apply_wal_with_fallback

        class _TracingConn(sqlite3.Connection):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.executed = []

            def execute(self, sql, params=()):
                self.executed.append(sql)
                return super().execute(sql, params)

        db_path = tmp_path / "fresh.db"
        conn = _TracingConn(str(db_path))
        try:
            result = apply_wal_with_fallback(conn)
        finally:
            conn.close()

        assert result == "wal"
        assert any("journal_mode=WAL" in sql for sql in conn.executed), (
            "set-pragma must fire on a fresh (non-WAL) connection"
        )






    def test_apply_wal_concurrent_connects_no_eio(self, tmp_path):
        """20 threads calling connect() on the same DB must not see disk I/O error."""
        import sys
        import threading
        import sqlite3
        from hermes_state import apply_wal_with_fallback

        db_path = tmp_path / "concurrent.db"
        errors = []

        def _connect_cycle():
            for _ in range(5):
                try:
                    conn = sqlite3.connect(str(db_path))
                    apply_wal_with_fallback(conn)
                    conn.close()
                except sqlite3.OperationalError as exc:
                    if "disk i/o error" in str(exc).lower():
                        errors.append(exc)

        threads = [threading.Thread(target=_connect_cycle) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"disk I/O errors from concurrent connects: {errors}"

        # Linux-only: no (deleted) WAL/SHM FDs should accumulate.
        if sys.platform == "linux":
            import os

            fd_dir = f"/proc/{os.getpid()}/fd"
            deleted_fds = []
            for fd_name in os.listdir(fd_dir):
                try:
                    target = os.readlink(os.path.join(fd_dir, fd_name))
                    if "(deleted)" in target and (
                        "wal" in target.lower() or "shm" in target.lower()
                    ):
                        deleted_fds.append(target)
                except OSError:
                    pass
            assert not deleted_fds, f"stale deleted WAL/SHM FDs: {deleted_fds}"




    def test_returns_wal_not_delete_from_probe(self, tmp_path):
        """Early-return only on 'wal'; 'delete' or 'memory' must fall through to set-pragma."""
        import sqlite3
        from hermes_state import apply_wal_with_fallback

        class _TracingConn(sqlite3.Connection):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.executed = []

            def execute(self, sql, params=()):
                self.executed.append(sql)
                return super().execute(sql, params)

        # Fresh DB is in "delete" mode — probe returns "delete", must NOT early-return.
        db_path = tmp_path / "delete_mode.db"
        conn = _TracingConn(str(db_path))
        try:
            result = apply_wal_with_fallback(conn)
        finally:
            conn.close()

        assert result == "wal"
        assert any("journal_mode=WAL" in sql for sql in conn.executed), (
            "set-pragma must fire when probe returns 'delete'"
        )


class TestSessionArchive:
    """Soft-archiving hides a session from default listings without deleting it."""

    def _seed(self, db, sid, *, archived=False):
        db.create_session(session_id=sid, source="cli")
        db.append_message(session_id=sid, role="user", content=f"hello from {sid}")
        if archived:
            db.set_session_archived(sid, True)

    def test_set_session_archived_roundtrip(self, db):
        self._seed(db, "s1")
        assert db.set_session_archived("s1", True) is True
        assert db.get_session("s1")["archived"] == 1
        assert db.set_session_archived("s1", False) is True
        assert db.get_session("s1")["archived"] == 0


    def test_archived_excluded_by_default(self, db):
        self._seed(db, "live")
        self._seed(db, "hidden", archived=True)

        ids = [s["id"] for s in db.list_sessions_rich()]
        assert ids == ["live"]
        assert db.session_count() == 1



class TestSessionPinAndStaleArchive:
    """Pin as a durable keep flag + last-activity-based stale auto-archive."""

    def _pinned(self, db, sid):
        row = db._conn.execute(
            "SELECT pinned FROM sessions WHERE id = ?", (sid,)
        ).fetchone()
        return row["pinned"] if row is not None else None

    def _make_idle(self, db, sid, *, days_idle, source="cli"):
        """A session whose latest activity was ``days_idle`` days ago."""
        db.create_session(session_id=sid, source=source)
        db.append_message(session_id=sid, role="user", content=f"msg {sid}")
        old = time.time() - days_idle * 86400
        db._conn.execute("UPDATE sessions SET started_at = ? WHERE id = ?", (old, sid))
        db._conn.execute(
            "UPDATE messages SET timestamp = ? WHERE session_id = ?", (old, sid)
        )
        db._conn.commit()

    # ── pin flag ──────────────────────────────────────────────────────────
    def test_set_session_pinned_roundtrip(self, db):
        db.create_session(session_id="s1", source="cli")
        assert db.set_session_pinned("s1", True) is True
        assert self._pinned(db, "s1") == 1
        assert db.set_session_pinned("s1", False) is True
        assert self._pinned(db, "s1") == 0



    # ── pinned back-fill past the page window ─────────────────────────────
    def test_pinned_session_survives_the_limit_window(self, db):
        """A pin outlives recency: paging must not evict a pinned row.

        Without ``include_pinned`` the desktop's Pinned section renders empty
        for any conversation that has aged off the sidebar page.
        """
        for i in range(6):
            self._make_idle(db, f"s{i}", days_idle=6 - i)
        db.set_session_pinned("s0", True)  # the oldest — off a 3-row page

        def ids(**kw):
            return [
                s["id"]
                for s in db.list_sessions_rich(
                    limit=3, min_message_count=1, order_by_last_active=True, **kw
                )
            ]

        page = ids()
        assert "s0" not in page, "precondition: the pin is off the page"

        with_pins = ids(include_pinned=True)
        assert "s0" in with_pins
        # The page itself is untouched; the pin is additive.
        assert with_pins[:3] == page
        assert len(with_pins) == len(page) + 1




    # ── stale archive ─────────────────────────────────────────────────────


    def test_pinned_sessions_are_spared(self, db):
        self._make_idle(db, "keep", days_idle=10)
        db.set_session_pinned("keep", True)

        assert db.archive_stale_sessions(3) == 0
        assert db.get_session("keep")["archived"] == 0
        # Opting out of the pin guard sweeps it.
        assert db.archive_stale_sessions(3, exclude_pinned=False) == 1
        assert db.get_session("keep")["archived"] == 1




    # ── throttled wrapper ─────────────────────────────────────────────────



class TestSessionIdSearch:
    """Session id search backs Desktop's Search Sessions UX."""

    def _seed(self, db, sid, *, content="ordinary message", archived=False, source="cli"):
        db.create_session(session_id=sid, source=source, model="test-model")
        db.append_message(session_id=sid, role="user", content=content)
        if archived:
            db.set_session_archived(sid, True)

    def test_search_sessions_by_id_matches_exact_prefix_and_substring(self, db):
        self._seed(db, "20260603_090200_abcd12", content="content without id")
        self._seed(db, "20260602_111111_other99", content="other content")

        assert [s["id"] for s in db.search_sessions_by_id("20260603_090200_abcd12")] == [
            "20260603_090200_abcd12"
        ]
        assert [s["id"] for s in db.search_sessions_by_id("20260603")] == ["20260603_090200_abcd12"]
        assert [s["id"] for s in db.search_sessions_by_id("ABCD12")] == ["20260603_090200_abcd12"]






class TestListCronJobRuns:
    """``list_cron_job_runs`` powers the desktop cron run-history endpoint.

    It must scope to exactly one job's runs via an id prefix range (not a
    substring), order newest-first, enrich with preview/last_active, and stay
    bounded by the requested window rather than the whole cron history.
    """

    def _seed_run(self, db, job_id: str, idx: int, started_at: float):
        sid = f"cron_{job_id}_{idx:08d}"
        db.create_session(session_id=sid, source="cron")
        db.append_message(sid, role="user", content=f"run {idx} for {job_id}")
        db.append_message(sid, role="assistant", content="done")
        db.end_session(sid, "completed")
        db._conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = ?", (started_at, sid)
        )
        db._conn.commit()
        return sid

    def test_scopes_to_job_newest_first_and_enriched(self, db):
        base = 1_700_000_000.0
        # Target job: 5 runs, ascending started_at.
        for i in range(5):
            self._seed_run(db, "alpha", i, base + i * 60)
        # A different job that must not leak in.
        for i in range(3):
            self._seed_run(db, "beta", i, base + i * 60)

        runs = db.list_cron_job_runs("alpha", limit=20)

        assert len(runs) == 5
        assert all(r["id"].startswith("cron_alpha_") for r in runs)
        # Newest started_at first.
        sts = [r["started_at"] for r in runs]
        assert sts == sorted(sts, reverse=True)
        # Enriched like list_sessions_rich.
        assert runs[0]["preview"].startswith("run 4 for alpha")
        assert runs[0]["last_active"] >= runs[0]["started_at"]



    def test_limit_and_offset_paging(self, db):
        base = 1_700_000_000.0
        for i in range(10):
            self._seed_run(db, "alpha", i, base + i * 60)

        page1 = db.list_cron_job_runs("alpha", limit=4, offset=0)
        page2 = db.list_cron_job_runs("alpha", limit=4, offset=4)

        assert len(page1) == 4
        assert len(page2) == 4
        assert {r["id"] for r in page1}.isdisjoint({r["id"] for r in page2})
        # Combined window is still newest-first and contiguous.
        combined = [r["started_at"] for r in page1 + page2]
        assert combined == sorted(combined, reverse=True)



def test_gateway_session_peer_round_trip_and_recovery(db):
    db.create_session(
        "gw-session",
        "telegram",
        user_id="user-1",
        session_key="agent:main:telegram:dm:chat-1",
        chat_id="chat-1",
        chat_type="dm",
        thread_id=None,
    )
    db.append_message("gw-session", "user", "hello")

    row = db.get_session("gw-session")
    assert row["session_key"] == "agent:main:telegram:dm:chat-1"
    assert row["chat_id"] == "chat-1"
    assert row["chat_type"] == "dm"

    recovered = db.find_latest_gateway_session_for_peer(
        source="telegram",
        user_id="user-1",
        session_key="agent:main:telegram:dm:chat-1",
        chat_id="chat-1",
        chat_type="dm",
    )
    assert recovered["id"] == "gw-session"












def test_find_session_by_origin_matching_rules(db):
    db.create_session(
        "gw-o1", "telegram", user_id="u1",
        session_key="agent:main:telegram:group:c9:u1", chat_id="c9", chat_type="group",
    )
    db.create_session(
        "gw-o2", "telegram", user_id="u2",
        session_key="agent:main:telegram:group:c9:u2", chat_id="c9", chat_type="group",
    )

    # Exact user match wins.
    assert db.find_session_by_origin(
        platform="telegram", chat_id="c9", user_id="u2"
    ) == "gw-o2"
    # Unknown user among multiple distinct users -> None (no contamination).
    assert db.find_session_by_origin(
        platform="telegram", chat_id="c9", user_id="u3"
    ) is None
    # No user given + multiple distinct users -> None.
    assert db.find_session_by_origin(platform="telegram", chat_id="c9") is None
    # Ended sessions are ignored: only gw-o1 remains as a live candidate.
    # A single remaining candidate is returned even without an exact user
    # match — mirrors the original sessions.json scan semantics.
    db.end_session("gw-o2", "session_reset")
    assert db.find_session_by_origin(
        platform="telegram", chat_id="c9", user_id="u2"
    ) == "gw-o1"
    # Single remaining candidate resolves without user_id.
    assert db.find_session_by_origin(platform="telegram", chat_id="c9") == "gw-o1"
    # Thread filter.
    db.create_session(
        "gw-th", "discord", user_id="u9",
        session_key="agent:main:discord:thread:t7", chat_id="ch7",
        chat_type="thread", thread_id="t7",
    )
    assert db.find_session_by_origin(
        platform="discord", chat_id="ch7", thread_id="t7"
    ) == "gw-th"
    assert db.find_session_by_origin(
        platform="discord", chat_id="ch7", thread_id="other"
    ) is None












def test_refresh_compression_lock_requires_holder_and_preserves_reclaimability(db, monkeypatch):
    db.create_session("s1", "cli")

    monkeypatch.setattr(hermes_state.time, "time", lambda: 1000.0)
    assert db.try_acquire_compression_lock("s1", "holder-a", ttl_seconds=10.0) is True

    original_expires = db._conn.execute(
        "SELECT expires_at FROM compression_locks WHERE session_id = ?",
        ("s1",),
    ).fetchone()[0]

    monkeypatch.setattr(hermes_state.time, "time", lambda: 1005.0)
    assert db.refresh_compression_lock("s1", "holder-a", ttl_seconds=10.0) is True
    refreshed_expires = db._conn.execute(
        "SELECT expires_at FROM compression_locks WHERE session_id = ?",
        ("s1",),
    ).fetchone()[0]
    assert refreshed_expires > original_expires

    assert db.refresh_compression_lock("s1", "holder-b", ttl_seconds=10.0) is False

    monkeypatch.setattr(hermes_state.time, "time", lambda: 1016.0)
    assert db.try_acquire_compression_lock("s1", "holder-b", ttl_seconds=10.0) is True




def test_refresh_cannot_resurrect_a_lock_already_reclaimed(db, monkeypatch):
    """Once a competitor owns the row, the old holder's refresh must fail.

    The guard is the ``holder`` match, not the clock: a reclaim replaces
    ``holder``, so the previous owner's UPDATE matches nothing.
    """
    db.create_session("s1", "cli")

    monkeypatch.setattr(hermes_state.time, "time", lambda: 1000.0)
    assert db.try_acquire_compression_lock("s1", "holder-a", ttl_seconds=10.0) is True

    # holder-a's lease lapses and holder-b legitimately reclaims it.
    monkeypatch.setattr(hermes_state.time, "time", lambda: 1020.0)
    assert db.try_acquire_compression_lock("s1", "holder-b", ttl_seconds=10.0) is True

    # holder-a coming back late must NOT steal it back.
    assert db.refresh_compression_lock("s1", "holder-a", ttl_seconds=10.0) is False
    current = db._conn.execute(
        "SELECT holder FROM compression_locks WHERE session_id = ?",
        ("s1",),
    ).fetchone()[0]
    assert current == "holder-b"


# =========================================================================
# compact_rows — lightweight column projection (issue #47414)
# =========================================================================

class TestCompactRows:
    """list_sessions_rich and _get_session_rich_row with compact_rows=True
    must omit system_prompt but return all other metadata fields."""

    def _create(self, db, sid, *, system_prompt="big blob " * 500):
        db.create_session(session_id=sid, source="cli", model="m")
        db.update_system_prompt(sid, system_prompt)
        return sid

    def test_compact_rows_omits_system_prompt(self, db):
        self._create(db, "s1")
        rows = db.list_sessions_rich(compact_rows=True)
        assert len(rows) == 1
        assert "system_prompt" not in rows[0]




    def test_get_session_rich_row_compact_omits_system_prompt(self, db):
        self._create(db, "s1", system_prompt="should be gone")
        row = db._get_session_rich_row("s1", compact_rows=True)
        assert row is not None
        assert "system_prompt" not in row
        assert row["id"] == "s1"






# =========================================================================
# get_messages pagination (salvage follow-up for #60347)
# =========================================================================

class TestGetMessagesPagination:
    """get_messages(limit=, offset=) pages in insertion order; the default
    (limit=None) returns the full transcript unchanged."""

    def _seed(self, db, n=10):
        db.create_session(session_id="s1", source="cli")
        for i in range(n):
            db.append_message("s1", "user" if i % 2 == 0 else "assistant", f"msg-{i}")

    def test_default_returns_all_messages(self, db):
        self._seed(db)
        messages = db.get_messages("s1")
        assert [m["content"] for m in messages] == [f"msg-{i}" for i in range(10)]


    def test_window_query_bounded_work(self, db):
        """Perf contract: get_messages_around must seek by index, not scan
        the session's whole message history. Measured behaviorally via
        SQLite progress-handler steps (behavior contracts over snapshots,
        AGENTS.md — no EXPLAIN text). Calibrated on this seed (3000
        messages): indexed = ~12 handler calls, unindexed full-session
        scan = ~855. Threshold 300: >25x headroom above the indexed path,
        ~3x below the scan. Same pattern as the loader call-count pins in
        tests/tools/test_approval_config_readonly.py."""
        self._seed(db, n=3000)
        mid = db.get_messages("s1", limit=1, offset=1500)[0]["id"]
        steps = [0]

        def progress():
            steps[0] += 1
            return 0

        db._conn.set_progress_handler(progress, 100)
        try:
            db.get_messages_around("s1", mid, window=20)
        finally:
            db._conn.set_progress_handler(None, 0)
        assert steps[0] < 300, (
            f"get_messages_around executed {steps[0]}x100 VM steps — the "
            "session-history scan is back (idx_messages_session_id missing "
            "or unused)")


    def test_window_results_identical_with_and_without_index(self, db):
        """The index must not change results: identical windows at probe
        points across the session, with and without it."""
        self._seed(db, n=500)
        ids = [m["id"] for m in db.get_messages("s1")]
        probes = (ids[0], ids[len(ids) // 2], ids[-1])
        with_index = [db.get_messages_around("s1", mid, window=5)
                      for mid in probes]
        db._conn.execute("DROP INDEX idx_messages_session_id")
        without_index = [db.get_messages_around("s1", mid, window=5)
                         for mid in probes]
        assert with_index == without_index

    def test_limit_pages_in_insertion_order(self, db):
        self._seed(db)
        page1 = db.get_messages("s1", limit=4, offset=0)
        page2 = db.get_messages("s1", limit=4, offset=4)
        page3 = db.get_messages("s1", limit=4, offset=8)
        assert [m["content"] for m in page1] == ["msg-0", "msg-1", "msg-2", "msg-3"]
        assert [m["content"] for m in page2] == ["msg-4", "msg-5", "msg-6", "msg-7"]
        assert [m["content"] for m in page3] == ["msg-8", "msg-9"]





# =========================================================================
# Lone-surrogate persistence
# =========================================================================

class TestLoneSurrogatePersistence:
    """sqlite3 encodes bound str params as UTF-8 and raises UnicodeEncodeError
    on lone surrogates (U+D800..U+DFFF). Tool results scraped from the web can
    carry them, so a single such code point aborted the whole message write —
    and because run_agent swallows the failure with a warning, the session then
    silently stopped persisting for the rest of its life.
    """

    DIRTY = "scraped \ud835 price"

    def test_append_message_survives_lone_surrogate_content(self, db):
        db.create_session("s1", source="cli")
        db.append_message("s1", "assistant", "hello world")
        db.append_message("s1", "tool", self.DIRTY, tool_name="web_search")

        rows = db.get_messages("s1")
        assert len(rows) == 2
        # Surrogate replaced with U+FFFD; the surrounding text is intact.
        assert rows[1]["content"] == "scraped � price"




    # -- sibling raw-str bind sites (follow-up widening of the same bug class)




    def test_set_latest_user_api_content_survives_lone_surrogate(self, db):
        db.create_session("s1", source="cli")
        db.append_message("s1", "user", "turn text")
        assert db.set_latest_user_api_content("s1", "turn text", self.DIRTY) == 1



class TestDisplayMetadataPersistence:
    """Round-trip display_kind/display_metadata through every write path."""

    def test_append_message_round_trips_display_fields(self, db):
        db.create_session("s1", source="cli")
        meta = {"task_count": 2, "delegation_id": "del-1"}
        db.append_message(
            "s1", "user", "event text",
            display_kind="async_delegation_complete",
            display_metadata=meta,
        )
        conv = db.get_messages_as_conversation("s1")
        assert conv[0]["display_kind"] == "async_delegation_complete"
        assert conv[0]["display_metadata"] == meta

    def test_replace_messages_preserves_display_metadata(self, db):
        db.create_session("s1", source="cli")
        meta = {"task_count": 3, "delegation_id": "del-2", "duration_seconds": 12.5}
        db.append_message(
            "s1", "user", "event",
            display_kind="async_delegation_complete",
            display_metadata=meta,
        )
        # Reload via get_messages_as_conversation (which decodes display fields)
        # then replace_messages (which re-inserts via _insert_message_rows).
        conv = db.get_messages_as_conversation("s1")
        db.replace_messages("s1", conv)
        reloaded = db.get_messages_as_conversation("s1")
        assert reloaded[0]["display_kind"] == "async_delegation_complete"
        assert reloaded[0]["display_metadata"] == meta



class TestDisplayMetadataReadPaths:
    """Every message read path must hand back the decoded dict.

    Returning the raw column instead reaches the desktop as a string, where
    ``'task_count' in meta`` throws and fails the whole session resume.
    """

    META = {
        "delegation_id": "deleg_0d84d484",
        "task_count": 1,
        "completed_count": 1,
        "failed_count": 0,
        "duration_seconds": 193.55,
    }

    @staticmethod
    def _seed(db):
        db.create_session("s1", source="desktop")
        message_id = db.append_message(
            "s1", "user", "event",
            display_kind="async_delegation_complete",
            display_metadata=TestDisplayMetadataReadPaths.META,
        )
        return message_id, db.append_message("s1", "assistant", "anchor")

    @staticmethod
    def _read(db, reader, message_id, anchor_id):
        if reader == "get_messages":
            return db.get_messages("s1")[0]
        if reader == "get_messages_around":
            return db.get_messages_around("s1", message_id, window=0)["window"][0]
        if reader == "get_anchored_view":
            view = db.get_anchored_view("s1", anchor_id, window=0, bookend=1)
            return view["bookend_start"][0]
        return db.get_messages_as_conversation("s1")[0]

    READERS = ("get_messages", "get_messages_around", "get_anchored_view", "conversation")

    @pytest.mark.parametrize("reader", READERS)
    def test_every_reader_decodes_display_metadata(self, db, reader):
        message_id, anchor_id = self._seed(db)
        assert self._read(db, reader, message_id, anchor_id)["display_metadata"] == self.META


    @pytest.mark.parametrize("reader", READERS)
    @pytest.mark.parametrize("raw", ["", "{not-json", "[]", '"text"', "0"])
    def test_every_reader_drops_unusable_display_metadata(self, db, reader, raw):
        """Bad presentation metadata must not take the message down with it."""
        message_id, anchor_id = self._seed(db)

        def _corrupt(conn):
            conn.execute(
                "UPDATE messages SET display_metadata = ? WHERE id = ?",
                (raw, message_id),
            )

        db._execute_write(_corrupt)
        message = self._read(db, reader, message_id, anchor_id)
        assert message.get("display_metadata") is None
        assert message["content"] == "event"

    def test_export_import_round_trip_keeps_metadata_decodable(self, db, tmp_path):
        """The read leak used to write a permanently double-encoded row here.

        ``export_session`` reads through ``get_messages``, so an undecoded
        string went back through ``_insert_message_rows`` and got re-dumped.
        """
        self._seed(db)
        blob = db.export_session("s1")
        assert isinstance(blob["messages"][0]["display_metadata"], dict)

        target = SessionDB(db_path=tmp_path / "imported.db")
        try:
            target.import_sessions([json.loads(json.dumps(blob))])
            assert target.get_messages_as_conversation("s1")[0]["display_metadata"] == self.META
            assert target.get_messages("s1")[0]["display_metadata"] == self.META
        finally:
            target.close()




class TestGatewayRoutingPkHeal:
    """Legacy gateway_routing tables (session_key-only PK) get rebuilt on open.

    Early builds of the #59203 routing-index migration created gateway_routing
    with ``session_key TEXT PRIMARY KEY`` and no ``scope`` column. The column
    reconciler ADDs ``scope`` but cannot change the PK, so on those databases
    every routing save failed ("ON CONFLICT clause does not match any PRIMARY
    KEY or UNIQUE constraint" / "UNIQUE constraint failed:
    gateway_routing.session_key") and spammed warnings on each save.
    """

    LEGACY_SQL = """
        CREATE TABLE gateway_routing (
            session_key TEXT PRIMARY KEY,
            entry_json TEXT NOT NULL,
            updated_at REAL NOT NULL
        , "scope" TEXT DEFAULT '')
    """

    def _make_legacy_db(self, tmp_path, rows=()):
        db_path = tmp_path / "state.db"
        conn = sqlite3.connect(db_path)
        conn.execute(self.LEGACY_SQL)
        conn.executemany(
            "INSERT INTO gateway_routing (scope, session_key, entry_json, updated_at) "
            "VALUES (?, ?, ?, ?)",
            list(rows),
        )
        conn.commit()
        conn.close()
        return db_path

    def _pk_cols(self, db):
        rows = db._conn.execute('PRAGMA table_info("gateway_routing")').fetchall()
        cols = sorted(
            ((r["pk"], r["name"]) for r in rows if r["pk"]),
        )
        return [name for _, name in cols]

    def test_legacy_pk_rebuilt_to_composite(self, tmp_path):
        db_path = self._make_legacy_db(
            tmp_path, rows=[("/home/u/.hermes/sessions", "agent:main:telegram:dm:1", "{}", 1.0)]
        )
        db = SessionDB(db_path=db_path)
        try:
            assert self._pk_cols(db) == ["scope", "session_key"]
            # Existing rows survive the rebuild.
            entries = db.load_gateway_routing_entries(scope="/home/u/.hermes/sessions")
            assert entries == {"agent:main:telegram:dm:1": "{}"}
        finally:
            db.close()



    def test_current_shape_left_untouched(self, tmp_path, db):
        """A DB born with the composite PK is not rebuilt (idempotence)."""
        db.save_gateway_routing_entry("k1", "{}", scope="s")
        assert self._pk_cols(db) == ["scope", "session_key"]
        # Re-running the heal is a no-op.
        cur = db._conn.cursor()
        db._heal_gateway_routing_pk(cur)
        assert db.load_gateway_routing_entries(scope="s") == {"k1": "{}"}


class TestApplyDatabasePragmas:
    """Config-driven WAL-sizing pragma application (database: section)."""

    @staticmethod
    def _patch_cfg(monkeypatch, cfg):
        monkeypatch.setattr(
            "hermes_cli.config.load_config_readonly",
            lambda: cfg,
        )

    def test_honors_wal_autocheckpoint_from_config(self, tmp_path, monkeypatch):
        import sqlite3
        from hermes_state import apply_database_pragmas

        conn = sqlite3.connect(str(tmp_path / "pragmas.db"))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            self._patch_cfg(monkeypatch, {"database": {"wal_autocheckpoint": 250}})
            apply_database_pragmas(conn, db_label="test.db")
            assert conn.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == 250
        finally:
            conn.close()

    def test_honors_journal_size_limit_from_config(self, tmp_path, monkeypatch):
        import sqlite3
        from hermes_state import apply_database_pragmas

        conn = sqlite3.connect(str(tmp_path / "pragmas.db"))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            self._patch_cfg(
                monkeypatch, {"database": {"journal_size_limit": 10485760}}
            )
            apply_database_pragmas(conn, db_label="test.db")
            assert (
                conn.execute("PRAGMA journal_size_limit").fetchone()[0] == 10485760
            )
        finally:
            conn.close()

    def test_noop_when_database_section_missing(self, tmp_path, monkeypatch):
        import sqlite3
        from hermes_state import apply_database_pragmas

        conn = sqlite3.connect(str(tmp_path / "pragmas.db"))
        try:
            conn.execute("PRAGMA journal_mode=DELETE")
            self._patch_cfg(monkeypatch, {})
            apply_database_pragmas(conn, db_label="test.db")
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        finally:
            conn.close()

    def test_never_touches_journal_mode(self, tmp_path, monkeypatch):
        """journal_mode is owned by apply_wal_with_fallback — a database:
        journal_mode entry must NOT cause a second, unguarded mode switch."""
        import sqlite3
        from hermes_state import apply_database_pragmas

        conn = sqlite3.connect(str(tmp_path / "pragmas.db"))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            self._patch_cfg(monkeypatch, {"database": {"journal_mode": "delete"}})
            apply_database_pragmas(conn, db_label="test.db")
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        finally:
            conn.close()

    def test_ignores_non_integer_values(self, tmp_path, monkeypatch):
        import sqlite3
        from hermes_state import apply_database_pragmas

        conn = sqlite3.connect(str(tmp_path / "pragmas.db"))
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            before = conn.execute("PRAGMA wal_autocheckpoint").fetchone()[0]
            self._patch_cfg(
                monkeypatch, {"database": {"wal_autocheckpoint": "lots"}}
            )
            apply_database_pragmas(conn, db_label="test.db")
            assert conn.execute("PRAGMA wal_autocheckpoint").fetchone()[0] == before
        finally:
            conn.close()
