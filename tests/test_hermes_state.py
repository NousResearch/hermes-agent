"""Tests for hermes_state.py — SessionDB SQLite CRUD, FTS5 search, export."""

import sqlite3
import time
import json
import threading
from pathlib import Path
from unittest import mock

import pytest

import hermes_state
import hermes_state_wal
import hermes_state_common
from agent.session_activity import ActivityProvenance, build_activity_snapshot
from hermes_state import SessionDB
from hermes_state_common import FTS_SQL, FTS_STORAGE_VERSION, SCHEMA_SQL, SCHEMA_VERSION


def _activity_snapshot(db, session_id):
    """Durable activity snapshot for *session_id* (what gateway/delegate readers build from the row)."""
    row = db.get_session(session_id)
    return build_activity_snapshot(
        last_activity_at=row.get("last_activity_at"),
        last_activity_description=row.get("last_activity_description"),
        last_activity_provenance=row.get("last_activity_provenance"),
    )


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


@pytest.fixture(autouse=True)
def _no_fts_rebuild_throttle(monkeypatch):
    """Zero the FTS-rebuild inter-chunk throttle for every test in this file.

    ``optimize_fts_storage`` sleeps ``max(_FTS_REBUILD_MIN_PAUSE,
    chunk_cost * _FTS_REBUILD_DUTY_FACTOR)`` between chunks so a LIVE
    gateway/CLI sharing the DB isn't starved of the write lock. Tests run
    against a private tmp-path DB with no concurrent process — the sleep
    protects nobody and was pure dead time (measured: 4.1s of a 4.6s
    migration test was time.sleep; ~20s across the file, whose total was
    ~52s). The duty-cycle POLICY (sleep >= 4x chunk cost) stays covered by
    the production constants themselves; no test asserts on wall-clock
    pacing.
    """
    monkeypatch.setattr(SessionDB, "_FTS_REBUILD_MIN_PAUSE", 0.0)
    monkeypatch.setattr(SessionDB, "_FTS_REBUILD_DUTY_FACTOR", 0.0)


# =========================================================================
# Connection lifecycle
# =========================================================================


class TestConnectionLifecycle:
    def test_failed_writable_open_does_not_leak_tracked_connection(
        self, tmp_path, monkeypatch
    ):
        """A failed schema init must close the connection opened before it."""
        from hermes_cli.sqlite_safe_read import has_live_connection

        db_path = tmp_path / "state.db"
        opened = []
        real_connect = hermes_state._connect_tracked_db

        def capture_connect(*args, **kwargs):
            conn = real_connect(*args, **kwargs)
            opened.append(conn)
            return conn

        monkeypatch.setattr(hermes_state, "_connect_tracked_db", capture_connect)
        monkeypatch.setattr(
            SessionDB,
            "_init_schema",
            mock.Mock(side_effect=RuntimeError("schema init failed")),
        )

        try:
            with pytest.raises(RuntimeError, match="schema init failed"):
                SessionDB(db_path=db_path)
            assert has_live_connection(db_path) is False
        finally:
            for conn in opened:
                try:
                    conn.close()
                except Exception:
                    pass

    def test_failed_wal_read_open_does_not_leak_tracked_connection(
        self, tmp_path, monkeypatch
    ):
        """A post-open read setup failure must close its unregistered conn."""
        from hermes_cli import sqlite_safe_read

        db_path = tmp_path / "state.db"
        db = SessionDB(db_path=db_path)
        opened = []
        real_connect = hermes_state._connect_tracked_db
        real_pragmas = hermes_state.apply_database_pragmas

        def capture_connect(*args, **kwargs):
            conn = real_connect(*args, **kwargs)
            opened.append(conn)
            return conn

        def fail_pragmas(*args, **kwargs):
            raise RuntimeError("read setup failed")

        monkeypatch.setattr(hermes_state, "_connect_tracked_db", capture_connect)
        monkeypatch.setattr(hermes_state, "apply_database_pragmas", fail_pragmas)
        before = dict(sqlite_safe_read._live_connections)
        db._wal_active = True

        try:
            with pytest.raises(RuntimeError, match="read setup failed"):
                db._get_read_conn()
            assert sqlite_safe_read._live_connections == before
        finally:
            monkeypatch.setattr(
                hermes_state, "apply_database_pragmas", real_pragmas
            )
            for conn in opened:
                try:
                    conn.close()
                except Exception:
                    pass
            db.close()

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

    def test_writable_close_uses_passive_checkpoint(self, tmp_path):
        db_path = tmp_path / "state.db"
        writable = SessionDB(db_path=db_path)
        executed = []
        writable._conn.set_trace_callback(executed.append)

        writable.close()

        # close() must NOT TRUNCATE: transient per-cron-run connections firing
        # full WAL resets race the gateway's live writer and corrupt B-tree
        # pages (issue #45383). It uses PASSIVE instead.
        assert not any(
            "pragma wal_checkpoint(truncate)" == " ".join(sql.lower().split())
            for sql in executed
        )
        assert any(
            "pragma wal_checkpoint(passive)" == " ".join(sql.lower().split())
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

    def test_read_only_open_retries_transient_wal_ioerr(self, tmp_path, monkeypatch):
        """A transient SQLITE_IOERR on a read-only open must retry, not raise.

        A ``mode=ro`` connection cannot perform WAL recovery (recovery would
        need to write the -shm index, which read-only mode refuses), so a
        concurrent checkpoint / WAL reset / frame-flush on the writer side can
        surface "disk I/O error" to a reader on a perfectly healthy database
        (#100436). The transition window is millisecond-scale; a bounded retry
        must let the open succeed instead of 500-ing the /api/sessions poll
        and every other read-only opener.
        """
        import sqlite3

        from hermes_cli.sqlite_safe_read import has_live_connection

        db_path = tmp_path / "state.db"
        writable = SessionDB(db_path=db_path)
        writable.create_session("wal-race", source="cli")
        writable.close()

        real_connect = hermes_state._connect_tracked_db
        attempts = []

        def flaky_connect(*args, **kwargs):
            attempts.append(kwargs.get("uri"))
            if len(attempts) == 1:
                # First open lands inside the writer's WAL transition window.
                raise sqlite3.OperationalError("disk I/O error")
            return real_connect(*args, **kwargs)

        monkeypatch.setattr(hermes_state, "_connect_tracked_db", flaky_connect)
        # Keep the test fast: one backoff tick is enough; the retry budget
        # itself is exercised by the attempt count below.
        monkeypatch.setattr(hermes_state, "_READ_ONLY_IOERR_RETRY_BACKOFF_S", 0.0)

        read_only = SessionDB(db_path=db_path, read_only=True)
        try:
            assert read_only._fts_enabled is True
            matches = read_only.search_messages("wal-race")
        finally:
            read_only.close()

        assert len(attempts) >= 2, "the transient IOERR must be retried"
        assert has_live_connection(db_path) is False  # no leaked connections

    def test_read_only_open_exhausts_retry_budget_for_persistent_ioerr(
        self, tmp_path, monkeypatch
    ):
        """A persistent SQLITE_IOERR must exhaust the budget and raise.

        The retry exists to ride out a millisecond WAL transition — a
        storage layer that keeps failing after the full budget is genuinely
        broken and must surface the error (and not loop forever).
        """
        import sqlite3

        db_path = tmp_path / "state.db"
        writable = SessionDB(db_path=db_path)
        writable.create_session("broken-disk", source="cli")
        writable.close()

        attempts = []

        def bad_connect(*args, **kwargs):
            attempts.append(1)
            raise sqlite3.OperationalError("disk I/O error")

        monkeypatch.setattr(hermes_state, "_connect_tracked_db", bad_connect)
        monkeypatch.setattr(hermes_state, "_READ_ONLY_IOERR_RETRY_BACKOFF_S", 0.0)
        budget = hermes_state._READ_ONLY_IOERR_RETRY_ATTEMPTS

        with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
            SessionDB(db_path=db_path, read_only=True)

        # budget + 1 = the initial attempt plus `budget` retries.
        assert len(attempts) == budget + 1


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


    def test_branch_resume_does_not_include_parent_messages_added_after_fork(self, db):
        """A branch owns its copied transcript, not the parent's later turns."""
        db.create_session("parent", source="tui")
        db.append_message("parent", role="user", content="before branch")
        db.append_message("parent", role="assistant", content="initial answer")

        db.create_session(
            "branch",
            source="tui",
            parent_session_id="parent",
            model_config={"_branched_from": "parent"},
        )
        db.append_message("branch", role="user", content="before branch")
        db.append_message("branch", role="assistant", content="initial answer")

        # The original conversation can be resumed after the fork. Those new
        # rows must not leak into the already-created branch's transcript.
        db.append_message("parent", role="user", content="after branch")
        db.append_message("parent", role="assistant", content="later answer")

        _, display_history = db.get_resume_conversations("branch")

        assert [message["content"] for message in display_history] == [
            "before branch",
            "initial answer",
        ]
        assert [
            message["content"]
            for message in db.get_messages_as_conversation("branch", include_ancestors=True)
        ] == ["before branch", "initial answer"]
        assert db.get_ancestor_display_prefix("branch") == []





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
            assert "content" not in results[0]
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
        # Results never carry full content; snippet + metadata only.
        assert all("content" not in r for r in results)






    def test_search_returns_context(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="Tell me about Kubernetes")
        db.append_message("s1", role="assistant", content="Kubernetes is an orchestrator.")

        results = db.search_messages("Kubernetes")
        assert len(results) == 2
        assert "context" in results[0]
        assert isinstance(results[0]["context"], list)
        assert len(results[0]["context"]) > 0

    def test_search_fields_project_results_without_changing_default(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="Tell me about Kubernetes")
        db.append_message("s1", role="assistant", content="Kubernetes is an orchestrator.")

        projected = db.search_messages(
            "Kubernetes", fields=("session_id", "role", "snippet")
        )
        default = db.search_messages("Kubernetes")

        assert len(projected) == len(default) == 2
        assert all(set(row) == {"session_id", "role", "snippet"} for row in projected)
        assert [
            (row["session_id"], row["role"], row["snippet"])
            for row in projected
        ] == [
            (row["session_id"], row["role"], row["snippet"])
            for row in default
        ]
        assert all("context" in row and row["context"] for row in default)

    def test_search_projection_skips_context_enrichment_queries(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="before")
        db.append_message("s1", role="assistant", content="projectionneedle")
        db.append_message("s1", role="user", content="after")

        statements = []
        read_conn = db._get_read_conn() or db._conn
        traced_connections = [db._conn]
        if read_conn is not db._conn:
            traced_connections.append(read_conn)
        for conn in traced_connections:
            conn.set_trace_callback(statements.append)

        def context_query_count():
            normalized = (" ".join(sql.upper().split()) for sql in statements)
            return sum("WITH TARGET AS (" in sql for sql in normalized)

        try:
            projected = db.search_messages(
                "projectionneedle", fields=("session_id", "snippet")
            )
            assert len(projected) == 1
            assert context_query_count() == 0

            full = db.search_messages(
                "projectionneedle", fields=("session_id", "context")
            )
            assert len(full) == 1
            assert full[0]["context"]
            assert context_query_count() == 1

            default = db.search_messages("projectionneedle")
            assert len(default) == 1
            assert default[0]["context"]
            assert context_query_count() == 2
        finally:
            for conn in traced_connections:
                conn.set_trace_callback(None)

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






    def test_session_count_ge_empty(self, db):
        """session_count_ge should return False for 0 sessions."""
        assert db.session_count_ge(1) is False
        assert db.session_count_ge(2) is False

    def test_session_count_ge_at_threshold(self, db):
        """session_count_ge should True when count >= n."""
        db.create_session("s1", "cli")
        assert db.session_count_ge(1) is True
        assert db.session_count_ge(2) is False

        db.create_session("s2", "telegram")
        assert db.session_count_ge(1) is True
        assert db.session_count_ge(2) is True
        assert db.session_count_ge(3) is False

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




    def test_export_nonexistent(self, db):
        assert db.export_session("nope") is None









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
        assert db.count_open_prune_matches(older_than_days=90) == 1

    def test_open_prune_match_count_applies_other_filters(self, db):
        db.create_session(session_id="matching-open", source="cron")
        db.create_session(session_id="other-source", source="cli")
        db.create_session(session_id="ended", source="cron")
        db.end_session("ended", "completed")
        old = time.time() - 200 * 86400
        db._conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id IN (?, ?, ?)",
            (old, "matching-open", "other-source", "ended"),
        )
        db._conn.commit()

        assert db.count_open_prune_matches(
            older_than_days=90, source="cron", archived=False
        ) == 1
        assert {row["id"] for row in db.list_prune_candidates(
            older_than_days=90, source="cron", archived=False
        )} == {"ended"}





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






    def test_title_like_underscore_is_literal_not_a_wildcard(self, db):
        """``_`` is a single-character wildcard in SQL LIKE, so an unescaped
        filter deletes sessions the operator never selected. The filters are
        documented (and shown in the CLI confirmation) as substring matches.
        """
        self._mk(db, "target", title="user_auth refactor")
        self._mk(db, "bystander1", title="user-auth review")
        self._mk(db, "bystander2", title="userXauth notes")
        self._mk(db, "bystander3", title="user auth meeting")

        rows = db.list_prune_candidates(title_like="user_auth")
        assert {r["id"] for r in rows} == {"target"}

        pruned = db.prune_sessions(older_than_days=None, title_like="user_auth")
        assert pruned == 1
        for survivor in ("bystander1", "bystander2", "bystander3"):
            assert db.get_session(survivor) is not None

    def test_percent_in_filter_does_not_select_everything(self, db):
        """``%`` matches any run of characters — a bare one would delete the
        whole table."""
        self._mk(db, "a", title="alpha")
        self._mk(db, "b", title="beta")
        self._mk(db, "pct", title="100% coverage run")

        # Only the title that really contains a percent sign matches.
        assert {r["id"] for r in db.list_prune_candidates(title_like="%")} == {"pct"}
        assert {r["id"] for r in db.list_prune_candidates(title_like="100%")} == {"pct"}

    def test_branch_like_underscore_is_literal(self, db):
        """Branch names carry underscores routinely."""
        self._mk_rich(db, "want", git_branch="fix/session_prune")
        self._mk_rich(db, "other", git_branch="fix/session-prune")

        rows = db.list_prune_candidates(branch_like="session_prune")
        assert {r["id"] for r in rows} == {"want"}

    def test_model_like_underscore_is_literal(self, db):
        self._mk_rich(db, "want", model="vendor/model_mini")
        self._mk_rich(db, "other", model="vendor/model-mini")

        rows = db.list_prune_candidates(model_like="model_mini")
        assert {r["id"] for r in rows} == {"want"}

    def test_plain_substring_filters_still_match(self, db):
        """Guard against over-escaping: ordinary filters keep working, and a
        literal backslash in the needle is matched as itself."""
        self._mk(db, "smoke", title="Codex Smoke Test")
        self._mk_rich(db, "winpath", title=r"build C:\tmp artifacts")

        assert {r["id"] for r in db.list_prune_candidates(title_like="smoke")} == {"smoke"}
        assert {r["id"] for r in db.list_prune_candidates(title_like=r"c:\tmp")} == {"winpath"}

    def test_cwd_prefix_underscore_is_literal_not_a_wildcard(self, db):
        """``_`` is a LIKE wildcard but an ordinary character in a path, so an
        unescaped prefix also matched a same-length sibling directory — and
        prune_sessions deletes what it matches."""
        self._mk(db, "target", cwd="/home/me/my_project/src")
        self._mk(db, "sibling", cwd="/home/me/myXproject/src")

        rows = db.list_prune_candidates(cwd_prefix="/home/me/my_project")
        assert {r["id"] for r in rows} == {"target"}

        pruned = db.prune_sessions(older_than_days=None, cwd_prefix="/home/me/my_project")
        assert pruned == 1
        assert db.get_session("sibling") is not None

    def test_cwd_prefix_percent_does_not_select_everything(self, db):
        self._mk(db, "a", cwd="/home/me/one")
        self._mk(db, "b", cwd="/home/me/two")

        assert db.list_prune_candidates(cwd_prefix="/home/me/%") == []

    def test_cwd_prefix_still_matches_the_directory_and_its_children(self, db):
        """Control: the prefix must keep matching itself and anything under it."""
        self._mk(db, "root", cwd="/home/me/proj")
        self._mk(db, "child", cwd="/home/me/proj/src")
        self._mk(db, "outside", cwd="/home/me/other")

        rows = db.list_prune_candidates(cwd_prefix="/home/me/proj")
        assert {r["id"] for r in rows} == {"root", "child"}

    def test_cwd_prefix_windows_separator_arm(self, db):
        """The backslash child arm (``{esc}\\\\%`` in the pattern) must keep
        matching Windows children while ``_`` stays literal — a guard against
        'simplifying' the quadruple backslash."""
        self._mk(db, "win_root", cwd=r"C:\Users\me\my_project")
        self._mk(db, "win_child", cwd=r"C:\Users\me\my_project\src")
        self._mk(db, "win_sibling", cwd=r"C:\Users\me\myXproject\src")

        rows = db.list_prune_candidates(cwd_prefix=r"C:\Users\me\my_project")
        assert {r["id"] for r in rows} == {"win_root", "win_child"}

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
        from hermes_state_wal import is_sqlite_wal_reset_vulnerable

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
        assert db.get_meta("telegram_dm_topic_schema_version") == "3"
        db.close()







    def test_schema_sql_is_source_of_truth(self, db):
        """Every column in SCHEMA_SQL exists in the live database.

        This is the architectural invariant: SCHEMA_SQL declares the
        desired schema, _reconcile_columns ensures it matches reality.
        """
        from hermes_state_common import SCHEMA_SQL

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


class TestReconcileColumnsErrorHandling:
    """_reconcile_columns must not bury migration failures (#79531/#80037).

    A locked ALTER used to be swallowed at DEBUG: startup "succeeded" with a
    half-reconciled schema and every session-list read then 500ed with
    "no such column" until an unrelated writable open. The contract now:
    duplicate-column races stay quiet, lock/busy propagates (so the open-time
    lock patience retries the whole init), everything else warns.
    """

    class _FailingAlterCursor:
        """Pass through to a real cursor, failing ALTER TABLE with ``exc``."""

        def __init__(self, real_cursor, exc):
            self._real = real_cursor
            self._exc = exc

        def execute(self, sql, *args, **kwargs):
            if sql.lstrip().upper().startswith("ALTER TABLE"):
                raise self._exc
            return self._real.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._real, name)

    def _db_missing_column(self, tmp_path):
        """A store whose sessions table lacks last_read_at."""
        db_path = tmp_path / "state.db"
        seed = SessionDB(db_path=db_path)
        seed.close()
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("ALTER TABLE sessions DROP COLUMN last_read_at")
            conn.commit()
        finally:
            conn.close()
        return db_path

    def test_locked_alter_propagates(self, tmp_path):
        """database-is-locked must escape _reconcile_columns, not vanish.

        Propagation is what lets _connect_and_init_with_lock_patience retry
        the whole init with jittered backoff instead of serving a store
        that is silently behind SCHEMA_SQL.
        """
        db_path = self._db_missing_column(tmp_path)
        conn = sqlite3.connect(str(db_path))
        try:
            stale = SessionDB.__new__(SessionDB)
            stale._conn = conn
            cursor = self._FailingAlterCursor(
                conn.cursor(),
                sqlite3.OperationalError("database is locked"),
            )
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                stale._reconcile_columns(cursor)
        finally:
            conn.close()

    def test_duplicate_column_race_stays_quiet(self, tmp_path, caplog):
        """A duplicate-column race is expected and must not warn or raise."""
        import logging

        db_path = self._db_missing_column(tmp_path)
        conn = sqlite3.connect(str(db_path))
        try:
            stale = SessionDB.__new__(SessionDB)
            stale._conn = conn
            cursor = self._FailingAlterCursor(
                conn.cursor(),
                sqlite3.OperationalError(
                    "duplicate column name: last_read_at"
                ),
            )
            with caplog.at_level(logging.WARNING, logger="hermes_state"):
                stale._reconcile_columns(cursor)
        finally:
            conn.close()
        assert not [
            r for r in caplog.records if "reconcile" in r.getMessage()
        ]

    def test_other_alter_failures_warn(self, tmp_path, caplog):
        """Schema mistakes (e.g. un-ADDable NOT NULL) log at WARNING."""
        import logging

        db_path = self._db_missing_column(tmp_path)
        conn = sqlite3.connect(str(db_path))
        try:
            stale = SessionDB.__new__(SessionDB)
            stale._conn = conn
            cursor = self._FailingAlterCursor(
                conn.cursor(),
                sqlite3.OperationalError(
                    "Cannot add a NOT NULL column with default value NULL"
                ),
            )
            with caplog.at_level(logging.WARNING, logger="hermes_state"):
                stale._reconcile_columns(cursor)
        finally:
            conn.close()
        warnings = [
            r
            for r in caplog.records
            if r.levelno >= logging.WARNING
            and "reconcile" in r.getMessage()
        ]
        assert warnings, "un-ADDable column failure must be logged at WARNING+"

    def test_locked_alter_is_retried_by_open_lock_patience(self, tmp_path, monkeypatch):
        """End-to-end: a transiently locked ALTER heals on open retry.

        The lock-patience wrapper retries on OperationalError raised out of
        _connect_and_init; before this fix _reconcile_columns caught the
        error internally so the retry never saw it and the store stayed
        stale forever.
        """
        db_path = self._db_missing_column(tmp_path)

        original = SessionDB._reconcile_columns
        calls = {"n": 0}

        def flaky_reconcile(self, cursor):
            calls["n"] += 1
            if calls["n"] == 1:
                raise sqlite3.OperationalError("database is locked")
            return original(self, cursor)

        monkeypatch.setattr(SessionDB, "_reconcile_columns", flaky_reconcile)
        # Keep the retry fast — patience budget is 20s by default.
        monkeypatch.setattr(SessionDB, "_WRITE_RETRY_SLOW_MIN_S", 0.001)
        monkeypatch.setattr(SessionDB, "_WRITE_RETRY_SLOW_MAX_S", 0.005)

        healed = SessionDB(db_path=db_path)
        try:
            cols = {
                r[1]
                for r in healed._conn.execute(
                    'PRAGMA table_info("sessions")'
                ).fetchall()
            }
        finally:
            healed.close()
        assert calls["n"] >= 2, "lock patience must retry the init"
        assert "last_read_at" in cols


class TestFtsRebuildLoopWithoutTrigram:
    """A trigram-less SQLite build must not re-index the store on every open.

    The three ``messages_fts_trigram_*`` triggers are declared only by the
    trigram DDL, whose ``CREATE VIRTUAL TABLE ... tokenize='trigram'`` needs a
    tokenizer SQLite only gained in 3.34 — Ubuntu 20.04 (3.31), RHEL/CentOS 8
    (3.26) and Amazon Linux 2 all ship older. ``_ensure_fts_schema``
    soft-fails that DDL there by design, so those three triggers can never
    exist, and startup's "are all six canonical triggers present?" check was
    therefore permanently unsatisfiable: the full FTS repair ran on every
    single ``SessionDB`` open, holding the write lock, and never converged.

    The v23 repair also clears the deferred-rebuild resume markers, so an
    interrupted ``hermes sessions optimize-storage`` silently lost its place
    every time the store was reopened.
    """

    @staticmethod
    def _trace(monkeypatch, statements, *, trigram):
        """Record every statement SessionDB executes during an open.

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

        monkeypatch.setattr("hermes_state.sqlite3.connect", connect)

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
        db = SessionDB(db_path=db_path)
        try:
            db.create_session(session_id="s1", source="cli")
            for i in range(5):
                db.append_message("s1", role="user", content=f"payload {i} zebra")
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
        from hermes_state_common import (
            FTS_SQL,
            FTS_TRIGRAM_SQL,
            LEGACY_FTS_SQL,
            LEGACY_FTS_TRIGRAM_SQL,
            _FTS_TRIGGERS,
        )
        from hermes_state_schema import _FTS_BASE_TRIGGERS, _FTS_TRIGRAM_TRIGGERS

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
            db = SessionDB(db_path=db_path)
            try:
                assert db._trigram_available is False
                assert self._rebuilds(statements) == []
                # The narrowed gate must not have cost us a working index.
                assert len(db.search_messages("zebra")) == 5
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
            db = SessionDB(db_path=db_path)
            try:
                assert db._db_has_legacy_inline_fts(db._conn.cursor()) is True
                assert db._trigram_available is False
                assert self._legacy_wipes(statements) == []
                assert len(db.search_messages("zebra")) == 5
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

        db = SessionDB(db_path=db_path)
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

        db = SessionDB(db_path=db_path)
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

        db = SessionDB(db_path=db_path)
        try:
            db._conn.execute("DROP TRIGGER messages_fts_insert")
            db._conn.commit()
        finally:
            db.close()

        statements.clear()
        db = SessionDB(db_path=db_path)
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
        db = SessionDB(db_path=db_path)
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

        db = SessionDB(db_path=db_path)
        trigram_available = db._trigram_available
        try:
            if not trigram_available:
                pytest.skip("this SQLite build has no trigram tokenizer")
            db._conn.execute("DROP TRIGGER messages_fts_trigram_insert")
            db._conn.commit()
        finally:
            db.close()

        statements.clear()
        db = SessionDB(db_path=db_path)
        try:
            assert db._trigram_available is True
            assert len(self._rebuilds(statements)) > 0
        finally:
            db.close()


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





    def test_last_active_prefers_session_activity_heartbeat(self, db):
        """Mid-turn agent heartbeats must advance last_active without new messages (#72016)."""
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "hello")
        with db._lock:
            db._conn.execute(
                "UPDATE messages SET timestamp=? WHERE session_id=? AND role=?",
                (1_700_000_000.0, "s1", "user"),
            )
            db._conn.commit()

        before = db.list_sessions_rich()[0]["last_active"]
        heartbeat = 1_700_000_500.0
        db.touch_session_activity(
            "s1",
            heartbeat,
            description="starting API call #1",
            provenance=ActivityProvenance.UNKNOWN,
        )
        after = db.list_sessions_rich()[0]["last_active"]
        assert after == heartbeat
        assert after > before

        row = db.get_session("s1")
        assert row["last_activity_at"] == heartbeat
        assert row["last_activity_description"] == "starting API call #1"
        assert row["last_activity_provenance"] == "unknown"

        activity = _activity_snapshot(db, "s1")
        assert activity["last_activity_at"] == heartbeat
        assert activity["last_activity_description"] == "starting API call #1"
        assert "phase" not in activity

        # Never move last_activity_at backwards.
        db.touch_session_activity("s1", heartbeat - 100, description="ignored")
        assert db.get_session("s1")["last_activity_at"] == heartbeat
        assert db.get_session("s1")["last_activity_description"] == "starting API call #1"

    def test_clear_session_activity_labels_keeps_timestamp(self, db):
        """Turn-end label clear must wipe desc/provenance without moving ts."""
        db.create_session("s1", "cli")
        heartbeat = 1_700_000_500.0
        db.touch_session_activity(
            "s1",
            heartbeat,
            description="compressing context",
            provenance=ActivityProvenance.AGENT_COMPRESSION,
        )
        row = db.get_session("s1")
        assert row["last_activity_at"] == heartbeat
        assert row["last_activity_description"] == "compressing context"
        assert row["last_activity_provenance"] == "agent.compression"

        db.clear_session_activity_labels("s1")
        row = db.get_session("s1")
        assert row["last_activity_at"] == heartbeat
        assert row["last_activity_description"] == ""
        assert row["last_activity_provenance"] == "unknown"
        activity = _activity_snapshot(db, "s1")
        assert activity["last_activity_at"] == heartbeat
        assert activity["last_activity_description"] == ""
        assert activity["last_activity_provenance"] == "unknown"

    def test_last_active_uses_newer_message_over_stale_heartbeat(self, db):
        """Rate-limited heartbeats can lag message writes; last_active must take max."""
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "hello")
        with db._lock:
            db._conn.execute(
                "UPDATE messages SET timestamp=? WHERE session_id=?",
                (1_700_000_800.0, "s1"),
            )
            db._conn.commit()
        db.touch_session_activity("s1", 1_700_000_500.0, description="api")  # older than message
        assert db.list_sessions_rich()[0]["last_active"] == 1_700_000_800.0

    def test_list_gateway_sessions_last_active_uses_activity_heartbeat(self, db):
        db.create_session(
            "gw-1",
            "telegram",
            session_key="agent:main:telegram:dm:c1",
            chat_id="c1",
            chat_type="dm",
        )
        db.append_message("gw-1", "user", "ping")
        with db._lock:
            db._conn.execute(
                "UPDATE messages SET timestamp=? WHERE session_id=?",
                (1_700_000_000.0, "gw-1"),
            )
            db._conn.commit()

        heartbeat = 1_700_000_900.0
        db.touch_session_activity(
            "gw-1",
            heartbeat,
            description="compressing context",
        )
        rows = db.list_gateway_sessions(active_only=True)
        assert len(rows) == 1
        assert rows[0]["last_active"] == heartbeat
        activity = _activity_snapshot(db, "gw-1")
        assert activity["last_activity_description"] == "compressing context"

    def test_order_by_last_active_surfaces_recently_touched_older_session_first(self, db):
        t0 = 1709500000.0
        db.create_session("old", "cli")
        db.create_session("new", "cli")







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

    @pytest.mark.parametrize(
        "end_reason",
        [
            "session_reset",
            "session_switch",
            "idle",
            "daily",
            "suspended",
            "resume_pending_expired",
        ],
    )
    def test_rich_list_keeps_legacy_reset_children_visible(self, db, end_reason):
        from hermes_state_common import _ephemeral_child_sql

        lane_key = "agent:main:telegram:dm:lane"
        parent_id = f"parent_{end_reason}"
        child_id = f"child_{end_reason}"
        db.create_session(parent_id, "telegram", session_key=lane_key)
        db.end_session(parent_id, end_reason)
        # No _reset_from marker: this is the on-disk shape written before the
        # marker existed. The unchanged routing key proves a reset boundary.
        db.create_session(
            child_id,
            "telegram",
            session_key=lane_key,
            parent_session_id=parent_id,
        )

        listed = [row["id"] for row in db.list_sessions_rich(source="telegram")]
        assert {parent_id, child_id}.issubset(listed)
        assert db.session_count(source="telegram", exclude_children=True) == 2
        assert db.session_count_by_source(exclude_children=True)["telegram"] == 2
        ephemeral = db._conn.execute(
            f"SELECT s.id FROM sessions s WHERE {_ephemeral_child_sql('s')}"
        ).fetchall()
        assert child_id not in {row["id"] for row in ephemeral}

    def test_reset_parent_does_not_surface_unrelated_child(self, db):
        db.create_session(
            "reset_parent",
            "telegram",
            session_key="agent:main:telegram:dm:lane",
        )
        db.end_session("reset_parent", "session_reset")
        db.create_session(
            "unrelated_child",
            "tool",
            session_key="delegate:other",
            parent_session_id="reset_parent",
        )

        listed = [row["id"] for row in db.list_sessions_rich()]
        assert "unrelated_child" not in listed
        assert db.session_count(exclude_children=True) == 1

    def test_resume_walker_does_not_cross_reset_boundary(self, db):
        """resolve_resume_session_id must not redirect a reset parent's resume
        into the post-reset conversation — that would restore the exact
        context the user reset away. Covers both the durable marker and the
        legacy markerless shape."""
        lane_key = "agent:main:telegram:dm:lane"
        # Marker shape (rows written by current gateway code).
        db.create_session("walk_parent", "telegram", session_key=lane_key)
        db.append_message("walk_parent", "user", "before reset")
        db.end_session("walk_parent", "session_reset")
        db.create_session(
            "walk_child",
            "telegram",
            session_key=lane_key,
            parent_session_id="walk_parent",
            model_config={"_reset_from": "walk_parent"},
        )
        db.append_message("walk_child", "user", "after reset")
        assert db.resolve_resume_session_id("walk_parent") == "walk_parent"

        # Legacy markerless shape (pre-marker on-disk rows).
        lane2 = "agent:main:telegram:dm:lane2"
        db.create_session("legacy_parent", "telegram", session_key=lane2)
        db.append_message("legacy_parent", "user", "before reset")
        db.end_session("legacy_parent", "session_reset")
        db.create_session(
            "legacy_child",
            "telegram",
            session_key=lane2,
            parent_session_id="legacy_parent",
        )
        db.append_message("legacy_child", "user", "after reset")
        assert db.resolve_resume_session_id("legacy_parent") == "legacy_parent"

    # Compression-tip following (the walker's original purpose) is pinned by
    # tests/hermes_state/test_resolve_resume_session_id.py
    # ::test_follows_compression_tip_when_parent_retains_messages.

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

    def test_list_serves_full_lineage_ids_for_projected_rows(self, db):
        """The projected tip row must carry every chain id. Root and tip
        alone are not enough client-side: a persisted tile or route can hold
        a MIDDLE segment's id (it was the tip when opened), and without the
        intermediates that surface cannot prove it names this conversation —
        which is how one chat ends up open twice after a compaction."""
        import time as _time
        self._build_compression_chain(db, _time.time() - 3600)
        db.create_session("solo", "cli")
        db.append_message("solo", "user", "standalone")
        db._conn.commit()

        sessions = db.list_sessions_rich(source="cli", limit=20)
        tip_row = next(s for s in sessions if s["id"] == "tip1")
        assert tip_row["_lineage_ids"] == ["root1", "mid1", "tip1"]
        solo_row = next(s for s in sessions if s["id"] == "solo")
        assert solo_row.get("_lineage_ids") is None



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

    def test_list_projects_multiple_independent_chains_in_one_call(self, db):
        """Two unrelated compression chains in the same page must each
        resolve to their own tip, not get cross-mixed by the batched tip-row
        fetch (regression test for the single-query batch in
        _get_session_rich_rows_batch — a wrong id->row mapping there would
        silently swap one chain's data onto the other)."""
        import time as _time

        t0 = _time.time() - 7200
        self._build_compression_chain(db, t0)

        # Second, independent chain — same shape, different ids/content.
        db.create_session("root2", "cli")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0 + 100, "root2"))
        db.append_message("root2", "user", "second conversation start")
        db._conn.execute(
            "UPDATE sessions SET ended_at=?, end_reason=? WHERE id=?",
            (t0 + 200, "compression", "root2"),
        )
        db.create_session("tip2", "cli", parent_session_id="root2")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0 + 201, "tip2"))
        db.append_message("tip2", "user", "second conversation continuation")
        db.update_session_cwd("tip2", "/tmp/workspaces/second")
        db._conn.commit()

        sessions = db.list_sessions_rich(source="cli", limit=20)
        ids = [s["id"] for s in sessions]
        assert "root1" not in ids and "root2" not in ids
        assert "tip1" in ids and "tip2" in ids

        tip1_row = next(s for s in sessions if s["id"] == "tip1")
        tip2_row = next(s for s in sessions if s["id"] == "tip2")
        assert tip1_row["_lineage_root_id"] == "root1"
        assert tip1_row["preview"].startswith("latest message")
        assert tip2_row["_lineage_root_id"] == "root2"
        assert tip2_row["preview"].startswith("second conversation continuation")
        assert tip2_row["cwd"] == "/tmp/workspaces/second"

    def test_list_batches_tip_row_fetch_into_one_query(self, db, monkeypatch):
        """Projection must resolve tip rows for a whole page in one batched
        query, not one _get_session_rich_row() call per compression root."""
        import time as _time

        t0 = _time.time() - 7200
        self._build_compression_chain(db, t0)
        db.create_session("root2", "cli")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0 + 100, "root2"))
        db.append_message("root2", "user", "second conversation start")
        db._conn.execute(
            "UPDATE sessions SET ended_at=?, end_reason=? WHERE id=?",
            (t0 + 200, "compression", "root2"),
        )
        db.create_session("tip2", "cli", parent_session_id="root2")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0 + 201, "tip2"))
        db.append_message("tip2", "user", "second continuation")
        db._conn.commit()

        batch_calls = []
        single_calls = []
        original_batch = db._get_session_rich_rows_batch
        original_single = db._get_session_rich_row

        def counting_batch(session_ids, **kwargs):
            batch_calls.append(list(session_ids))
            return original_batch(session_ids, **kwargs)

        def counting_single(session_id, **kwargs):
            single_calls.append(session_id)
            return original_single(session_id, **kwargs)

        monkeypatch.setattr(db, "_get_session_rich_rows_batch", counting_batch)
        monkeypatch.setattr(db, "_get_session_rich_row", counting_single)

        sessions = db.list_sessions_rich(source="cli", limit=20)
        assert len(sessions) >= 2  # sanity: both chains actually surfaced

        # Two compression roots resolved with exactly one batched call, and
        # zero single-row calls — not one single-row call per root.
        assert len(batch_calls) == 1
        assert set(batch_calls[0]) == {"tip1", "tip2"}
        assert single_calls == []




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
        monkeypatch.setattr(db, "_freelist_ratio", lambda: 0.5)  # ratio gate open
        vacuum_calls = []
        monkeypatch.setattr(db, "vacuum", lambda: vacuum_calls.append(True))

        result = db.maybe_auto_prune_and_vacuum(min_interval_hours=0)

        assert result["vacuumed"] is True
        assert vacuum_calls == [True]
        assert db.get_meta("last_vacuum") is not None

    def test_auto_maintenance_skips_recent_vacuum(self, db, monkeypatch):
        monkeypatch.setattr(db, "prune_sessions", lambda **_kwargs: 3)
        monkeypatch.setattr(db, "_freelist_ratio", lambda: 0.5)  # ratio gate open
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
        monkeypatch.setattr(db, "_freelist_ratio", lambda: 0.5)  # ratio gate open
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
        monkeypatch.setattr(db, "_freelist_ratio", lambda: 0.5)  # ratio gate open
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

    # ── freelist-ratio gate (#54189) ─────────────────────────────────────
    def test_auto_maintenance_skips_vacuum_below_freelist_ratio(self, db, monkeypatch):
        """A prune that frees few pages on a dense DB must NOT trigger VACUUM."""
        monkeypatch.setattr(db, "prune_sessions", lambda **_kwargs: 1)
        monkeypatch.setattr(db, "_freelist_ratio", lambda: 0.05)
        vacuum_calls = []
        monkeypatch.setattr(db, "vacuum", lambda: vacuum_calls.append(True))

        result = db.maybe_auto_prune_and_vacuum(min_interval_hours=0)

        assert result["pruned"] == 1
        assert result["vacuumed"] is False
        assert result["freelist_ratio"] == 0.05
        assert vacuum_calls == []
        assert db.get_meta("last_vacuum") is None
        # The prune itself still counts as a maintenance run.
        assert db.get_meta("last_auto_prune") is not None

    def test_auto_maintenance_vacuums_above_freelist_ratio(self, db, monkeypatch):
        monkeypatch.setattr(db, "prune_sessions", lambda **_kwargs: 1)
        monkeypatch.setattr(db, "_freelist_ratio", lambda: 0.40)
        vacuum_calls = []
        monkeypatch.setattr(db, "vacuum", lambda: vacuum_calls.append(True))

        result = db.maybe_auto_prune_and_vacuum(min_interval_hours=0)

        assert result["vacuumed"] is True
        assert result["freelist_ratio"] == 0.40
        assert vacuum_calls == [True]

    def test_auto_maintenance_freelist_ratio_exactly_at_threshold_skips(self, db, monkeypatch):
        """Gate is strictly greater-than: 25.0% reclaimable does not VACUUM."""
        from hermes_state_common import AUTO_VACUUM_MIN_FREELIST_RATIO

        monkeypatch.setattr(db, "prune_sessions", lambda **_kwargs: 1)
        monkeypatch.setattr(db, "_freelist_ratio", lambda: AUTO_VACUUM_MIN_FREELIST_RATIO)
        vacuum_calls = []
        monkeypatch.setattr(db, "vacuum", lambda: vacuum_calls.append(True))

        result = db.maybe_auto_prune_and_vacuum(min_interval_hours=0)

        assert result["vacuumed"] is False
        assert vacuum_calls == []

    def test_auto_maintenance_unknown_freelist_ratio_falls_back_to_time_throttle(self, db, monkeypatch):
        """If the pragmas cannot be read, don't silently disable VACUUM forever."""
        monkeypatch.setattr(db, "prune_sessions", lambda **_kwargs: 1)
        monkeypatch.setattr(db, "_freelist_ratio", lambda: None)
        vacuum_calls = []
        monkeypatch.setattr(db, "vacuum", lambda: vacuum_calls.append(True))

        result = db.maybe_auto_prune_and_vacuum(min_interval_hours=0)

        assert result["vacuumed"] is True
        assert result["freelist_ratio"] is None
        assert vacuum_calls == [True]

    def test_auto_maintenance_ratio_gate_threshold_is_overridable(self, db, monkeypatch):
        monkeypatch.setattr(db, "prune_sessions", lambda **_kwargs: 1)
        monkeypatch.setattr(db, "_freelist_ratio", lambda: 0.10)
        vacuum_calls = []
        monkeypatch.setattr(db, "vacuum", lambda: vacuum_calls.append(True))

        result = db.maybe_auto_prune_and_vacuum(
            min_interval_hours=0, min_vacuum_freelist_ratio=0.05
        )

        assert result["vacuumed"] is True
        assert vacuum_calls == [True]

    def test_freelist_ratio_reads_real_pragmas(self, db):
        """Real-DB check: freeing most of the file pushes the ratio past the gate."""
        from hermes_state_common import AUTO_VACUUM_MIN_FREELIST_RATIO

        db.create_session(session_id="keep", source="cli")
        db.append_message(session_id="keep", role="user", content="hi")
        for i in range(6):
            sid = f"bulk{i}"
            db.create_session(session_id=sid, source="cli")
            for _ in range(20):
                db.append_message(session_id=sid, role="assistant", content="z" * 4000)
        db._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        dense = db._freelist_ratio()
        assert dense is not None and dense < AUTO_VACUUM_MIN_FREELIST_RATIO

        for i in range(6):
            db.delete_session(f"bulk{i}")
        db._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        sparse = db._freelist_ratio()
        assert sparse is not None and sparse > AUTO_VACUUM_MIN_FREELIST_RATIO

    def test_wal_size_limit_is_bounded(self, db):
        """journal_size_limit must be a finite bound, not SQLite's -1 default.

        Contract, not a snapshot: assert the limit is positive (so the WAL is
        truncated back at checkpoints) rather than pinning the exact byte
        count, which is a tunable.
        """
        mode = db._conn.execute("PRAGMA journal_mode").fetchone()[0]
        if str(mode).lower() != "wal":
            pytest.skip("WAL unavailable on this filesystem")
        limit = db._conn.execute("PRAGMA journal_size_limit").fetchone()[0]
        assert limit > 0, "unbounded WAL: state.db-wal never returns disk to the OS"

    def test_vacuum_leaves_wal_truncated(self, db, tmp_path):
        """VACUUM must not strand a giant WAL beside the database.

        VACUUM rewrites every page through the write-ahead log. Without a
        checkpoint *after* it, a 3 GB database leaves a 3 GB state.db-wal
        behind — `sessions optimize` then consumes far more disk than it
        frees, which is the opposite of its purpose.
        """
        mode = db._conn.execute("PRAGMA journal_mode").fetchone()[0]
        if str(mode).lower() != "wal":
            pytest.skip("WAL unavailable on this filesystem")

        db.create_session(session_id="s1", source="cli")
        for i in range(500):
            db.append_message(
                session_id="s1", role="user", content=f"padding message {i} " * 20
            )
        db.vacuum()

        wal = Path(str(db.db_path) + "-wal")
        if wal.exists():
            limit = db._conn.execute("PRAGMA journal_size_limit").fetchone()[0]
            assert wal.stat().st_size <= max(limit, 0) or wal.stat().st_size == 0, (
                f"WAL left at {wal.stat().st_size} bytes after VACUUM"
            )


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

    def test_first_run_prunes_and_skips_vacuum_when_little_reclaimable(self, db):
        """Pruning two empty sessions frees almost nothing → prune yes, VACUUM no."""
        self._make_old_ended(db, "old1", days_old=100)
        self._make_old_ended(db, "old2", days_old=100)
        db.create_session(session_id="new", source="cli")  # active, must survive

        result = db.maybe_auto_prune_and_vacuum(retention_days=90)
        assert result["skipped"] is False
        assert result["pruned"] == 2
        assert result["vacuumed"] is False  # freelist ratio gate (#54189)
        assert result["freelist_ratio"] is not None
        assert result["freelist_ratio"] <= 0.25
        assert result.get("error") is None
        assert db.get_session("old1") is None
        assert db.get_session("old2") is None
        assert db.get_session("new") is not None

    def test_first_run_prunes_and_vacuums_when_mostly_reclaimable(self, db):
        """Pruning the bulk of the file's pages crosses the 25% gate → VACUUM runs."""
        db.create_session(session_id="new", source="cli")  # active, must survive
        db.append_message(session_id="new", role="user", content="hi")
        for i in range(6):
            sid = f"old{i}"
            self._make_old_ended(db, sid, days_old=100)
            for _ in range(20):
                db.append_message(session_id=sid, role="assistant", content="z" * 4000)
            # Keep the row aged: append_message bumps activity, prune ages by
            # latest message, so push the message timestamps back too.
            db._conn.execute(
                "UPDATE messages SET timestamp = ? WHERE session_id = ?",
                (time.time() - 100 * 86400, sid),
            )
        db._conn.commit()

        result = db.maybe_auto_prune_and_vacuum(retention_days=90)
        assert result["skipped"] is False
        assert result["pruned"] == 6
        assert result["freelist_ratio"] > 0.25
        assert result["vacuumed"] is True
        assert result.get("error") is None
        assert db.get_session("new") is not None
        assert db.get_meta("last_vacuum") is not None

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
            from hermes_state_common import SCHEMA_VERSION
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

    @pytest.mark.parametrize("with_message", [False, True])
    def test_v23_rebuild_from_trigram_tool_calls_projection(
        self, tmp_path, with_message
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

        db = SessionDB(db_path=db_path)
        try:
            assert db._conn is not None
            assert db.fts_optimize_available() is True
            assert db.get_meta("fts_storage_version") == "1"

            original_ensure = db._ensure_fts_schema

            def interrupt_after_demote(cursor, table_name, ddl):
                if table_name == "messages_fts_trigram":
                    raise RuntimeError("injected trigram rebuild interruption")
                return original_ensure(cursor, table_name, ddl)

            db._ensure_fts_schema = interrupt_after_demote
            with pytest.raises(RuntimeError, match="injected trigram"):
                db.optimize_fts_storage(vacuum=False)

            db.close()
            db = SessionDB(db_path=db_path)
            assert db._conn is not None
            assert db.get_meta("fts_rebuild_high_water") is None
            assert db.get_meta("fts_rebuild_progress") is None
            assert db._has_fts_trash(db._conn) is True
            assert db.fts_optimize_available() is True
            assert db.get_meta("fts_storage_version") == "1"
            if with_message:
                assert db._conn.execute(
                    "SELECT 1 FROM messages_fts_trigram "
                    "WHERE messages_fts_trigram MATCH '部署完成' LIMIT 1"
                ).fetchone()

            result = db.optimize_fts_storage(vacuum=False)
            assert result["ok"] is True

            if with_message:
                # messages_fts stays in the tool-calls search path.
                assert len(db.search_messages("UNIQUE_TOOLCALL_TOKEN_43701")) == 1
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
        from hermes_state_common import FTS_SQL, FTS_TRIGRAM_SQL

        conn = db._conn
        db._drop_fts_triggers(conn)
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

        db = SessionDB(db_path=db_path)
        try:
            assert len(db.search_messages("deployment")) == 1
            self._simulate_pre_fix_demote_crash_window(db)
            # Crash window shape: no markers, trash present, empty index.
            assert db.get_meta("fts_rebuild_high_water") is None
            assert db.get_meta("fts_rebuild_progress") is None
            assert db._has_fts_trash(db._conn) is True
            assert db._conn.execute(
                "SELECT COUNT(*) FROM messages_fts_docsize"
            ).fetchone()[0] == 0
            assert len(db.search_messages("deployment")) == 0

            # Still offered (trash and/or empty-index heal).
            assert db.fts_optimize_available() is True

            result = db.optimize_fts_storage(vacuum=False)
            assert result["ok"] is True
            assert db.fts_rebuild_status() is None
            assert db.fts_optimize_available() is False
            assert db.get_meta("fts_storage_version") == str(
                hermes_state_common.FTS_STORAGE_VERSION
            )
            assert db._conn.execute(
                "SELECT name FROM sqlite_master WHERE name LIKE '%_v22_trash%'"
            ).fetchall() == []
            # Historical rows searchable again; index fully populated.
            assert len(db.search_messages("deployment")) == 1
            assert len(db.search_messages("TOOLBLOB")) == 1
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

        db = SessionDB(db_path=db_path)
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
                (str(hermes_state_common.FTS_STORAGE_VERSION),),
            )
            db._conn.commit()

            assert db.get_meta("fts_rebuild_high_water") is None
            assert db._has_fts_trash(db._conn) is False
            assert db._fts_external_index_empty_with_messages(db._conn) is True
            # Must still be offered despite the premature stamp.
            assert db.fts_optimize_available() is True
            assert len(db.search_messages("deployment")) == 0

            result = db.optimize_fts_storage(vacuum=False)
            assert result["ok"] is True
            assert len(db.search_messages("deployment")) == 1
            assert db.get_meta("fts_storage_version") == str(
                hermes_state_common.FTS_STORAGE_VERSION
            )
            assert db.fts_optimize_available() is False
        finally:
            db.close()

    def test_optimize_heals_high_water_without_progress(self, tmp_path):
        """high_water without progress used to make fts_rebuild_step return
        False immediately (treated as finished by another process), then
        settle stamped success while the marker remained. Re-seed progress
        and complete the empty-index backfill."""
        db_path = tmp_path / "v22.db"
        self._build_v22_db(db_path)
        db = SessionDB(db_path=db_path)
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
            assert db.fts_optimize_available() is True
            # Empty index: base FTS MATCH finds nothing (gap LIKE may still
            # supplement when high_water is set — that is intentional).
            assert db._conn.execute(
                "SELECT COUNT(*) FROM messages_fts_docsize"
            ).fetchone()[0] == 0

            result = db.optimize_fts_storage(vacuum=False)
            assert result["ok"] is True
            assert db.get_meta("fts_rebuild_high_water") is None
            assert db.get_meta("fts_rebuild_progress") is None
            n_msg = db._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            n_fts = db._conn.execute(
                "SELECT COUNT(*) FROM messages_fts_docsize"
            ).fetchone()[0]
            assert n_fts == n_msg
            assert len(db.search_messages("deployment")) == 1
            assert db.fts_optimize_available() is False
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
        db = SessionDB(db_path=db_path)
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

            result = db.optimize_fts_storage(vacuum=False)
            assert result["ok"] is True
            n_msg = db._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            n_fts = db._conn.execute(
                "SELECT COUNT(*) FROM messages_fts_docsize"
            ).fetchone()[0]
            # Exactly one index entry per message: no replay duplicates.
            assert n_fts == n_msg
            assert len(db.search_messages("deployment")) == 1
            db._conn.execute(
                "INSERT INTO messages_fts(messages_fts, rank) VALUES('integrity-check', 1)"
            )
        finally:
            db.close()

    def test_repair_bookkeeping_reseeds_missing_progress(self, tmp_path):
        """Unit: high_water without progress gets progress='0' without
        forcing a full marker reset when a real backfill is already claimed."""
        db = SessionDB(db_path=tmp_path / "fresh.db")
        try:
            db.create_session(session_id="s1", source="cli")
            db.append_message("s1", role="user", content="bookkeeping needle")
            db.set_meta("fts_rebuild_high_water", "42")
            db._conn.execute(
                "DELETE FROM state_meta WHERE key = ?", ("fts_rebuild_progress",)
            )
            db._conn.commit()
            db._repair_optimize_bookkeeping()
            assert db.get_meta("fts_rebuild_high_water") == "42"
            assert db.get_meta("fts_rebuild_progress") == "0"
        finally:
            db.close()

    def test_demote_writes_markers_before_empty_schema(self, tmp_path):
        """Demote must commit rebuild markers before createscript builds the
        empty v23 tables — so a crash between stage and ensure still leaves
        a resumable claim rather than an unmarked empty index."""
        db_path = tmp_path / "v22.db"
        self._build_v22_db(db_path)
        db = SessionDB(db_path=db_path)
        try:
            # Patch ensure to fail *after* the staged write commits, simulating
            # death mid schema-create. Markers must already be durable.
            orig_ensure = db._ensure_fts_schema
            calls = {"n": 0}

            def boom(cursor, table_name, ddl):
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
                return orig_ensure(cursor, table_name, ddl)

            db._ensure_fts_schema = boom  # type: ignore[method-assign]
            try:
                db._demote_legacy_fts_to_trash()
                raise AssertionError("demote should have raised")
            except sqlite3.OperationalError as exc:
                assert "simulated crash" in str(exc)

            # Staged demote survived: markers + trash, no successful stamp.
            assert db.get_meta("fts_rebuild_high_water") is not None
            assert db.get_meta("fts_rebuild_progress") == "0"
            assert db._has_fts_trash(db._conn) is True
            assert db.get_meta("fts_storage_version") is None

            # Restore ensure and resume — full optimize completes.
            db._ensure_fts_schema = orig_ensure  # type: ignore[method-assign]
            result = db.optimize_fts_storage(vacuum=False)
            assert result["ok"] is True
            assert len(db.search_messages("deployment")) == 1
            assert db.fts_optimize_available() is False
        finally:
            db.close()

    def test_optimize_settle_refuses_pending_backfill(self, tmp_path):
        """Settle must not stamp while high_water markers remain."""
        db = SessionDB(db_path=tmp_path / "fresh.db")
        try:
            db.create_session(session_id="s1", source="cli")
            db.append_message("s1", role="user", content="settle guard needle")
            # Plant markers without going through demote.
            db.set_meta("fts_rebuild_high_water", "1")
            db.set_meta("fts_rebuild_progress", "0")
            # The public contract: optimize returns ok=False when still
            # pending. Simulate an unfinishable backfill by stubbing the
            # chunk step to a no-op while markers stay.
            db.fts_rebuild_step = lambda: False  # type: ignore[method-assign]
            result = db.optimize_fts_storage(vacuum=False)
            assert result["ok"] is False
            assert result.get("reason") == "backfill_incomplete"
            assert db.get_meta("fts_storage_version") is None
            assert db.get_meta("fts_rebuild_high_water") is not None
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

    def test_fts_teardown_single_key_high_water_drains_and_drops(self, tmp_path):
        """#79324: single-column-key trash tables drain via a high-water
        marker so each chunk only scans rows after the previous chunk.

        Builds a large trash table with a rowid-like integer PK, then drives
        ``_fts_teardown_trash_step`` to completion. Verifies every row is
        removed, the marker advances monotonically, the marker is cleared,
        and the table is dropped at the end.
        """
        db = SessionDB(db_path=tmp_path / "trash.db")
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

            assert db._has_fts_trash(conn) is True

            steps = 0
            while db._fts_teardown_trash_step():
                steps += 1
                assert steps < 100, "teardown never finished"

            # All rows gone, table dropped, marker cleaned up.
            assert db._has_fts_trash(conn) is False
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
        db = SessionDB(db_path=tmp_path / "trash.db")
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
            assert db._fts_teardown_trash_step() is True
            assert db._fts_teardown_trash_step() is True
            marker = db.get_meta("fts_teardown_fts_v22_trash_messages_fts_data_progress")
            assert marker is not None
            assert int(marker) == 1000  # 2 chunks x 500 rows

            remaining = conn.execute(
                "SELECT COUNT(*) FROM fts_v22_trash_messages_fts_data"
            ).fetchone()[0]
            assert remaining == 200

            # Resume: drains the rest, drops the table.
            while db._fts_teardown_trash_step():
                pass
            assert db._has_fts_trash(conn) is False
            assert db.get_meta("fts_teardown_fts_v22_trash_messages_fts_data_progress") is None
        finally:
            db.close()

    def test_fts_teardown_compound_key_keeps_legacy_path(self, tmp_path):
        """#79324: multi-column-PK trash tables (small by construction) keep
        the legacy chunked delete — the high-water path only applies to
        single-column keys."""
        db = SessionDB(db_path=tmp_path / "trash.db")
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
            while db._fts_teardown_trash_step():
                steps += 1
                assert steps < 10

            assert db._has_fts_trash(conn) is False
            assert conn.execute(
                "SELECT name FROM sqlite_master WHERE name = "
                "'fts_v22_trash_messages_fts_idx'"
            ).fetchone() is None
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
            hermes_state_wal, "is_sqlite_wal_reset_vulnerable", lambda version_info=None: False
        )


    def test_sets_wal_on_fresh_connection(self, tmp_path):
        """Probe sees 'delete', then set-pragma runs and returns 'wal'."""
        import sqlite3
        from hermes_state_wal import apply_wal_with_fallback

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
        from hermes_state_wal import apply_wal_with_fallback

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
        from hermes_state_wal import apply_wal_with_fallback

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


@pytest.mark.parametrize(
    "persisted_session_key",
    ["agent:main:telegram:dm:chat-1", None],
    ids=["exact-key", "peer-fallback"],
)
def test_gateway_session_recovery_does_not_cross_newer_reset_boundary(
    db, persisted_session_key
):
    """A newer session_reset row fences recovery for the peer (#68539).

    Recovery must never reach *behind* an intentional /new boundary and
    resurrect an older still-open row — if the newest boundary row for the
    peer is reset-ended, recovery returns nothing.
    """
    peer = {
        "user_id": "user-1",
        "session_key": persisted_session_key,
        "chat_id": "chat-1",
        "chat_type": "dm",
    }
    db.create_session("gw-before-reset", "telegram", **peer)
    db.append_message("gw-before-reset", "user", "old context")
    db.create_session("gw-reset", "telegram", **peer)
    db.append_message("gw-reset", "user", "/new")
    db.end_session("gw-reset", "session_reset")

    assert db.find_latest_gateway_session_for_peer(
        source="telegram",
        user_id="user-1",
        session_key="agent:main:telegram:dm:chat-1",
        chat_id="chat-1",
        chat_type="dm",
    ) is None


def test_peer_fallback_never_adopts_a_sibling_profiles_row(tmp_path, monkeypatch):
    """#74285: the peer-tuple fallback is fenced by the store's own profile.

    A Telegram DM peer tuple (chat_id == user_id, no thread) is identical for
    every bot, so a legacy sibling-profile row sitting in this store — written
    before the per-profile partition — must lose to the older own row, and
    with no own row recovery must return nothing rather than the sibling's.
    """
    import hermes_state

    root = tmp_path / "hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    store = SessionDB(db_path=root / "state.db")  # owner: default
    try:
        peer = {"user_id": "42", "chat_id": "42", "chat_type": "dm"}
        store.create_session("sibling", "telegram", session_key="agent:bot2:telegram:dm:42",
                             profile_name="bot2", **peer)
        store.append_message("sibling", "user", "bot2's conversation")

        def recover():
            return store.find_latest_gateway_session_for_peer(
                source="telegram", session_key="agent:main:telegram:dm:42", **peer
            )

        assert recover() is None  # only the sibling exists: fail closed

        store.create_session("own", "telegram", session_key="agent:main:telegram:dm:42:old", **peer)
        store.append_message("own", "user", "default's conversation")
        store._execute_write(
            lambda c: c.execute("UPDATE sessions SET last_activity_at = 1 WHERE id = 'own'")
        )
        assert recover()["id"] == "own"  # older own row beats newer sibling row
    finally:
        store.close()


def test_child_inherits_parent_profile_only_within_its_key_namespace(db):
    """#88381: parent→child ``profile_name`` COALESCE is fenced by ``agent:<ns>:``.

    A default child (``agent:main:``) forked from a sibling profile's row must
    not be durably mislabelled as that profile's; same-namespace and keyless
    (CLI/subagent) children keep inheriting.
    """
    db.create_session("parent", "telegram", session_key="agent:bot2:telegram:dm:42",
                      profile_name="bot2")
    db.create_session("cross", "telegram", parent_session_id="parent",
                      session_key="agent:main:telegram:dm:42")
    db.create_session("same", "telegram", parent_session_id="parent",
                      session_key="agent:bot2:telegram:dm:42:r2")
    db.create_session("keyless", "cli", parent_session_id="parent")
    assert db.get_session("cross")["profile_name"] is None
    assert db.get_session("same")["profile_name"] == "bot2"
    assert db.get_session("keyless")["profile_name"] == "bot2"












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

    def test_batch_compact_rows_omits_system_prompt_keeps_git_fields(self, db):
        """_get_session_rich_rows_batch(compact_rows=True) must apply the same
        schema-derived compact projection as the single-row path: no
        system_prompt blob, but git_branch/git_repo_root still present."""
        self._create(db, "s1", system_prompt="should be gone")
        db.update_session_cwd("s1", "/tmp/w1", git_branch="main", git_repo_root="/tmp/w1")
        rows = db._get_session_rich_rows_batch(["s1"], compact_rows=True)
        assert set(rows) == {"s1"}
        row = rows["s1"]
        assert "system_prompt" not in row
        assert row["git_branch"] == "main"
        assert row["git_repo_root"] == "/tmp/w1"

    def test_compression_tip_projection_threads_compact_rows(self, db):
        """list_sessions_rich(compact_rows=True) must thread compact_rows
        through the batched tip-row fetch: the projected tip row must lack
        system_prompt but keep git metadata (guards the call site at the
        projection loop, not just the batch helper)."""
        import time as _time

        t0 = _time.time() - 3600
        db.create_session("rootc", "cli")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0, "rootc"))
        db.append_message("rootc", "user", "start")
        db._conn.execute(
            "UPDATE sessions SET ended_at=?, end_reason=? WHERE id=?",
            (t0 + 100, "compression", "rootc"),
        )
        db.create_session("tipc", "cli", parent_session_id="rootc")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0 + 101, "tipc"))
        db.append_message("tipc", "user", "continuation")
        db.update_system_prompt("tipc", "big blob " * 500)
        db.update_session_cwd("tipc", "/tmp/w2", git_branch="dev", git_repo_root="/tmp/w2")
        db._conn.commit()

        rows = db.list_sessions_rich(source="cli", compact_rows=True)
        tip = next(s for s in rows if s["id"] == "tipc")
        assert tip["_lineage_root_id"] == "rootc"
        assert "system_prompt" not in tip
        assert tip["git_branch"] == "dev"
        assert tip["git_repo_root"] == "/tmp/w2"






# =========================================================================
# get_messages pagination (salvage follow-up for #60347)
# =========================================================================

class TestGetMessagesPagination:
    """get_messages(limit=, offset=) pages in insertion order; the default
    (limit=None) returns the full transcript unchanged."""

    def _seed(self, db, n=10):
        db.create_session(session_id="s1", source="cli")
        # One write transaction for the whole seed: per-row append_message
        # pays a commit (and, off WAL, an fsync) per message, which at
        # n=3000 was ~10s of pure seeding before the query under test ran.
        db.append_messages_batch(
            "s1",
            [
                {
                    "role": "user" if i % 2 == 0 else "assistant",
                    "content": f"msg-{i}",
                }
                for i in range(n)
            ],
        )

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

    def test_latest_pages_count_back_from_newest_but_remain_chronological(self, db):
        self._seed(db)
        page1 = db.get_messages("s1", limit=4, offset=0, latest=True)
        page2 = db.get_messages("s1", limit=4, offset=4, latest=True)
        page3 = db.get_messages("s1", limit=4, offset=8, latest=True)
        assert [m["content"] for m in page1] == ["msg-6", "msg-7", "msg-8", "msg-9"]
        assert [m["content"] for m in page2] == ["msg-2", "msg-3", "msg-4", "msg-5"]
        assert [m["content"] for m in page3] == ["msg-0", "msg-1"]

    def test_after_id_keyset_pages_forward_in_insertion_order(self, db):
        self._seed(db)
        page1 = db.get_messages("s1", limit=4, after_id=0)
        assert [m["content"] for m in page1] == ["msg-0", "msg-1", "msg-2", "msg-3"]
        page2 = db.get_messages("s1", limit=4, after_id=page1[-1]["id"])
        assert [m["content"] for m in page2] == ["msg-4", "msg-5", "msg-6", "msg-7"]
        page3 = db.get_messages("s1", limit=4, after_id=page2[-1]["id"])
        assert [m["content"] for m in page3] == ["msg-8", "msg-9"]
        with pytest.raises(ValueError):
            db.get_messages("s1", limit=4, after_id=0, latest=True)
        with pytest.raises(ValueError):
            db.get_messages("s1", limit=4, after_id=0, offset=2)

    def test_resume_safety_counts_active_rows_across_lineage(self, db):
        db.create_session(session_id="root", source="cli")
        db.append_messages_batch(
            "root",
            [{"role": "user", "content": f"root-{i}"} for i in range(3)],
        )
        db.create_session(
            session_id="tip",
            source="compression",
            parent_session_id="root",
        )
        db.append_messages_batch(
            "tip",
            [{"role": "assistant", "content": f"tip-{i}"} for i in range(2)],
        )

        assert db.get_resume_message_count("tip") == 5
        with pytest.raises(hermes_state.SessionResumeTooLargeError) as exc_info:
            db.assert_resume_safe("tip", max_messages=4)
        assert exc_info.value.message_count == 5
        assert exc_info.value.limit == 4

    def test_resume_safety_tip_only_counts_the_tip_segment(self, db):
        """A deep compression lineage behind a small tip resumes tip-only.

        The Desktop Bot Chat shape: many compaction segments (~29k rows of
        lineage) and a small live tip. Callers that never materialize the
        ancestors (deferred / omit_messages / lazy resume, tip-only model
        restore) must be bounded by the tip alone, and the message must name
        the scope it counted.
        """
        prev = None
        for i in range(6):
            sid = f"seg-{i}"
            kwargs = {"parent_session_id": prev} if prev else {}
            db.create_session(session_id=sid, source="tui", **kwargs)
            db.append_messages_batch(
                sid,
                [{"role": "user", "content": f"{sid}-{j}"} for j in range(4)],
            )
            if i < 5:
                db.end_session(sid, "compression")
            prev = sid

        assert db.get_resume_message_count("seg-5") == 24
        assert db.get_resume_message_count("seg-5", tip_only=True) == 4
        with pytest.raises(hermes_state.SessionResumeTooLargeError) as full:
            db.assert_resume_safe("seg-5", max_messages=10)
        assert "across its lineage" in str(full.value)
        assert db.assert_resume_safe("seg-5", max_messages=10, tip_only=True) == 4
        with pytest.raises(hermes_state.SessionResumeTooLargeError) as tip:
            db.assert_resume_safe("seg-5", max_messages=3, tip_only=True)
        assert tip.value.message_count == 4
        assert "in its tip segment" in str(tip.value)

    def test_resume_guard_counts_exactly_what_a_branch_resume_loads(self, db):
        """An explicit /branch copy owns its transcript: the guard and the
        resume readers must agree that its lineage is itself alone."""
        db.create_session(session_id="parent", source="tui")
        db.append_messages_batch(
            "parent",
            [{"role": "user", "content": f"parent-{i}"} for i in range(6)],
        )
        db.create_session(
            session_id="branch",
            source="tui",
            parent_session_id="parent",
            model_config={"_branched_from": "parent"},
        )
        db.append_messages_batch(
            "branch",
            [{"role": "user", "content": f"branch-{i}"} for i in range(2)],
        )

        _, display = db.get_resume_conversations("branch")
        assert len(display) == 2
        assert db.get_ancestor_display_prefix("branch") == []
        # Before: the guard walked parent_session_id and counted 8, so a branch
        # could be refused for rows a resume would never load.
        assert db.get_resume_message_count("branch") == 2
        assert db.assert_resume_safe("branch", max_messages=5) == 2

    def test_export_safety_is_bounded_to_the_requested_active_segment(self, db):
        db.create_session(session_id="root", source="cli")
        db.append_messages_batch(
            "root",
            [{"role": "user", "content": f"root-{i}"} for i in range(3)],
        )
        db.create_session(
            session_id="tip",
            source="compression",
            parent_session_id="root",
        )
        db.append_messages_batch(
            "tip",
            [{"role": "assistant", "content": f"tip-{i}"} for i in range(2)],
        )

        assert db.assert_export_safe("tip", max_messages=2) == 2
        with pytest.raises(hermes_state.SessionExportTooLargeError) as exc_info:
            db.assert_export_safe("root", max_messages=2)
        assert exc_info.value.session_id == "root"
        assert exc_info.value.message_count == 3
        assert exc_info.value.limit == 2

    def test_zero_limit_disables_resume_and_export_guards(self, db, monkeypatch):
        """sessions.max_*_messages: 0 disables the guard entirely."""
        db.create_session(session_id="big", source="cli")
        db.append_messages_batch(
            "big",
            [{"role": "user", "content": f"msg-{i}"} for i in range(5)],
        )

        # A small explicit limit rejects...
        with pytest.raises(hermes_state.SessionResumeTooLargeError):
            db.assert_resume_safe("big", max_messages=2)
        with pytest.raises(hermes_state.SessionExportTooLargeError):
            db.assert_export_safe("big", max_messages=2)

        # ...but a config-resolved limit of 0 disables both guards: no raise,
        # and no counting work at all (returns 0 — callers use the raise side
        # effect only).
        monkeypatch.setattr(hermes_state, "resolved_max_resume_messages", lambda: 0)
        monkeypatch.setattr(hermes_state, "resolved_max_export_messages", lambda: 0)
        assert db.assert_resume_safe("big") == 0
        assert db.assert_export_safe("big") == 0
        # An explicit 0 disables too, independent of config.
        assert db.assert_resume_safe("big", max_messages=0) == 0
        assert db.assert_export_safe("big", max_messages=0) == 0

    def test_guard_limits_resolve_from_config_at_call_time(self, db, monkeypatch):
        db.create_session(session_id="cfg", source="cli")
        db.append_messages_batch(
            "cfg",
            [{"role": "user", "content": f"msg-{i}"} for i in range(4)],
        )

        monkeypatch.setattr(hermes_state, "resolved_max_resume_messages", lambda: 3)
        monkeypatch.setattr(hermes_state, "resolved_max_export_messages", lambda: 3)
        with pytest.raises(hermes_state.SessionResumeTooLargeError) as resume_exc:
            db.assert_resume_safe("cfg")
        assert resume_exc.value.limit == 3
        with pytest.raises(hermes_state.SessionExportTooLargeError) as export_exc:
            db.assert_export_safe("cfg")
        assert export_exc.value.limit == 3





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

    def test_ignores_non_integer_performance_values(self, tmp_path, monkeypatch):
        """Garbage cache_size/mmap_size/temp_store values must be rejected."""
        import sqlite3
        from hermes_state import apply_database_pragmas

        conn = sqlite3.connect(str(tmp_path / "pragmas.db"))
        try:
            before = {
                name: conn.execute(f"PRAGMA {name}").fetchone()[0]
                for name in ("cache_size", "mmap_size", "temp_store")
            }
            self._patch_cfg(
                monkeypatch,
                {
                    "database": {
                        "cache_size": "big",
                        "mmap_size": [256],
                        "temp_store": "ram please",
                    }
                },
            )
            apply_database_pragmas(conn, db_label="test.db")
            after = {
                name: conn.execute(f"PRAGMA {name}").fetchone()[0]
                for name in ("cache_size", "mmap_size", "temp_store")
            }
            assert after == before
        finally:
            conn.close()


class TestInsightsToolCallIndex:
    """The Insights assistant tool-call scan has a predicate-aligned index.

    ``InsightsEngine._get_tool_usage`` / ``_get_skill_usage`` filter messages by
    ``role = 'assistant' AND tool_calls IS NOT NULL``.  A partial index over that
    predicate keeps the scan off the full ``messages`` table on a large state.db.
    """

    _INDEX = "idx_messages_assistant_calls_by_session"

    def _index_defn(self, conn):
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
            (self._INDEX,),
        ).fetchone()
        return row["sql"] if row else None

    def test_index_created_on_fresh_db(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "fresh.db")
        try:
            sql = self._index_defn(db._conn)
            assert sql is not None, "partial index missing on a fresh database"
            # Partial predicate must match the queried rows exactly.
            assert "role = 'assistant'" in sql
            assert "tool_calls IS NOT NULL" in sql
        finally:
            db.close()

    def test_index_created_on_existing_db(self, tmp_path):
        """Reopening a DB that predates the index must create it (SCHEMA_SQL is
        re-run on every open; role/tool_calls are original base columns)."""
        db_path = tmp_path / "legacy.db"
        db = SessionDB(db_path=db_path)
        # Simulate a database created before the index shipped.
        db._conn.execute(f"DROP INDEX IF EXISTS {self._INDEX}")
        db._conn.commit()
        assert self._index_defn(db._conn) is None
        db.close()

        db2 = SessionDB(db_path=db_path)
        try:
            assert self._index_defn(db2._conn) is not None, (
                "index not recreated when reopening an existing database"
            )
        finally:
            db2.close()

    def test_index_predicate_is_partial(self, db):
        """The index covers only the assistant tool-call rows Insights reads.

        Query-plan coverage (that the Insights queries actually select this
        index, for both scopes, without ANALYZE) lives with the queries in
        tests/agent/test_insights.py.
        """
        sql = self._index_defn(db._conn)
        assert sql is not None
        assert "WHERE" in sql
        assert "role = 'assistant'" in sql
        assert "tool_calls IS NOT NULL" in sql
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
        seeded = SessionDB(db_path=db_path)
        try:
            seeded.create_session(session_id="s1", source="cli")
            for i in range(n):
                seeded.append_message(
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
            "hermes_state.sqlite3.connect", connect_without_trigram
        )
        db = SessionDB(db_path=db_path)
        try:
            assert db._trigram_available is False
            # A trigram-less runtime leaves no trigram index on disk.
            db._conn.execute("DROP TABLE IF EXISTS messages_fts_trigram")
            db._conn.commit()
            assert db._fts_table_exists("messages_fts_trigram") is False

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
            db._fts_rebuild_finish()

            # The sweep ran to completion: markers cleared…
            assert db.get_meta("fts_rebuild_high_water") is None
            assert db.get_meta("fts_rebuild_progress") is None
            # …and the base index is still usable (the fix must not disable
            # real search to dodge the error).
            assert db.search_messages("zebra")
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
            "hermes_state.sqlite3.connect", connect_without_trigram
        )
        db = SessionDB(db_path=db_path)
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

            result = db.optimize_fts_storage(vacuum=False)
            assert result["ok"] is True
            assert db.get_meta("fts_rebuild_high_water") is None
            assert db.search_messages("zebra")
        finally:
            db.close()



class TestPerformancePragmasEndToEnd:
    """E2E guard for PR #71755: config-gated cache_size / mmap_size /
    temp_store must reach EVERY connection type (writer, read-only
    cross-profile attach, WAL per-thread reader) — and default installs
    (no ``database:`` keys) must see byte-identical SQLite defaults.

    NOTE: SQLite's compiled-in default for ``cache_size`` is already
    ``-2000``, so the configured value here is ``-16000`` — a value the
    test can actually discriminate from the default (a reverted prod
    change must FAIL this test, not accidentally pass it).
    """

    PRAGMAS = ("cache_size", "mmap_size", "temp_store")
    CONFIGURED = {"cache_size": -16000, "mmap_size": 1048576, "temp_store": 2}

    @staticmethod
    def _read(conn):
        return {
            name: conn.execute(f"PRAGMA {name}").fetchone()[0]
            for name in ("cache_size", "mmap_size", "temp_store")
        }

    @staticmethod
    def _sqlite_defaults(tmp_path):
        import sqlite3

        conn = sqlite3.connect(str(tmp_path / "baseline.db"))
        try:
            return {
                name: conn.execute(f"PRAGMA {name}").fetchone()[0]
                for name in ("cache_size", "mmap_size", "temp_store")
            }
        finally:
            conn.close()

    def _fresh_home(self, tmp_path, monkeypatch, config_text=None):
        import hermes_state

        # Local venvs may bundle a WAL-reset-vulnerable SQLite (e.g. 3.46.0),
        # which would silently disable WAL and skip the per-thread reader
        # path. Force WAL eligibility so _get_read_conn is truly exercised
        # (established pattern used by the WAL tests above).
        monkeypatch.setattr(
            hermes_state_wal, "is_sqlite_wal_reset_vulnerable",
            lambda version_info=None: False,
        )
        home = tmp_path / "hermes_home"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        if config_text is not None:
            (home / "config.yaml").write_text(config_text)
        return home

    def test_configured_pragmas_reach_all_connection_types(
        self, tmp_path, monkeypatch
    ):
        from hermes_state import SessionDB

        home = self._fresh_home(
            tmp_path,
            monkeypatch,
            "database:\n"
            "  cache_size: -16000\n"
            "  temp_store: 2\n"
            "  mmap_size: 1048576\n",
        )
        db_path = home / "state.db"
        db = SessionDB(db_path=db_path)
        try:
            # Writer connection.
            assert self._read(db._conn) == self.CONFIGURED
            # WAL per-thread reader.
            rconn = db._get_read_conn()
            assert rconn is not None, "WAL reader expected on local filesystem"
            assert self._read(rconn) == self.CONFIGURED
        finally:
            db.close()

        # Read-only cross-profile attach.
        ro = SessionDB(db_path=db_path, read_only=True)
        try:
            assert self._read(ro._conn) == self.CONFIGURED
        finally:
            ro.close()

    def test_defaults_unchanged_without_config(self, tmp_path, monkeypatch):
        """No database: keys in config.yaml → SQLite defaults untouched."""
        from hermes_state import SessionDB

        defaults = self._sqlite_defaults(tmp_path)
        home = self._fresh_home(tmp_path, monkeypatch, config_text=None)
        db_path = home / "state.db"
        db = SessionDB(db_path=db_path)
        try:
            assert self._read(db._conn) == defaults
            rconn = db._get_read_conn()
            if rconn is not None:
                assert self._read(rconn) == defaults
        finally:
            db.close()

        ro = SessionDB(db_path=db_path, read_only=True)
        try:
            assert self._read(ro._conn) == defaults
        finally:
            ro.close()


class TestFts5SanitizerCharacterClass:
    """Every character FTS5 rejects outside a quoted phrase must be stripped.

    A survivor reaches MATCH raw and raises, which the execute site swallows
    into zero results — so the search silently finds nothing rather than
    erroring. Assertions run the sanitized text against a real FTS5 table.
    """

    @staticmethod
    def _fts_table():
        import sqlite3

        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(content)")
        conn.execute(
            "INSERT INTO t (content) VALUES "
            "('meet me at user host about gateway run py it s 50 a b')"
        )
        return conn

    @staticmethod
    def _sanitize(query):
        from hermes_state_search import SessionSearchMixin

        return SessionSearchMixin._sanitize_fts5_query(query)

    @pytest.mark.parametrize(
        "query",
        [
            "it's",                 # apostrophe — ordinary prose
            "gateway/run.py",       # path separator
            "user@host",            # email / handle
            "a,b",                  # comma
            "why?",                 # question mark
            "e=mc2",                # equals
            "a;b", "a!b", "a&b", "a|b", "x~y",
            "#tag", "$dollar", "[bracket]", "<tag>",
            r"C:\path\file",        # backslash
        ],
    )
    def test_query_stays_parsable(self, query):
        conn = self._fts_table()
        sanitized = self._sanitize(query)
        if not sanitized.strip():
            return
        # Raises sqlite3.OperationalError if a special character survived.
        conn.execute("SELECT count(*) FROM t WHERE t MATCH ?", (sanitized,)).fetchone()

    def test_plain_terms_are_untouched(self):
        assert self._sanitize("hello world").split() == ["hello", "world"]

    def test_quoted_phrase_survives(self):
        assert '"exact phrase"' in self._sanitize('"exact phrase"')

    def test_hyphen_dotted_term_still_quoted(self):
        # Step 5's behaviour must not regress: my-app.config.ts stays one term.
        assert '"my-app.config.ts"' in self._sanitize("my-app.config.ts")

    def test_prefix_star_still_works(self):
        conn = self._fts_table()
        sanitized = self._sanitize("gate*")
        rows = conn.execute(
            "SELECT count(*) FROM t WHERE t MATCH ?", (sanitized,)
        ).fetchone()
        assert rows[0] == 1

    def test_percent_stripped_for_non_cjk_query(self):
        # % is kept only for the CJK LIKE fallback; a non-CJK query never
        # reaches that fallback, so % must be stripped before MATCH.
        conn = self._fts_table()
        sanitized = self._sanitize("50%")
        assert "%" not in sanitized
        conn.execute(
            "SELECT count(*) FROM t WHERE t MATCH ?", (sanitized,)
        ).fetchone()

    def test_percent_preserved_for_cjk_query(self):
        # The CJK LIKE fallback builds its own pattern from the sanitized
        # text; keep % intact there (pre-existing contract).
        sanitized = self._sanitize("完成50%")
        assert "%" in sanitized
