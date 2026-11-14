"""The conversation store's session and message functions on a real StoreHolder: session create, end and update, message append and read, timestamp round trips, and session and message counts."""

import contextlib
import json
import re
import sqlite3
import threading
import time

import pytest

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite.conversation._store.messages import (
    append_message,
    get_ancestor_display_prefix,
    get_messages,
    get_messages_as_conversation,
    get_resume_conversations,
    message_count,
    replace_messages,
)
from persistence.sqlite.conversation._store.search import search_messages
from persistence.sqlite.conversation._store.sessions import (
    create_session,
    end_session,
    get_session,
    session_count,
    session_count_ge,
    update_session_cwd,
    update_session_model,
)
from persistence.sqlite.conversation._store.usage import update_session_billing_route, update_token_counts


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


# =========================================================================
# Session lifecycle
# =========================================================================

class TestSessionLifecycle:
    def test_create_and_get_session(self, db):
        sid = create_session(db,
            session_id="s1",
            source="cli",
            model="test-model",
        )
        assert sid == "s1"

        session = get_session(db, "s1")
        assert session is not None
        assert session["source"] == "cli"
        assert session["model"] == "test-model"
        assert session["ended_at"] is None

    def test_branch_resume_does_not_include_parent_messages_added_after_fork(self, db):
        """A branch owns its copied transcript, not the parent's later turns."""
        create_session(db, "parent", source="tui")
        append_message(db, "parent", role="user", content="before branch")
        append_message(db, "parent", role="assistant", content="initial answer")

        create_session(db,
            "branch",
            source="tui",
            parent_session_id="parent",
            model_config={"_branched_from": "parent"},
        )
        append_message(db, "branch", role="user", content="before branch")
        append_message(db, "branch", role="assistant", content="initial answer")

        # The original conversation can be resumed after the fork. Those new
        # rows must not leak into the already-created branch's transcript.
        append_message(db, "parent", role="user", content="after branch")
        append_message(db, "parent", role="assistant", content="later answer")

        _, display_history = get_resume_conversations(db, "branch")

        assert [message["content"] for message in display_history] == [
            "before branch",
            "initial answer",
        ]
        assert [
            message["content"]
            for message in get_messages_as_conversation(db, "branch", include_ancestors=True)
        ] == ["before branch", "initial answer"]
        assert get_ancestor_display_prefix(db, "branch") == []

    def test_update_session_cwd_persists_git_branch(self, db):
        create_session(db, session_id="s1", source="cli")
        update_session_cwd(db, "s1", "/work/repo", git_branch="pets-feature")

        session = get_session(db, "s1")
        assert session["cwd"] == "/work/repo"
        assert session["git_branch"] == "pets-feature"

    def test_end_session_first_reason_wins_across_concurrent_connections(
        self, db
    ):
        """Concurrent finalizers perform one transition, not last-write-wins."""
        import threading

        create_session(db, session_id="s1", source="cron")
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

        peer = StoreHolder(db_path=db.db_path)
        barrier = threading.Barrier(2)
        errors = []

        def _end(session_db, reason):
            try:
                barrier.wait(timeout=5)
                end_session(session_db, "s1", reason)
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
            assert get_session(db, "s1")["end_reason"] == audit_rows[0]["reason"]
        finally:
            peer.close()

    def test_update_session_model_clears_browser_lock_and_preserves_lineage(self, db):
        """A later /model switch must replace, not compete with, a Browser lock."""
        create_session(db,
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

        update_session_model(db, "s1", "anthropic/claude-opus-4.8")

        session = get_session(db, "s1")
        model_config = json.loads(session["model_config"])
        assert session["model"] == "anthropic/claude-opus-4.8"
        assert "browser_model_lock" not in model_config
        assert model_config["_branched_from"] == "parent-session"

    def test_first_accounted_route_replaces_all_route_fields_atomically(self, db):
        create_session(db, session_id="route", source="cli", model="primary")
        update_session_billing_route(db,
            "route", provider="primary-provider",
            base_url="https://primary.example/v1", billing_mode="api_key",
        )
        update_token_counts(db,
            "route", model="fallback", billing_provider="fallback-provider",
            billing_base_url=None, billing_mode=None, api_call_count=1,
        )
        row = get_session(db, "route")
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

        monkeypatch.setattr("persistence.sqlite._holder.sqlite3.connect", connect_without_trigram)
        db = StoreHolder(db_path=db_path)
        try:
            create_session(db, session_id="s1", source="cli")
            append_message(db, "s1", role="user", content="大别山项目计划书")
            append_message(db, "s1", role="user", content="长江大桥设计方案")

            # 3+ CJK chars would normally use trigram, but it's unavailable.
            # Must fall back to LIKE and still return results.
            results = search_messages(db, "大别山")
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
        create_session(db, session_id="s1", source="cli")
        append_message(db, "s1", role="user", content="Hello")
        append_message(db, "s1", role="assistant", content="Hi there!")

        messages = get_messages(db, "s1")
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"

    def test_settled_open_issues_no_main_db_writes(self, tmp_path, monkeypatch):
        """Opening a database that needs no repair must not execute any write statement.

        A write statement takes the write lock even when it changes nothing, so an
        unconditional INSERT OR IGNORE / UPDATE / marker stamp blocks every open behind
        a sibling process's transaction. The FTS5 capability probe on ``temp`` is exempt.
        """
        db_path = tmp_path / "state.db"
        StoreHolder(db_path=db_path).close()  # mints stamp + FTS layout marker
        StoreHolder(db_path=db_path).close()

        writes = []
        real_connect = sqlite3.connect

        def tracing_connect(*args, **kwargs):
            conn = real_connect(*args, **kwargs)
            conn.set_trace_callback(
                lambda stmt: writes.append(stmt)
                if re.match(r"\s*(INSERT|UPDATE|DELETE|REPLACE|ALTER|DROP\s+TRIGGER)\b", stmt, re.IGNORECASE) and "temp." not in stmt
                else None
            )
            return conn

        monkeypatch.setattr(sqlite3, "connect", tracing_connect)
        StoreHolder(db_path=db_path).close()
        assert writes == []

    def test_open_completes_while_sibling_holds_write_lock(self, tmp_path):
        """A settled read-write open must not wait on another connection's write transaction."""
        db_path = tmp_path / "state.db"
        StoreHolder(db_path=db_path).close()
        StoreHolder(db_path=db_path).close()

        holder = sqlite3.connect(db_path, timeout=60)
        holder.execute("BEGIN IMMEDIATE")
        holder.execute("UPDATE state_meta SET value = value WHERE key = 'nonexistent'")
        release = threading.Timer(4.0, holder.rollback)
        release.start()
        try:
            started = time.perf_counter()
            StoreHolder(db_path=db_path).close()
            elapsed = time.perf_counter() - started
        finally:
            release.cancel()
            with contextlib.suppress(sqlite3.Error):
                holder.rollback()
            holder.close()
        # Pre-fix this waited for the whole 4 s hold (retry loop around the busy timeout).
        assert elapsed < 2.0, f"open blocked on the write lock for {elapsed:.3f}s"

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

        session_db = StoreHolder(db_path=db_path)
        try:
            active = session_db._conn.execute(
                "SELECT active FROM messages WHERE content = 'old hidden turn'"
            ).fetchone()[0]
            assert active == 1
            assert len(get_messages_as_conversation(session_db, "s1")) == 1
        finally:
            session_db.close()

    def test_get_messages_as_conversation_strips_leaked_memory_context(self, db):
        create_session(db, session_id="s1", source="cli")
        append_message(db,
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

        conv = get_messages_as_conversation(db, "s1")
        assert len(conv) == 1
        assert conv[0]["role"] == "assistant"
        assert conv[0]["content"] == "Visible answer"
        assert isinstance(conv[0].get("timestamp"), float)

    def test_reasoning_persisted_and_restored(self, db):
        """Reasoning text is stored for assistant messages and restored by
        get_messages_as_conversation() so providers receive coherent multi-turn
        reasoning context."""
        create_session(db, session_id="s1", source="telegram")
        append_message(db, "s1", role="user", content="create a cron job")
        append_message(db,
            "s1",
            role="assistant",
            content=None,
            tool_calls=[{"function": {"name": "cronjob", "arguments": "{}"}, "id": "c1", "type": "function"}],
            reasoning="I should call the cronjob tool to schedule this.",
        )
        append_message(db, "s1", role="tool", content='{"job_id": "abc"}', tool_call_id="c1")

        conv = get_messages_as_conversation(db, "s1")
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
        create_session(db, session_id="s1", source="cli")
        ts = 1_234_567.0
        mid = append_message(db, "s1", role="user", content="hello",
                                timestamp=ts)
        msgs = get_messages(db, "s1")
        assert len(msgs) == 1
        assert msgs[0]["timestamp"] == ts
        assert msgs[0]["id"] == mid
        raw = self._raw_timestamps(db, "s1")
        assert raw == [ts]

    def test_replace_messages_preserves_timestamps(self, db):
        """Message dicts with ``timestamp`` passed to ``replace_messages``
        retain those timestamps after the rewrite."""
        create_session(db, session_id="s1", source="cli")
        msgs_in = [
            {"role": "user", "content": "first", "timestamp": 100.0},
            {"role": "assistant", "content": "second", "timestamp": 200.0},
            {"role": "user", "content": "third", "timestamp": 300.0},
        ]
        replace_messages(db, "s1", msgs_in)
        msgs_out = get_messages(db, "s1")
        assert [m["timestamp"] for m in msgs_out] == [100.0, 200.0, 300.0]
        assert self._raw_timestamps(db, "s1") == [100.0, 200.0, 300.0]

    def test_compression_replace_roundtrip_preserves_timestamps(self, db):
        """Compression-style rewrite: replace_messages with dicts loaded from
        get_messages_as_conversation must keep the surviving messages'
        original timestamps (#28841)."""
        timestamps = [1_500_000_000.0, 1_500_000_100.0, 1_500_000_200.0]
        create_session(db, session_id="s1", source="cli")
        for i, ts in enumerate(timestamps):
            append_message(db,
                "s1",
                role="user" if i % 2 == 0 else "assistant",
                content=f"msg-{i}",
                timestamp=ts,
            )

        history = get_messages_as_conversation(db, "s1")
        # Simulate a compression that keeps the last two turns verbatim and
        # prepends a fresh summary message (no timestamp — falls back to now).
        compressed = [{"role": "user", "content": "[summary]"}] + history[-2:]
        replace_messages(db, "s1", compressed)

        raw = self._raw_timestamps(db, "s1")
        assert len(raw) == 3
        assert raw[1:] == timestamps[-2:]
        assert raw[0] > timestamps[-1]  # summary stamped with a current time


# =========================================================================
# Counts
# =========================================================================

class TestCounts:

    def test_session_count_by_source(self, db):
        create_session(db, session_id="s1", source="cli")
        create_session(db, session_id="s2", source="telegram")
        create_session(db, session_id="s3", source="cli")
        assert session_count(db, source="cli") == 2
        assert session_count(db, source="telegram") == 1

    def test_session_count_ge_at_threshold(self, db):
        """session_count_ge should True when count >= n."""
        create_session(db, "s1", "cli")
        assert session_count_ge(db, 1) is True
        assert session_count_ge(db, 2) is False

        create_session(db, "s2", "telegram")
        assert session_count_ge(db, 1) is True
        assert session_count_ge(db, 2) is True
        assert session_count_ge(db, 3) is False

    def test_message_count_total(self, db):
        assert message_count(db) == 0
        create_session(db, session_id="s1", source="cli")
        append_message(db, "s1", role="user", content="Hello")
        append_message(db, "s1", role="assistant", content="Hi")
        assert message_count(db) == 2
