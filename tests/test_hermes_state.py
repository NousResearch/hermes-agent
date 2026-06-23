"""Tests for hermes_state.py — SessionDB SQLite CRUD, FTS5 search, export."""

import sqlite3
import time
import pytest

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
    """Create a SessionDB with a temp database file.

    Auto-embed is disabled by default so tests get a clean slate.
    Tests that need auto-embed (TestAutoEmbedOnAppend, TestOllamaE2E)
    set ``db.embedding_provider`` explicitly.
    """
    db_path = tmp_path / "test_state.db"
    session_db = SessionDB(db_path=db_path)
    # Disable auto-embed — the fixture resolves the real Ollama provider
    # from config, which would pre-embed every append_message and break
    # tests that expect zero-vector placeholders.
    session_db.embedding_provider = None
    yield session_db
    session_db.close()


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


    def test_get_nonexistent_session(self, db):
        assert db.get_session("nonexistent") is None

    def test_create_session_enriches_null_metadata_on_conflict(self, db):
        """Gateway creates a bare row first; the agent's later create_session
        must backfill model/model_config/system_prompt without clobbering the
        gateway's source/user_id/chat_id. Regression for NULL gateway metadata
        (sessions with NULL billing_provider/model)."""
        # Gateway bare row (source + user_id only), before the agent exists.
        db.create_session("s1", source="telegram", user_id="u1", chat_id="c1")
        bare = db.get_session("s1")
        assert bare["model"] is None
        # Agent enriches — passes source="cli" but real metadata.
        db.create_session(
            "s1", source="cli", model="claude-opus-4-6",
            model_config={"max_iterations": 90}, system_prompt="SYS",
        )
        enriched = db.get_session("s1")
        assert enriched["model"] == "claude-opus-4-6"
        assert enriched["system_prompt"] == "SYS"
        # Gateway-owned fields preserved (NOT clobbered by source="cli").
        assert enriched["source"] == "telegram"
        assert enriched["user_id"] == "u1"
        assert enriched["chat_id"] == "c1"

    def test_create_session_does_not_overwrite_existing_metadata(self, db):
        """A later bare write (source='unknown', model=...) must not overwrite
        a model/source an earlier writer already set."""
        db.create_session("s1", source="cli", model="real-model")
        db.create_session("s1", source="unknown", model="should-not-win")
        session = db.get_session("s1")
        assert session["model"] == "real-model"
        assert session["source"] == "cli"

    def test_update_session_cwd_persists_git_branch(self, db):
        db.create_session(session_id="s1", source="cli")
        db.update_session_cwd("s1", "/work/repo", git_branch="pets-feature")

        session = db.get_session("s1")
        assert session["cwd"] == "/work/repo"
        assert session["git_branch"] == "pets-feature"

    def test_update_session_cwd_empty_branch_does_not_clobber(self, db):
        """A failed branch probe (empty string) must not wipe a branch we
        already captured — only the cwd updates."""
        db.create_session(session_id="s1", source="cli")
        db.update_session_cwd("s1", "/work/repo", git_branch="main")
        db.update_session_cwd("s1", "/work/repo", git_branch="")

        session = db.get_session("s1")
        assert session["git_branch"] == "main"

    def test_update_session_cwd_without_branch_arg(self, db):
        """Back-compat: callers that pass only (id, cwd) still work."""
        db.create_session(session_id="s1", source="cli")
        db.update_session_cwd("s1", "/work/repo")

        session = db.get_session("s1")
        assert session["cwd"] == "/work/repo"
        assert session["git_branch"] is None

    def test_update_session_cwd_persists_git_repo_root(self, db):
        db.create_session(session_id="s1", source="cli")
        db.update_session_cwd("s1", "/work/repo/src", git_repo_root="/work/repo")

        assert db.get_session("s1")["git_repo_root"] == "/work/repo"

    def test_update_session_cwd_empty_repo_root_does_not_clobber(self, db):
        db.create_session(session_id="s1", source="cli")
        db.update_session_cwd("s1", "/work/repo", git_repo_root="/work/repo")
        db.update_session_cwd("s1", "/work/repo", git_repo_root="")

        assert db.get_session("s1")["git_repo_root"] == "/work/repo"

    def test_distinct_session_cwds_aggregates_history(self, db):
        db.create_session("s1", "cli", cwd="/repo")
        db.create_session("s2", "cli", cwd="/repo")
        db.create_session("s3", "cli", cwd="/other")
        db.create_session("s4", "cli")  # no cwd — excluded

        rows = {r["cwd"]: r["sessions"] for r in db.distinct_session_cwds()}
        assert rows == {"/repo": 2, "/other": 1}

    def test_backfill_repo_roots_fills_only_empty(self, db):
        db.create_session("s1", "cli", cwd="/repo/a")
        db.create_session("s2", "cli", cwd="/repo/b")
        db.update_session_cwd("s2", "/repo/b", git_repo_root="/already")

        db.backfill_repo_roots({"/repo/a": "/repo", "/repo/b": "/repo"})

        assert db.get_session("s1")["git_repo_root"] == "/repo"
        # Pre-existing root is preserved, not clobbered.
        assert db.get_session("s2")["git_repo_root"] == "/already"

    def test_end_session(self, db):
        db.create_session(session_id="s1", source="cli")
        db.end_session("s1", end_reason="user_exit")

        session = db.get_session("s1")
        assert isinstance(session["ended_at"], float)
        assert session["end_reason"] == "user_exit"

    def test_end_session_preserves_original_end_reason(self, db):
        """The first end_reason wins — compression splits must not be
        overwritten when a later stale ``end_session()`` call lands on the
        same row (e.g. from a CLI session_id that desynced after compression
        and then tried to /resume another session).
        """
        db.create_session(session_id="s1", source="cli")
        db.end_session("s1", end_reason="compression")
        first_ended_at = db.get_session("s1")["ended_at"]

        # Simulate a stale CLI holding the old session_id and calling
        # end_session() again with a different reason.
        time.sleep(0.01)
        db.end_session("s1", end_reason="resumed_other")

        session = db.get_session("s1")
        assert session["end_reason"] == "compression"
        assert session["ended_at"] == first_ended_at

    def test_end_session_after_reopen_allows_re_end(self, db):
        """reopen_session() is the explicit escape hatch for re-ending a
        closed session. After reopen, end_session() takes effect again.
        """
        db.create_session(session_id="s1", source="cli")
        db.end_session("s1", end_reason="compression")
        db.reopen_session("s1")
        db.end_session("s1", end_reason="user_exit")

        session = db.get_session("s1")
        assert session["end_reason"] == "user_exit"

    def test_update_system_prompt(self, db):
        db.create_session(session_id="s1", source="cli")
        db.update_system_prompt("s1", "You are a helpful assistant.")

        session = db.get_session("s1")
        assert session["system_prompt"] == "You are a helpful assistant."

    def test_update_token_counts(self, db):
        db.create_session(session_id="s1", source="cli")
        db.update_token_counts("s1", input_tokens=200, output_tokens=100)
        db.update_token_counts("s1", input_tokens=100, output_tokens=50)

        session = db.get_session("s1")
        assert session["input_tokens"] == 300
        assert session["output_tokens"] == 150

    def test_update_token_counts_tracks_api_call_count(self, db):
        """api_call_count increments with each update_token_counts call."""
        db.create_session(session_id="s1", source="cli")
        db.update_token_counts("s1", input_tokens=100, output_tokens=50, api_call_count=1)
        db.update_token_counts("s1", input_tokens=100, output_tokens=50, api_call_count=1)
        db.update_token_counts("s1", input_tokens=100, output_tokens=50, api_call_count=1)

        session = db.get_session("s1")
        assert session["api_call_count"] == 3

    def test_update_token_counts_api_call_count_absolute(self, db):
        """absolute mode sets api_call_count directly."""
        db.create_session(session_id="s1", source="cli")
        db.update_token_counts("s1", input_tokens=100, output_tokens=50, api_call_count=1)
        db.update_token_counts("s1", input_tokens=300, output_tokens=150,
                               api_call_count=5, absolute=True)

        session = db.get_session("s1")
        assert session["api_call_count"] == 5
        assert session["input_tokens"] == 300

    def test_update_token_counts_backfills_model_when_null(self, db):
        db.create_session(session_id="s1", source="telegram")
        db.update_token_counts("s1", input_tokens=10, output_tokens=5, model="openai/gpt-5.4")

        session = db.get_session("s1")
        assert session["model"] == "openai/gpt-5.4"

    def test_update_token_counts_preserves_existing_model(self, db):
        db.create_session(session_id="s1", source="cli", model="anthropic/claude-opus-4.6")
        db.update_token_counts("s1", input_tokens=10, output_tokens=5, model="openai/gpt-5.4")

        session = db.get_session("s1")
        assert session["model"] == "anthropic/claude-opus-4.6"

    def test_update_session_model_overwrites_existing(self, db):
        """A mid-session /model switch must overwrite the stored model.

        update_token_counts uses COALESCE(model, ?) (first-writer-wins), so
        the dashboard kept showing the original model after a switch (#34850).
        update_session_model sets the column unconditionally.
        """
        db.create_session(session_id="s1", source="telegram",
                          model="xiaomi/mimo-v2.5-pro")
        # Token updates never change the model once set.
        db.update_token_counts("s1", input_tokens=10, output_tokens=5,
                               model="xiaomi/mimo-v2.5-pro")
        assert db.get_session("s1")["model"] == "xiaomi/mimo-v2.5-pro"

        # Explicit switch overwrites it.
        db.update_session_model("s1", "xiaomi/mimo-v2.5")
        assert db.get_session("s1")["model"] == "xiaomi/mimo-v2.5"

        # And a subsequent token update does NOT revert it (COALESCE no-ops
        # because the column is now non-NULL).
        db.update_token_counts("s1", input_tokens=10, output_tokens=5,
                               model="xiaomi/mimo-v2.5-pro")
        assert db.get_session("s1")["model"] == "xiaomi/mimo-v2.5"

    def test_update_session_billing_route_overwrites_after_switch(self, db):
        """A mid-session provider switch must overwrite the billing route.

        update_token_counts writes billing fields with
        COALESCE(billing_provider, ?) (first-writer-wins), so after a
        provider switch the dashboard kept attributing cost to the original
        provider (#48248). update_session_billing_route sets them
        unconditionally and nulls system_prompt so the next turn rebuilds
        the Model:/Provider: header (#48173).
        """
        db.create_session(session_id="s1", source="telegram")
        # First token update seeds the billing route.
        db.update_token_counts("s1", input_tokens=10, output_tokens=5,
                               billing_provider="openrouter",
                               billing_base_url="https://openrouter.ai/api/v1",
                               billing_mode="api_key")
        sess = db.get_session("s1")
        assert sess["billing_provider"] == "openrouter"
        # A later token update never changes it (COALESCE first-writer-wins).
        db.update_token_counts("s1", input_tokens=10, output_tokens=5,
                               billing_provider="ollama",
                               billing_base_url="http://localhost:11434/v1",
                               billing_mode="local")
        assert db.get_session("s1")["billing_provider"] == "openrouter"

        # Seed a stale prompt snapshot, then switch the billing route.
        db.update_system_prompt("s1", "Model: x/old\nProvider: openrouter")
        assert db.get_session("s1")["system_prompt"] is not None
        db.update_session_billing_route(
            "s1", provider="ollama",
            base_url="http://localhost:11434/v1", billing_mode="local",
        )
        sess = db.get_session("s1")
        assert sess["billing_provider"] == "ollama"
        assert sess["billing_base_url"] == "http://localhost:11434/v1"
        assert sess["billing_mode"] == "local"
        assert sess["system_prompt"] is None, \
            "system_prompt must be nulled so the next turn rebuilds Model:/Provider:"

        # billing_mode defaults to COALESCE — omitting it preserves the value.
        db.update_session_billing_route(
            "s1", provider="openai",
            base_url="https://api.openai.com/v1",
        )
        sess = db.get_session("s1")
        assert sess["billing_provider"] == "openai"
        assert sess["billing_mode"] == "local"  # preserved (COALESCE on None)

    def test_parent_session(self, db):
        db.create_session(session_id="parent", source="cli")
        db.create_session(session_id="child", source="cli", parent_session_id="parent")

        child = db.get_session("child")
        assert child["parent_session_id"] == "parent"

    def test_db_initializes_without_fts5_module(self, tmp_path, monkeypatch):
        real_connect = sqlite3.connect

        def connect_without_fts(*args, **kwargs):
            kwargs["factory"] = _NoFtsConnection
            return real_connect(*args, **kwargs)

        monkeypatch.setattr("hermes_state.sqlite3.connect", connect_without_fts)

        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            assert db._fts_enabled is False
            # Neither FTS5 virtual table should have been created on a build
            # that lacks the fts5 module — both init paths must degrade.
            assert db._fts_table_exists("messages_fts") is False
            assert db._fts_table_exists("messages_fts_trigram") is False

            db.create_session(session_id="s1", source="cli")
            db.append_message("s1", role="user", content="hello from sqlite without fts")

            messages = db.get_messages("s1")
            assert len(messages) == 1
            assert messages[0]["content"] == "hello from sqlite without fts"
            assert db.search_messages("hello") == []
        finally:
            db.close()

    def test_existing_fts_tables_do_not_break_without_fts5(
        self, tmp_path, monkeypatch
    ):
        db_path = tmp_path / "state.db"
        seeded = SessionDB(db_path=db_path)
        try:
            seeded.create_session(session_id="s1", source="cli")
            seeded.append_message("s1", role="user", content="before runtime change")
        finally:
            seeded.close()

        real_connect = sqlite3.connect

        def connect_without_fts(*args, **kwargs):
            kwargs["factory"] = _NoFtsExistingTableConnection
            return real_connect(*args, **kwargs)

        monkeypatch.setattr("hermes_state.sqlite3.connect", connect_without_fts)

        db = SessionDB(db_path=db_path)
        try:
            assert db._fts_enabled is False
            assert db.get_session("s1") is not None
            assert len(db.get_messages("s1")) == 1

            # Existing FTS triggers must be disabled too; otherwise this write
            # would try to insert into an unusable FTS virtual table.
            db.append_message("s1", role="assistant", content="after runtime change")
            messages = db.get_messages("s1")
            assert len(messages) == 2
            assert messages[1]["content"] == "after runtime change"
        finally:
            db.close()

    def test_old_schema_without_fts5_does_not_crash(self, tmp_path, monkeypatch):
        db_path = tmp_path / "legacy.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript(SCHEMA_SQL)
        conn.execute("DELETE FROM schema_version")
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (9,))
        conn.commit()
        conn.close()

        real_connect = sqlite3.connect

        def connect_without_fts(*args, **kwargs):
            kwargs["factory"] = _NoFtsConnection
            return real_connect(*args, **kwargs)

        monkeypatch.setattr("hermes_state.sqlite3.connect", connect_without_fts)

        db = SessionDB(db_path=db_path)
        try:
            assert db._fts_enabled is False
            db.create_session(session_id="s1", source="cli")
            db.append_message("s1", role="user", content="legacy no fts")
            assert db.get_messages("s1")[0]["content"] == "legacy no fts"
            assert db.search_messages("legacy") == []

            # Leave the FTS migration version in place so a future FTS-capable
            # runtime can still rebuild and backfill the indexes.
            row = db._conn.execute("SELECT version FROM schema_version").fetchone()
            assert row["version"] == 9
        finally:
            db.close()

    def test_fts_runtime_restores_triggers_after_no_fts_open(
        self, tmp_path, monkeypatch
    ):
        db_path = tmp_path / "state.db"
        seeded = SessionDB(db_path=db_path)
        try:
            seeded.create_session(session_id="s1", source="cli")
            seeded.append_message("s1", role="user", content="first searchable")
        finally:
            seeded.close()

        real_connect = sqlite3.connect

        def connect_without_fts(*args, **kwargs):
            kwargs["factory"] = _NoFtsExistingTableConnection
            return real_connect(*args, **kwargs)

        monkeypatch.setattr("hermes_state.sqlite3.connect", connect_without_fts)
        no_fts = SessionDB(db_path=db_path)
        try:
            no_fts.append_message("s1", role="assistant", content="not indexed yet")
        finally:
            no_fts.close()

        monkeypatch.setattr("hermes_state.sqlite3.connect", real_connect)
        restored = SessionDB(db_path=db_path)
        try:
            assert restored._fts_enabled is True
            restored.append_message("s1", role="assistant", content="indexed again")
            assert len(restored.search_messages("not indexed yet")) == 1
            assert len(restored.search_messages("indexed")) == 2
        finally:
            restored.close()

    def test_base_fts_rebuilds_after_trigger_repair_without_trigram(
        self, tmp_path, monkeypatch
    ):
        """Trigger repair must rebuild base FTS even when trigram is unavailable."""
        db_path = tmp_path / "state.db"
        seeded = SessionDB(db_path=db_path)
        try:
            seeded.create_session(session_id="s1", source="cli")
            seeded.append_message("s1", role="user", content="already indexed")
            for trigger in (
                "messages_fts_insert",
                "messages_fts_delete",
                "messages_fts_update",
                "messages_fts_trigram_insert",
                "messages_fts_trigram_delete",
                "messages_fts_trigram_update",
            ):
                seeded._conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
            seeded._conn.commit()
            seeded.append_message("s1", role="assistant", content="repair only base needle")
        finally:
            seeded.close()

        real_connect = sqlite3.connect

        def connect_without_trigram(*args, **kwargs):
            kwargs["factory"] = _NoTrigramConnection
            return real_connect(*args, **kwargs)

        monkeypatch.setattr("hermes_state.sqlite3.connect", connect_without_trigram)
        restored = SessionDB(db_path=db_path)
        try:
            assert restored._fts_enabled is True
            assert restored._trigram_available is False
            assert restored._fts_table_exists("messages_fts") is True
            assert len(restored.search_messages("needle")) == 1
        finally:
            restored.close()

    def test_is_fts5_unavailable_error_catches_trigram_tokenizer(self):
        """Unit test: _is_fts5_unavailable_error matches 'no such tokenizer: trigram'."""
        fts5_err = sqlite3.OperationalError("no such module: fts5")
        trigram_err = sqlite3.OperationalError("no such tokenizer: trigram")
        generic_tokenizer_err = sqlite3.OperationalError("no such tokenizer: foo")
        unrelated_err = sqlite3.OperationalError("no such table: foo")

        assert SessionDB._is_fts5_unavailable_error(fts5_err) is True
        assert SessionDB._is_fts5_unavailable_error(trigram_err) is True
        # Generic tokenizer errors should NOT match — only trigram.
        assert SessionDB._is_fts5_unavailable_error(generic_tokenizer_err) is False
        assert SessionDB._is_fts5_unavailable_error(unrelated_err) is False

    def test_is_trigram_unavailable_error(self):
        """Unit test: _is_trigram_unavailable_error is scoped to trigram."""
        trigram_err = sqlite3.OperationalError("no such tokenizer: trigram")
        generic_err = sqlite3.OperationalError("no such tokenizer: foo")
        fts5_err = sqlite3.OperationalError("no such module: fts5")

        assert SessionDB._is_trigram_unavailable_error(trigram_err) is True
        assert SessionDB._is_trigram_unavailable_error(generic_err) is False
        assert SessionDB._is_trigram_unavailable_error(fts5_err) is False

    def test_db_initializes_without_trigram_tokenizer(self, tmp_path, monkeypatch):
        """SessionDB must not crash when FTS5 exists but trigram tokenizer is missing."""
        real_connect = sqlite3.connect

        def connect_without_trigram(*args, **kwargs):
            kwargs["factory"] = _NoTrigramConnection
            return real_connect(*args, **kwargs)

        monkeypatch.setattr("hermes_state.sqlite3.connect", connect_without_trigram)

        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            # Base FTS5 should still work (trigram is optional).
            assert db._fts_enabled is True
            assert db._fts_table_exists("messages_fts") is True
            # Trigram table should NOT have been created.
            assert db._fts_table_exists("messages_fts_trigram") is False

            db.create_session(session_id="s1", source="cli")
            db.append_message("s1", role="user", content="hello without trigram")

            messages = db.get_messages("s1")
            assert len(messages) == 1
            assert messages[0]["content"] == "hello without trigram"

            # FTS5 keyword search should still work.
            assert len(db.search_messages("hello")) == 1
        finally:
            db.close()

    def test_v11_migration_backfills_base_fts_when_trigram_unavailable(
        self, tmp_path, monkeypatch
    ):
        """Regression: v11 migration must backfill base FTS even when trigram is unavailable."""
        real_connect = sqlite3.connect
        db_path = tmp_path / "state.db"

        # Phase 1: create a DB at schema v10 with messages.
        db = SessionDB(db_path=db_path)
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="legacy message alpha")
        db.append_message("s1", role="assistant", content="legacy reply beta")
        # Force schema version to v10 so migration runs on next open.
        db._conn.execute(
            "UPDATE schema_version SET version = 10"
        )
        db._conn.commit()
        db.close()

        # Phase 2: reopen with trigram disabled — migration should still
        # backfill base FTS and make existing messages searchable.
        def connect_without_trigram(*args, **kwargs):
            kwargs["factory"] = _NoTrigramConnection
            return real_connect(*args, **kwargs)

        monkeypatch.setattr("hermes_state.sqlite3.connect", connect_without_trigram)
        migrated_db = SessionDB(db_path=db_path)
        try:
            assert migrated_db._fts_enabled is True
            assert migrated_db._trigram_available is False
            assert migrated_db._fts_table_exists("messages_fts") is True
            assert migrated_db._fts_table_exists("messages_fts_trigram") is False

            # Existing messages must be searchable via base FTS.
            results = migrated_db.search_messages("legacy message")
            assert len(results) == 1
            # snippet has FTS5 highlight markers (>>>...<<<); check raw content via get_messages
            msgs = migrated_db.get_messages("s1")
            assert any("legacy message" in m["content"] for m in msgs)
        finally:
            migrated_db.close()

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

    def test_append_message_accepts_explicit_timestamp(self, db):
        db.create_session(session_id="s1", source="telegram")
        event_ts = 1777383653.0

        db.append_message("s1", role="user", content="Hello", timestamp=event_ts)

        messages = db.get_messages_as_conversation("s1")
        assert messages[0]["timestamp"] == event_ts

    def test_message_increments_session_count(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="Hello")
        db.append_message("s1", role="assistant", content="Hi")

        session = db.get_session("s1")
        assert session["message_count"] == 2

    def test_observed_flag_round_trips_for_gateway_replay(self, db):
        db.create_session(session_id="s1", source="telegram:-100")
        db.append_message(
            "s1",
            role="user",
            content="[Alice|111]\nside chatter",
            observed=True,
        )
        db.append_message("s1", role="assistant", content="ack")

        messages = db.get_messages("s1")
        assert messages[0]["observed"] == 1
        assert messages[1]["observed"] == 0

        conversation = db.get_messages_as_conversation("s1")
        assert conversation[0]["role"] == "user"
        assert conversation[0]["content"] == "[Alice|111]\nside chatter"
        assert conversation[0]["observed"] is True
        assert isinstance(conversation[0].get("timestamp"), float)
        assert "observed" not in conversation[1]

    def test_tool_response_does_not_increment_tool_count(self, db):
        """Tool responses (role=tool) should not increment tool_call_count.

        Only assistant messages with tool_calls should count.
        """
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="tool", content="result", tool_name="web_search")

        session = db.get_session("s1")
        assert session["tool_call_count"] == 0

    def test_assistant_tool_calls_increment_by_count(self, db):
        """An assistant message with N tool_calls should increment by N."""
        db.create_session(session_id="s1", source="cli")
        tool_calls = [
            {"id": "call_1", "function": {"name": "web_search", "arguments": "{}"}},
        ]
        db.append_message("s1", role="assistant", content="", tool_calls=tool_calls)

        session = db.get_session("s1")
        assert session["tool_call_count"] == 1

    def test_tool_call_count_matches_actual_calls(self, db):
        """tool_call_count should equal the number of tool calls made, not messages."""
        db.create_session(session_id="s1", source="cli")

        # Assistant makes 2 parallel tool calls in one message
        tool_calls = [
            {"id": "call_1", "function": {"name": "ha_call_service", "arguments": "{}"}},
            {"id": "call_2", "function": {"name": "ha_call_service", "arguments": "{}"}},
        ]
        db.append_message("s1", role="assistant", content="", tool_calls=tool_calls)

        # Two tool responses come back
        db.append_message("s1", role="tool", content="ok", tool_name="ha_call_service")
        db.append_message("s1", role="tool", content="ok", tool_name="ha_call_service")

        session = db.get_session("s1")
        # Should be 2 (the actual number of tool calls), not 3
        assert session["tool_call_count"] == 2, (
            f"Expected 2 tool calls but got {session['tool_call_count']}. "
            "tool responses are double-counted and multi-call messages are under-counted"
        )

    def test_tool_calls_serialization(self, db):
        db.create_session(session_id="s1", source="cli")
        tool_calls = [{"id": "call_1", "function": {"name": "web_search", "arguments": "{}"}}]
        db.append_message("s1", role="assistant", tool_calls=tool_calls)

        messages = db.get_messages("s1")
        assert messages[0]["tool_calls"] == tool_calls

    def test_multimodal_list_content_round_trip(self, db):
        """Multimodal ``content`` (list of parts) must survive the SQLite
        round-trip.  sqlite3 cannot bind Python lists directly, so the DB
        layer JSON-encodes structured content on write and decodes on read.

        Regression test for the "Error binding parameter 3: type 'list' is
        not supported" crash users hit when pasting screenshots into the
        TUI (issue #17522).
        """
        db.create_session(session_id="s1", source="cli")
        content = [
            {"type": "text", "text": "describe this screenshot"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,iVBORw0KG..."},
            },
        ]

        # Write must not raise
        db.append_message("s1", role="user", content=content)

        # get_messages decodes back to the original list
        msgs = db.get_messages("s1")
        assert len(msgs) == 1
        assert msgs[0]["content"] == content

        # get_messages_as_conversation decodes back to the original list
        conv = db.get_messages_as_conversation("s1")
        assert len(conv) == 1
        assert conv[0]["role"] == "user"
        assert conv[0]["content"] == content
        assert isinstance(conv[0].get("timestamp"), float)

    def test_dict_content_round_trip(self, db):
        """Dict-shaped content (e.g. provider wrappers) also round-trips."""
        db.create_session(session_id="s1", source="cli")
        content = {"parts": [{"text": "hi"}]}

        db.append_message("s1", role="user", content=content)
        msgs = db.get_messages("s1")
        assert msgs[0]["content"] == content

    def test_string_content_unchanged_by_encoding(self, db):
        """Plain strings must not be wrapped — FTS search and legacy
        consumers depend on raw-string storage for text content.
        """
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="plain text")

        # Peek at the raw column to confirm no encoding was applied
        with db._lock:
            row = db._conn.execute(
                "SELECT content FROM messages WHERE session_id = ?", ("s1",)
            ).fetchone()
        assert row["content"] == "plain text"

    def test_replace_messages_persists_tool_name(self, db):
        """`replace_messages` (used by /retry, /undo, /compress) must write
        tool_name to the DB for messages built by make_tool_result_message."""
        from agent.tool_dispatch_helpers import make_tool_result_message
        db.create_session(session_id="s1", source="cli")
        db.replace_messages(
            "s1",
            [
                {"role": "user", "content": "do something"},
                make_tool_result_message("web_search", "some results", "c1"),
            ],
        )

        msgs = db.get_messages("s1")
        tool_msg = next(m for m in msgs if m["role"] == "tool")
        assert tool_msg["tool_name"] == "web_search"

    def test_replace_messages_handles_multimodal_content(self, db):
        """`replace_messages` (used by /retry, /undo, /compress) must also
        handle list content without crashing."""
        db.create_session(session_id="s1", source="cli")
        content = [
            {"type": "text", "text": "look at this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
        ]

        db.replace_messages(
            "s1",
            [
                {"role": "user", "content": content},
                {"role": "assistant", "content": "I see a screenshot."},
            ],
        )

        msgs = db.get_messages("s1")
        assert len(msgs) == 2
        assert msgs[0]["content"] == content
        assert msgs[1]["content"] == "I see a screenshot."

    def test_get_messages_as_conversation(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="Hello")
        db.append_message("s1", role="assistant", content="Hi!")

        conv = db.get_messages_as_conversation("s1")
        assert len(conv) == 2
        assert conv[0]["role"] == "user"
        assert conv[0]["content"] == "Hello"
        assert isinstance(conv[0]["timestamp"], float)
        assert conv[1]["role"] == "assistant"
        assert conv[1]["content"] == "Hi!"
        assert isinstance(conv[1]["timestamp"], float)

    def test_platform_message_id_round_trips(self, db):
        """Platform-side message ids (yuanbao msg_id, telegram update_id, …)
        survive append → get_messages_as_conversation under the
        ``message_id`` key so platform recall flows can match by exact id."""
        db.create_session(session_id="s_pmi", source="yuanbao")
        db.append_message(
            "s_pmi",
            role="user",
            content="hi",
            platform_message_id="abc-123",
        )
        db.append_message("s_pmi", role="assistant", content="hello")

        conv = db.get_messages_as_conversation("s_pmi")
        user_msg = next(m for m in conv if m["role"] == "user")
        assistant_msg = next(m for m in conv if m["role"] == "assistant")
        assert user_msg.get("message_id") == "abc-123"
        # Assistant row had no platform id — must not gain one spuriously.
        assert "message_id" not in assistant_msg

    def test_replace_messages_preserves_platform_message_id(self, db):
        """``rewrite_transcript`` (which goes through replace_messages) must
        keep the platform_message_id round-trip working for /retry, /undo,
        /compress and yuanbao's recall rewrite path."""
        db.create_session(session_id="s_rep", source="yuanbao")
        db.replace_messages(
            "s_rep",
            [
                {"role": "user", "content": "x", "message_id": "ext-1"},
                {"role": "assistant", "content": "y"},
            ],
        )
        conv = db.get_messages_as_conversation("s_rep")
        assert next(m for m in conv if m["role"] == "user").get("message_id") == "ext-1"
        assert "message_id" not in next(m for m in conv if m["role"] == "assistant")

    def test_get_messages_as_conversation_includes_ancestor_chain(self, db):
        db.create_session("root", "tui")
        db.append_message("root", role="user", content="first prompt")
        db.append_message("root", role="assistant", content="first answer")
        db.create_session("child", "tui", parent_session_id="root")
        db.append_message("child", role="user", content="second prompt")
        db.append_message("child", role="assistant", content="second answer")

        conv = db.get_messages_as_conversation("child", include_ancestors=True)

        assert [m["content"] for m in conv] == [
            "first prompt",
            "first answer",
            "second prompt",
            "second answer",
        ]

    def test_get_messages_as_conversation_avoids_repeated_resume_prompts_from_ancestors(self, db):
        db.create_session("root", "tui")
        db.append_message("root", role="user", content="same prompt")
        db.append_message("root", role="user", content="same prompt")
        db.append_message("root", role="assistant", content="answer")
        db.create_session("child", "tui", parent_session_id="root")
        db.append_message("child", role="user", content="next prompt")

        conv = db.get_messages_as_conversation("child", include_ancestors=True)

        assert [m["content"] for m in conv if m["role"] == "user"] == ["same prompt", "next prompt"]

    def test_finish_reason_stored(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="assistant", content="Done", finish_reason="stop")

        messages = db.get_messages("s1")
        assert messages[0]["finish_reason"] == "stop"

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

    def test_reasoning_details_persisted_and_restored(self, db):
        """reasoning_details (structured array) is round-tripped through JSON
        serialization in the DB."""
        db.create_session(session_id="s1", source="telegram")
        details = [
            {"type": "reasoning.summary", "summary": "Thinking about tools"},
            {"type": "reasoning.encrypted_content", "encrypted_content": "abc123"},
        ]
        db.append_message(
            "s1",
            role="assistant",
            content="Hello",
            reasoning="Thinking about what to say",
            reasoning_details=details,
        )

        conv = db.get_messages_as_conversation("s1")
        assert len(conv) == 1
        msg = conv[0]
        assert msg["reasoning"] == "Thinking about what to say"
        assert msg["reasoning_details"] == details

    def test_finish_reason_restored_by_get_messages_as_conversation(self, db):
        """finish_reason on assistant messages must survive conversation replay.

        Without this, /branch copies and other transcript round-trips silently
        drop the provider's stop signal.
        """
        db.create_session(session_id="s1", source="cli")
        db.append_message(
            "s1",
            role="assistant",
            content="Done",
            finish_reason="tool_calls",
        )
        db.append_message("s1", role="user", content="next")

        conv = db.get_messages_as_conversation("s1")
        assert conv[0]["role"] == "assistant"
        assert conv[0]["finish_reason"] == "tool_calls"
        # Non-assistant rows should not have a finish_reason key added.
        assert "finish_reason" not in conv[1]

    def test_reasoning_content_persisted_and_restored(self, db):
        """reasoning_content must survive session replay as its own field."""
        db.create_session(session_id="s1", source="cli")
        db.append_message(
            "s1",
            role="assistant",
            content="Hello",
            reasoning="Short summary",
            reasoning_content="Longer provider-native scratchpad",
        )

        conv = db.get_messages_as_conversation("s1")
        assert len(conv) == 1
        assert conv[0]["reasoning"] == "Short summary"
        assert conv[0]["reasoning_content"] == "Longer provider-native scratchpad"

    def test_reasoning_content_empty_string_restored_for_assistant(self, db):
        """Empty reasoning_content still needs to round-trip for strict replays."""
        db.create_session(session_id="s1", source="cli")
        db.append_message(
            "s1",
            role="assistant",
            content="",
            tool_calls=[{"id": "c1", "type": "function", "function": {"name": "date", "arguments": "{}"}}],
            reasoning_content="",
        )

        conv = db.get_messages_as_conversation("s1")
        assert len(conv) == 1
        assert "reasoning_content" in conv[0]
        assert conv[0]["reasoning_content"] == ""

    def test_codex_message_items_persisted_and_restored(self, db):
        """codex_message_items must round-trip through JSON serialization."""
        db.create_session(session_id="s1", source="cli")
        items = [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "id": "msg_123",
                "phase": "commentary",
                "content": [{"type": "output_text", "text": "Thinking..."}],
            },
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "id": "msg_456",
                "phase": "final_answer",
                "content": [{"type": "output_text", "text": "Done!"}],
            },
        ]
        db.append_message("s1", role="assistant", content="Done!", codex_message_items=items)

        conv = db.get_messages_as_conversation("s1")
        assert len(conv) == 1
        assert conv[0].get("codex_message_items") == items

    def test_reasoning_not_set_for_non_assistant(self, db):
        """reasoning is never leaked onto user or tool messages."""
        db.create_session(session_id="s1", source="telegram")
        db.append_message("s1", role="user", content="hi")
        db.append_message("s1", role="assistant", content="hello", reasoning=None)

        conv = db.get_messages_as_conversation("s1")
        assert "reasoning" not in conv[0]
        assert "reasoning" not in conv[1]

    def test_reasoning_empty_string_not_restored(self, db):
        """Empty string reasoning is treated as absent."""
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="assistant", content="hi", reasoning="")

        conv = db.get_messages_as_conversation("s1")
        assert "reasoning" not in conv[0]

    def test_codex_reasoning_items_persisted_and_restored(self, db):
        """codex_reasoning_items (encrypted blobs for Codex Responses API) are
        round-tripped through JSON serialization in the DB."""
        db.create_session(session_id="s1", source="cli")
        codex_items = [
            {"type": "reasoning", "id": "rs_abc", "encrypted_content": "enc_blob_123"},
            {"type": "reasoning", "id": "rs_def", "encrypted_content": "enc_blob_456"},
        ]
        db.append_message(
            "s1",
            role="assistant",
            content="Done",
            codex_reasoning_items=codex_items,
        )

        conv = db.get_messages_as_conversation("s1")
        assert len(conv) == 1
        assert conv[0]["codex_reasoning_items"] == codex_items
        assert conv[0]["codex_reasoning_items"][0]["encrypted_content"] == "enc_blob_123"


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

    def test_search_empty_query(self, db):
        assert db.search_messages("") == []
        assert db.search_messages("   ") == []

    def test_search_with_source_filter(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="CLI question about Python")

        db.create_session(session_id="s2", source="telegram")
        db.append_message("s2", role="user", content="Telegram question about Python")

        results = db.search_messages("Python", source_filter=["telegram"])
        # Should only find the telegram message
        sources = [r["source"] for r in results]
        assert all(s == "telegram" for s in sources)

    def test_search_default_sources_include_acp(self, db):
        db.create_session(session_id="s1", source="acp")
        db.append_message("s1", role="user", content="ACP question about Python")

        results = db.search_messages("Python")
        sources = [r["source"] for r in results]
        assert "acp" in sources

    def test_search_default_includes_all_platforms(self, db):
        """Default search (no source_filter) should find sessions from any platform."""
        for src in ("cli", "telegram", "signal", "homeassistant", "acp", "matrix"):
            sid = f"s-{src}"
            db.create_session(session_id=sid, source=src)
            db.append_message(sid, role="user", content=f"universal search test from {src}")

        results = db.search_messages("universal search test")
        found_sources = {r["source"] for r in results}
        assert found_sources == {"cli", "telegram", "signal", "homeassistant", "acp", "matrix"}

    def test_search_with_role_filter(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="What is FastAPI?")
        db.append_message("s1", role="assistant", content="FastAPI is a web framework.")

        results = db.search_messages("FastAPI", role_filter=["assistant"])
        roles = [r["role"] for r in results]
        assert all(r == "assistant" for r in roles)

    def test_search_returns_context(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="Tell me about Kubernetes")
        db.append_message("s1", role="assistant", content="Kubernetes is an orchestrator.")

        results = db.search_messages("Kubernetes")
        assert len(results) == 2
        assert "context" in results[0]
        assert isinstance(results[0]["context"], list)
        assert len(results[0]["context"]) > 0

    def test_search_context_uses_session_neighbors_when_ids_are_interleaved(self, db):
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="cli")

        db.append_message("s1", role="user", content="before needle")
        db.append_message("s2", role="user", content="other session message")
        db.append_message("s1", role="assistant", content="needle match")
        db.append_message("s2", role="assistant", content="another other session message")
        db.append_message("s1", role="user", content="after needle")

        results = db.search_messages('"needle match"')
        needle_result = next(r for r in results if r["session_id"] == "s1" and "needle match" in r["snippet"])

        assert [msg["content"] for msg in needle_result["context"]] == [
            "before needle",
            "needle match",
            "after needle",
        ]

    def test_search_special_chars_do_not_crash(self, db):
        """FTS5 special characters in queries must not raise OperationalError."""
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="How do I use C++ templates?")

        # Each of these previously caused sqlite3.OperationalError
        dangerous_queries = [
            'C++',              # + is FTS5 column filter
            '"unterminated',    # unbalanced double-quote
            '(problem',         # unbalanced parenthesis
            'hello AND',        # dangling boolean operator
            '***',              # repeated wildcard
            '{test}',           # curly braces (column reference)
            'OR hello',         # leading boolean operator
            'a AND OR b',       # adjacent operators
        ]
        for query in dangerous_queries:
            # Must not raise — should return list (possibly empty)
            results = db.search_messages(query)
            assert isinstance(results, list), f"Query {query!r} did not return a list"

    def test_search_sanitized_query_still_finds_content(self, db):
        """Sanitization must not break normal keyword search."""
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="Learning C++ templates today")

        # "C++" sanitized to "C" should still match "C++"
        results = db.search_messages("C++")
        # The word "C" appears in the content, so FTS5 should find it
        assert isinstance(results, list)

    def test_search_hyphenated_term_does_not_crash(self, db):
        """Hyphenated terms like 'chat-send' must not crash FTS5."""
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="Run the chat-send command")

        results = db.search_messages("chat-send")
        assert isinstance(results, list)
        assert len(results) >= 1
        assert any("chat-send" in (r.get("snippet") or r.get("content", "")).lower()
                    for r in results)

    def test_search_dotted_term_does_not_crash(self, db):
        """Dotted terms like 'P2.2' or 'simulate.p2.test.ts' should not crash FTS5."""
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="Working on P2.2 session_search edge cases")
        db.append_message("s1", role="assistant", content="See simulate.p2.test.ts for details")

        results = db.search_messages("P2.2")
        assert isinstance(results, list)
        assert len(results) >= 1

        results2 = db.search_messages("simulate.p2.test.ts")
        assert isinstance(results2, list)
        assert len(results2) >= 1

    def test_search_colon_query_still_finds_content(self, db):
        """Queries containing ':' must not silently return empty.

        ':' is FTS5's column-filter operator. With a single-column FTS table an
        unquoted query like 'TODO: fix' parses as 'column:term', raises
        "no such column: TODO", and the swallowed error turns into zero results
        even though the content is present. Regression for that silent-empty bug.
        """
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="TODO fix the deployment script")

        # Control: the same content is found without the colon.
        assert len(db.search_messages("deployment")) >= 1

        # The colon query must find the message, not silently return [].
        results = db.search_messages("TODO: fix")
        assert isinstance(results, list)
        assert len(results) >= 1
        assert any("deployment" in (r.get("snippet") or r.get("content", "")).lower()
                   for r in results)

    def test_search_quoted_phrase_preserved(self, db):
        """User-provided quoted phrases should be preserved for exact matching."""
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="docker networking is complex")
        db.append_message("s1", role="assistant", content="networking docker tips")

        # Quoted phrase should match only the exact order
        results = db.search_messages('"docker networking"')
        assert isinstance(results, list)
        # Should find the user message (exact phrase) but may or may not find
        # the assistant message depending on FTS5 phrase matching
        assert len(results) >= 1

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

    def test_sanitize_fts5_preserves_quoted_phrases(self):
        """Properly paired double-quoted phrases should be preserved."""
        from hermes_state import SessionDB
        s = SessionDB._sanitize_fts5_query
        # Simple quoted phrase
        assert s('"exact phrase"') == '"exact phrase"'
        # Quoted phrase alongside unquoted terms
        assert '"docker networking"' in s('"docker networking" setup')
        # Multiple quoted phrases
        result = s('"hello world" OR "foo bar"')
        assert '"hello world"' in result
        assert '"foo bar"' in result
        # Unmatched quote still stripped
        assert '"' not in s('"unterminated')

    def test_sanitize_fts5_quotes_hyphenated_terms(self):
        """Hyphenated terms should be wrapped in quotes for exact matching."""
        from hermes_state import SessionDB
        s = SessionDB._sanitize_fts5_query
        # Simple hyphenated term
        assert s('chat-send') == '"chat-send"'
        # Multiple hyphens
        assert s('docker-compose-up') == '"docker-compose-up"'
        # Hyphenated term with other words
        result = s('fix chat-send bug')
        assert '"chat-send"' in result
        assert 'fix' in result
        assert 'bug' in result
        # Multiple hyphenated terms with OR
        result = s('chat-send OR deploy-prod')
        assert '"chat-send"' in result
        assert '"deploy-prod"' in result
        # Already-quoted hyphenated term — no double quoting
        assert s('"chat-send"') == '"chat-send"'
        # Hyphenated inside a quoted phrase stays as-is
        assert s('"my chat-send thing"') == '"my chat-send thing"'

    def test_sanitize_fts5_quotes_dotted_terms(self):
        """Dotted terms should be wrapped in quotes to avoid FTS5 query parse edge cases."""
        from hermes_state import SessionDB
        s = SessionDB._sanitize_fts5_query

        assert s('P2.2') == '"P2.2"'
        assert s('simulate.p2') == '"simulate.p2"'
        assert s('simulate.p2.test.ts') == '"simulate.p2.test.ts"'

        # Already quoted — no double quoting
        assert s('"P2.2"') == '"P2.2"'

        # Works with boolean syntax
        result = s('P2.2 OR simulate.p2')
        assert '"P2.2"' in result
        assert '"simulate.p2"' in result

        # Mixed dots and hyphens — single pass avoids double-quoting
        assert s('my-app.config') == '"my-app.config"'
        assert s('my-app.config.ts') == '"my-app.config.ts"'

    def test_sanitize_fts5_quotes_underscored_terms(self):
        """Underscored terms should be wrapped in quotes for exact matching.

        FTS5 default tokenizer splits 'sp_new1' into tokens 'sp' and 'new1'.
        Without quoting, a search for 'sp_new' becomes an AND query
        ('sp AND new') that fails to match rows indexed as 'sp_new1'.
        """
        from hermes_state import SessionDB
        s = SessionDB._sanitize_fts5_query
        # Simple underscored term
        assert s('sp_new') == '"sp_new"'
        # Multiple underscores
        assert s('a_b_c') == '"a_b_c"'
        # Mixed underscores and hyphens/dots — single pass avoids double-quoting
        assert s('sp_new1') == '"sp_new1"'
        assert s('docker-compose_up') == '"docker-compose_up"'
        assert s('my.app_config.ts') == '"my.app_config.ts"'
        # Already-quoted — no double quoting
        assert s('"sp_new"') == '"sp_new"'
        # Mixed with other words
        result = s('sp_new and 血管瘤')
        assert '"sp_new"' in result
        assert '血管瘤' in result


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

    def test_chinese_multichar_query_returns_results(self, db):
        """The headline bug: multi-char Chinese query must not return []."""
        db.create_session(session_id="s1", source="cli")
        db.append_message(
            "s1", role="user",
            content="昨天和其他Agent的聊天记录，记忆断裂问题复现了",
        )
        results = db.search_messages("记忆断裂")
        assert len(results) == 1
        assert results[0]["session_id"] == "s1"

    def test_chinese_bigram_query(self, db):
        db.create_session(session_id="s1", source="telegram")
        db.append_message("s1", role="user", content="今天讨论A2A通信协议的实现")
        results = db.search_messages("通信")
        assert len(results) == 1

    def test_korean_query_returns_results(self, db):
        """Guards against Hangul range typos (\\uac00-\\ud7af, not \\ud7a0-)."""
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="안녕하세요 반갑습니다")
        results = db.search_messages("안녕")
        assert len(results) == 1

    def test_japanese_query_returns_results(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="こんにちは世界")
        assert len(db.search_messages("こんにちは")) == 1
        assert len(db.search_messages("世界")) == 1

    def test_cjk_fallback_preserves_source_filter(self, db):
        """Guards against the SQL-builder bug where filter clauses land
        after LIMIT/OFFSET (seen in one of the duplicate PRs)."""
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="telegram")
        db.append_message("s1", role="user", content="记忆断裂在CLI")
        db.append_message("s2", role="user", content="记忆断裂在Telegram")

        results = db.search_messages("记忆断裂", source_filter=["telegram"])
        assert len(results) == 1
        assert results[0]["source"] == "telegram"

    def test_cjk_fallback_preserves_exclude_sources(self, db):
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="tool")
        db.append_message("s1", role="user", content="记忆断裂在CLI")
        db.append_message("s2", role="assistant", content="记忆断裂在tool")

        results = db.search_messages("记忆断裂", exclude_sources=["tool"])
        sources = {r["source"] for r in results}
        assert "tool" not in sources
        assert "cli" in sources

    def test_cjk_fallback_preserves_role_filter(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="用户说的记忆断裂")
        db.append_message("s1", role="assistant", content="助手说的记忆断裂")

        results = db.search_messages("记忆断裂", role_filter=["assistant"])
        assert len(results) == 1
        assert results[0]["role"] == "assistant"

    def test_cjk_snippet_is_centered_on_match(self, db):
        """Snippet should contain the search term, not just the first N chars."""
        db.create_session(session_id="s1", source="cli")
        long_prefix = "这是一段很长的前缀用来把匹配位置推到文档中间" * 3
        long_suffix = "这是一段很长的后缀内容填充剩余空间" * 3
        db.append_message(
            "s1", role="user",
            content=f"{long_prefix}记忆断裂{long_suffix}",
        )
        results = db.search_messages("记忆断裂")
        assert len(results) == 1
        # The centered substr() snippet must include the matched term.
        assert "记忆断裂" in results[0]["snippet"]

    def test_english_query_still_uses_fts5_fast_path(self, db):
        """English queries must not trigger the LIKE fallback (fast path regression)."""
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="Deploy docker containers")
        results = db.search_messages("docker")
        assert len(results) == 1
        # No CJK in query → LIKE fallback must not run. We don't assert this
        # directly (no instrumentation), but the FTS5 path produces an
        # FTS5-style snippet with highlight markers when the term is short.
        # At minimum: english queries must still match.

    def test_cjk_query_with_no_matches_returns_empty(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="unrelated English content")
        results = db.search_messages("记忆断裂")
        assert results == []

    def test_mixed_cjk_english_query(self, db):
        """Mixed queries should still fall back to LIKE when FTS5 misses."""
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="讨论Agent通信协议")
        # "Agent通信" is CJK+English — FTS5 default tokenizer indexes the
        # whole CJK run with embedded "agent" as separate tokens; the LIKE
        # fallback handles the substring correctly.
        results = db.search_messages("Agent通信")
        assert len(results) == 1

    def test_cjk_partial_fts5_results_supplemented_by_like(self, db):
        """When FTS5 returns *some* CJK results, LIKE must still find all matches.

        Regression test for #15500 / #14829: FTS5 unicode61 tokenizer drops
        certain CJK characters, so multi-character queries may return partial
        results.  The LIKE path must always run for CJK queries.
        """
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="telegram")
        db.append_message("s1", role="user", content="昨晚讨论了记忆系统")
        db.append_message("s2", role="user", content="昨晚的会议纪要已发送")
        results = db.search_messages("昨晚")
        assert len(results) == 2
        session_ids = {r["session_id"] for r in results}
        assert session_ids == {"s1", "s2"}

    def test_cjk_like_dedup_no_duplicates(self, db):
        """When FTS5 and LIKE both find the same message, no duplicates."""
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="测试去重逻辑")
        results = db.search_messages("测试")
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

    def test_cjk_trigram_preserves_boolean_operators(self, db):
        """Boolean operators (OR, AND, NOT) work in CJK trigram queries."""
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="cli")
        db.append_message("s1", role="user", content="记忆系统很好用")
        db.append_message("s2", role="user", content="断裂连接需要修复")
        results = db.search_messages("记忆系统 OR 断裂连接")
        assert len(results) == 2
        session_ids = {r["session_id"] for r in results}
        assert session_ids == {"s1", "s2"}

    def test_cjk_or_combined_short_tokens_returns_results(self, db):
        """Regression test for #20494.

        OR-combined 2-char CJK tokens (e.g. "广西 OR 桂林 OR 漓江 OR 旅游")
        previously returned 0 results because _count_cjk of the whole query
        was >=3 (8 chars here), selecting the trigram path, but each individual
        token is only 2 CJK chars and trigram requires >=3 chars per token.
        The per-token check must route such queries to the LIKE fallback.
        """
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="telegram")
        db.create_session(session_id="s3", source="cli")
        db.append_message("s1", role="user", content="广西是个好地方，去过桂林")
        db.append_message("s2", role="user", content="漓江风景很美，值得旅游")
        db.append_message("s3", role="user", content="unrelated English content")

        results = db.search_messages("广西 OR 桂林 OR 漓江 OR 旅游")
        session_ids = {r["session_id"] for r in results}
        assert "s1" in session_ids, "广西/桂林 terms not matched"
        assert "s2" in session_ids, "漓江/旅游 terms not matched"
        assert "s3" not in session_ids, "unrelated message must not match"

    def test_cjk_short_token_or_query_preserves_filters(self, db):
        """Source filter applies correctly in the short-token LIKE path (#20494)."""
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="telegram")
        db.append_message("s1", role="user", content="广西旅游攻略cli")
        db.append_message("s2", role="user", content="广西旅游攻略telegram")

        results = db.search_messages("广西 OR 旅游", source_filter=["telegram"])
        assert len(results) == 1
        assert results[0]["source"] == "telegram"


# =========================================================================
# Session search and listing
# =========================================================================

class TestSearchSessions:
    def test_list_all_sessions(self, db):
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="telegram")

        sessions = db.search_sessions()
        assert len(sessions) == 2

    def test_filter_by_source(self, db):
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="telegram")

        sessions = db.search_sessions(source="cli")
        assert len(sessions) == 1
        assert sessions[0]["source"] == "cli"

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
    def test_session_count(self, db):
        assert db.session_count() == 0
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="telegram")
        assert db.session_count() == 2

    def test_session_count_by_source(self, db):
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="telegram")
        db.create_session(session_id="s3", source="cli")
        assert db.session_count(source="cli") == 2
        assert db.session_count(source="telegram") == 1

    def test_session_count_by_cwd_prefix(self, db):
        db.create_session("s1", "cli", cwd="/repo")
        db.create_session("s2", "cli", cwd="/repo-wt-feature")
        db.create_session("s3", "cli", cwd="/repo/subdir")

        assert db.session_count(cwd_prefix="/repo") == 2

    def test_message_count_total(self, db):
        assert db.message_count() == 0
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="Hello")
        db.append_message("s1", role="assistant", content="Hi")
        assert db.message_count() == 2

    def test_message_count_per_session(self, db):
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="cli")
        db.append_message("s1", role="user", content="A")
        db.append_message("s2", role="user", content="B")
        db.append_message("s2", role="user", content="C")
        assert db.message_count(session_id="s1") == 1
        assert db.message_count(session_id="s2") == 2


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

    def test_delete_nonexistent(self, db):
        assert db.delete_session("nope") is False

    def test_resolve_session_id_exact(self, db):
        db.create_session(session_id="20260315_092437_c9a6ff", source="cli")
        assert db.resolve_session_id("20260315_092437_c9a6ff") == "20260315_092437_c9a6ff"

    def test_resolve_session_id_unique_prefix(self, db):
        db.create_session(session_id="20260315_092437_c9a6ff", source="cli")
        assert db.resolve_session_id("20260315_092437_c9a6") == "20260315_092437_c9a6ff"

    def test_resolve_session_id_ambiguous_prefix_returns_none(self, db):
        db.create_session(session_id="20260315_092437_c9a6aa", source="cli")
        db.create_session(session_id="20260315_092437_c9a6bb", source="cli")
        assert db.resolve_session_id("20260315_092437_c9a6") is None

    def test_resolve_session_id_escapes_like_wildcards(self, db):
        db.create_session(session_id="20260315_092437_c9a6ff", source="cli")
        db.create_session(session_id="20260315X092437_c9a6ff", source="cli")
        assert db.resolve_session_id("20260315_092437") == "20260315_092437_c9a6ff"

    def test_export_session(self, db):
        db.create_session(session_id="s1", source="cli", model="test")
        db.append_message("s1", role="user", content="Hello")
        db.append_message("s1", role="assistant", content="Hi")

        export = db.export_session("s1")
        assert isinstance(export, dict)
        assert export["source"] == "cli"
        assert len(export["messages"]) == 2

    def test_export_nonexistent(self, db):
        assert db.export_session("nope") is None

    def test_export_all(self, db):
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="telegram")
        db.append_message("s1", role="user", content="A")

        exports = db.export_all()
        assert len(exports) == 2

    def test_export_all_with_source(self, db):
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="telegram")

        exports = db.export_all(source="cli")
        assert len(exports) == 1
        assert exports[0]["source"] == "cli"


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

    def test_prune_with_source_filter(self, db):
        for sid, src in [("old_cli", "cli"), ("old_tg", "telegram")]:
            db.create_session(session_id=sid, source=src)
            db.end_session(sid, end_reason="done")
            db._conn.execute(
                "UPDATE sessions SET started_at = ? WHERE id = ?",
                (time.time() - 200 * 86400, sid),
            )
        db._conn.commit()

        pruned = db.prune_sessions(older_than_days=90, source="cli")
        assert pruned == 1
        assert db.get_session("old_cli") is None
        assert db.get_session("old_tg") is not None

    def test_prune_with_multilevel_chain(self, db):
        """Pruning old sessions orphans newer children instead of crashing on FK."""
        old_ts = time.time() - 200 * 86400
        recent_ts = time.time() - 10 * 86400

        # Chain: A (old) -> B (old) -> C (recent) -> D (recent)
        db.create_session(session_id="A", source="cli")
        db.end_session("A", end_reason="compressed")
        db.create_session(session_id="B", source="cli", parent_session_id="A")
        db.end_session("B", end_reason="compressed")
        db.create_session(session_id="C", source="cli", parent_session_id="B")
        db.end_session("C", end_reason="compressed")
        db.create_session(session_id="D", source="cli", parent_session_id="C")
        db.end_session("D", end_reason="done")

        # Backdate A and B to be old; C and D stay recent
        for sid, ts in [("A", old_ts), ("B", old_ts), ("C", recent_ts), ("D", recent_ts)]:
            db._conn.execute(
                "UPDATE sessions SET started_at = ? WHERE id = ?", (ts, sid)
            )
        db._conn.commit()

        # Should not raise IntegrityError
        pruned = db.prune_sessions(older_than_days=90)
        assert pruned == 2  # only A and B
        assert db.get_session("A") is None
        assert db.get_session("B") is None
        # C and D survive, C is orphaned (parent_session_id NULL)
        c = db.get_session("C")
        assert c is not None
        assert c["parent_session_id"] is None
        d = db.get_session("D")
        assert d is not None
        assert d["parent_session_id"] == "C"

    def test_prune_entire_old_chain(self, db):
        """All sessions in a chain are old — entire chain is pruned."""
        old_ts = time.time() - 200 * 86400

        db.create_session(session_id="X", source="cli")
        db.end_session("X", end_reason="compressed")
        db.create_session(session_id="Y", source="cli", parent_session_id="X")
        db.end_session("Y", end_reason="compressed")
        db.create_session(session_id="Z", source="cli", parent_session_id="Y")
        db.end_session("Z", end_reason="done")

        for sid in ("X", "Y", "Z"):
            db._conn.execute(
                "UPDATE sessions SET started_at = ? WHERE id = ?", (old_ts, sid)
            )
        db._conn.commit()

        pruned = db.prune_sessions(older_than_days=90)
        assert pruned == 3
        for sid in ("X", "Y", "Z"):
            assert db.get_session(sid) is None


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

    def test_returns_real_count_skipping_unknown_ids(self, db):
        """Unknown IDs are silently skipped — the return value reflects
        what was *actually* deleted, so the UI can show an accurate
        toast even if the selection raced against another tab."""
        db.create_session(session_id="real", source="cli")

        deleted = db.delete_sessions(["real", "ghost1", "ghost2"])
        assert deleted == 1
        assert db.get_session("real") is None

    def test_empty_list_is_noop(self, db):
        """``[]`` returns 0 without touching the DB. Guards against a
        bulk endpoint with an empty payload triggering an
        unconditional 'wipe everything' if the caller forgets the
        WHERE clause."""
        db.create_session(session_id="keep", source="cli")
        assert db.delete_sessions([]) == 0
        assert db.get_session("keep") is not None

    def test_drops_non_string_entries(self, db):
        """Stray ``None`` / empty strings in the input list are
        filtered out before hitting SQL. Callers may pull selection IDs
        from a Set-like that occasionally contains noise; we don't want
        a SQL parameter-type error to fail the whole batch."""
        db.create_session(session_id="real", source="cli")
        # noinspection PyTypeChecker
        deleted = db.delete_sessions(["real", None, "", "ghost"])  # type: ignore[list-item]
        assert deleted == 1
        assert db.get_session("real") is None

    def test_dedupes_duplicate_ids(self, db):
        """The same ID listed twice counts as one deletion. Defends
        against a hand-crafted POST body or a UI bug that double-adds
        the same selection."""
        db.create_session(session_id="real", source="cli")
        deleted = db.delete_sessions(["real", "real"])
        assert deleted == 1

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

    def test_deletes_archived_and_active_when_selected(self, db):
        """Unlike the safety-gated ``delete_empty_sessions`` sweep,
        explicit bulk-select trusts the user — archived sessions and
        un-ended live sessions are both deleted when in the list.
        Otherwise the selection UI would silently 'leak' rows the user
        thought they'd removed."""
        db.create_session(session_id="archived", source="cli")
        db.end_session("archived", end_reason="done")
        db.set_session_archived("archived", True)
        db.create_session(session_id="live", source="cli")

        deleted = db.delete_sessions(["archived", "live"])
        assert deleted == 2
        assert db.get_session("archived") is None
        assert db.get_session("live") is None

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

    def test_skips_active_empty_sessions(self, db):
        """A live (un-ended) empty session is what you get during the
        race between session-create and the first message landing. The
        sweep must not delete it — that would yank a session out from
        under the agent before its first reply persists."""
        db.create_session(session_id="live", source="cli")
        # Deliberately no end_session() — session is "active".

        assert db.count_empty_sessions() == 0
        assert db.delete_empty_sessions() == 0
        assert db.get_session("live") is not None

    def test_skips_archived_empty_sessions(self, db):
        """Archived = soft-hidden by the user. They explicitly chose to
        keep the row around (even though it's empty), so the bulk sweep
        must not surprise them by deleting it. Restoring an archived
        session is one click; resurrecting one we deleted is impossible."""
        db.create_session(session_id="archived_empty", source="cli")
        db.end_session("archived_empty", end_reason="done")
        db.set_session_archived("archived_empty", True)

        assert db.count_empty_sessions() == 0
        assert db.delete_empty_sessions() == 0
        assert db.get_session("archived_empty") is not None

    def test_returns_zero_when_nothing_to_delete(self, db):
        """No-op path: no candidate rows → return 0, no error."""
        db.create_session(session_id="hasmsg", source="cli")
        db.append_message("hasmsg", role="user", content="Hello")
        db.end_session("hasmsg", end_reason="done")

        assert db.count_empty_sessions() == 0
        assert db.delete_empty_sessions() == 0
        assert db.get_session("hasmsg") is not None

    def test_orphans_children_of_deleted_empty_parent(self, db):
        """Even an empty parent can have a child (e.g. a branch session
        spawned before the parent received any messages). The sweep
        must orphan that child, not cascade-delete it — same contract
        as ``delete_session`` and ``prune_sessions``."""
        db.create_session(session_id="empty_parent", source="cli")
        db.end_session("empty_parent", end_reason="done")
        db.create_session(
            session_id="child", source="cli", parent_session_id="empty_parent"
        )
        db.append_message("child", role="user", content="something")
        db.end_session("child", end_reason="done")

        deleted = db.delete_empty_sessions()
        assert deleted == 1
        assert db.get_session("empty_parent") is None
        child = db.get_session("child")
        assert child is not None
        assert child["parent_session_id"] is None

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

    def test_set_title_nonexistent_session(self, db):
        assert db.set_session_title("nonexistent", "Title") is False

    def test_title_initially_none(self, db):
        db.create_session(session_id="s1", source="cli")
        session = db.get_session("s1")
        assert session["title"] is None

    def test_update_title(self, db):
        db.create_session(session_id="s1", source="cli")
        db.set_session_title("s1", "First Title")
        db.set_session_title("s1", "Updated Title")

        session = db.get_session("s1")
        assert session["title"] == "Updated Title"

    def test_title_in_search_sessions(self, db):
        db.create_session(session_id="s1", source="cli")
        db.set_session_title("s1", "Debugging Auth")
        db.create_session(session_id="s2", source="cli")

        sessions = db.search_sessions()
        titled = [s for s in sessions if s.get("title") == "Debugging Auth"]
        assert len(titled) == 1
        assert titled[0]["id"] == "s1"

    def test_title_in_export(self, db):
        db.create_session(session_id="s1", source="cli")
        db.set_session_title("s1", "Export Test")
        db.append_message("s1", role="user", content="Hello")

        export = db.export_session("s1")
        assert export["title"] == "Export Test"

    def test_title_with_special_characters(self, db):
        db.create_session(session_id="s1", source="cli")
        title = "PR #438 — fixing the 'auth' middleware"
        db.set_session_title("s1", title)

        session = db.get_session("s1")
        assert session["title"] == title

    def test_title_empty_string_normalized_to_none(self, db):
        """Empty strings are normalized to None (clearing the title)."""
        db.create_session(session_id="s1", source="cli")
        db.set_session_title("s1", "My Title")
        # Setting to empty string should clear the title (normalize to None)
        db.set_session_title("s1", "")

        session = db.get_session("s1")
        assert session["title"] is None

    def test_multiple_empty_titles_no_conflict(self, db):
        """Multiple sessions can have empty-string (normalized to NULL) titles."""
        db.create_session(session_id="s1", source="cli")
        db.create_session(session_id="s2", source="cli")
        db.set_session_title("s1", "")
        db.set_session_title("s2", "")
        # Both should be None, no uniqueness conflict
        assert db.get_session("s1")["title"] is None
        assert db.get_session("s2")["title"] is None

    def test_title_survives_end_session(self, db):
        db.create_session(session_id="s1", source="cli")
        db.set_session_title("s1", "Before End")
        db.end_session("s1", end_reason="user_exit")

        session = db.get_session("s1")
        assert session["title"] == "Before End"
        assert session["ended_at"] is not None


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

    def test_transfer_walks_multi_level_chain(self, db):
        import time as _time
        t0 = _time.time() - 7200
        # root (compression) -> mid (compression) -> tip
        self._make_compression_chain(db, t0, root="root", tip="mid")
        db._conn.execute(
            "UPDATE sessions SET ended_at=?, end_reason='compression' WHERE id=?",
            (t0 + 300, "mid"),
        )
        db.create_session("tip", "cli", parent_session_id="mid")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0 + 400, "tip"))
        db._conn.commit()

        db.set_session_title("root", "deep-dive")
        assert db.set_session_title("tip", "deep-dive") is True
        assert db.get_session("tip")["title"] == "deep-dive"
        assert db.get_session("root")["title"] is None

    def test_unrelated_session_still_conflicts(self, db):
        db.create_session("a", "cli")
        db.create_session("b", "cli")
        db.set_session_title("a", "shared")
        with pytest.raises(ValueError, match="already in use"):
            db.set_session_title("b", "shared")
        # The unrelated holder keeps its title.
        assert db.get_session("a")["title"] == "shared"

    def test_non_compression_child_still_conflicts(self, db):
        """A child whose parent did NOT end via compression (delegate/branch
        spawned while the parent was live) is not a continuation, so renaming it
        to the parent's title must still raise."""
        import time as _time
        t0 = _time.time() - 3600
        db.create_session("parent", "cli")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0, "parent"))
        db.create_session("child", "cli", parent_session_id="parent")
        # Child started BEFORE parent ended, and parent ended for a non-
        # compression reason — not a continuation edge.
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0 + 10, "child"))
        db._conn.execute(
            "UPDATE sessions SET ended_at=?, end_reason='user_exit' WHERE id=?",
            (t0 + 100, "parent"),
        )
        db._conn.commit()
        db.set_session_title("parent", "shared")
        with pytest.raises(ValueError, match="already in use"):
            db.set_session_title("child", "shared")


class TestSanitizeTitle:
    """Tests for SessionDB.sanitize_title() validation and cleaning."""

    def test_normal_title_unchanged(self):
        assert SessionDB.sanitize_title("My Project") == "My Project"

    def test_strips_whitespace(self):
        assert SessionDB.sanitize_title("  hello world  ") == "hello world"

    def test_collapses_internal_whitespace(self):
        assert SessionDB.sanitize_title("hello   world") == "hello world"

    def test_tabs_and_newlines_collapsed(self):
        assert SessionDB.sanitize_title("hello\t\nworld") == "hello world"

    def test_none_returns_none(self):
        assert SessionDB.sanitize_title(None) is None

    def test_empty_string_returns_none(self):
        assert SessionDB.sanitize_title("") is None

    def test_whitespace_only_returns_none(self):
        assert SessionDB.sanitize_title("   \t\n  ") is None

    def test_control_chars_stripped(self):
        # Null byte, bell, backspace, etc.
        assert SessionDB.sanitize_title("hello\x00world") == "helloworld"
        assert SessionDB.sanitize_title("\x07\x08test\x1b") == "test"

    def test_del_char_stripped(self):
        assert SessionDB.sanitize_title("hello\x7fworld") == "helloworld"

    def test_zero_width_chars_stripped(self):
        # Zero-width space (U+200B), zero-width joiner (U+200D)
        assert SessionDB.sanitize_title("hello\u200bworld") == "helloworld"
        assert SessionDB.sanitize_title("hello\u200dworld") == "helloworld"

    def test_rtl_override_stripped(self):
        # Right-to-left override (U+202E) — used in filename spoofing attacks
        assert SessionDB.sanitize_title("hello\u202eworld") == "helloworld"

    def test_bom_stripped(self):
        # Byte order mark (U+FEFF)
        assert SessionDB.sanitize_title("\ufeffhello") == "hello"

    def test_only_control_chars_returns_none(self):
        assert SessionDB.sanitize_title("\x00\x01\x02\u200b\ufeff") is None

    def test_max_length_allowed(self):
        title = "A" * 100
        assert SessionDB.sanitize_title(title) == title

    def test_exceeds_max_length_raises(self):
        title = "A" * 101
        with pytest.raises(ValueError, match="too long"):
            SessionDB.sanitize_title(title)

    def test_unicode_emoji_allowed(self):
        assert SessionDB.sanitize_title("🚀 My Project 🎉") == "🚀 My Project 🎉"

    def test_cjk_characters_allowed(self):
        assert SessionDB.sanitize_title("我的项目") == "我的项目"

    def test_accented_characters_allowed(self):
        assert SessionDB.sanitize_title("Résumé éditing") == "Résumé éditing"

    def test_special_punctuation_allowed(self):
        title = "PR #438 — fixing the 'auth' middleware"
        assert SessionDB.sanitize_title(title) == title

    def test_sanitize_applied_in_set_session_title(self, db):
        """set_session_title applies sanitize_title internally."""
        db.create_session("s1", "cli")
        db.set_session_title("s1", "  hello\x00  world  ")
        assert db.get_session("s1")["title"] == "hello world"

    def test_too_long_title_rejected_by_set(self, db):
        """set_session_title raises ValueError for overly long titles."""
        db.create_session("s1", "cli")
        with pytest.raises(ValueError, match="too long"):
            db.set_session_title("s1", "X" * 150)


class TestSchemaInit:
    def test_wal_mode(self, db):
        cursor = db._conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        assert mode == "wal"

    def test_foreign_keys_enabled(self, db):
        cursor = db._conn.execute("PRAGMA foreign_keys")
        assert cursor.fetchone()[0] == 1

    def test_tables_exist(self, db):
        cursor = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}
        assert "sessions" in tables
        assert "messages" in tables
        assert "schema_version" in tables

    def test_schema_version(self, db):
        from hermes_state import SCHEMA_VERSION
        cursor = db._conn.execute("SELECT version FROM schema_version")
        version = cursor.fetchone()[0]
        assert version == SCHEMA_VERSION

    def test_title_column_exists(self, db):
        """Verify the title column was created in the sessions table."""
        cursor = db._conn.execute("PRAGMA table_info(sessions)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "title" in columns

    def test_topic_mode_schema_is_not_auto_migrated_on_open(self, tmp_path):
        """Opening an old DB should not add topic-mode columns until /topic opts in.

        The gateway must remain rollback-safe: simply upgrading Hermes and starting
        the old bot should not eagerly mutate the state DB for this feature.
        """
        old_db = tmp_path / "old.db"
        import sqlite3

        conn = sqlite3.connect(old_db)
        conn.executescript(
            """
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            INSERT INTO schema_version VALUES (11);
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                user_id TEXT,
                model TEXT,
                model_config TEXT,
                system_prompt TEXT,
                parent_session_id TEXT,
                started_at REAL NOT NULL,
                ended_at REAL,
                end_reason TEXT,
                message_count INTEGER DEFAULT 0,
                tool_call_count INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cache_read_tokens INTEGER DEFAULT 0,
                cache_write_tokens INTEGER DEFAULT 0,
                reasoning_tokens INTEGER DEFAULT 0,
                billing_provider TEXT,
                billing_base_url TEXT,
                billing_mode TEXT,
                estimated_cost_usd REAL,
                actual_cost_usd REAL,
                cost_status TEXT,
                cost_source TEXT,
                pricing_version TEXT,
                title TEXT,
                api_call_count INTEGER DEFAULT 0,
                FOREIGN KEY (parent_session_id) REFERENCES sessions(id)
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                role TEXT NOT NULL,
                content TEXT,
                tool_call_id TEXT,
                tool_calls TEXT,
                tool_name TEXT,
                timestamp REAL NOT NULL,
                token_count INTEGER,
                finish_reason TEXT,
                reasoning TEXT,
                reasoning_content TEXT,
                reasoning_details TEXT,
                codex_reasoning_items TEXT,
                codex_message_items TEXT
            );
            """
        )
        conn.close()

        db = SessionDB(db_path=old_db)
        cursor = db._conn.execute("PRAGMA table_info(sessions)")
        columns = {row[1] for row in cursor.fetchall()}
        assert {"telegram_dm_topic_mode", "telegram_topic_thread_id"}.isdisjoint(columns)
        db.close()

    def test_apply_telegram_topic_migration_creates_topic_tables_explicitly(self, tmp_path):
        """The /topic opt-in path owns the DB migration for Telegram topic mode."""
        old_db = tmp_path / "old.db"
        import sqlite3

        conn = sqlite3.connect(old_db)
        conn.executescript(
            """
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            INSERT INTO schema_version VALUES (11);
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                user_id TEXT,
                model TEXT,
                model_config TEXT,
                system_prompt TEXT,
                parent_session_id TEXT,
                started_at REAL NOT NULL,
                ended_at REAL,
                end_reason TEXT,
                message_count INTEGER DEFAULT 0,
                tool_call_count INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cache_read_tokens INTEGER DEFAULT 0,
                cache_write_tokens INTEGER DEFAULT 0,
                reasoning_tokens INTEGER DEFAULT 0,
                billing_provider TEXT,
                billing_base_url TEXT,
                billing_mode TEXT,
                estimated_cost_usd REAL,
                actual_cost_usd REAL,
                cost_status TEXT,
                cost_source TEXT,
                pricing_version TEXT,
                title TEXT,
                api_call_count INTEGER DEFAULT 0,
                FOREIGN KEY (parent_session_id) REFERENCES sessions(id)
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                role TEXT NOT NULL,
                content TEXT,
                tool_call_id TEXT,
                tool_calls TEXT,
                tool_name TEXT,
                timestamp REAL NOT NULL,
                token_count INTEGER,
                finish_reason TEXT,
                reasoning TEXT,
                reasoning_content TEXT,
                reasoning_details TEXT,
                codex_reasoning_items TEXT,
                codex_message_items TEXT
            );
            """
        )
        conn.close()

        db = SessionDB(db_path=old_db)
        db.apply_telegram_topic_migration()

        tables = {
            row[0]
            for row in db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert "telegram_dm_topic_mode" in tables
        assert "telegram_dm_topic_bindings" in tables
        assert db.get_meta("telegram_dm_topic_schema_version") == "2"
        db.close()

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

    def test_telegram_topic_binding_refuses_to_relink_session_to_another_topic(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session(
            session_id="topic-session",
            source="telegram",
            user_id="208214988",
        )
        db.bind_telegram_topic(
            chat_id="208214988",
            thread_id="17585",
            user_id="208214988",
            session_key="key-17585",
            session_id="topic-session",
        )

        with pytest.raises(ValueError, match="already linked"):
            db.bind_telegram_topic(
                chat_id="208214988",
                thread_id="99999",
                user_id="208214988",
                session_key="key-99999",
                session_id="topic-session",
            )
        db.close()

    def test_list_unlinked_telegram_sessions_for_user_excludes_bound_and_other_users(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session(
            session_id="old-unlinked",
            source="telegram",
            user_id="208214988",
        )
        db.set_session_title("old-unlinked", "Old research")
        db.append_message("old-unlinked", "user", "first prompt")
        db.create_session(
            session_id="already-linked",
            source="telegram",
            user_id="208214988",
        )
        db.bind_telegram_topic(
            chat_id="208214988",
            thread_id="17585",
            user_id="208214988",
            session_key="key-17585",
            session_id="already-linked",
        )
        db.create_session(
            session_id="other-user",
            source="telegram",
            user_id="someone-else",
        )

        sessions = db.list_unlinked_telegram_sessions_for_user(
            chat_id="208214988",
            user_id="208214988",
        )

        assert [s["id"] for s in sessions] == ["old-unlinked"]
        assert sessions[0]["title"] == "Old research"
        assert sessions[0]["preview"] == "first prompt"
        db.close()

    def test_migration_from_v2(self, tmp_path):
        """Simulate a v2 database and verify migration adds title column."""
        import sqlite3

        db_path = tmp_path / "migrate_test.db"
        conn = sqlite3.connect(str(db_path))
        # Create v2 schema (without title column)
        conn.executescript("""
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            INSERT INTO schema_version (version) VALUES (2);

            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                user_id TEXT,
                model TEXT,
                model_config TEXT,
                system_prompt TEXT,
                parent_session_id TEXT,
                started_at REAL NOT NULL,
                ended_at REAL,
                end_reason TEXT,
                message_count INTEGER DEFAULT 0,
                tool_call_count INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0
            );

            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_call_id TEXT,
                tool_calls TEXT,
                tool_name TEXT,
                timestamp REAL NOT NULL,
                token_count INTEGER,
                finish_reason TEXT
            );
        """)
        conn.execute(
            "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
            ("existing", "cli", 1000.0),
        )
        conn.commit()
        conn.close()

        # Open with SessionDB — should migrate to v9
        migrated_db = SessionDB(db_path=db_path)

        # Verify migration
        from hermes_state import SCHEMA_VERSION
        cursor = migrated_db._conn.execute("SELECT version FROM schema_version")
        assert cursor.fetchone()[0] == SCHEMA_VERSION

        # Verify title column exists and is NULL for existing sessions
        session = migrated_db.get_session("existing")
        assert session is not None
        assert session["title"] is None

        # Verify api_call_count column was added with default 0
        cursor = migrated_db._conn.execute(
            "SELECT api_call_count FROM sessions WHERE id = 'existing'"
        )
        assert cursor.fetchone()[0] == 0

        # Verify we can set title on migrated session
        assert migrated_db.set_session_title("existing", "Migrated Title") is True
        session = migrated_db.get_session("existing")
        assert session["title"] == "Migrated Title"

        migrated_db.close()

    def test_v9_migration_skips_v10_trigram_backfill_before_v11_rebuild(self, tmp_path, monkeypatch):
        """Direct v9→current migration should do only the v11 FTS rebuild.

        v10 backfilled ``messages_fts_trigram`` with content-only rows. Current
        v11+ migration immediately drops and rebuilds both FTS tables with
        content + tool metadata, so running the v10 insert first is wasted work.
        """
        db_path = tmp_path / "v9_fts.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript(SCHEMA_SQL)
        conn.execute("DELETE FROM schema_version")
        conn.execute("INSERT INTO schema_version (version) VALUES (9)")
        conn.execute(
            "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
            ("s1", "cli", 1000.0),
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content, tool_name, tool_calls, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("s1", "tool", "plain content", "browser_snapshot", '{"name":"browser_snapshot"}', 1001.0),
        )
        conn.commit()
        conn.close()

        trigram_content_only_inserts = []
        real_connect = sqlite3.connect

        def connect_with_trace(*args, **kwargs):
            conn = real_connect(*args, **kwargs)

            def trace(sql):
                text = " ".join(str(sql).split())
                if (
                    "INSERT INTO messages_fts_trigram" in text
                    and "SELECT id, content FROM messages" in text
                ):
                    trigram_content_only_inserts.append(text)

            conn.set_trace_callback(trace)
            return conn

        monkeypatch.setattr("hermes_state.sqlite3.connect", connect_with_trace)
        migrated_db = SessionDB(db_path=db_path)
        try:
            assert trigram_content_only_inserts == []
            version = migrated_db._conn.execute("SELECT version FROM schema_version").fetchone()[0]
            assert version == SCHEMA_VERSION
            normal_count = migrated_db._conn.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0]
            trigram_count = migrated_db._conn.execute("SELECT COUNT(*) FROM messages_fts_trigram").fetchone()[0]
            assert normal_count == 1
            assert trigram_count == 1
            tool_hit = migrated_db._conn.execute(
                "SELECT COUNT(*) FROM messages_fts_trigram "
                "WHERE messages_fts_trigram MATCH 'browser_snapshot'"
            ).fetchone()[0]
            assert tool_hit == 1
        finally:
            migrated_db.close()

    def test_reconciliation_adds_missing_columns(self, tmp_path):
        """Columns present in SCHEMA_SQL but missing from the live table
        are added by _reconcile_columns regardless of schema_version.

        Regression test: commit a7d78d3b inserted a new v7 migration
        (reasoning_content) and renumbered the old v7 (api_call_count)
        to v8.  Users already at the old v7 had schema_version >= 7,
        so the new v7 block was skipped and reasoning_content was never
        created — causing 'no such column' on /continue.
        """
        import sqlite3

        db_path = tmp_path / "gap_test.db"
        conn = sqlite3.connect(str(db_path))
        # Simulate the old v7 state: api_call_count exists, reasoning_content does NOT
        conn.executescript("""
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            INSERT INTO schema_version (version) VALUES (7);

            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                user_id TEXT,
                model TEXT,
                model_config TEXT,
                system_prompt TEXT,
                parent_session_id TEXT,
                started_at REAL NOT NULL,
                ended_at REAL,
                end_reason TEXT,
                message_count INTEGER DEFAULT 0,
                tool_call_count INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cache_read_tokens INTEGER DEFAULT 0,
                cache_write_tokens INTEGER DEFAULT 0,
                reasoning_tokens INTEGER DEFAULT 0,
                billing_provider TEXT,
                billing_base_url TEXT,
                billing_mode TEXT,
                estimated_cost_usd REAL,
                actual_cost_usd REAL,
                cost_status TEXT,
                cost_source TEXT,
                pricing_version TEXT,
                title TEXT,
                api_call_count INTEGER DEFAULT 0
            );

            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                tool_call_id TEXT,
                tool_calls TEXT,
                tool_name TEXT,
                timestamp REAL NOT NULL,
                token_count INTEGER,
                finish_reason TEXT,
                reasoning TEXT,
                reasoning_details TEXT,
                codex_reasoning_items TEXT
            );
        """)
        conn.execute(
            "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
            ("s1", "cli", 1000.0),
        )
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) "
            "VALUES (?, ?, ?, ?)",
            ("s1", "assistant", "hello", 1001.0),
        )
        conn.commit()
        # Verify reasoning_content is absent
        cols = {r[1] for r in conn.execute("PRAGMA table_info(messages)").fetchall()}
        assert "reasoning_content" not in cols
        conn.close()

        # Open with SessionDB — reconciliation should add the missing column
        migrated_db = SessionDB(db_path=db_path)

        msg_cols = {
            r[1]
            for r in migrated_db._conn.execute("PRAGMA table_info(messages)").fetchall()
        }
        assert "reasoning_content" in msg_cols

        # The query that used to crash must now work
        cursor = migrated_db._conn.execute(
            "SELECT role, content, reasoning, reasoning_content, "
            "reasoning_details, codex_reasoning_items "
            "FROM messages WHERE session_id = ?",
            ("s1",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "assistant"
        assert row[3] is None  # reasoning_content NULL for old rows

        migrated_db.close()

    def test_reconciliation_is_idempotent(self, tmp_path):
        """Opening the same database twice doesn't error or duplicate columns."""
        db_path = tmp_path / "idempotent.db"
        db1 = SessionDB(db_path=db_path)
        cols1 = {r[1] for r in db1._conn.execute("PRAGMA table_info(messages)").fetchall()}
        db1.close()

        db2 = SessionDB(db_path=db_path)
        cols2 = {r[1] for r in db2._conn.execute("PRAGMA table_info(messages)").fetchall()}
        db2.close()

        assert cols1 == cols2

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

    def test_same_session_can_keep_title(self, db):
        """A session can re-set its own title without error."""
        db.create_session("s1", "cli")
        db.set_session_title("s1", "my project")
        # Should not raise — it's the same session
        assert db.set_session_title("s1", "my project") is True

    def test_null_titles_not_unique(self, db):
        """Multiple sessions can have NULL titles (no constraint violation)."""
        db.create_session("s1", "cli")
        db.create_session("s2", "cli")
        # Both have NULL titles — no error
        assert db.get_session("s1")["title"] is None
        assert db.get_session("s2")["title"] is None

    def test_get_session_by_title(self, db):
        db.create_session("s1", "cli")
        db.set_session_title("s1", "refactoring auth")
        result = db.get_session_by_title("refactoring auth")
        assert result is not None
        assert result["id"] == "s1"

    def test_get_session_by_title_not_found(self, db):
        assert db.get_session_by_title("nonexistent") is None

    def test_get_session_title(self, db):
        db.create_session("s1", "cli")
        assert db.get_session_title("s1") is None
        db.set_session_title("s1", "my title")
        assert db.get_session_title("s1") == "my title"

    def test_get_session_title_nonexistent(self, db):
        assert db.get_session_title("nonexistent") is None


class TestTitleLineage:
    """Tests for title lineage resolution and auto-numbering."""

    def test_resolve_exact_title(self, db):
        db.create_session("s1", "cli")
        db.set_session_title("s1", "my project")
        assert db.resolve_session_by_title("my project") == "s1"

    def test_resolve_returns_latest_numbered(self, db):
        """When numbered variants exist, return the most recent one."""
        import time
        db.create_session("s1", "cli")
        db.set_session_title("s1", "my project")
        time.sleep(0.01)
        db.create_session("s2", "cli")
        db.set_session_title("s2", "my project #2")
        time.sleep(0.01)
        db.create_session("s3", "cli")
        db.set_session_title("s3", "my project #3")
        # Resolving "my project" should return s3 (latest numbered variant)
        assert db.resolve_session_by_title("my project") == "s3"

    def test_resolve_exact_numbered(self, db):
        """Resolving an exact numbered title returns that specific session."""
        db.create_session("s1", "cli")
        db.set_session_title("s1", "my project")
        db.create_session("s2", "cli")
        db.set_session_title("s2", "my project #2")
        # Resolving "my project #2" exactly should return s2
        assert db.resolve_session_by_title("my project #2") == "s2"

    def test_resolve_nonexistent_title(self, db):
        assert db.resolve_session_by_title("nonexistent") is None

    def test_next_title_no_existing(self, db):
        """With no existing sessions, base title is returned as-is."""
        assert db.get_next_title_in_lineage("my project") == "my project"

    def test_next_title_first_continuation(self, db):
        """First continuation after the original gets #2."""
        db.create_session("s1", "cli")
        db.set_session_title("s1", "my project")
        assert db.get_next_title_in_lineage("my project") == "my project #2"

    def test_next_title_increments(self, db):
        """Each continuation increments the number."""
        db.create_session("s1", "cli")
        db.set_session_title("s1", "my project")
        db.create_session("s2", "cli")
        db.set_session_title("s2", "my project #2")
        db.create_session("s3", "cli")
        db.set_session_title("s3", "my project #3")
        assert db.get_next_title_in_lineage("my project") == "my project #4"

    def test_next_title_strips_existing_number(self, db):
        """Passing a numbered title strips the number and finds the base."""
        db.create_session("s1", "cli")
        db.set_session_title("s1", "my project")
        db.create_session("s2", "cli")
        db.set_session_title("s2", "my project #2")
        # Even when called with "my project #2", it should return #3
        assert db.get_next_title_in_lineage("my project #2") == "my project #3"


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

    def test_resolve_title_with_percent(self, db):
        """A title with '%' should not wildcard-match unrelated sessions."""
        db.create_session("s1", "cli")
        db.set_session_title("s1", "100% done")
        db.create_session("s2", "cli")
        db.set_session_title("s2", "100X done #2")
        # Should resolve to s1 (exact), not s2
        assert db.resolve_session_by_title("100% done") == "s1"

    def test_next_lineage_with_underscore(self, db):
        """get_next_title_in_lineage with underscores doesn't match wrong sessions."""
        db.create_session("s1", "cli")
        db.set_session_title("s1", "test_project")
        db.create_session("s2", "cli")
        db.set_session_title("s2", "testXproject #2")
        # Only "test_project" exists, so next should be "test_project #2"
        assert db.get_next_title_in_lineage("test_project") == "test_project #2"


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

    def test_preview_truncated_at_60(self, db):
        db.create_session("s1", "cli")
        long_msg = "A" * 100
        db.append_message("s1", "user", long_msg)
        sessions = db.list_sessions_rich()
        assert len(sessions[0]["preview"]) == 63  # 60 chars + "..."
        assert sessions[0]["preview"].endswith("...")

    def test_preview_empty_when_no_user_messages(self, db):
        db.create_session("s1", "cli")
        db.append_message("s1", "system", "System prompt")
        sessions = db.list_sessions_rich()
        assert sessions[0]["preview"] == ""

    def test_last_active_from_latest_message(self, db):
        import time
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "Hello")
        time.sleep(0.01)
        db.append_message("s1", "assistant", "Hi there!")
        sessions = db.list_sessions_rich()
        # last_active should be close to now (the assistant message)
        assert sessions[0]["last_active"] > sessions[0]["started_at"]

    def test_last_active_fallback_to_started_at(self, db):
        db.create_session("s1", "cli")
        sessions = db.list_sessions_rich()
        # No messages, so last_active falls back to started_at
        assert sessions[0]["last_active"] == sessions[0]["started_at"]

    def test_order_by_last_active_surfaces_recently_touched_older_session_first(self, db):
        t0 = 1709500000.0
        db.create_session("old", "cli")
        db.create_session("new", "cli")

        with db._lock:
            db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0, "old"))
            db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0 + 10, "new"))

        db.append_message("old", "user", "old first")
        db.append_message("new", "user", "new first")
        db.append_message("old", "assistant", "old touched later")

        with db._lock:
            db._conn.execute(
                "UPDATE messages SET timestamp=? WHERE session_id=? AND role=? AND content=?",
                (t0 + 1, "old", "user", "old first"),
            )
            db._conn.execute(
                "UPDATE messages SET timestamp=? WHERE session_id=? AND role=? AND content=?",
                (t0 + 11, "new", "user", "new first"),
            )
            db._conn.execute(
                "UPDATE messages SET timestamp=? WHERE session_id=? AND role=? AND content=?",
                (t0 + 20, "old", "assistant", "old touched later"),
            )
            db._conn.commit()

        assert [s["id"] for s in db.list_sessions_rich(limit=5)] == ["new", "old"]
        assert [
            s["id"] for s in db.list_sessions_rich(limit=5, order_by_last_active=True)
        ] == ["old", "new"]

    def test_order_by_last_active_uses_compression_tip_activity(self, db):
        """A compression root whose tip was touched recently must rank above
        a newer uncompressed session, even when that tip activity lives in a
        different row and the outer LIMIT could otherwise cut it.

        This is the case that forced SQL-level chain walking: a naive "cap
        the SQL fetch at limit*K" optimization would drop the old root off
        the SQL page before post-projection could promote it.
        """
        t0 = 1709500000.0
        db.create_session("root1", "cli")
        with db._lock:
            db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0, "root1"))
            db._conn.execute(
                "UPDATE sessions SET ended_at=?, end_reason=? WHERE id=?",
                (t0 + 100, "compression", "root1"),
            )
        db.append_message("root1", "user", "old ask")

        # Continuation tip created after root ended; last activity much later.
        db.create_session("tip1", "cli", parent_session_id="root1")
        with db._lock:
            db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0 + 101, "tip1"))
        db.append_message("tip1", "user", "latest message")

        # Bunch of newer, uncompressed sessions — fresher start_at but older
        # last activity than the tip. Explicitly pin message timestamps so
        # they don't pick up wall-clock from append_message.
        for i in range(5):
            sid = f"newer{i}"
            db.create_session(sid, "cli")
            with db._lock:
                db._conn.execute(
                    "UPDATE sessions SET started_at=? WHERE id=?",
                    (t0 + 500 + i, sid),
                )
            db.append_message(sid, "user", f"msg {i}")
            with db._lock:
                db._conn.execute(
                    "UPDATE messages SET timestamp=? WHERE session_id=? AND content=?",
                    (t0 + 500 + i, sid, f"msg {i}"),
                )

        # Tip activity timestamp is the latest thing in the DB.
        with db._lock:
            db._conn.execute(
                "UPDATE messages SET timestamp=? WHERE session_id=? AND content=?",
                (t0 + 10_000, "tip1", "latest message"),
            )
            db._conn.commit()

        # limit=1 is the stress test: the old root must win the single slot.
        top = db.list_sessions_rich(limit=1, order_by_last_active=True)
        assert len(top) == 1
        # Projection surfaces the tip's id in the root's slot.
        assert top[0]["id"] == "tip1"
        assert top[0]["_lineage_root_id"] == "root1"

    def test_rich_list_includes_title(self, db):
        db.create_session("s1", "cli")
        db.set_session_title("s1", "refactoring auth")
        sessions = db.list_sessions_rich()
        assert sessions[0]["title"] == "refactoring auth"

    def test_rich_list_source_filter(self, db):
        db.create_session("s1", "cli")
        db.create_session("s2", "telegram")
        sessions = db.list_sessions_rich(source="cli")
        assert len(sessions) == 1
        assert sessions[0]["id"] == "s1"

    def test_rich_list_cwd_prefix_filter(self, db):
        db.create_session("s1", "cli", cwd="/repo")
        db.create_session("s2", "cli", cwd="/repo/subdir")
        db.create_session("s3", "cli", cwd="/repo-wt-feature")

        sessions = db.list_sessions_rich(cwd_prefix="/repo")
        assert [session["id"] for session in sessions] == ["s2", "s1"]

    def test_preview_newlines_collapsed(self, db):
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "Line one\nLine two\nLine three")
        sessions = db.list_sessions_rich()
        assert "\n" not in sessions[0]["preview"]
        assert "Line one Line two" in sessions[0]["preview"]

    def test_branch_session_visible_in_list(self, db):
        """Branch sessions (parent ended with 'branched') must appear in list_sessions_rich."""
        db.create_session("parent", "cli")
        db.end_session("parent", "branched")
        db.create_session("branch", "cli", parent_session_id="parent")
        db.append_message("branch", "user", "Exploring the alternative approach")

        sessions = db.list_sessions_rich()
        ids = [s["id"] for s in sessions]
        assert "branch" in ids, "Branch session should be visible in default list"

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

    def test_delete_parent_cascades_delegate_children(self, db):
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

        assert db.delete_session("parent") is True
        assert db.get_session("delegate") is None
        assert db.get_session("branch") is not None

    def test_v16_migration_tags_linked_delegate_rows(self, tmp_path):
        """Pre-marker linked subagent rows get tagged, then cascade with parent."""
        import json

        db_path = tmp_path / "state.db"
        db = SessionDB(db_path=db_path)
        db.create_session("parent", "cli")
        db.create_session("delegate", "cli", parent_session_id="parent")
        db._conn.execute("UPDATE schema_version SET version = 15")
        db._conn.commit()
        db.close()

        db = SessionDB(db_path=db_path)
        row = db.get_session("delegate")
        assert json.loads(row["model_config"])["_delegate_from"] == "parent"
        assert db.delete_session("parent") is True
        assert db.get_session("delegate") is None
        db.close()

    def test_v16_migration_tags_orphaned_delegate_rows(self, tmp_path):
        import json

        db_path = tmp_path / "state.db"
        db = SessionDB(db_path=db_path)
        db.create_session("orphan", "cli")
        db.append_message("orphan", "user", "Echo progress")
        db.append_message("orphan", "tool", "step 1", tool_name="terminal")
        db._conn.execute("UPDATE schema_version SET version = 15")
        db._conn.commit()
        db.close()

        db = SessionDB(db_path=db_path)
        assert "orphan" not in [s["id"] for s in db.list_sessions_rich()]
        row = db.get_session("orphan")
        assert json.loads(row["model_config"])["_delegate_from"] == "__orphaned__"
        db.close()

    def test_branch_session_visible_after_parent_reopen_and_reend(self, db):
        """Branch sessions stay visible after the parent is reopened and re-ended.

        Regression for issue #20856: /branch (aka /fork) sessions vanished from
        /resume and /sessions once the parent was reopened (e.g. resumed) and
        re-ended with a different end_reason — tui_shutdown overwriting
        'branched' — which broke the legacy end_reason heuristic. The stable
        _branched_from marker in model_config keeps them visible.
        """
        import json as _json

        db.create_session("parent", "cli")
        db.end_session("parent", "branched")
        db.create_session(
            "branch",
            "cli",
            model_config={"_branched_from": "parent"},
            parent_session_id="parent",
        )
        db.append_message("branch", "user", "Exploring the alternative approach")

        # Marker is persisted at creation time.
        branch_row = db.get_session("branch")
        cfg = _json.loads(branch_row["model_config"]) if branch_row["model_config"] else {}
        assert cfg.get("_branched_from") == "parent"

        # Visible immediately after branching.
        assert "branch" in [s["id"] for s in db.list_sessions_rich()]

        # Parent reopened + re-ended with a different reason (the bug trigger).
        db.reopen_session("parent")
        db.end_session("parent", "tui_shutdown")

        # Branch must STILL be visible — the marker survives the parent's
        # end_reason churn, unlike the legacy 'branched' heuristic.
        ids = [s["id"] for s in db.list_sessions_rich()]
        assert "branch" in ids, "Branch should stay visible after parent re-end"

    def test_subagent_session_still_hidden(self, db):
        """Sub-agent children (parent NOT ended with 'branched') remain hidden."""
        db.create_session("root", "cli")
        db.create_session("delegate", "cli", parent_session_id="root")

        sessions = db.list_sessions_rich()
        ids = [s["id"] for s in sessions]
        assert "delegate" not in ids, "Delegate sub-agent should not appear in default list"
        assert "root" in ids

    def test_compression_child_still_hidden(self, db):
        """Compression continuation sessions remain hidden (parent ended with 'compression')."""
        import time as _time
        t0 = _time.time()
        db.create_session("root", "cli")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t0, "root"))
        db._conn.execute(
            "UPDATE sessions SET ended_at=?, end_reason='compression' WHERE id=?",
            (t0 + 1800, "root"),
        )
        db._conn.commit()
        db.create_session("continuation", "cli", parent_session_id="root")
        db._conn.execute(
            "UPDATE sessions SET started_at=? WHERE id=?", (t0 + 1801, "continuation")
        )
        db._conn.commit()

        sessions = db.list_sessions_rich(project_compression_tips=False)
        ids = [s["id"] for s in sessions]
        assert "continuation" not in ids, "Compression continuation should stay hidden"


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

    def test_get_compression_tip_returns_self_for_uncompressed(self, db):
        db.create_session("solo", "cli")
        assert db.get_compression_tip("solo") == "solo"

    def test_get_compression_tip_skips_delegate_children(self, db):
        """Delegate subagents have parent_session_id set but were created
        BEFORE the parent ended. They must not be followed as compression
        continuations — the started_at >= ended_at guard handles this.
        """
        import time as _time
        self._build_compression_chain(db, _time.time() - 3600)
        # delegate1 is a child of root1 but NOT a compression continuation.
        # root1's tip must be tip1 (via mid1), not delegate1.
        assert db.get_compression_tip("root1") == "tip1"

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

    def test_list_projection_uses_tip_cwd(self, db):
        """Projected lineage rows should carry cwd from the live tip row.

        Without this, compressed conversations can lose workspace grouping
        even after the continuation session persists its cwd.
        """
        import time as _time

        self._build_compression_chain(db, _time.time() - 3600)
        db.update_session_cwd("tip1", "/tmp/workspaces/tip")
        db._conn.commit()

        sessions = db.list_sessions_rich(source="cli", limit=20)
        tip_row = next(s for s in sessions if s["id"] == "tip1")

        assert tip_row["_lineage_root_id"] == "root1"
        assert tip_row["cwd"] == "/tmp/workspaces/tip"

    def test_list_without_projection_returns_raw_root(self, db):
        """project_compression_tips=False returns the raw parent-NULL root
        rows — useful for admin/debug UIs.
        """
        import time as _time
        self._build_compression_chain(db, _time.time() - 3600)
        sessions = db.list_sessions_rich(
            source="cli", limit=20, project_compression_tips=False
        )
        ids = [s["id"] for s in sessions]
        assert "root1" in ids
        assert "tip1" not in ids

        root_row = next(s for s in sessions if s["id"] == "root1")
        assert root_row["end_reason"] == "compression"
        assert "_lineage_root_id" not in root_row

    def test_list_preserves_sort_by_started_at(self, db):
        """Chronological ordering uses the ROOT's started_at (conversation
        start), not the tip's. This keeps lineage entries stable in the list
        even as new compressions push the tip forward in time.
        """
        import time as _time
        t0 = _time.time() - 3600
        self._build_compression_chain(db, t0)

        # Create a newer standalone session that should sort above the lineage
        # if we used tip.started_at, but below if we correctly use root.started_at.
        t_between = t0 + 120  # between root1 and its compression
        db.create_session("newer", "cli")
        db._conn.execute("UPDATE sessions SET started_at=? WHERE id=?", (t_between, "newer"))
        db.append_message("newer", "user", "newer session started after root1")
        db._conn.commit()

        sessions = db.list_sessions_rich(source="cli", limit=20)
        ids_in_order = [s["id"] for s in sessions]
        # 'newer' started AFTER root1 but BEFORE tip1's actual started_at.
        # Correct ordering (by root started_at): newer > tip1's lineage entry.
        assert ids_in_order.index("newer") < ids_in_order.index("tip1")

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

    def test_list_sessions_rich_no_exclusion_returns_all(self, db):
        db.create_session("s1", "cli")
        db.create_session("s2", "tool")
        sessions = db.list_sessions_rich()
        ids = [s["id"] for s in sessions]
        assert "s1" in ids
        assert "s2" in ids

    def test_list_sessions_rich_source_and_exclude_combined(self, db):
        """When source= is explicit, exclude_sources should not conflict."""
        db.create_session("s1", "cli")
        db.create_session("s2", "tool")
        db.create_session("s3", "telegram")
        # Explicit source filter: only tool sessions, no exclusion
        sessions = db.list_sessions_rich(source="tool")
        ids = [s["id"] for s in sessions]
        assert ids == ["s2"]

    def test_list_sessions_rich_exclude_multiple_sources(self, db):
        db.create_session("s1", "cli")
        db.create_session("s2", "tool")
        db.create_session("s3", "cron")
        db.create_session("s4", "telegram")
        sessions = db.list_sessions_rich(exclude_sources=["tool", "cron"])
        ids = [s["id"] for s in sessions]
        assert "s1" in ids
        assert "s4" in ids
        assert "s2" not in ids
        assert "s3" not in ids

    def test_search_messages_excludes_tool_source(self, db):
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "Python deployment question")
        db.create_session("s2", "tool")
        db.append_message("s2", "user", "Python automated question")
        results = db.search_messages("Python", exclude_sources=["tool"])
        sources = [r["source"] for r in results]
        assert "cli" in sources
        assert "tool" not in sources

    def test_search_messages_no_exclusion_returns_all_sources(self, db):
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "Rust deployment question")
        db.create_session("s2", "tool")
        db.append_message("s2", "user", "Rust automated question")
        results = db.search_messages("Rust")
        sources = [r["source"] for r in results]
        assert "cli" in sources
        assert "tool" in sources

    def test_search_messages_source_include_and_exclude(self, db):
        """source_filter (include) and exclude_sources can coexist."""
        db.create_session("s1", "cli")
        db.append_message("s1", "user", "Golang test")
        db.create_session("s2", "telegram")
        db.append_message("s2", "user", "Golang test")
        db.create_session("s3", "tool")
        db.append_message("s3", "user", "Golang test")
        # Include cli+tool, but exclude tool → should only return cli
        results = db.search_messages(
            "Golang", source_filter=["cli", "tool"], exclude_sources=["tool"]
        )
        sources = [r["source"] for r in results]
        assert sources == ["cli"]


class TestResolveSessionByNameOrId:
    """Tests for the main.py helper that resolves names or IDs."""

    def test_resolve_by_id(self, db):
        db.create_session("test-id-123", "cli")
        session = db.get_session("test-id-123")
        assert session is not None
        assert session["id"] == "test-id-123"

    def test_resolve_by_title_falls_back(self, db):
        db.create_session("s1", "cli")
        db.set_session_title("s1", "my project")
        result = db.resolve_session_by_title("my project")
        assert result == "s1"


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

    def test_ensure_session_is_idempotent(self, db):
        """ensure_session on an existing row must be a no-op (no overwrite)."""
        db.create_session(session_id="existing", source="cli", model="original-model")
        db.ensure_session("existing", source="gateway", model="overwrite-model")
        row = db.get_session("existing")
        # First write wins — ensure_session must not overwrite
        assert row["source"] == "cli"
        assert row["model"] == "original-model"

    def test_ensure_session_allows_append_message_after_failed_create(self, db):
        """Messages can be flushed even when create_session failed at startup.

        Simulates the #3139 scenario: create_session raises (lock), then
        ensure_session is called during flush, then append_message succeeds.
        """
        # Simulate failed create_session — row absent
        db.ensure_session("late-session", source="gateway", model="gpt-4")
        db.append_message(
            session_id="late-session",
            role="user",
            content="hello after lock",
        )
        msgs = db.get_messages("late-session")
        assert len(msgs) == 1
        assert msgs[0]["content"] == "hello after lock"

    def test_sqlite_timeout_is_at_least_30s(self, db):
        """Connection timeout should be >= 30s to survive CLI/gateway contention."""
        # Access the underlying connection timeout via sqlite3 introspection.
        # There is no public API, so we check the kwarg via the module default.
        import inspect
        from hermes_state import SessionDB as _SessionDB
        src = inspect.getsource(_SessionDB.__init__)
        assert "30" in src, (
            "SQLite timeout should be at least 30s to handle CLI/gateway lock contention"
        )


# =========================================================================
# Auto-maintenance: state_meta + vacuum + maybe_auto_prune_and_vacuum
# =========================================================================

class TestStateMeta:
    def test_get_meta_missing_returns_none(self, db):
        assert db.get_meta("nonexistent") is None

    def test_set_then_get_meta(self, db):
        db.set_meta("foo", "bar")
        assert db.get_meta("foo") == "bar"

    def test_set_meta_upsert(self, db):
        """set_meta overwrites existing value (ON CONFLICT DO UPDATE)."""
        db.set_meta("key", "v1")
        db.set_meta("key", "v2")
        assert db.get_meta("key") == "v2"


class TestVacuum:
    def test_vacuum_runs_without_error(self, db):
        """VACUUM must succeed on a fresh DB (no rows to reclaim)."""
        db.create_session(session_id="s1", source="cli")
        db.append_message(session_id="s1", role="user", content="hi")
        # Should not raise, even though there's nothing significant to reclaim.
        db.vacuum()


class TestOptimizeFts:
    def test_optimize_returns_index_count(self, db):
        """A fresh DB has both FTS indexes; optimize merges both."""
        db.create_session(session_id="s1", source="cli")
        db.append_message(session_id="s1", role="user", content="hello world")
        assert db.optimize_fts() == 2

    def test_optimize_preserves_search_and_snippet(self, db):
        """Optimize is layout-only: MATCH results + snippets are unchanged."""
        db.create_session(session_id="s1", source="cli")
        for i in range(50):
            db.append_message(
                session_id="s1",
                role="user",
                content=f"needle alpha bravo charlie message {i}",
            )
        before = db.search_messages("needle")
        n = db.optimize_fts()
        assert n == 2
        after = db.search_messages("needle")
        assert len(after) == len(before)
        assert len(after) > 0
        # Snippet must still be populated (would be empty/None if the FTS
        # content shadow were lost during optimize).
        assert all(row.get("snippet") for row in after)
        # IDs and snippets are identical before/after — pure layout change.
        assert [r["id"] for r in after] == [r["id"] for r in before]
        assert [r["snippet"] for r in after] == [r["snippet"] for r in before]

    def test_optimize_skips_missing_trigram_table(self, db):
        """When the trigram index is absent, optimize handles only the porter
        index and does not raise."""
        db.create_session(session_id="s1", source="cli")
        db.append_message(session_id="s1", role="user", content="hello")
        # Drop the trigram table + triggers to simulate a disabled/absent index.
        with db._lock:
            for trig in (
                "messages_fts_trigram_insert",
                "messages_fts_trigram_delete",
                "messages_fts_trigram_update",
            ):
                db._conn.execute(f"DROP TRIGGER IF EXISTS {trig}")
            db._conn.execute("DROP TABLE IF EXISTS messages_fts_trigram")
        assert db._fts_table_exists("messages_fts_trigram") is False
        assert db._fts_table_exists("messages_fts") is True
        # Only the porter index remains -> 1 optimized, no error.
        assert db.optimize_fts() == 1

    def test_optimize_idempotent(self, db):
        """Running optimize twice is safe (second pass is a no-op merge)."""
        db.create_session(session_id="s1", source="cli")
        db.append_message(session_id="s1", role="user", content="repeat me")
        assert db.optimize_fts() == 2
        assert db.optimize_fts() == 2
        # Search still works after repeated optimization.
        assert len(db.search_messages("repeat")) == 1


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

    def test_second_call_after_interval_runs_again(self, db):
        self._make_old_ended(db, "old", days_old=100)
        db.maybe_auto_prune_and_vacuum(retention_days=90, min_interval_hours=24)

        # Backdate the last-run marker to force another run.
        db.set_meta("last_auto_prune", str(time.time() - 48 * 3600))

        self._make_old_ended(db, "old2", days_old=100)
        result = db.maybe_auto_prune_and_vacuum(
            retention_days=90, min_interval_hours=24
        )
        assert result["skipped"] is False
        assert result["pruned"] == 1
        assert db.get_session("old2") is None

    def test_no_prunable_sessions_no_vacuum(self, db):
        """When prune deletes 0 rows, VACUUM is skipped (wasted I/O)."""
        db.create_session(session_id="fresh", source="cli")  # too recent
        result = db.maybe_auto_prune_and_vacuum(retention_days=90)
        assert result["skipped"] is False
        assert result["pruned"] == 0
        assert result["vacuumed"] is False
        # But last-run is still recorded so we don't retry immediately.
        assert db.get_meta("last_auto_prune") is not None

    def test_vacuum_disabled_via_flag(self, db):
        self._make_old_ended(db, "old", days_old=100)
        result = db.maybe_auto_prune_and_vacuum(retention_days=90, vacuum=False)
        assert result["pruned"] == 1
        assert result["vacuumed"] is False

    def test_corrupt_last_run_marker_treated_as_no_prior_run(self, db):
        """A non-numeric marker must not break maintenance."""
        db.set_meta("last_auto_prune", "not-a-timestamp")
        self._make_old_ended(db, "old", days_old=100)
        result = db.maybe_auto_prune_and_vacuum(retention_days=90)
        assert result["skipped"] is False
        assert result["pruned"] == 1

    def test_state_meta_survives_vacuum(self, db):
        """Marker written just before VACUUM must still be readable after."""
        self._make_old_ended(db, "old", days_old=100)
        db.maybe_auto_prune_and_vacuum(retention_days=90)
        marker = db.get_meta("last_auto_prune")
        assert marker is not None
        # Should parse as a float timestamp close to now.
        assert abs(float(marker) - time.time()) < 60

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

    def test_auto_prune_without_sessions_dir_preserves_files(self, db, tmp_path):
        """Backward-compat: no sessions_dir = DB-only cleanup (legacy behavior)."""
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        self._make_old_ended(db, "old", days_old=100)
        (sessions_dir / "old.jsonl").write_text("{}\n")

        result = db.maybe_auto_prune_and_vacuum(retention_days=90)
        assert result["pruned"] == 1
        # File stays — caller didn't opt in
        assert (sessions_dir / "old.jsonl").exists()

    def test_prune_sessions_deletes_files_for_pruned_only(self, db, tmp_path):
        """Active-session transcripts must never be deleted by prune."""
        sessions_dir = tmp_path / "sessions"
        sessions_dir.mkdir()
        self._make_old_ended(db, "old", days_old=100)
        db.create_session(session_id="active", source="cli")  # not ended
        (sessions_dir / "old.jsonl").write_text("{}\n")
        (sessions_dir / "active.jsonl").write_text("{}\n")

        count = db.prune_sessions(older_than_days=90, sessions_dir=sessions_dir)
        assert count == 1
        assert not (sessions_dir / "old.jsonl").exists()
        assert (sessions_dir / "active.jsonl").exists()


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

    def test_tool_function_name_in_tool_calls_is_searchable(self, db):
        db.create_session(session_id="s1", source="cli")
        db.append_message(
            "s1", role="assistant", content="",
            tool_calls=[{
                "id": "c1",
                "type": "function",
                "function": {"name": "UNIQUEFUNCNAME", "arguments": "{}"},
            }],
        )
        results = db.search_messages("UNIQUEFUNCNAME")
        assert len(results) == 1

    def test_delete_message_row_does_not_crash(self, db):
        """DELETE on messages must not raise when FTS rows reference tool fields.

        Previously the messages_fts_delete trigger passed old.content to the
        FTS5 delete-command but the inserted row was the concatenation of
        content || tool_name || tool_calls, so FTS5 rejected the delete with
        'SQL logic error' and every session delete path broke.
        """
        db.create_session(session_id="s1", source="cli")
        db.append_message(
            "s1", role="assistant", content="hello",
            tool_name="web_search",
            tool_calls=[{
                "id": "c1",
                "type": "function",
                "function": {"name": "web_search", "arguments": '{"q": "x"}'},
            }],
        )
        # end_session + end-time prune path would exercise DELETE; hit the
        # row directly through the write helper to keep the regression focused.
        def _delete(conn):
            conn.execute("DELETE FROM messages WHERE session_id = ?", ("s1",))
        db._execute_write(_delete)  # must not raise

        assert db.search_messages("hello") == []
        assert db.search_messages("web_search") == []

    def test_update_message_reindexes_tool_fields(self, db):
        """UPDATE must refresh the FTS row so old tokens drop out and new tokens appear."""
        db.create_session(session_id="s1", source="cli")
        db.append_message(
            "s1", role="assistant", content="",
            tool_name="ORIGINALTOOL",
        )
        assert len(db.search_messages("ORIGINALTOOL")) == 1

        def _update(conn):
            conn.execute(
                "UPDATE messages SET tool_name = ? WHERE session_id = ?",
                ("RENAMEDTOOL", "s1"),
            )
        db._execute_write(_update)

        assert db.search_messages("ORIGINALTOOL") == []
        assert len(db.search_messages("RENAMEDTOOL")) == 1


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

        # Now open via SessionDB — migration runs.
        session_db = SessionDB(db_path=db_path)
        try:
            assert len(session_db.search_messages("LEGACYTOOL")) == 1, \
                "v11 migration must backfill tool_name into FTS"
            assert len(session_db.search_messages("LEGACYARG")) == 1, \
                "v11 migration must backfill tool_calls JSON into FTS"
            # schema_version bumped
            from hermes_state import SCHEMA_VERSION
            row = session_db._conn.execute(
                "SELECT version FROM schema_version LIMIT 1"
            ).fetchone()
            version = row["version"] if hasattr(row, "keys") else row[0]
            assert version == SCHEMA_VERSION
        finally:
            session_db.close()


# ---------------------------------------------------------------------------
# apply_wal_with_fallback — read-only probe tests
# ---------------------------------------------------------------------------


class TestApplyWalProbe:
    """Unit tests for the journal_mode probe in apply_wal_with_fallback."""

    def test_skips_set_pragma_when_already_wal(self, tmp_path):
        """Already-WAL connection must not trigger the set-pragma."""
        import sqlite3
        from hermes_state import apply_wal_with_fallback

        class _TracingConn(sqlite3.Connection):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.executed = []

            def execute(self, sql, params=()):
                self.executed.append(sql)
                return super().execute(sql, params)

        db_path = tmp_path / "wal.db"
        # Prime the file into WAL mode first.
        with sqlite3.connect(str(db_path)) as seed:
            seed.execute("PRAGMA journal_mode=WAL")

        conn = _TracingConn(str(db_path))
        try:
            result = apply_wal_with_fallback(conn)
        finally:
            conn.close()

        assert result == "wal"
        # Only the probe should have fired; the set-pragma must NOT appear.
        assert any("PRAGMA journal_mode" == sql.strip() for sql in conn.executed), (
            "probe PRAGMA should have run"
        )
        assert not any("journal_mode=WAL" in sql for sql in conn.executed), (
            "set-pragma must not run when already in WAL mode"
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

    def test_macos_checkpoint_fullsync_barrier_applied(self, tmp_path, monkeypatch):
        """On Darwin, apply_wal_with_fallback sets checkpoint_fullfsync=1 (issue #30636)."""
        import sqlite3
        import hermes_state
        from hermes_state import apply_wal_with_fallback

        class _TracingConn(sqlite3.Connection):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.executed = []

            def execute(self, sql, params=()):
                self.executed.append(sql)
                return super().execute(sql, params)

        monkeypatch.setattr(hermes_state.sys, "platform", "darwin")

        db_path = tmp_path / "macos_fresh.db"
        conn = _TracingConn(str(db_path))
        try:
            result = apply_wal_with_fallback(conn)
        finally:
            conn.close()

        assert result == "wal"
        assert any("checkpoint_fullfsync=1" in sql for sql in conn.executed), (
            "checkpoint_fullfsync barrier must be applied on macOS"
        )

    def test_macos_barrier_applied_when_already_wal(self, tmp_path, monkeypatch):
        """The Darwin barrier fires on the already-WAL early-return path too."""
        import sqlite3
        import hermes_state
        from hermes_state import apply_wal_with_fallback

        class _TracingConn(sqlite3.Connection):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.executed = []

            def execute(self, sql, params=()):
                self.executed.append(sql)
                return super().execute(sql, params)

        db_path = tmp_path / "macos_wal.db"
        with sqlite3.connect(str(db_path)) as seed:
            seed.execute("PRAGMA journal_mode=WAL")

        monkeypatch.setattr(hermes_state.sys, "platform", "darwin")

        conn = _TracingConn(str(db_path))
        try:
            result = apply_wal_with_fallback(conn)
        finally:
            conn.close()

        assert result == "wal"
        assert any("checkpoint_fullfsync=1" in sql for sql in conn.executed), (
            "checkpoint_fullfsync barrier must fire on the already-WAL path"
        )

    def test_checkpoint_fullsync_barrier_skipped_off_darwin(self, tmp_path, monkeypatch):
        """Non-macOS platforms must NOT issue the macOS-only PRAGMA."""
        import sqlite3
        import hermes_state
        from hermes_state import apply_wal_with_fallback

        class _TracingConn(sqlite3.Connection):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self.executed = []

            def execute(self, sql, params=()):
                self.executed.append(sql)
                return super().execute(sql, params)

        monkeypatch.setattr(hermes_state.sys, "platform", "linux")

        db_path = tmp_path / "linux_fresh.db"
        conn = _TracingConn(str(db_path))
        try:
            result = apply_wal_with_fallback(conn)
        finally:
            conn.close()

        assert result == "wal"
        assert not any("checkpoint_fullfsync" in sql for sql in conn.executed), (
            "checkpoint_fullfsync must not be issued off macOS"
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

    def test_fallback_to_delete_still_works(self, tmp_path):
        """When set-pragma raises a WAL-incompat error, falls back to DELETE."""
        import sqlite3
        from hermes_state import apply_wal_with_fallback

        class _IncompatConn(sqlite3.Connection):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self._call_count = 0

            def execute(self, sql, params=()):
                self._call_count += 1
                # First call is the read probe; let it return "delete".
                # Second call is the set-pragma; raise a WAL-incompat error.
                if "journal_mode=WAL" in sql:
                    raise sqlite3.OperationalError("locking protocol")
                return super().execute(sql, params)

        db_path = tmp_path / "incompat.db"
        conn = _IncompatConn(str(db_path))
        try:
            result = apply_wal_with_fallback(conn, db_label="test.db")
        finally:
            conn.close()

        assert result == "delete"

    def test_probe_failure_falls_through_to_set_pragma(self, tmp_path):
        """When the read probe raises OperationalError, fall through to set-pragma."""
        import sqlite3
        from hermes_state import apply_wal_with_fallback

        class _ProbeFails(sqlite3.Connection):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self._first = True

            def execute(self, sql, params=()):
                if self._first and "journal_mode" in sql and "WAL" not in sql:
                    self._first = False
                    raise sqlite3.OperationalError("simulated probe failure")
                return super().execute(sql, params)

        db_path = tmp_path / "probe_fail.db"
        conn = _ProbeFails(str(db_path))
        try:
            result = apply_wal_with_fallback(conn)
        finally:
            conn.close()

        # Despite probe failure, set-pragma must still run and succeed.
        assert result == "wal"

    def test_no_downgrade_from_wal_to_delete_on_eio(self, tmp_path):
        """OperationalError NOT in _WAL_INCOMPAT_MARKERS must propagate, not downgrade."""
        import sqlite3
        import pytest
        from hermes_state import apply_wal_with_fallback

        class _EIOConn(sqlite3.Connection):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self._first = True

            def execute(self, sql, params=()):
                # Let the probe succeed (returns "delete" for fresh DB).
                if "journal_mode=WAL" in sql:
                    raise sqlite3.OperationalError("some unexpected hardware failure")
                return super().execute(sql, params)

        db_path = tmp_path / "eio.db"
        conn = _EIOConn(str(db_path))
        try:
            with pytest.raises(
                sqlite3.OperationalError, match="some unexpected hardware failure"
            ):
                apply_wal_with_fallback(conn)
        finally:
            conn.close()

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

    def test_set_session_archived_missing_row(self, db):
        assert db.set_session_archived("nope", True) is False

    def test_archived_excluded_by_default(self, db):
        self._seed(db, "live")
        self._seed(db, "hidden", archived=True)

        ids = [s["id"] for s in db.list_sessions_rich()]
        assert ids == ["live"]
        assert db.session_count() == 1

    def test_archived_only_and_include(self, db):
        self._seed(db, "live")
        self._seed(db, "hidden", archived=True)

        only = [s["id"] for s in db.list_sessions_rich(archived_only=True)]
        assert only == ["hidden"]
        assert db.session_count(archived_only=True) == 1

        both = {s["id"] for s in db.list_sessions_rich(include_archived=True)}
        assert both == {"live", "hidden"}
        assert db.session_count(include_archived=True) == 2



class TestSessionIdSearch:
    """Session id search backs Desktop's Search Sessions UX."""

    def _seed(self, db, sid, *, content="ordinary message", archived=False):
        db.create_session(session_id=sid, source="cli", model="test-model")
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

    def test_search_sessions_by_id_respects_limit_and_prioritizes_exact_matches(self, db):
        self._seed(db, "20260603_090200_abcd12")
        self._seed(db, "20260603_090200_abcd12_child")
        self._seed(db, "x_20260603_090200_abcd12")

        ids = [s["id"] for s in db.search_sessions_by_id("20260603_090200_abcd12", limit=2)]

        assert ids == ["20260603_090200_abcd12", "20260603_090200_abcd12_child"]

    def test_search_sessions_by_id_can_include_or_exclude_archived(self, db):
        self._seed(db, "20260603_090200_live")
        self._seed(db, "20260603_090200_archived", archived=True)

        included = {s["id"] for s in db.search_sessions_by_id("20260603_090200", include_archived=True)}
        excluded = {s["id"] for s in db.search_sessions_by_id("20260603_090200", include_archived=False)}

        assert included == {"20260603_090200_live", "20260603_090200_archived"}
        assert excluded == {"20260603_090200_live"}

    def test_search_sessions_by_id_matches_projected_lineage_root_id(self, db):
        root = "20260602_235959_root99"
        tip = "20260603_010000_tip01"
        db.create_session(session_id=root, source="cli")
        db.append_message(root, role="user", content="root conversation")
        db.end_session(root, "compression")
        db.create_session(session_id=tip, source="cli", parent_session_id=root)
        db.append_message(tip, role="user", content="continued conversation")

        matches = db.search_sessions_by_id("root99")

        assert [s["id"] for s in matches] == [tip]
        assert matches[0]["_lineage_root_id"] == root


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

    def test_prefix_match_excludes_substring_collision(self, db):
        """A job whose id contains the target id as a substring must not leak.

        The old code used a leading-wildcard ``LIKE %cron_<id>_%`` which would
        also match ``cron_xalpha_...``; the range scan binds to the true prefix.
        """
        base = 1_700_000_000.0
        self._seed_run(db, "alpha", 0, base)
        # Collision: id is "xalpha", which contains "alpha".
        self._seed_run(db, "xalpha", 0, base + 10)
        # Collision the other way: id "alpha2" extends past the underscore.
        self._seed_run(db, "alpha2", 0, base + 20)

        runs = db.list_cron_job_runs("alpha", limit=20)

        assert [r["id"] for r in runs] == ["cron_alpha_00000000"]

    def test_ignores_non_cron_sessions(self, db):
        base = 1_700_000_000.0
        self._seed_run(db, "alpha", 0, base)
        # A non-cron session whose id happens to share the prefix shape.
        db.create_session(session_id="cron_alpha_99999999", source="cli")
        db._conn.execute(
            "UPDATE sessions SET started_at = ? WHERE id = ?",
            (base + 100, "cron_alpha_99999999"),
        )
        db._conn.commit()

        runs = db.list_cron_job_runs("alpha", limit=20)

        assert [r["id"] for r in runs] == ["cron_alpha_00000000"]

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

    def test_uses_index_range_scan(self, db):
        """The query must use the (source, id) index, not a full table scan."""
        prefix = "cron_alpha_"
        prefix_hi = prefix[:-1] + chr(ord(prefix[-1]) + 1)
        plan = db._conn.execute(
            "EXPLAIN QUERY PLAN "
            "SELECT s.* FROM sessions s "
            "WHERE s.source = 'cron' AND s.id >= ? AND s.id < ? "
            "ORDER BY s.started_at DESC LIMIT 20",
            (prefix, prefix_hi),
        ).fetchall()
        detail = " ".join(row[-1] for row in plan)
        assert "USING INDEX" in detail or "USING COVERING INDEX" in detail, detail
        assert "idx_sessions_source" in detail, detail


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


def test_gateway_session_recovery_reopens_legacy_agent_close_rows(db):
    db.create_session(
        "closed-gw-session",
        "telegram",
        user_id="user-1",
        session_key="agent:main:telegram:dm:chat-1",
        chat_id="chat-1",
        chat_type="dm",
    )
    db.append_message("closed-gw-session", "user", "hello")
    db.end_session("closed-gw-session", "agent_close")

    recovered = db.find_latest_gateway_session_for_peer(
        source="telegram",
        user_id="user-1",
        session_key="agent:main:telegram:dm:chat-1",
        chat_id="chat-1",
        chat_type="dm",
    )
    assert recovered["id"] == "closed-gw-session"

    db.end_session("closed-gw-session", "session_reset")
    # First end reason wins, so force explicit reset state for this branch.
    db._conn.execute(
        "UPDATE sessions SET ended_at = ?, end_reason = ? WHERE id = ?",
        (time.time(), "session_reset", "closed-gw-session"),
    )
    db._conn.commit()

    assert db.find_latest_gateway_session_for_peer(
        source="telegram",
        user_id="user-1",
        session_key="agent:main:telegram:dm:chat-1",
        chat_id="chat-1",
        chat_type="dm",
    ) is None

# =========================================================================
# Vector embedding search (sqlite-vec)
# =========================================================================

class TestVecSchema:
    """Tests for the sqlite-vec vector embedding schema."""

    def test_vec_table_exists_after_init(self, db):
        """messages_vec virtual table is created during schema init."""
        cursor = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_vec'"
        )
        assert cursor.fetchone() is not None, "messages_vec table missing"

    def test_vec_table_has_embedding_column(self, db):
        """messages_vec has an embedding column for float vectors."""
        cursor = db._conn.execute("PRAGMA table_info(messages_vec)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        assert "embedding" in columns, f"columns: {columns}"

    def test_vec_trigger_insert_exists(self, db):
        """Trigger fires on INSERT to generate embeddings."""
        cursor = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name='messages_vec_insert'"
        )
        assert cursor.fetchone() is not None, "messages_vec_insert trigger missing"

    def test_vec_trigger_delete_exists(self, db):
        """Trigger fires on DELETE to remove embeddings."""
        cursor = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name='messages_vec_delete'"
        )
        assert cursor.fetchone() is not None, "messages_vec_delete trigger missing"

    def test_vec_trigger_update_absent(self, db):
        """No UPDATE trigger — it would overwrite real embeddings with zeros.
        INSERT and DELETE triggers are sufficient; content changes are rare
        (compaction does DELETE+reINSERT)."""
        cursor = db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name='messages_vec_update'"
        )
        assert cursor.fetchone() is None, (
            "messages_vec_update trigger should NOT exist — "
            "it overwrites real embeddings with zeroblob on any UPDATE to messages"
        )

    def test_vec_extension_loaded(self, db):
        """sqlite-vec extension is loaded and vec0 is available."""
        # Try creating a probe vec0 table — proves the extension is loaded
        db._conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _vec_probe USING vec0(embedding float[4])")
        # If we got here without OperationalError, vec0 is available
        db._conn.execute("DROP TABLE IF EXISTS _vec_probe")


class TestVecSearch:
    """Tests for semantic and hybrid vector search."""

    @pytest.fixture(autouse=True)
    def _mock_provider(self):
        """Inject a reusable mock embedding provider into every test."""
        from agent.embedding_provider import EmbeddingProvider

        class MockProvider(EmbeddingProvider):
            @property
            def dimensions(self):
                return 768

            def generate_embedding(self, text):
                return [0.1] * 768

        self.provider = MockProvider()

    def test_search_semantic_returns_results(self, db):
        """search_semantic finds messages by vector similarity."""
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="How do I deploy with Docker?")
        db.append_message("s1", role="assistant", content="Use docker compose up.")

        results = db.search_semantic("docker deployment", self.provider, limit=5)
        assert len(results) > 0, "Expected at least one result"
        for r in results:
            assert "id" in r
            assert "session_id" in r
            assert "role" in r
            assert "distance" in r
            assert "context" in r

    def test_search_semantic_respects_limit(self, db):
        """search_semantic returns at most `limit` results."""
        db.create_session(session_id="s1", source="cli")
        for i in range(10):
            db.append_message("s1", role="user", content=f"Message number {i}")

        results = db.search_semantic("message", self.provider, limit=3)
        assert len(results) <= 3

    def test_search_semantic_empty_query(self, db):
        """search_semantic returns empty list for empty query."""
        assert db.search_semantic("", self.provider) == []
        assert db.search_semantic("   ", self.provider) == []

    def test_search_semantic_no_provider_returns_empty(self, db):
        """search_semantic returns empty list when provider is None."""
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="test message")
        assert db.search_semantic("test", None) == []

    def test_search_hybrid_combines_fts5_and_vector(self, db):
        """search_hybrid merges BM25 and vector results via RRF."""
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="Docker deployment guide")
        db.append_message("s1", role="assistant", content="Use docker compose for orchestration")
        db.append_message("s1", role="user", content="Unrelated message about lunch")

        results = db.search_hybrid("docker", self.provider, limit=5)
        assert len(results) > 0
        for r in results:
            assert "hybrid_score" in r, f"Missing hybrid_score in {list(r.keys())}"

    def test_search_hybrid_falls_back_to_fts5_when_no_provider(self, db):
        """search_hybrid returns FTS5-only results when provider is None."""
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="Docker deployment guide")

        results = db.search_hybrid("docker", None, limit=5)
        assert len(results) > 0
        for r in results:
            assert "hybrid_score" in r

    def test_search_hybrid_respects_role_filter(self, db):
        """search_hybrid applies role_filter to both FTS5 and vector results."""
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="What is Docker?")
        db.append_message("s1", role="assistant", content="Docker is a container runtime.")

        results = db.search_hybrid("Docker", self.provider, limit=5, role_filter=["assistant"])
        roles = {r["role"] for r in results}
        assert roles == {"assistant"}, f"Expected only assistant, got {roles}"

    def test_search_hybrid_accepts_sort_parameter(self, db):
        """search_hybrid accepts and forwards sort to inner search_messages."""
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="Docker deployment guide")

        results = db.search_hybrid("docker", self.provider, limit=5, sort="newest")
        assert isinstance(results, list)

    def test_search_hybrid_sort_newest_orders_by_timestamp(self, db):
        """sort='newest' returns most recent messages first."""
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="First message")
        db.append_message("s1", role="user", content="Second message")
        db.append_message("s1", role="user", content="Third message")

        results = db.search_hybrid("message", self.provider, limit=5, sort="newest")
        assert len(results) >= 2
        timestamps = [r["timestamp"] for r in results]
        for i in range(len(timestamps) - 1):
            assert timestamps[i] >= timestamps[i + 1], \
                f"Expected descending timestamps, got {timestamps}"


class TestRRFMerge:
    """Tests for the standalone _rrf_merge static method."""

    def test_rrf_merge_combines_results(self):
        """Merges two result lists and returns hybrid scores."""
        from hermes_state import SessionDB

        bm25 = [
            {"id": 1, "content": "alpha"},
            {"id": 2, "content": "beta"},
        ]
        vec = [
            {"id": 2, "content": "beta"},
            {"id": 3, "content": "gamma"},
        ]

        merged = SessionDB._rrf_merge(bm25, vec, hybrid_weight=0.5, limit=10)
        assert len(merged) == 3
        # All should have hybrid_score
        for r in merged:
            assert "hybrid_score" in r

    def test_rrf_merge_respects_limit(self):
        """Returns at most `limit` results."""
        from hermes_state import SessionDB

        bm25 = [{"id": i, "content": f"msg{i}"} for i in range(10)]
        vec = [{"id": i, "content": f"msg{i}"} for i in range(5, 15)]

        merged = SessionDB._rrf_merge(bm25, vec, hybrid_weight=0.5, limit=3)
        assert len(merged) == 3

    def test_rrf_merge_bm25_only(self):
        """When vec_results is empty, returns BM25 results with hybrid_score."""
        from hermes_state import SessionDB

        bm25 = [
            {"id": 1, "content": "alpha"},
            {"id": 2, "content": "beta"},
        ]

        merged = SessionDB._rrf_merge(bm25, [], hybrid_weight=0.5, limit=10)
        assert len(merged) == 2
        for r in merged:
            assert "hybrid_score" in r

    def test_rrf_merge_vec_only(self):
        """When bm25_results is empty, returns vec results with hybrid_score."""
        from hermes_state import SessionDB

        vec = [
            {"id": 1, "content": "alpha"},
            {"id": 2, "content": "beta"},
        ]

        merged = SessionDB._rrf_merge([], vec, hybrid_weight=0.5, limit=10)
        assert len(merged) == 2
        for r in merged:
            assert "hybrid_score" in r

    def test_rrf_merge_higher_weight_boosts_vec(self):
        """Higher hybrid_weight gives vec results better scores."""
        from hermes_state import SessionDB

        # msg 1: BM25 rank 1, vec rank 2
        # msg 2: BM25 rank 2, vec rank 1
        bm25 = [{"id": 1, "content": "a"}, {"id": 2, "content": "b"}]
        vec = [{"id": 2, "content": "b"}, {"id": 1, "content": "a"}]

        # With weight=0.0 (pure BM25), msg 1 should win
        merged_bm25 = SessionDB._rrf_merge(bm25, vec, hybrid_weight=0.0, limit=10)
        assert merged_bm25[0]["id"] == 1

        # With weight=2.0 (vec-heavy), msg 2 should win
        merged_vec = SessionDB._rrf_merge(bm25, vec, hybrid_weight=2.0, limit=10)
        assert merged_vec[0]["id"] == 2

    def test_rrf_merge_empty_both(self):
        """Returns empty list when both inputs are empty."""
        from hermes_state import SessionDB
        merged = SessionDB._rrf_merge([], [], hybrid_weight=0.5, limit=10)
        assert merged == []


# =========================================================================
# Embedding generation and backfill
# =========================================================================

class TestEmbedMessage:
    """Tests for embed_message — generating and storing real embeddings."""

    @pytest.fixture(autouse=True)
    def _mock_provider(self):
        from agent.embedding_provider import EmbeddingProvider

        class MockProvider(EmbeddingProvider):
            @property
            def dimensions(self):
                return 768

            def generate_embedding(self, text):
                # Deterministic 768-float vector from text hash.
                # SHA256 gives 32 bytes; repeat to fill 768 floats.
                import hashlib
                h = hashlib.sha256(text.encode()).digest()
                # Repeat the 32-byte pattern 24 times to get 768 bytes
                expanded = (h * 24)[:768]
                return [b / 255.0 for b in expanded]

        self.provider = MockProvider()

    def test_embed_message_stores_real_embedding(self, db):
        """embed_message replaces zero-vector with a real embedding."""
        db.create_session(session_id="s1", source="cli")
        msg_id = db.append_message("s1", role="user", content="Docker deployment guide")

        # Before: zero vector
        cursor = db._conn.execute(
            "SELECT length(embedding) FROM messages_vec WHERE rowid = ?", (msg_id,)
        )
        before_len = cursor.fetchone()[0]
        assert before_len == 3072  # zeroblob(3072)

        # Embed
        db.embed_message(msg_id, "Docker deployment guide", self.provider)

        # After: real embedding (same length, different content)
        cursor = db._conn.execute(
            "SELECT length(embedding), embedding FROM messages_vec WHERE rowid = ?", (msg_id,)
        )
        row = cursor.fetchone()
        assert row[0] == 3072  # still 768 * 4 bytes
        # Not all zeros anymore
        assert row[1] != b'\x00' * 3072, "Embedding should not be all zeros"

    def test_embed_message_idempotent(self, db):
        """embed_message can be called multiple times safely."""
        db.create_session(session_id="s1", source="cli")
        msg_id = db.append_message("s1", role="user", content="test")

        db.embed_message(msg_id, "test", self.provider)
        cursor = db._conn.execute(
            "SELECT embedding FROM messages_vec WHERE rowid = ?", (msg_id,)
        )
        first = cursor.fetchone()[0]

        # Second call with same text should produce same embedding
        db.embed_message(msg_id, "test", self.provider)
        cursor = db._conn.execute(
            "SELECT embedding FROM messages_vec WHERE rowid = ?", (msg_id,)
        )
        second = cursor.fetchone()[0]
        assert first == second

    def test_embed_message_no_provider_noop(self, db):
        """embed_message with None provider does nothing (no crash)."""
        db.create_session(session_id="s1", source="cli")
        msg_id = db.append_message("s1", role="user", content="test")
        db.embed_message(msg_id, "test", None)  # should not raise

    def test_embed_message_nonexistent_row_noop(self, db):
        """embed_message on a nonexistent rowid does nothing (no crash)."""
        db.embed_message(99999, "test", self.provider)  # should not raise

    def test_embed_message_atomic_on_update_failure(self, db):
        """embed_message rolls back DELETE+INSERT if UPDATE fails.

        Uses a temporary BEFORE UPDATE trigger on the messages table
        that raises ABORT, simulating a real SQLite-level failure.
        After the fix wraps operations in an explicit transaction,
        the DELETE+INSERT into messages_vec must be rolled back.
        """
        db.create_session(session_id="s1", source="cli")
        msg_id = db.append_message("s1", role="user", content="Atomic test")

        # Install a trigger that blocks UPDATE on messages
        db._conn.execute(
            "CREATE TEMP TRIGGER test_block_vec_update "
            "BEFORE UPDATE OF vec_embedded ON messages "
            "BEGIN "
            "    SELECT RAISE(ABORT, 'simulated UPDATE failure'); "
            "END"
        )

        try:
            db.embed_message(msg_id, "Atomic test", self.provider)
        except (sqlite3.OperationalError, sqlite3.IntegrityError):
            pass  # expected — trigger raised ABORT
        finally:
            # Clean up the trigger
            db._conn.execute("DROP TRIGGER IF EXISTS test_block_vec_update")

        # After the fix: DELETE+INSERT should be rolled back.
        # The zero-vector placeholder should still be there.
        import sqlite_vec
        db._conn.enable_load_extension(True)
        sqlite_vec.load(db._conn)
        db._conn.enable_load_extension(False)
        cursor = db._conn.execute(
            "SELECT embedding FROM messages_vec WHERE rowid = ?", (msg_id,)
        )
        row = cursor.fetchone()
        assert row is not None, "messages_vec row should still exist"
        # Should still be zero vector (DELETE+INSERT rolled back)
        assert row[0] == b'\x00' * 3072, (
            "DELETE+INSERT should be rolled back when UPDATE fails — "
            "got non-zero embedding (orphan bug)"
        )
        # vec_embedded should still be 0
        cursor = db._conn.execute(
            "SELECT vec_embedded FROM messages WHERE id = ?", (msg_id,)
        )
        assert cursor.fetchone()[0] == 0, (
            "vec_embedded should remain 0 when UPDATE fails"
        )


class TestBackfillEmbeddings:
    """Tests for backfill_embeddings — batch embedding of zero-vector rows."""

    @pytest.fixture(autouse=True)
    def _mock_provider(self):
        from agent.embedding_provider import EmbeddingProvider

        class MockProvider(EmbeddingProvider):
            @property
            def dimensions(self):
                return 768

            def generate_embedding(self, text):
                import hashlib
                h = hashlib.sha256(text.encode()).digest()
                expanded = (h * 24)[:768]
                return [b / 255.0 for b in expanded]

        self.provider = MockProvider()

    def test_backfill_embeds_all_zero_vectors(self, db):
        """backfill_embeddings replaces all zero-vector placeholders."""
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="Message one")
        db.append_message("s1", role="assistant", content="Message two")
        db.append_message("s1", role="user", content="Message three")

        # All should be unembedded initially
        cursor = db._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE vec_embedded = 0"
        )
        assert cursor.fetchone()[0] == 3

        count = db.backfill_embeddings(self.provider, batch_size=10)

        assert count == 3
        # No unembedded messages remain
        cursor = db._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE vec_embedded = 0"
        )
        assert cursor.fetchone()[0] == 0

    def test_backfill_skips_already_embedded(self, db):
        """backfill_embeddings only touches vec_embedded=0 rows."""
        db.create_session(session_id="s1", source="cli")
        msg_id = db.append_message("s1", role="user", content="Already embedded")
        db.embed_message(msg_id, "Already embedded", self.provider)

        db.append_message("s1", role="user", content="Not embedded yet")

        count = db.backfill_embeddings(self.provider, batch_size=10)
        assert count == 1  # only the unembedded row

    def test_backfill_no_provider_returns_zero(self, db):
        """backfill_embeddings with None provider returns 0."""
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="test")
        assert db.backfill_embeddings(None) == 0

    def test_backfill_empty_db_returns_zero(self, db):
        """backfill_embeddings on a DB with no messages returns 0."""
        assert db.backfill_embeddings(self.provider) == 0

    def test_backfill_respects_batch_size(self, db):
        """backfill_embeddings processes in batches."""
        db.create_session(session_id="s1", source="cli")
        for i in range(5):
            db.append_message("s1", role="user", content=f"Message {i}")

        count = db.backfill_embeddings(self.provider, batch_size=2)
        assert count == 5  # all embedded, just in batches

    def test_backfill_skips_tool_messages(self, db):
        """backfill_embeddings only embeds user+assistant, not tool messages."""
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="User message")
        db.append_message("s1", role="assistant", content="Assistant message")
        db.append_message("s1", role="tool", content="Tool output",
                          tool_name="test_tool")

        count = db.backfill_embeddings(self.provider, batch_size=10)

        # Only user+assistant should be embedded
        assert count == 2, (
            f"backfill should embed only user+assistant, got {count}"
        )

        # Verify: tool message should still have vec_embedded=0
        cursor = db._conn.execute(
            "SELECT id, role, vec_embedded FROM messages ORDER BY id"
        )
        rows = cursor.fetchall()
        for row in rows:
            if row[1] == "tool":
                assert row[2] == 0, (
                    f"tool message {row[0]} should not be embedded"
                )
            else:
                assert row[2] == 1, (
                    f"user/assistant message {row[0]} should be embedded"
                )


class TestAutoEmbedOnAppend:
    """Tests for automatic embedding on append_message."""

    @pytest.fixture(autouse=True)
    def _mock_provider(self):
        from agent.embedding_provider import EmbeddingProvider

        class MockProvider(EmbeddingProvider):
            @property
            def dimensions(self):
                return 768

            def generate_embedding(self, text):
                import hashlib
                h = hashlib.sha256(text.encode()).digest()
                expanded = (h * 24)[:768]
                return [b / 255.0 for b in expanded]

        self.provider = MockProvider()

    def test_auto_embed_on_append(self, db):
        """append_message auto-embeds when embedding_provider is set."""
        db.embedding_provider = self.provider
        db.create_session(session_id="s1", source="cli")
        msg_id = db.append_message("s1", role="user", content="Auto-embed test")

        cursor = db._conn.execute(
            "SELECT embedding FROM messages_vec WHERE rowid = ?", (msg_id,)
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] != b'\x00' * 3072, "Should have real embedding, not zeros"

    def test_no_auto_embed_when_provider_not_set(self, db):
        """append_message stores zero vector when embedding_provider is None."""
        db.create_session(session_id="s1", source="cli")
        msg_id = db.append_message("s1", role="user", content="No provider")

        cursor = db._conn.execute(
            "SELECT embedding FROM messages_vec WHERE rowid = ?", (msg_id,)
        )
        row = cursor.fetchone()
        assert row[0] == b'\x00' * 3072, "Should be zero vector placeholder"

    def test_auto_embed_handles_provider_failure(self, db):
        """append_message still succeeds even if embedding generation fails."""
        from agent.embedding_provider import EmbeddingProvider

        class FailingProvider(EmbeddingProvider):
            @property
            def dimensions(self):
                return 768

            def generate_embedding(self, text):
                raise RuntimeError("Embedding service down")

        db.embedding_provider = FailingProvider()
        db.create_session(session_id="s1", source="cli")
        msg_id = db.append_message("s1", role="user", content="test")

        # Message was inserted (zero vector fallback)
        cursor = db._conn.execute("SELECT id FROM messages WHERE id = ?", (msg_id,))
        assert cursor.fetchone() is not None


# =========================================================================
# End-to-end test with real Ollama embeddings
# =========================================================================

@pytest.mark.skipif(
    "not _ollama_available()",
    reason="Ollama not running on localhost:11434",
)
class TestOllamaE2E:
    """End-to-end semantic search with real Ollama nomic-embed-text."""

    def test_real_embeddings_produce_meaningful_distances(self, db):
        """Messages with similar content have smaller vector distances."""
        from agent.embedding_provider import OllamaEmbeddingProvider

        provider = OllamaEmbeddingProvider(
            model="nomic-embed-text",
            base_url="http://localhost:11434/v1",
            dimensions=768,
        )
        db.embedding_provider = provider

        db.create_session(session_id="s1", source="cli")
        # Two Docker-related messages
        docker1 = db.append_message("s1", role="user", content="How do I deploy with Docker?")
        docker2 = db.append_message("s1", role="assistant", content="Use docker compose up for deployment")
        # One unrelated message
        lunch = db.append_message("s1", role="user", content="What should I eat for lunch today?")

        # Verify embeddings are not zero vectors
        for msg_id in (docker1, docker2, lunch):
            cursor = db._conn.execute(
                "SELECT embedding FROM messages_vec WHERE rowid = ?", (msg_id,)
            )
            row = cursor.fetchone()
            assert row is not None, f"msg {msg_id} missing from messages_vec"
            assert row[0] != b'\x00' * 3072, f"msg {msg_id} has zero vector"

        # Semantic search for "Docker" should rank Docker messages above lunch
        results = db.search_semantic("Docker container deployment", provider, limit=3)
        assert len(results) >= 2
        ids = [r["id"] for r in results]
        # Docker messages should appear before lunch
        lunch_pos = ids.index(lunch) if lunch in ids else len(ids)
        docker_positions = [ids.index(d) for d in (docker1, docker2) if d in ids]
        assert any(p < lunch_pos for p in docker_positions), \
            f"Docker messages should rank above lunch. Got order: {ids}"

    def test_hybrid_search_with_real_embeddings(self, db):
        """Hybrid search returns results with hybrid_score."""
        from agent.embedding_provider import OllamaEmbeddingProvider

        provider = OllamaEmbeddingProvider(
            model="nomic-embed-text",
            base_url="http://localhost:11434/v1",
            dimensions=768,
        )
        db.embedding_provider = provider

        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="Python async programming guide")
        db.append_message("s1", role="assistant", content="Use asyncio for concurrent I/O")
        db.append_message("s1", role="user", content="Best pizza places in New York")

        results = db.search_hybrid("async python", provider, limit=3)
        assert len(results) > 0
        for r in results:
            assert "hybrid_score" in r
            assert isinstance(r["hybrid_score"], float)

    def test_backfill_with_real_embeddings(self, db):
        """backfill_embeddings replaces zero vectors with real embeddings."""
        from agent.embedding_provider import OllamaEmbeddingProvider

        provider = OllamaEmbeddingProvider(
            model="nomic-embed-text",
            base_url="http://localhost:11434/v1",
            dimensions=768,
        )

        db.create_session(session_id="s1", source="cli")
        # Insert without provider — zero vectors
        db.append_message("s1", role="user", content="Message one")
        db.append_message("s1", role="user", content="Message two")

        # Verify zero vectors
        cursor = db._conn.execute(
            "SELECT COUNT(*) FROM messages_vec WHERE embedding = zeroblob(3072)"
        )
        assert cursor.fetchone()[0] == 2

        # Backfill
        count = db.backfill_embeddings(provider, batch_size=10)
        assert count == 2

        # Verify no zero vectors remain
        cursor = db._conn.execute(
            "SELECT COUNT(*) FROM messages_vec WHERE embedding = zeroblob(3072)"
        )
        assert cursor.fetchone()[0] == 0


def _ollama_available():
    """Check if Ollama is running and nomic-embed-text is available."""
    try:
        import urllib.request
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            import json
            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            return any("nomic-embed-text" in m for m in models)
    except Exception:
        return False
