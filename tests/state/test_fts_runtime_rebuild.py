"""Runtime FTS-corruption self-heal on the SessionDB write path (#65637 class).

A corrupted FTS5 shadow table (``messages_fts_data``) makes every message
write raise ``sqlite3.DatabaseError: database disk image is malformed``
through the FTS sync triggers, while the canonical ``messages`` rows stay
intact. Before this fix the gateway swallowed the failure at debug level and
the in-memory session advanced while disk silently fell behind — surfacing
later as "Persisted transcript lagged live cached history" amnesia.

The fix: ``_execute_write`` first attempts a one-shot in-place FTS rebuild.
If corruption persists, it records a durable stale marker, detaches the FTS
sync triggers, and retries the canonical write. Search degrades to ``LIKE``
until a later open atomically rebuilds the index and restores the triggers.
"""

import sqlite3

import pytest

from hermes_state import (
    FTS_STALE_KEY,
    LEGACY_FTS_SQL,
    LEGACY_FTS_TRIGRAM_SQL,
    SCHEMA_SQL,
    SessionDB,
    _FTS_TRIGGERS,
)


@pytest.fixture
def db(tmp_path):
    d = SessionDB(db_path=tmp_path / "state.db")
    yield d
    try:
        d.close()
    except Exception:
        pass


def _corrupt_fts(db_path):
    raw = sqlite3.connect(str(db_path))
    raw.execute(
        "UPDATE messages_fts_data SET block = X'DEADBEEFDEADBEEFDEADBEEFDEADBEEF'"
    )
    raw.commit()
    raw.close()


def _corrupt_trigram_fts(db_path):
    raw = sqlite3.connect(str(db_path))
    raw.execute(
        "UPDATE messages_fts_trigram_data "
        "SET block = X'DEADBEEFDEADBEEFDEADBEEFDEADBEEF'"
    )
    raw.commit()
    raw.close()


def _message_contents(db_path):
    raw = sqlite3.connect(str(db_path))
    rows = raw.execute("SELECT content FROM messages ORDER BY id").fetchall()
    raw.close()
    return [r[0] for r in rows]


def _meta_value(db_path, key):
    raw = sqlite3.connect(str(db_path))
    row = raw.execute(
        "SELECT value FROM state_meta WHERE key = ?", (key,)
    ).fetchone()
    raw.close()
    return None if row is None else row[0]


def _base_fts_triggers(db_path):
    raw = sqlite3.connect(str(db_path))
    rows = raw.execute(
        "SELECT name FROM sqlite_master WHERE type = 'trigger' "
        f"AND name IN ({','.join('?' for _ in _FTS_TRIGGERS)})",
        _FTS_TRIGGERS,
    ).fetchall()
    raw.close()
    return {row[0] for row in rows}


class TestRuntimeFtsRebuild:
    def test_corruption_error_classification_covers_both_sqlite_messages(self):
        """SQLite's message for a corrupt FTS index varies by version: older
        builds raise the generic malformed-image error, newer builds raise an
        FTS5-specific one. Both must trigger the self-heal."""
        assert SessionDB._is_fts_write_corruption_error(
            sqlite3.DatabaseError("database disk image is malformed")
        )
        assert SessionDB._is_fts_write_corruption_error(
            sqlite3.DatabaseError(
                'fts5: corrupt structure record for table "messages_fts"'
            )
        )
        assert not SessionDB._is_fts_write_corruption_error(
            sqlite3.DatabaseError("no such table: nothing_fts_related")
        )

    def test_append_self_heals_after_fts_corruption(self, db, tmp_path):
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "hello world")

        _corrupt_fts(tmp_path / "state.db")

        # Before the fix this raised DatabaseError and the row was lost.
        msg_id = db.append_message("s1", "user", "healed append")
        assert msg_id is not None
        assert _message_contents(tmp_path / "state.db") == [
            "hello world",
            "healed append",
        ]

    def test_search_works_after_self_heal(self, db, tmp_path):
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "before corruption")
        _corrupt_fts(tmp_path / "state.db")
        db.append_message("s1", "user", "searchable needle text")

        raw = sqlite3.connect(str(tmp_path / "state.db"))
        hits = raw.execute(
            "SELECT rowid FROM messages_fts WHERE messages_fts MATCH 'needle'"
        ).fetchall()
        raw.close()
        assert len(hits) == 1

    def test_search_messages_self_heals_after_fts_corruption(self, db, tmp_path):
        """A read-only session that only SEARCHES (no write after corruption)
        must self-heal too. The MATCH read raises the corruption class
        (DatabaseError / 'fts5: corrupt structure record'), NOT the
        OperationalError that search_messages caught — so before this fix the
        search crashed until a write or restart rebuilt the index.
        """
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "a searchable needle here")

        _corrupt_fts(tmp_path / "state.db")
        # Injected via a raw connection, so no write on THIS instance has
        # consumed the one-shot rebuild yet.
        assert db._fts_runtime_rebuild_attempted is False

        results = db.search_messages("needle")

        assert db._fts_runtime_rebuild_attempted is True  # the search rebuilt it
        assert results  # non-empty: the rebuilt index matched the query
        assert any("needle" in (r.get("snippet") or "") for r in results)

    def test_trigram_search_self_heals_after_fts_corruption(self, db, tmp_path):
        """The CJK/trigram MATCH branch has the same read-corruption exposure
        as the main FTS5 branch: it caught only OperationalError (query
        syntax), so a corrupt trigram shadow table raised DatabaseError
        straight out of search_messages. It must self-heal via the shared
        one-shot rebuild and answer from the rebuilt trigram index.
        """
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        if not db._trigram_available:
            pytest.skip("trigram tokenizer unavailable in this build")
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "关于大别山项目的进展报告")

        _corrupt_trigram_fts(tmp_path / "state.db")
        assert db._fts_runtime_rebuild_attempted is False

        # >=3 CJK chars per token → routed to the trigram branch.
        results = db.search_messages("大别山项目")

        assert db._fts_runtime_rebuild_attempted is True  # search rebuilt it
        assert results
        # The rebuilt trigram index answered (trigram snippets use >>> <<<),
        # i.e. we did not silently degrade to the LIKE fallback.
        assert any(">>>" in (r.get("snippet") or "") for r in results)


    def test_second_corruption_fails_open_and_rebuilds_on_reopen(
        self, db, tmp_path
    ):
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db_path = tmp_path / "state.db"
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "seed")
        _corrupt_fts(db_path)
        db.append_message("s1", "user", "first heal")  # consumes the one shot
        assert db._fts_runtime_rebuild_attempted is True

        # A second corruption must not strand the canonical transcript. The
        # derived indexes are detached and marked stale instead of looping.
        _corrupt_fts(db_path)
        db.append_message("s1", "user", "second corruption")
        assert _message_contents(db_path) == [
            "seed",
            "first heal",
            "second corruption",
        ]
        assert db._fts_stale is True
        assert _meta_value(db_path, FTS_STALE_KEY) == "1"
        assert _base_fts_triggers(db_path) == set()

        # Search remains available from canonical rows while FTS is stale.
        results = db.search_messages("second corruption")
        assert results
        assert any("second corruption" in row["snippet"] for row in results)

        # A later open atomically rebuilds all canonical rows before triggers
        # return, then clears the durable breadcrumb.
        db.close()
        reopened = SessionDB(db_path=db_path)
        try:
            assert reopened._fts_stale is False
            assert _meta_value(db_path, FTS_STALE_KEY) is None
            assert _base_fts_triggers(db_path) == set(_FTS_TRIGGERS)
            results = reopened.search_messages("second corruption")
            assert results
        finally:
            reopened.close()

    def test_failed_in_place_rebuild_fails_open(self, db, tmp_path, monkeypatch):
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db_path = tmp_path / "state.db"
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "seed")
        _corrupt_fts(db_path)

        def _failed_rebuild():
            raise sqlite3.DatabaseError("rebuild could not read corrupt FTS")

        monkeypatch.setattr(db, "rebuild_fts", _failed_rebuild)
        db.append_message("s1", "user", "canonical survives")

        assert _message_contents(db_path)[-1] == "canonical survives"
        assert _meta_value(db_path, FTS_STALE_KEY) == "1"
        assert _base_fts_triggers(db_path) == set()

    def test_stale_search_preserves_not_semantics(self, db, tmp_path, monkeypatch):
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db_path = tmp_path / "state.db"
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "python language guide")
        db.append_message("s1", "user", "python java interoperability")
        _corrupt_fts(db_path)

        monkeypatch.setattr(
            db,
            "rebuild_fts",
            lambda: (_ for _ in ()).throw(
                sqlite3.DatabaseError("rebuild could not read corrupt FTS")
            ),
        )
        db.append_message("s1", "user", "canonical write survives")
        assert db._fts_stale is True

        results = db.search_messages("python NOT java")
        snippets = [row["snippet"] for row in results]
        assert any("python language guide" in snippet for snippet in snippets)
        assert all("java" not in snippet for snippet in snippets)

    def test_existing_peer_observes_fail_open_marker(
        self, db, tmp_path, monkeypatch
    ):
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db_path = tmp_path / "state.db"
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "seed")
        peer = SessionDB(db_path=db_path)
        try:
            _corrupt_fts(db_path)

            def _failed_rebuild():
                raise sqlite3.DatabaseError("rebuild failed")

            monkeypatch.setattr(db, "rebuild_fts", _failed_rebuild)
            db.append_message("s1", "user", "visible through canonical search")

            assert peer._fts_stale is False
            results = peer.search_messages("canonical search")
            assert peer._fts_stale is True
            assert results
        finally:
            peer.close()

    def test_failed_startup_rebuild_keeps_fts_detached(
        self, db, tmp_path, monkeypatch
    ):
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db_path = tmp_path / "state.db"
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "seed")
        _corrupt_fts(db_path)
        monkeypatch.setattr(
            db,
            "rebuild_fts",
            lambda: (_ for _ in ()).throw(sqlite3.DatabaseError("still corrupt")),
        )
        db.append_message("s1", "user", "before restart")
        db.close()

        monkeypatch.setattr(
            SessionDB,
            "_recover_stale_fts",
            lambda self, cursor, legacy: False,
        )
        reopened = SessionDB(db_path=db_path)
        try:
            assert reopened._fts_stale is True
            assert _meta_value(db_path, FTS_STALE_KEY) == "1"
            assert _base_fts_triggers(db_path) == set()
            reopened.append_message("s1", "user", "after failed recovery")
            assert _message_contents(db_path)[-1] == "after failed recovery"
            assert reopened.search_messages("failed recovery")
        finally:
            reopened.close()

    def test_legacy_inline_fts_fails_open_and_recovers(self, tmp_path, monkeypatch):
        db_path = tmp_path / "legacy-state.db"
        raw = sqlite3.connect(str(db_path))
        raw.executescript(SCHEMA_SQL)
        try:
            raw.executescript(LEGACY_FTS_SQL + LEGACY_FTS_TRIGRAM_SQL)
        except sqlite3.OperationalError as exc:
            raw.close()
            pytest.skip(f"required FTS tokenizer unavailable: {exc}")
        raw.commit()
        raw.close()

        legacy = SessionDB(db_path=db_path)
        try:
            assert legacy._db_has_legacy_inline_fts(legacy._conn.cursor())
            legacy.create_session("s1", source="test")
            legacy.append_message("s1", "user", "legacy seed")
            _corrupt_fts(db_path)
            monkeypatch.setattr(
                legacy,
                "rebuild_fts",
                lambda: (_ for _ in ()).throw(
                    sqlite3.DatabaseError("legacy rebuild failed")
                ),
            )
            legacy.append_message("s1", "user", "legacy canonical survives")
            assert _message_contents(db_path)[-1] == "legacy canonical survives"
            assert _meta_value(db_path, FTS_STALE_KEY) == "1"
        finally:
            legacy.close()

        recovered = SessionDB(db_path=db_path)
        try:
            assert recovered._fts_stale is False
            assert _meta_value(db_path, FTS_STALE_KEY) is None
            assert recovered.search_messages("canonical survives")
        finally:
            recovered.close()

