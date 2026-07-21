"""Runtime FTS-corruption self-heal on the SessionDB write path (#65637 class).

A corrupted FTS5 shadow table (``messages_fts_data``) makes every message
write raise ``sqlite3.DatabaseError: database disk image is malformed``
through the FTS sync triggers, while the canonical ``messages`` rows stay
intact. Before this fix the gateway swallowed the failure at debug level and
the in-memory session advanced while disk silently fell behind — surfacing
later as "Persisted transcript lagged live cached history" amnesia.

The fix: ``_execute_write`` detects the malformed-image class, performs a
one-shot in-place FTS rebuild (FTS5 ``'rebuild'`` command — index rewritten
from canonical rows, no messages touched), and retries the failed write.
"""

import sqlite3

import pytest

from hermes_state import SessionDB


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


def _message_contents(db_path):
    raw = sqlite3.connect(str(db_path))
    rows = raw.execute("SELECT content FROM messages ORDER BY id").fetchall()
    raw.close()
    return [r[0] for r in rows]


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

    def test_rebuild_is_one_shot_per_instance(self, db, tmp_path):
        if not db._fts_enabled:
            pytest.skip("FTS5 unavailable in this build")
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "seed")
        _corrupt_fts(tmp_path / "state.db")
        db.append_message("s1", "user", "first heal")  # consumes the one shot
        assert db._fts_runtime_rebuild_attempted is True

        # Corrupt again: the guard must NOT loop — the write now propagates.
        _corrupt_fts(tmp_path / "state.db")
        with pytest.raises(sqlite3.DatabaseError):
            db.append_message("s1", "user", "second corruption")

    def test_non_fts_errors_still_propagate(self, db):
        db.create_session("s1", source="test")

        def _bad(conn):
            raise sqlite3.IntegrityError("NOT NULL constraint failed: x.y")

        with pytest.raises(sqlite3.IntegrityError):
            db._execute_write(_bad)
        # The guard must not have been consumed by an unrelated error class.
        assert db._fts_runtime_rebuild_attempted is False

    def test_lock_retry_path_unchanged(self, db):
        """A locked error still follows the jitter-retry path, untouched by
        the DatabaseError handler (OperationalError is caught first)."""
        calls = {"n": 0}

        def _flaky(conn):
            calls["n"] += 1
            if calls["n"] < 3:
                raise sqlite3.OperationalError("database is locked")
            return "ok"

        assert db._execute_write(_flaky) == "ok"
        assert calls["n"] == 3
        assert db._fts_runtime_rebuild_attempted is False
