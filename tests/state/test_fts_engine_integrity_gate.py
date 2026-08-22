"""Engine-version-gated FTS5 integrity verification at SessionDB open (#86027).

Upgrading the SQLite engine underneath an existing state.db (v0.18.2 →
v0.20.1 class) can leave a legacy inline ``messages_fts_trigram`` index that
only the writing engine can read: MATCH keeps answering, trigger writes keep
succeeding, and the damage surfaces only on ``DELETE FROM`` / the FTS5
``'integrity-check'`` command as ``malformed inverted index``. With the sync
triggers intact nothing on the open path ever noticed — the DB silently
carried a broken derived index.

The fix: ``_init_schema`` runs the FTS5 ``'integrity-check'`` special
command on every existing FTS table once per engine version (the passing
version is stamped into state_meta under ``fts_integrity_engine``), repairs
failures in place via ``'rebuild'``, and only then stamps the marker.

CI cannot install the old engine, so the corruption is fabricated by
overwriting the inverted-index blocks (``%_data``) with garbage: the stored
content (``%_content``) stays intact, which is exactly the shape the
'rebuild' repair recovers from, while 'integrity-check' — and, for the
trigram table, a plain ``DELETE FROM`` — raise DatabaseError.
"""

import logging
import sqlite3

import pytest

from hermes_state import (
    FTS_INTEGRITY_ENGINE_KEY,
    LEGACY_FTS_SQL,
    LEGACY_FTS_TRIGRAM_SQL,
    SCHEMA_SQL,
    SessionDB,
    _FTS_TRIGGERS,
)


def _make_legacy_db(db_path, texts=("alpha needle one", "beta needle two")):
    """Legacy v22-shape store: inline FTS tables, live triggers, indexed rows."""
    raw = sqlite3.connect(str(db_path))
    raw.executescript(SCHEMA_SQL)
    try:
        raw.executescript(LEGACY_FTS_SQL + LEGACY_FTS_TRIGRAM_SQL)
    except sqlite3.OperationalError as exc:
        raw.close()
        pytest.skip(f"required FTS tokenizer unavailable: {exc}")
    raw.execute(
        "INSERT INTO sessions (id, source, started_at) VALUES ('s1', 'test', 1.0)"
    )
    for i, text in enumerate(texts):
        raw.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) "
            "VALUES ('s1', 'user', ?, ?)",
            (text, 1.0 + i),
        )
    # Commit the trigger-driven FTS writes BEFORE any corruption: rewriting
    # %_data blocks in the same transaction as pending FTS inserts makes the
    # commit itself fail inside fts5's consistency checks.
    raw.commit()
    raw.close()


def _corrupt_fts_index(db_path, table):
    """Overwrite the inverted-index blocks so 'integrity-check' (and, for the
    trigram table, plain DELETE) raise DatabaseError while %_content stays
    intact — i.e. a shape the 'rebuild' repair fully recovers from."""
    raw = sqlite3.connect(str(db_path))
    raw.execute(
        f"UPDATE {table}_data SET block = X'DEADBEEFDEADBEEFDEADBEEFDEADBEEF'"
    )
    raw.commit()
    raw.close()


def _integrity_check_ok(db_path, table):
    raw = sqlite3.connect(str(db_path))
    try:
        raw.execute(f"INSERT INTO {table}({table}) VALUES('integrity-check')")
        return True
    except sqlite3.DatabaseError:
        return False
    finally:
        try:
            raw.rollback()
        except sqlite3.Error:
            pass
        raw.close()


def _match_count(db_path, table, term):
    raw = sqlite3.connect(str(db_path))
    try:
        return raw.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {table} MATCH ?", (term,)
        ).fetchone()[0]
    finally:
        raw.close()


def _meta_value(db_path, key):
    raw = sqlite3.connect(str(db_path))
    try:
        row = raw.execute(
            "SELECT value FROM state_meta WHERE key = ?", (key,)
        ).fetchone()
        return None if row is None else row[0]
    finally:
        raw.close()


def _set_meta(db_path, key, value):
    raw = sqlite3.connect(str(db_path))
    raw.execute(
        "INSERT INTO state_meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
    raw.commit()
    raw.close()


def _live_fts_triggers(db_path):
    raw = sqlite3.connect(str(db_path))
    try:
        rows = raw.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' "
            f"AND name IN ({','.join('?' for _ in _FTS_TRIGGERS)})",
            _FTS_TRIGGERS,
        ).fetchall()
        return {row[0] for row in rows}
    finally:
        raw.close()


class TestFtsEngineIntegrityGate:
    def test_open_detects_and_repairs_engine_left_corruption(self, tmp_path, caplog):
        """T1: a legacy DB whose trigram index the current engine cannot read
        is repaired at open via 'rebuild' and stamped with the engine marker."""
        db_path = tmp_path / "legacy-state.db"
        _make_legacy_db(db_path)
        assert _match_count(db_path, "messages_fts_trigram", "needle") == 2
        _corrupt_fts_index(db_path, "messages_fts_trigram")

        with caplog.at_level(logging.WARNING, logger="hermes_state"):
            db = SessionDB(db_path=db_path)
        try:
            assert _integrity_check_ok(db_path, "messages_fts")
            assert _integrity_check_ok(db_path, "messages_fts_trigram")
            assert _match_count(db_path, "messages_fts", "needle") == 2
            assert _match_count(db_path, "messages_fts_trigram", "needle") == 2
            assert (
                _meta_value(db_path, FTS_INTEGRITY_ENGINE_KEY)
                == sqlite3.sqlite_version
            )
            assert any(
                "messages_fts_trigram" in record.message
                for record in caplog.records
                if record.levelno == logging.WARNING
            )
        finally:
            db.close()

    def test_matching_engine_marker_skips_the_gate(self, tmp_path):
        """T2: the marker equals the running engine → no re-verification, so a
        (re-introduced) inconsistency survives the open untouched."""
        db_path = tmp_path / "legacy-state.db"
        _make_legacy_db(db_path)
        _corrupt_fts_index(db_path, "messages_fts_trigram")
        _set_meta(db_path, FTS_INTEGRITY_ENGINE_KEY, sqlite3.sqlite_version)

        db = SessionDB(db_path=db_path)
        try:
            assert not _integrity_check_ok(db_path, "messages_fts_trigram")
            assert (
                _meta_value(db_path, FTS_INTEGRITY_ENGINE_KEY)
                == sqlite3.sqlite_version
            )
        finally:
            db.close()

    def test_engine_change_reverifies_and_repairs(self, tmp_path):
        """T3: a marker from a different engine version does not satisfy the
        gate — the sweep runs, repairs, and re-stamps the current version."""
        db_path = tmp_path / "legacy-state.db"
        _make_legacy_db(db_path)
        _corrupt_fts_index(db_path, "messages_fts_trigram")
        _set_meta(db_path, FTS_INTEGRITY_ENGINE_KEY, "3.0.0")

        db = SessionDB(db_path=db_path)
        try:
            assert _integrity_check_ok(db_path, "messages_fts_trigram")
            assert _match_count(db_path, "messages_fts_trigram", "needle") == 2
            assert (
                _meta_value(db_path, FTS_INTEGRITY_ENGINE_KEY)
                == sqlite3.sqlite_version
            )
        finally:
            db.close()

    def test_legacy_trigger_repair_survives_delete_failure(self, tmp_path):
        """T4: the triggers-need-repair backfill starts with ``DELETE FROM``
        the inline tables; on the malformed-index class that DELETE itself
        raises, so the rebuild must fall back to drop-recreate-backfill
        instead of failing the open."""
        db_path = tmp_path / "legacy-state.db"
        _make_legacy_db(db_path)
        raw = sqlite3.connect(str(db_path))
        raw.execute("DROP TRIGGER messages_fts_trigram_delete")
        raw.commit()
        raw.close()
        _corrupt_fts_index(db_path, "messages_fts_trigram")

        db = SessionDB(db_path=db_path)  # must not raise
        try:
            assert _integrity_check_ok(db_path, "messages_fts")
            assert _integrity_check_ok(db_path, "messages_fts_trigram")
            assert _match_count(db_path, "messages_fts", "needle") == 2
            assert _match_count(db_path, "messages_fts_trigram", "needle") == 2
            assert _live_fts_triggers(db_path) == set(_FTS_TRIGGERS)
        finally:
            db.close()

    def test_read_only_open_never_touches_the_index(self, tmp_path):
        """T5: a read-only open neither repairs nor stamps the marker."""
        db_path = tmp_path / "legacy-state.db"
        _make_legacy_db(db_path)
        _corrupt_fts_index(db_path, "messages_fts_trigram")

        ro = SessionDB(db_path=db_path, read_only=True)
        try:
            assert not _integrity_check_ok(db_path, "messages_fts_trigram")
            assert _meta_value(db_path, FTS_INTEGRITY_ENGINE_KEY) is None
        finally:
            ro.close()

    def test_capability_signature_stamped_and_skips_resweep(self, tmp_path, monkeypatch):
        """T10: a host whose probe cannot see a table (e.g. no cjk
        tokenizer) sweeps at most once per engine+capability pair — the
        stamp encodes the missing-capability signature, so a same-shape
        reopen skips, while the bare-version stamp semantics are unchanged
        on fully capable hosts."""
        db_path = tmp_path / "legacy-state.db"
        _make_legacy_db(db_path)

        real_probe = SessionDB._fts_table_probe

        def incapable_probe(self, cursor, table_name):
            if table_name == "messages_fts_cjk":
                return None  # capability missing, table unverifiable here
            return real_probe(self, cursor, table_name)

        monkeypatch.setattr(SessionDB, "_fts_table_probe", incapable_probe)
        db = SessionDB(db_path=db_path)
        db.close()
        expected_stamp = f"{sqlite3.sqlite_version}|missing=messages_fts_cjk"
        assert _meta_value(db_path, FTS_INTEGRITY_ENGINE_KEY) == expected_stamp

        # A same-capability reopen skips the sweep entirely: corruption
        # introduced after the stamp survives untouched (matching-marker
        # semantics, now signature-scoped).
        _corrupt_fts_index(db_path, "messages_fts_trigram")
        db = SessionDB(db_path=db_path)
        db.close()
        assert not _integrity_check_ok(db_path, "messages_fts_trigram")
        assert (
            _meta_value(db_path, FTS_INTEGRITY_ENGINE_KEY) == expected_stamp
        )

    def test_v23_external_content_layout_repaired(self, tmp_path):
        """T6: the same gate covers the v23 external-content layout (the
        'rebuild' command re-reads the canonical messages table). The DB is
        born on the current engine — which stamps the marker at creation —
        so the engine change is simulated by stamping an older version, as
        an upgraded install would carry."""
        db_path = tmp_path / "state.db"
        db = SessionDB(db_path=db_path)
        if not db._fts_enabled:
            db.close()
            pytest.skip("FTS5 unavailable in this build")
        db.create_session("s1", source="test")
        db.append_message("s1", "user", "alpha needle one")
        db.append_message("s1", "user", "beta needle two")
        db.close()

        _corrupt_fts_index(db_path, "messages_fts")
        _set_meta(db_path, FTS_INTEGRITY_ENGINE_KEY, "3.0.0")

        reopened = SessionDB(db_path=db_path)
        try:
            assert _integrity_check_ok(db_path, "messages_fts")
            assert _integrity_check_ok(db_path, "messages_fts_trigram")
            assert _match_count(db_path, "messages_fts", "needle") == 2
            assert (
                _meta_value(db_path, FTS_INTEGRITY_ENGINE_KEY)
                == sqlite3.sqlite_version
            )
        finally:
            reopened.close()


class _FakeConnection:
    def __init__(self):
        self.rollbacks = 0

    def rollback(self):
        self.rollbacks += 1


class _FailingCursor:
    """Cursor double for _rebuild_legacy_fts_indexes failure paths: SQL
    containing a configured marker raises; everything else succeeds. The
    atomic fallback runs as one executescript, so markers match by
    containment (the fallback script contains no fast-path DELETE)."""

    def __init__(self, failures):
        self.failures = failures
        self.executed = []
        self.params = []
        self.connection = _FakeConnection()

    def _maybe_raise(self, sql):
        for marker, exc in self.failures.items():
            if marker in sql:
                raise exc

    def execute(self, sql, params=()):
        self.executed.append(sql)
        self.params.append(params)
        self._maybe_raise(sql)
        return self

    def executescript(self, sql):
        self.executed.append(sql)
        self.params.append(())
        self._maybe_raise(sql)


def _stale_breadcrumb_dropped(cursor):
    """True if the fts_stale breadcrumb write was attempted (any params
    recorded alongside a state_meta upsert)."""
    from hermes_state import FTS_STALE_KEY

    return any(
        sql.startswith("INSERT INTO state_meta") and p and p[0] == FTS_STALE_KEY
        for sql, p in zip(cursor.executed, cursor.params)
    )


class TestLegacyRebuildFallbackFailures:
    def test_fallback_own_failure_logs_and_continues(self, caplog):
        """T7: DELETE fails with the malformed class and the atomic recovery
        script ALSO fails → rollback attempted, logger.error with the
        `hermes sessions repair` hint, the fts_stale breadcrumb is dropped
        (a possibly-half-recreated table must not be trusted by the next
        open), and no exception escapes."""
        cursor = _FailingCursor(
            {
                "DELETE FROM messages_fts": sqlite3.DatabaseError(
                    "database disk image is malformed"
                ),
                "DROP TABLE IF EXISTS messages_fts;": sqlite3.DatabaseError(
                    "malformed inverted index for FTS5 table"
                ),
            }
        )
        with caplog.at_level(logging.ERROR, logger="hermes_state"):
            SessionDB._rebuild_legacy_fts_indexes(cursor, include_trigram=False)
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert "messages_fts" in errors[0].message
        assert "hermes sessions repair" in errors[0].message
        assert _stale_breadcrumb_dropped(cursor)
        assert cursor.connection.rollbacks >= 1
        # the atomic recovery script ran (single BEGIN IMMEDIATE-wrapped
        # executescript) and failed before any separate backfill execute
        assert any(sql.startswith("BEGIN IMMEDIATE;") for sql in cursor.executed)
        assert not any(
            sql.startswith("INSERT INTO messages_fts(") for sql in cursor.executed
        )

    def test_post_fallback_backfill_failure_does_not_raise(self, caplog):
        """T8: DELETE fails, but the failure inside the atomic recovery
        script is at the backfill INSERT → rollback + error log + repair
        hint + fts_stale breadcrumb (an empty recreated table would
        otherwise pass 'integrity-check' and freeze a silently-dead index
        under the stamped marker), and nothing escapes to fail the open."""
        cursor = _FailingCursor(
            {
                "DELETE FROM messages_fts": sqlite3.DatabaseError(
                    "database disk image is malformed"
                ),
                "INSERT INTO messages_fts(rowid, content)": sqlite3.DatabaseError(
                    "database disk image is malformed"
                ),
            }
        )
        with caplog.at_level(logging.ERROR, logger="hermes_state"):
            SessionDB._rebuild_legacy_fts_indexes(cursor, include_trigram=False)
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert "hermes sessions repair" in errors[0].message
        assert _stale_breadcrumb_dropped(cursor)
        assert cursor.connection.rollbacks >= 1
        # the recovery ran its full shape inside one atomic script
        scripts = [s for s in cursor.executed if s.startswith("BEGIN IMMEDIATE;")]
        assert len(scripts) == 1
        assert "DROP TRIGGER IF EXISTS messages_fts_insert" in scripts[0]
        assert "DROP TABLE IF EXISTS messages_fts;" in scripts[0]
        assert "CREATE VIRTUAL TABLE" in scripts[0]
        assert "INSERT INTO messages_fts(rowid, content)" in scripts[0]
        assert scripts[0].rstrip().endswith("COMMIT;")

    def test_transient_delete_error_reraises_without_fallback(self, caplog):
        """T9: a lock/busy-style DELETE failure is NOT the corruption class
        — it must re-raise for the open-retry path (pre-fallback behavior)
        and must not run the drop-recreate script or drop the fts_stale
        breadcrumb (classification adopted from the #86062 review round)."""
        cursor = _FailingCursor(
            {
                "DELETE FROM messages_fts": sqlite3.OperationalError(
                    "database is locked"
                )
            }
        )
        with pytest.raises(sqlite3.OperationalError, match="database is locked"):
            SessionDB._rebuild_legacy_fts_indexes(cursor, include_trigram=False)
        assert not any(
            sql.startswith("BEGIN IMMEDIATE;") for sql in cursor.executed
        )
        assert not _stale_breadcrumb_dropped(cursor)
        assert cursor.connection.rollbacks == 0
