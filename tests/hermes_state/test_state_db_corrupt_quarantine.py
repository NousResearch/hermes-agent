"""Quarantine of a live SessionDB handle after structural (non-FTS) corruption.

Field evidence (the #90837 lost/reordered-page-write class): a gateway kept
retrying writes for ~50 minutes after ``gateway_routing`` reported
``database disk image is malformed``; on SIGTERM the close-time
``PRAGMA wal_checkpoint(PASSIVE)`` then wrote 15 pages to the wrong page
numbers (page 1 received a ``messages_fts_trigram_data`` leaf) and the file
stopped opening at all. Once structural corruption is observed on a handle
the only safe policy is to stop touching the file.
"""

import sqlite3

import pytest

from hermes_state import SessionDB, StateDbCorruptError


class _MalformedConn:
    """Connection proxy whose every execute reports bare SQLITE_CORRUPT."""

    def __init__(self, real_conn):
        self._real = real_conn

    def execute(self, *args, **kwargs):
        raise sqlite3.DatabaseError("database disk image is malformed")

    def __getattr__(self, name):
        return getattr(self._real, name)


class TestQuarantineAfterStructuralCorruption:
    def test_structural_corruption_sets_sticky_flag_and_raises_typed(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        real_conn = db._conn
        try:
            db.create_session(session_id="s1", source="cli", model="test")
            db._conn = _MalformedConn(real_conn)
            with pytest.raises(StateDbCorruptError, match="malformed") as excinfo:
                db.create_session(session_id="s2", source="cli", model="test")
            assert isinstance(excinfo.value.__cause__, sqlite3.DatabaseError)
            assert db._db_corrupt is True
            # Structural damage must never be mistaken for FTS-scoped damage.
            assert db._fts_stale is False
        finally:
            db._conn = real_conn
            db.close()


class _RecordingConn:
    """Connection proxy that records every SQL text and delegates."""

    def __init__(self, real_conn):
        self._real = real_conn
        self.recorded = []

    def execute(self, sql, *args, **kwargs):
        self.recorded.append(str(sql))
        return self._real.execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


def _quarantined_db(tmp_path):
    """A SessionDB whose first corrupt write already tripped the quarantine."""
    db = SessionDB(db_path=tmp_path / "state.db")
    real_conn = db._conn
    db.create_session(session_id="s1", source="cli", model="test")
    db._conn = _MalformedConn(real_conn)
    with pytest.raises(StateDbCorruptError):
        db.create_session(session_id="s2", source="cli", model="test")
    db._conn = real_conn
    assert db._db_corrupt is True
    return db, real_conn


class TestQuarantinedHandleStopsTouchingTheFile:
    def test_subsequent_writes_fail_fast_without_touching_connection(self, tmp_path):
        db, real_conn = _quarantined_db(tmp_path)
        recorder = _RecordingConn(real_conn)
        db._conn = recorder
        try:
            with pytest.raises(StateDbCorruptError):
                db.create_session(session_id="s3", source="cli", model="test")
            assert recorder.recorded == []
        finally:
            db._conn = real_conn
            db.close()

    def test_close_skips_wal_checkpoint_when_quarantined(self, tmp_path, caplog):
        db, real_conn = _quarantined_db(tmp_path)
        recorder = _RecordingConn(real_conn)
        db._conn = recorder
        with caplog.at_level("WARNING", logger="hermes_state"):
            db.close()
        assert not any("wal_checkpoint" in sql for sql in recorder.recorded)
        assert db._conn is None
        assert any(
            "Skipping the close-time WAL checkpoint" in rec.getMessage()
            and "hermes sessions recover" in rec.getMessage()
            for rec in caplog.records
        )

    def test_reopen_after_close_refused_when_quarantined(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        db, real_conn = _quarantined_db(tmp_path)
        db.close()
        reopen = MagicMock()
        monkeypatch.setattr("hermes_state._connect_tracked_db", reopen)
        with pytest.raises(StateDbCorruptError, match="structural corruption"):
            db.create_session(session_id="s4", source="cli", model="test")
        reopen.assert_not_called()
        # The read fallback after close() goes through the same reopen path.
        with pytest.raises(StateDbCorruptError, match="refusing to reopen"):
            db.get_session("s1")
        reopen.assert_not_called()


class TestQuarantineScope:
    def test_fts_scoped_corruption_does_not_trip_flag(self, tmp_path):
        """Corrupt FTS shadow tables keep the existing fail-open detach path."""
        path = tmp_path / "state.db"
        db = SessionDB(db_path=path)
        db.create_session(session_id="s1", source="cli", model="test")
        db.append_message("s1", role="user", content="hello world")
        raw = sqlite3.connect(str(path))
        raw.execute(
            "UPDATE messages_fts_data SET block = X'DEADBEEFDEADBEEFDEADBEEFDEADBEEF'"
        )
        raw.commit()
        raw.close()
        try:
            db.append_message("s1", role="user", content="healed append")
            assert db._db_corrupt is False
            assert db._fts_stale is True
            assert db._fts_enabled is False
        finally:
            db.close()

    def test_replaced_file_takes_precedence_over_corrupt(self, tmp_path):
        import os

        from hermes_state import StateDbReplacedError

        live = tmp_path / "state.db"
        other = tmp_path / "other.db"
        db = SessionDB(db_path=live)
        real_conn = db._conn
        try:
            db.create_session(session_id="s1", source="cli", model="test")
            if db._db_file_identity is None:
                pytest.skip("filesystem does not expose st_dev/st_ino")
            alt = SessionDB(db_path=other)
            alt.create_session("other", "cli")
            alt.close()
            os.replace(other, live)
            db._conn = _MalformedConn(real_conn)
            with pytest.raises(StateDbReplacedError):
                db.create_session(session_id="s2", source="cli", model="test")
            assert db._db_replaced is True
            assert db._db_corrupt is False
        finally:
            db._conn = real_conn
            db.close()

    def test_classify_persistence_error_maps_quarantine_to_corrupt(self):
        from hermes_state import _STATE_DB_CORRUPT_MSG, classify_persistence_error

        assert classify_persistence_error(StateDbCorruptError("x")) == "corrupt"
        # The stringified form (RPC boundaries) must classify the same way.
        assert classify_persistence_error(_STATE_DB_CORRUPT_MSG) == "corrupt"


@pytest.fixture
def _clean_registry():
    import hermes_state_registry as registry

    registry.close_all()
    registry._generations.clear()
    registry._retired.clear()
    yield registry
    registry.close_all()
    registry._generations.clear()
    registry._retired.clear()


class TestSharedRegistry:
    def test_holders_share_quarantine_and_close_all_skips_checkpoint(
        self, tmp_path, _clean_registry
    ):
        registry = _clean_registry
        path = tmp_path / "state.db"
        holder_a = registry.acquire(path)
        holder_b = registry.acquire(path)
        assert holder_a is holder_b
        real_conn = holder_a._conn
        holder_a.create_session(session_id="s1", source="cli", model="test")

        holder_a._conn = _MalformedConn(real_conn)
        with pytest.raises(StateDbCorruptError):
            holder_a.create_session(session_id="s2", source="cli", model="test")
        recorder = _RecordingConn(real_conn)
        holder_b._conn = recorder

        with pytest.raises(StateDbCorruptError):
            holder_b.create_session(session_id="s3", source="cli", model="test")

        registry.close_all()
        assert not any("wal_checkpoint" in sql for sql in recorder.recorded)
        assert holder_a._conn is None
