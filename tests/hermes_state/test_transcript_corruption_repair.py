"""Offline transcript-corruption repair for worker transcript writes (#115571).

Field report: while saving a worker transcript Hermes raised a storage
failure (completed=false, failed=true) with PRAGMA integrity_check
reporting damaged structure in the messages table and three indexes
(idx_messages_session_active, idx_messages_session_id,
idx_messages_session), and no repair was attempted.

Contract under test:
* on transcript-write sqlite3.DatabaseError the write path runs
  PRAGMA integrity_check and rebuilds messages + its indexes into
  a fresh file offline, then atomically swaps it in;
* the original file is never deleted or truncated on failure (fail closed):
  an unhealable image raises with the live bytes bit-identical;
* the core repair_state_db_schema strategy ladder also incorporates
  the messages table rebuild strategy to heal structural transcript corruption.
"""

import hashlib
import sqlite3

import pytest

from hermes_state import SessionDB, StateDbCorruptError
from hermes_state_repair import (
    repair_state_db_schema,
    repair_transcript_corruption_offline,
)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _checkpoint_and_detach(path):
    """Fold WAL frames into the main file and drop sidecars so tests flip
    bytes in the only image SQLite will read."""
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.commit()
    finally:
        conn.close()
    for suffix in ("-wal", "-shm", "-journal"):
        try:
            (path.parent / (path.name + suffix)).unlink()
        except FileNotFoundError:
            pass


def _integrity_lines(path):
    conn = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
    try:
        return [r[0] for r in conn.execute("PRAGMA integrity_check")]
    finally:
        conn.close()


def _damage_mentions_transcript(lines):
    return any(
        ("message" in line.lower() or "idx_messages" in line)
        and line.strip().lower() != "ok"
        for line in lines
    )


def _corrupt_messages_index_page(path):
    """Corrupt a messages-index b-tree page until PRAGMA integrity_check
    reports transcript-scoped damage (the #115571 verdict shape:
    ``row N missing from index idx_messages_*``).

    Flipping the low bytes of an index root page (cell-pointer array) breaks
    the index walk deterministically. Page 1 (sqlite_master) is never touched:
    schema damage is a different repair's job.
    """
    conn = sqlite3.connect(str(path))
    try:
        page_size = conn.execute("PRAGMA page_size").fetchone()[0]
        rows = conn.execute(
            "SELECT name, rootpage FROM sqlite_master "
            "WHERE type = 'index' AND tbl_name = 'messages' AND rootpage > 1 "
            "ORDER BY rootpage DESC"
        ).fetchall()
    finally:
        conn.close()
    assert rows, "expected messages indexes on pages > 1"
    raw = bytearray(path.read_bytes())
    for _name, rootpage in rows:
        base = (rootpage - 1) * page_size
        for delta in [8] + list(range(9, min(page_size, 512), 16)):
            off = base + delta
            if off >= len(raw):
                continue
            raw[off] ^= 0xFF
            path.write_bytes(raw)
            try:
                lines = _integrity_lines(path)
            except sqlite3.DatabaseError:
                lines = None
            if lines is not None and _damage_mentions_transcript(lines):
                return lines
            raw[off] ^= 0xFF  # benign flip, or an unscoped (raising) break: restore
            path.write_bytes(raw)
    pytest.fail("could not induce integrity-check-visible messages-index damage")


def _smash_messages_table_root(path):
    """Point the messages table at a nonexistent page: the table becomes
    unreadable, so no salvage can prove lossless recovery (fail-closed case)."""
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("PRAGMA writable_schema=ON")
        conn.execute("UPDATE sqlite_master SET rootpage = 999999 WHERE name = 'messages'")
        conn.commit()
    finally:
        conn.close()


def _expected_contents(n_pairs):
    out = []
    for i in range(n_pairs):
        out.append("q%d" % i)
        out.append("a%d" % i)
    return out


@pytest.fixture()
def transcript_db(tmp_path):
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    db.create_session("sess-tcr", source="cli", model="test")
    for i in range(5):
        db.append_message(session_id="sess-tcr", role="user", content="q%d" % i)
        db.append_message(session_id="sess-tcr", role="assistant", content="a%d" % i)
    db.close()
    _checkpoint_and_detach(path)
    return path


class TestOfflineTranscriptRepair:
    def test_heals_corrupt_messages_index_preserving_rows(self, transcript_db):
        damage = _corrupt_messages_index_page(transcript_db)
        assert _damage_mentions_transcript(damage)

        report = repair_transcript_corruption_offline(transcript_db)

        assert report["repaired"] is True, report
        assert _integrity_lines(transcript_db) == ["ok"]
        conn = sqlite3.connect("file:%s?mode=ro" % transcript_db, uri=True)
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = 'sess-tcr'"
            ).fetchone()[0]
            assert count == 10
            contents = [
                r[0]
                for r in conn.execute(
                    "SELECT content FROM messages WHERE session_id = 'sess-tcr' "
                    "ORDER BY id"
                )
            ]
            assert contents == _expected_contents(5)
            indexes = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'index' "
                    "AND tbl_name = 'messages'"
                )
            }
            assert {
                "idx_messages_session",
                "idx_messages_session_id",
                "idx_messages_session_active",
            } <= indexes
        finally:
            conn.close()

    def test_repair_state_db_schema_rebuilds_corrupt_messages_table(self, transcript_db):
        damage = _corrupt_messages_index_page(transcript_db)
        assert _damage_mentions_transcript(damage)

        report = repair_state_db_schema(transcript_db, backup=False)

        assert report["repaired"] is True, report
        assert _integrity_lines(transcript_db) == ["ok"]
        conn = sqlite3.connect("file:%s?mode=ro" % transcript_db, uri=True)
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = 'sess-tcr'"
            ).fetchone()[0]
            assert count == 10
        finally:
            conn.close()

    def test_unhealable_damage_fails_closed_original_untouched(self, transcript_db):
        # Break the messages table root: nothing can be salvaged, so the repair
        # must report unhealed with the live bytes bit-identical.
        _smash_messages_table_root(transcript_db)
        before = _sha256(transcript_db)

        report = repair_transcript_corruption_offline(transcript_db)

        assert report["repaired"] is False, report
        assert report["error"], report
        assert _sha256(transcript_db) == before
        assert not list(transcript_db.parent.glob("state.db.transcript-rebuild*"))

    def test_clean_db_reports_clean_without_touching(self, transcript_db):
        assert _integrity_lines(transcript_db) == ["ok"]
        before = _sha256(transcript_db)

        report = repair_transcript_corruption_offline(transcript_db)

        assert report["repaired"] is False
        assert report.get("strategy") == "clean"
        assert _sha256(transcript_db) == before


class _FailOnceConn:
    """Raise one structural DatabaseError, then delegate to the live conn."""

    def __init__(self, real_conn, exc):
        self._real = real_conn
        self._exc = exc

    def execute(self, *args, **kwargs):
        if self._exc is not None:
            exc, self._exc = self._exc, None
            raise exc
        return self._real.execute(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


class TestTranscriptWriteRepairHook:
    def test_write_error_with_clean_image_fails_closed(self, tmp_path, caplog):
        path = tmp_path / "state.db"
        db = SessionDB(db_path=path)
        try:
            db.create_session("sess-tcr", source="cli", model="test")
            db.append_message(session_id="sess-tcr", role="user", content="hi")
            before = _sha256(path)
            real_conn = db._conn
            db._conn = _FailOnceConn(
                real_conn, sqlite3.DatabaseError("database disk image is malformed")
            )
            try:
                with caplog.at_level("WARNING", logger="hermes_state"):
                    with pytest.raises(StateDbCorruptError):
                        db.append_message(
                            session_id="sess-tcr", role="user", content="lost?"
                        )
            finally:
                if isinstance(db._conn, _FailOnceConn):
                    db._conn = real_conn
            # Clean image: no swap, original bytes bit-identical, quarantine held.
            assert _sha256(path) == before
            assert db._db_corrupt is True
            assert any(
                "integrity" in rec.getMessage().lower() for rec in caplog.records
            )
        finally:
            db.close()

    def test_write_error_with_healable_damage_repairs_and_lands_write(
        self, transcript_db
    ):
        damage = _corrupt_messages_index_page(transcript_db)
        assert _damage_mentions_transcript(damage)
        db = SessionDB(db_path=transcript_db)
        try:
            real_conn = db._conn
            db._conn = _FailOnceConn(
                real_conn, sqlite3.DatabaseError("database disk image is malformed")
            )
            try:
                row_id = db.append_message(
                    session_id="sess-tcr", role="user", content="after-repair"
                )
            finally:
                if isinstance(db._conn, _FailOnceConn):
                    db._conn = real_conn
            assert isinstance(row_id, int)
            assert _integrity_lines(transcript_db) == ["ok"]
            assert db.get_messages("sess-tcr")[-1]["content"] == "after-repair"
        finally:
            db.close()
