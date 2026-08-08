"""Tests for the state.db byte-identical duplicate-message hole (mechanism 1).

A profile state.db was observed carrying two message rows with the same
(session_id, role, content, timestamp-to-the-millisecond) — e.g. pair
id 638962 == id 640246, ts REAL 1785411770.4681301 — so the same report
rendered twice in the Desktop chat view. Any residual writer path (a poll
loop, a race in a flush, a missed compaction rebaseline) could persist a
byte-identical row because the ``messages`` table had no uniqueness guard.

The fix boundary: dedupe applies ONLY within the ACTIVE (active=1) row
class. The in-place compaction flow INTENTIONALLY keeps the same
content+timestamp once as (active=0, compacted=1) — the archived original —
and once as (active=1, compacted=0) — the re-inserted live copy. A
constraint spanning (active, compacted) would forbid that pair and break
compaction durability.
"""

import sqlite3
import tempfile
from pathlib import Path

CONTENT = "quarterly report: revenue up 12%"
TS = 1785411770.4681301  # fractional-millisecond REAL, as in the live DB


def _raw_counts(db_path):
    """Direct sqlite read — independent of the SessionDB Python API."""
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            "SELECT COUNT(*), "
            "COALESCE(SUM(active = 1 AND compacted = 0), 0), "
            "COALESCE(SUM(active = 0 AND compacted = 1), 0) "
            "FROM messages WHERE content = ?",
            (CONTENT,),
        ).fetchone()
    finally:
        conn.close()


class TestAppendMessageIdempotence:
    def test_append_message_idempotent_duplicate_insert(self):
        """Re-appending a byte-identical (session, role, content, timestamp)
        message must not create a second row, double-count the session, or
        double-index FTS."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "t.db"
            db = SessionDB(db_path=db_path)
            try:
                sid = "20260730_170159_f8432a"
                db.create_session(sid, "cli", model="test/model")

                first_id = db.append_message(
                    session_id=sid, role="assistant",
                    content=CONTENT, timestamp=TS,
                )
                # The exact same write again — the idempotence-hole path.
                second_id = db.append_message(
                    session_id=sid, role="assistant",
                    content=CONTENT, timestamp=TS,
                )

                assert second_id == first_id, (
                    "idempotent re-append must resolve to the existing row"
                )
                # Direct DB read: exactly ONE message row for this content.
                total, active, archived = _raw_counts(db_path)
                assert total == 1, (
                    f"duplicate insert persisted: {total} rows share the same "
                    "content"
                )
                assert (active, archived) == (1, 0)
                # Session counters must track the single persisted row.
                assert db.get_session(sid)["message_count"] == 1
                # FTS got exactly one entry — the duplicate insert must not
                # fire the messages_fts_insert trigger for a skipped row.
                fts_count = db._conn.execute(
                    "SELECT COUNT(*) FROM messages_fts"
                ).fetchone()[0]
                assert fts_count == 1, (
                    "skipped duplicate insert still fired the FTS insert "
                    "trigger"
                )
            finally:
                db.close()


class TestInPlaceCompactionDuplicateBoundary:
    def test_in_place_compaction_keeps_single_live_copy_and_archive(self):
        """The mechanism-2 pair must survive the dedupe guard while a
        duplicate replay is still collapsed.

        Flow: an active message is soft-archived by in-place compaction
        (active=0, compacted=1) and the same content+timestamp is
        re-inserted as the new active row — the legitimate archive+active
        pair. A residual replay then re-appends the identical row a second
        time. The store must end with exactly ONE live copy AND the archive
        copy intact — not two live rows, and the archive copy NOT swallowed.
        """
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "t.db"
            db = SessionDB(db_path=db_path)
            try:
                sid = "20260730_170159_f8432a"
                db.create_session(sid, "cli", model="test/model")
                db.append_message(
                    session_id=sid, role="assistant",
                    content=CONTENT, timestamp=TS,
                )

                # In-place compaction: soft-archive the live row, re-insert
                # the same content+timestamp as the fresh active copy.
                db.archive_and_compact(
                    sid,
                    [{"role": "assistant", "content": CONTENT,
                      "timestamp": TS}],
                )

                # Residual duplicate writer (race / poll-loop / missed
                # rebaseline) re-appends the identical row.
                db.append_message(
                    session_id=sid, role="assistant",
                    content=CONTENT, timestamp=TS,
                )

                total, active, archived = _raw_counts(db_path)
                assert total == 2, (
                    f"the archive+live pair must total 2 rows, found {total}"
                )
                assert active == 1, (
                    f"duplicate replay persisted: {active} live active=1 "
                    "copies of the same content+timestamp"
                )
                assert archived == 1, (
                    "the compaction archive copy (active=0, compacted=1) "
                    "must survive the dedupe constraint — it is intentional"
                )
                # Live count tracks just the active row.
                assert db.get_session(sid)["message_count"] == 1
            finally:
                db.close()


class TestMigrationCollapsesExistingDuplicates:
    def test_upgrade_open_collapses_dirty_active_rows_and_guards_future(self):
        """An existing DB that already carries byte-identical ACTIVE duplicate
        rows (written before the guard existed) must open cleanly: the
        migration keeps the first copy, preserves the intentional
        archive+live pair, installs the guard index, and blocks new dupes.
        """
        from hermes_state import SessionDB
        from hermes_state_common import SCHEMA_VERSION

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "t.db"
            db = SessionDB(db_path=db_path)
            sid = "20260730_170159_f8432a"
            db.create_session(sid, "cli", model="test/model")
            db.close()

            # Recreate a pre-upgrade dirty state: guard index absent, two
            # live byte-identical rows, plus the intentional compaction pair
            # (same content+timestamp archived AND active). Plain INSERTs
            # only succeed while the index is dropped — as before the fix.
            raw = sqlite3.connect(db_path)
            try:
                raw.execute("DROP INDEX idx_messages_active_dedupe")
                ts_pair = TS + 1000.0
                for i in range(2):  # byte-identical live duplicates
                    raw.execute(
                        "INSERT INTO messages (session_id, role, content, "
                        "timestamp, active, compacted) VALUES (?, 'assistant', "
                        "?, ?, 1, 0)",
                        (sid, CONTENT, TS),
                    )
                raw.execute(  # archived original of the compaction pair
                    "INSERT INTO messages (session_id, role, content, "
                    "timestamp, active, compacted) VALUES (?, 'assistant', "
                    "?, ?, 0, 1)",
                    (sid, CONTENT, ts_pair),
                )
                raw.execute(  # its live re-insert — the legitimate pair
                    "INSERT INTO messages (session_id, role, content, "
                    "timestamp, active, compacted) VALUES (?, 'assistant', "
                    "?, ?, 1, 0)",
                    (sid, CONTENT, ts_pair),
                )
                raw.commit()
            finally:
                raw.close()

            db = SessionDB(db_path=db_path)  # open runs the migration
            try:
                rows = db._conn.execute(
                    "SELECT id, active, compacted FROM messages "
                    "WHERE content = ? ORDER BY id",
                    (CONTENT,),
                ).fetchall()
                live = [r for r in rows if r["active"] == 1]
                archived = [r for r in rows if r["active"] == 0]
                assert len(rows) == 3 and len(live) == 2, (
                    f"expected merged duplicate + intact pair, found {rows}"
                )
                assert len(archived) == 1 and archived[0]["compacted"] == 1
                # The surviving duplicate copy is the FIRST-inserted row.
                assert live[0]["id"] == 1
                # The guard index landed and schema bookkeeping advanced.
                assert db._conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'index' "
                    "AND name = 'idx_messages_active_dedupe'"
                ).fetchone() is not None
                assert db._conn.execute(
                    "SELECT version FROM schema_version"
                ).fetchone()[0] == SCHEMA_VERSION
                # New duplicate writes are blocked from here on.
                db.append_message(
                    session_id=sid, role="assistant",
                    content=CONTENT, timestamp=TS,
                )
                live_after = db._conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE content = ? "
                    "AND active = 1 AND timestamp = ?",
                    (CONTENT, TS),
                ).fetchone()[0]
                assert live_after == 1
            finally:
                db.close()


class TestNullActiveMigrationOrdering:
    """Regression for review thread on hermes_state_schema.py:L355.

    Legacy DBs may carry rows with ``active IS NULL`` (older reconciler
    builds omitted the NOT NULL DEFAULT 1).  If the v24 duplicate-cleanup
    DELETE runs BEFORE those NULL rows are normalised to active=1, the
    DELETE misses them (it only targets active=1).  Later the NULL→1
    UPDATE would then collide with the already-created unique index.

    The fix normalises NULL active → 1 BEFORE the cleanup DELETE so the
    DELETE sees and collapses the formerly-NULL duplicates too.
    """

    def test_null_active_duplicates_collapsed_before_index_creation(self):
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "t.db"
            db = SessionDB(db_path=db_path)
            sid = "20260730_170159_f8432a"
            db.create_session(sid, "cli", model="test/model")
            db.close()

            # Recreate pre-upgrade dirty state: two NULL-active rows with
            # the same (session_id, role, content, timestamp) — the legacy
            # double-write that the NULL→1 repair would later turn into
            # colliding active=1 rows.
            #
            # Legacy DBs had ``active`` added by the reconciler WITHOUT a
            # NOT NULL constraint (#51646), so we relax the column before
            # inserting NULLs.  PRAGMA writable_schema requires a reconnect
            # for the change to take effect.
            raw = sqlite3.connect(db_path)
            try:
                raw.execute("DROP INDEX IF EXISTS idx_messages_active_dedupe")
                raw.execute("PRAGMA writable_schema = 1")
                raw.execute(
                    "UPDATE sqlite_master SET sql = REPLACE(sql, "
                    "'active INTEGER NOT NULL DEFAULT 1', 'active INTEGER') "
                    "WHERE type = 'table' AND name = 'messages'"
                )
                raw.execute("PRAGMA writable_schema = 0")
                raw.commit()
            finally:
                raw.close()

            # Reopen — the relaxed schema is now active.
            raw = sqlite3.connect(db_path)
            try:
                for _ in range(2):
                    raw.execute(
                        "INSERT INTO messages (session_id, role, content, "
                        "timestamp, active, compacted) VALUES "
                        "(?, 'assistant', ?, ?, NULL, 0)",
                        (sid, CONTENT, TS),
                    )
                raw.commit()
            finally:
                raw.close()

            # Re-opening runs the migration: NULL→1 first, then DELETE
            # duplicates, then create index.  If the ordering is wrong,
            # the unique-index creation will raise IntegrityError.
            db = SessionDB(db_path=db_path)
            try:
                live = db._conn.execute(
                    "SELECT COUNT(*) FROM messages "
                    "WHERE content = ? AND active = 1 AND timestamp = ?",
                    (CONTENT, TS),
                ).fetchone()[0]
                assert live == 1, (
                    f"NULL-active duplicates not collapsed: {live} live rows"
                )
                # Index must exist — migration completed without error.
                assert db._conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'index' "
                    "AND name = 'idx_messages_active_dedupe'"
                ).fetchone() is not None
            finally:
                db.close()


class TestNullContentBoundary:
    """Documented limitation for review thread on hermes_state_common.py:L317.

    SQLite UNIQUE indexes treat NULL as distinct, so the
    idx_messages_active_dedupe index does NOT deduplicate rows with
    content = NULL.  The migration cleanup (GROUP BY, which collapses NULL
    into one group) handles EXISTING NULL-content duplicates, but NEW
    NULL-content duplicate inserts bypass the runtime guard.

    This is an explicit, documented boundary — NULL-content messages are
    rare (tool-call-only rows with no text body) and typically carry
    distinct tool_call_id values, so true duplicates are unlikely in
    practice.
    """

    def test_null_content_duplicate_insert_is_not_guarded(self):
        """Two append_message calls with content=None and the same timestamp
        create two rows — the index does not block this.  This test
        documents the known limitation; if a future fix adds a null-safe
        key (e.g. COALESCE(content, '') generated column), this test
        should be updated to assert deduplication."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "t.db"
            db = SessionDB(db_path=db_path)
            try:
                sid = "20260730_170159_f8432a"
                db.create_session(sid, "cli", model="test/model")

                db.append_message(
                    session_id=sid, role="assistant",
                    content=None, timestamp=TS,
                )
                db.append_message(
                    session_id=sid, role="assistant",
                    content=None, timestamp=TS,
                )

                # Known limitation: both rows persist (NULL is distinct in
                # SQLite UNIQUE indexes).  The guard does NOT fire here.
                count = db._conn.execute(
                    "SELECT COUNT(*) FROM messages "
                    "WHERE content IS NULL AND active = 1 AND timestamp = ?",
                    (TS,),
                ).fetchone()[0]
                assert count == 2, (
                    f"expected 2 unguarded NULL-content rows, found {count}"
                )
            finally:
                db.close()

    def test_null_content_existing_duplicates_collapsed_on_migration(self):
        """The migration cleanup DELETE uses GROUP BY, which treats NULL as
        a single group.  Existing NULL-content duplicate rows ARE cleaned
        up on upgrade, even though the runtime index does not guard them
        going forward."""
        from hermes_state import SessionDB

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "t.db"
            db = SessionDB(db_path=db_path)
            sid = "20260730_170159_f8432a"
            db.create_session(sid, "cli", model="test/model")
            db.close()

            # Pre-upgrade: two NULL-content active duplicates.
            raw = sqlite3.connect(db_path)
            try:
                raw.execute("DROP INDEX IF EXISTS idx_messages_active_dedupe")
                for _ in range(2):
                    raw.execute(
                        "INSERT INTO messages (session_id, role, content, "
                        "timestamp, active, compacted) VALUES "
                        "(?, 'assistant', NULL, ?, 1, 0)",
                        (sid, TS),
                    )
                raw.commit()
            finally:
                raw.close()

            db = SessionDB(db_path=db_path)
            try:
                count = db._conn.execute(
                    "SELECT COUNT(*) FROM messages "
                    "WHERE content IS NULL AND active = 1 AND timestamp = ?",
                    (TS,),
                ).fetchone()[0]
                assert count == 1, (
                    f"migration should collapse NULL-content duplicates to "
                    f"1 row, found {count}"
                )
            finally:
                db.close()
