"""FTS index damage must stay index-scoped on the ranked search path.

A ``messages_fts`` posting whose ``_docsize`` row is gone (left behind by an
external-content 'delete' that was handed different text than was indexed) makes
bm25 raise ``SQLITE_CORRUPT_VTAB``. ``ORDER BY rank`` runs bm25 inside FTS5's
nested sorter statement, and SQLite masks that statement's error to the primary
code, so the search read surfaced bare ``SQLITE_CORRUPT``: the store latched as
structurally corrupt and every session create/resume answered
``storage_unavailable`` until restart, although the canonical tables were intact.

The second half covers how such postings were produced in the field: a release
before FTS storage v3 reopening a realigned store re-stamps the retired
``fts_tool_full_content_high_water`` marker and reinstalls id-gated triggers, whose
'delete' sends FULL content for tool rows the v3 rebuild indexed as a prefix.
"""

import sqlite3

import pytest

import persistence.sqlite._schema as schema

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite._schema_ddl import FTS_TOOL_CONTENT_PREFIX_CHARS
from persistence.sqlite._state.health import STORAGE_CORRUPT, STORAGE_OK, reset_storage_state, storage_state
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.search import search_messages
from persistence.sqlite.conversation._store.sessions import create_session

# The v3 projection cuts tool rows at FTS_TOOL_CONTENT_PREFIX_CHARS; this body puts the
# cut inside "fragmenttail", so the index holds the fragment "frag" while the full
# body (what a pre-v3 'delete' sends) only ever contains "fragmenttail".
_LEAD = "lead "
_FILLER = "z" * (FTS_TOOL_CONTENT_PREFIX_CHARS - len(_LEAD) - 5)
CUT_TOOL_ROW = f"{_LEAD}{_FILLER} fragmenttail after the cut"

# The delete trigger a pre-v3 release installs (id-gated by the retired marker).
_PRE_V3_DELETE_TRIGGER = f"""
CREATE TRIGGER messages_fts_delete AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content, tool_name, tool_calls)
    VALUES ('delete', old.id,
            CASE WHEN old.role = 'tool'
                  AND old.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                                         WHERE key = 'fts_tool_full_content_high_water'), -1)
                 THEN substr(COALESCE(old.content, ''), 1, {FTS_TOOL_CONTENT_PREFIX_CHARS})
                 ELSE old.content END,
            old.tool_name, old.tool_calls);
END
"""


def _open(path):
    holder = StoreHolder(db_path=path)
    if not holder._fts_enabled:
        holder.close()
        pytest.skip("SQLite FTS5 unavailable")
    return holder


def _reopen(path):
    """Reopen a store _open accepted: FTS going missing here is a migration failure, not a skip."""
    holder = StoreHolder(db_path=path)
    assert holder._fts_enabled
    return holder


@pytest.fixture(autouse=True)
def _clean_latch():
    reset_storage_state()
    yield
    reset_storage_state()


def _orphan_posting(holder, session_id: str) -> None:
    """Index 'orphanword keepword', then 'delete' it with only 'keepword': FTS5 drops the
    row's _docsize entry and the 'orphanword' posting stays behind."""
    row_id = append_message(holder, session_id, role="user", content="orphanword keepword")
    holder._execute_write(lambda conn: conn.execute(
        "INSERT INTO messages_fts(messages_fts, rowid, content, tool_name, tool_calls) "
        "VALUES('delete', ?, 'keepword', NULL, NULL)", (row_id,)))


def test_ranked_search_over_orphaned_posting_stays_index_scoped(tmp_path):
    path = tmp_path / "state.db"
    holder = _open(path)
    try:
        create_session(holder, "s", source="cli")
        append_message(holder, "s", role="user", content="orphanword survives in this row")
        _orphan_posting(holder, "s")

        rows = search_messages(holder, "orphanword")

        assert any("survives" in (row.get("snippet") or "") for row in rows)
        assert holder._fts_stale
        assert storage_state(path) == STORAGE_OK
        create_session(holder, "after", source="cli")
    finally:
        holder.close()


def test_ranked_search_on_a_read_only_handle_answers_from_canonical_rows(tmp_path):
    """A read-only handle cannot persist the detach; it still answers this and later searches by LIKE."""
    path = tmp_path / "state.db"
    holder = _open(path)
    try:
        create_session(holder, "s", source="cli")
        append_message(holder, "s", role="user", content="orphanword survives in this row")
        _orphan_posting(holder, "s")
    finally:
        holder.close()

    reader = StoreHolder(db_path=path, read_only=True)
    try:
        assert reader._fts_enabled
        for _ in range(2):
            rows = search_messages(reader, "orphanword")
            assert any("survives" in (row.get("snippet") or "") for row in rows)
        assert storage_state(path) == STORAGE_OK
    finally:
        reader.close()
    raw = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        assert raw.execute("SELECT 1 FROM state_meta WHERE key = 'fts_stale'").fetchone() is None
    finally:
        raw.close()


def test_pre_v3_reopen_of_a_realigned_store_realigns_again(tmp_path):
    path = tmp_path / "state.db"
    holder = _open(path)
    try:
        create_session(holder, "s", source="cli")
        old_row = append_message(holder, "s", role="tool", content=CUT_TOOL_ROW, tool_name="terminal")
        row_id = append_message(holder, "s", role="tool", content=CUT_TOOL_ROW, tool_name="terminal")
        # What a pre-v3 release does on open of this v3 store: stamp the marker at MAX(id)
        # and swap in its id-gated delete trigger.
        holder._execute_write(lambda conn: conn.executescript(
            "INSERT OR REPLACE INTO state_meta(key, value) "
            f"VALUES('fts_tool_full_content_high_water', '{row_id}');"
            f"DROP TRIGGER messages_fts_delete;{_PRE_V3_DELETE_TRIGGER};"))
        holder._execute_write(lambda conn: conn.execute("DELETE FROM messages WHERE id = ?", (old_row,)))
    finally:
        holder.close()

    reopened = _reopen(path)
    try:
        assert reopened.get_meta("fts_tool_full_content_high_water") is None
        trigger_sql = reopened._conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'messages_fts_delete'").fetchone()[0]
        assert "fts_tool_full_content_high_water" not in trigger_sql

        hits = reopened._conn.execute("SELECT rowid FROM messages_fts WHERE messages_fts MATCH 'frag*' ORDER BY rank")
        assert [row[0] for row in hits] == [row_id]

        reopened._execute_write(lambda conn: conn.execute("DELETE FROM messages WHERE id = ?", (row_id,)))

        reopened._conn.execute(
            "SELECT rowid FROM messages_fts WHERE messages_fts MATCH 'frag*' ORDER BY rank").fetchall()
        reopened._conn.execute("INSERT INTO messages_fts(messages_fts, rank) VALUES('integrity-check', 1)")
        assert storage_state(path) == STORAGE_OK
    finally:
        reopened.close()


def test_pre_v3_reopen_of_an_empty_realigned_store_opens(tmp_path):
    """The realign of an emptied store commits its DDL and the marker removal together."""
    path = tmp_path / "state.db"
    holder = _open(path)
    try:
        create_session(holder, "s", source="cli")
        row_id = append_message(holder, "s", role="tool", content=CUT_TOOL_ROW, tool_name="terminal")
        holder._execute_write(lambda conn: conn.executescript(
            "DELETE FROM messages; INSERT OR REPLACE INTO state_meta(key, value) "
            f"VALUES('fts_tool_full_content_high_water', '{row_id}');"))
    finally:
        holder.close()

    reopened = _reopen(path)
    try:
        assert reopened.get_meta("fts_tool_full_content_high_water") is None
        append_message(reopened, "s", role="user", content="indexed after the realign")
        assert search_messages(reopened, "realign")
    finally:
        reopened.close()


def test_an_interrupted_realign_keeps_the_old_index_and_the_marker(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    holder = _open(path)
    try:
        create_session(holder, "s", source="cli")
        row_id = append_message(holder, "s", role="user", content="survivorword stays searchable")
        holder._execute_write(lambda conn: conn.execute(
            "INSERT OR REPLACE INTO state_meta(key, value) "
            "VALUES('fts_tool_full_content_high_water', ?)", (str(row_id),)))
    finally:
        holder.close()

    real_ddl = schema._execute_ddl_script_transactional

    def ddl_then_crash(cursor, ddl):
        real_ddl(cursor, ddl)
        if ddl == schema.FTS_SQL:
            raise RuntimeError("interrupted realign")

    monkeypatch.setattr(schema, "_execute_ddl_script_transactional", ddl_then_crash)
    with pytest.raises(RuntimeError, match="interrupted realign"):
        StoreHolder(db_path=path)
    monkeypatch.undo()

    raw = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        assert raw.execute(
            "SELECT 1 FROM state_meta WHERE key = 'fts_tool_full_content_high_water'").fetchone() is not None
        assert raw.execute(
            "SELECT rowid FROM messages_fts WHERE messages_fts MATCH 'survivorword'").fetchall() == [(row_id,)]
    finally:
        raw.close()
    reopened = _reopen(path)
    try:
        assert reopened.get_meta("fts_tool_full_content_high_water") is None
        assert search_messages(reopened, "survivorword")
    finally:
        reopened.close()


def test_trigram_fallback_over_orphaned_posting_fails_open(tmp_path):
    """The zero-hit trigram retry is a ranked MATCH too: its damage must detach, not raise,
    and the triggering request still answers from canonical rows."""
    path = tmp_path / "state.db"
    holder = _open(path)
    try:
        if not holder._trigram_available:
            pytest.skip("trigram tokenizer unavailable")
        create_session(holder, "s", source="cli")
        row_id = append_message(holder, "s", role="user", content="xyzorphan qqkeep")
        holder._execute_write(lambda conn: conn.execute(
            "INSERT INTO messages_fts_trigram(messages_fts_trigram, rowid, content, tool_name) "
            "VALUES('delete', ?, 'qqkeep', NULL)", (row_id,)))
        holder._execute_write(lambda conn: conn.execute(
            "INSERT INTO messages_fts(messages_fts, rowid, content, tool_name, tool_calls) "
            "VALUES('delete', ?, 'xyzorphan qqkeep', NULL, NULL)", (row_id,)))  # unicode61 misses -> trigram retry

        rows = search_messages(holder, "yzorph")

        assert any("xyzorphan" in (row.get("snippet") or "") for row in rows)
        assert storage_state(path) == STORAGE_OK
    finally:
        holder.close()


def test_or_relaxed_retry_over_orphaned_posting_answers_from_canonical_rows(tmp_path):
    path = tmp_path / "state.db"
    holder = _open(path)
    try:
        create_session(holder, "s", source="cli")
        append_message(holder, "s", role="user", content="orphanword survives in this row")
        _orphan_posting(holder, "s")

        for _ in range(2):  # the triggering request and the next, already-detached one agree
            rows = search_messages(holder, "missingterm orphanword")
            assert any("survives" in (row.get("snippet") or "") for row in rows)
        assert storage_state(path) == STORAGE_OK
    finally:
        holder.close()


def test_a_transient_ioerr_in_the_rank_probe_replays_the_read(tmp_path, monkeypatch):
    """A probe that hits a disk I/O error proves nothing: the read is retried, not latched."""
    path = tmp_path / "state.db"
    holder = _open(path)
    try:
        create_session(holder, "s", source="cli")
        append_message(holder, "s", role="user", content="orphanword survives in this row")
        _orphan_posting(holder, "s")
        original = holder._read_retrying_ioerr
        probe_failures = []

        class FlakyProbe:
            def __init__(self, conn):
                self._conn = conn

            @property
            def in_transaction(self):
                return self._conn.in_transaction

            def execute(self, sql, *args):
                if sql.startswith("SELECT bm25(") and not probe_failures:
                    probe_failures.append(sql)
                    raise sqlite3.OperationalError("disk I/O error")
                return self._conn.execute(sql, *args)

        monkeypatch.setattr(holder, "_read_retrying_ioerr", lambda fn: original(lambda conn: fn(FlakyProbe(conn))))
        rows = search_messages(holder, "orphanword")

        assert len(probe_failures) == 1
        assert any("survives" in (row.get("snippet") or "") for row in rows)
        assert storage_state(path) == STORAGE_OK
    finally:
        holder.close()


def test_a_plain_ioerr_inside_a_caller_handling_corruption_stays_an_ioerr(tmp_path, monkeypatch):
    """Only the probe's own IOERR carries a corruption to restore; an unrelated exception the caller is
    handling never becomes the search's error."""
    path = tmp_path / "state.db"
    holder = _open(path)
    try:
        create_session(holder, "s", source="cli")
        append_message(holder, "s", role="user", content="healthyword")
        original = holder._read_retrying_ioerr

        class IoerrRankedRead:
            def __init__(self, conn):
                self._conn = conn

            @property
            def in_transaction(self):
                return self._conn.in_transaction

            def execute(self, sql, *args):
                if "ORDER BY rank" in sql:
                    raise sqlite3.OperationalError("disk I/O error")
                return self._conn.execute(sql, *args)

        monkeypatch.setattr(holder, "_read_retrying_ioerr", lambda fn: original(lambda conn: fn(IoerrRankedRead(conn))))
        try:
            raise sqlite3.DatabaseError("database disk image is malformed")
        except sqlite3.DatabaseError:
            assert search_messages(holder, "healthyword") == []
        assert storage_state(path) == STORAGE_OK
    finally:
        holder.close()


def test_structural_error_the_index_does_not_reproduce_still_latches(tmp_path, monkeypatch):
    """A bare SQLITE_CORRUPT the index alone does not reproduce stays structural and latches."""
    path = tmp_path / "state.db"
    holder = _open(path)
    try:
        create_session(holder, "s", source="cli")
        append_message(holder, "s", role="user", content="healthyword")
        original = holder._read_retrying_ioerr

        class CorruptRankedRead:
            def __init__(self, conn):
                self._conn = conn

            @property
            def in_transaction(self):
                return self._conn.in_transaction

            def execute(self, sql, *args):
                if "ORDER BY rank" in sql:
                    exc = sqlite3.DatabaseError("database disk image is malformed")
                    exc.sqlite_errorcode = sqlite3.SQLITE_CORRUPT
                    raise exc
                return self._conn.execute(sql, *args)

        monkeypatch.setattr(holder, "_read_retrying_ioerr", lambda fn: original(lambda conn: fn(CorruptRankedRead(conn))))
        with pytest.raises(sqlite3.DatabaseError) as raised:
            search_messages(holder, "healthyword")
        assert raised.value.sqlite_errorcode == sqlite3.SQLITE_CORRUPT
        assert storage_state(path) == STORAGE_CORRUPT
    finally:
        holder.close()


@pytest.mark.parametrize("probe_error, retries", [("database is locked", False), ("disk I/O error", True)])
def test_an_inconclusive_rank_probe_keeps_the_corruption_closed(tmp_path, monkeypatch, probe_error, retries):
    """A probe that never answers (a non-IOERR failure, or an IOERR outlasting the retry budget)
    leaves the bare SQLITE_CORRUPT as the read reported it: raised and latched, never an empty answer."""
    path = tmp_path / "state.db"
    holder = _open(path)
    try:
        create_session(holder, "s", source="cli")
        append_message(holder, "s", role="user", content="healthyword")
        original = holder._read_retrying_ioerr
        ranked_reads = []

        class CorruptReadFailingProbe:
            def __init__(self, conn):
                self._conn = conn

            @property
            def in_transaction(self):
                return self._conn.in_transaction

            def execute(self, sql, *args):
                if "ORDER BY rank" in sql:
                    ranked_reads.append(sql)
                    exc = sqlite3.DatabaseError("database disk image is malformed")
                    exc.sqlite_errorcode = sqlite3.SQLITE_CORRUPT
                    raise exc
                if sql.startswith("SELECT bm25("):
                    raise sqlite3.OperationalError(probe_error)
                return self._conn.execute(sql, *args)

        monkeypatch.setattr(holder, "_read_retrying_ioerr",
                            lambda fn: original(lambda conn: fn(CorruptReadFailingProbe(conn))))
        with pytest.raises(sqlite3.DatabaseError) as raised:
            search_messages(holder, "healthyword")
        assert raised.value.sqlite_errorcode == sqlite3.SQLITE_CORRUPT
        assert storage_state(path) == STORAGE_CORRUPT
        assert (len(ranked_reads) > 1) is retries
    finally:
        holder.close()


def test_an_opener_that_waited_for_admission_skips_the_rebuild_a_peer_finished(tmp_path, monkeypatch):
    """Two openers saw the marker; the one admitted second re-reads it under the lock and does not rebuild."""
    path = tmp_path / "state.db"
    holder = _open(path)
    try:
        create_session(holder, "s", source="cli")
        row_id = append_message(holder, "s", role="user", content="raceword")
    finally:
        holder.close()
    first, second = _reopen(path), _reopen(path)
    try:
        first._execute_write(lambda conn: conn.execute(
            "INSERT OR REPLACE INTO state_meta(key, value) "
            "VALUES('fts_tool_full_content_high_water', ?)", (str(row_id),)))
        rebuilds = []
        for name, opener in (("first", first), ("second", second)):
            opener._conn.set_trace_callback(
                lambda stmt, name=name: rebuilds.append(name) if "VALUES('rebuild')" in stmt else None)
        real_admitted = schema._run_admitted_startup_rebuild

        def peer_finishes_first(holder, cursor, rebuild_fn):
            if holder is first:  # the peer's own admission runs as usual
                schema._migrate_misaligned_fts_source(second, second._conn.cursor(), legacy=False)
            real_admitted(holder, cursor, rebuild_fn)

        monkeypatch.setattr(schema, "_run_admitted_startup_rebuild", peer_finishes_first)
        schema._migrate_misaligned_fts_source(first, first._conn.cursor(), legacy=False)

        assert rebuilds == ["second"]
        assert first.get_meta("fts_tool_full_content_high_water") is None
        assert search_messages(first, "raceword")
    finally:
        first.close()
        second.close()
