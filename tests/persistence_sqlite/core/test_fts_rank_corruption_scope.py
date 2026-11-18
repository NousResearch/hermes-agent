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

from persistence.sqlite._holder import StoreHolder
from persistence.sqlite._schema_ddl import FTS_TOOL_CONTENT_PREFIX_CHARS
from persistence.sqlite._state.health import STORAGE_CORRUPT, STORAGE_OK, reset_storage_state, storage_state
from persistence.sqlite.conversation._store.messages import append_message
from persistence.sqlite.conversation._store.search import _rescope_ranked_fts_corruption, search_messages
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

        # Answered (LIKE fail-open over canonical rows), not raised; the file is not latched.
        assert any("survives" in (row.get("snippet") or "") for row in rows)
        assert holder._fts_stale  # the fail-open path detached the index for the repair paths
        assert storage_state(path) == STORAGE_OK
        create_session(holder, "after", source="cli")  # writes still accepted
    finally:
        holder.close()


def test_pre_v3_reopen_of_a_realigned_store_realigns_again(tmp_path):
    path = tmp_path / "state.db"
    holder = _open(path)
    try:
        create_session(holder, "s", source="cli")
        row_id = append_message(holder, "s", role="tool", content=CUT_TOOL_ROW, tool_name="terminal")
        # What a pre-v3 release does on open of this v3 store: stamp the marker at MAX(id)
        # and swap in its id-gated delete trigger.
        holder._execute_write(lambda conn: conn.executescript(
            "INSERT OR REPLACE INTO state_meta(key, value) "
            f"VALUES('fts_tool_full_content_high_water', '{row_id}');"
            f"DROP TRIGGER messages_fts_delete;{_PRE_V3_DELETE_TRIGGER};"))
    finally:
        holder.close()

    reopened = _open(path)
    try:
        assert reopened.get_meta("fts_tool_full_content_high_water") is None
        trigger_sql = reopened._conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'messages_fts_delete'").fetchone()[0]
        assert "fts_tool_full_content_high_water" not in trigger_sql

        reopened._execute_write(lambda conn: conn.execute("DELETE FROM messages WHERE id = ?", (row_id,)))

        # No orphaned "frag" posting: the ranked read and FTS5's strict checker both pass.
        reopened._conn.execute(
            "SELECT rowid FROM messages_fts WHERE messages_fts MATCH 'frag*' ORDER BY rank").fetchall()
        reopened._conn.execute("INSERT INTO messages_fts(messages_fts, rank) VALUES('integrity-check', 1)")
        assert storage_state(path) == STORAGE_OK
    finally:
        reopened.close()


def test_pre_v3_reopen_of_an_empty_realigned_store_opens(tmp_path):
    """The realign of an emptied store runs inside a savepoint; its DDL must not commit it away."""
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

    reopened = _open(path)
    try:
        assert reopened.get_meta("fts_tool_full_content_high_water") is None
        append_message(reopened, "s", role="user", content="indexed after the realign")
        assert search_messages(reopened, "realign")
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

        rows = search_messages(holder, "missingterm orphanword")

        assert any("survives" in (row.get("snippet") or "") for row in rows)
        assert storage_state(path) == STORAGE_OK
    finally:
        holder.close()


def test_genuine_corruption_on_the_ranked_path_still_latches(tmp_path):
    """The rescope only re-attributes when the index itself reports damage: a bare
    SQLITE_CORRUPT the FTS-only probe cannot reproduce stays structural and latches."""
    path = tmp_path / "state.db"
    holder = _open(path)
    try:
        create_session(holder, "s", source="cli")
        append_message(holder, "s", role="user", content="healthyword")
        rescope = _rescope_ranked_fts_corruption(holder, "messages_fts", "healthyword*", "ORDER BY rank")
        assert _rescope_ranked_fts_corruption(holder, "messages_fts", "x", "ORDER BY m.timestamp DESC") is None

        def corrupt(_conn):
            exc = sqlite3.DatabaseError("database disk image is malformed")
            exc.sqlite_errorcode = sqlite3.SQLITE_CORRUPT
            raise exc

        with pytest.raises(sqlite3.DatabaseError) as raised:
            holder._read_retrying_ioerr(corrupt, rescope=rescope)
        assert raised.value.sqlite_errorcode == sqlite3.SQLITE_CORRUPT
        assert storage_state(path) == STORAGE_CORRUPT
    finally:
        holder.close()
