"""FTS storage-layout drift: one structural answer shared by the stamp, the update notice and doctor.

Regression for #129927. The ``fts_storage_version`` stamp was written by a base-index realign that never
looked at the trigram vtable, doctor trusted that stamp, and the shape check only knew two historical
layouts, so the next layout bump would be stamped current while still stale.
"""

from __future__ import annotations

import contextlib
import io
import os
import pathlib
import re
import sqlite3

import pytest

import hermes_state_common
from hermes_state import SessionDB


def _require_trigram(db: SessionDB) -> None:
    if not db._trigram_available:
        db.close()
        pytest.skip("trigram tokenizer unavailable in this SQLite build")


def _install_v1_trigram(db: SessionDB) -> None:
    """The trigram layout `optimize-storage` produced between #65798 and #101729 (projects ``tool_calls``)."""
    db._conn.executescript(
        """
        DROP TRIGGER messages_fts_trigram_insert;
        DROP TRIGGER messages_fts_trigram_delete;
        DROP TRIGGER messages_fts_trigram_update;
        DROP TABLE messages_fts_trigram;
        DROP VIEW messages_fts_trigram_src;
        CREATE VIEW messages_fts_trigram_src AS
            SELECT id, role, content, tool_name, tool_calls FROM messages WHERE role <> 'tool';
        CREATE VIRTUAL TABLE messages_fts_trigram USING fts5(
            content, tool_name, tool_calls,
            content='messages_fts_trigram_src', content_rowid='id', tokenize='trigram'
        );
        INSERT INTO messages_fts_trigram(messages_fts_trigram) VALUES('rebuild');
        """
    )


def _install_raw_messages_base(db: SessionDB) -> None:
    """The pre-v3 base index: external-content over raw ``messages`` (realigned on the next open)."""
    names = [r[0] for r in db._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='trigger' AND name IN "
        "('messages_fts_insert','messages_fts_delete','messages_fts_update')")]
    db._conn.executescript(
        "".join(f"DROP TRIGGER {n};" for n in names)
        + """
        DROP TABLE messages_fts;
        CREATE VIRTUAL TABLE messages_fts USING fts5(
            content, tool_name, tool_calls, content='messages', content_rowid='id'
        );
        INSERT INTO messages_fts(messages_fts) VALUES('rebuild');
        """
    )


def _stamp(db_path: pathlib.Path):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = conn.execute("SELECT value FROM state_meta WHERE key = 'fts_storage_version'").fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _seed(db: SessionDB) -> None:
    db.create_session("cli", source="cli")
    db.append_message("cli", role="user", content="deploy the payment service")
    db.append_message(
        "cli", role="assistant", content="running",
        tool_calls=[{"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}],
    )


@pytest.fixture
def v1_trigram_db(tmp_path):
    """tek's state: optimized at #65798 (v1 trigram, raw-messages base), then reopened by current code."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    _require_trigram(db)
    _seed(db)
    _install_v1_trigram(db)
    _install_raw_messages_base(db)
    db._conn.execute("UPDATE state_meta SET value = '1' WHERE key = 'fts_storage_version'")
    db._conn.commit()
    db.close()
    SessionDB(db_path=db_path).close()  # the gateway/CLI open that realigns only the base index
    return db_path


def test_realign_does_not_stamp_current_while_the_trigram_is_stale(v1_trigram_db):
    assert _stamp(v1_trigram_db) != str(hermes_state_common.FTS_STORAGE_VERSION)


def test_doctor_offers_optimize_storage_for_a_stale_trigram(v1_trigram_db, monkeypatch):
    from hermes_cli import doctor_state
    from hermes_state_dbfile import collect_state_db_stats

    stats = collect_state_db_stats(v1_trigram_db)
    monkeypatch.setattr(doctor_state, "STATE_DB_SIZE_WARN_BYTES", 0)
    rows = doctor_state._render_state_db_stats(stats)
    assert any("hermes sessions optimize-storage" in detail for _, _, detail in rows)


def test_fresh_store_is_stamped_on_creation_and_doctor_stays_quiet(tmp_path, monkeypatch):
    from hermes_cli import doctor_state
    from hermes_state_dbfile import collect_state_db_stats

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    _seed(db)
    db.close()
    assert _stamp(db_path) == str(hermes_state_common.FTS_STORAGE_VERSION)
    monkeypatch.setattr(doctor_state, "STATE_DB_SIZE_WARN_BYTES", 0)
    rows = doctor_state._render_state_db_stats(collect_state_db_stats(db_path))
    assert not any("hermes sessions optimize-storage" in detail for _, _, detail in rows)


def _notice_for(db_path: pathlib.Path, monkeypatch) -> str:
    """Run the `hermes update` notice against *db_path* with the size gate satisfied for state.db only."""
    from hermes_cli import update_cmd_maint
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: db_path.parent)
    real_stat = pathlib.Path.stat

    def _big(path, *args, **kwargs):
        st = real_stat(path, *args, **kwargs)
        if path.name == "state.db":
            return os.stat_result((st.st_mode, st.st_ino, st.st_dev, st.st_nlink, st.st_uid, st.st_gid,
                                   10 * 1024 ** 3, st.st_atime, st.st_mtime, st.st_ctime))
        return st

    monkeypatch.setattr(pathlib.Path, "stat", _big)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        update_cmd_maint._print_fts_optimize_available_notice()
    return buf.getvalue()


def test_notice_for_a_stale_trigram_does_not_claim_the_legacy_reclaim(v1_trigram_db, monkeypatch):
    out = _notice_for(v1_trigram_db, monkeypatch)
    assert "hermes sessions optimize-storage" in out
    assert "60%" not in out and "old storage layout" not in out.lower()


def test_notice_for_a_legacy_inline_store_keeps_the_reclaim_estimate(tmp_path, monkeypatch):
    from hermes_state_common import LEGACY_FTS_SQL

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    _seed(db)
    names = [r[0] for r in db._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'messages_fts%'")]
    db._conn.executescript(
        "".join(f"DROP TRIGGER {n};" for n in names)
        + "DROP TABLE IF EXISTS messages_fts_cjk; DROP TABLE IF EXISTS messages_fts_trigram;"
        + "DROP VIEW IF EXISTS messages_fts_trigram_src; DROP TABLE messages_fts;"
        + LEGACY_FTS_SQL
    )
    db._conn.commit()
    db.close()
    assert "60%" in _notice_for(db_path, monkeypatch)


def test_a_base_index_the_next_open_realigns_is_not_offered_as_opt_in_work(tmp_path, monkeypatch):
    """A raw-``messages`` base index with a current trigram is realigned automatically on the next read-write
    open, so the update notice (read-only, before that open) must not send the user to optimize-storage."""
    import hermes_state_fts

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    _require_trigram(db)
    _seed(db)
    _install_raw_messages_base(db)
    db._conn.execute("UPDATE state_meta SET value = '2' WHERE key = 'fts_storage_version'")
    db._conn.commit()
    db.close()

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        assert hermes_state_fts.fts_layout_drift(conn.cursor()) == []
    finally:
        conn.close()
    assert "optimize-storage" not in _notice_for(db_path, monkeypatch)
    SessionDB(db_path=db_path).close()
    assert _stamp(db_path) == str(hermes_state_common.FTS_STORAGE_VERSION)


def test_a_layout_change_the_code_has_not_shipped_yet_is_reported_as_drift(tmp_path, monkeypatch):
    """After a DDL change appends its new form to the registry, stores on the previous form are drift."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    _seed(db)
    db.close()
    assert _stamp(db_path) == str(hermes_state_common.FTS_STORAGE_VERSION)

    import hermes_state_fts
    bumped = hermes_state_common.FTS_SQL.replace(
        "content_rowid='id'\n);", "content_rowid='id',\n    tokenize='porter unicode61'\n);", 1)
    assert bumped != hermes_state_common.FTS_SQL
    monkeypatch.setattr(hermes_state_fts, "FTS_SQL", bumped)
    layouts = dict(hermes_state_fts.FTS_VTABLE_LAYOUTS)
    layouts["messages_fts"] = (*layouts["messages_fts"], hermes_state_fts._expected_vtable_sql(bumped, "messages_fts"))
    monkeypatch.setattr(hermes_state_fts, "FTS_VTABLE_LAYOUTS", layouts)

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        drift = hermes_state_fts.fts_layout_drift(conn.cursor())
    finally:
        conn.close()
    assert "messages_fts" in drift


def test_the_layout_registry_ends_with_the_shipping_ddl():
    """A change to the FTS DDL must append its stored form, keeping the previous one recognisable as old."""
    import hermes_state_fts
    from hermes_state_common import FTS_SQL, FTS_TRIGRAM_SQL, LEGACY_FTS_SQL, LEGACY_FTS_TRIGRAM_SQL

    layouts = hermes_state_fts.FTS_VTABLE_LAYOUTS
    assert layouts["messages_fts"][-1] == hermes_state_fts._expected_vtable_sql(FTS_SQL, "messages_fts")
    assert layouts["messages_fts_trigram"][-1] == hermes_state_fts._expected_vtable_sql(
        FTS_TRIGRAM_SQL, "messages_fts_trigram")
    assert hermes_state_fts._expected_vtable_sql(LEGACY_FTS_SQL, "messages_fts") in layouts["messages_fts"][:-1]
    assert hermes_state_fts._expected_vtable_sql(
        LEGACY_FTS_TRIGRAM_SQL, "messages_fts_trigram") in layouts["messages_fts_trigram"][:-1]


def _install_newer_base_layout(db: SessionDB) -> str:
    """A base-index definition this code does not recognise (a newer release's), stamped with a newer version."""
    from hermes_state_common import FTS_SQL, FTS_STORAGE_VERSION

    sql = db._conn.execute("SELECT sql FROM sqlite_master WHERE name = 'messages_fts'").fetchone()[0]
    db._conn.executescript(
        "DROP TRIGGER messages_fts_insert; DROP TRIGGER messages_fts_delete; DROP TRIGGER messages_fts_update;"
        "DROP TABLE messages_fts;")
    newer = sql.replace("content_rowid='id'", "content_rowid='id', tokenize='porter unicode61'")
    db._conn.execute(newer)
    db._conn.executescript(FTS_SQL)
    db._conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
    db._conn.execute("UPDATE state_meta SET value = ? WHERE key = 'fts_storage_version'", (str(FTS_STORAGE_VERSION + 1),))
    db._conn.commit()
    return newer


def _forbid_demote(db: SessionDB, monkeypatch) -> None:
    def _demote():
        pytest.fail("optimize-storage demoted an index it does not recognise as an old layout")
    monkeypatch.setattr(db, "_demote_legacy_fts_to_trash", _demote)


def test_a_layout_from_a_newer_release_is_left_alone(tmp_path, monkeypatch):
    """Downgrade: an unrecognised (newer) definition is neither rebuilt nor re-stamped as this release's."""
    from hermes_state_common import FTS_STORAGE_VERSION

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    _seed(db)
    _install_newer_base_layout(db)
    db.close()

    db = SessionDB(db_path=db_path)
    try:
        assert not db.fts_optimize_available()
        _forbid_demote(db, monkeypatch)
        db.optimize_fts_storage(vacuum=False)
    finally:
        db.close()
    assert _stamp(db_path) == str(FTS_STORAGE_VERSION + 1)


def test_a_quoted_table_name_in_the_stored_definition_is_still_current(tmp_path, monkeypatch):
    """ALTER TABLE ... RENAME round trips store the name quoted; the layout is unchanged and must not be rebuilt."""
    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    _require_trigram(db)
    _seed(db)
    db._conn.execute("ALTER TABLE messages_fts_trigram RENAME TO renamed_trigram")
    db._conn.execute("ALTER TABLE renamed_trigram RENAME TO messages_fts_trigram")
    stored = db._conn.execute("SELECT sql FROM sqlite_master WHERE name = 'messages_fts_trigram'").fetchone()[0]
    assert '"messages_fts_trigram"' in stored
    db._conn.execute("DELETE FROM state_meta WHERE key = 'fts_storage_version'")
    db._conn.commit()
    db.close()

    db = SessionDB(db_path=db_path)
    try:
        _forbid_demote(db, monkeypatch)
        db.optimize_fts_storage(vacuum=False)
    finally:
        db.close()
    assert _stamp(db_path) == str(hermes_state_common.FTS_STORAGE_VERSION)


def test_optimize_settle_refuses_pending_backfill(tmp_path):
    """Settle must not stamp while high_water markers remain."""
    db = SessionDB(db_path=tmp_path / "fresh.db")
    try:
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", role="user", content="settle guard needle")
        # A fresh store is stamped on creation; clear it so the assertion reads settle's own write.
        db._conn.execute("DELETE FROM state_meta WHERE key = 'fts_storage_version'")
        db._conn.commit()
        # Plant markers without going through demote.
        db.set_meta("fts_rebuild_high_water", "1")
        db.set_meta("fts_rebuild_progress", "0")
        # The public contract: optimize returns ok=False when still
        # pending. Simulate an unfinishable backfill by stubbing the
        # chunk step to a no-op while markers stay.
        db.fts_rebuild_step = lambda: False  # type: ignore[method-assign]
        result = db.optimize_fts_storage(vacuum=False)
        assert result["ok"] is False
        assert result.get("reason") == "backfill_incomplete"
        assert db.get_meta("fts_storage_version") is None
        assert db.get_meta("fts_rebuild_high_water") is not None
    finally:
        db.close()


def test_a_stale_base_source_view_is_replaced_before_the_store_is_stamped(tmp_path):
    from hermes_state_common import FTS_STORAGE_VERSION, FTS_TOOL_CONTENT_PREFIX_CHARS

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    _seed(db)
    db.append_message("cli", role="tool", content="x" * FTS_TOOL_CONTENT_PREFIX_CHARS + " trailingneedle")
    current_view = db._conn.execute("SELECT sql FROM sqlite_master WHERE name = 'messages_fts_src'").fetchone()[0]
    db._conn.execute("DROP VIEW messages_fts_src")
    db._conn.execute("CREATE VIEW messages_fts_src AS SELECT id, content, tool_name, tool_calls FROM messages")
    db._conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
    db._conn.execute("DELETE FROM state_meta WHERE key = 'fts_storage_version'")
    db._conn.commit()
    db.close()

    db = SessionDB(db_path=db_path)
    try:
        view = db._conn.execute("SELECT sql FROM sqlite_master WHERE name = 'messages_fts_src'").fetchone()[0]
        assert " ".join(view.split()) == " ".join(current_view.split())
        assert db.search_messages("trailingneedle") == []
        db._conn.execute("INSERT INTO messages_fts(messages_fts, rank) VALUES('integrity-check', 1)")
    finally:
        db.close()
    assert _stamp(db_path) == str(FTS_STORAGE_VERSION)


def _interrupted_raw_base_backfill(db_path: pathlib.Path) -> int:
    """An older release's optimize-storage stopped mid-backfill: raw-``messages`` base index, trigram table and the
    shared markers left behind, one of two rows indexed. Returns the id of the row the backfill had not reached."""
    db = SessionDB(db_path=db_path)
    _require_trigram(db)
    db.create_session("s1", source="cli")
    first_id = db.append_message("s1", role="user", content="already indexed")
    last_id = db.append_message("s1", role="user", content="大别山项目 deployment")
    db._reset_fts_index_to_empty(db._conn)
    _install_raw_messages_base(db)
    db._conn.execute("DELETE FROM messages_fts")
    db._conn.execute(
        "INSERT INTO messages_fts(rowid, content, tool_name, tool_calls) "
        "SELECT id, content, tool_name, tool_calls FROM messages WHERE id = ?", (first_id,))
    db._conn.execute(
        "INSERT INTO messages_fts_trigram(rowid, content, tool_name) "
        "SELECT id, content, tool_name FROM messages WHERE id = ?", (first_id,))
    db.set_meta("fts_storage_version", "2")
    db.set_meta("fts_rebuild_high_water", str(last_id))
    db.set_meta("fts_rebuild_progress", str(first_id))
    db._conn.commit()
    db.close()
    return last_id


def test_the_base_realign_finishes_a_pending_trigram_backfill(tmp_path):
    """The open-time base-index realign clears the shared backfill markers, so it must also complete the trigram."""
    db_path = tmp_path / "state.db"
    last_id = _interrupted_raw_base_backfill(db_path)
    db = SessionDB(db_path=db_path)
    try:
        assert [row["id"] for row in db.search_messages("大别山")] == [last_id]
    finally:
        db.close()


class _NoTrigramTokenizerCursor(sqlite3.Cursor):
    """A SQLite build without the trigram tokenizer opening a store whose trigram table already exists."""

    _TRIGRAM = re.compile(r"\b(?:FROM|INTO)\s+messages_fts_trigram\b(?!_)|tokenize='trigram'", re.IGNORECASE)

    def execute(self, sql, parameters=()):
        if self._TRIGRAM.search(sql):
            raise sqlite3.OperationalError("no such tokenizer: trigram")
        return super().execute(sql, parameters)

    def executescript(self, sql_script):
        if self._TRIGRAM.search(sql_script):
            raise sqlite3.OperationalError("no such tokenizer: trigram")
        return super().executescript(sql_script)


class _NoTrigramTokenizerConnection(sqlite3.Connection):
    def cursor(self, factory=None):
        return super().cursor(factory or _NoTrigramTokenizerCursor)


def test_the_base_realign_without_the_trigram_tokenizer_keeps_the_store_open_and_resumes_later(tmp_path, monkeypatch):
    """Without the tokenizer the realign still opens the store with base search; the trigram index is rebuilt by the
    first open that has the tokenizer."""
    db_path = tmp_path / "state.db"
    last_id = _interrupted_raw_base_backfill(db_path)
    real_connect = sqlite3.connect

    def connect(*args, **kwargs):
        kwargs["factory"] = _NoTrigramTokenizerConnection
        return real_connect(*args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr("hermes_state.sqlite3.connect", connect)
        db = SessionDB(db_path=db_path)
        try:
            assert [row["id"] for row in db.search_messages("deployment")] == [last_id]
        finally:
            db.close()

    db = SessionDB(db_path=db_path)
    try:
        assert [row["id"] for row in db.search_messages("大别山")] == [last_id]
    finally:
        db.close()


def test_a_stale_source_view_does_not_replace_a_newer_base_index(tmp_path):
    """Downgrade with a differing source view: the open-time repair leaves an unrecognised base index alone."""
    from hermes_state_common import FTS_STORAGE_VERSION

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    _seed(db)
    newer = _install_newer_base_layout(db)
    db._conn.execute("DROP VIEW messages_fts_src")
    db._conn.execute("CREATE VIEW messages_fts_src AS SELECT id, content, tool_name, tool_calls FROM messages")
    db._conn.commit()
    db.close()

    SessionDB(db_path=db_path).close()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        assert conn.execute("SELECT sql FROM sqlite_master WHERE name = 'messages_fts'").fetchone()[0] == newer
    finally:
        conn.close()
    assert _stamp(db_path) == str(FTS_STORAGE_VERSION + 1)


def test_optimize_storage_on_a_newer_layout_finishes_its_work_without_restamping(tmp_path):
    """Leftover old-index tables on a downgraded store are torn down and reported as done; the stamp stays."""
    from hermes_state_common import FTS_STORAGE_VERSION

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    _seed(db)
    _install_newer_base_layout(db)
    db._conn.execute("CREATE TABLE fts_v22_trash_messages_fts_data (docid INTEGER PRIMARY KEY, block BLOB)")
    db._conn.commit()
    db.close()

    db = SessionDB(db_path=db_path)
    try:
        result = db.optimize_fts_storage(vacuum=False)
        assert result["ok"] is True, result
        assert not db._has_fts_trash(db._conn)
    finally:
        db.close()
    assert _stamp(db_path) == str(FTS_STORAGE_VERSION + 1)


def test_doctor_offers_the_resume_after_backfill_before_teardown(tmp_path, monkeypatch):
    """optimize-storage interrupted after backfill but before trash teardown: doctor agrees with the update notice."""
    from hermes_cli import doctor_state
    from hermes_state_dbfile import collect_state_db_stats

    db_path = tmp_path / "state.db"
    db = SessionDB(db_path=db_path)
    _require_trigram(db)
    _seed(db)
    db._demote_legacy_fts_to_trash()
    while db.fts_rebuild_step():
        pass
    assert db.get_meta("fts_rebuild_high_water") is None and db._has_fts_trash(db._conn)
    db.close()

    assert "optimize-storage" in _notice_for(db_path, monkeypatch)
    monkeypatch.setattr(doctor_state, "STATE_DB_SIZE_WARN_BYTES", 0)
    rows = doctor_state._render_state_db_stats(collect_state_db_stats(db_path))
    assert any("hermes sessions optimize-storage" in detail for _, _, detail in rows)


@pytest.mark.parametrize("mode, heading", [("advise", "Search index update available"),
                                           ("require", "required by your notice policy")])
def test_the_follow_up_notice_makes_no_size_or_effort_claim(v1_trigram_db, monkeypatch, mode, heading):
    from hermes_cli import config as hermes_config

    monkeypatch.setattr(hermes_config, "load_config", lambda: {"sessions": {"fts_optimize_notice": mode}})
    out = _notice_for(v1_trigram_db, monkeypatch)
    assert heading in out
    assert "%" not in out and "smaller" not in out.lower()
