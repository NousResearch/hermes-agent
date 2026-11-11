"""Retirement is atomic across connections and health checks leave writers free."""

import os
import sqlite3
from pathlib import Path

import pytest

import hermes_state_repair as repair
from hermes_state import SessionDB
from hermes_state_fts import drop_retired_cjk_index
from tests.hermes_state import test_fts_cjk_index_retirement as retirement
from tests.hermes_state.test_fts_cjk_index_retirement import (
    _cjk_residue,
    _store_snapshot,
)

cjk_db = retirement.cjk_db


def test_retirement_excludes_an_interleaved_retirement(cjk_db):
    competing_results = []

    class InterleavedCursor(sqlite3.Cursor):
        def execute(self, sql, parameters=()):
            if sql.startswith("ALTER TABLE messages_fts_cjk_data RENAME"):
                with sqlite3.connect(cjk_db, isolation_level=None, timeout=0) as second:
                    try:
                        competing_results.append(drop_retired_cjk_index(second.cursor()))
                    except sqlite3.OperationalError as exc:
                        competing_results.append(str(exc))
            return super().execute(sql, parameters)

    with sqlite3.connect(cjk_db, isolation_level=None) as first:
        assert drop_retired_cjk_index(first.cursor(factory=InterleavedCursor))
        assert not first.in_transaction
    assert competing_results == ["database is locked"]
    SessionDB(db_path=cjk_db).close()
    assert _cjk_residue(cjk_db) == []


def test_repair_integrity_scan_allows_a_concurrent_writer(cjk_db, monkeypatch):
    before = _store_snapshot(cjk_db)
    connect = repair._connect_repair_durable
    writes = []

    def trace(sql):
        if sql.upper() == "PRAGMA INTEGRITY_CHECK":
            with sqlite3.connect(cjk_db, timeout=0) as writer:
                try:
                    writer.execute("UPDATE sessions SET source = source WHERE id = 's1'")
                    writer.commit()
                    writes.append("ok")
                except sqlite3.OperationalError as exc:
                    writes.append(str(exc))

    def traced_connect(*args, **kwargs):
        conn = connect(*args, **kwargs)
        conn.set_trace_callback(trace)
        return conn

    monkeypatch.setattr(repair, "_connect_repair_durable", traced_connect)
    assert repair._db_opens_cleanly(cjk_db) is None
    assert writes == ["ok"]
    assert _store_snapshot(cjk_db) == before


@pytest.mark.parametrize("outer_transaction", [False, True])
def test_failed_retirement_preserves_the_store(cjk_db, outer_transaction):
    before = _store_snapshot(cjk_db)

    class FailingCursor(sqlite3.Cursor):
        def execute(self, sql, parameters=()):
            if sql.startswith("ALTER TABLE messages_fts_cjk_data RENAME"):
                raise sqlite3.OperationalError("injected rename failure")
            return super().execute(sql, parameters)

    with sqlite3.connect(cjk_db, isolation_level=None) as conn:
        if outer_transaction:
            conn.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.OperationalError, match="injected rename failure"):
            drop_retired_cjk_index(conn.cursor(factory=FailingCursor))
        assert conn.in_transaction is outer_transaction
        if outer_transaction:
            conn.rollback()
    assert _store_snapshot(cjk_db) == before


def test_successful_retirement_leaves_the_callers_transaction_open(cjk_db):
    before = _store_snapshot(cjk_db)
    with sqlite3.connect(cjk_db, isolation_level=None) as conn:
        conn.execute("BEGIN IMMEDIATE")
        assert drop_retired_cjk_index(conn.cursor())
        assert conn.in_transaction
        conn.rollback()
    assert _store_snapshot(cjk_db) == before


def test_a_copy_the_os_refuses_is_not_reported_as_damage(cjk_db, monkeypatch):
    before = _store_snapshot(cjk_db)

    def fail_snapshot(*args, **kwargs):
        raise PermissionError("snapshot storage unavailable")

    monkeypatch.setattr(repair, "_copy_database_snapshot", fail_snapshot)
    reason = repair._db_opens_cleanly(cjk_db)
    assert isinstance(reason, repair.HealthCheckUnavailable) and "snapshot storage unavailable" in reason
    assert _store_snapshot(cjk_db) == before


def test_retirement_rechecks_after_acquiring_the_write_lock(cjk_db):
    class InterleavedCursor(sqlite3.Cursor):
        def execute(self, sql, parameters=()):
            if sql == "BEGIN IMMEDIATE":
                with sqlite3.connect(cjk_db, isolation_level=None) as second:
                    assert drop_retired_cjk_index(second.cursor())
            return super().execute(sql, parameters)

    with sqlite3.connect(cjk_db, isolation_level=None) as first:
        assert not drop_retired_cjk_index(first.cursor(factory=InterleavedCursor))
        assert not first.in_transaction
    assert _cjk_residue(cjk_db) == []



def test_a_full_temporary_volume_is_neither_healthy_nor_damage(cjk_db, monkeypatch, capsys):
    """The probe judges a copy of the store; a copy that does not fit says nothing either way about the store."""
    import argparse

    import hermes_state
    from hermes_cli import sessions_cmd

    before = _store_snapshot(cjk_db)

    def full_volume(*args, **kwargs):
        raise sqlite3.OperationalError("database or disk is full")

    monkeypatch.setattr(repair, "_copy_database_snapshot", full_volume)
    assert isinstance(repair._db_opens_cleanly(cjk_db), repair.HealthCheckUnavailable)

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", cjk_db)
    for check_only in (True, False):
        assert sessions_cmd._cmd_repair(argparse.Namespace(check_only=check_only, no_backup=True)) == 1
        assert "Could not check" in capsys.readouterr().out

    monkeypatch.setattr(repair, "_live_writer_holds_db", lambda _path: False)
    report = repair.repair_state_db_schema(cjk_db, backup=False)
    assert not report["repaired"] and report["strategy"] is None
    assert "could not check state.db before repair" in report["error"]
    assert _store_snapshot(cjk_db) == before


def test_no_usable_temporary_directory_is_not_reported_as_damage(cjk_db, monkeypatch):
    import tempfile

    def no_tempdir(*args, **kwargs):
        raise FileNotFoundError("No usable temporary directory found")

    monkeypatch.setattr(tempfile, "TemporaryDirectory", no_tempdir)
    monkeypatch.setattr(tempfile, "gettempdir", no_tempdir)
    reason = repair._db_opens_cleanly(cjk_db)
    assert isinstance(reason, repair.HealthCheckUnavailable) and "No usable temporary directory" in reason


@pytest.mark.platforms("posix")
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root writes into chmod 500 dirs")
def test_a_copy_sqlite_cannot_write_is_not_reported_as_damage(cjk_db, monkeypatch, tmp_path):
    import tempfile

    unwritable = tmp_path / "unwritable"
    unwritable.mkdir()
    real = tempfile.TemporaryDirectory
    monkeypatch.setattr(tempfile, "TemporaryDirectory", lambda **kw: real(dir=unwritable, **kw))
    real_conn = repair._repair_conn

    def readonly_destination(path, *args, **kwargs):
        if Path(path).parent.parent == unwritable:
            os.chmod(Path(path).parent, 0o500)
        return real_conn(path, *args, **kwargs)

    monkeypatch.setattr(repair, "_repair_conn", readonly_destination)
    try:
        reason = repair._db_opens_cleanly(cjk_db)
    finally:
        for child in unwritable.iterdir():
            os.chmod(child, 0o700)
    assert isinstance(reason, repair.HealthCheckUnavailable), reason



@pytest.mark.parametrize("damage", ["missing", "unreadable"])
def test_base_fts_damage_beside_the_retired_index_routes_to_repair(cjk_db, damage):
    with sqlite3.connect(cjk_db, isolation_level=None) as conn:
        if damage == "missing":
            conn.execute("DROP TABLE messages_fts_config")
        else:
            conn.execute("DELETE FROM messages_fts_config")
            conn.execute("INSERT INTO messages_fts_config VALUES ('version', 99)")
    reason = repair._db_opens_cleanly(cjk_db)
    assert reason is not None and not isinstance(reason, repair.HealthCheckUnavailable), reason



def test_repair_heals_base_fts_damage_beside_the_retired_index(cjk_db, monkeypatch):
    import contextlib
    import gc

    gc.collect()  # the fixture's outside-write probe leaves a raw connection for GC
    with contextlib.closing(sqlite3.connect(cjk_db, isolation_level=None)) as conn:
        conn.execute("DELETE FROM messages_fts_config")
        conn.execute("INSERT INTO messages_fts_config VALUES ('version', 99)")
    monkeypatch.setattr(repair, "_live_writer_holds_db", lambda _path: False)
    report = repair.repair_state_db_schema(cjk_db, backup=False)
    assert report["repaired"], report
    assert repair._db_opens_cleanly(cjk_db) is None
    with sqlite3.connect(cjk_db) as raw:
        assert raw.execute("SELECT content FROM messages").fetchall() == [("东京都",)]


@pytest.mark.platforms("posix")
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root writes read-only files")
def test_a_copy_that_turns_read_only_is_not_reported_as_damage(cjk_db, monkeypatch):
    copy = repair._copy_database_snapshot
    frozen = []

    def copy_then_freeze(source, destination, **kwargs):
        copy(source, destination, **kwargs)
        destination.chmod(0o400)
        destination.parent.chmod(0o500)
        frozen.append(destination.parent)

    monkeypatch.setattr(repair, "_copy_database_snapshot", copy_then_freeze)
    try:
        reason = repair._db_opens_cleanly(cjk_db)
    finally:
        for probe_dir in frozen:
            if probe_dir.exists():
                probe_dir.chmod(0o700)
    assert isinstance(reason, repair.HealthCheckUnavailable), reason


def test_retirement_succeeds_beside_an_index_whose_tokenizer_is_missing(cjk_db):
    """Renaming the retired shadows must not re-parse another index's triggers: a build without trigram would fail
    there before the open's own tokenizer-unavailable handling runs. Discriminates only on SQLite whose modern rename
    re-parses triggers (3.33.0 and 3.45.1 fail without the fix); 3.53.1 passes either way, so a green run there is
    not coverage of the fix."""
    import contextlib

    with contextlib.closing(sqlite3.connect(cjk_db, isolation_level=None)) as raw:
        raw.execute("PRAGMA writable_schema=ON")
        raw.execute("UPDATE sqlite_master SET sql = replace(sql, \"tokenize='trigram'\", \"tokenize='absent_trigram'\") "
                    "WHERE name = 'messages_fts_trigram'")
        version = raw.execute("PRAGMA schema_version").fetchone()[0]
        raw.execute(f"PRAGMA schema_version={version + 1}")
        raw.execute("PRAGMA writable_schema=OFF")
    with contextlib.closing(sqlite3.connect(cjk_db, isolation_level=None)) as conn:
        assert drop_retired_cjk_index(conn.cursor())
        assert conn.execute("PRAGMA legacy_alter_table").fetchone()[0] == 0
    assert [name for name in _cjk_residue(cjk_db) if not name.startswith("fts_")] == []



@pytest.mark.platforms("posix")
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root writes read-only files")
@pytest.mark.parametrize("mode", [0o400, 0o000], ids=["read-only", "unopenable"])
def test_a_copy_failing_its_final_scan_is_not_reported_as_damage(cjk_db, monkeypatch, mode):
    """Copy and retirement succeed; the copy then turns read-only or unopenable, so only its own scan fails."""
    import hermes_state_fts

    retire = hermes_state_fts.drop_retired_cjk_index
    frozen = []

    def retire_then_freeze(cursor):
        removed = retire(cursor)
        path = Path(cursor.connection.execute("PRAGMA database_list").fetchone()[2])
        if path != Path(cjk_db):
            for item in path.parent.iterdir():
                item.chmod(mode)
            path.parent.chmod(0o500)
            frozen.append(path.parent)
        return removed

    monkeypatch.setattr(hermes_state_fts, "drop_retired_cjk_index", retire_then_freeze)
    try:
        reason = repair._db_opens_cleanly(cjk_db)
    finally:
        for probe_dir in frozen:
            if probe_dir.exists():
                probe_dir.chmod(0o700)
                for item in probe_dir.iterdir():
                    item.chmod(0o600)
    assert frozen
    assert isinstance(reason, repair.HealthCheckUnavailable), reason



def test_corruption_found_by_the_final_scan_stays_damage(cjk_db, monkeypatch):
    """A SQLITE_CORRUPT raised by the copy's own scan is the store's damage, not a temporary-file failure."""
    probe = repair._db_opens_cleanly

    def corrupt_scan(path):
        if path == cjk_db:
            return probe(path)
        error = sqlite3.DatabaseError("database disk image is malformed")
        error.sqlite_errorcode = 11  # SQLITE_CORRUPT
        raise error

    monkeypatch.setattr(repair, "_db_opens_cleanly", corrupt_scan)
    reason = probe(cjk_db)
    assert reason == "database disk image is malformed"
    assert not isinstance(reason, repair.HealthCheckUnavailable)
