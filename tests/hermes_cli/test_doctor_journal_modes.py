"""Tests for doctor's per-database journal-mode report.

`hermes doctor` lists each Hermes-managed database with its journal mode and
flags databases that are in WAL while the linked SQLite carries the WAL-reset
bug (https://sqlite.org/wal.html#walresetbug). The probe reads the file header
only — it never opens the database through the SQLite engine, because even a
read-only engine open creates -wal/-shm sidecar files next to a WAL database.
"""

import os
import re
import sqlite3

import pytest

import hermes_cli.doctor as doctor
from hermes_cli.sqlite_safe_read import (
    connect_tracked,
    has_live_connection,
    track_connection,
    untrack_connection,
)

VULNERABLE = (3, 50, 4)
FIXED_VERSIONS = [(3, 51, 3), (3, 52, 0), (3, 50, 7), (3, 44, 6)]

EXPOSED_TEXT = "exposed to the WAL-reset bug"


def _make_db(path, journal_mode=None):
    conn = sqlite3.connect(path)
    try:
        if journal_mode:
            conn.execute(f"PRAGMA journal_mode={journal_mode}")
        conn.execute("CREATE TABLE t (x INTEGER)")
        conn.commit()
    finally:
        conn.close()


def _sidecars(directory):
    return sorted(
        p.name for p in directory.iterdir() if p.name.endswith(("-wal", "-shm"))
    )


@pytest.fixture
def clean_registry():
    """Isolate a test from the module-level connection registry.

    Clears on both sides, not just teardown: a test that leaks a tracked
    connection (an earlier failure, or a test that does not take this
    fixture) would otherwise leave the registry dirty and make the *next*
    test's refusal assertion pass for the wrong reason.
    """
    import hermes_cli.sqlite_safe_read as mod

    def _clear():
        with mod._live_lock:
            mod._live_connections.clear()

    _clear()
    try:
        yield
    finally:
        _clear()


class TestReadJournalMode:
    def test_reads_wal(self, tmp_path):
        db = tmp_path / "state.db"
        _make_db(db, journal_mode="WAL")

        mode, error = doctor._read_journal_mode(db)

        assert mode == "wal"
        assert error is None

    def test_reads_rollback(self, tmp_path):
        db = tmp_path / "state.db"
        _make_db(db)

        mode, error = doctor._read_journal_mode(db)

        assert mode == "rollback"
        assert error is None

    def test_probe_creates_no_wal_sidecars(self, tmp_path):
        db = tmp_path / "state.db"
        _make_db(db, journal_mode="WAL")
        assert _sidecars(tmp_path) == []

        assert doctor._read_journal_mode(db) == ("wal", None)

        assert _sidecars(tmp_path) == []

    def test_missing_file_reports_error_and_does_not_create_it(self, tmp_path):
        db = tmp_path / "missing.db"

        mode, error = doctor._read_journal_mode(db)

        assert mode is None
        assert error
        assert not db.exists()

    def test_empty_file_reports_error(self, tmp_path):
        db = tmp_path / "state.db"
        db.touch()

        mode, error = doctor._read_journal_mode(db)

        assert mode is None
        assert error == "file is empty"

    def test_short_file_reports_error(self, tmp_path):
        db = tmp_path / "state.db"
        db.write_bytes(b"SQLite f")

        mode, error = doctor._read_journal_mode(db)

        assert mode is None
        assert "not a database" in error

    def test_corrupt_file_reports_error(self, tmp_path):
        db = tmp_path / "state.db"
        db.write_bytes(b"this is not a sqlite database" * 4)

        mode, error = doctor._read_journal_mode(db)

        assert mode is None
        assert "not a database" in error

    def test_locked_database_is_still_readable(self, tmp_path):
        db = tmp_path / "state.db"
        _make_db(db)
        holder = sqlite3.connect(db, isolation_level=None)
        try:
            holder.execute("BEGIN EXCLUSIVE")

            assert doctor._read_journal_mode(db) == ("rollback", None)
        finally:
            holder.close()

    @pytest.mark.skipif(os.name == "nt", reason="chmod is a no-op on Windows")
    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
    def test_read_only_directory_is_still_readable(self, tmp_path):
        db = tmp_path / "state.db"
        _make_db(db, journal_mode="WAL")
        os.chmod(tmp_path, 0o555)
        try:
            assert doctor._read_journal_mode(db) == ("wal", None)
        finally:
            os.chmod(tmp_path, 0o755)
        assert _sidecars(tmp_path) == []

    def test_does_not_mutate_database_files(self, tmp_path):
        wal_db = tmp_path / "wal.db"
        rollback_db = tmp_path / "plain.db"
        _make_db(wal_db, journal_mode="WAL")
        _make_db(rollback_db)
        wal_bytes = wal_db.read_bytes()
        rollback_bytes = rollback_db.read_bytes()

        assert doctor._read_journal_mode(wal_db) == ("wal", None)
        assert doctor._read_journal_mode(rollback_db) == ("rollback", None)

        assert wal_db.read_bytes() == wal_bytes
        assert rollback_db.read_bytes() == rollback_bytes
        assert _sidecars(tmp_path) == []


class TestLiveConnectionSafety:
    """The probe must not raw-open a database this process has connections to.

    close() on any descriptor cancels every POSIX advisory lock the process
    holds on that file, so a byte-probe run while a connection is live drops
    that connection's locks — including the EXCLUSIVE lock a VACUUM holds
    mid-rewrite. run_doctor is reachable in-process (the dashboard console
    imports and calls it directly while holding live SessionDB connections),
    so the probe must defer to the registry rather than open the file.
    """

    def test_probe_is_refused_while_a_tracked_connection_is_live(
        self, tmp_path, clean_registry
    ):
        db = tmp_path / "state.db"
        _make_db(db, journal_mode="WAL")

        track_connection(db)
        try:
            assert has_live_connection(db)

            mode, error = doctor._read_journal_mode(db)

            assert mode is None
            assert error == "database is open in this process"
        finally:
            untrack_connection(db)

    def test_probe_is_refused_for_a_real_tracked_connection(
        self, tmp_path, clean_registry
    ):
        """The same, through connect_tracked — the path SessionDB actually takes."""
        db = tmp_path / "state.db"
        _make_db(db, journal_mode="WAL")

        conn = connect_tracked(db)
        try:
            assert has_live_connection(db)

            mode, error = doctor._read_journal_mode(db)

            assert mode is None
            assert error == "database is open in this process"
        finally:
            conn.close()

    def test_probe_resumes_once_the_connection_closes(self, tmp_path, clean_registry):
        db = tmp_path / "state.db"
        _make_db(db, journal_mode="WAL")

        conn = connect_tracked(db)
        assert doctor._read_journal_mode(db)[0] is None
        conn.close()

        assert not has_live_connection(db)
        assert doctor._read_journal_mode(db) == ("wal", None)

    def test_refusal_creates_no_new_sidecars(self, tmp_path, clean_registry):
        db = tmp_path / "state.db"
        _make_db(db, journal_mode="WAL")

        conn = connect_tracked(db)
        try:
            before = _sidecars(tmp_path)

            doctor._read_journal_mode(db)

            assert _sidecars(tmp_path) == before
        finally:
            conn.close()

    def test_report_degrades_instead_of_probing_a_live_database(
        self, tmp_path, capsys, clean_registry
    ):
        db = tmp_path / "state.db"
        _make_db(db, journal_mode="WAL")

        conn = connect_tracked(db)
        try:
            doctor._report_database_journal_modes(tmp_path, VULNERABLE)
        finally:
            conn.close()

        out = capsys.readouterr().out
        assert "state.db: journal mode could not be read" in out
        assert "database is open in this process" in out
        assert "cannot rule out WAL exposure" in out

    def test_an_untracked_lock_holder_does_not_block_the_probe(self, tmp_path):
        """Only this process's *registered* connections gate the read.

        A plain sqlite3.connect elsewhere is not in the registry, and a lock
        held by another process is irrelevant — neither can be cancelled by a
        close() we never perform. Guards against over-correcting into refusing
        every read.
        """
        db = tmp_path / "state.db"
        _make_db(db)
        holder = sqlite3.connect(db, isolation_level=None)
        try:
            holder.execute("BEGIN EXCLUSIVE")

            assert doctor._read_journal_mode(db) == ("rollback", None)
        finally:
            holder.close()


class TestUnreadableReason:
    def test_missing_file_keeps_the_os_error_text(self, tmp_path):
        reason = doctor._unreadable_reason(tmp_path / "gone.db")

        assert "No such file or directory" in reason

    @pytest.mark.skipif(os.name == "nt", reason="chmod is a no-op on Windows")
    @pytest.mark.skipif(
        # os.geteuid is POSIX-only, and a skipif condition is evaluated at
        # collection time — calling it unguarded would raise AttributeError
        # and take the whole module down on Windows.
        hasattr(os, "geteuid") and os.geteuid() == 0,
        reason="root ignores file permissions",
    )
    def test_unreadable_file_is_reported_as_permission_denied(self, tmp_path):
        db = tmp_path / "state.db"
        _make_db(db)
        os.chmod(db, 0o000)
        try:
            mode, error = doctor._read_journal_mode(db)
        finally:
            os.chmod(db, 0o644)

        assert mode is None
        assert "permission denied" in error.lower()

    def test_reason_does_not_open_the_file(self, tmp_path, monkeypatch):
        """_unreadable_reason must answer from metadata only.

        It runs on database paths, so taking a descriptor would reintroduce
        the very close() this module's guard exists to prevent.
        """
        db = tmp_path / "state.db"
        _make_db(db)

        def _fail(*args, **kwargs):
            raise AssertionError("_unreadable_reason must not open the file")

        monkeypatch.setattr("builtins.open", _fail)

        assert doctor._unreadable_reason(db) == "file could not be read"


class TestReportDatabaseJournalModes:
    def test_vulnerable_runtime_wal_db_is_exposed(self, tmp_path, capsys):
        _make_db(tmp_path / "state.db", journal_mode="WAL")

        doctor._report_database_journal_modes(tmp_path, VULNERABLE)

        out = capsys.readouterr().out
        assert "state.db is in WAL mode" in out
        assert EXPOSED_TEXT in out

    def test_vulnerable_runtime_rollback_db_is_listed_not_exposed(self, tmp_path, capsys):
        _make_db(tmp_path / "state.db")

        doctor._report_database_journal_modes(tmp_path, VULNERABLE)

        out = capsys.readouterr().out
        assert "state.db: rollback journal mode" in out
        assert EXPOSED_TEXT not in out

    @pytest.mark.parametrize("version", FIXED_VERSIONS)
    def test_fixed_runtime_wal_db_is_not_exposed(self, tmp_path, capsys, version):
        _make_db(tmp_path / "state.db", journal_mode="WAL")

        doctor._report_database_journal_modes(tmp_path, version)

        out = capsys.readouterr().out
        assert "state.db: WAL journal mode" in out
        assert EXPOSED_TEXT not in out
        assert "⚠" not in out

    def test_lists_every_managed_database(self, tmp_path, capsys):
        _make_db(tmp_path / "state.db", journal_mode="WAL")
        _make_db(tmp_path / "projects.db")
        _make_db(tmp_path / "kanban.db")
        board = tmp_path / "kanban" / "boards" / "myboard"
        board.mkdir(parents=True)
        _make_db(board / "kanban.db", journal_mode="WAL")

        doctor._report_database_journal_modes(tmp_path, VULNERABLE)

        out = capsys.readouterr().out
        assert "state.db is in WAL mode" in out
        assert "projects.db: rollback journal mode" in out
        assert "kanban.db: rollback journal mode" in out
        assert "kanban/boards/myboard/kanban.db is in WAL mode" in out

    def test_missing_databases_are_skipped(self, tmp_path, capsys):
        doctor._report_database_journal_modes(tmp_path, VULNERABLE)

        out = capsys.readouterr().out
        assert "state.db" not in out
        assert EXPOSED_TEXT not in out

    def test_locked_database_does_not_crash_or_block(self, tmp_path, capsys):
        db = tmp_path / "state.db"
        _make_db(db)
        holder = sqlite3.connect(db, isolation_level=None)
        try:
            holder.execute("BEGIN EXCLUSIVE")

            doctor._report_database_journal_modes(tmp_path, VULNERABLE)
        finally:
            holder.close()

        out = capsys.readouterr().out
        assert "state.db: rollback journal mode" in out

    @pytest.mark.skipif(os.name == "nt", reason="chmod is a no-op on Windows")
    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
    def test_unreadable_database_does_not_crash(self, tmp_path, capsys):
        db = tmp_path / "state.db"
        _make_db(db)
        os.chmod(db, 0o000)
        try:
            doctor._report_database_journal_modes(tmp_path, VULNERABLE)
        finally:
            os.chmod(db, 0o644)

        out = capsys.readouterr().out
        assert "state.db: journal mode could not be read" in out
        assert "cannot rule out WAL exposure" in out

    def test_corrupt_database_does_not_crash(self, tmp_path, capsys):
        (tmp_path / "state.db").write_bytes(b"garbage bytes, not sqlite" * 8)

        doctor._report_database_journal_modes(tmp_path, VULNERABLE)

        out = capsys.readouterr().out
        assert "state.db: journal mode could not be read" in out

    def test_read_error_is_informational_on_fixed_runtime(self, tmp_path, capsys):
        (tmp_path / "state.db").write_bytes(b"garbage bytes, not sqlite" * 8)

        doctor._report_database_journal_modes(tmp_path, (3, 51, 3))

        out = capsys.readouterr().out
        assert "state.db: journal mode could not be read" in out
        assert "cannot rule out WAL exposure" not in out
        assert "⚠" not in out

    def test_report_creates_no_wal_sidecars(self, tmp_path, capsys):
        db = tmp_path / "state.db"
        _make_db(db, journal_mode="WAL")
        db_bytes = db.read_bytes()

        doctor._report_database_journal_modes(tmp_path, VULNERABLE)

        assert _sidecars(tmp_path) == []
        assert db.read_bytes() == db_bytes


class TestSizeAndRepairHint:
    def test_exposed_databases_report_size_and_repair_hint(self, tmp_path, capsys):
        db = tmp_path / "state.db"
        _make_db(db, journal_mode="WAL")
        doctor._report_database_journal_modes(tmp_path, VULNERABLE)
        out = capsys.readouterr().out
        # _format_size picks the unit (a fresh test DB is KB-scale).
        assert re.search(r"\(\d[\d.]* [KMGT]?B\)", out)
        assert "To clear the exposure:" in out

    def test_no_repair_hint_when_nothing_is_exposed(self, tmp_path, capsys):
        _make_db(tmp_path / "state.db", journal_mode="DELETE")
        doctor._report_database_journal_modes(tmp_path, VULNERABLE)
        assert "To clear the exposure:" not in capsys.readouterr().out

    def test_no_repair_hint_on_a_fixed_runtime(self, tmp_path, capsys):
        _make_db(tmp_path / "state.db", journal_mode="WAL")
        doctor._report_database_journal_modes(tmp_path, FIXED_VERSIONS[0])
        assert "To clear the exposure:" not in capsys.readouterr().out

    def test_size_failure_does_not_crash(self, tmp_path, capsys):
        assert doctor._format_db_size(tmp_path / "gone.db") == "size unknown"
