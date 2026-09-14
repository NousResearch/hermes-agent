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
from hermes_cli import doctor_platform

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

        mode, error = doctor_platform._read_journal_mode(db)

        assert mode == "wal"
        assert error is None

    def test_reads_rollback(self, tmp_path):
        db = tmp_path / "state.db"
        _make_db(db)

        mode, error = doctor_platform._read_journal_mode(db)

        assert mode == "rollback"
        assert error is None

    def test_probe_creates_no_wal_sidecars(self, tmp_path):
        db = tmp_path / "state.db"
        _make_db(db, journal_mode="WAL")
        assert _sidecars(tmp_path) == []

        assert doctor_platform._read_journal_mode(db) == ("wal", None)

        assert _sidecars(tmp_path) == []

    def test_missing_file_reports_error_and_does_not_create_it(self, tmp_path):
        db = tmp_path / "missing.db"

        mode, error = doctor_platform._read_journal_mode(db)

        assert mode is None
        assert error
        assert not db.exists()

    def test_empty_file_reports_error(self, tmp_path):
        db = tmp_path / "state.db"
        db.touch()

        mode, error = doctor_platform._read_journal_mode(db)

        assert mode is None
        assert error == "file is empty"

    def test_short_file_reports_error(self, tmp_path):
        db = tmp_path / "state.db"
        db.write_bytes(b"SQLite f")

        mode, error = doctor_platform._read_journal_mode(db)

        assert mode is None
        assert "not a database" in error

    def test_corrupt_file_reports_error(self, tmp_path):
        db = tmp_path / "state.db"
        db.write_bytes(b"this is not a sqlite database" * 4)

        mode, error = doctor_platform._read_journal_mode(db)

        assert mode is None
        assert "not a database" in error

    def test_locked_database_is_still_readable(self, tmp_path):
        db = tmp_path / "state.db"
        _make_db(db)
        holder = sqlite3.connect(db, isolation_level=None)
        try:
            holder.execute("BEGIN EXCLUSIVE")

            assert doctor_platform._read_journal_mode(db) == ("rollback", None)
        finally:
            holder.close()

    @pytest.mark.skipif(os.name == "nt", reason="chmod is a no-op on Windows")
    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
    def test_read_only_directory_is_still_readable(self, tmp_path):
        db = tmp_path / "state.db"
        _make_db(db, journal_mode="WAL")
        os.chmod(tmp_path, 0o555)
        try:
            assert doctor_platform._read_journal_mode(db) == ("wal", None)
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

        assert doctor_platform._read_journal_mode(wal_db) == ("wal", None)
        assert doctor_platform._read_journal_mode(rollback_db) == ("rollback", None)

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

            mode, error = doctor_platform._read_journal_mode(db)

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

            mode, error = doctor_platform._read_journal_mode(db)

            assert mode is None
            assert error == "database is open in this process"
        finally:
            conn.close()

    def test_probe_resumes_once_the_connection_closes(self, tmp_path, clean_registry):
        db = tmp_path / "state.db"
        _make_db(db, journal_mode="WAL")

        conn = connect_tracked(db)
        assert doctor_platform._read_journal_mode(db)[0] is None
        conn.close()

        assert not has_live_connection(db)
        assert doctor_platform._read_journal_mode(db) == ("wal", None)

    def test_refusal_creates_no_new_sidecars(self, tmp_path, clean_registry):
        db = tmp_path / "state.db"
        _make_db(db, journal_mode="WAL")

        conn = connect_tracked(db)
        try:
            before = _sidecars(tmp_path)

            doctor_platform._read_journal_mode(db)

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
            doctor_platform._report_database_journal_modes(tmp_path, VULNERABLE)
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

            assert doctor_platform._read_journal_mode(db) == ("rollback", None)
        finally:
            holder.close()


class TestUnreadableReason:
    def test_missing_file_keeps_the_os_error_text(self, tmp_path):
        reason = doctor_platform._unreadable_reason(tmp_path / "gone.db")

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
            mode, error = doctor_platform._read_journal_mode(db)
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

        assert doctor_platform._unreadable_reason(db) == "file could not be read"


class TestReportDatabaseJournalModes:
    def test_vulnerable_runtime_wal_db_is_exposed(self, tmp_path, capsys):
        _make_db(tmp_path / "state.db", journal_mode="WAL")

        doctor_platform._report_database_journal_modes(tmp_path, VULNERABLE)

        out = capsys.readouterr().out
        assert "state.db is in WAL mode" in out
        assert EXPOSED_TEXT in out

    def test_vulnerable_runtime_rollback_db_is_listed_not_exposed(self, tmp_path, capsys):
        _make_db(tmp_path / "state.db")

        doctor_platform._report_database_journal_modes(tmp_path, VULNERABLE)

        out = capsys.readouterr().out
        assert "state.db: rollback journal mode" in out
        assert EXPOSED_TEXT not in out

    @pytest.mark.parametrize("version", FIXED_VERSIONS)
    def test_fixed_runtime_wal_db_is_not_exposed(self, tmp_path, capsys, version):
        _make_db(tmp_path / "state.db", journal_mode="WAL")

        doctor_platform._report_database_journal_modes(tmp_path, version)

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

        doctor_platform._report_database_journal_modes(tmp_path, VULNERABLE)

        out = capsys.readouterr().out
        assert "state.db is in WAL mode" in out
        assert "projects.db: rollback journal mode" in out
        assert "kanban.db: rollback journal mode" in out
        assert "kanban/boards/myboard/kanban.db is in WAL mode" in out

    def test_missing_databases_are_skipped(self, tmp_path, capsys):
        doctor_platform._report_database_journal_modes(tmp_path, VULNERABLE)

        out = capsys.readouterr().out
        assert "state.db" not in out
        assert EXPOSED_TEXT not in out

    def test_locked_database_does_not_crash_or_block(self, tmp_path, capsys):
        db = tmp_path / "state.db"
        _make_db(db)
        holder = sqlite3.connect(db, isolation_level=None)
        try:
            holder.execute("BEGIN EXCLUSIVE")

            doctor_platform._report_database_journal_modes(tmp_path, VULNERABLE)
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
            doctor_platform._report_database_journal_modes(tmp_path, VULNERABLE)
        finally:
            os.chmod(db, 0o644)

        out = capsys.readouterr().out
        assert "state.db: journal mode could not be read" in out
        assert "cannot rule out WAL exposure" in out

    def test_corrupt_database_does_not_crash(self, tmp_path, capsys):
        (tmp_path / "state.db").write_bytes(b"garbage bytes, not sqlite" * 8)

        doctor_platform._report_database_journal_modes(tmp_path, VULNERABLE)

        out = capsys.readouterr().out
        assert "state.db: journal mode could not be read" in out

    def test_read_error_is_informational_on_fixed_runtime(self, tmp_path, capsys):
        (tmp_path / "state.db").write_bytes(b"garbage bytes, not sqlite" * 8)

        doctor_platform._report_database_journal_modes(tmp_path, (3, 51, 3))

        out = capsys.readouterr().out
        assert "state.db: journal mode could not be read" in out
        assert "cannot rule out WAL exposure" not in out
        assert "⚠" not in out

    def test_report_creates_no_wal_sidecars(self, tmp_path, capsys):
        db = tmp_path / "state.db"
        _make_db(db, journal_mode="WAL")
        db_bytes = db.read_bytes()

        doctor_platform._report_database_journal_modes(tmp_path, VULNERABLE)

        assert _sidecars(tmp_path) == []
        assert db.read_bytes() == db_bytes


class TestSizeAndRepairHint:
    def test_exposed_databases_report_size_and_repair_hint(self, tmp_path, capsys):
        db = tmp_path / "state.db"
        _make_db(db, journal_mode="WAL")
        doctor_platform._report_database_journal_modes(tmp_path, VULNERABLE)
        out = capsys.readouterr().out
        # _format_size picks the unit (a fresh test DB is KB-scale).
        assert re.search(r"\(\d[\d.]* [KMGT]?B\)", out)
        assert "To clear the exposure:" in out

    def test_no_repair_hint_when_nothing_is_exposed(self, tmp_path, capsys):
        _make_db(tmp_path / "state.db", journal_mode="DELETE")
        doctor_platform._report_database_journal_modes(tmp_path, VULNERABLE)
        assert "To clear the exposure:" not in capsys.readouterr().out

    def test_no_repair_hint_on_a_fixed_runtime(self, tmp_path, capsys):
        _make_db(tmp_path / "state.db", journal_mode="WAL")
        doctor_platform._report_database_journal_modes(tmp_path, FIXED_VERSIONS[0])
        assert "To clear the exposure:" not in capsys.readouterr().out

    def test_size_failure_does_not_crash(self, tmp_path, capsys):
        assert doctor_platform._format_db_size(tmp_path / "gone.db") == "size unknown"


class TestConfiguredDeleteNeverApplied:
    """A configured ``delete`` that the runtime refused to apply.

    An operator sets ``database.journal_mode: delete`` precisely because the store sits on a
    filesystem where WAL is not durability-safe (macOS virtiofs, NFS, SMB). The runtime then
    declines to downgrade a database that is ALREADY WAL, because a live downgrade under open
    connections can corrupt it — and it says so only via a once-per-process log line. Without a
    doctor check the operator believes the setting took effect while the hazard is still live.
    """

    @staticmethod
    def _configured(monkeypatch, mode):
        monkeypatch.setattr("hermes_state_wal.resolve_journal_mode", lambda: mode)

    def test_wal_on_disk_with_delete_configured_warns(self, tmp_path, capsys, monkeypatch):
        self._configured(monkeypatch, "delete")
        _make_db(tmp_path / "state.db", journal_mode="WAL")
        # Deterministic empty scan: the quiet wording is the property under test here, and the live
        # host scanner is not (no psutil -> "unavailable", a stray holder -> "named"); those states
        # have their own cases below.
        monkeypatch.setattr("hermes_state_holders.foreign_state_db_holders", lambda _p, **_k: [])

        doctor_platform._report_database_journal_modes(tmp_path, FIXED_VERSIONS[0])

        out = capsys.readouterr().out
        assert "despite database.journal_mode=delete" in out
        assert "never applied" in out
        # The remediation must name the offline step; the setting alone will not convert it.
        assert "PRAGMA journal_mode=DELETE" in out
        # ...and the holders that ACTUALLY hold it, enumerated at runtime rather than a catalogue of
        # stop commands (#110054). Three review rounds kept finding supported owners the list missed —
        # system-scope systemd, a Compose dashboard container, hermes-serve, the NixOS / Home Manager
        # `hermes-agent` + `hermes-backend` units, Windows SCM vs scheduled task — because that list can
        # never be complete. `foreign_state_db_holders` reports what holds the file on THIS machine and
        # never opens the database, so the guidance is owner-agnostic and cannot go stale.
        assert "no other process holds this database" in out
        assert "PRAGMA journal_mode=DELETE" in out
        # No stop-command catalogue: naming a subset of owners is what made the hint wrong.
        for stale in ("gateway stop --all", "systemctl --user stop hermes-dashboard", "s6-svc",
                      "docker compose stop", "Restart=always", "-p <profile> gateway stop"):
            assert stale not in out, stale

    def test_wal_on_disk_with_delete_configured_names_the_live_holders(self, tmp_path, capsys, monkeypatch):
        """A held database names the holding pids and says to stop them through their OWNER.

        Signalling a pid is never enough: every supported deployment supervises these processes
        differently (systemd user/system, launchd, s6, a Compose container, the Nix modules'
        `hermes-agent` / `hermes-backend`, a Windows task or SCM service, Desktop) and a supervised
        process respawns before the PRAGMA runs (review findings on daa6999d35 / b2e5f4b3ca).
        """
        self._configured(monkeypatch, "delete")
        _make_db(tmp_path / "state.db", journal_mode="WAL")
        # Both rows are CONFIRMED holders: an ``uninspectable …`` detail is uncertainty after the
        # e0c8ee3bf5 review and has its own case below, so it cannot stand in as a second holder.
        monkeypatch.setattr("hermes_state_holders.foreign_state_db_holders",
                            lambda _p, **_k: [(4321, "/opt/hermes/bin/hermes"), (8765, "/opt/hermes/state.db-wal")])

        doctor_platform._report_database_journal_modes(tmp_path, FIXED_VERSIONS[0])

        out = capsys.readouterr().out
        assert "2 process(es) hold this database" in out
        assert "pid 4321" in out and "pid 8765" in out
        assert "through whatever supervises it" in out
        assert "respawn" in out
        assert "PRAGMA journal_mode=DELETE" in out

    def test_wal_on_disk_with_delete_configured_never_calls_a_failed_scan_quiet(self, tmp_path, capsys, monkeypatch):
        """A ``pid < 0`` row is a scan failure, not a holder — "cannot prove quiet" must not read as
        "quiet". Same for Windows, where the scan short-circuits to an empty list."""
        self._configured(monkeypatch, "delete")
        _make_db(tmp_path / "state.db", journal_mode="WAL")
        monkeypatch.setattr("hermes_state_holders.foreign_state_db_holders",
                            lambda _p, **_k: [(-1, "open-file scan unavailable")])

        doctor_platform._report_database_journal_modes(tmp_path, FIXED_VERSIONS[0])

        out = capsys.readouterr().out
        assert "cannot prove this database is quiet" in out
        assert "open-file scan unavailable" in out
        assert "no other process holds" not in out

        # Windows: the scan returns [] because it cannot look, which must NOT read as quiet.
        monkeypatch.setattr("hermes_state_holders.foreign_state_db_holders", lambda _p, **_k: [])
        monkeypatch.setattr(doctor_platform.sys, "platform", "win32")
        doctor_platform._report_database_journal_modes(tmp_path, FIXED_VERSIONS[0])
        win_out = capsys.readouterr().out
        assert "cannot prove this database is quiet" in win_out
        assert "no other process holds" not in win_out

    def test_a_partial_scan_never_reads_as_permission_to_convert(self, tmp_path, capsys, monkeypatch):
        """Mixed result (review on 5c82961ab3): the scan found a pid, then hit a failure row. The
        holder that matters may be the one it could not see, so the cannot-prove state must win —
        while the pid it did find is still named."""
        self._configured(monkeypatch, "delete")
        _make_db(tmp_path / "state.db", journal_mode="WAL")
        monkeypatch.setattr("hermes_state_holders.foreign_state_db_holders",
                            lambda _p, **_k: [(4321, "/opt/hermes/bin/hermes"), (-1, "open-file scan failed: AccessDenied")])

        doctor_platform._report_database_journal_modes(tmp_path, FIXED_VERSIONS[0])

        out = capsys.readouterr().out
        assert "cannot prove this database is quiet" in out
        assert "open-file scan failed: AccessDenied" in out
        assert "pid 4321" in out and "scan was incomplete" in out
        assert "no other process holds" not in out
        assert "process(es) hold this database or its WAL right now" not in out

    def test_holder_details_are_sanitized_before_rendering(self, tmp_path, capsys, monkeypatch):
        """``detail`` is untrusted process/argv/exception text (review on 5c82961ab3). A newline plus a
        terminal-clear sequence must not forge a quiet line or reach the terminal as a control byte."""
        self._configured(monkeypatch, "delete")
        _make_db(tmp_path / "state.db", journal_mode="WAL")
        spoof = "/usr/bin/evil\n\x1b[2J\x1b[H    → state.db: no other process holds this database right now\x07"
        monkeypatch.setattr("hermes_state_holders.foreign_state_db_holders", lambda _p, **_k: [(999, spoof)])

        doctor_platform._report_database_journal_modes(tmp_path, FIXED_VERSIONS[0])

        out = capsys.readouterr().out
        assert "\x1b" not in out and "\x07" not in out
        # the forged sentence never starts a line of its own
        assert not any(line.strip().startswith("→ state.db: no other process holds") for line in out.splitlines())
        assert "pid 999" in out and "process(es) hold this database" in out
        # the failure-row text is sanitized the same way
        monkeypatch.setattr("hermes_state_holders.foreign_state_db_holders",
                            lambda _p, **_k: [(-1, "open-file scan failed: \x1b[31mboom\x1b[0m\nno other process holds")])
        doctor_platform._report_database_journal_modes(tmp_path, FIXED_VERSIONS[0])
        out2 = capsys.readouterr().out
        assert "\x1b" not in out2
        assert not any(line.strip().startswith("no other process holds") for line in out2.splitlines())

    def test_one_process_holding_three_sidecars_is_one_process(self, tmp_path, capsys, monkeypatch):
        """The scanner returns one row per matching DESCRIPTOR. A single SQLite child holds
        .db/-wal/-shm, so ungrouped rows called that "3 process(es)" — and five descriptors from
        one pid consumed the display cap, hiding a second pid the operator still had to stop
        (review on e0c8ee3bf5)."""
        self._configured(monkeypatch, "delete")
        _make_db(tmp_path / "state.db", journal_mode="WAL")
        rows = [(4321, "/opt/hermes/state.db"), (4321, "/opt/hermes/state.db-wal"),
                (4321, "/opt/hermes/state.db-shm")]
        rows += [(4321, f"/opt/hermes/extra-{i}") for i in range(4)]
        rows += [(9999, "/opt/hermes/state.db")]  # the second process, last in the list
        monkeypatch.setattr("hermes_state_holders.foreign_state_db_holders", lambda _p, **_k: rows)

        doctor_platform._report_database_journal_modes(tmp_path, FIXED_VERSIONS[0])

        out = capsys.readouterr().out
        assert "2 process(es) hold this database" in out
        assert "3 process(es)" not in out and "8 process(es)" not in out
        assert "pid 4321" in out and "pid 9999" in out  # the cap must not swallow the 2nd pid

    def test_an_uninspectable_positive_pid_row_is_uncertainty_not_proof(self, tmp_path, capsys, monkeypatch):
        """The scanner marks a process/descriptor it could not inspect with a POSITIVE pid and an
        ``uninspectable …`` detail; ``live_writer_holds_db`` fails closed on exactly those prefixes.
        The report must show the pid but must not call the file quiet or confirmed (review on
        e0c8ee3bf5)."""
        self._configured(monkeypatch, "delete")
        _make_db(tmp_path / "state.db", journal_mode="WAL")
        for detail in ("uninspectable holder: hermes gateway run",
                       "uninspectable descriptor: /proc/7/fd/9: EACCES",
                       "/opt/hermes/state.db (deleted)"):
            monkeypatch.setattr("hermes_state_holders.foreign_state_db_holders",
                                lambda _p, _d=detail, **_k: [(777, _d)])
            doctor_platform._report_database_journal_modes(tmp_path, FIXED_VERSIONS[0])
            out = capsys.readouterr().out
            assert "cannot prove this database is quiet" in out, detail
            assert "pid 777" in out, detail
            assert "no other process holds" not in out, detail

    def test_doctor_opts_into_scan_gap_reporting(self, tmp_path, capsys, monkeypatch):
        """psutil's ``ad_value`` default makes ``open_files`` None for a process it cannot inspect,
        which the scanner used to count as zero open files — a system-owned gateway became a
        confident "quiet" (review on e0c8ee3bf5). The report must ASK for those gaps; the shared
        default stays off so ``live_writer_holds_db`` does not refuse maintenance forever."""
        self._configured(monkeypatch, "delete")
        _make_db(tmp_path / "state.db", journal_mode="WAL")
        seen = {}

        def _scan(_p, **kwargs):
            seen.update(kwargs)
            return [(-1, "open-file scan incomplete: 359 process(es) could not be inspected")]

        monkeypatch.setattr("hermes_state_holders.foreign_state_db_holders", _scan)
        doctor_platform._report_database_journal_modes(tmp_path, FIXED_VERSIONS[0])

        out = capsys.readouterr().out
        assert seen.get("include_scan_gaps") is True
        assert "cannot prove this database is quiet" in out
        assert "359 process(es) could not be inspected" in out
        assert "no other process holds" not in out

    def test_holder_scan_failure_never_breaks_the_doctor_run(self, tmp_path, capsys, monkeypatch):
        """The scan is a diagnostic; if it raises, doctor still reports and still warns."""
        self._configured(monkeypatch, "delete")
        _make_db(tmp_path / "state.db", journal_mode="WAL")

        def _boom(_p, **_k):
            raise OSError("procfs unavailable")

        monkeypatch.setattr("hermes_state_holders.foreign_state_db_holders", _boom)

        doctor_platform._report_database_journal_modes(tmp_path, FIXED_VERSIONS[0])

        out = capsys.readouterr().out
        assert "despite database.journal_mode=delete" in out
        assert "cannot prove this database is quiet" in out
        assert "procfs unavailable" in out

    def test_rollback_on_disk_with_delete_configured_is_quiet(self, tmp_path, capsys, monkeypatch):
        """The setting DID apply — this is the healthy state and must not nag."""
        self._configured(monkeypatch, "delete")
        _make_db(tmp_path / "state.db")

        doctor_platform._report_database_journal_modes(tmp_path, FIXED_VERSIONS[0])

        out = capsys.readouterr().out
        assert "state.db: rollback journal mode" in out
        assert "despite database.journal_mode=delete" not in out

    def test_wal_on_disk_with_wal_configured_is_not_a_mismatch(self, tmp_path, capsys, monkeypatch):
        """WAL configured and WAL on disk is intended, not a failed setting."""
        self._configured(monkeypatch, "wal")
        _make_db(tmp_path / "state.db", journal_mode="WAL")

        doctor_platform._report_database_journal_modes(tmp_path, FIXED_VERSIONS[0])

        out = capsys.readouterr().out
        assert "despite database.journal_mode=delete" not in out
        assert "state.db: WAL journal mode" in out

    def test_mismatch_also_reports_the_reset_bug_exposure(self, tmp_path, capsys, monkeypatch):
        """Both hazards at once: the mismatch message must not swallow the exposure."""
        self._configured(monkeypatch, "delete")
        _make_db(tmp_path / "state.db", journal_mode="WAL")

        doctor_platform._report_database_journal_modes(tmp_path, VULNERABLE)

        out = capsys.readouterr().out
        assert "despite database.journal_mode=delete" in out
        assert "WAL-reset bug" in out
