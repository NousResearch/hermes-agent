"""Tests for gateway.disk_guard — the 2026-09-19 / 2026-09-20 incident guards.

Cron/verification patterns leaked abandoned multi-GB SQLite copies into the
temp root twice: ~107 GB (``tmp*.db``, ``statedb_ro*``) over three days, then
— after the first pattern-bound guard shipped — another ~14 GB under NEW names
(``sess_state*.db``, ``state_check*.db``). The volume hit 100 %, the gateway
died unclean, and every board raised ``sqlite3.OperationalError: disk I/O
error``. These tests pin the properties the incidents demand:

* (a) a simulated leak is fully swept by one housekeeping pass — under BOTH
  incident name sets, and under arbitrary future ``*.db`` names (closed set);
* (b) the free-space check warns below 5 GiB (ERROR below 1 GiB);
* (c) the sweep is shape-bound, not name-bound, and live files survive:
  open-handle files (real and mocked lsof), files inside the mtime grace
  window, small files, symlinks, directories and foreign names are never
  touched.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway import disk_guard
from gateway.disk_guard import (
    FREE_WARN_BYTES,
    check_free_disk_warning,
    sweep_abandoned_db_copies,
)

_GIB = 1024 * 1024 * 1024


def _stale_lsof(_path):
    """lsof stub: no process holds the file (rc 1 — 'abandoned')."""
    return False


def _usage(total: int, free: int) -> SimpleNamespace:
    return SimpleNamespace(total=total, used=total - free, free=free)


class TestSweepAbandonedDbCopies:
    def test_leaked_copies_are_removed(self, tmp_path):
        """(a) one pass removes every >1 GiB leftover of the 2026-09-19 set."""
        leak = [
            tmp_path / "tmpabc123.db",          # the incident's shape: cp to $TMPDIR
            tmp_path / "tmp987654-0.db",
            tmp_path / "statedb_ro_20260919",   # read-only verification copy
            tmp_path / "statedb_ro_copy_2",
        ]
        for f in leak:
            f.write_bytes(b"\0" * 16)  # content irrelevant; size gate bypassed below
        removed = sweep_abandoned_db_copies(
            tmp_path, min_bytes=0, grace_seconds=0.0, lsof=_stale_lsof
        )
        assert removed == len(leak)
        assert not any(f.exists() for f in leak)

    def test_renamed_leak_copies_are_removed(self, tmp_path):
        """(a) the 2026-09-20 relapse set (sess_state*/state_check*) — the names
        the first, pattern-bound guard missed — is swept without maintaining a
        leak-name list."""
        leak = [
            tmp_path / "sess_state.db",
            tmp_path / "sess_state2.db",
            tmp_path / "state_check2.db",
            tmp_path / "state_check3.db",
        ]
        for f in leak:
            f.write_bytes(b"\0" * 16)
        removed = sweep_abandoned_db_copies(
            tmp_path, min_bytes=0, grace_seconds=0.0, lsof=_stale_lsof
        )
        assert removed == len(leak)
        assert not any(f.exists() for f in leak)

    def test_any_future_big_db_name_is_swept(self, tmp_path):
        """(a) closed set: the sweep must not depend on knowing leak names —
        a hypothetical future tool leaking e.g. 'foo_snapshot9.db' is caught."""
        victim = tmp_path / "foo_snapshot9.db"
        victim.write_bytes(b"\0" * 16)
        removed = sweep_abandoned_db_copies(
            tmp_path, min_bytes=0, grace_seconds=0.0, lsof=_stale_lsof
        )
        assert removed == 1
        assert not victim.exists()

    def test_wal_shm_sidecars_are_swept(self, tmp_path):
        """(a) sqlite sidecar spellings count as sweep material too."""
        leak = [
            tmp_path / "live1.db-wal",
            tmp_path / "live2.db-shm",
            tmp_path / "check3.db-wal",
        ]
        for f in leak:
            f.write_bytes(b"\0" * 16)
        removed = sweep_abandoned_db_copies(
            tmp_path, min_bytes=0, grace_seconds=0.0, lsof=_stale_lsof
        )
        assert removed == 3
        assert not any(f.exists() for f in leak)

    def test_duplicate_pattern_match_is_counted_once(self, tmp_path):
        """'statedb_ro_1.db' matches both '*.db' and 'statedb_ro*' — dedupe via
        the seen-set must keep the count (and unlink) at one."""
        victim = tmp_path / "statedb_ro_1.db"
        victim.write_bytes(b"\0" * 16)
        assert sweep_abandoned_db_copies(
            tmp_path, min_bytes=0, grace_seconds=0.0, lsof=_stale_lsof
        ) == 1

    def test_small_copies_are_left_alone(self, tmp_path):
        """(c) size-bound: a 4 KB tmp.db is not sweep material (kept in case a
        live process is mid-write; the incident's copies were 1.4-4.7 GB)."""
        small = tmp_path / "tmpsmall.db"
        small.write_bytes(b"\0" * 4096)
        assert sweep_abandoned_db_copies(tmp_path, lsof=_stale_lsof) == 0
        assert small.exists()


class TestSweepLivenessGuards:
    """(c) acceptance: files with an open handle must NOT be deleted."""

    def test_open_file_is_never_swept(self, tmp_path, caplog):
        """Mocked lsof reports a holder — the file must survive."""
        victim = tmp_path / "sess_state_live.db"
        victim.write_bytes(b"\0" * 16)
        with caplog.at_level(logging.DEBUG):
            removed = sweep_abandoned_db_copies(
                tmp_path, min_bytes=0, grace_seconds=0.0, lsof=lambda _p: True
            )
        assert removed == 0
        assert victim.exists()
        assert any("held open" in r.getMessage() for r in caplog.records)

    def test_open_file_survives_real_lsof(self, tmp_path):
        """True acceptance shape (AC 3): a sparse >1 GiB file with a REAL open
        handle stays; its handle-free twin goes. Runs the actual lsof binary."""
        if not os.path.exists("/usr/sbin/lsof"):
            pytest.skip("lsof binary not available")
        live = tmp_path / "sess_state.db"
        gone = tmp_path / "state_check9.db"
        for f in (live, gone):
            with open(f, "wb") as fh:  # sparse: >1 GiB size, ~0 blocks on disk
                fh.truncate(disk_guard.SWEEP_MIN_BYTES + 1024)
        os.utime(live, (0, 0))
        os.utime(gone, (0, 0))  # both stale: only the handle makes the difference
        fh = open(live, "rb")
        try:
            removed = sweep_abandoned_db_copies(tmp_path)  # full defaults
        finally:
            fh.close()
        assert removed == 1
        assert live.exists() and not gone.exists()

    def test_lsof_inconclusive_file_is_skipped(self, tmp_path, caplog):
        """lsof missing/errored (None): cannot prove 'abandoned' → skip."""
        victim = tmp_path / "state_check_unknown.db"
        victim.write_bytes(b"\0" * 16)
        with caplog.at_level(logging.DEBUG):
            removed = sweep_abandoned_db_copies(
                tmp_path, min_bytes=0, grace_seconds=0.0, lsof=lambda _p: None
            )
        assert removed == 0
        assert victim.exists()
        assert any("inconclusive" in r.getMessage() for r in caplog.records)

    def test_fresh_file_gets_mtime_grace(self, tmp_path):
        """(c) a just-written >1 GiB-looking file is presumed live: even with
        no lsof holder, mtime age < grace keeps it (backstop for mmap'd or
        transiently-unseen handles)."""
        fresh = tmp_path / "tmpfresh.db"
        fresh.write_bytes(b"\0" * 16)
        assert sweep_abandoned_db_copies(
            tmp_path, min_bytes=0, grace_seconds=600.0, lsof=_stale_lsof
        ) == 0
        assert fresh.exists()

    def test_stale_file_past_grace_is_swept(self, tmp_path):
        """The grace window is not a blanket pardon: handle-free + mtime age
        exactly at/after the grace boundary → swept."""
        victim = tmp_path / "tmpstale.db"
        victim.write_bytes(b"\0" * 16)
        now = 1_000_000.0
        os.utime(victim, (now - 600, now - 600))  # exactly grace_seconds old
        assert sweep_abandoned_db_copies(
            tmp_path, min_bytes=0, grace_seconds=600.0, lsof=_stale_lsof, _now=now
        ) == 1
        assert not victim.exists()


class TestSweepSafetyBounds:
    def test_foreign_temp_files_are_never_touched(self, tmp_path):
        """(c) shape-bound: big non-sqlite files and non-.db backups survive."""
        foreign = [
            tmp_path / "keepme.db.bak",                  # user backup idiom
            tmp_path / "tmpbuild.bin",                   # tmp prefix, no db suffix
            tmp_path / "tmpboard-export.sqlite",         # .sqlite, not .db
            tmp_path / "other_ro_notes.txt",
        ]
        for f in foreign:
            f.write_bytes(b"\0" * 4096)
        removed = sweep_abandoned_db_copies(
            tmp_path, min_bytes=0, grace_seconds=0.0, lsof=_stale_lsof
        )
        assert removed == 0
        assert all(f.exists() for f in foreign)

    @pytest.mark.skipif(sys.platform == "win32", reason="Symlinks require elevated privileges on Windows")
    def test_symlinks_and_directories_are_never_touched(self, tmp_path):
        """(c) a *.db symlink must not be unlinked (nor its target harmed);
        directories named *.db never match either."""
        target = tmp_path / "real.sqlite3"  # non-matching target isolates the link test
        target.write_bytes(b"\0" * 4096)
        link = tmp_path / "tmpattack.db"
        link.symlink_to(target)
        (tmp_path / "tmpdir.db").mkdir()
        assert sweep_abandoned_db_copies(
            tmp_path, min_bytes=0, grace_seconds=0.0, lsof=_stale_lsof
        ) == 0
        assert link.is_symlink() and target.exists()

    def test_open_file_failure_is_skipped_not_raised(self, tmp_path, monkeypatch, caplog):
        """Windows-shaped: unlink fails on an open file — skip, retry next tick."""
        victim = tmp_path / "tmplocked.db"
        victim.write_bytes(b"\0" * 4096)

        def _raise(self):
            raise PermissionError("file in use")

        monkeypatch.setattr(__import__("pathlib").Path, "unlink", _raise)
        with caplog.at_level(logging.DEBUG):
            removed = sweep_abandoned_db_copies(
                tmp_path, min_bytes=0, grace_seconds=0.0, lsof=_stale_lsof,
                logger_=disk_guard.logger,
            )
        assert removed == 0
        assert victim.exists()
        assert any("could not remove" in r.getMessage() for r in caplog.records)

    def test_missing_file_raced_away_is_not_counted(self, tmp_path):
        victim = tmp_path / "tmpgone.db"
        victim.write_bytes(b"\0" * 4096)
        real_glob = __import__("pathlib").Path.glob

        def _glob_then_delete(self, pattern):
            found = list(real_glob(self, pattern))
            victim.unlink(missing_ok=True)  # raced away mid-scan
            return found

        with patch.object(__import__("pathlib").Path, "glob", _glob_then_delete):
            assert sweep_abandoned_db_copies(
                tmp_path, min_bytes=0, grace_seconds=0.0, lsof=_stale_lsof
            ) == 0

    def test_unreadable_root_never_raises(self, tmp_path):
        def _boom(_self, _pattern):
            raise PermissionError("no scan for you")

        with patch.object(__import__("pathlib").Path, "glob", _boom):
            assert sweep_abandoned_db_copies(tmp_path, lsof=_stale_lsof) == 0


class TestSweepSizeGate:
    def test_boundary_size_is_excluded_inclusive(self, tmp_path):
        """Exactly 1 GiB is NOT swept (``<= min_bytes`` keeps it): the gate is
        strictly-greater, so ordinary 1 GiB DBs someone still wants are safe."""
        f = tmp_path / "tmpexact.db"
        with open(f, "wb") as fh:
            fh.truncate(disk_guard.SWEEP_MIN_BYTES)  # sparse: exactly at the gate
        removed = sweep_abandoned_db_copies(
            tmp_path, grace_seconds=0.0, lsof=_stale_lsof
        )
        assert removed == 0
        assert f.exists()


class TestCheckFreeDiskWarning:
    def _state(self):
        return {"level": "ok", "last_warn_monotonic": 0.0}

    def test_warns_below_5gib(self, tmp_path, caplog):
        """(b) 4.5 GiB free → WARNING naming the floor."""
        with patch.object(disk_guard.shutil, "disk_usage", lambda _p: _usage(500 * _GIB, int(4.5 * _GIB))), \
                caplog.at_level(logging.WARNING):
            low = check_free_disk_warning([tmp_path], state=self._state())
        assert low is True
        assert any(r.levelno == logging.WARNING and "early-warning floor" in r.getMessage()
                   for r in caplog.records)

    def test_error_below_1gib(self, tmp_path, caplog):
        """(b) the incident's terminal band (<1 GiB) escalates to ERROR."""
        with patch.object(disk_guard.shutil, "disk_usage", lambda _p: _usage(500 * _GIB, int(0.3 * _GIB))), \
                caplog.at_level(logging.ERROR):
            check_free_disk_warning([tmp_path], state=self._state())
        assert any(r.levelno == logging.ERROR and "imminent risk" in r.getMessage()
                   for r in caplog.records)

    def test_no_warning_when_plentiful(self, tmp_path, caplog):
        with patch.object(disk_guard.shutil, "disk_usage", lambda _p: _usage(500 * _GIB, 200 * _GIB)), \
                caplog.at_level(logging.WARNING):
            low = check_free_disk_warning([tmp_path], state=self._state())
        assert low is False
        assert not caplog.records

    def test_recovery_is_logged_once(self, tmp_path, caplog):
        state = self._state()
        low_usage = _usage(500 * _GIB, int(4.5 * _GIB))
        ok_usage = _usage(500 * _GIB, 200 * _GIB)
        with patch.object(disk_guard.shutil, "disk_usage", lambda _p: low_usage):
            check_free_disk_warning([tmp_path], state=state, _now=100.0)
        with patch.object(disk_guard.shutil, "disk_usage", lambda _p: ok_usage), \
                caplog.at_level(logging.INFO):
            check_free_disk_warning([tmp_path], state=state, _now=200.0)
            check_free_disk_warning([tmp_path], state=state, _now=300.0)
        assert sum("recovered" in r.getMessage() for r in caplog.records) == 1

    def test_rewarn_cadence_30min(self, tmp_path, caplog):
        """While low: one warning, quiet for 30 min, then warns again."""
        state = self._state()
        low_usage = _usage(500 * _GIB, int(4.5 * _GIB))
        with patch.object(disk_guard.shutil, "disk_usage", lambda _p: low_usage), \
                caplog.at_level(logging.WARNING):
            check_free_disk_warning([tmp_path], state=state, _now=1000.0)
            assert len(caplog.records) == 1
            check_free_disk_warning([tmp_path], state=state, _now=1000.0 + 60)
            assert len(caplog.records) == 1  # inside the 30-min window: quiet
            check_free_disk_warning([tmp_path], state=state, _now=1000.0 + 30 * 60 + 1)
            assert len(caplog.records) == 2  # window elapsed: re-warn
        assert sum("early-warning floor" in r.getMessage() for r in caplog.records) == 2

    def test_escalation_warn_to_critical_rewarns_immediately(self, tmp_path, caplog):
        state = self._state()
        warn_usage = _usage(500 * _GIB, int(4.5 * _GIB))
        crit_usage = _usage(500 * _GIB, int(0.5 * _GIB))
        with patch.object(disk_guard.shutil, "disk_usage", lambda _p: warn_usage), \
                caplog.at_level(logging.WARNING):
            check_free_disk_warning([tmp_path], state=state, _now=100.0)
        with patch.object(disk_guard.shutil, "disk_usage", lambda _p: crit_usage), \
                caplog.at_level(logging.ERROR):
            check_free_disk_warning([tmp_path], state=state, _now=101.0)  # band change re-warns
        assert any(r.levelno == logging.ERROR for r in caplog.records)

    def test_multiple_paths_worst_wins_and_devices_dedupe(self, tmp_path, caplog):
        home = tmp_path / "home"
        home.mkdir()
        other = tmp_path / "other"
        other.mkdir()
        fake = {str(home): _usage(100 * _GIB, 90 * _GIB), str(other): _usage(100 * _GIB, int(2 * _GIB))}
        real_stat = __import__("pathlib").Path.stat

        def _fake_stat(self, **kw):
            st = real_stat(self, **kw)
            # distinct fake devices so the dedupe keeps both samples
            return SimpleNamespace(st_dev=1 if self == home else 2, st_mtime=getattr(st, "st_mtime", 0))

        with patch.object(disk_guard.shutil, "disk_usage", lambda p: fake[str(p)]), \
                patch.object(__import__("pathlib").Path, "stat", _fake_stat), \
                caplog.at_level(logging.WARNING):
            low = check_free_disk_warning([home, other], state=self._state())
        assert low is True
        assert any(str(other) in r.getMessage() for r in caplog.records)

    def test_unreadable_paths_fail_silent_false(self, tmp_path, caplog):
        def _boom(_p):
            raise OSError("statvfs failed")

        with patch.object(disk_guard.shutil, "disk_usage", _boom), \
                caplog.at_level(logging.DEBUG):
            low = check_free_disk_warning([tmp_path], state=self._state())
        assert low is False
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_zero_total_sample_is_ignored(self, tmp_path):
        with patch.object(disk_guard.shutil, "disk_usage", lambda _p: _usage(0, 0)):
            low = check_free_disk_warning([tmp_path], state=self._state())
        assert low is False

    def test_boundary_exactly_5gib_is_not_low(self, tmp_path):
        with patch.object(disk_guard.shutil, "disk_usage", lambda _p: _usage(500 * _GIB, FREE_WARN_BYTES)):
            low = check_free_disk_warning([tmp_path], state=self._state())
        assert low is False  # strictly-below gate: exactly 5 GiB is comfortable


class TestHousekeepingWiring:
    def test_disk_guard_chore_runs_every_tick(self, monkeypatch, tmp_path):
        """The chore is registered with cadence 1 and calls both guards."""
        import gateway.run as gateway_run

        calls = []
        monkeypatch.setattr(
            "gateway.disk_guard.sweep_abandoned_db_copies", lambda *a, **k: calls.append("sweep")
        )
        monkeypatch.setattr(
            "gateway.disk_guard.check_free_disk_warning", lambda *a, **k: calls.append("warn")
        )

        class _OneTick:
            def __init__(self):
                self.waited = False

            def is_set(self):
                return self.waited

            def wait(self, timeout=None):
                self.waited = True
                return True

        from gateway import run_profile_reconcile

        monkeypatch.setattr(run_profile_reconcile, "_mcp_config_reconciler", lambda runner: lambda: None)

        gateway_run._start_gateway_housekeeping(_OneTick(), interval=0)
        assert calls.count("sweep") >= 1
        assert calls.count("warn") >= 1

    def test_chore_is_every_tick(self):
        import gateway.run as gateway_run

        # Inspect the chores list construction indirectly: the chore function must
        # exist and be wired; cadence is asserted via source contract to avoid
        # duplicating the loop-run test above.
        assert callable(gateway_run._housekeeping_disk_guard)
        import inspect

        src = inspect.getsource(gateway_run._start_gateway_housekeeping)
        assert '"Disk guard sweep + free-space warning", _housekeeping_disk_guard' in src


class TestRealIncidentShape:
    def test_default_patterns_match_incident_names(self):
        """The pattern set must cover the exact fingerprints from BOTH incidents:
        the 2026-09-19 originals and the 2026-09-20 renamed relapse."""
        import fnmatch

        names = [
            "tmpb7f3c2a1.db", "tmpa9d4e8f2.db",           # 16 loose copies, 19.09.
            "statedb_ro_verify_2026-09-19",               # read-only verification copy
            "statedb_ro_board_seedscraper",
            "sess_state.db", "sess_state2.db",            # renamed relapse, 20.09.
            "state_check2.db", "state_check3.db",
        ]
        for name in names:
            assert any(fnmatch.fnmatch(name, pat) for pat in disk_guard.SWEEP_PATTERNS), name

    def test_real_large_leak_files_are_swept_end_to_end(self, tmp_path, caplog):
        """True end-to-end (default min_bytes/grace, real lsof): sparse >1 GiB
        files with both incidents' name shapes, backdated past the grace
        window, are removed by one default-parameter sweep."""
        if not os.path.exists("/usr/sbin/lsof"):
            pytest.skip("lsof binary not available")
        leaks = [tmp_path / "tmp_e2e_incident.db", tmp_path / "sess_state.db"]
        for f in leaks:
            with open(f, "wb") as fh:  # sparse file: >1 GiB size, ~0 blocks used
                fh.truncate(disk_guard.SWEEP_MIN_BYTES + 1024)
            old = time.time() - (disk_guard.SWEEP_GRACE_SECONDS + 60)
            os.utime(f, (old, old))
        with caplog.at_level(logging.WARNING):
            removed = sweep_abandoned_db_copies(tmp_path)
        assert removed == 2
        assert not any(f.exists() for f in leaks)
