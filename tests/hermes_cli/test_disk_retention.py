"""Tests for hermes_cli.disk_retention — the disk guardian layer (OOF-250 / OOF-269).

Covers:
- truncate_log_tail (in-place tail truncation of diag logs)
- prune_files (age/count/total-size pruning)
- protected-path guards (never touch user data)
- run_retention_sweep (family coverage + exception-proofing)
- sweep_and_log (never raises)
- disk_status (low-space thresholds)
- disk_usage_summary
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli import disk_retention as dr


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A fake HERMES_HOME with the standard layout."""
    h = tmp_path / "hermes"
    for sub in ("logs", "sessions", "memories", "cache/images", "cache/audio",
                "cache/documents", "cache/screenshots"):
        (h / sub).mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


def _write(path: Path, size: int, content: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = content * 79 + b"\n"
    with open(path, "wb") as f:
        while f.tell() < size:
            f.write(line)
    return path


def _age(path: Path, days: float) -> None:
    old = os.path.getmtime(path) - days * 86400
    os.utime(path, (old, old))


# ---------------------------------------------------------------------------
# truncate_log_tail
# ---------------------------------------------------------------------------


class TestTruncateLogTail:
    def test_under_cap_untouched(self, home):
        f = _write(home / "logs" / "boot.log", 1000)
        reclaimed = dr.truncate_log_tail(f, max_bytes=2000, keep_bytes=500, home=home)
        assert reclaimed == 0
        assert f.stat().st_size == pytest.approx(1000, abs=100)

    def test_over_cap_truncates_in_place(self, home):
        f = _write(home / "logs" / "boot.log", 100_000)
        original_size = f.stat().st_size
        reclaimed = dr.truncate_log_tail(f, max_bytes=10_000, keep_bytes=2_000, home=home)
        assert reclaimed > 0
        new_size = f.stat().st_size
        assert new_size < original_size
        assert new_size <= 2_000 + len(dr._TRUNCATION_MARKER)
        # Same inode — in-place truncation, not unlink+recreate (OOF-2:
        # deleted-but-open files don't free space).
        assert f.exists()
        data = f.read_bytes()
        assert data.startswith(dr._TRUNCATION_MARKER)

    def test_kept_tail_starts_at_line_boundary(self, home):
        f = home / "logs" / "boot.log"
        f.write_bytes(b"".join(f"line-{i:06d}\n".encode() for i in range(10_000)))
        dr.truncate_log_tail(f, max_bytes=1_000, keep_bytes=500, home=home)
        body = f.read_bytes()[len(dr._TRUNCATION_MARKER):]
        assert body.startswith(b"line-")

    def test_appender_keeps_working_after_truncation(self, home):
        """An O_APPEND writer must continue appending after truncation."""
        f = _write(home / "logs" / "boot.log", 50_000)
        with open(f, "a", encoding="utf-8") as appender:
            dr.truncate_log_tail(f, max_bytes=10_000, keep_bytes=1_000, home=home)
            appender.write("post-truncation line\n")
        assert b"post-truncation line" in f.read_bytes()

    def test_missing_file_returns_zero(self, home):
        assert dr.truncate_log_tail(
            home / "logs" / "nope.log", max_bytes=10, keep_bytes=5, home=home
        ) == 0

    def test_protected_file_untouched(self, home):
        f = _write(home / "state.db", 100_000)
        assert dr.truncate_log_tail(f, max_bytes=10, keep_bytes=5, home=home) == 0
        assert f.stat().st_size >= 100_000

    def test_file_outside_home_untouched(self, home, tmp_path):
        f = _write(tmp_path / "outside.log", 100_000)
        assert dr.truncate_log_tail(f, max_bytes=10, keep_bytes=5, home=home) == 0
        assert f.stat().st_size >= 100_000


# ---------------------------------------------------------------------------
# prune_files
# ---------------------------------------------------------------------------


class TestPruneFiles:
    def test_age_prune_removes_old_keeps_new(self, home):
        old = _write(home / "cache" / "images" / "old.jpg", 100)
        new = _write(home / "cache" / "images" / "new.jpg", 100)
        _age(old, days=10)
        removed, reclaimed = dr.prune_files(
            (home / "cache" / "images").iterdir(), max_age_days=3, home=home
        )
        assert removed == 1
        assert reclaimed >= 100
        assert not old.exists()
        assert new.exists()

    def test_keep_count_protects_newest_even_when_expired(self, home):
        files = []
        for i in range(6):
            f = _write(home / f"state.db.malformed-{i}", 100)
            _age(f, days=30 + i)  # all expired; file 0 is newest
            files.append(f)
        removed, _ = dr.prune_files(
            home.glob("state.db.malformed-*"),
            keep_count=5, max_age_days=14, home=home,
        )
        assert removed == 1
        assert not files[5].exists()  # oldest removed
        assert all(f.exists() for f in files[:5])

    def test_max_total_bytes_removes_oldest_first(self, home):
        files = []
        for i in range(4):
            f = _write(home / "cache" / "audio" / f"a{i}.ogg", 1000)
            _age(f, days=i)  # a0 newest, a3 oldest
            files.append(f)
        removed, _ = dr.prune_files(
            (home / "cache" / "audio").iterdir(),
            max_total_bytes=2500, home=home,
        )
        assert removed == 2
        assert not files[3].exists() and not files[2].exists()
        assert files[0].exists() and files[1].exists()

    def test_never_prunes_protected_dirs(self, home):
        f = _write(home / "sessions" / "chat.jsonl", 100)
        _age(f, days=999)
        removed, _ = dr.prune_files([f], max_age_days=1, home=home)
        assert removed == 0
        assert f.exists()

    def test_never_prunes_protected_names(self, home):
        f = _write(home / "state.db", 100)
        _age(f, days=999)
        removed, _ = dr.prune_files([f], max_age_days=1, home=home)
        assert removed == 0
        assert f.exists()

    def test_vanished_file_is_skipped(self, home):
        removed, reclaimed = dr.prune_files(
            [home / "cache" / "images" / "ghost.jpg"], max_age_days=0, home=home
        )
        assert (removed, reclaimed) == (0, 0)


# ---------------------------------------------------------------------------
# disk_status
# ---------------------------------------------------------------------------


class _FakeUsage:
    def __init__(self, total, free):
        self.total = total
        self.free = free
        self.used = total - free


class TestDiskStatus:
    def test_healthy_disk(self, home):
        with patch.object(dr.shutil, "disk_usage",
                          return_value=_FakeUsage(1_000_000_000, 500_000_000)):
            status = dr.disk_status(
                home, min_free_bytes=200 * 1024 * 1024, min_free_percent=10.0
            )
        assert status["low_space"] is False
        assert status["free_bytes"] == 500_000_000
        assert status["percent_free"] == 50.0

    def test_low_space_by_bytes(self, home):
        # 15% free but only 150MB — bytes threshold trips.
        with patch.object(dr.shutil, "disk_usage",
                          return_value=_FakeUsage(1_000_000_000, 150_000_000)):
            status = dr.disk_status(
                home, min_free_bytes=200 * 1024 * 1024, min_free_percent=10.0
            )
        assert status["low_space"] is True

    def test_low_space_by_percent(self, home):
        # 500MB free but only 5% — percent threshold trips.
        with patch.object(dr.shutil, "disk_usage",
                          return_value=_FakeUsage(10_000_000_000, 500_000_000)):
            status = dr.disk_status(
                home, min_free_bytes=200 * 1024 * 1024, min_free_percent=10.0
            )
        assert status["low_space"] is True

    def test_thresholds_from_config(self, home):
        cfg = {
            "retention": dict(dr._DEFAULT_DISK_CONFIG["retention"]),
            "low_space": {"min_free_bytes": 1, "min_free_percent": 0.0},
        }
        with patch.object(dr, "get_disk_config", return_value=cfg), \
             patch.object(dr.shutil, "disk_usage",
                          return_value=_FakeUsage(1_000, 999)):
            status = dr.disk_status(home)
        assert status["low_space"] is False
        assert status["min_free_bytes"] == 1


# ---------------------------------------------------------------------------
# disk_usage_summary
# ---------------------------------------------------------------------------


class TestDiskUsageSummary:
    def test_reports_family_sizes(self, home):
        _write(home / "logs" / "agent.log", 5_000)
        _write(home / "sessions" / "x.jsonl", 3_000)
        _write(home / "state.db", 2_000)
        _write(home / "state.db.malformed-backup-20260817_000000", 1_000)
        summary = dr.disk_usage_summary(home)
        assert summary["logs"] >= 5_000
        assert summary["sessions"] >= 3_000
        assert summary["state_db"] >= 2_000
        assert summary["state_db_backups"] >= 1_000

    def test_missing_dirs_are_omitted(self, home):
        summary = dr.disk_usage_summary(home)
        assert "photon" not in summary


# ---------------------------------------------------------------------------
# run_retention_sweep
# ---------------------------------------------------------------------------


def _cfg(**retention_overrides):
    cfg = {
        "retention": dict(dr._DEFAULT_DISK_CONFIG["retention"]),
        "low_space": dict(dr._DEFAULT_DISK_CONFIG["low_space"]),
    }
    cfg["retention"].update(retention_overrides)
    return cfg


class TestRetentionSweep:
    def test_truncates_unrotated_diag_logs(self, home):
        boot = _write(home / "logs" / "container-boot.log", 5_000_000)
        report = dr.run_retention_sweep(home, _cfg())
        assert report["bytes_reclaimed"] > 0
        assert boot.stat().st_size < 5_000_000
        assert report["families"]["diag_logs"]["bytes_reclaimed"] > 0

    def test_skips_rotating_handler_managed_logs(self, home):
        agent = _write(home / "logs" / "agent.log", 5_000_000)
        backup = _write(home / "logs" / "agent.log.1", 5_000_000)
        dr.run_retention_sweep(home, _cfg())
        assert agent.stat().st_size >= 5_000_000
        assert backup.stat().st_size >= 5_000_000

    def test_prunes_db_malformed_backups(self, home):
        # Uses the exact writer contract from hermes_state._backup_db_file:
        # state.db.malformed-backup-<stamp> (+ optional -wal/-shm sidecars).
        files = []
        for i in range(8):
            f = _write(home / f"state.db.malformed-backup-2026081{i}_000000", 100)
            _age(f, days=20 + i)
            files.append(f)
        report = dr.run_retention_sweep(home, _cfg())
        survivors = list(home.glob("state.db.malformed-backup-*"))
        assert len(survivors) == 5  # keep_count default
        assert report["families"]["db_backups"]["files_removed"] == 3

    def test_state_db_itself_never_touched(self, home):
        db = _write(home / "state.db", 50_000_000)
        wal = _write(home / "state.db-wal", 10_000_000)
        dr.run_retention_sweep(home, _cfg())
        assert db.stat().st_size >= 50_000_000
        assert wal.stat().st_size >= 10_000_000

    def test_sessions_never_touched(self, home):
        s = _write(home / "sessions" / "chat.jsonl", 10_000_000)
        _age(s, days=999)
        dr.run_retention_sweep(home, _cfg())
        assert s.exists()

    def test_media_cache_backstop_prunes_old_audio(self, home):
        old = _write(home / "cache" / "audio" / "old.ogg", 1_000)
        new = _write(home / "cache" / "audio" / "new.ogg", 1_000)
        _age(old, days=10)
        report = dr.run_retention_sweep(home, _cfg())
        assert not old.exists()
        assert new.exists()
        assert report["families"]["media_caches"]["files_removed"] == 1

    def test_disabled_config_is_noop(self, home):
        f = _write(home / "logs" / "container-boot.log", 5_000_000)
        report = dr.run_retention_sweep(home, _cfg(enabled=False))
        assert report["enabled"] is False
        assert f.stat().st_size >= 5_000_000

    def test_family_failure_does_not_stop_other_families(self, home):
        old = _write(home / "cache" / "audio" / "old.ogg", 1_000)
        _age(old, days=10)
        with patch.object(dr, "truncate_log_tail", side_effect=RuntimeError("boom")):
            _write(home / "logs" / "container-boot.log", 5_000_000)
            report = dr.run_retention_sweep(home, _cfg())
        # diag_logs family failed…
        assert any("diag_logs" in e for e in report["errors"])
        # …but the media cache family still ran.
        assert not old.exists()

    def test_sweep_never_raises_even_on_config_failure(self, home):
        with patch.object(dr, "get_disk_config", side_effect=RuntimeError("cfg boom")):
            report = dr.run_retention_sweep(home, None)
        assert report["errors"]


# ---------------------------------------------------------------------------
# sweep_and_log
# ---------------------------------------------------------------------------


class TestSweepAndLog:
    def test_logs_one_line_and_returns_report(self, home, caplog):
        _write(home / "logs" / "container-boot.log", 5_000_000)
        with caplog.at_level("INFO", logger=dr.__name__):
            report = dr.sweep_and_log()
        assert report["bytes_reclaimed"] > 0
        lines = [r for r in caplog.records if "Disk retention sweep" in r.getMessage()]
        assert len(lines) == 1
        assert "reclaimed" in lines[0].getMessage()

    def test_never_raises_when_sweep_crashes(self, home):
        with patch.object(dr, "run_retention_sweep", side_effect=RuntimeError("boom")):
            report = dr.sweep_and_log()
        assert report["errors"] == ["boom"]

    def test_never_raises_when_disk_status_crashes(self, home):
        with patch.object(dr, "disk_status", side_effect=OSError("statvfs fail")):
            report = dr.sweep_and_log()
        assert "bytes_reclaimed" in report

    def test_warns_on_low_space(self, home, caplog):
        low = {
            "path": str(home), "total_bytes": 100, "free_bytes": 1,
            "percent_free": 1.0, "min_free_bytes": 50,
            "min_free_percent": 10.0, "low_space": True,
        }
        with patch.object(dr, "disk_status", return_value=low), \
             caplog.at_level("WARNING", logger=dr.__name__):
            dr.sweep_and_log()
        assert any("low_space=True" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Adversarial hardening (independent review round: symlink/hardlink escape,
# backup family contract, config sanitization, unlink accounting)
# ---------------------------------------------------------------------------


class TestTruncationInodeSafety:
    """The hard contract: retention must never mutate protected inodes,
    even when a swept directory contains a link pointing at them."""

    def test_symlink_to_state_db_is_refused(self, home):
        db = home / "state.db"
        db.write_bytes(b"X" * 10_000)
        evil = home / "logs" / "evil.log"
        evil.symlink_to("../state.db")
        got = dr.truncate_log_tail(evil, max_bytes=1_000, keep_bytes=100, home=home)
        assert got == 0
        assert db.stat().st_size == 10_000

    def test_hardlink_to_state_db_is_refused(self, home):
        db = home / "state.db"
        db.write_bytes(b"X" * 10_000)
        evil = home / "logs" / "evil2.log"
        os.link(db, evil)
        got = dr.truncate_log_tail(evil, max_bytes=1_000, keep_bytes=100, home=home)
        assert got == 0
        assert db.stat().st_size == 10_000

    def test_symlink_to_unprotected_file_outside_home_is_refused(self, home, tmp_path):
        target = tmp_path / "other.log"
        target.write_bytes(b"Y" * 10_000)
        evil = home / "logs" / "link.log"
        evil.symlink_to(target)
        got = dr.truncate_log_tail(evil, max_bytes=1_000, keep_bytes=100, home=home)
        assert got == 0
        assert target.stat().st_size == 10_000

    def test_sweep_with_hostile_links_never_touches_state_db(self, home):
        db = home / "state.db"
        db.write_bytes(b"D" * 5_000_000)
        (home / "logs" / "sneaky.log").symlink_to("../state.db")
        try:
            os.link(db, home / "logs" / "sneaky2.log")
        except OSError:
            pass
        report = dr.run_retention_sweep(home=home)
        assert db.stat().st_size == 5_000_000
        assert isinstance(report, dict)

    def test_fifo_is_refused(self, home):
        fifo = home / "logs" / "pipe.log"
        try:
            os.mkfifo(fifo)
        except (AttributeError, OSError):
            pytest.skip("mkfifo unavailable")
        got = dr.truncate_log_tail(fifo, max_bytes=0, keep_bytes=0, home=home)
        assert got == 0


class TestPruneLinkSafety:
    def test_prune_skips_symlinks_and_hardlinks(self, home):
        db = home / "state.db"
        db.write_bytes(b"Z" * 1_000)
        d = home / "cache" / "audio"
        link = d / "old-link.mp3"
        link.symlink_to(db)
        hard = d / "old-hard.mp3"
        os.link(db, hard)
        _age(link, 30)
        _age(hard, 30)
        removed, _ = dr.prune_files(d.iterdir(), max_age_days=1, home=home)
        assert removed == 0
        assert db.exists() and db.stat().st_size == 1_000


class TestDbBackupFamily:
    """prune_db_backup_family honours the hermes_state writer contract:
    exact prefix, base+sidecars as one unit, keep-count in sets."""

    def _mk_set(self, home, stamp: str, *, age_days: float = 0.0):
        base = home / f"state.db.malformed-backup-{stamp}"
        base.write_bytes(b"B" * 100)
        wal = home / (base.name + "-wal")
        wal.write_bytes(b"W" * 50)
        if age_days:
            _age(base, age_days)
            _age(wal, age_days)
        return base, wal

    def test_keeps_newest_sets_with_sidecars(self, home):
        for i in range(8):
            self._mk_set(home, f"2026010{i}_000000", age_days=90)
        removed, reclaimed = dr.prune_db_backup_family(
            home, keep_count=5, max_age_days=14
        )
        bases = sorted(
            p.name for p in home.glob("state.db.malformed-backup-*")
            if not p.name.endswith(("-wal", "-shm"))
        )
        wals = sorted(
            p.name for p in home.glob("state.db.malformed-backup-*-wal")
        )
        assert len(bases) == 5
        assert len(wals) == 5
        for w in wals:
            assert w[: -len("-wal")] in bases, f"orphaned sidecar {w}"
        assert removed == 6  # 3 bases + 3 wals
        assert reclaimed == 3 * 150

    def test_prefix_neighbours_never_match(self, home):
        for name in ("state.db.repair-attempts.json", "state.db-wal",
                     "state.db.corrupt.abc.bak"):
            f = home / name
            f.write_bytes(b"N")
            _age(f, 365)
        self._mk_set(home, "20260101_000000", age_days=365)
        dr.prune_db_backup_family(home, keep_count=0, max_age_days=1)
        assert (home / "state.db.repair-attempts.json").exists()
        assert (home / "state.db-wal").exists()
        assert (home / "state.db.corrupt.abc.bak").exists()

    def test_orphaned_sidecar_pruned_once_expired(self, home):
        orphan = home / "state.db.malformed-backup-20260101_000000-wal"
        orphan.write_bytes(b"W" * 10)
        _age(orphan, 365)
        removed, _ = dr.prune_db_backup_family(home, keep_count=5, max_age_days=14)
        assert removed == 1
        assert not orphan.exists()

    def test_fresh_sets_kept_even_beyond_keep_count(self, home):
        for i in range(8):
            self._mk_set(home, f"2026010{i}_000000")  # fresh mtime
        removed, _ = dr.prune_db_backup_family(home, keep_count=5, max_age_days=14)
        assert removed == 0


class TestConfigSanitization:
    def _with_user_config(self, monkeypatch, disk_section):
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda: {"disk": disk_section},
        )

    def test_malformed_values_fall_back_to_defaults(self, home, monkeypatch):
        self._with_user_config(monkeypatch, {
            "retention": {
                "diag_log_max_bytes": "x",
                "diag_log_keep_bytes": None,
                "cache_max_age_hours": {},
                "db_backup_keep_count": float("nan"),
                "sweep_interval_minutes": -5,
            },
            "low_space": {"min_free_bytes": "lots", "min_free_percent": [1]},
        })
        cfg = dr.get_disk_config()
        d = dr._DEFAULT_DISK_CONFIG
        assert cfg["retention"]["diag_log_max_bytes"] == d["retention"]["diag_log_max_bytes"]
        assert cfg["retention"]["diag_log_keep_bytes"] == d["retention"]["diag_log_keep_bytes"]
        assert cfg["retention"]["cache_max_age_hours"] == d["retention"]["cache_max_age_hours"]
        assert cfg["retention"]["db_backup_keep_count"] == d["retention"]["db_backup_keep_count"]
        assert cfg["retention"]["sweep_interval_minutes"] == d["retention"]["sweep_interval_minutes"]
        assert cfg["low_space"]["min_free_bytes"] == d["low_space"]["min_free_bytes"]
        assert cfg["low_space"]["min_free_percent"] == d["low_space"]["min_free_percent"]

    def test_negative_sizes_never_weaponize_truncation(self, home, monkeypatch):
        self._with_user_config(monkeypatch, {
            "retention": {"diag_log_max_bytes": -1, "diag_log_keep_bytes": -1},
        })
        cfg = dr.get_disk_config()
        assert cfg["retention"]["diag_log_max_bytes"] >= 4096
        assert cfg["retention"]["diag_log_keep_bytes"] >= 0

    def test_string_false_disables(self, home, monkeypatch):
        self._with_user_config(monkeypatch, {"retention": {"enabled": "false"}})
        assert dr.get_disk_config()["retention"]["enabled"] is False

    def test_string_true_enables(self, home, monkeypatch):
        self._with_user_config(monkeypatch, {"retention": {"enabled": "true"}})
        assert dr.get_disk_config()["retention"]["enabled"] is True

    def test_unknown_string_uses_default(self, home, monkeypatch):
        self._with_user_config(monkeypatch, {"retention": {"enabled": "maybe"}})
        assert dr.get_disk_config()["retention"]["enabled"] is True


class TestUnlinkAccounting:
    def test_failed_unlink_does_not_over_delete_newer_files(self, home):
        d = home / "cache" / "audio"
        files = []
        for i in range(5):
            f = d / f"f{i}.mp3"
            f.write_bytes(b"A" * 100)
            _age(f, 5 - i)  # f0 oldest
            files.append(f)

        real_unlink = dr._unlink
        blocked = files[0]

        def flaky_unlink(path):
            if path == blocked:
                return False
            return real_unlink(path)

        with patch.object(dr, "_unlink", side_effect=flaky_unlink):
            removed, _ = dr.prune_files(
                d.iterdir(), max_total_bytes=350, home=home
            )
        # Budget needs 150 bytes freed => oldest two candidates (f0 blocked,
        # f1 removed). f2..f4 must survive — no compensation deletes.
        assert files[1].exists() is False
        assert all(f.exists() for f in (files[2], files[3], files[4]))
        assert removed == 1
