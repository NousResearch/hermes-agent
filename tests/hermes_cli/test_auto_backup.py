"""Tests for scheduled auto-backup and `hermes backup --list` (#12238).

Covers:

  1. maybe_create_auto_backup — disabled-by-default gate, schedule parsing,
     interval gating via last_run_at, archive creation, keep_last pruning,
     failure stamping (no retry-hammering), custom dir override.
  2. list_backup_archives / run_backup_list — kind classification, ordering,
     empty-state output.
"""

from __future__ import annotations

import json
import zipfile
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from hermes_cli import backup as B


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_home(tmp_path: Path) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("model:\n  provider: openrouter\n")
    (home / "skills").mkdir()
    (home / "skills" / "SKILL.md").write_text("# skill\n")
    return home


def _set_cfg(monkeypatch, cfg: dict) -> None:
    monkeypatch.setattr(B, "_get_backup_config", lambda: cfg)


def _state(home: Path) -> dict:
    path = home / "backups" / B._AUTO_STATE_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _write_state(home: Path, state: dict) -> None:
    backups = home / "backups"
    backups.mkdir(exist_ok=True)
    (backups / B._AUTO_STATE_FILE).write_text(json.dumps(state))


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

class TestScheduleParsing:
    def test_named_schedules(self):
        assert B._auto_backup_interval_hours({"schedule": "hourly"}) == 1.0
        assert B._auto_backup_interval_hours({"schedule": "daily"}) == 24.0
        assert B._auto_backup_interval_hours({"schedule": "weekly"}) == 168.0

    def test_default_is_daily(self):
        assert B._auto_backup_interval_hours({}) == 24.0

    def test_numeric_hours(self):
        assert B._auto_backup_interval_hours({"schedule": 6}) == 6.0
        assert B._auto_backup_interval_hours({"schedule": "12"}) == 12.0

    def test_numeric_floor_one_hour(self):
        assert B._auto_backup_interval_hours({"schedule": 0}) == 1.0

    def test_garbage_falls_back_to_daily(self):
        assert B._auto_backup_interval_hours({"schedule": "fortnightly"}) == 24.0
        assert B._auto_backup_interval_hours({"schedule": True}) == 24.0

    def test_enabled_parsing(self):
        assert B._auto_backup_enabled({}) is False
        assert B._auto_backup_enabled({"enabled": True}) is True
        assert B._auto_backup_enabled({"enabled": "true"}) is True
        assert B._auto_backup_enabled({"enabled": "false"}) is False

    def test_keep_floor_is_one(self):
        assert B._auto_backup_keep({"keep_last": 0}) == 1
        assert B._auto_backup_keep({"keep_last": "junk"}) == B._AUTO_DEFAULT_KEEP


# ---------------------------------------------------------------------------
# maybe_create_auto_backup
# ---------------------------------------------------------------------------

class TestMaybeCreateAutoBackup:
    def test_disabled_by_default(self, tmp_path, monkeypatch):
        home = _make_home(tmp_path)
        _set_cfg(monkeypatch, {})
        assert B.maybe_create_auto_backup(hermes_home=home) is None
        assert not (home / "backups").exists()

    def test_first_run_creates_archive(self, tmp_path, monkeypatch):
        home = _make_home(tmp_path)
        _set_cfg(monkeypatch, {"enabled": True, "schedule": "daily"})
        result = B.maybe_create_auto_backup(hermes_home=home)
        assert result is not None
        assert result.name.startswith("auto-")
        assert result.suffix == ".zip"
        assert zipfile.is_zipfile(result)
        with zipfile.ZipFile(result) as zf:
            assert "config.yaml" in zf.namelist()
        state = _state(home)
        assert state["last_status"] == "ok"
        assert state["last_run_at"]

    def test_not_due_returns_none(self, tmp_path, monkeypatch):
        home = _make_home(tmp_path)
        _set_cfg(monkeypatch, {"enabled": True, "schedule": "daily"})
        now = datetime.now(timezone.utc)
        _write_state(home, {"last_run_at": (now - timedelta(hours=2)).isoformat()})
        assert B.maybe_create_auto_backup(hermes_home=home, now=now) is None

    def test_due_after_interval(self, tmp_path, monkeypatch):
        home = _make_home(tmp_path)
        _set_cfg(monkeypatch, {"enabled": True, "schedule": "daily"})
        now = datetime.now(timezone.utc)
        _write_state(home, {"last_run_at": (now - timedelta(hours=25)).isoformat()})
        result = B.maybe_create_auto_backup(hermes_home=home, now=now)
        assert result is not None

    def test_hourly_schedule(self, tmp_path, monkeypatch):
        home = _make_home(tmp_path)
        _set_cfg(monkeypatch, {"enabled": True, "schedule": "hourly"})
        now = datetime.now(timezone.utc)
        _write_state(home, {"last_run_at": (now - timedelta(minutes=61)).isoformat()})
        assert B.maybe_create_auto_backup(hermes_home=home, now=now) is not None

    def test_unparseable_last_run_recovers(self, tmp_path, monkeypatch):
        home = _make_home(tmp_path)
        _set_cfg(monkeypatch, {"enabled": True})
        _write_state(home, {"last_run_at": "not-a-date"})
        assert B.maybe_create_auto_backup(hermes_home=home) is not None

    def test_naive_last_run_treated_as_utc(self, tmp_path, monkeypatch):
        home = _make_home(tmp_path)
        _set_cfg(monkeypatch, {"enabled": True, "schedule": "daily"})
        now = datetime.now(timezone.utc)
        naive = (now - timedelta(hours=2)).replace(tzinfo=None)
        _write_state(home, {"last_run_at": naive.isoformat()})
        assert B.maybe_create_auto_backup(hermes_home=home, now=now) is None

    def test_prunes_beyond_keep_last(self, tmp_path, monkeypatch):
        home = _make_home(tmp_path)
        _set_cfg(monkeypatch, {"enabled": True, "keep_last": 2})
        backups = home / "backups"
        backups.mkdir()
        for i in range(3):
            (backups / f"auto-2026-01-0{i + 1}-000000.zip").write_bytes(b"old")
        # Manual + pre-update archives in the same dir must never be touched.
        (backups / "pre-update-2026-01-01-000000.zip").write_bytes(b"keep")
        (backups / "my-manual.zip").write_bytes(b"keep")

        result = B.maybe_create_auto_backup(hermes_home=home)
        assert result is not None
        autos = sorted(p.name for p in backups.glob("auto-*.zip"))
        assert len(autos) == 2
        assert result.name in autos
        assert (backups / "pre-update-2026-01-01-000000.zip").exists()
        assert (backups / "my-manual.zip").exists()

    def test_failure_stamps_state_no_hammering(self, tmp_path, monkeypatch):
        home = _make_home(tmp_path)
        _set_cfg(monkeypatch, {"enabled": True, "schedule": "daily"})
        monkeypatch.setattr(B, "_write_full_zip_backup", lambda out, root: None)
        now = datetime.now(timezone.utc)
        assert B.maybe_create_auto_backup(hermes_home=home, now=now) is None
        state = _state(home)
        assert state["last_status"] == "failed"
        # The failure stamped last_run_at, so an immediate re-poll is gated.
        monkeypatch.undo()
        _set_cfg(monkeypatch, {"enabled": True, "schedule": "daily"})
        assert B.maybe_create_auto_backup(hermes_home=home, now=now) is None

    def test_custom_dir(self, tmp_path, monkeypatch):
        home = _make_home(tmp_path)
        dest = tmp_path / "external-drive"
        _set_cfg(monkeypatch, {"enabled": True, "dir": str(dest)})
        result = B.maybe_create_auto_backup(hermes_home=home)
        assert result is not None
        assert result.parent == dest

    def test_missing_home_returns_none(self, tmp_path, monkeypatch):
        _set_cfg(monkeypatch, {"enabled": True})
        assert B.maybe_create_auto_backup(hermes_home=tmp_path / "nope") is None

    def test_never_raises_on_config_error(self, tmp_path, monkeypatch):
        """A broken config load degrades to 'disabled', not an exception."""
        import hermes_cli.config as config_mod

        home = _make_home(tmp_path)

        def boom():
            raise RuntimeError("config exploded")

        monkeypatch.setattr(config_mod, "load_config", boom)
        assert B._get_backup_config() == {}
        assert B.maybe_create_auto_backup(hermes_home=home) is None

    def test_archive_restores_with_import_validation(self, tmp_path, monkeypatch):
        """The auto-backup zip passes the same validation `hermes import` uses."""
        home = _make_home(tmp_path)
        _set_cfg(monkeypatch, {"enabled": True})
        result = B.maybe_create_auto_backup(hermes_home=home)
        with zipfile.ZipFile(result) as zf:
            ok, reason = B._validate_backup_zip(zf)
        assert ok, reason


# ---------------------------------------------------------------------------
# list_backup_archives / run_backup_list
# ---------------------------------------------------------------------------

class TestListArchives:
    def test_classifies_kinds(self, tmp_path, monkeypatch):
        home = _make_home(tmp_path)
        _set_cfg(monkeypatch, {})
        backups = home / "backups"
        backups.mkdir()
        (backups / "auto-2026-01-01-000000.zip").write_bytes(b"a")
        (backups / "pre-update-2026-01-02-000000.zip").write_bytes(b"b")
        (backups / "pre-migration-2026-01-03-000000.zip").write_bytes(b"c")
        (backups / "hand-rolled.zip").write_bytes(b"d")
        (backups / "not-a-backup.txt").write_text("ignored")

        archives = B.list_backup_archives(hermes_home=home)
        kinds = {a["path"].name: a["kind"] for a in archives}
        assert kinds == {
            "auto-2026-01-01-000000.zip": "auto",
            "pre-update-2026-01-02-000000.zip": "pre-update",
            "pre-migration-2026-01-03-000000.zip": "pre-migration",
            "hand-rolled.zip": "manual",
        }

    def test_includes_custom_dir(self, tmp_path, monkeypatch):
        home = _make_home(tmp_path)
        dest = tmp_path / "elsewhere"
        dest.mkdir()
        (dest / "auto-2026-01-01-000000.zip").write_bytes(b"a")
        _set_cfg(monkeypatch, {"dir": str(dest)})
        archives = B.list_backup_archives(hermes_home=home)
        assert [a["path"].parent for a in archives] == [dest]

    def test_sorted_newest_first(self, tmp_path, monkeypatch):
        import os
        home = _make_home(tmp_path)
        _set_cfg(monkeypatch, {})
        backups = home / "backups"
        backups.mkdir()
        older = backups / "auto-2026-01-01-000000.zip"
        newer = backups / "auto-2026-01-02-000000.zip"
        older.write_bytes(b"a")
        newer.write_bytes(b"b")
        os.utime(older, (1000000000, 1000000000))
        os.utime(newer, (2000000000, 2000000000))
        archives = B.list_backup_archives(hermes_home=home)
        assert [a["path"].name for a in archives] == [newer.name, older.name]

    def test_run_backup_list_empty(self, tmp_path, monkeypatch, capsys):
        home = _make_home(tmp_path)
        _set_cfg(monkeypatch, {})
        monkeypatch.setattr(B, "get_default_hermes_root", lambda: home)
        B.run_backup_list(Namespace())
        out = capsys.readouterr().out
        assert "No backup archives found" in out

    def test_run_backup_list_output(self, tmp_path, monkeypatch, capsys):
        home = _make_home(tmp_path)
        _set_cfg(monkeypatch, {})
        monkeypatch.setattr(B, "get_default_hermes_root", lambda: home)
        backups = home / "backups"
        backups.mkdir()
        (backups / "auto-2026-01-01-000000.zip").write_bytes(b"x" * 2048)
        B.run_backup_list(Namespace())
        out = capsys.readouterr().out
        assert "auto-2026-01-01-000000.zip" in out
        assert "[auto]" in out
        assert "hermes import" in out
