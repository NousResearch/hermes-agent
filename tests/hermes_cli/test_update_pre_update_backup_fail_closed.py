"""Fail-closed pre-update backup: a requested backup that fails aborts the update.

Regression tests for #114592: ``hermes update --yes`` must not proceed when
the pre-update backup fails. Explicit opt-out (``--no-backup`` / config off)
and empty homes (nothing to protect) still proceed.
"""

from types import SimpleNamespace

import pytest

import hermes_cli.update_cmd_maint as _maint


def _args(**overrides):
    base = {"no_backup": False, "backup": False}
    base.update(overrides)
    return SimpleNamespace(**base)


def test_quick_snapshot_raise_aborts(monkeypatch, capsys):
    monkeypatch.setattr(_maint, "_resolve_pre_update_backup_mode", lambda args: "quick")

    def _boom():
        raise OSError("disk full")

    monkeypatch.setattr(_maint, "_run_quick_snapshots", _boom)
    with pytest.raises(SystemExit) as exc:
        _maint._run_pre_update_backup(_args())
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "--no-backup" in out


def test_full_mode_quick_raise_aborts(monkeypatch):
    monkeypatch.setattr(_maint, "_resolve_pre_update_backup_mode", lambda args: "full")

    def _boom():
        raise OSError("snapshot io error")

    monkeypatch.setattr(_maint, "_run_quick_snapshots", _boom)
    with pytest.raises(SystemExit):
        _maint._run_pre_update_backup(_args())


def test_full_mode_zip_failure_aborts(monkeypatch):
    monkeypatch.setattr(_maint, "_resolve_pre_update_backup_mode", lambda args: "full")
    monkeypatch.setattr(_maint, "_run_quick_snapshots", lambda: "snap-123")
    monkeypatch.setattr(_maint, "_run_full_backup", lambda: None)
    with pytest.raises(SystemExit) as exc:
        _maint._run_pre_update_backup(_args())
    assert exc.value.code == 1


def test_full_mode_success_returns_snapshot(monkeypatch):
    from pathlib import Path

    monkeypatch.setattr(_maint, "_resolve_pre_update_backup_mode", lambda args: "full")
    monkeypatch.setattr(_maint, "_run_quick_snapshots", lambda: "snap-123")
    monkeypatch.setattr(_maint, "_run_full_backup", lambda: Path("/tmp/bak.zip"))
    assert _maint._run_pre_update_backup(_args()) == "snap-123"


def test_empty_home_none_proceeds(monkeypatch):
    monkeypatch.setattr(_maint, "_resolve_pre_update_backup_mode", lambda args: "quick")
    monkeypatch.setattr(_maint, "_run_quick_snapshots", lambda: None)
    assert _maint._run_pre_update_backup(_args()) is None


def test_explicit_opt_out_proceeds(monkeypatch):
    monkeypatch.setattr(_maint, "_resolve_pre_update_backup_mode", lambda args: "off")
    assert _maint._run_pre_update_backup(_args(no_backup=True)) is None
