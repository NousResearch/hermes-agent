"""Configurable pre-update snapshot size cap (issue #66726)."""

from __future__ import annotations

from argparse import Namespace
import sys
import types


def _install_backup_stub(monkeypatch, seen, snap_id="snap-test"):
    """Stub hermes_cli.backup import site used by the quick-snapshot runner."""

    def create_quick_snapshot(**kwargs):
        seen.update(kwargs)
        return snap_id

    def verify_sqlite_integrity(*_a, **_k):
        return {"valid": True}

    stub = types.ModuleType("hermes_cli.backup")
    stub.create_quick_snapshot = create_quick_snapshot
    stub.create_pre_update_snapshots_all_profiles = lambda **_k: {}
    stub.verify_sqlite_integrity = verify_sqlite_integrity
    stub._quick_snapshot_root = lambda *_a, **_k: None
    monkeypatch.setitem(sys.modules, "hermes_cli.backup", stub)


def _write_updates(home, body: str, monkeypatch):
    from hermes_cli import config as cfg_mod

    home.mkdir()
    (home / "config.yaml").write_text(body, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(cfg_mod, "get_hermes_home", lambda: home)


def test_default_max_is_1gib(monkeypatch, tmp_path):
    from hermes_cli.update_cmd_maint import _resolve_pre_update_snapshot_max_file_size

    _write_updates(tmp_path / ".hermes", "updates: {}\n", monkeypatch)
    assert _resolve_pre_update_snapshot_max_file_size() == 1 << 30


def test_config_raises_cap_to_4gib(monkeypatch, tmp_path):
    from hermes_cli.update_cmd_maint import _resolve_pre_update_snapshot_max_file_size

    _write_updates(
        tmp_path / ".hermes",
        "updates:\n  pre_update_snapshot_max_mb: 4096\n",
        monkeypatch,
    )
    assert _resolve_pre_update_snapshot_max_file_size() == 4096 << 20


def test_boolean_config_keeps_default_cap(monkeypatch, tmp_path):
    from hermes_cli.update_cmd_maint import _resolve_pre_update_snapshot_max_file_size

    _write_updates(
        tmp_path / ".hermes",
        "updates:\n  pre_update_snapshot_max_mb: true\n",
        monkeypatch,
    )
    assert _resolve_pre_update_snapshot_max_file_size() == 1 << 30


def test_zero_disables_cap(monkeypatch, tmp_path):
    from hermes_cli.update_cmd_maint import _resolve_pre_update_snapshot_max_file_size

    _write_updates(
        tmp_path / ".hermes",
        "updates:\n  pre_update_snapshot_max_mb: 0\n",
        monkeypatch,
    )
    assert _resolve_pre_update_snapshot_max_file_size() == 0


def test_run_pre_update_backup_passes_raised_cap(monkeypatch, tmp_path):
    from hermes_cli.update_cmd_maint import _run_pre_update_backup

    _write_updates(
        tmp_path / ".hermes",
        "updates:\n  pre_update_backup: quick\n  pre_update_snapshot_max_mb: 2048\n",
        monkeypatch,
    )
    seen = {}
    _install_backup_stub(monkeypatch, seen, snap_id="snap-test")

    out = _run_pre_update_backup(Namespace(no_backup=False, backup=False))
    assert out == "snap-test"
    assert seen.get("max_file_size") == 2048 << 20
    assert seen.get("label") == "pre-update"


def test_run_pre_update_backup_zero_passes_none_cap(monkeypatch, tmp_path):
    """Config ``0`` reaches create_quick_snapshot as max_file_size=None, not 0."""
    from hermes_cli.update_cmd_maint import _run_pre_update_backup

    _write_updates(
        tmp_path / ".hermes",
        "updates:\n  pre_update_backup: quick\n  pre_update_snapshot_max_mb: 0\n",
        monkeypatch,
    )
    seen = {}
    _install_backup_stub(monkeypatch, seen, snap_id="snap-none")

    out = _run_pre_update_backup(Namespace(no_backup=False, backup=False))
    assert out == "snap-none"
    assert "max_file_size" in seen
    assert seen["max_file_size"] is None
    assert seen.get("label") == "pre-update"


def test_raised_copy_cap_does_not_scale_integrity_max_bytes(monkeypatch, tmp_path):
    """Raising the copy cap must not pass that size as verify_sqlite_integrity max_bytes."""
    from hermes_cli.update_cmd_maint import (
        _PRE_UPDATE_SNAPSHOT_MAX_FILE_SIZE,
        _run_pre_update_backup,
    )

    _write_updates(
        tmp_path / ".hermes",
        "updates:\n  pre_update_backup: quick\n  pre_update_snapshot_max_mb: 4096\n",
        monkeypatch,
    )
    seen_copy = {}
    seen_verify = {}

    def create_quick_snapshot(**kwargs):
        seen_copy.update(kwargs)
        return "snap-integrity"

    def verify_sqlite_integrity(*_a, **kwargs):
        seen_verify.update(kwargs)
        return {"valid": True}

    stub = types.ModuleType("hermes_cli.backup")
    stub.create_quick_snapshot = create_quick_snapshot
    stub.create_pre_update_snapshots_all_profiles = lambda **_k: {}
    stub.verify_sqlite_integrity = verify_sqlite_integrity
    stub._quick_snapshot_root = lambda *_a, **_k: tmp_path
    monkeypatch.setitem(sys.modules, "hermes_cli.backup", stub)

    home = tmp_path / ".hermes"
    (home / "state.db").write_bytes(b"x")

    out = _run_pre_update_backup(Namespace(no_backup=False, backup=False))
    assert out == "snap-integrity"
    assert seen_copy.get("max_file_size") == 4096 << 20
    assert seen_verify.get("max_bytes") == _PRE_UPDATE_SNAPSHOT_MAX_FILE_SIZE
