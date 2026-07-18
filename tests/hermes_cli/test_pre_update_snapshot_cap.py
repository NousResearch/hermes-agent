"""Configurable pre-update snapshot size cap (issue #66726)."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path


def test_default_max_is_1gib(monkeypatch, tmp_path):
    from hermes_cli import main as cli_main
    from hermes_cli import config as cfg_mod

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("updates: {}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(cfg_mod, "get_hermes_home", lambda: home)

    assert cli_main._resolve_pre_update_snapshot_max_file_size() == 1 << 30


def test_config_raises_cap_to_4gib(monkeypatch, tmp_path):
    from hermes_cli import main as cli_main
    from hermes_cli import config as cfg_mod

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "updates:\n  pre_update_snapshot_max_mb: 4096\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(cfg_mod, "get_hermes_home", lambda: home)

    assert cli_main._resolve_pre_update_snapshot_max_file_size() == 4096 << 20


def test_zero_disables_cap(monkeypatch, tmp_path):
    from hermes_cli import main as cli_main
    from hermes_cli import config as cfg_mod

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "updates:\n  pre_update_snapshot_max_mb: 0\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(cfg_mod, "get_hermes_home", lambda: home)

    assert cli_main._resolve_pre_update_snapshot_max_file_size() == 0


def test_run_pre_update_backup_passes_raised_cap(monkeypatch, tmp_path):
    from hermes_cli import main as cli_main
    from hermes_cli import config as cfg_mod
    import hermes_cli.backup as backup_mod

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "updates:\n  pre_update_backup: quick\n  pre_update_snapshot_max_mb: 2048\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(cfg_mod, "get_hermes_home", lambda: home)

    seen = {}

    def fake_create_quick_snapshot(**kwargs):
        seen.update(kwargs)
        return "snap-test"

    monkeypatch.setattr(backup_mod, "create_quick_snapshot", fake_create_quick_snapshot)
    # ensure import path used inside _run_pre_update_backup sees the patch
    monkeypatch.setattr(
        "hermes_cli.backup.create_quick_snapshot",
        fake_create_quick_snapshot,
        raising=False,
    )

    # Patch at the import site used by function body
    import sys
    import types

    class _B:
        @staticmethod
        def create_quick_snapshot(**kwargs):
            seen.update(kwargs)
            return "snap-test"

    monkeypatch.setitem(sys.modules, "hermes_cli.backup", _B)

    out = cli_main._run_pre_update_backup(Namespace(no_backup=False, backup=False))
    assert out == "snap-test"
    assert seen.get("max_file_size") == 2048 << 20
    assert seen.get("label") == "pre-update"
