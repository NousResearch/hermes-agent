"""`hermes import` and the built-in memory store's lock files."""

import zipfile
from argparse import Namespace
from pathlib import Path

import pytest

import hermes_cli.gateway as gateway_mod
import hermes_cli.gateway_setup_service as service_setup


@pytest.fixture(autouse=True)
def _no_real_gateway_service(monkeypatch):
    """run_import() may start a gateway service after a restore; never touch the host's."""
    monkeypatch.setattr(service_setup, "ensure_gateway_service", lambda **kw: False)
    monkeypatch.setattr(gateway_mod, "_is_service_running", lambda: False)


def test_import_keeps_the_live_memory_lock_inode_and_restores_the_memory_file(tmp_path, monkeypatch):
    """An older archive carrying ``memories/MEMORY.md.lock`` ahead of ``MEMORY.md`` must not swap
    the live lock inode: a save holding the old inode would no longer exclude the restore."""
    hermes_home = tmp_path / ".hermes"
    (hermes_home / "memories").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    live_lock = hermes_home / "memories" / "MEMORY.md.lock"
    live_lock.write_text("", encoding="utf-8")
    inode = live_lock.stat().st_ino

    zip_path = tmp_path / "backup.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("memories/MEMORY.md.lock", "")
        zf.writestr("memories/MEMORY.md", "restored entry")
        zf.writestr("config.yaml", "model: test\n")

    from hermes_cli.backup import run_import
    run_import(Namespace(zipfile=str(zip_path), force=True))

    assert live_lock.stat().st_ino == inode
    assert (hermes_home / "memories" / "MEMORY.md").read_text(encoding="utf-8-sig") == "restored entry"
