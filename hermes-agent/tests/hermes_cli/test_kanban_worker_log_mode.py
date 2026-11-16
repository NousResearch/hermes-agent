"""Kanban worker logs are durable and may quote secrets: never world-readable (#63594)."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd

pytestmark = pytest.mark.platforms("posix")


@pytest.fixture
def log_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return kb.worker_log_path("t_mode")


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_new_worker_log_is_not_world_readable(log_path):
    old = os.umask(0o002)
    try:
        with kbd._open_worker_log(SimpleNamespace(id="t_mode"), None) as f:
            f.write(b"x")
    finally:
        os.umask(old)
    assert _mode(log_path) & 0o007 == 0
    assert _mode(log_path) & 0o600 == 0o600


def test_existing_world_readable_log_and_its_backup_are_tightened(log_path, monkeypatch):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_bytes(b"old secret output\n" * 10)
    log_path.chmod(0o664)
    monkeypatch.setattr("hermes_cli.kanban_worker_log.worker_log_rotation_config", lambda: (16, 1))

    with kbd._open_worker_log(SimpleNamespace(id="t_mode"), None) as f:
        f.write(b"x")

    backup = log_path.with_suffix(".log.1")
    assert _mode(backup) == 0o660  # group bits kept for shared-group homes
    assert _mode(log_path) & 0o007 == 0
