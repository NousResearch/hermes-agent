from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_cli.dev_sync import DevSyncError, SubprocessRunner, _sync_venv, run, SyncReport


class FakeRunner(SubprocessRunner):
    def __init__(self, *, code: int = 0, watch_code: int = 0):
        self.code = code
        self.watch_code = watch_code
        self.commands = []
        self.watched = []

    def run(self, cmd, *, cwd=None, env=None):
        self.commands.append((cmd, cwd, env))
        return subprocess.CompletedProcess(cmd, self.code, stdout="", stderr="boom" if self.code else "")

    def watch(self, commands):
        self.watched = commands
        return self.watch_code


def test_venv_creation_failure_is_fatal(tmp_path):
    runner = FakeRunner(code=7)
    with patch("hermes_cli.managed_uv.ensure_uv", return_value="uv"):
        with pytest.raises(DevSyncError, match="venv creation failed: boom"):
            _sync_venv(tmp_path, SyncReport(), runner)


def test_watch_runs_selected_frontend_processes_after_sync(tmp_path):
    for package in ("ui-tui", "web"):
        directory = tmp_path / package
        directory.mkdir()
        (directory / "package.json").write_text("{}")
        if package == "ui-tui":
            (directory / "dist").mkdir()
            (directory / "dist" / "entry.js").write_text("built")
    manifest = tmp_path / "hermes_cli" / "web_dist" / ".vite"
    manifest.mkdir(parents=True)
    (manifest / "manifest.json").write_text("{}")
    stamps = tmp_path / ".hermes-dev"
    stamps.mkdir()

    runner = FakeRunner()
    with (
        patch("hermes_cli.dev_sync._tui_stamp") as tui_stamp,
        patch("hermes_cli.dev_sync._web_stamp") as web_stamp,
        patch("hermes_constants.find_node_executable", return_value="npm"),
    ):
        tui_stamp.return_value.needs_build.return_value = False
        web_stamp.return_value.needs_build.return_value = False
        run(tmp_path, only=["tui", "web"], watch=True, runner=runner)

    assert runner.watched == [
        (["npm", "run", "dev"], tmp_path / "ui-tui"),
        (["npm", "run", "dev"], tmp_path / "web"),
    ]


def test_watch_child_failure_is_fatal(tmp_path):
    runner = FakeRunner(watch_code=9)
    with (
        patch("hermes_cli.dev_sync._build_tui", side_effect=lambda root, report, runner: report),
        patch("hermes_constants.find_node_executable", return_value="npm"),
    ):
        with pytest.raises(DevSyncError, match="watch process exited with code 9"):
            run(tmp_path, only=["tui"], watch=True, runner=runner)
