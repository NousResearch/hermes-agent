"""The node-bootstrap spawns must use ``_find_bash``, never a bare ``"bash"`` argv: on Windows
``CreateProcess`` resolves the bare name to System32's WSL launcher (#116818)."""

from __future__ import annotations

import subprocess

import pytest

import hermes_constants
from hermes_cli import main_tui_launch
from tools.environments import local


def _drive_node_bootstrap():
    return hermes_constants._run_node_bootstrap("ensure_node", timeout=5)


def _drive_tui_node():
    return main_tui_launch._ensure_tui_node()


@pytest.mark.parametrize("drive", [_drive_node_bootstrap, _drive_tui_node], ids=["node_bootstrap", "tui_node"])
def test_node_bootstrap_spawns_resolved_bash(monkeypatch, tmp_path, drive):
    fixture_bash = str(tmp_path / "git" / "bin" / "bash.exe")
    monkeypatch.setattr(local, "_find_bash", lambda: fixture_bash)
    monkeypatch.setattr(main_tui_launch.shutil, "which", lambda name: None)  # force the bootstrap path
    monkeypatch.delenv("HERMES_SKIP_NODE_BOOTSTRAP", raising=False)
    monkeypatch.setattr(hermes_constants, "_NODE_BOOTSTRAP_SCRIPT", tmp_path / "node-bootstrap.sh")
    (tmp_path / "node-bootstrap.sh").write_text("", encoding="utf-8")
    argvs = []

    def fake_run(argv, **kwargs):
        argvs.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(main_tui_launch.subprocess, "run", fake_run)

    drive()

    assert [argv[0] for argv in argvs] == [fixture_bash]
