"""Tests for the top-level `./hermes` launcher script."""

import os
import runpy
import subprocess
import sys
import types
from pathlib import Path

import pytest


def test_launcher_delegates_to_argparse_entrypoint(monkeypatch):
    """`./hermes` should use `hermes_cli.main`, not the legacy Fire wrapper."""
    launcher_path = Path(__file__).resolve().parents[2] / "hermes"
    called = []

    fake_main_module = types.ModuleType("hermes_cli.main")

    def fake_main():
        called.append("hermes_cli.main")
        return 7

    fake_main_module.main = fake_main
    monkeypatch.setitem(sys.modules, "hermes_cli.main", fake_main_module)

    fake_cli_module = types.ModuleType("cli")

    def legacy_cli_main(*args, **kwargs):
        raise AssertionError("launcher should not import cli.main")

    fake_cli_module.main = legacy_cli_main
    monkeypatch.setitem(sys.modules, "cli", fake_cli_module)

    fake_fire_module = types.ModuleType("fire")

    def legacy_fire(*args, **kwargs):
        raise AssertionError("launcher should not invoke fire.Fire")

    fake_fire_module.Fire = legacy_fire
    monkeypatch.setitem(sys.modules, "fire", fake_fire_module)

    monkeypatch.setattr(sys, "argv", [str(launcher_path), "gateway", "status"])

    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(launcher_path), run_name="__main__")

    assert called == ["hermes_cli.main"]
    assert exc.value.code == 7


def test_launcher_propagates_cron_failure(tmp_path):
    """The tracked launcher must preserve failures returned by delegated commands."""
    repo_root = Path(__file__).resolve().parents[2]
    launcher_path = repo_root / "hermes"
    env = os.environ.copy()
    env["HERMES_HOME"] = str(tmp_path)

    result = subprocess.run(
        [sys.executable, str(launcher_path), "cron", "pause", "definitely-missing-job"],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )

    assert result.returncode == 1
    assert "Job with ID or name 'definitely-missing-job' not found" in result.stdout
