"""Tests for the top-level `./hermes` launcher script."""

import runpy
import sys
import types
from pathlib import Path


def test_launcher_delegates_to_argparse_entrypoint(monkeypatch):
    """`./hermes` should use `hermes_cli.main`, not the legacy Fire wrapper."""
    launcher_path = Path(__file__).resolve().parents[2] / "hermes"
    called = []

    fake_main_module = types.ModuleType("hermes_cli.main")

    def fake_main():
        called.append("hermes_cli.main")

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

    runpy.run_path(str(launcher_path), run_name="__main__")

    assert called == ["hermes_cli.main"]


def test_installation_command_refuses_unpublished_launcher(monkeypatch, tmp_path):
    """PM workspace copies resolve a store Python without ever publishing
    .hermes/bin — the returned command must stay executable everywhere
    instead of embedding a missing path (issue #125043)."""
    from hermes_cli import _launchers

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(_launchers, "resolve_store_python", lambda root: Path("/usr/bin/python3"))

    command = _launchers.installation_command(workspace, ["gateway", "run"])
    assert str(workspace / ".hermes" / "bin" / "hermes") not in command
    assert command[0] == "/usr/bin/python3" and command[1:3] == ["-I", "-c"]
    assert command[-2:] == ["gateway", "run"]

    # A published, executable launcher keeps the exact-install path.
    local = workspace / ".hermes" / "bin"
    local.mkdir(parents=True)
    shim = local / "hermes"
    shim.write_text("#!/bin/sh\n", encoding="utf-8")
    shim.chmod(0o755)
    assert _launchers.installation_command(workspace, ["gateway", "run"])[0] == str(shim)
    assert _launchers.installation_command(
        workspace, module="gateway.cgroup_cleanup")[1] == "--run-module"

    # A non-executable launcher is as unusable as a missing one.
    shim.chmod(0o644)
    assert _launchers.installation_command(workspace, ["gateway", "run"])[0] != str(shim)
