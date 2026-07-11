"""Tests for the top-level `./hermes` launcher script."""

import runpy
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

    # The launcher wraps the call in ``raise SystemExit(main())``; a ``None``
    # return exits 0, which runpy surfaces as SystemExit.
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(launcher_path), run_name="__main__")

    assert not excinfo.value.code
    assert called == ["hermes_cli.main"]


def test_launcher_propagates_nonzero_exit_status(monkeypatch):
    """A handler status returned by ``main()`` must become the process exit code.

    Guards the completeness gap in #62810: the launcher used to end with a bare
    ``main()`` call, discarding the integer status so ``./hermes`` exited 0 even
    when the installed ``hermes`` / ``python -m hermes_cli.main`` exited nonzero.
    """
    launcher_path = Path(__file__).resolve().parents[2] / "hermes"

    fake_main_module = types.ModuleType("hermes_cli.main")

    def fake_main():
        return 2

    fake_main_module.main = fake_main
    monkeypatch.setitem(sys.modules, "hermes_cli.main", fake_main_module)
    monkeypatch.setattr(sys, "argv", [str(launcher_path), "checkpoints", "clear"])

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(launcher_path), run_name="__main__")

    assert excinfo.value.code == 2
