"""Which argv shapes skip the pre-parse startup prelude (``hermes_cli.main_startup``).

The prelude sweeps quarantined exes and stale bytecode, recovers gateways a killed
update left paused, bootstraps the install and prints the PM verdict. None of that
can change what ``hermes --help`` or a read-only command prints, so those shapes
must not pay for it — while every real dispatch still must.
"""

from __future__ import annotations

import sys

import pytest

from hermes_cli import main_startup
from hermes_cli.main_startup import run_startup_prelude, skips_startup_prelude


@pytest.mark.parametrize("argv", [
    ["--help"],
    ["-h"],
    ["chat", "--help"],
    ["gateway", "install", "--help"],
    ["--version"],
    ["-V"],
    ["-p", "coder", "--version"],
    ["config"],
    ["config", "get", "model"],
    ["config", "show"],
    ["config", "path"],
    ["config", "env-path"],
    ["kanban", "list"],
    ["kanban", "ls", "--status", "ready"],
    ["kanban", "show", "t_abc123"],
    ["kanban", "boards"],
])
def test_read_only_shapes_skip_the_prelude(argv):
    assert skips_startup_prelude(argv), f"{argv} cannot be affected by the prelude"


@pytest.mark.parametrize("argv", [
    [],
    ["chat"],
    ["-z", "hello"],
    ["--tui"],
    ["update"],
    ["gateway", "run"],
    ["gateway", "status"],
    ["kanban", "create", "--title", "x"],
    ["kanban", "stats"],
    ["config", "set", "model", "gpt"],
    ["completion"],
])
def test_every_other_shape_runs_the_prelude(argv):
    assert not skips_startup_prelude(argv), f"{argv} still needs the prelude"


def test_update_defers_only_the_boot_bootstrap():
    """A killed update strands gateways: `update` must keep that recovery, not skip it."""
    assert ("update",) in main_startup._BOOT_BOOTSTRAP_OWNED_SHAPES
    assert not skips_startup_prelude(["update"])


def _forbidden(name: str):
    def boom(*args, **kwargs):
        raise AssertionError(f"{name} must not run for --help")

    return boom


def test_main_help_runs_no_startup_side_effects(monkeypatch, capsys):
    """End to end through the real entry point: `hermes --help` touches none of it."""
    from hermes_cli import main as cli_main

    for name in (
        "_cleanup_quarantined_exes",
        "_sweep_stale_bytecode_if_checkout_changed",
        "_recover_update_debts_on_startup",
    ):
        monkeypatch.setattr(cli_main, name, _forbidden(name))
    monkeypatch.setattr(
        "hermes_cli.boot_bootstrap.maybe_run_boot_bootstrap",
        _forbidden("maybe_run_boot_bootstrap"),
    )
    monkeypatch.setattr("hermes_cli.venv_sync.check_runtime", _forbidden("check_runtime"))

    monkeypatch.setattr(sys, "argv", ["hermes", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        cli_main.main()

    assert exit_info.value.code == 0
    assert "usage: hermes" in capsys.readouterr().out
