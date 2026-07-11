"""Regression tests for top-level Hermes CLI exit-code propagation."""

import sys

from hermes_cli import main as cli


def test_main_returns_subcommand_exit_code(monkeypatch):
    monkeypatch.setattr(cli, "cmd_kanban", lambda _args: 7)
    monkeypatch.setattr(sys, "argv", ["hermes", "kanban", "list"])

    assert cli.main() == 7