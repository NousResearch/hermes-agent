"""
Tests for Hermes CLI cross-platform subcommands (`platform`, `runtime`, `connect`).
"""

import pytest
from hermes_cli.subcommands.platform import main as platform_main
from hermes_cli.subcommands.runtime import main as runtime_main
from hermes_cli.subcommands.connect import main as connect_main


def test_platform_subcommand(capsys):
    platform_main()
    captured = capsys.readouterr()
    assert "=== Hermes Platform Information ===" in captured.out
    assert "Platform Type:" in captured.out


def test_runtime_subcommand(capsys):
    runtime_main()
    captured = capsys.readouterr()
    assert "=== Hermes Runtimes ===" in captured.out
    assert "local_default" in captured.out


def test_connect_subcommand(capsys):
    connect_main()
    captured = capsys.readouterr()
    assert "=== Hermes Remote Connection Manager ===" in captured.out
