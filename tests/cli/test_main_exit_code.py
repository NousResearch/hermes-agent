"""Tests for top-level CLI exit-code propagation."""
from __future__ import annotations

import sys

import pytest

from hermes_cli import main as hermes_main


def test_main_exits_with_integer_command_return(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["hermes", "version"])
    monkeypatch.setattr(hermes_main, "cmd_version", lambda _args: 7)

    with pytest.raises(SystemExit) as exc_info:
        hermes_main.main()

    assert exc_info.value.code == 7
