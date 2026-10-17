"""``process_state`` owns only a real Hermes entry point.

The launcher runs its runtime as ``python -I -c <bootstrap> serve …`` (see
``_launchers.runtime_command``) and a plain ``-c`` merely carrying the name is
not identity, so ownership is matched on the *normalized* argv.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from hermes_cli import windows_ssh_runtime as rt
from hermes_cli._launchers import runtime_command

PYTHON = "C:/py/python.exe"
HERMES_PATH = "C:/x/hermes-agent/hermes.exe"
NONCE = "0123456789abcdef"
OTHER_NONCE = "fedcba9876543210"
CREATE_TIME = 12345.0
CREATE_TIME_NS = int(CREATE_TIME * 1_000_000_000)


class _NoSuchProcess(Exception):
    pass


class _AccessDenied(Exception):
    pass


class _FakeProcess:
    def __init__(self, argv: list[str]) -> None:
        self._argv = argv

    def create_time(self) -> float:
        return CREATE_TIME

    def cmdline(self) -> list[str]:
        return list(self._argv)

    def is_running(self) -> bool:
        return True


def _launcher_argv(*, isolated: bool = True) -> list[str]:
    """The published launcher command: ``python -I -c <bootstrap> serve …``."""
    command = [*runtime_command(Path("C:/x/hermes-agent"), [], python=PYTHON), "serve"]
    if isolated:
        command.append("--isolated")
    return [*command, "--host", "127.0.0.1", "--port", "0", "--ssh-owner-nonce", NONCE]


def _state(monkeypatch, argv: list[str], nonce: str = NONCE) -> dict:
    monkeypatch.setitem(sys.modules, "psutil", types.SimpleNamespace(
        Process=lambda pid: _FakeProcess(argv),
        NoSuchProcess=_NoSuchProcess, AccessDenied=_AccessDenied))
    return rt.process_state(4242, CREATE_TIME_NS, HERMES_PATH, nonce)


@pytest.mark.parametrize("argv, nonce, expected", [
    pytest.param(_launcher_argv(), NONCE, True, id="launcher-bootstrap"),
    pytest.param([PYTHON, "-m", "hermes_cli.main", "serve", "--isolated",
                  "--ssh-owner-nonce", NONCE], NONCE, True, id="direct-module"),
    pytest.param([PYTHON, "-c", "print('hermes_cli.main')", "serve", "--isolated",
                  "--ssh-owner-nonce", NONCE], NONCE, False, id="foreign-inline-source"),
    pytest.param(_launcher_argv(), OTHER_NONCE, False, id="wrong-nonce"),
    pytest.param(_launcher_argv(isolated=False), NONCE, False, id="missing-isolated"),
])
def test_process_state_ownership(monkeypatch, argv, nonce, expected):
    state = _state(monkeypatch, argv, nonce)
    assert state["owned"] is expected
    assert state["indeterminate"] is False
