"""The guard repo scripts call to require an activated shell.

``scripts/_activation.py`` is stdlib-only and importable before the environment
it checks for exists. Only one part of it cannot be eyeballed from a single
host: ``activation_command()`` names a different command per shell, so a POSIX
run cannot see the Windows string and vice versa.
"""

from __future__ import annotations

import pytest

from scripts._activation import ACTIVATION_ENV_VAR, activation_command, require_activation


def test_returns_when_activated_and_exits_naming_the_command_when_not(monkeypatch, capsys):
    monkeypatch.setenv(ACTIVATION_ENV_VAR, "<installed-state path>")
    require_activation()  # must return, not raise

    monkeypatch.delenv(ACTIVATION_ENV_VAR)
    with pytest.raises(SystemExit) as excinfo:
        require_activation()
    assert excinfo.value.code == 1
    assert activation_command() in capsys.readouterr().err


@pytest.mark.platforms("posix")
def test_command_is_the_posix_source_line():
    assert activation_command() == "source ./activate"


@pytest.mark.platforms("windows")
def test_command_is_powershell_on_a_native_windows_shell(monkeypatch):
    monkeypatch.delenv("MSYSTEM", raising=False)
    monkeypatch.delenv("SHELL", raising=False)
    assert activation_command() == ". .\\activate.ps1"