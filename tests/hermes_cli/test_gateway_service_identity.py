"""_system_service_identity() must degrade cleanly when the POSIX account database is absent.

test_gateway_service.py skips wholesale on win32 (``importorskip("pwd")``), so the guard that turns a
missing ``pwd``/``grp`` module into a ValueError is proven here instead — a file that runs on every host.
"""

import sys

import pytest

from hermes_cli import gateway as gateway_cli


def test_identity_without_posix_pwd_database_raises_valueerror(monkeypatch):
    """win32 has no ``pwd``/``grp``: a non-root identity request must surface a ValueError,
    not a raw ModuleNotFoundError that escapes the CLI (#137195)."""
    monkeypatch.delenv("SUDO_USER", raising=False)
    monkeypatch.setenv("USER", "somebody-such-user")
    monkeypatch.setenv("LOGNAME", "somebody-such-user")
    monkeypatch.setitem(sys.modules, "pwd", None)
    monkeypatch.setitem(sys.modules, "grp", None)

    with pytest.raises(ValueError, match="(?i)posix pwd/grp lookup is unavailable"):
        gateway_cli._system_service_identity(run_as_user=None)
