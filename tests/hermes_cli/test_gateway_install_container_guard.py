"""Container guard on the systemd install path.

``hermes gateway install`` routes to ``_install_systemd_from_cli()`` whenever the systemd
backend matches, and a systemd container deliberately matches (``supports_systemd_services()``
returns True there so system units keep working). Without a container check the default
user-scope install writes the unit and its ``default.target.wants`` symlink into the home
directory — commonly the host's own home bind-mounted into the container — so the host's
``systemd --user`` manager enables the same unit and a second gateway starts outside the
container against the same bot token.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

import hermes_cli.gateway as gateway


def _install_args():
    return SimpleNamespace(
        start_now=None,
        start_on_login=None,
    )


class TestUserScopeInstallRefusedInContainer:
    def test_user_scope_refused_inside_container(self, monkeypatch, capsys):
        monkeypatch.setattr(gateway, "is_container", lambda: True)
        monkeypatch.setattr(gateway, "is_wsl", lambda: False)
        monkeypatch.setattr(
            gateway,
            "systemd_install",
            lambda **kw: pytest.fail(
                "systemd_install must not run for a user-scope install inside a container"
            ),
        )

        with pytest.raises(SystemExit) as exc_info:
            gateway._install_systemd_from_cli(
                _install_args(), force=False, system=False, run_as_user=None
            )
        assert exc_info.value.code == 1

        out = capsys.readouterr().out
        assert (
            "user-scope gateway service cannot be installed inside a container" in out
        )
        assert "hermes gateway run" in out
        assert "--system --run-as-user" in out

    def test_system_scope_still_installable_inside_container(self, monkeypatch, capsys):
        monkeypatch.setattr(gateway, "is_container", lambda: True)
        monkeypatch.setattr(gateway, "is_wsl", lambda: False)
        monkeypatch.setattr(gateway, "systemd_install", lambda **kw: None)
        monkeypatch.setattr(gateway, "systemd_start", lambda **kw: None)

        gateway._install_systemd_from_cli(
            _install_args(), force=False, system=True, run_as_user="tomek"
        )

        out = capsys.readouterr().out
        assert "cannot be installed inside a container" not in out

    def test_user_scope_still_installable_on_plain_host(self, monkeypatch, capsys):
        monkeypatch.setattr(gateway, "is_container", lambda: False)
        monkeypatch.setattr(gateway, "is_wsl", lambda: False)
        monkeypatch.setattr(gateway, "systemd_install", lambda **kw: None)
        monkeypatch.setattr(gateway, "systemd_start", lambda **kw: None)

        gateway._install_systemd_from_cli(
            _install_args(), force=False, system=False, run_as_user=None
        )

        out = capsys.readouterr().out
        assert "cannot be installed inside a container" not in out
