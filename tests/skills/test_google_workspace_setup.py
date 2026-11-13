"""Google Workspace setup dependency ownership and gws-native auth coverage.

Native auth cases relocated from the removed test_google_oauth_setup.py.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pm
import pytest


SETUP_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/scripts/setup.py"
)


@pytest.fixture()
def setup_module(monkeypatch, tmp_path):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.syspath_prepend(str(SETUP_PATH.parent))
    spec = importlib.util.spec_from_file_location("google_workspace_setup", SETUP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("error", [None, pm.InstallError("venv", "sync refused")])
def test_explicit_install_uses_pm_and_reports_restart(setup_module, monkeypatch, capsys, error):
    sync = Mock(side_effect=error)
    monkeypatch.setattr(pm, "sync_venv", sync)
    # Even a successful old-interpreter probe must not bypass explicit sync.
    monkeypatch.setattr(pm, "ensure_import", Mock(side_effect=AssertionError("not a sync")))
    monkeypatch.setattr("subprocess.check_call", Mock(side_effect=AssertionError("ambient install")))

    assert setup_module.install_deps() is (error is None)
    sync.assert_called_once_with(["google"], explicit=True)
    output = capsys.readouterr().out
    if error is None:
        assert "restart" in output.lower()
    else:
        assert "sync refused" in output


def test_auth_uses_pm_import_check(setup_module, monkeypatch):
    ensure = Mock()
    monkeypatch.setattr(pm, "ensure_import", ensure)
    monkeypatch.setattr(pm, "sync_venv", Mock(side_effect=AssertionError("explicit sync during auth")))
    monkeypatch.setattr("subprocess.check_call", Mock(side_effect=AssertionError("ambient install")))

    setup_module._ensure_deps()

    ensure.assert_called_once_with("google")


class TestGwsNativeAuth:
    """setup.py --check / --check-live behavior in gws-native mode."""

    def test_check_auth_returns_true_when_gws_native(self, setup_module, monkeypatch, capsys):
        """No Hermes token but gws authed on its own → AUTHENTICATED via gws."""
        assert not setup_module.TOKEN_PATH.exists()
        monkeypatch.setattr(setup_module, "gws_native_authed", lambda: True)
        assert setup_module.check_auth() is True
        out = capsys.readouterr().out
        assert "AUTHENTICATED: via gws CLI" in out

    def test_check_auth_false_when_no_token_and_no_gws(self, setup_module, monkeypatch, capsys):
        """No token and gws not authed → NOT_AUTHENTICATED."""
        assert not setup_module.TOKEN_PATH.exists()
        monkeypatch.setattr(setup_module, "gws_native_authed", lambda: False)
        assert setup_module.check_auth() is False
        out = capsys.readouterr().out
        assert "NOT_AUTHENTICATED" in out

    def test_check_live_uses_gws_when_native(self, setup_module, monkeypatch, capsys):
        """In gws-native mode, --check-live validates via a real gws call, not the token client."""
        assert not setup_module.TOKEN_PATH.exists()
        monkeypatch.setattr(setup_module, "gws_native_authed", lambda: True)
        monkeypatch.setattr(setup_module, "gws_live_check", lambda: (True, "ok"))
        assert setup_module.check_auth_live() is True
        assert "LIVE_CHECK_OK" in capsys.readouterr().out

    def test_check_live_reports_disabled_client_from_gws(self, setup_module, monkeypatch, capsys):
        """A disabled_client error surfaced by gws is reported, not swallowed."""
        assert not setup_module.TOKEN_PATH.exists()
        monkeypatch.setattr(setup_module, "gws_native_authed", lambda: True)
        monkeypatch.setattr(
            setup_module, "gws_live_check", lambda: (False, "disabled_client: the client is disabled")
        )
        assert setup_module.check_auth_live() is False
        out = capsys.readouterr().out
        assert "LIVE_CHECK_FAILED" in out
        assert "disabled" in out.lower()
