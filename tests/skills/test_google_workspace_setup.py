"""Google Workspace setup: PM-owned dependencies and auth checks."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pm
import pytest


SETUP_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/scripts/setup.py"
)


@pytest.fixture()
def setup_module(monkeypatch):
    # setup.py exposes sibling imports for direct script execution.
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


def _without_hermes_token(setup_module, monkeypatch, tmp_path, gws_status):
    monkeypatch.setattr(setup_module, "TOKEN_PATH", tmp_path / "google_token.json")
    monkeypatch.setattr(setup_module, "_gws_native_status", lambda: gws_status)


def test_gws_auth_status_ignores_inherited_credentials_file(setup_module, monkeypatch):
    """The check must see the same gws login google_api.py uses without a Hermes token."""
    monkeypatch.setenv("HERMES_GWS_BIN", "/usr/bin/gws")
    monkeypatch.setenv("GOOGLE_WORKSPACE_CLI_CREDENTIALS_FILE", "/stale/credentials.json")
    status = {"auth_method": "oauth2", "token_valid": True}
    run = Mock(return_value=Mock(stdout=json.dumps(status)))
    monkeypatch.setattr(setup_module.subprocess, "run", run)

    assert setup_module._gws_native_status() == status
    assert run.call_args.args[0] == ["/usr/bin/gws", "auth", "status"]
    assert "GOOGLE_WORKSPACE_CLI_CREDENTIALS_FILE" not in run.call_args.kwargs["env"]

    setup_module._gws_native_status.cache_clear()
    run.return_value = Mock(stdout=json.dumps({"auth_method": "none"}))
    assert setup_module._gws_native_status() is None


@pytest.mark.parametrize(("gws_status", "ok", "expected"), [
    (None, False, "NOT_AUTHENTICATED"),
    ({"auth_method": "oauth2", "token_valid": False, "token_error": "Token has been expired or revoked."},
     False, "GWS_TOKEN_INVALID: Token has been expired or revoked."),
    ({"auth_method": "oauth2", "token_valid": True, "user": "a@example.com"},
     True, "AUTHENTICATED: Using gws CLI credentials (a@example.com)"),
    ({"auth_method": "oauth2", "token_valid": True, "scopes": ["https://www.googleapis.com/auth/calendar"]},
     True, "AUTHENTICATED (partial): gws login valid but missing 7 scopes"),
])
def test_check_auth_falls_back_to_gws_login(setup_module, monkeypatch, tmp_path, capsys, gws_status, ok, expected):
    _without_hermes_token(setup_module, monkeypatch, tmp_path, gws_status)

    assert setup_module.check_auth() is ok
    assert expected in capsys.readouterr().out


@pytest.mark.parametrize(("user", "ok", "expected"), [
    ("a@example.com", True, "LIVE_CHECK_OK"),
    (None, False, "LIVE_CHECK_FAILED"),
])
def test_check_live_on_gws_login_requires_userinfo_call(setup_module, monkeypatch, tmp_path, capsys, user, ok, expected):
    gws_status = {"auth_method": "oauth2", "token_valid": True}
    if user:
        gws_status["user"] = user
    _without_hermes_token(setup_module, monkeypatch, tmp_path, gws_status)

    assert setup_module.check_auth_live() is ok
    assert expected in capsys.readouterr().out
