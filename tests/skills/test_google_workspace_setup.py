"""Google Workspace setup delegates dependency ownership to PM."""

from __future__ import annotations

import importlib.util
import sys
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


# --- scope-mode selection (--mode full|read-only) ---------------------------------


@pytest.fixture()
def auth_env(setup_module, monkeypatch, tmp_path):
    """Run get_auth_url() against a fake OAuth flow and capture the requested scopes."""
    client_secret = tmp_path / "google_client_secret.json"
    client_secret.write_text('{"installed": {"client_id": "x", "client_secret": "y"}}')
    monkeypatch.setattr(setup_module, "CLIENT_SECRET_PATH", client_secret)
    monkeypatch.setattr(setup_module, "PENDING_AUTH_PATH", tmp_path / "pending.json")
    monkeypatch.setattr(setup_module, "_ensure_deps", lambda: None)

    captured = {}

    class FakeFlow:
        code_verifier = "verifier"

        @classmethod
        def from_client_secrets_file(cls, path, *, scopes, **kwargs):
            captured["scopes"] = list(scopes)
            return cls()

        def authorization_url(self, **kwargs):
            return "https://accounts.google.com/o/oauth2/auth?fake", "state-123"

    fake_module = type("fake_flow_module", (), {"Flow": FakeFlow})()
    monkeypatch.setitem(sys.modules, "google_auth_oauthlib.flow", fake_module)
    return captured


def test_read_only_mode_requests_only_readonly_scopes(setup_module, auth_env):
    setup_module.get_auth_url(mode="read-only")
    scopes = auth_env["scopes"]
    assert scopes == setup_module.READ_ONLY_SCOPES
    assert all(scope.endswith(".readonly") for scope in scopes)
    assert "https://www.googleapis.com/auth/gmail.send" not in scopes
    assert "https://www.googleapis.com/auth/gmail.modify" not in scopes


def test_full_mode_requests_legacy_scope_set(setup_module, auth_env):
    setup_module.get_auth_url(mode="full")
    scopes = auth_env["scopes"]
    assert scopes == setup_module.FULL_SCOPES
    assert "https://www.googleapis.com/auth/gmail.send" in scopes
    assert "https://www.googleapis.com/auth/gmail.modify" in scopes


def test_default_mode_is_full_unchanged_behavior(setup_module, auth_env):
    # Regression guard: omitting --mode must request exactly the pre-change SCOPES.
    setup_module.get_auth_url()
    assert auth_env["scopes"] == setup_module.SCOPES


def test_cli_mode_flag_defaults_to_full(setup_module, monkeypatch):
    calls = []
    monkeypatch.setattr(setup_module, "get_auth_url", lambda **kw: calls.append(kw))
    monkeypatch.setattr(sys, "argv", ["setup.py", "--auth-url"])
    setup_module.main()
    assert calls == [{"mode": "full"}]


def test_cli_mode_flag_passes_read_only(setup_module, monkeypatch):
    calls = []
    monkeypatch.setattr(setup_module, "get_auth_url", lambda **kw: calls.append(kw))
    monkeypatch.setattr(sys, "argv", ["setup.py", "--auth-url", "--mode", "read-only"])
    setup_module.main()
    assert calls == [{"mode": "read-only"}]
