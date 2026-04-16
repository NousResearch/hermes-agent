"""Regression tests for Google Workspace OAuth setup.

These tests cover the headless/manual auth-code flow where the browser step and
code exchange happen in separate process invocations.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/scripts/setup.py"
)


def test_fallback_path_resolves_to_hermes_agent_root():
    """Upward search finds hermes_constants.py when walking from the script's location."""
    found = next(
        (p for p in SCRIPT_PATH.resolve().parents if (p / "hermes_constants.py").exists()),
        None,
    )
    assert found is not None, "hermes_constants.py not found in any parent directory"
    assert (found / "hermes_constants.py").exists()


class FakeCredentials:
    def __init__(self, payload=None):
        self._payload = payload or {
            "token": "access-token",
            "refresh_token": "refresh-token",
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_id": "client-id",
            "client_secret": "client-secret",
            "scopes": [
                "https://www.googleapis.com/auth/gmail.readonly",
                "https://www.googleapis.com/auth/gmail.send",
                "https://www.googleapis.com/auth/gmail.modify",
                "https://www.googleapis.com/auth/calendar",
                "https://www.googleapis.com/auth/drive.readonly",
                "https://www.googleapis.com/auth/contacts.readonly",
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/documents.readonly",
            ],
        }

    def to_json(self):
        return json.dumps(self._payload)


class FakeFlow:
    created = []
    default_state = "generated-state"
    default_verifier = "generated-code-verifier"
    credentials_payload = None
    fetch_error = None

    def __init__(
        self,
        client_secrets_file,
        scopes,
        *,
        redirect_uri=None,
        state=None,
        code_verifier=None,
        autogenerate_code_verifier=False,
    ):
        self.client_secrets_file = client_secrets_file
        self.scopes = scopes
        self.redirect_uri = redirect_uri
        self.state = state
        self.code_verifier = code_verifier
        self.autogenerate_code_verifier = autogenerate_code_verifier
        self.authorization_kwargs = None
        self.fetch_token_calls = []
        self.credentials = FakeCredentials(self.credentials_payload)

        if autogenerate_code_verifier and not self.code_verifier:
            self.code_verifier = self.default_verifier
        if not self.state:
            self.state = self.default_state

    @classmethod
    def reset(cls):
        cls.created = []
        cls.default_state = "generated-state"
        cls.default_verifier = "generated-code-verifier"
        cls.credentials_payload = None
        cls.fetch_error = None

    @classmethod
    def from_client_secrets_file(cls, client_secrets_file, scopes, **kwargs):
        inst = cls(client_secrets_file, scopes, **kwargs)
        cls.created.append(inst)
        return inst

    def authorization_url(self, **kwargs):
        self.authorization_kwargs = kwargs
        return f"https://auth.example/authorize?state={self.state}", self.state

    def fetch_token(self, **kwargs):
        self.fetch_token_calls.append(kwargs)
        if self.fetch_error:
            raise self.fetch_error


@pytest.fixture
def setup_module(monkeypatch, tmp_path):
    FakeFlow.reset()

    google_auth_module = types.ModuleType("google_auth_oauthlib")
    flow_module = types.ModuleType("google_auth_oauthlib.flow")
    flow_module.Flow = FakeFlow
    google_auth_module.flow = flow_module
    monkeypatch.setitem(sys.modules, "google_auth_oauthlib", google_auth_module)
    monkeypatch.setitem(sys.modules, "google_auth_oauthlib.flow", flow_module)

    spec = importlib.util.spec_from_file_location("google_workspace_setup_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    monkeypatch.setattr(module, "_ensure_deps", lambda: None)
    monkeypatch.setattr(module, "CLIENT_SECRET_PATH", tmp_path / "google_client_secret.json")
    monkeypatch.setattr(module, "TOKEN_PATH", tmp_path / "google_token.json")
    monkeypatch.setattr(module, "PENDING_AUTH_PATH", tmp_path / "google_oauth_pending.json", raising=False)

    client_secret = {
        "installed": {
            "client_id": "client-id",
            "client_secret": "client-secret",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }
    module.CLIENT_SECRET_PATH.write_text(json.dumps(client_secret))
    return module


class TestGetAuthUrl:
    def test_persists_state_and_code_verifier_for_later_exchange(self, setup_module, capsys):
        setup_module.get_auth_url()

        out = capsys.readouterr().out.strip()
        assert out == "https://auth.example/authorize?state=generated-state"

        saved = json.loads(setup_module.PENDING_AUTH_PATH.read_text())
        assert saved["state"] == "generated-state"
        assert saved["code_verifier"] == "generated-code-verifier"

        flow = FakeFlow.created[-1]
        assert flow.autogenerate_code_verifier is True
        assert flow.authorization_kwargs == {"access_type": "offline", "prompt": "consent"}


class TestExchangeAuthCode:
    def test_reuses_saved_pkce_material_for_plain_code(self, setup_module):
        setup_module.PENDING_AUTH_PATH.write_text(
            json.dumps({"state": "saved-state", "code_verifier": "saved-verifier"})
        )

        setup_module.exchange_auth_code("4/test-auth-code")

        flow = FakeFlow.created[-1]
        assert flow.state == "saved-state"
        assert flow.code_verifier == "saved-verifier"
        assert flow.fetch_token_calls == [{"code": "4/test-auth-code"}]
        assert json.loads(setup_module.TOKEN_PATH.read_text())["token"] == "access-token"
        assert not setup_module.PENDING_AUTH_PATH.exists()

    def test_extracts_code_from_redirect_url_and_checks_state(self, setup_module):
        setup_module.PENDING_AUTH_PATH.write_text(
            json.dumps({"state": "saved-state", "code_verifier": "saved-verifier"})
        )

        setup_module.exchange_auth_code(
            "http://localhost:1/?code=4/extracted-code&state=saved-state&scope=gmail"
        )

        flow = FakeFlow.created[-1]
        assert flow.fetch_token_calls == [{"code": "4/extracted-code"}]

    def test_rejects_state_mismatch(self, setup_module, capsys):
        setup_module.PENDING_AUTH_PATH.write_text(
            json.dumps({"state": "saved-state", "code_verifier": "saved-verifier"})
        )

        with pytest.raises(SystemExit):
            setup_module.exchange_auth_code(
                "http://localhost:1/?code=4/extracted-code&state=wrong-state"
            )

        out = capsys.readouterr().out
        assert "state mismatch" in out.lower()
        assert not setup_module.TOKEN_PATH.exists()

    def test_requires_pending_auth_session(self, setup_module, capsys):
        with pytest.raises(SystemExit):
            setup_module.exchange_auth_code("4/test-auth-code")

        out = capsys.readouterr().out
        assert "run --auth-url first" in out.lower()
        assert not setup_module.TOKEN_PATH.exists()

    def test_keeps_pending_auth_session_when_exchange_fails(self, setup_module, capsys):
        setup_module.PENDING_AUTH_PATH.write_text(
            json.dumps({"state": "saved-state", "code_verifier": "saved-verifier"})
        )
        FakeFlow.fetch_error = Exception("invalid_grant: Missing code verifier")

        with pytest.raises(SystemExit):
            setup_module.exchange_auth_code("4/test-auth-code")

        out = capsys.readouterr().out
        assert "token exchange failed" in out.lower()
        assert setup_module.PENDING_AUTH_PATH.exists()
        assert not setup_module.TOKEN_PATH.exists()

    def test_accepts_narrower_scopes_with_warning(self, setup_module, capsys):
        """Partial scopes are accepted with a warning (gws migration: v2.0)."""
        setup_module.PENDING_AUTH_PATH.write_text(
            json.dumps({"state": "saved-state", "code_verifier": "saved-verifier"})
        )
        setup_module.TOKEN_PATH.write_text(json.dumps({"token": "***", "scopes": setup_module.SCOPES}))
        FakeFlow.credentials_payload = {
            "token": "***",
            "refresh_token": "***",
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_id": "client-id",
            "client_secret": "client-secret",
            "scopes": [
                "https://www.googleapis.com/auth/drive.readonly",
                "https://www.googleapis.com/auth/spreadsheets",
            ],
        }

        setup_module.exchange_auth_code("4/test-auth-code")

        out = capsys.readouterr().out
        assert "warning" in out.lower()
        assert "missing" in out.lower()
        # Token is saved (partial scopes accepted)
        assert setup_module.TOKEN_PATH.exists()
        # Pending auth is cleaned up
        assert not setup_module.PENDING_AUTH_PATH.exists()


class TestInstallDeps:
    def test_returns_true_when_already_installed(self, setup_module):
        """No subprocess calls when packages are already importable."""
        import sys
        from unittest.mock import patch, MagicMock

        fake_googleapiclient = MagicMock()
        fake_google_auth = MagicMock()

        with patch.dict(sys.modules, {
            "googleapiclient": fake_googleapiclient,
            "google_auth_oauthlib": fake_google_auth,
        }), patch("subprocess.check_call") as mock_call:
            result = setup_module.install_deps()

        assert result is True
        mock_call.assert_not_called()

    def test_uses_uv_when_available(self, setup_module):
        """uv pip install --python sys.executable is tried first when uv is on PATH."""
        import subprocess
        import sys
        from unittest.mock import patch, MagicMock

        # Remove cached imports so install_deps() sees them as missing
        saved = {}
        for mod in ("googleapiclient", "google_auth_oauthlib"):
            saved[mod] = sys.modules.pop(mod, None)

        calls = []
        def fake_check_call(cmd, **kwargs):
            calls.append(list(cmd))

        try:
            with patch.object(setup_module.shutil, "which", return_value="/usr/bin/uv"), \
                 patch("subprocess.check_call", side_effect=fake_check_call):
                result = setup_module.install_deps()
        finally:
            for mod, val in saved.items():
                if val is not None:
                    sys.modules[mod] = val
                else:
                    sys.modules.pop(mod, None)

        assert result is True
        assert any(c[0] == "/usr/bin/uv" and "pip" in c for c in calls)

    def test_falls_back_to_pip_user_when_uv_missing(self, setup_module):
        """Falls back through pip → pip --user when uv is not found."""
        import subprocess
        import sys
        from unittest.mock import patch

        saved = {}
        for mod in ("googleapiclient", "google_auth_oauthlib"):
            saved[mod] = sys.modules.pop(mod, None)

        calls = []
        def fake_check_call(cmd, **kwargs):
            calls.append(list(cmd))
            if "--user" not in cmd:
                raise subprocess.CalledProcessError(1, cmd)

        try:
            with patch.object(setup_module.shutil, "which", return_value=None), \
                 patch("subprocess.check_call", side_effect=fake_check_call):
                result = setup_module.install_deps()
        finally:
            for mod, val in saved.items():
                if val is not None:
                    sys.modules[mod] = val
                else:
                    sys.modules.pop(mod, None)

        assert result is True
        assert any("--user" in c for c in calls)

    def test_returns_false_when_all_methods_fail(self, setup_module, capsys):
        """Returns False and prints error when all install methods fail."""
        import subprocess
        import sys
        from unittest.mock import patch

        saved = {}
        for mod in ("googleapiclient", "google_auth_oauthlib"):
            saved[mod] = sys.modules.pop(mod, None)

        def always_fail(cmd, **kwargs):
            raise subprocess.CalledProcessError(1, cmd)

        try:
            with patch.object(setup_module.shutil, "which", return_value=None), \
                 patch("subprocess.check_call", side_effect=always_fail):
                result = setup_module.install_deps()
        finally:
            for mod, val in saved.items():
                if val is not None:
                    sys.modules[mod] = val
                else:
                    sys.modules.pop(mod, None)

        assert result is False
        out = capsys.readouterr().out
        assert "ERROR" in out or "failed" in out.lower()
