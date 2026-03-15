"""Regression tests for Google Workspace OAuth setup.

These tests cover the headless/manual auth-code flow where the browser step and
code exchange happen in separate process invocations.
"""

from __future__ import annotations

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


class FakeCredentials:
    def __init__(self, payload=None):
        self._payload = payload or {
            "token": "access-token",
            "refresh_token": "refresh-token",
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_id": "client-id",
            "client_secret": "client-secret",
            "scopes": ["scope-a"],
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
    monkeypatch.setattr(module, "LAST_AUTH_URL_PATH", tmp_path / "google_oauth_last_url.txt", raising=False)

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


class TestResolveServices:
    def test_reduces_to_requested_services(self, setup_module):
        services, scopes = setup_module._resolve_services("email,calendar")
        assert services == ["email", "calendar"]
        assert scopes == [
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/gmail.send",
            "https://www.googleapis.com/auth/gmail.modify",
            "https://www.googleapis.com/auth/calendar",
        ]


class TestGetAuthUrl:
    def test_persists_state_verifier_scopes_and_last_url(self, setup_module, capsys):
        setup_module.get_auth_url("email,calendar", output_format="json")

        out = json.loads(capsys.readouterr().out)
        assert out["success"] is True
        assert out["auth_url"] == "https://auth.example/authorize?state=generated-state"
        assert out["services"] == ["email", "calendar"]
        assert Path(out["auth_url_file"]).read_text() == out["auth_url"]

        saved = json.loads(setup_module.PENDING_AUTH_PATH.read_text())
        assert saved["state"] == "generated-state"
        assert saved["code_verifier"] == "generated-code-verifier"
        assert saved["services"] == ["email", "calendar"]
        assert saved["scopes"] == [
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/gmail.send",
            "https://www.googleapis.com/auth/gmail.modify",
            "https://www.googleapis.com/auth/calendar",
        ]

        flow = FakeFlow.created[-1]
        assert flow.autogenerate_code_verifier is True
        assert flow.authorization_kwargs == {"access_type": "offline", "prompt": "consent"}


class TestExchangeAuthCode:
    def test_reuses_saved_pkce_material_for_plain_code(self, setup_module):
        setup_module.PENDING_AUTH_PATH.write_text(
            json.dumps(
                {
                    "state": "saved-state",
                    "code_verifier": "saved-verifier",
                    "services": ["email", "calendar"],
                    "scopes": ["scope-a", "scope-b"],
                }
            )
        )

        setup_module.exchange_auth_code("4/test-auth-code")

        flow = FakeFlow.created[-1]
        assert flow.state == "saved-state"
        assert flow.code_verifier == "saved-verifier"
        assert flow.scopes == ["scope-a", "scope-b"]
        assert flow.fetch_token_calls == [{"code": "4/test-auth-code"}]
        assert json.loads(setup_module.TOKEN_PATH.read_text())["token"] == "access-token"
        assert not setup_module.PENDING_AUTH_PATH.exists()

    def test_extracts_code_from_redirect_url_and_checks_state(self, setup_module):
        setup_module.PENDING_AUTH_PATH.write_text(
            json.dumps(
                {
                    "state": "saved-state",
                    "code_verifier": "saved-verifier",
                    "services": ["email"],
                    "scopes": ["scope-a"],
                }
            )
        )

        setup_module.exchange_auth_code(
            "http://localhost:1/?code=4/extracted-code&state=saved-state&scope=gmail"
        )

        flow = FakeFlow.created[-1]
        assert flow.fetch_token_calls == [{"code": "4/extracted-code"}]

    def test_state_mismatch_regenerates_fresh_url(self, setup_module, capsys):
        setup_module.PENDING_AUTH_PATH.write_text(
            json.dumps(
                {
                    "state": "saved-state",
                    "code_verifier": "saved-verifier",
                    "services": ["email", "calendar"],
                    "scopes": ["scope-a", "scope-b"],
                }
            )
        )
        FakeFlow.default_state = "replacement-state"
        FakeFlow.default_verifier = "replacement-verifier"

        with pytest.raises(SystemExit):
            setup_module.exchange_auth_code(
                "http://localhost:1/?code=4/extracted-code&state=wrong-state",
                output_format="json",
            )

        out = json.loads(capsys.readouterr().out)
        assert out["success"] is False
        assert out["fresh_auth_url"] == "https://auth.example/authorize?state=replacement-state"

        saved = json.loads(setup_module.PENDING_AUTH_PATH.read_text())
        assert saved["state"] == "replacement-state"
        assert saved["code_verifier"] == "replacement-verifier"
        assert saved["services"] == ["email", "calendar"]

    def test_requires_pending_auth_session(self, setup_module, capsys):
        with pytest.raises(SystemExit):
            setup_module.exchange_auth_code("4/test-auth-code")

        out = capsys.readouterr().out
        assert "run --auth-url first" in out.lower()
        assert not setup_module.TOKEN_PATH.exists()

    def test_failed_exchange_regenerates_fresh_url(self, setup_module, capsys):
        setup_module.PENDING_AUTH_PATH.write_text(
            json.dumps(
                {
                    "state": "saved-state",
                    "code_verifier": "saved-verifier",
                    "services": ["email"],
                    "scopes": ["scope-a"],
                }
            )
        )
        FakeFlow.default_state = "replacement-state"
        FakeFlow.default_verifier = "replacement-verifier"
        FakeFlow.fetch_error = Exception("invalid_grant: Missing code verifier")

        with pytest.raises(SystemExit):
            setup_module.exchange_auth_code("4/test-auth-code", output_format="json")

        out = json.loads(capsys.readouterr().out)
        assert out["success"] is False
        assert out["fresh_auth_url"] == "https://auth.example/authorize?state=replacement-state"
        assert setup_module.PENDING_AUTH_PATH.exists()
        assert not setup_module.TOKEN_PATH.exists()

    def test_check_auth_rejects_missing_requested_scopes(self, setup_module, capsys):
        setup_module.TOKEN_PATH.write_text(
            json.dumps(
                {
                    "token": "access-token",
                    "refresh_token": "refresh-token",
                    "scopes": ["https://www.googleapis.com/auth/gmail.readonly"],
                }
            )
        )

        ok = setup_module.check_auth("email,calendar")
        out = capsys.readouterr().out
        assert ok is False
        assert "missing scopes" in out.lower()
