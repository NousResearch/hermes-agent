"""Scoped Google Workspace OAuth setup contracts (#123445 / #21774)."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest


SETUP_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/scripts/setup.py"
)


class _FakeCredentials:
    granted_scopes = None

    def __init__(self):
        self.valid = True
        self.expired = False
        self.refresh_token = "refresh-token"

    def to_json(self):
        return json.dumps(
            {
                "token": "access-token",
                "refresh_token": self.refresh_token,
                "token_uri": "https://oauth2.googleapis.com/token",
                "client_id": "client-id",
                "client_secret": "client-secret",
            }
        )


class _FakeFlow:
    created = []

    def __init__(self, scopes, state=None, code_verifier=None, autogenerate_code_verifier=False):
        self.scopes = list(scopes)
        self.state = state or "generated-state"
        self.code_verifier = code_verifier or "generated-verifier"
        self.credentials = _FakeCredentials()
        self.fetch_token_calls = []
        self.__class__.created.append(self)

    @classmethod
    def from_client_secrets_file(cls, _path, scopes, **kwargs):
        return cls(
            scopes,
            state=kwargs.get("state"),
            code_verifier=kwargs.get("code_verifier"),
            autogenerate_code_verifier=kwargs.get("autogenerate_code_verifier", False),
        )

    def authorization_url(self, **_kwargs):
        return "https://auth.example/authorize?state=generated-state", self.state

    def fetch_token(self, **kwargs):
        self.fetch_token_calls.append(kwargs)


@pytest.fixture()
def setup_module(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(SETUP_PATH.parent))
    spec = importlib.util.spec_from_file_location("google_workspace_scoped_setup", SETUP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(module, "_ensure_deps", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "CLIENT_SECRET_PATH", tmp_path / "client.json")
    monkeypatch.setattr(module, "TOKEN_PATH", tmp_path / "token.json")
    monkeypatch.setattr(module, "PENDING_AUTH_PATH", tmp_path / "pending.json")
    monkeypatch.setattr(module, "LAST_AUTH_URL_PATH", tmp_path / "last-url.txt")
    module.CLIENT_SECRET_PATH.write_text('{"installed": {}}', encoding="utf-8")

    flow_mod = types.ModuleType("google_auth_oauthlib.flow")
    flow_mod.Flow = _FakeFlow
    package = types.ModuleType("google_auth_oauthlib")
    package.flow = flow_mod
    monkeypatch.setitem(sys.modules, "google_auth_oauthlib", package)
    monkeypatch.setitem(sys.modules, "google_auth_oauthlib.flow", flow_mod)
    _FakeFlow.created.clear()
    return module


def test_documented_service_subset_controls_authorization_scopes(setup_module, capsys):
    setup_module.get_auth_url("email,calendar", "json")

    payload = json.loads(capsys.readouterr().out)
    expected = [*setup_module.GMAIL_SCOPES, *setup_module.CALENDAR_SCOPES]
    assert payload["services"] == ["email", "calendar"]
    assert payload["scopes"] == expected
    assert _FakeFlow.created[-1].scopes == expected
    assert setup_module.DRIVE_SCOPES[0] not in expected
    assert setup_module.DOCS_SCOPES[0] not in expected

    pending = json.loads(setup_module.PENDING_AUTH_PATH.read_text(encoding="utf-8"))
    assert pending["services"] == ["email", "calendar"]
    assert pending["requested_scopes"] == expected
    assert setup_module.LAST_AUTH_URL_PATH.read_text(encoding="utf-8") == payload["auth_url"]


def test_requested_scope_metadata_prevents_false_partial_warning(setup_module):
    payload = {
        "services": ["calendar"],
        "requested_scopes": setup_module.CALENDAR_SCOPES,
        "scopes": setup_module.CALENDAR_SCOPES,
    }
    assert setup_module._missing_scopes_from_payload(payload) == []


def test_unknown_service_fails_closed(setup_module, capsys):
    with pytest.raises(SystemExit) as exc:
        setup_module.get_auth_url("calendar,not-a-service", "json")

    assert exc.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"] == "invalid_services"
    assert not setup_module.PENDING_AUTH_PATH.exists()


def test_main_accepts_documented_auth_url_flags(setup_module, monkeypatch):
    calls = []
    monkeypatch.setattr(
        setup_module,
        "get_auth_url",
        lambda services="all", output_format="text": calls.append((services, output_format)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["setup.py", "--auth-url", "--services", "email,calendar", "--format", "json"],
    )

    setup_module.main()
    assert calls == [("email,calendar", "json")]


def test_json_dependency_error_is_one_machine_readable_document(monkeypatch, capsys):
    monkeypatch.syspath_prepend(str(SETUP_PATH.parent))
    spec = importlib.util.spec_from_file_location("google_workspace_dependency_json", SETUP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(
        module.pm,
        "ensure_import",
        Mock(side_effect=RuntimeError("restart Hermes to activate")),
    )
    with pytest.raises(SystemExit) as exc:
        module._ensure_deps("json")

    assert exc.value.code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"] == "google_dependencies_unavailable"
    assert "restart Hermes" in payload["message"]


def test_live_check_uses_oauth_refresh_not_calendar_scope(setup_module, monkeypatch, capsys):
    monkeypatch.setattr(setup_module, "check_auth", lambda quiet=False: True)
    refreshed = []

    class Credentials:
        refresh_token = "refresh-token"

        @classmethod
        def from_authorized_user_file(cls, _path):
            return cls()

        def refresh(self, _request):
            refreshed.append(True)

    credentials_mod = types.ModuleType("google.oauth2.credentials")
    credentials_mod.Credentials = Credentials
    oauth2_mod = types.ModuleType("google.oauth2")
    oauth2_mod.credentials = credentials_mod
    requests_mod = types.ModuleType("google.auth.transport.requests")
    requests_mod.Request = lambda: object()
    transport_mod = types.ModuleType("google.auth.transport")
    transport_mod.requests = requests_mod
    auth_mod = types.ModuleType("google.auth")
    auth_mod.transport = transport_mod
    google_mod = types.ModuleType("google")
    google_mod.oauth2 = oauth2_mod

    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.oauth2", oauth2_mod)
    monkeypatch.setitem(sys.modules, "google.oauth2.credentials", credentials_mod)
    monkeypatch.setitem(sys.modules, "google.auth", auth_mod)
    monkeypatch.setitem(sys.modules, "google.auth.transport", transport_mod)
    monkeypatch.setitem(sys.modules, "google.auth.transport.requests", requests_mod)
    # If the old Calendar-specific path is reached, fail loudly.
    monkeypatch.setitem(sys.modules, "googleapiclient.discovery", None)

    assert setup_module.check_auth_live() is True
    assert refreshed == [True]
    assert "OAuth refresh succeeded" in capsys.readouterr().out


def test_readonly_service_variants_do_not_request_write_scopes(setup_module, capsys):
    setup_module.get_auth_url("calendar-readonly,drive-readonly", "json")

    payload = json.loads(capsys.readouterr().out)
    assert payload["services"] == ["calendar-readonly", "drive-readonly"]
    assert payload["scopes"] == [
        "https://www.googleapis.com/auth/calendar.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    assert setup_module.CALENDAR_SCOPES[0] not in payload["scopes"]
    assert setup_module.DRIVE_SCOPES[0] not in payload["scopes"]
