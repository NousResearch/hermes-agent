"""Regression tests for Google Workspace OAuth error handling."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


SETUP_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/scripts/setup.py"
)


class FakeRefreshError(Exception):
    pass


class FakeHttpError(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(body)
        self.resp = SimpleNamespace(status=status)


class FakeCredentials:
    loader = None

    def __init__(self, *, valid=True, expired=False, refresh_token="refresh-token", refresh_error=None):
        self.valid = valid
        self.expired = expired
        self.refresh_token = refresh_token
        self.refresh_error = refresh_error

    @classmethod
    def from_authorized_user_file(cls, path, scopes=None):
        if cls.loader is not None:
            return cls.loader(path, scopes=scopes)
        return cls()

    def refresh(self, _request):
        if self.refresh_error is not None:
            raise self.refresh_error
        self.valid = True
        self.expired = False

    def to_json(self):
        return json.dumps({"token": "refreshed", "refresh_token": self.refresh_token})


def _install_google_fakes(monkeypatch, *, build):
    google = types.ModuleType("google")
    google_auth = types.ModuleType("google.auth")
    google_auth_exceptions = types.ModuleType("google.auth.exceptions")
    google_auth_exceptions.RefreshError = FakeRefreshError
    google_auth_transport = types.ModuleType("google.auth.transport")
    google_auth_requests = types.ModuleType("google.auth.transport.requests")
    google_auth_requests.Request = type("Request", (), {})
    google_auth_transport.requests = google_auth_requests
    google_auth.transport = google_auth_transport
    google_auth.exceptions = google_auth_exceptions

    google_oauth2 = types.ModuleType("google.oauth2")
    google_credentials = types.ModuleType("google.oauth2.credentials")
    google_credentials.Credentials = FakeCredentials
    google_oauth2.credentials = google_credentials

    google_api = types.ModuleType("googleapiclient")
    google_discovery = types.ModuleType("googleapiclient.discovery")
    google_discovery.build = build
    google_errors = types.ModuleType("googleapiclient.errors")
    google_errors.HttpError = FakeHttpError
    google_api.discovery = google_discovery
    google_api.errors = google_errors

    for name, module in {
        "google": google,
        "google.auth": google_auth,
        "google.auth.exceptions": google_auth_exceptions,
        "google.auth.transport": google_auth_transport,
        "google.auth.transport.requests": google_auth_requests,
        "google.oauth2": google_oauth2,
        "google.oauth2.credentials": google_credentials,
        "googleapiclient": google_api,
        "googleapiclient.discovery": google_discovery,
        "googleapiclient.errors": google_errors,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture
def setup_module(monkeypatch, tmp_path):
    FakeCredentials.loader = None
    _install_google_fakes(monkeypatch, build=lambda *_args, **_kwargs: None)
    spec = importlib.util.spec_from_file_location("google_oauth_setup_test", SETUP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.TOKEN_PATH = tmp_path / "google_token.json"
    module._ensure_deps = lambda: None
    yield module
    FakeCredentials.loader = None


def _write_token(module):
    module.TOKEN_PATH.write_text(
        json.dumps({"token": "access", "refresh_token": "refresh"}),
        encoding="utf-8",
    )


def test_structured_refresh_error_wins_over_message_text(setup_module):
    error = FakeRefreshError(
        "invalid_client: misleading text",
        {"error": "invalid_grant"},
    )
    assert setup_module._extract_oauth_error_code(error) == "invalid_grant"


def test_compound_oauth_code_is_not_classified(setup_module):
    assert setup_module._extract_oauth_error_code(Exception("invalid_grant_type: nope")) == ""


@pytest.mark.parametrize(
    ("code", "label"),
    [("disabled_client", "OAUTH_CLIENT_DISABLED"), ("invalid_grant", "TOKEN_REVOKED")],
)
def test_check_auth_classifies_refresh_errors(setup_module, capsys, code, label):
    error = FakeRefreshError(f"{code}: failure", {"error": code})
    creds = FakeCredentials(valid=False, expired=True, refresh_error=error)
    FakeCredentials.loader = lambda _path, scopes=None: creds
    _write_token(setup_module)

    assert setup_module.check_auth() is False
    output = capsys.readouterr().out
    assert f"{label}:" in output


def test_check_auth_generic_refresh_error_is_not_misclassified(setup_module, capsys):
    creds = FakeCredentials(valid=False, expired=True, refresh_error=RuntimeError("network down"))
    FakeCredentials.loader = lambda _path, scopes=None: creds
    _write_token(setup_module)

    assert setup_module.check_auth() is False
    output = capsys.readouterr().out
    assert "REFRESH_FAILED:" in output
    assert "OAUTH_CLIENT_DISABLED" not in output
    assert "TOKEN_REVOKED" not in output


def test_check_auth_live_success(setup_module, monkeypatch, capsys):
    FakeCredentials.loader = lambda _path, scopes=None: FakeCredentials()
    _write_token(setup_module)

    class CalendarList:
        def list(self, **_kwargs):
            return self

        def execute(self):
            return {"items": []}

    class Service:
        def calendarList(self):
            return CalendarList()

    monkeypatch.setitem(sys.modules["googleapiclient.discovery"].__dict__, "build", lambda *_a, **_k: Service())

    assert setup_module.check_auth_live() is True
    assert "LIVE_CHECK_OK:" in capsys.readouterr().out


def test_check_auth_live_disabled_client(setup_module, monkeypatch, capsys):
    FakeCredentials.loader = lambda _path, scopes=None: FakeCredentials()
    _write_token(setup_module)

    class CalendarList:
        def list(self, **_kwargs):
            return self

        def execute(self):
            raise FakeRefreshError("disabled_client: disabled", {"error": "disabled_client"})

    class Service:
        def calendarList(self):
            return CalendarList()

    monkeypatch.setitem(sys.modules["googleapiclient.discovery"].__dict__, "build", lambda *_a, **_k: Service())

    assert setup_module.check_auth_live() is False
    assert "OAuth client or account disabled" in capsys.readouterr().out


def test_check_auth_live_scope_failure_is_partial_success(setup_module, monkeypatch, capsys):
    FakeCredentials.loader = lambda _path, scopes=None: FakeCredentials()
    _write_token(setup_module)

    class CalendarList:
        def list(self, **_kwargs):
            return self

        def execute(self):
            raise FakeHttpError(403, "access_token_scope_insufficient")

    class Service:
        def calendarList(self):
            return CalendarList()

    monkeypatch.setitem(sys.modules["googleapiclient.discovery"].__dict__, "build", lambda *_a, **_k: Service())

    assert setup_module.check_auth_live() is True
    assert "LIVE_CHECK_PARTIAL:" in capsys.readouterr().out
