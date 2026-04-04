"""Regression tests for Google Workspace API credential validation."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/scripts/google_api.py"
)


class FakeAuthorizedCredentials:
    def __init__(self, *, valid=True, expired=False, refresh_token="refresh-token"):
        self.valid = valid
        self.expired = expired
        self.refresh_token = refresh_token
        self.refresh_calls = 0

    def refresh(self, _request):
        self.refresh_calls += 1
        self.valid = True
        self.expired = False

    def to_json(self):
        return json.dumps({
            "token": "refreshed-token",
            "refresh_token": self.refresh_token,
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
        })


class FakeCredentialsFactory:
    creds = FakeAuthorizedCredentials()

    @classmethod
    def from_authorized_user_file(cls, _path, _scopes):
        return cls.creds


@pytest.fixture
def google_api_module(monkeypatch, tmp_path):
    google_module = types.ModuleType("google")
    oauth2_module = types.ModuleType("google.oauth2")
    credentials_module = types.ModuleType("google.oauth2.credentials")
    credentials_module.Credentials = FakeCredentialsFactory
    auth_module = types.ModuleType("google.auth")
    transport_module = types.ModuleType("google.auth.transport")
    requests_module = types.ModuleType("google.auth.transport.requests")
    requests_module.Request = object

    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.oauth2", oauth2_module)
    monkeypatch.setitem(sys.modules, "google.oauth2.credentials", credentials_module)
    monkeypatch.setitem(sys.modules, "google.auth", auth_module)
    monkeypatch.setitem(sys.modules, "google.auth.transport", transport_module)
    monkeypatch.setitem(sys.modules, "google.auth.transport.requests", requests_module)

    spec = importlib.util.spec_from_file_location("google_workspace_api_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    monkeypatch.setattr(module, "TOKEN_PATH", tmp_path / "google_token.json")
    return module


def _write_token(path: Path, scopes):
    path.write_text(json.dumps({
        "token": "access-token",
        "refresh_token": "refresh-token",
        "token_uri": "https://oauth2.googleapis.com/token",
        "client_id": "client-id",
        "client_secret": "client-secret",
        "scopes": scopes,
    }))


def test_get_credentials_rejects_missing_scopes(google_api_module, capsys):
    FakeCredentialsFactory.creds = FakeAuthorizedCredentials(valid=True)
    _write_token(google_api_module.TOKEN_PATH, [
        "https://www.googleapis.com/auth/drive.readonly",
        "https://www.googleapis.com/auth/spreadsheets",
    ])

    with pytest.raises(SystemExit):
        google_api_module.get_credentials()

    err = capsys.readouterr().err
    assert "missing google workspace scopes" in err.lower()
    assert "gmail.send" in err


def test_get_credentials_accepts_full_scope_token(google_api_module):
    FakeCredentialsFactory.creds = FakeAuthorizedCredentials(valid=True)
    _write_token(google_api_module.TOKEN_PATH, list(google_api_module.SCOPES))

    creds = google_api_module.get_credentials()

    assert creds is FakeCredentialsFactory.creds
