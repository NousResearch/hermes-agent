"""Regression tests for Google Workspace OAuth token permissions."""

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="POSIX permission bits are not available on this platform",
)


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills/productivity/google-workspace/scripts/setup.py"
)


class FakeCredentials:
    granted_scopes = None

    def to_json(self):
        return json.dumps(
            {
                "token": "access-token",
                "refresh_token": "refresh-token",
                "token_uri": "https://oauth2.googleapis.com/token",
                "client_id": "client-id",
                "client_secret": "client-secret",
            }
        )


class FakeFlow:
    def __init__(self, *args, **kwargs):
        self.credentials = FakeCredentials()

    @classmethod
    def from_client_secrets_file(cls, *args, **kwargs):
        return cls()

    def fetch_token(self, **kwargs):
        return None


@pytest.fixture
def setup_module(monkeypatch, tmp_path):
    google_auth_module = types.ModuleType("google_auth_oauthlib")
    flow_module = types.ModuleType("google_auth_oauthlib.flow")
    flow_module.Flow = FakeFlow
    google_auth_module.flow = flow_module
    monkeypatch.setitem(sys.modules, "google_auth_oauthlib", google_auth_module)
    monkeypatch.setitem(sys.modules, "google_auth_oauthlib.flow", flow_module)

    spec = importlib.util.spec_from_file_location(
        "google_workspace_setup_permissions_test",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    monkeypatch.setattr(module, "_ensure_deps", lambda: None)
    monkeypatch.setattr(module, "CLIENT_SECRET_PATH", tmp_path / "google_client_secret.json")
    monkeypatch.setattr(module, "TOKEN_PATH", tmp_path / "google_token.json")
    monkeypatch.setattr(module, "PENDING_AUTH_PATH", tmp_path / "google_oauth_pending.json")

    module.CLIENT_SECRET_PATH.write_text('{"installed": {}}', encoding="utf-8")
    module.PENDING_AUTH_PATH.write_text(
        json.dumps({"state": "state", "code_verifier": "verifier"}),
        encoding="utf-8",
    )
    return module


def test_auth_code_exchange_creates_owner_only_token(setup_module):
    setup_module.exchange_auth_code("authorization-code")

    assert setup_module.TOKEN_PATH.stat().st_mode & 0o777 == 0o600


def test_auth_code_exchange_repairs_existing_token_permissions(setup_module):
    setup_module.TOKEN_PATH.write_text('{"token": "old"}', encoding="utf-8")
    setup_module.TOKEN_PATH.chmod(0o644)

    setup_module.exchange_auth_code("authorization-code")

    assert setup_module.TOKEN_PATH.stat().st_mode & 0o777 == 0o600


def test_interrupted_serialization_preserves_existing_token(
    setup_module,
    monkeypatch,
):
    original = '{"token": "existing-refresh-token"}'
    setup_module.TOKEN_PATH.write_text(original, encoding="utf-8")
    storage_module = sys.modules[setup_module.write_private_json.__module__]

    def fail_during_dump(data, file, indent):
        file.write('{"token":')
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(storage_module.json, "dump", fail_during_dump)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        setup_module.write_private_json(
            setup_module.TOKEN_PATH,
            {"token": "replacement"},
        )

    assert setup_module.TOKEN_PATH.read_text(encoding="utf-8") == original
    assert list(setup_module.TOKEN_PATH.parent.glob("google_token.tmp.*")) == []


def test_setup_refresh_repairs_existing_token_permissions(
    setup_module,
    monkeypatch,
):
    class RefreshingCredentials:
        expired = True
        refresh_token = "refresh-token"
        valid = False

        def refresh(self, request):
            self.expired = False
            self.valid = True

        def to_json(self):
            return FakeCredentials().to_json()

    class CredentialsFactory:
        @staticmethod
        def from_authorized_user_file(filename):
            assert filename == str(setup_module.TOKEN_PATH)
            return RefreshingCredentials()

    google_module = types.ModuleType("google")
    oauth2_module = types.ModuleType("google.oauth2")
    credentials_module = types.ModuleType("google.oauth2.credentials")
    credentials_module.Credentials = CredentialsFactory
    auth_module = types.ModuleType("google.auth")
    transport_module = types.ModuleType("google.auth.transport")
    requests_module = types.ModuleType("google.auth.transport.requests")
    requests_module.Request = lambda: object()

    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.oauth2", oauth2_module)
    monkeypatch.setitem(sys.modules, "google.oauth2.credentials", credentials_module)
    monkeypatch.setitem(sys.modules, "google.auth", auth_module)
    monkeypatch.setitem(sys.modules, "google.auth.transport", transport_module)
    monkeypatch.setitem(sys.modules, "google.auth.transport.requests", requests_module)

    setup_module.TOKEN_PATH.write_text('{"token": "old"}', encoding="utf-8")
    setup_module.TOKEN_PATH.chmod(0o644)

    assert setup_module.check_auth() is True
    assert setup_module.TOKEN_PATH.stat().st_mode & 0o777 == 0o600
