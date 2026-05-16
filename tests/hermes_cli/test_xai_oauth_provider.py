"""Basic tests for the xAI OAuth provider."""

import json
from pathlib import Path

import pytest

from hermes_cli.auth import (
    _import_grok_cli_into_hermes,
    get_xai_oauth_auth_status,
    resolve_xai_oauth_runtime_credentials,
)


def test_get_xai_oauth_auth_status_no_credentials(tmp_path, monkeypatch):
    """Should return logged_in=False when no credentials exist."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    status = get_xai_oauth_auth_status()
    assert status["logged_in"] is False
    assert status["provider"] == "xai-oauth"


def test_read_grok_cli_auth(tmp_path, monkeypatch):
    """_read_grok_cli_auth should correctly parse a mock Grok CLI auth.json."""
    from hermes_cli.auth import _read_grok_cli_auth, GROK_CLI_AUTH_PATH

    # Point GROK_CLI_AUTH_PATH to temp directory
    fake_auth = tmp_path / ".grok" / "auth.json"
    fake_auth.parent.mkdir(parents=True)
    monkeypatch.setattr("hermes_cli.auth.GROK_CLI_AUTH_PATH", fake_auth)

    grok_data = {
        "https://auth.x.ai::b1a00492-073a-47ea-816f-4c329264a828": {
            "key": "eyJhbG...test",
            "refresh_token": "refresh-123",
            "expires_at": "2099-01-01T00:00:00Z",
            "email": "test@example.com",
            "auth_mode": "oidc",
        }
    }
    fake_auth.write_text(json.dumps(grok_data))

    creds = _read_grok_cli_auth()
    assert creds is not None
    assert creds["access_token"] == "eyJhbG...test"
    assert creds["refresh_token"] == "refresh-123"
    assert creds["email"] == "test@example.com"
    assert creds["expires_at"] == "2099-01-01T00:00:00Z"


def test_resolve_xai_oauth_no_creds_raises(tmp_path, monkeypatch):
    """Should raise a clear error when no credentials are available."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    with pytest.raises(Exception) as exc:
        resolve_xai_oauth_runtime_credentials()

    assert "Not logged into xAI via OAuth" in str(exc.value)