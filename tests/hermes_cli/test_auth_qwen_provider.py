"""Tests for Qwen OAuth provider authentication (hermes_cli/auth.py).

Covers: _qwen_cli_auth_path, _read_qwen_cli_tokens, _save_qwen_cli_tokens,
_qwen_access_token_is_expiring, _refresh_qwen_cli_tokens,
resolve_qwen_runtime_credentials, get_qwen_auth_status.
"""

import json
import os
import stat
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.auth import (
    AuthError,
    DEFAULT_QWEN_BASE_URL,
    QWEN_ACCESS_TOKEN_REFRESH_SKEW_SECONDS,
    _qwen_cli_auth_path,
    _read_qwen_cli_tokens,
    _save_qwen_cli_tokens,
    _qwen_access_token_is_expiring,
    _refresh_qwen_cli_tokens,
    resolve_qwen_runtime_credentials,
    get_qwen_auth_status,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_qwen_tokens(
    access_token="test-access-token",
    refresh_token="test-refresh-token",
    expiry_date=None,
    **extra,
):
    """Create a minimal Qwen CLI OAuth credential dict."""
    if expiry_date is None:
        # 1 hour from now in milliseconds
        expiry_date = int((time.time() + 3600) * 1000)
    data = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "Bearer",
        "expiry_date": expiry_date,
        "resource_url": "portal.qwen.ai",
    }
    data.update(extra)
    return data


def _write_qwen_creds(tmp_path, tokens=None):
    """Write tokens to the Qwen CLI credentials file and return the path."""
    qwen_dir = tmp_path / ".qwen"
    qwen_dir.mkdir(parents=True, exist_ok=True)
    creds_path = qwen_dir / "oauth_creds.json"
    if tokens is None:
        tokens = _make_qwen_tokens()
    creds_path.write_text(json.dumps(tokens), encoding="utf-8")
    return creds_path


@pytest.fixture()
def qwen_env(tmp_path, monkeypatch):
    """Redirect _qwen_cli_auth_path to tmp_path/.qwen/oauth_creds.json."""
    creds_path = tmp_path / ".qwen" / "oauth_creds.json"
    monkeypatch.setattr(
        "hermes_cli.auth._qwen_cli_auth_path", lambda: creds_path
    )
    return tmp_path


# ---------------------------------------------------------------------------
# _qwen_cli_auth_path
# ---------------------------------------------------------------------------

def test_qwen_cli_auth_path_returns_expected_location():
    path = _qwen_cli_auth_path()
    assert path == Path.home() / ".qwen" / "oauth_creds.json"


# ---------------------------------------------------------------------------
# _read_qwen_cli_tokens
# ---------------------------------------------------------------------------

def test_read_qwen_cli_tokens_success(qwen_env):
    tokens = _make_qwen_tokens(access_token="my-access")
    _write_qwen_creds(qwen_env, tokens)
    result = _read_qwen_cli_tokens()
    assert result["access_token"] == "my-access"
    assert result["refresh_token"] == "test-refresh-token"


def test_read_qwen_cli_tokens_missing_file(qwen_env):
    with pytest.raises(AuthError) as exc:
        _read_qwen_cli_tokens()
    assert exc.value.code == "qwen_auth_missing"


def test_read_qwen_cli_tokens_invalid_json(qwen_env):
    creds_path = qwen_env / ".qwen" / "oauth_creds.json"
    creds_path.parent.mkdir(parents=True, exist_ok=True)
    creds_path.write_text("not json{{{", encoding="utf-8")
    with pytest.raises(AuthError) as exc:
        _read_qwen_cli_tokens()
    assert exc.value.code == "qwen_auth_read_failed"


def test_read_qwen_cli_tokens_non_dict(qwen_env):
    creds_path = qwen_env / ".qwen" / "oauth_creds.json"
    creds_path.parent.mkdir(parents=True, exist_ok=True)
    creds_path.write_text(json.dumps(["a", "b"]), encoding="utf-8")
    with pytest.raises(AuthError) as exc:
        _read_qwen_cli_tokens()
    assert exc.value.code == "qwen_auth_invalid"


# ---------------------------------------------------------------------------
# _save_qwen_cli_tokens
# ---------------------------------------------------------------------------

def test_save_qwen_cli_tokens_roundtrip(qwen_env):
    tokens = _make_qwen_tokens(access_token="saved-token")
    saved_path = _save_qwen_cli_tokens(tokens)
    assert saved_path.exists()
    loaded = json.loads(saved_path.read_text(encoding="utf-8"))
    assert loaded["access_token"] == "saved-token"


def test_save_qwen_cli_tokens_creates_parent(qwen_env):
    tokens = _make_qwen_tokens()
    saved_path = _save_qwen_cli_tokens(tokens)
    assert saved_path.parent.exists()


def test_save_qwen_cli_tokens_permissions(qwen_env):
    tokens = _make_qwen_tokens()
    saved_path = _save_qwen_cli_tokens(tokens)
    mode = saved_path.stat().st_mode
    assert mode & stat.S_IRUSR  # owner read
    assert mode & stat.S_IWUSR  # owner write
    assert not (mode & stat.S_IRGRP)  # no group read
    assert not (mode & stat.S_IROTH)  # no other read


# ---------------------------------------------------------------------------
# _qwen_access_token_is_expiring
# ---------------------------------------------------------------------------

def test_expiring_token_not_expired():
    # 1 hour from now in milliseconds
    future_ms = int((time.time() + 3600) * 1000)
    assert not _qwen_access_token_is_expiring(future_ms)


def test_expiring_token_already_expired():
    # 1 hour ago in milliseconds
    past_ms = int((time.time() - 3600) * 1000)
    assert _qwen_access_token_is_expiring(past_ms)


def test_expiring_token_within_skew():
    # Just inside the default skew window
    near_ms = int((time.time() + QWEN_ACCESS_TOKEN_REFRESH_SKEW_SECONDS - 5) * 1000)
    assert _qwen_access_token_is_expiring(near_ms)


def test_expiring_token_none_returns_true():
    assert _qwen_access_token_is_expiring(None)


def test_expiring_token_non_numeric_returns_true():
    assert _qwen_access_token_is_expiring("not-a-number")


# ---------------------------------------------------------------------------
# _refresh_qwen_cli_tokens
# ---------------------------------------------------------------------------

def test_refresh_qwen_cli_tokens_success(qwen_env):
    tokens = _make_qwen_tokens(refresh_token="old-refresh")

    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "access_token": "new-access",
        "refresh_token": "new-refresh",
        "expires_in": 7200,
    }

    with patch("hermes_cli.auth.httpx") as mock_httpx:
        mock_httpx.post.return_value = resp
        result = _refresh_qwen_cli_tokens(tokens)

    assert result["access_token"] == "new-access"
    assert result["refresh_token"] == "new-refresh"
    assert "expiry_date" in result


def test_refresh_qwen_cli_tokens_preserves_old_refresh_if_not_in_response(qwen_env):
    tokens = _make_qwen_tokens(refresh_token="keep-me")

    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "access_token": "new-access",
        # No refresh_token in response — should keep old one
        "expires_in": 3600,
    }

    with patch("hermes_cli.auth.httpx") as mock_httpx:
        mock_httpx.post.return_value = resp
        result = _refresh_qwen_cli_tokens(tokens)

    assert result["refresh_token"] == "keep-me"


def test_refresh_qwen_cli_tokens_missing_refresh_token():
    tokens = {"access_token": "at", "refresh_token": ""}
    with pytest.raises(AuthError) as exc:
        _refresh_qwen_cli_tokens(tokens)
    assert exc.value.code == "qwen_refresh_token_missing"


def test_refresh_qwen_cli_tokens_http_error(qwen_env):
    tokens = _make_qwen_tokens()

    resp = MagicMock()
    resp.status_code = 401
    resp.text = "unauthorized"

    with patch("hermes_cli.auth.httpx") as mock_httpx:
        mock_httpx.post.return_value = resp
        with pytest.raises(AuthError) as exc:
            _refresh_qwen_cli_tokens(tokens)
    assert exc.value.code == "qwen_refresh_failed"


def test_refresh_qwen_cli_tokens_network_error(qwen_env):
    tokens = _make_qwen_tokens()

    with patch("hermes_cli.auth.httpx") as mock_httpx:
        mock_httpx.post.side_effect = ConnectionError("timeout")
        with pytest.raises(AuthError) as exc:
            _refresh_qwen_cli_tokens(tokens)
    assert exc.value.code == "qwen_refresh_failed"


def test_refresh_qwen_cli_tokens_invalid_json_response(qwen_env):
    tokens = _make_qwen_tokens()

    resp = MagicMock()
    resp.status_code = 200
    resp.json.side_effect = ValueError("bad json")

    with patch("hermes_cli.auth.httpx") as mock_httpx:
        mock_httpx.post.return_value = resp
        with pytest.raises(AuthError) as exc:
            _refresh_qwen_cli_tokens(tokens)
    assert exc.value.code == "qwen_refresh_invalid_json"


def test_refresh_qwen_cli_tokens_missing_access_token_in_response(qwen_env):
    tokens = _make_qwen_tokens()

    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"something": "but no access_token"}

    with patch("hermes_cli.auth.httpx") as mock_httpx:
        mock_httpx.post.return_value = resp
        with pytest.raises(AuthError) as exc:
            _refresh_qwen_cli_tokens(tokens)
    assert exc.value.code == "qwen_refresh_invalid_response"


def test_refresh_qwen_cli_tokens_default_expires_in(qwen_env):
    """When expires_in is missing, default to 6 hours."""
    tokens = _make_qwen_tokens()

    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"access_token": "new"}

    with patch("hermes_cli.auth.httpx") as mock_httpx:
        mock_httpx.post.return_value = resp
        result = _refresh_qwen_cli_tokens(tokens)

    # Verify expiry_date is roughly now + 6h (within 60s tolerance)
    expected_ms = int(time.time() * 1000) + 6 * 60 * 60 * 1000
    assert abs(result["expiry_date"] - expected_ms) < 60_000


def test_refresh_qwen_cli_tokens_saves_to_disk(qwen_env):
    tokens = _make_qwen_tokens()

    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "access_token": "disk-check",
        "expires_in": 3600,
    }

    with patch("hermes_cli.auth.httpx") as mock_httpx:
        mock_httpx.post.return_value = resp
        _refresh_qwen_cli_tokens(tokens)

    # Verify it was persisted
    creds_path = qwen_env / ".qwen" / "oauth_creds.json"
    assert creds_path.exists()
    saved = json.loads(creds_path.read_text(encoding="utf-8"))
    assert saved["access_token"] == "disk-check"


# ---------------------------------------------------------------------------
# resolve_qwen_runtime_credentials
# ---------------------------------------------------------------------------

def test_resolve_qwen_runtime_credentials_fresh_token(qwen_env):
    tokens = _make_qwen_tokens(access_token="fresh-at")
    _write_qwen_creds(qwen_env, tokens)

    creds = resolve_qwen_runtime_credentials(refresh_if_expiring=False)
    assert creds["provider"] == "qwen-oauth"
    assert creds["api_key"] == "fresh-at"
    assert creds["base_url"] == DEFAULT_QWEN_BASE_URL
    assert creds["source"] == "qwen-cli"


def test_resolve_qwen_runtime_credentials_triggers_refresh(qwen_env):
    # Write an expired token
    expired_ms = int((time.time() - 3600) * 1000)
    tokens = _make_qwen_tokens(access_token="old", expiry_date=expired_ms)
    _write_qwen_creds(qwen_env, tokens)

    refreshed = _make_qwen_tokens(access_token="refreshed-at")

    with patch(
        "hermes_cli.auth._refresh_qwen_cli_tokens", return_value=refreshed
    ) as mock_refresh:
        creds = resolve_qwen_runtime_credentials()
    mock_refresh.assert_called_once()
    assert creds["api_key"] == "refreshed-at"


def test_resolve_qwen_runtime_credentials_force_refresh(qwen_env):
    tokens = _make_qwen_tokens(access_token="old-at")
    _write_qwen_creds(qwen_env, tokens)

    refreshed = _make_qwen_tokens(access_token="force-refreshed")

    with patch(
        "hermes_cli.auth._refresh_qwen_cli_tokens", return_value=refreshed
    ) as mock_refresh:
        creds = resolve_qwen_runtime_credentials(force_refresh=True)
    mock_refresh.assert_called_once()
    assert creds["api_key"] == "force-refreshed"


def test_resolve_qwen_runtime_credentials_missing_access_token(qwen_env):
    tokens = _make_qwen_tokens(access_token="")
    _write_qwen_creds(qwen_env, tokens)

    with pytest.raises(AuthError) as exc:
        resolve_qwen_runtime_credentials(refresh_if_expiring=False)
    assert exc.value.code == "qwen_access_token_missing"


def test_resolve_qwen_runtime_credentials_base_url_env_override(qwen_env, monkeypatch):
    tokens = _make_qwen_tokens(access_token="at")
    _write_qwen_creds(qwen_env, tokens)
    monkeypatch.setenv("HERMES_QWEN_BASE_URL", "https://custom.qwen.ai/v1")

    creds = resolve_qwen_runtime_credentials(refresh_if_expiring=False)
    assert creds["base_url"] == "https://custom.qwen.ai/v1"


# ---------------------------------------------------------------------------
# get_qwen_auth_status
# ---------------------------------------------------------------------------

def test_get_qwen_auth_status_logged_in(qwen_env):
    tokens = _make_qwen_tokens(access_token="status-at")
    _write_qwen_creds(qwen_env, tokens)

    status = get_qwen_auth_status()
    assert status["logged_in"] is True
    assert status["api_key"] == "status-at"


def test_get_qwen_auth_status_not_logged_in(qwen_env):
    # No credentials file
    status = get_qwen_auth_status()
    assert status["logged_in"] is False
    assert "error" in status
