"""Tests for Qwen OAuth provider authentication (hermes_cli/auth.py).

Covers: _qwen_cli_auth_path, _read_qwen_cli_tokens, _save_qwen_cli_tokens,
_qwen_access_token_is_expiring, _refresh_qwen_cli_tokens,
resolve_qwen_runtime_credentials, get_qwen_auth_status.
"""

import json
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
    get_qwen_auth_status_local,
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





# ---------------------------------------------------------------------------
# _save_qwen_cli_tokens
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# _qwen_access_token_is_expiring
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# _refresh_qwen_cli_tokens
# ---------------------------------------------------------------------------





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


def test_resolve_qwen_runtime_credentials_missing_access_token(qwen_env):
    tokens = _make_qwen_tokens(access_token="")
    _write_qwen_creds(qwen_env, tokens)

    with pytest.raises(AuthError) as exc:
        resolve_qwen_runtime_credentials(refresh_if_expiring=False)
    assert exc.value.code == "qwen_access_token_missing"


# ---------------------------------------------------------------------------
# get_qwen_auth_status
# ---------------------------------------------------------------------------

def test_get_qwen_auth_status_logged_in(qwen_env):
    tokens = _make_qwen_tokens(access_token="status-at")
    _write_qwen_creds(qwen_env, tokens)

    status = get_qwen_auth_status()
    assert status["logged_in"] is True
    assert status["api_key"] == "status-at"



def test_get_qwen_auth_status_local_does_not_refresh_expired_token(qwen_env):
    expired_ms = int((time.time() - 3600) * 1000)
    tokens = _make_qwen_tokens(access_token="old-at", expiry_date=expired_ms)
    _write_qwen_creds(qwen_env, tokens)

    with patch("hermes_cli.auth._refresh_qwen_cli_tokens") as mock_refresh:
        status = get_qwen_auth_status_local()

    mock_refresh.assert_not_called()
    assert status["logged_in"] is False
    assert "expired" in status["error"]


def test_get_qwen_auth_status_local_reports_fresh_token(qwen_env):
    expires_ms = int((time.time() + 3600) * 1000)
    tokens = _make_qwen_tokens(access_token="local-at", expiry_date=expires_ms)
    _write_qwen_creds(qwen_env, tokens)

    status = get_qwen_auth_status_local()

    assert status["logged_in"] is True
    assert status["source"] == "qwen-cli"
    assert status["expires_at_ms"] == expires_ms

def test_get_qwen_auth_status_refreshes_expired_token(qwen_env):
    expired_ms = int((time.time() - 3600) * 1000)
    tokens = _make_qwen_tokens(access_token="old-at", expiry_date=expired_ms)
    _write_qwen_creds(qwen_env, tokens)

    refreshed = _make_qwen_tokens(access_token="refreshed-at")

    with patch(
        "hermes_cli.auth._refresh_qwen_cli_tokens", return_value=refreshed
    ) as mock_refresh:
        status = get_qwen_auth_status()

    mock_refresh.assert_called_once()
    assert status["logged_in"] is True
    assert status["api_key"] == "refreshed-at"


def test_model_flow_qwen_oauth_stale_token_shows_reauth_guidance(qwen_env, monkeypatch, capsys):
    from hermes_cli.main import _model_flow_qwen_oauth

    expired_ms = int((time.time() - 3600) * 1000)
    tokens = _make_qwen_tokens(access_token="dead-at", expiry_date=expired_ms)
    _write_qwen_creds(qwen_env, tokens)

    monkeypatch.setattr(
        "hermes_cli.auth._refresh_qwen_cli_tokens",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AuthError(
                "Qwen refresh rejected. Re-run 'qwen auth qwen-oauth'.",
                provider="qwen-oauth",
                code="qwen_refresh_failed",
            )
        ),
    )

    prompt_called = {"value": False}
    update_called = {"value": False}

    monkeypatch.setattr(
        "hermes_cli.auth._prompt_model_selection",
        lambda *args, **kwargs: prompt_called.__setitem__("value", True),
    )
    monkeypatch.setattr(
        "hermes_cli.auth._update_config_for_provider",
        lambda *args, **kwargs: update_called.__setitem__("value", True),
    )

    _model_flow_qwen_oauth({}, current_model="qwen3-coder-plus")

    out = capsys.readouterr().out
    assert "Run: qwen auth qwen-oauth" in out
    assert "Qwen refresh rejected" in out
    assert prompt_called["value"] is False
    assert update_called["value"] is False
