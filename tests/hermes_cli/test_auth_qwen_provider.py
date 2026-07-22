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


def test_get_qwen_auth_status_expired_unrefreshable_token_is_not_logged_in(qwen_env):
    expired_ms = int((time.time() - 3600) * 1000)
    tokens = _make_qwen_tokens(access_token="dead-at", expiry_date=expired_ms)
    _write_qwen_creds(qwen_env, tokens)

    with patch(
        "hermes_cli.auth._refresh_qwen_cli_tokens",
        side_effect=AuthError(
            "Qwen OAuth refresh failed. The `qwen auth` CLI subcommand was "
            "removed in Qwen CLI 0.19.x — run `qwen` interactively and use "
            "`/auth` to re-authenticate, or update ~/.qwen/oauth_creds.json.",
            provider="qwen-oauth",
            code="qwen_refresh_failed",
        ),
    ) as mock_refresh:
        status = get_qwen_auth_status()

    mock_refresh.assert_called_once()
    assert status["logged_in"] is False
    assert "0.19.x" in status["error"]
    assert "/auth" in status["error"]
    # Old, removed subcommand must NOT appear in fresh guidance:
    assert "qwen auth qwen-oauth" not in status["error"]


def test_get_qwen_auth_status_not_logged_in(qwen_env):
    # No credentials file
    status = get_qwen_auth_status()
    assert status["logged_in"] is False
    assert "error" in status


def test_model_flow_qwen_oauth_stale_token_shows_reauth_guidance(qwen_env, monkeypatch, capsys):
    from hermes_cli.main import _model_flow_qwen_oauth

    expired_ms = int((time.time() - 3600) * 1000)
    tokens = _make_qwen_tokens(access_token="dead-at", expiry_date=expired_ms)
    _write_qwen_creds(qwen_env, tokens)

    monkeypatch.setattr(
        "hermes_cli.auth._refresh_qwen_cli_tokens",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AuthError(
                "Qwen OAuth refresh failed. The `qwen auth` CLI subcommand "
                "was removed in Qwen CLI 0.19.x — run `qwen` interactively "
                "and use `/auth` to re-authenticate.",
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
    # Must point users at the supported replacement path:
    assert "Run: qwen" in out
    assert "/auth" in out
    # Must NOT advertise the removed `qwen auth` subcommand:
    assert "qwen auth qwen-oauth" not in out
    # Underlying refresh error should be surfaced too:
    assert "Qwen OAuth refresh failed" in out
    assert prompt_called["value"] is False
    assert update_called["value"] is False


# ---------------------------------------------------------------------------
# Guidance strings — Qwen CLI 0.19.x removed `qwen auth`
# ---------------------------------------------------------------------------
#
# These tests pin the user-facing guidance produced by the Qwen OAuth code
# paths against the current installed Qwen CLI (0.19.x). The
# `qwen auth qwen-oauth` subcommand was removed upstream, so every error
# and prompt must point at `qwen` + `/auth` (or manual settings.json
# editing) instead. See:
#   https://qwenlm.github.io/qwen-code-docs/en/users/configuration/auth/


def test_missing_credentials_message_points_at_qwen_and_slash_auth(qwen_env):
    """_read_qwen_cli_tokens() must advertise the supported login path."""
    with pytest.raises(AuthError) as exc:
        _read_qwen_cli_tokens()
    assert exc.value.code == "qwen_auth_missing"
    msg = str(exc.value)
    # The interactive replacement the Qwen CLI docs prescribe:
    assert "0.19.x" in msg
    assert "/auth" in msg
    assert "qwen" in msg
    # The removed subcommand must NOT appear:
    assert "qwen auth qwen-oauth" not in msg
    # The credentials file path must still be mentioned so users can fix it manually:
    assert "oauth_creds.json" in msg or "settings.json" in msg


def test_refresh_token_missing_message_points_at_qwen_and_slash_auth(qwen_env):
    """Missing refresh token in oauth_creds.json must steer users to /auth."""
    # Write a token file with NO refresh_token field.
    tokens = _make_qwen_tokens(access_token="at", refresh_token="")
    _write_qwen_creds(qwen_env, tokens)

    with pytest.raises(AuthError) as exc:
        _refresh_qwen_cli_tokens(tokens)
    assert exc.value.code == "qwen_refresh_token_missing"
    msg = str(exc.value)
    assert "0.19.x" in msg
    assert "/auth" in msg
    assert "qwen auth qwen-oauth" not in msg


def test_refresh_failed_message_mentions_removed_subcommand_and_replacement(qwen_env):
    """A refresh 4xx response must explain the replacement, not the removed cmd."""
    expired_ms = int((time.time() - 3600) * 1000)
    tokens = _make_qwen_tokens(
        access_token="dead-at",
        refresh_token="rt",
        expiry_date=expired_ms,
    )
    _write_qwen_creds(qwen_env, tokens)

    fake_response = MagicMock()
    fake_response.status_code = 400
    fake_response.text = "invalid_grant"

    with patch("hermes_cli.auth.httpx.post", return_value=fake_response):
        with pytest.raises(AuthError) as exc:
            _refresh_qwen_cli_tokens(tokens)
    assert exc.value.code == "qwen_refresh_failed"
    msg = str(exc.value)
    assert "0.19.x" in msg
    assert "/auth" in msg
    assert "oauth_creds.json" in msg
    assert "qwen auth qwen-oauth" not in msg
    # The upstream response body should still be surfaced for debuggability:
    assert "invalid_grant" in msg


def test_access_token_missing_message_points_at_qwen_and_slash_auth(qwen_env, monkeypatch):
    """resolve_qwen_runtime_credentials must explain /auth when access is empty."""
    tokens = _make_qwen_tokens(access_token="", refresh_token="rt")
    _write_qwen_creds(qwen_env, tokens)

    with pytest.raises(AuthError) as exc:
        resolve_qwen_runtime_credentials(refresh_if_expiring=False)
    assert exc.value.code == "qwen_access_token_missing"
    msg = str(exc.value)
    assert "0.19.x" in msg
    assert "/auth" in msg
    assert "qwen auth qwen-oauth" not in msg


def test_model_flow_qwen_oauth_not_logged_in_guidance(qwen_env, monkeypatch, capsys):
    """`hermes model` → qwen-oauth when not logged in must show /auth guidance."""
    from hermes_cli.main import _model_flow_qwen_oauth

    # No credentials file at all → get_qwen_auth_status() returns logged_in=False.
    qwen_status = get_qwen_auth_status()
    assert qwen_status.get("logged_in") is False
    assert "error" in qwen_status

    prompt_called = {"value": False}
    update_called = {"value": False}
    monkeypatch.setattr(
        "hermes_cli.auth._prompt_model_selection",
        lambda *a, **kw: prompt_called.__setitem__("value", True),
    )
    monkeypatch.setattr(
        "hermes_cli.auth._update_config_for_provider",
        lambda *a, **kw: update_called.__setitem__("value", True),
    )

    _model_flow_qwen_oauth({}, current_model="qwen3-coder-plus")
    out = capsys.readouterr().out
    assert "Not logged into Qwen CLI OAuth" in out
    assert "Run: qwen" in out
    assert "/auth" in out
    # Still tells the user where to write tokens manually:
    assert "oauth_creds.json" in out or "settings.json" in out
    # Never the removed subcommand:
    assert "qwen auth qwen-oauth" not in out
    # And we must NOT have proceeded to model selection / config update:
    assert prompt_called["value"] is False
    assert update_called["value"] is False


def test_status_command_qwen_row_uses_slash_auth_guidance(qwen_env, capsys):
    """`hermes status` Qwen OAuth row must point users at /auth, not `qwen auth`.

    The Qwen status row is a literal f-string in hermes_cli/status.py —
    assert against the new message directly so the regression cannot sneak
    back in via a refactor.
    """
    # No credentials file → not logged in. get_qwen_auth_status is what the
    # status code uses, and it now must report an error so the row reads
    # "not logged in (run: qwen, then /auth — `qwen auth` was removed in 0.19.x)".
    qwen_status = get_qwen_auth_status()
    assert qwen_status.get("logged_in") is False

    # Reproduce the literal format used in hermes_cli/status.py:282 so a
    # future refactor that breaks the wording fails this test.
    expected = (
        "not logged in (run: qwen, then /auth — `qwen auth` was removed in 0.19.x)"
    )
    rendered = (
        f"  {'Qwen OAuth':<12}  {'not logged in (run: qwen, then /auth — `qwen auth` was removed in 0.19.x)'}"
    )
    # The actual format used by the row:
    assert expected in rendered
    # Make sure the OLD guidance is gone from the source file (no regression):
    src = Path("hermes_cli/status.py").read_text(encoding="utf-8")
    assert "run: qwen auth qwen-oauth" not in src
    # And the NEW guidance is present:
    assert "run: qwen, then /auth" in src


def test_qwen_cli_remove_hints_use_supported_replacement(qwen_env):
    """The qwen-cli suppression hint must NOT advertise the removed subcommand."""
    from agent.credential_sources import _remove_qwen_cli

    result = _remove_qwen_cli("qwen-oauth", removed=None)
    joined = "\n".join(result.hints)
    assert "Suppressed qwen-cli credential" in joined
    assert "oauth_creds.json" in joined
    # New guidance:
    assert "/auth" in joined
    assert "settings.json" in joined
    # Removed guidance must be gone:
    assert "hermes auth add qwen-oauth" not in joined
