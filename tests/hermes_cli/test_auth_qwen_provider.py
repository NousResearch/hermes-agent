"""Tests for Qwen OAuth provider authentication (hermes_cli/auth.py).

Covers: _qwen_cli_auth_path, _read_qwen_cli_tokens, _save_qwen_cli_tokens,
_qwen_access_token_is_expiring, _refresh_qwen_cli_tokens,
resolve_qwen_runtime_credentials, get_qwen_auth_status.
"""

import json
import time
from unittest.mock import patch

import pytest

from hermes_cli.auth import (
    AuthError,
    DEFAULT_QWEN_BASE_URL,
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


def _make_fake_qwen_response(*, status_code: int, text: str, content_type: str = "text/html"):
    """httpx.Response is hard to instantiate directly; assemble a MagicMock that quacks like one."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.headers = {"content-type": content_type}
    resp.json.side_effect = json.JSONDecodeError("Expecting value", text, 0) if not text.lstrip().startswith("{") else None
    if not text.lstrip().startswith("{"):
        # Make ``response.json()`` actually raise (MagicMock would otherwise return a Mock object).
        def _raise():
            raise json.JSONDecodeError("Expecting value", text, 0)
        resp.json.side_effect = _raise
    else:
        resp.json.return_value = json.loads(text)
    return resp


def test_refresh_qwen_cli_tokens_sends_qwen_code_user_agent(qwen_env):
    """Match the Qwen-style ``User-Agent`` that qwen-code-api sends during refresh;
    without it the OAuth endpoint returns a non-JSON body and the next ``response.json()``
    raises "Expecting value" (#7746)."""
    tokens = _make_qwen_tokens()
    body = json.dumps({"access_token": "new-at", "expires_in": 3600})
    fake = _make_fake_qwen_response(status_code=200, text=body, content_type="application/json")

    with (
        patch("hermes_cli.auth.httpx.post", return_value=fake) as mock_post,
        patch("hermes_cli.auth._save_qwen_cli_tokens"),
    ):
        _refresh_qwen_cli_tokens(tokens)

    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["headers"]["User-Agent"] == "QwenCode/0.14.0 (linux; x64)", (
        "Qwen refresh must send the QwenCode User-Agent so the OAuth endpoint returns JSON (#7746)"
    )


def test_refresh_qwen_cli_tokens_invalid_json_includes_diagnostics(qwen_env):
    """The original error message hid status/content-type/body, so a refresh that came back
    as HTML (WAF) or empty (server hung up) looked identical to bad-JSON. Surface the
    diagnostics in the AuthError so the user can tell them apart (#7746)."""
    tokens = _make_qwen_tokens()
    html_body = "<html><body>Login required</body></html>" * 20  # > 500 chars to test the preview cap
    fake = _make_fake_qwen_response(status_code=200, text=html_body, content_type="text/html; charset=utf-8")

    with (
        patch("hermes_cli.auth.httpx.post", return_value=fake),
        patch("hermes_cli.auth._save_qwen_cli_tokens"),
    ):
        with pytest.raises(AuthError) as exc:
            _refresh_qwen_cli_tokens(tokens)

    assert exc.value.code == "qwen_refresh_invalid_json"
    msg = str(exc.value)
    assert "status=200" in msg, f"diagnostic must include status code: {msg!r}"
    assert "content_type=" in msg, f"diagnostic must include content-type: {msg!r}"
    assert "text/html" in msg, f"content-type value should appear: {msg!r}"
    assert "body_preview=" in msg, f"diagnostic must include body preview: {msg!r}"
    assert "<html>" in msg, f"body preview should contain the actual body text: {msg!r}"


def test_refresh_qwen_cli_tokens_invalid_json_empty_body(qwen_env):
    """Counter-case: empty body is the other common shape of "invalid JSON"; the
    diagnostic must report the empty body preview without crashing on it (#7746)."""
    tokens = _make_qwen_tokens()
    fake = _make_fake_qwen_response(status_code=200, text="", content_type="application/json")

    with (
        patch("hermes_cli.auth.httpx.post", return_value=fake),
        patch("hermes_cli.auth._save_qwen_cli_tokens"),
    ):
        with pytest.raises(AuthError) as exc:
            _refresh_qwen_cli_tokens(tokens)

    assert exc.value.code == "qwen_refresh_invalid_json"
    assert "body_preview=''" in str(exc.value), (
        f"empty body should appear as an empty quoted preview: {str(exc.value)!r}"
    )





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
    from hermes_cli.model_setup_flows import _model_flow_qwen_oauth

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
    assert "Qwen refresh rejected" in out
    assert prompt_called["value"] is False
    assert update_called["value"] is False
