"""Dashboard OAuth error handlers must not echo credentials."""

from __future__ import annotations

from unittest.mock import patch
import time

import pytest
from fastapi import HTTPException


SECRET = "sk-oauthsecretvalue1234567890"


def test_safe_oauth_error_message_redacts_credentials():
    from hermes_cli.web_server import _safe_oauth_error_message

    message = _safe_oauth_error_message(
        RuntimeError(f"provider failed: access_token={SECRET}")
    )

    assert SECRET not in message
    assert "access_token=***" in message


def test_safe_oauth_error_message_fails_closed(monkeypatch):
    from hermes_cli import web_server

    def broken_redactor(*_args, **_kwargs):
        raise RuntimeError("redactor unavailable")

    monkeypatch.setattr("agent.redact.redact_sensitive_text", broken_redactor)

    message = web_server._safe_oauth_error_message(
        RuntimeError(f"provider failed: access_token={SECRET}")
    )

    assert message == "OAuth provider error"
    assert SECRET not in message


def test_resolve_provider_status_redacts_status_fn_exception():
    from hermes_cli.web_server import _resolve_provider_status

    def bad_status():
        raise ValueError(f"token expired: api_key={SECRET}")

    result = _resolve_provider_status("test", bad_status)

    assert result["logged_in"] is False
    assert SECRET not in result["error"]
    assert "api_key=***" in result["error"]


def test_resolve_provider_status_redacts_dispatch_exception(monkeypatch):
    from hermes_cli.web_server import _resolve_provider_status

    monkeypatch.setattr(
        "hermes_cli.auth.get_nous_auth_status",
        lambda: (_ for _ in ()).throw(ValueError(f"key={SECRET}")),
    )

    result = _resolve_provider_status("nous", None)

    assert result["logged_in"] is False
    assert SECRET not in result["error"]
    assert "key=***" in result["error"]


def test_nous_poller_stores_redacted_error_message():
    from hermes_cli.web_server import _oauth_sessions, _oauth_sessions_lock, _nous_poller

    session_id = "test-redact-nous"
    with _oauth_sessions_lock:
        _oauth_sessions[session_id] = {
            "status": "pending",
            "provider": "nous",
            "poll_fn": lambda: None,
            "created_at": time.time(),
            "portal_base_url": "https://portal.example.test",
            "client_id": "client-id",
            "device_code": "device-code",
            "interval": 1,
            "expires_at": time.time() + 600,
        }

    with patch(
        "hermes_cli.auth._poll_for_token",
        side_effect=ValueError(f"access_token={SECRET}"),
    ):
        _nous_poller(session_id)

    with _oauth_sessions_lock:
        sess = _oauth_sessions.pop(session_id, {})

    assert sess.get("status") == "error"
    assert SECRET not in sess.get("error_message", "")
    assert "access_token=***" in sess.get("error_message", "")


@pytest.mark.anyio
async def test_start_oauth_login_redacts_http_exception_detail(monkeypatch):
    from hermes_cli.web_server import start_oauth_login

    monkeypatch.setattr(
        "hermes_cli.web_server._require_token",
        lambda request: None,
    )
    monkeypatch.setattr(
        "hermes_cli.web_server._gc_oauth_sessions",
        lambda: None,
    )
    async def broken_start(*_args, **_kwargs):
        raise ValueError(f"refresh_token={SECRET}")

    monkeypatch.setattr(
        "hermes_cli.web_server._start_device_code_flow",
        broken_start,
    )

    with pytest.raises(HTTPException) as exc:
        await start_oauth_login("nous", request=None)

    assert exc.value.status_code == 500
    assert SECRET not in str(exc.value.detail)
    assert "refresh_token=***" in str(exc.value.detail)
