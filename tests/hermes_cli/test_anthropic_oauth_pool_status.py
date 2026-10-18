"""Anthropic card status must see a credential-pool OAuth login, not only its seed file.

Bug history (2026-10-18): an Anthropic OAuth login whose PKCE file
(``~/.hermes/.anthropic_oauth.json``) was removed — or whose file seeder got
suppressed — still lived on as a working ``credential_pool["anthropic"]`` row
(``source: manual:hermes_pkce``, HTTP 200 against ``/v1/models``). Every model
picker listed Anthropic as available while the Keys tab's "Accessi provider
(OAuth)" card said *not connected*, because the status probe only inspected the
PKCE file and ``ANTHROPIC_*`` env vars.

These tests pin the pool fallback and its exclusions: Claude Code rows belong to
their own card and never count here, and an expired token with no refresh path
is not a usable login.
"""
import os
import time
from unittest.mock import patch

import hermes_cli.web_server_oauth as _web_server_oauth

FUTURE_MS = int((time.time() + 6 * 3600) * 1000)
PAST_MS = int((time.time() - 3600) * 1000)
EMPTY_ENV = {"ANTHROPIC_API_KEY": "", "ANTHROPIC_TOKEN": "", "CLAUDE_CODE_OAUTH_TOKEN": ""}


def _status(pool_rows, *, pkce_file_creds=None):
    with (
        patch("agent.anthropic_credentials.read_hermes_oauth_credentials", return_value=pkce_file_creds),
        patch("hermes_cli.auth.read_credential_pool", return_value=pool_rows),
        patch.dict(os.environ, EMPTY_ENV),
    ):
        return _web_server_oauth._anthropic_oauth_status()


def test_pool_oauth_row_reports_connected():
    rows = [{
        "id": "15116e", "label": "anthropic-oauth-5", "auth_type": "oauth",
        "source": "manual:hermes_pkce", "access_token": "sk-ant-oat01-abcdef",
        "refresh_token": "sk-ant-ort01-ghijkl", "expires_at_ms": FUTURE_MS, "last_status": "ok",
    }]
    status = _status(rows)
    assert status["logged_in"] is True
    assert status["source"] == "credential_pool"
    assert "anthropic-oauth-5" in status["source_label"]
    assert status["has_refresh_token"] is True
    assert status["token_preview"].endswith("cdef")


def test_pkce_file_still_wins_over_pool():
    rows = [{"auth_type": "oauth", "source": "manual:hermes_pkce", "access_token": "pool-token",
             "refresh_token": "r", "expires_at_ms": FUTURE_MS}]
    status = _status(rows, pkce_file_creds={"accessToken": "file-token", "refreshToken": "r"})
    assert status["source"] == "hermes_pkce"


def test_claude_code_pool_row_does_not_mark_anthropic_connected():
    rows = [{"auth_type": "oauth", "source": "claude_code", "access_token": "cc-token",
             "refresh_token": "cc-refresh", "expires_at_ms": FUTURE_MS}]
    assert _status(rows)["logged_in"] is False


def test_expired_token_without_refresh_is_not_connected():
    rows = [{"auth_type": "oauth", "source": "manual:hermes_pkce", "access_token": "dead-token",
             "expires_at_ms": PAST_MS}]
    assert _status(rows)["logged_in"] is False


def test_expired_token_with_refresh_stays_connected():
    rows = [{"auth_type": "oauth", "source": "manual:hermes_pkce", "access_token": "old-token",
             "refresh_token": "refresh-me", "expires_at_ms": PAST_MS}]
    status = _status(rows)
    assert status["logged_in"] is True
    assert status["has_refresh_token"] is True


def test_api_key_row_in_pool_is_not_an_oauth_login():
    rows = [{"auth_type": "api_key", "source": "manual", "access_token": "sk-ant-api03-xyz"}]
    assert _status(rows)["logged_in"] is False
