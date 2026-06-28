"""Tests that web_server.py OAuth error handlers redact sensitive text."""

import pytest
from unittest.mock import patch


def _redact_side_effect(text, force=False):
    """Fake redact that brackets the text to prove it was called."""
    return f"[REDACTED:{text}]"


class TestResolveProviderStatusRedact:
    """_resolve_provider_status() must route errors through redact_sensitive_text."""

    @patch("agent.redact.redact_sensitive_text", side_effect=_redact_side_effect)
    def test_status_fn_exception_redacted(self, mock_redact):
        from hermes_cli.web_server import _resolve_provider_status

        def bad_status():
            raise ValueError("token_expired: secret_abc123")

        result = _resolve_provider_status("test", bad_status)
        assert result["logged_in"] is False
        assert "[REDACTED:" in result["error"]
        mock_redact.assert_called()

    @patch("agent.redact.redact_sensitive_text", side_effect=_redact_side_effect)
    def test_outer_fallback_exception_redacted(self, mock_redact):
        from hermes_cli.web_server import _resolve_provider_status

        with patch("hermes_cli.web_server.hauth") as mock_hauth:
            mock_hauth.get_nous_auth_status.side_effect = ValueError("key=sk-abc123")
            result = _resolve_provider_status("nous", None)
            assert result["logged_in"] is False
            assert "[REDACTED:" in result["error"]
            mock_redact.assert_called()


class TestDeviceCodePollerRedact:
    """Device-code pollers must redact error_message in session state."""

    @patch("agent.redact.redact_sensitive_text", side_effect=_redact_side_effect)
    def test_nous_poller_error_redacted(self, mock_redact):
        from hermes_cli.web_server import _oauth_sessions, _oauth_sessions_lock, _nous_poller

        session_id = "test-redact-nous"
        with _oauth_sessions_lock:
            _oauth_sessions[session_id] = {
                "status": "pending",
                "provider": "nous",
                "poll_fn": lambda: None,
            }

        with patch("hermes_cli.web_server.nous_poll_device_code", side_effect=ValueError("device_code=dg_abc123")):
            _nous_poller(session_id)

        with _oauth_sessions_lock:
            sess = _oauth_sessions.get(session_id, {})
        assert sess.get("status") == "error"
        assert "dg_abc123" not in sess.get("error_message", "")
        mock_redact.assert_called()

    @patch("agent.redact.redact_sensitive_text", side_effect=_redact_side_effect)
    def test_minimax_poller_error_redacted(self, mock_redact):
        from hermes_cli.web_server import _oauth_sessions, _oauth_sessions_lock, _minimax_poller

        session_id = "test-redact-minimax"
        with _oauth_sessions_lock:
            _oauth_sessions[session_id] = {
                "status": "pending",
                "provider": "minimax-oauth",
                "code_verifier": "verifier123",
                "user_code": "code123",
                "region": "us",
            }

        with patch("hermes_cli.web_server._minimax_poll_token", side_effect=ValueError("api_key=minimax_key_abc")):
            _minimax_poller(session_id)

        with _oauth_sessions_lock:
            sess = _oauth_sessions.get(session_id, {})
        assert sess.get("status") == "error"
        assert "minimax_key_abc" not in sess.get("error_message", "")
        mock_redact.assert_called()

    @patch("agent.redact.redact_sensitive_text", side_effect=_redact_side_effect)
    def test_codex_worker_error_redacted(self, mock_redact):
        from hermes_cli.web_server import _oauth_sessions, _oauth_sessions_lock, _codex_full_login_worker

        session_id = "test-redact-codex"
        with _oauth_sessions_lock:
            _oauth_sessions[session_id] = {
                "status": "pending",
                "provider": "openai-codex",
            }

        with patch("hermes_cli.web_server.httpx") as mock_httpx:
            mock_httpx.Client.side_effect = ValueError("bearer_token=sk-codex-abc")
            _codex_full_login_worker(session_id)

        with _oauth_sessions_lock:
            s = _oauth_sessions.get(session_id, {})
        assert s.get("status") == "error"
        assert "sk-codex-abc" not in s.get("error_message", "")
        mock_redact.assert_called()
