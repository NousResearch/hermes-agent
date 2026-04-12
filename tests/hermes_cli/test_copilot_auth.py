"""Tests for hermes_cli.copilot_auth — Copilot token validation, exchange, and resolution."""

import json
import os
import time
import pytest
from unittest.mock import patch, MagicMock


FAKE_SESSION_TOKEN = "tid=abc123;exp=9999999999;sku=copilot_enterprise_seat_quota;proxy-ep=proxy.enterprise.githubcopilot.com"
FAKE_EXPIRES_AT = time.time() + 3600


def _mock_exchange_success(github_token):
    from hermes_cli.copilot_auth import CopilotSessionToken

    return CopilotSessionToken(
        token=FAKE_SESSION_TOKEN,
        expires_at=FAKE_EXPIRES_AT,
        base_url="https://api.enterprise.githubcopilot.com",
        source="fetched:mock",
    )


class TestTokenValidation:
    """Token type validation."""

    def test_classic_pat_rejected(self):
        from hermes_cli.copilot_auth import validate_copilot_token

        valid, msg = validate_copilot_token("ghp_abcdefghijklmnop1234")
        assert valid is False
        assert "Classic Personal Access Tokens" in msg
        assert "ghp_" in msg

    def test_oauth_token_accepted(self):
        from hermes_cli.copilot_auth import validate_copilot_token

        valid, msg = validate_copilot_token("gho_abcdefghijklmnop1234")
        assert valid is True

    def test_fine_grained_pat_accepted(self):
        from hermes_cli.copilot_auth import validate_copilot_token

        valid, msg = validate_copilot_token("github_pat_abcdefghijklmnop1234")
        assert valid is True

    def test_github_app_token_accepted(self):
        from hermes_cli.copilot_auth import validate_copilot_token

        valid, msg = validate_copilot_token("ghu_abcdefghijklmnop1234")
        assert valid is True

    def test_empty_token_rejected(self):
        from hermes_cli.copilot_auth import validate_copilot_token

        valid, msg = validate_copilot_token("")
        assert valid is False


class TestNormalizeExpiry:
    """Expiry timestamp normalization (ms -> seconds)."""

    def test_seconds_unchanged(self):
        from hermes_cli.copilot_auth import _normalize_expiry

        ts = 1700000000.0
        assert _normalize_expiry(ts) == ts

    def test_milliseconds_converted(self):
        from hermes_cli.copilot_auth import _normalize_expiry

        ts_ms = 1700000000000.0
        assert _normalize_expiry(ts_ms) == 1700000000.0


class TestDeriveBaseUrl:
    """Base URL derivation from session token proxy-ep field."""

    def test_enterprise_proxy_ep(self):
        from hermes_cli.copilot_auth import _derive_base_url_from_token

        token = "tid=abc;proxy-ep=proxy.enterprise.githubcopilot.com;exp=999"
        assert (
            _derive_base_url_from_token(token)
            == "https://api.enterprise.githubcopilot.com"
        )

    def test_individual_proxy_ep(self):
        from hermes_cli.copilot_auth import _derive_base_url_from_token

        token = "tid=abc;proxy-ep=proxy.individual.githubcopilot.com;exp=999"
        assert (
            _derive_base_url_from_token(token)
            == "https://api.individual.githubcopilot.com"
        )

    def test_no_proxy_ep_returns_default(self):
        from hermes_cli.copilot_auth import (
            _derive_base_url_from_token,
            DEFAULT_COPILOT_API_BASE_URL,
        )

        token = "tid=abc;exp=999;sku=test"
        assert _derive_base_url_from_token(token) == DEFAULT_COPILOT_API_BASE_URL

    def test_empty_token_returns_default(self):
        from hermes_cli.copilot_auth import (
            _derive_base_url_from_token,
            DEFAULT_COPILOT_API_BASE_URL,
        )

        assert _derive_base_url_from_token("") == DEFAULT_COPILOT_API_BASE_URL

    def test_proxy_ep_with_https_prefix(self):
        from hermes_cli.copilot_auth import _derive_base_url_from_token

        token = "tid=abc;proxy-ep=https://proxy.enterprise.githubcopilot.com;exp=999"
        assert (
            _derive_base_url_from_token(token)
            == "https://api.enterprise.githubcopilot.com"
        )

    def test_malformed_proxy_ep_javascript(self):
        from hermes_cli.copilot_auth import (
            _derive_base_url_from_token,
            DEFAULT_COPILOT_API_BASE_URL,
        )

        token = "tid=abc;proxy-ep=javascript:alert(1);exp=999"
        assert _derive_base_url_from_token(token) == DEFAULT_COPILOT_API_BASE_URL

    def test_malformed_proxy_ep_empty_value(self):
        from hermes_cli.copilot_auth import (
            _derive_base_url_from_token,
            DEFAULT_COPILOT_API_BASE_URL,
        )

        token = "tid=abc;proxy-ep=;exp=999"
        assert _derive_base_url_from_token(token) == DEFAULT_COPILOT_API_BASE_URL


class TestSessionTokenCache:
    """Session token caching and loading."""

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        from hermes_cli.copilot_auth import (
            _save_session_token,
            _load_cached_session_token,
        )

        cache_path = tmp_path / "credentials" / "github-copilot.token.json"
        monkeypatch.setattr(
            "hermes_cli.copilot_auth._get_token_cache_path", lambda: cache_path
        )

        expires = time.time() + 3600
        _save_session_token(FAKE_SESSION_TOKEN, expires)

        loaded = _load_cached_session_token()
        assert loaded is not None
        assert loaded.token == FAKE_SESSION_TOKEN
        assert loaded.base_url == "https://api.enterprise.githubcopilot.com"

    def test_expired_token_not_loaded(self, tmp_path, monkeypatch):
        from hermes_cli.copilot_auth import (
            _save_session_token,
            _load_cached_session_token,
        )

        cache_path = tmp_path / "credentials" / "github-copilot.token.json"
        monkeypatch.setattr(
            "hermes_cli.copilot_auth._get_token_cache_path", lambda: cache_path
        )

        _save_session_token(FAKE_SESSION_TOKEN, time.time() - 10)

        loaded = _load_cached_session_token()
        assert loaded is None

    def test_missing_file_returns_none(self, tmp_path, monkeypatch):
        from hermes_cli.copilot_auth import _load_cached_session_token

        cache_path = tmp_path / "credentials" / "github-copilot.token.json"
        monkeypatch.setattr(
            "hermes_cli.copilot_auth._get_token_cache_path", lambda: cache_path
        )

        assert _load_cached_session_token() is None

    def test_cache_file_permissions(self, tmp_path, monkeypatch):
        from hermes_cli.copilot_auth import _save_session_token

        cache_path = tmp_path / "credentials" / "github-copilot.token.json"
        monkeypatch.setattr(
            "hermes_cli.copilot_auth._get_token_cache_path", lambda: cache_path
        )

        _save_session_token(FAKE_SESSION_TOKEN, time.time() + 3600)
        assert cache_path.stat().st_mode & 0o777 == 0o600


class TestTokenExchange:
    """GitHub token -> Copilot session token exchange."""

    def test_exchange_uses_cache_when_valid(self, tmp_path, monkeypatch):
        from hermes_cli.copilot_auth import (
            exchange_github_token_for_copilot_session,
            _save_session_token,
        )

        cache_path = tmp_path / "credentials" / "github-copilot.token.json"
        monkeypatch.setattr(
            "hermes_cli.copilot_auth._get_token_cache_path", lambda: cache_path
        )

        _save_session_token(FAKE_SESSION_TOKEN, time.time() + 3600)

        result = exchange_github_token_for_copilot_session("gho_unused")
        assert result.token == FAKE_SESSION_TOKEN
        assert "cache:" in result.source

    def test_exchange_calls_api_when_no_cache(self, tmp_path, monkeypatch):
        from hermes_cli.copilot_auth import exchange_github_token_for_copilot_session

        cache_path = tmp_path / "credentials" / "github-copilot.token.json"
        monkeypatch.setattr(
            "hermes_cli.copilot_auth._get_token_cache_path", lambda: cache_path
        )

        response_body = json.dumps(
            {
                "token": FAKE_SESSION_TOKEN,
                "expires_at": int(time.time()) + 1800,
            }
        ).encode()

        mock_resp = MagicMock()
        mock_resp.read.return_value = response_body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = exchange_github_token_for_copilot_session("gho_test")

        assert result.token == FAKE_SESSION_TOKEN
        assert "fetched:" in result.source
        assert cache_path.is_file()

    def test_exchange_raises_on_http_error(self, tmp_path, monkeypatch):
        from hermes_cli.copilot_auth import exchange_github_token_for_copilot_session
        import urllib.error

        cache_path = tmp_path / "credentials" / "github-copilot.token.json"
        monkeypatch.setattr(
            "hermes_cli.copilot_auth._get_token_cache_path", lambda: cache_path
        )

        exc = urllib.error.HTTPError(
            url="https://api.github.com/copilot_internal/v2/token",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=MagicMock(read=lambda: b'{"message":"Not Found"}'),
        )
        with patch("urllib.request.urlopen", side_effect=exc):
            with pytest.raises(RuntimeError, match="HTTP 404"):
                exchange_github_token_for_copilot_session("gho_bad")


class TestResolveToken:
    """Full resolution chain: env var -> GitHub token -> exchange -> session token."""

    def test_copilot_github_token_first_priority(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_token

        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "gho_copilot_first")
        monkeypatch.setenv("GH_TOKEN", "gho_gh_second")
        monkeypatch.setenv("GITHUB_TOKEN", "gho_github_third")
        with patch(
            "hermes_cli.copilot_auth.exchange_github_token_for_copilot_session",
            side_effect=_mock_exchange_success,
        ):
            token, base_url, source = resolve_copilot_token()
        assert token == FAKE_SESSION_TOKEN
        assert base_url == "https://api.enterprise.githubcopilot.com"
        assert "COPILOT_GITHUB_TOKEN" in source

    def test_gh_token_second_priority(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_token

        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.setenv("GH_TOKEN", "gho_gh_second")
        monkeypatch.setenv("GITHUB_TOKEN", "gho_github_third")
        with patch(
            "hermes_cli.copilot_auth.exchange_github_token_for_copilot_session",
            side_effect=_mock_exchange_success,
        ):
            token, base_url, source = resolve_copilot_token()
        assert token == FAKE_SESSION_TOKEN
        assert base_url == "https://api.enterprise.githubcopilot.com"
        assert "GH_TOKEN" in source

    def test_github_token_third_priority(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_token

        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.setenv("GITHUB_TOKEN", "gho_github_third")
        with patch(
            "hermes_cli.copilot_auth.exchange_github_token_for_copilot_session",
            side_effect=_mock_exchange_success,
        ):
            token, base_url, source = resolve_copilot_token()
        assert token == FAKE_SESSION_TOKEN
        assert base_url == "https://api.enterprise.githubcopilot.com"
        assert "GITHUB_TOKEN" in source

    def test_classic_pat_in_env_skipped(self, monkeypatch):
        """Classic PATs in env vars should be skipped, not returned."""
        from hermes_cli.copilot_auth import resolve_copilot_token

        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "ghp_classic_pat_nope")
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.setenv("GITHUB_TOKEN", "gho_valid_oauth")
        with patch(
            "hermes_cli.copilot_auth.exchange_github_token_for_copilot_session",
            side_effect=_mock_exchange_success,
        ):
            token, base_url, source = resolve_copilot_token()
        assert token == FAKE_SESSION_TOKEN
        assert "GITHUB_TOKEN" in source

    def test_gh_cli_fallback(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_token

        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        with (
            patch(
                "hermes_cli.copilot_auth._try_gh_cli_token", return_value="gho_from_cli"
            ),
            patch(
                "hermes_cli.copilot_auth.exchange_github_token_for_copilot_session",
                side_effect=_mock_exchange_success,
            ),
        ):
            token, base_url, source = resolve_copilot_token()
        assert token == FAKE_SESSION_TOKEN
        assert "gh auth token" in source

    def test_gh_cli_classic_pat_raises(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_token

        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        with patch(
            "hermes_cli.copilot_auth._try_gh_cli_token", return_value="ghp_classic"
        ):
            with pytest.raises(ValueError, match="classic PAT"):
                resolve_copilot_token()

    def test_no_token_returns_empty(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_token

        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        with patch("hermes_cli.copilot_auth._try_gh_cli_token", return_value=None):
            token, base_url, source = resolve_copilot_token()
        assert token == ""
        assert base_url == ""
        assert source == ""

    def test_exchange_failure_returns_empty(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_token

        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "gho_valid")
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        with patch(
            "hermes_cli.copilot_auth.exchange_github_token_for_copilot_session",
            side_effect=RuntimeError("HTTP 404"),
        ):
            token, base_url, source = resolve_copilot_token()
        assert token == ""
        assert base_url == ""
        assert source == ""


class TestRequestHeaders:
    """Copilot API header generation."""

    def test_default_headers_include_openai_intent(self):
        from hermes_cli.copilot_auth import copilot_request_headers

        headers = copilot_request_headers()
        assert headers["Openai-Intent"] == "conversation-edits"
        assert headers["User-Agent"] == "HermesAgent/1.0"
        assert "Editor-Version" in headers

    def test_agent_turn_sets_initiator(self):
        from hermes_cli.copilot_auth import copilot_request_headers

        headers = copilot_request_headers(is_agent_turn=True)
        assert headers["x-initiator"] == "agent"

    def test_user_turn_sets_initiator(self):
        from hermes_cli.copilot_auth import copilot_request_headers

        headers = copilot_request_headers(is_agent_turn=False)
        assert headers["x-initiator"] == "user"

    def test_vision_header(self):
        from hermes_cli.copilot_auth import copilot_request_headers

        headers = copilot_request_headers(is_vision=True)
        assert headers["Copilot-Vision-Request"] == "true"

    def test_no_vision_header_by_default(self):
        from hermes_cli.copilot_auth import copilot_request_headers

        headers = copilot_request_headers()
        assert "Copilot-Vision-Request" not in headers


class TestCopilotDefaultHeaders:
    """The models.py copilot_default_headers uses copilot_auth."""

    def test_includes_openai_intent(self):
        from hermes_cli.models import copilot_default_headers

        headers = copilot_default_headers()
        assert "Openai-Intent" in headers
        assert headers["Openai-Intent"] == "conversation-edits"

    def test_includes_x_initiator(self):
        from hermes_cli.models import copilot_default_headers

        headers = copilot_default_headers()
        assert "x-initiator" in headers


class TestApiModeSelection:
    """API mode selection matching opencode's shouldUseCopilotResponsesApi."""

    def test_gpt5_uses_responses(self):
        from hermes_cli.models import _should_use_copilot_responses_api

        assert _should_use_copilot_responses_api("gpt-5.4") is True
        assert _should_use_copilot_responses_api("gpt-5.4-mini") is True
        assert _should_use_copilot_responses_api("gpt-5.3-codex") is True
        assert _should_use_copilot_responses_api("gpt-5.2-codex") is True
        assert _should_use_copilot_responses_api("gpt-5.2") is True
        assert _should_use_copilot_responses_api("gpt-5.1-codex-max") is True

    def test_gpt5_mini_excluded(self):
        from hermes_cli.models import _should_use_copilot_responses_api

        assert _should_use_copilot_responses_api("gpt-5-mini") is False

    def test_gpt4_uses_chat(self):
        from hermes_cli.models import _should_use_copilot_responses_api

        assert _should_use_copilot_responses_api("gpt-4.1") is False
        assert _should_use_copilot_responses_api("gpt-4o") is False
        assert _should_use_copilot_responses_api("gpt-4o-mini") is False

    def test_non_gpt_uses_chat(self):
        from hermes_cli.models import _should_use_copilot_responses_api

        assert _should_use_copilot_responses_api("claude-sonnet-4.6") is False
        assert _should_use_copilot_responses_api("claude-opus-4.6") is False
        assert _should_use_copilot_responses_api("gemini-2.5-pro") is False
        assert _should_use_copilot_responses_api("grok-code-fast-1") is False


class TestEnvVarOrder:
    """PROVIDER_REGISTRY has correct env var order."""

    def test_copilot_env_vars_include_copilot_github_token(self):
        from hermes_cli.auth import PROVIDER_REGISTRY

        copilot = PROVIDER_REGISTRY["copilot"]
        assert "COPILOT_GITHUB_TOKEN" in copilot.api_key_env_vars
        assert copilot.api_key_env_vars[0] == "COPILOT_GITHUB_TOKEN"

    def test_copilot_env_vars_order_matches_docs(self):
        from hermes_cli.auth import PROVIDER_REGISTRY

        copilot = PROVIDER_REGISTRY["copilot"]
        assert copilot.api_key_env_vars == (
            "COPILOT_GITHUB_TOKEN",
            "GH_TOKEN",
            "GITHUB_TOKEN",
        )
