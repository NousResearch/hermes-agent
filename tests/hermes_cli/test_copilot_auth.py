"""Tests for hermes_cli.copilot_auth — Copilot token validation and resolution."""

import json
import os
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock


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

    def test_is_classic_pat(self):
        from hermes_cli.copilot_auth import is_classic_pat
        assert is_classic_pat("ghp_abc123") is True
        assert is_classic_pat("gho_abc123") is False
        assert is_classic_pat("github_pat_abc") is False
        assert is_classic_pat("") is False


class TestResolveToken:
    """Token resolution with env var priority."""

    def test_copilot_github_token_first_priority(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_token
        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "gho_copilot_first")
        monkeypatch.setenv("GH_TOKEN", "gho_gh_second")
        monkeypatch.setenv("GITHUB_TOKEN", "gho_github_third")
        token, source = resolve_copilot_token()
        assert token == "gho_copilot_first"
        assert source == "COPILOT_GITHUB_TOKEN"

    def test_gh_token_second_priority(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_token
        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.setenv("GH_TOKEN", "gho_gh_second")
        monkeypatch.setenv("GITHUB_TOKEN", "gho_github_third")
        token, source = resolve_copilot_token()
        assert token == "gho_gh_second"
        assert source == "GH_TOKEN"

    def test_github_token_third_priority(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_token
        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.setenv("GITHUB_TOKEN", "gho_github_third")
        token, source = resolve_copilot_token()
        assert token == "gho_github_third"
        assert source == "GITHUB_TOKEN"

    def test_classic_pat_in_env_skipped(self, monkeypatch):
        """Classic PATs in env vars should be skipped, not returned."""
        from hermes_cli.copilot_auth import resolve_copilot_token
        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "ghp_classic_pat_nope")
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.setenv("GITHUB_TOKEN", "gho_valid_oauth")
        token, source = resolve_copilot_token()
        # Should skip the ghp_ token and find the gho_ one
        assert token == "gho_valid_oauth"
        assert source == "GITHUB_TOKEN"

    def test_gh_cli_fallback(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_token
        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        with patch("hermes_cli.copilot_auth._try_gh_cli_token", return_value="gho_from_cli"):
            token, source = resolve_copilot_token()
        assert token == "gho_from_cli"
        assert source == "gh auth token"

    def test_gh_cli_classic_pat_raises(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_token
        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        with patch("hermes_cli.copilot_auth._try_gh_cli_token", return_value="ghp_classic"):
            with pytest.raises(ValueError, match="classic PAT"):
                resolve_copilot_token()

    def test_no_token_returns_empty(self, monkeypatch):
        from hermes_cli.copilot_auth import resolve_copilot_token
        monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        with patch("hermes_cli.copilot_auth._try_gh_cli_token", return_value=None):
            token, source = resolve_copilot_token()
        assert token == ""
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


class TestTokenExchange:
    def test_derive_copilot_api_base_url_uses_proxy_endpoint(self):
        from hermes_cli.copilot_auth import derive_copilot_api_base_url

        token = "abc123;proxy-ep=proxy.individual.githubcopilot.com;foo=bar"
        assert derive_copilot_api_base_url(token) == "https://api.individual.githubcopilot.com"

    def test_derive_copilot_api_base_url_rejects_non_github_hosts(self):
        from hermes_cli.copilot_auth import derive_copilot_api_base_url, COPILOT_API_BASE_URL

        token = "abc123;proxy-ep=evil-githubcopilot.com.evil.example;foo=bar"
        assert derive_copilot_api_base_url(token) == COPILOT_API_BASE_URL

    def test_derive_copilot_api_base_url_ignores_path_components(self):
        from hermes_cli.copilot_auth import derive_copilot_api_base_url

        token = "abc123;proxy-ep=proxy.individual.githubcopilot.com/v1;foo=bar"
        assert derive_copilot_api_base_url(token) == "https://api.individual.githubcopilot.com"

    def test_is_copilot_base_url_accepts_routed_domains(self):
        from hermes_cli.copilot_auth import is_copilot_base_url

        assert is_copilot_base_url("https://api.githubcopilot.com")
        assert is_copilot_base_url("https://api.individual.githubcopilot.com")
        assert is_copilot_base_url("https://api.business.githubcopilot.com")
        assert is_copilot_base_url("https://example.com") is False
        assert is_copilot_base_url("https://evil-githubcopilot.com") is False
        assert is_copilot_base_url("https://githubcopilot.com.evil.com") is False

    @pytest.mark.parametrize(
        "raw,expected_ms",
        [
            (1718452800, 1718452800000),
            (1718452800000, 1718452800000),
            ("1718452800", 1718452800000),
            ("2024-06-15T12:00:00+00:00", 1718452800000),
            ("2024-06-15T12:00:00Z", 1718452800000),
        ],
    )
    def test_parse_expires_at_variants(self, raw, expected_ms):
        from hermes_cli.copilot_auth import _parse_expires_at

        assert _parse_expires_at(raw) == expected_ms

    def test_parse_expires_at_invalid_raises(self):
        from hermes_cli.copilot_auth import _parse_expires_at

        with pytest.raises(ValueError):
            _parse_expires_at("not-a-date")

    def test_cache_round_trip_and_fingerprint(self, tmp_path, monkeypatch):
        from hermes_cli.copilot_auth import (
            _save_cached_copilot_api_token,
            get_cached_copilot_api_token,
        )

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _save_cached_copilot_api_token(
            "gho_source_token",
            "copilot_runtime_token",
            4_102_444_800_000,
            "https://api.individual.githubcopilot.com",
        )

        cached = get_cached_copilot_api_token("gho_source_token")
        assert cached is not None
        assert cached["token"] == "copilot_runtime_token"
        assert cached["base_url"] == "https://api.individual.githubcopilot.com"

        assert get_cached_copilot_api_token("gho_other_token") is None

    def test_cache_rejects_corrupt_file(self, tmp_path, monkeypatch):
        from hermes_cli.copilot_auth import get_cached_copilot_api_token

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        cache_path = Path(tmp_path) / "copilot_token.json"
        cache_path.write_text("{bad json", encoding="utf-8")

        assert get_cached_copilot_api_token("gho_source_token") is None

    def test_exchange_copilot_api_token_uses_cache(self, tmp_path, monkeypatch):
        from hermes_cli.copilot_auth import _save_cached_copilot_api_token, exchange_copilot_api_token

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _save_cached_copilot_api_token(
            "gho_source_token",
            "copilot_runtime_token",
            4_102_444_800_000,
            "https://api.individual.githubcopilot.com",
        )

        result = exchange_copilot_api_token("gho_source_token")
        assert result["token"] == "copilot_runtime_token"
        assert result["base_url"] == "https://api.individual.githubcopilot.com"
        assert "cache:" in result["source"]

    def test_exchange_copilot_api_token_fetches_and_saves(self, tmp_path, monkeypatch):
        from hermes_cli.copilot_auth import exchange_copilot_api_token

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                payload = {
                    "token": "abc123;proxy-ep=proxy.individual.githubcopilot.com",
                    "expires_at": "2024-06-15T12:00:00+00:00",
                }
                return json.dumps(payload).encode("utf-8")

        with patch("urllib.request.urlopen", return_value=_Resp()):
            result = exchange_copilot_api_token("gho_source_token")

        assert result["token"] == "abc123;proxy-ep=proxy.individual.githubcopilot.com"
        assert result["base_url"] == "https://api.individual.githubcopilot.com"
        assert result["expires_at_ms"] == 1718452800000

    def test_exchange_copilot_api_token_missing_token_raises(self, tmp_path, monkeypatch):
        from hermes_cli.copilot_auth import exchange_copilot_api_token

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))

        class _Resp:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps({"expires_at": 1718452800}).encode("utf-8")

        with patch("urllib.request.urlopen", return_value=_Resp()):
            with pytest.raises(ValueError, match="missing token"):
                exchange_copilot_api_token("gho_source_token")


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
        # COPILOT_GITHUB_TOKEN should be first
        assert copilot.api_key_env_vars[0] == "COPILOT_GITHUB_TOKEN"

    def test_copilot_env_vars_order_matches_docs(self):
        from hermes_cli.auth import PROVIDER_REGISTRY
        copilot = PROVIDER_REGISTRY["copilot"]
        assert copilot.api_key_env_vars == (
            "COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"
        )
