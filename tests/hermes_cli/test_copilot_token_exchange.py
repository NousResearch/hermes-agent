"""Tests for Copilot token exchange (raw GitHub token → Copilot API token)."""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clear_jwt_cache():
    """Reset the module-level JWT cache before each test."""
    import hermes_cli.copilot_auth as mod
    mod._jwt_cache.clear()
    mod._endpoint_cache.clear()
    yield
    mod._jwt_cache.clear()
    mod._endpoint_cache.clear()


class TestExchangeCopilotToken:
    """Tests for exchange_copilot_token()."""

    def _mock_urlopen(self, token="tid=abc;exp=123;sku=copilot_individual", expires_at=None):
        """Create a mock urlopen context manager returning a token response."""
        if expires_at is None:
            expires_at = time.time() + 1800
        resp_data = json.dumps({"token": token, "expires_at": expires_at}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = resp_data
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    @patch("urllib.request.urlopen")
    def test_exchanges_token_successfully(self, mock_urlopen):
        from hermes_cli.copilot_auth import exchange_copilot_token

        mock_urlopen.return_value = self._mock_urlopen(token="tid=abc;exp=999")
        api_token, expires_at = exchange_copilot_token("gho_test123")

        assert api_token == "tid=abc;exp=999"
        assert isinstance(expires_at, float)

        # Verify request was made with correct headers
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.get_header("Authorization") == "token gho_test123"
        assert "GitHubCopilotChat" in req.get_header("User-agent")

    @patch("urllib.request.urlopen")
    def test_caches_result(self, mock_urlopen):
        from hermes_cli.copilot_auth import exchange_copilot_token

        future = time.time() + 1800
        mock_urlopen.return_value = self._mock_urlopen(expires_at=future)

        exchange_copilot_token("gho_test123")
        exchange_copilot_token("gho_test123")

        assert mock_urlopen.call_count == 1

    @patch("urllib.request.urlopen")
    def test_refreshes_expired_cache(self, mock_urlopen):
        from hermes_cli.copilot_auth import exchange_copilot_token, _jwt_cache, _token_fingerprint

        # Seed cache with expired entry
        fp = _token_fingerprint("gho_test123")
        _jwt_cache[fp] = ("old_token", time.time() - 10)

        mock_urlopen.return_value = self._mock_urlopen(
            token="new_token", expires_at=time.time() + 1800
        )
        api_token, _ = exchange_copilot_token("gho_test123")

        assert api_token == "new_token"
        assert mock_urlopen.call_count == 1

    @patch("urllib.request.urlopen")
    def test_raises_on_empty_token(self, mock_urlopen):
        from hermes_cli.copilot_auth import exchange_copilot_token

        resp_data = json.dumps({"token": "", "expires_at": 0}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = resp_data
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        with pytest.raises(ValueError, match="empty token"):
            exchange_copilot_token("gho_test123")

    @patch("urllib.request.urlopen", side_effect=Exception("network error"))
    def test_raises_on_network_error(self, mock_urlopen):
        from hermes_cli.copilot_auth import exchange_copilot_token

        with pytest.raises(ValueError, match="network error"):
            exchange_copilot_token("gho_test123")


class TestGetCopilotApiToken:
    """Tests for get_copilot_api_token() — the fallback wrapper."""

    @patch("hermes_cli.copilot_auth.exchange_copilot_token")
    def test_returns_exchanged_token(self, mock_exchange):
        from hermes_cli.copilot_auth import get_copilot_api_token

        mock_exchange.return_value = ("exchanged_jwt", time.time() + 1800)
        assert get_copilot_api_token("gho_raw") == "exchanged_jwt"

    @patch("hermes_cli.copilot_auth.exchange_copilot_token", side_effect=ValueError("fail"))
    def test_falls_back_to_raw_token(self, mock_exchange):
        from hermes_cli.copilot_auth import get_copilot_api_token

        assert get_copilot_api_token("gho_raw") == "gho_raw"

    def test_empty_token_passthrough(self):
        from hermes_cli.copilot_auth import get_copilot_api_token

        assert get_copilot_api_token("") == ""


class TestTokenFingerprint:
    """Tests for _token_fingerprint()."""

    def test_consistent(self):
        from hermes_cli.copilot_auth import _token_fingerprint

        fp1 = _token_fingerprint("gho_abc123")
        fp2 = _token_fingerprint("gho_abc123")
        assert fp1 == fp2

    def test_different_tokens_different_fingerprints(self):
        from hermes_cli.copilot_auth import _token_fingerprint

        fp1 = _token_fingerprint("gho_abc123")
        fp2 = _token_fingerprint("gho_xyz789")
        assert fp1 != fp2

    def test_length(self):
        from hermes_cli.copilot_auth import _token_fingerprint

        assert len(_token_fingerprint("gho_test")) == 16


class TestCallerIntegration:
    """Test that callers correctly use token exchange."""

    @patch("hermes_cli.copilot_auth.resolve_copilot_token", return_value=("gho_raw", "GH_TOKEN"))
    @patch("hermes_cli.copilot_auth.get_copilot_api_token", return_value="exchanged_jwt")
    def test_auth_resolve_uses_exchange(self, mock_exchange, mock_resolve):
        from hermes_cli.auth import _resolve_api_key_provider_secret

        # Create a minimal pconfig mock
        pconfig = MagicMock()
        token, source = _resolve_api_key_provider_secret("copilot", pconfig)
        assert token == "exchanged_jwt"
        assert source == "GH_TOKEN"
        mock_exchange.assert_called_once_with("gho_raw")


class TestCopilotApiBaseUrl:
    """Regression tests for base-URL resolution (#50252).

    The token-exchange response advertises the account-specific API host under
    ``endpoints.api``. Previously it was dropped, leaving the Copilot provider
    to rely on a hardcoded default that could end up empty at chat time.
    """

    def _mock_urlopen(self, *, token="tid=abc", endpoints=None):
        import json as _json
        from unittest.mock import MagicMock as _MagicMock
        payload = {"token": token, "expires_at": time.time() + 1800}
        if endpoints is not None:
            payload["endpoints"] = endpoints
        mock_resp = _MagicMock()
        mock_resp.read.return_value = _json.dumps(payload).encode()
        mock_resp.__enter__ = _MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = _MagicMock(return_value=False)
        return mock_resp

    @patch("urllib.request.urlopen")
    def test_exchange_caches_advertised_endpoint(self, mock_urlopen):
        from hermes_cli.copilot_auth import (
            exchange_copilot_token,
            _endpoint_cache,
            _token_fingerprint,
        )

        mock_urlopen.return_value = self._mock_urlopen(
            endpoints={"api": "https://copilot-proxy.example.com/"}
        )
        exchange_copilot_token("gho_test123")

        fp = _token_fingerprint("gho_test123")
        assert _endpoint_cache[fp] == "https://copilot-proxy.example.com"

    @patch("urllib.request.urlopen")
    def test_base_url_uses_advertised_endpoint(self, mock_urlopen):
        from hermes_cli.copilot_auth import get_copilot_api_base_url

        mock_urlopen.return_value = self._mock_urlopen(
            endpoints={"api": "https://copilot-proxy.example.com"}
        )
        assert get_copilot_api_base_url("gho_test123") == "https://copilot-proxy.example.com"

    @patch("urllib.request.urlopen")
    def test_base_url_falls_back_to_default_when_no_endpoint(self, mock_urlopen):
        from hermes_cli.copilot_auth import (
            get_copilot_api_base_url,
            DEFAULT_COPILOT_API_BASE_URL,
        )

        mock_urlopen.return_value = self._mock_urlopen()
        assert get_copilot_api_base_url("gho_test123") == DEFAULT_COPILOT_API_BASE_URL

    def test_base_url_empty_token_returns_default(self):
        from hermes_cli.copilot_auth import (
            get_copilot_api_base_url,
            DEFAULT_COPILOT_API_BASE_URL,
        )

        assert get_copilot_api_base_url("") == DEFAULT_COPILOT_API_BASE_URL

    @patch("urllib.request.urlopen", side_effect=Exception("network error"))
    def test_base_url_exchange_failure_returns_default(self, mock_urlopen):
        from hermes_cli.copilot_auth import (
            get_copilot_api_base_url,
            DEFAULT_COPILOT_API_BASE_URL,
        )

        assert get_copilot_api_base_url("gho_test123") == DEFAULT_COPILOT_API_BASE_URL


class TestResolveCredentialsCopilotBaseUrl:
    """The provider resolver must always return a non-empty Copilot base URL."""

    @patch("hermes_cli.copilot_auth.resolve_copilot_token", return_value=("gho_raw", "GH_TOKEN"))
    @patch("hermes_cli.copilot_auth.get_copilot_api_token", return_value="exchanged_jwt")
    @patch("hermes_cli.copilot_auth.get_copilot_api_base_url", return_value="https://copilot-proxy.example.com")
    def test_resolver_uses_exchange_endpoint(self, mock_base, mock_tok, mock_resolve, monkeypatch):
        monkeypatch.delenv("COPILOT_API_BASE_URL", raising=False)
        from hermes_cli.auth import resolve_api_key_provider_credentials

        creds = resolve_api_key_provider_credentials("copilot")
        assert creds["base_url"] == "https://copilot-proxy.example.com"

    @patch("hermes_cli.copilot_auth.resolve_copilot_token", return_value=("gho_raw", "GH_TOKEN"))
    @patch("hermes_cli.copilot_auth.get_copilot_api_token", return_value="exchanged_jwt")
    @patch("hermes_cli.copilot_auth.get_copilot_api_base_url", return_value="https://api.githubcopilot.com")
    def test_resolver_never_empty_with_blank_env(self, mock_base, mock_tok, mock_resolve, monkeypatch):
        monkeypatch.setenv("COPILOT_API_BASE_URL", "   ")
        from hermes_cli.auth import resolve_api_key_provider_credentials

        creds = resolve_api_key_provider_credentials("copilot")
        assert creds["base_url"]
        assert creds["base_url"] == "https://api.githubcopilot.com"
