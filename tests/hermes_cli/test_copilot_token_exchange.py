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
    mod._subscription_warned.clear()
    yield
    mod._jwt_cache.clear()
    mod._subscription_warned.clear()


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

    @patch("hermes_cli.copilot_auth._classify_copilot_account", return_value="free")
    @patch("urllib.request.urlopen")
    def test_404_with_free_account_raises_subscription_error(
        self, mock_urlopen, mock_classify
    ):
        """HTTP 404 + /copilot_internal/user reports free SKU = real subscription error."""
        import urllib.error

        from hermes_cli.copilot_auth import (
            CopilotSubscriptionError,
            exchange_copilot_token,
        )

        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.github.com/copilot_internal/v2/token",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        )
        with pytest.raises(CopilotSubscriptionError, match="no active paid Copilot"):
            exchange_copilot_token("gho_test123")
        mock_classify.assert_called_once_with("gho_test123")

    @patch("hermes_cli.copilot_auth._classify_copilot_account", return_value="paid")
    @patch("urllib.request.urlopen")
    def test_404_with_paid_account_falls_through_to_value_error(
        self, mock_urlopen, mock_classify
    ):
        """HTTP 404 + paid SKU = wrong OAuth app, not a subscription problem.

        Regression: gh CLI's OAuth app gets 404 from /copilot_internal/v2/token
        even for paid accounts; the raw token still works for chat completions.
        Must NOT raise CopilotSubscriptionError or paying users see a misleading
        'no subscription' error.
        """
        import urllib.error

        from hermes_cli.copilot_auth import (
            CopilotSubscriptionError,
            exchange_copilot_token,
        )

        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.github.com/copilot_internal/v2/token",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        )
        with pytest.raises(ValueError) as excinfo:
            exchange_copilot_token("gho_test123")
        assert not isinstance(excinfo.value, CopilotSubscriptionError)

    @patch("hermes_cli.copilot_auth._classify_copilot_account", return_value="unknown")
    @patch("urllib.request.urlopen")
    def test_404_with_unknown_account_falls_through_to_value_error(
        self, mock_urlopen, mock_classify
    ):
        """If we can't classify the account (probe failed/missing sku), don't
        block — prefer silent raw-token fallback over a false-positive."""
        import urllib.error

        from hermes_cli.copilot_auth import (
            CopilotSubscriptionError,
            exchange_copilot_token,
        )

        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.github.com/copilot_internal/v2/token",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        )
        with pytest.raises(ValueError) as excinfo:
            exchange_copilot_token("gho_test123")
        assert not isinstance(excinfo.value, CopilotSubscriptionError)

    @patch("urllib.request.urlopen")
    def test_other_http_errors_are_plain_value_error(self, mock_urlopen):
        """Non-404 HTTPErrors stay as ValueError (not subscription error)."""
        import urllib.error

        from hermes_cli.copilot_auth import (
            CopilotSubscriptionError,
            exchange_copilot_token,
        )

        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.github.com/copilot_internal/v2/token",
            code=500,
            msg="Server Error",
            hdrs=None,
            fp=None,
        )
        with pytest.raises(ValueError) as excinfo:
            exchange_copilot_token("gho_test123")
        assert not isinstance(excinfo.value, CopilotSubscriptionError)


class TestClassifyCopilotAccount:
    """Tests for _classify_copilot_account() — disambiguates 404 from /v2/token."""

    def _mock_user_response(self, payload):
        resp_data = json.dumps(payload).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = resp_data
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    @patch("urllib.request.urlopen")
    def test_free_limited_classified_as_free(self, mock_urlopen):
        from hermes_cli.copilot_auth import _classify_copilot_account

        mock_urlopen.return_value = self._mock_user_response(
            {"access_type_sku": "free_limited_copilot", "chat_enabled": True}
        )
        assert _classify_copilot_account("gho_test") == "free"

    @patch("urllib.request.urlopen")
    def test_monthly_subscriber_classified_as_paid(self, mock_urlopen):
        """Real-world signal: vincez-builder's account had this SKU."""
        from hermes_cli.copilot_auth import _classify_copilot_account

        mock_urlopen.return_value = self._mock_user_response(
            {"access_type_sku": "monthly_subscriber_quota", "copilot_plan": "individual"}
        )
        assert _classify_copilot_account("gho_test") == "paid"

    @patch("urllib.request.urlopen")
    def test_missing_sku_classified_as_unknown(self, mock_urlopen):
        from hermes_cli.copilot_auth import _classify_copilot_account

        mock_urlopen.return_value = self._mock_user_response({"chat_enabled": True})
        assert _classify_copilot_account("gho_test") == "unknown"

    @patch("urllib.request.urlopen", side_effect=Exception("network down"))
    def test_network_failure_classified_as_unknown(self, mock_urlopen):
        from hermes_cli.copilot_auth import _classify_copilot_account

        assert _classify_copilot_account("gho_test") == "unknown"


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

    def test_subscription_error_propagates(self):
        """Subscription failures must NOT silently fall back to raw token —
        the raw token is useless against api.githubcopilot.com and would
        produce a misleading 'model_not_supported' for every model."""
        from hermes_cli.copilot_auth import (
            CopilotSubscriptionError,
            get_copilot_api_token,
        )

        with patch(
            "hermes_cli.copilot_auth.exchange_copilot_token",
            side_effect=CopilotSubscriptionError("no sub"),
        ):
            with pytest.raises(CopilotSubscriptionError):
                get_copilot_api_token("gho_raw")

    def test_subscription_warning_logged_once_per_token(self, caplog):
        """The WARNING fires once per fingerprint, not on every retry."""
        import logging

        from hermes_cli.copilot_auth import (
            CopilotSubscriptionError,
            get_copilot_api_token,
        )

        caplog.set_level(logging.WARNING, logger="hermes_cli.copilot_auth")
        with patch(
            "hermes_cli.copilot_auth.exchange_copilot_token",
            side_effect=CopilotSubscriptionError("no sub"),
        ):
            for _ in range(3):
                with pytest.raises(CopilotSubscriptionError):
                    get_copilot_api_token("gho_raw")

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, "expected exactly one WARNING per token"

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

    @patch("hermes_cli.copilot_auth.resolve_copilot_token", return_value=("gho_raw", "GH_TOKEN"))
    def test_auth_resolve_surfaces_subscription_error(self, mock_resolve):
        """Free-tier accounts produce a structured AuthError with the
        copilot_subscription_required code so format_auth_error can render
        actionable guidance."""
        from hermes_cli.auth import AuthError, _resolve_api_key_provider_secret
        from hermes_cli.copilot_auth import CopilotSubscriptionError

        pconfig = MagicMock()
        with patch(
            "hermes_cli.copilot_auth.get_copilot_api_token",
            side_effect=CopilotSubscriptionError("no sub"),
        ):
            with pytest.raises(AuthError) as excinfo:
                _resolve_api_key_provider_secret("copilot", pconfig)

        assert excinfo.value.code == "copilot_subscription_required"
        assert excinfo.value.provider == "copilot"
        assert excinfo.value.relogin_required is False

    def test_format_auth_error_subscription_message(self):
        """format_auth_error renders friendly guidance for the new code."""
        from hermes_cli.auth import AuthError, format_auth_error

        err = AuthError(
            "no sub",
            provider="copilot",
            code="copilot_subscription_required",
            relogin_required=False,
        )
        msg = format_auth_error(err)
        assert "no active paid Copilot subscription" in msg
        assert "github.com/settings/copilot" in msg
        assert "hermes auth add copilot" in msg
