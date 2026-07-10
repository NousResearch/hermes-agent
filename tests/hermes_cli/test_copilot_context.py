"""Tests for Copilot live /models context-window resolution."""

from __future__ import annotations

import time
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.models import (
    get_copilot_model_catalog_cached,
    get_copilot_model_context,
    get_copilot_reasoning_efforts,
)


# Sample catalog items mimicking the Copilot /models API response
_SAMPLE_CATALOG = [
    {
        "id": "claude-opus-4.6-1m",
        "capabilities": {
            "type": "chat",
            "limits": {"max_prompt_tokens": 1000000, "max_output_tokens": 64000},
        },
    },
    {
        "id": "gpt-4.1",
        "capabilities": {
            "type": "chat",
            "limits": {"max_prompt_tokens": 128000, "max_output_tokens": 32768},
        },
    },
    {
        "id": "claude-sonnet-4",
        "capabilities": {
            "type": "chat",
            "limits": {"max_prompt_tokens": 200000, "max_output_tokens": 64000},
        },
    },
    {
        "id": "model-without-limits",
        "capabilities": {"type": "chat"},
    },
    {
        "id": "model-zero-limit",
        "capabilities": {
            "type": "chat",
            "limits": {"max_prompt_tokens": 0},
        },
    },
    {
        "id": "gpt-5.6-sol",
        "capabilities": {
            "type": "chat",
            "limits": {"max_prompt_tokens": 922000},
            "supports": {
                "reasoning_effort": [
                    "none",
                    "low",
                    "medium",
                    "high",
                    "xhigh",
                    "max",
                ]
            },
        },
        "supported_endpoints": ["/responses"],
    },
]


@pytest.fixture(autouse=True)
def _clear_cache():
    """Reset module-level cache before each test."""
    import hermes_cli.models as mod

    mod._copilot_catalog_cache = {}
    mod._copilot_catalog_failed_time = {}
    yield
    mod._copilot_catalog_cache = {}
    mod._copilot_catalog_failed_time = {}


class TestGetCopilotModelContext:
    """Tests for get_copilot_model_context()."""

    @patch("urllib.request.urlopen")
    def test_authenticated_catalog_fetch_never_falls_back_to_anonymous(self, urlopen):
        from hermes_cli.models import fetch_github_model_catalog

        urlopen.side_effect = urllib.error.HTTPError(
            "https://api.githubcopilot.com/models",
            403,
            "forbidden",
            hdrs=None,
            fp=None,
        )

        assert fetch_github_model_catalog(api_key="account-a-secret") is None
        assert urlopen.call_count == 1
        request = urlopen.call_args.args[0]
        assert request.get_header("Authorization") == "Bearer account-a-secret"

    @patch("urllib.request.urlopen")
    def test_anonymous_catalog_fetch_stays_anonymous(self, urlopen):
        from hermes_cli.models import fetch_github_model_catalog

        response = MagicMock()
        response.read.return_value = b'{"data": [{"id": "gpt-5.6-sol"}]}'
        response.__enter__.return_value = response
        response.__exit__.return_value = False
        urlopen.return_value = response

        assert fetch_github_model_catalog(api_key=None) == [{"id": "gpt-5.6-sol"}]
        assert urlopen.call_count == 1
        request = urlopen.call_args.args[0]
        assert request.get_header("Authorization") is None

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=_SAMPLE_CATALOG)
    def test_catalog_cache_is_shared_by_context_and_reasoning(self, mock_fetch):
        assert get_copilot_model_context("gpt-5.6-sol") == 922_000
        assert get_copilot_reasoning_efforts("gpt-5.6-sol")[-2:] == [
            "xhigh",
            "max",
        ]
        assert mock_fetch.call_count == 1

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=_SAMPLE_CATALOG)
    def test_cached_catalog_returns_same_object(self, mock_fetch):
        first = get_copilot_model_catalog_cached("test-token")
        second = get_copilot_model_catalog_cached("test-token")
        assert first is second
        assert mock_fetch.call_count == 1

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=_SAMPLE_CATALOG)
    def test_returns_max_prompt_tokens(self, mock_fetch):
        assert get_copilot_model_context("claude-opus-4.6-1m") == 1_000_000
        assert get_copilot_model_context("gpt-4.1") == 128_000

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=_SAMPLE_CATALOG)
    def test_returns_none_for_unknown_model(self, mock_fetch):
        assert get_copilot_model_context("nonexistent-model") is None

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=_SAMPLE_CATALOG)
    def test_skips_models_without_limits(self, mock_fetch):
        assert get_copilot_model_context("model-without-limits") is None

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=_SAMPLE_CATALOG)
    def test_skips_zero_limit(self, mock_fetch):
        assert get_copilot_model_context("model-zero-limit") is None

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=_SAMPLE_CATALOG)
    def test_caches_results(self, mock_fetch):
        get_copilot_model_context("gpt-4.1")
        get_copilot_model_context("claude-sonnet-4")
        # Only one API call despite two lookups
        assert mock_fetch.call_count == 1

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=_SAMPLE_CATALOG)
    def test_cache_expires(self, mock_fetch):
        import hermes_cli.models as mod

        get_copilot_model_context("gpt-4.1")
        assert mock_fetch.call_count == 1

        # Expire the account-scoped shared catalog.
        key = mod._copilot_catalog_cache_key(None)
        catalog, _ = mod._copilot_catalog_cache[key]
        mod._copilot_catalog_cache[key] = (catalog, time.time() - 7200)
        get_copilot_model_context("gpt-4.1")
        assert mock_fetch.call_count == 2

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=None)
    def test_expired_catalog_is_dropped_when_refresh_fails(self, mock_fetch):
        import hermes_cli.models as mod

        key = mod._copilot_catalog_cache_key("account-a")
        stale_catalog = [
            {
                "id": "gpt-5.6-sol",
                "capabilities": {
                    "type": "chat",
                    "supports": {"reasoning_effort": ["low", "max"]},
                },
            }
        ]
        mod._copilot_catalog_cache[key] = (stale_catalog, time.time() - 7200)

        assert get_copilot_model_catalog_cached("account-a") is None
        assert key not in mod._copilot_catalog_cache
        assert mock_fetch.call_count == 1
        assert get_copilot_model_catalog_cached("account-a") is None
        assert mock_fetch.call_count == 1

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=None)
    def test_failed_fetch_is_negative_cached_per_account(self, mock_fetch):
        assert get_copilot_reasoning_efforts("gpt-5.6-sol", "account-a") == [
            "minimal",
            "low",
            "medium",
            "high",
        ]
        get_copilot_reasoning_efforts("gpt-5.6-sol", "account-a")
        assert mock_fetch.call_count == 1
        get_copilot_reasoning_efforts("gpt-5.6-sol", "account-b")
        assert mock_fetch.call_count == 2

    @patch("hermes_cli.models.fetch_github_model_catalog")
    def test_catalog_cache_does_not_cross_copilot_accounts(self, mock_fetch):
        catalog_a = [
            {
                "id": "gpt-5.6-sol",
                "capabilities": {
                    "type": "chat",
                    "supports": {"reasoning_effort": ["low", "high"]},
                },
            }
        ]
        catalog_b = [
            {
                "id": "gpt-5.6-sol",
                "capabilities": {
                    "type": "chat",
                    "supports": {"reasoning_effort": ["low", "high", "max"]},
                },
            }
        ]
        mock_fetch.side_effect = [catalog_a, catalog_b]

        assert get_copilot_reasoning_efforts("gpt-5.6-sol", "account-a") == [
            "low",
            "high",
        ]
        assert get_copilot_reasoning_efforts("gpt-5.6-sol", "account-b") == [
            "low",
            "high",
            "max",
        ]
        assert mock_fetch.call_count == 2

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=None)
    def test_returns_none_when_catalog_unavailable(self, mock_fetch):
        assert get_copilot_model_context("gpt-4.1") is None

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=[])
    def test_returns_none_for_empty_catalog(self, mock_fetch):
        assert get_copilot_model_context("gpt-4.1") is None


class TestModelMetadataCopilotIntegration:
    """Test that get_model_context_length() uses Copilot live API for copilot provider."""

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=_SAMPLE_CATALOG)
    def test_copilot_provider_uses_live_api(self, mock_fetch):
        from agent.model_metadata import get_model_context_length

        ctx = get_model_context_length("claude-opus-4.6-1m", provider="copilot")
        assert ctx == 1_000_000

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=_SAMPLE_CATALOG)
    def test_copilot_acp_provider_uses_live_api(self, mock_fetch):
        from agent.model_metadata import get_model_context_length

        ctx = get_model_context_length("claude-sonnet-4", provider="copilot-acp")
        assert ctx == 200_000

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=None)
    def test_falls_through_when_catalog_unavailable(self, mock_fetch):
        from agent.model_metadata import get_model_context_length

        # Should not raise, should fall through to models.dev or defaults
        ctx = get_model_context_length("gpt-4.1", provider="copilot")
        assert isinstance(ctx, int)
        assert ctx > 0
