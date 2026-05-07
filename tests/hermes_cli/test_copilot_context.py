"""Tests for Copilot live /models context-window resolution."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from hermes_cli.models import get_copilot_model_context


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
]


@pytest.fixture(autouse=True)
def _clear_cache():
    """Reset module-level cache before each test."""
    import hermes_cli.models as mod

    mod._copilot_context_cache = {}
    mod._copilot_context_cache_time = {}
    mod._copilot_catalog_api_key_cache = {}
    yield
    mod._copilot_context_cache = {}
    mod._copilot_context_cache_time = {}
    mod._copilot_catalog_api_key_cache = {}


class TestGetCopilotModelContext:
    """Tests for get_copilot_model_context()."""

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
    def test_cache_is_scoped_to_api_key(self, mock_fetch):
        assert get_copilot_model_context("gpt-4.1", api_key="token-a") == 128_000
        assert get_copilot_model_context("gpt-4.1", api_key="token-b") == 128_000
        assert get_copilot_model_context("gpt-4.1", api_key="token-a") == 128_000
        assert mock_fetch.call_count == 2

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=_SAMPLE_CATALOG)
    @patch("hermes_cli.models._resolve_copilot_catalog_api_key_details", return_value=("gh-token", float("inf")))
    def test_resolves_catalog_api_key_when_not_provided(self, mock_resolve, mock_fetch):
        assert get_copilot_model_context("gpt-4.1") == 128_000
        assert get_copilot_model_context("gpt-4.1") == 128_000
        mock_resolve.assert_called_once_with()
        mock_fetch.assert_called_once_with(api_key="gh-token", retry_without_auth=False)

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=_SAMPLE_CATALOG)
    @patch("hermes_cli.models._resolve_copilot_catalog_api_key_details", return_value=("gh-token", float("inf")))
    def test_no_key_cache_hit_skips_token_refresh(self, mock_resolve, mock_fetch):
        import hermes_cli.models as mod

        assert get_copilot_model_context("gpt-4.1") == 128_000
        state_key = next(iter(mod._copilot_catalog_api_key_cache))
        assert mod._copilot_catalog_api_key_cache[state_key][0] == "gh-token"

        assert get_copilot_model_context("gpt-4.1") == 128_000
        assert mock_resolve.call_count == 1
        assert mock_fetch.call_count == 1
        mock_fetch.assert_called_once_with(api_key="gh-token", retry_without_auth=False)

    @patch(
        "hermes_cli.models.fetch_github_model_catalog",
        side_effect=[_SAMPLE_CATALOG, _SAMPLE_CATALOG],
    )
    @patch(
        "hermes_cli.models._resolve_copilot_catalog_api_key_details",
        side_effect=[("", 0.0), ("gh-token", float("inf"))],
    )
    def test_anonymous_cache_invalidates_on_auth_change(self, mock_resolve, mock_fetch, monkeypatch):
        assert get_copilot_model_context("gpt-4.1") == 128_000
        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "fresh-token")
        assert get_copilot_model_context("gpt-4.1") == 128_000
        assert mock_resolve.call_count == 2
        assert mock_fetch.call_count == 2
        assert mock_fetch.call_args_list[0].kwargs.get("retry_without_auth") is True
        assert mock_fetch.call_args_list[1].kwargs.get("retry_without_auth") is False

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=_SAMPLE_CATALOG)
    @patch("hermes_cli.models._resolve_copilot_catalog_api_key_details", return_value=("gh-token-a", 1.0))
    def test_fresh_cached_context_skips_token_refresh(self, mock_resolve, mock_fetch):
        import hermes_cli.models as mod

        assert get_copilot_model_context("gpt-4.1") == 128_000
        state_key = next(iter(mod._copilot_catalog_api_key_cache))
        mod._copilot_catalog_api_key_cache[state_key] = ("gh-token-a", 0.0)

        assert get_copilot_model_context("gpt-4.1") == 128_000
        assert mock_resolve.call_count == 1
        assert mock_fetch.call_count == 1

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=_SAMPLE_CATALOG)
    @patch(
        "hermes_cli.models._resolve_copilot_catalog_api_key_details",
        side_effect=[("gh-token-a", float("inf")), ("gh-token-b", float("inf"))],
    )
    def test_catalog_api_key_cache_respects_gh_host(self, mock_resolve, mock_fetch, monkeypatch):
        monkeypatch.setenv("COPILOT_GH_HOST", "github.com")
        assert get_copilot_model_context("gpt-4.1") == 128_000
        monkeypatch.setenv("COPILOT_GH_HOST", "github.example.com")
        assert get_copilot_model_context("gpt-4.1") == 128_000
        assert mock_resolve.call_count == 2
        assert mock_fetch.call_count == 2

    @patch(
        "hermes_cli.models.fetch_github_model_catalog",
        side_effect=[_SAMPLE_CATALOG, _SAMPLE_CATALOG, _SAMPLE_CATALOG],
    )
    @patch(
        "hermes_cli.models._resolve_copilot_catalog_api_key_details",
        side_effect=[("gh-token-a", 1.0), ("", 0.0), ("gh-token-b", float("inf"))],
    )
    def test_expired_cached_catalog_token_refreshes_without_memoizing_empty(self, mock_resolve, mock_fetch):
        import hermes_cli.models as mod

        assert get_copilot_model_context("gpt-4.1") == 128_000
        state_key = next(iter(mod._copilot_catalog_api_key_cache))
        token, expires_at = mod._copilot_catalog_api_key_cache[state_key]
        assert token == "gh-token-a"
        mod._copilot_catalog_api_key_cache[state_key] = (token, 0.0)
        cache_id = next(iter(mod._copilot_context_cache))
        mod._copilot_context_cache_time[cache_id] = time.time() - 7200

        assert get_copilot_model_context("gpt-4.1") == 128_000
        assert mod._copilot_catalog_api_key_cache[state_key] == ("gh-token-a", 0.0)
        assert mock_fetch.call_args_list[1].kwargs.get("api_key") is None
        assert mock_fetch.call_args_list[1].kwargs.get("retry_without_auth") is True

        mod._copilot_catalog_api_key_cache[state_key] = ("gh-token-a", 0.0)
        mod._copilot_context_cache_time[cache_id] = time.time() - 7200
        assert get_copilot_model_context("gpt-4.1") == 128_000
        assert mod._copilot_catalog_api_key_cache[state_key][0] == "gh-token-b"
        assert mock_fetch.call_args_list[2].kwargs.get("api_key") == "gh-token-b"
        assert mock_fetch.call_args_list[2].kwargs.get("retry_without_auth") is False
        assert mock_resolve.call_count == 3
        assert mock_fetch.call_count == 3

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=_SAMPLE_CATALOG)
    def test_cache_expires(self, mock_fetch):
        import hermes_cli.models as mod

        get_copilot_model_context("gpt-4.1")
        assert mock_fetch.call_count == 1

        # Expire the cache
        cache_id = next(iter(mod._copilot_context_cache))
        mod._copilot_context_cache_time[cache_id] = time.time() - 7200
        get_copilot_model_context("gpt-4.1")
        assert mock_fetch.call_count == 2

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=None)
    def test_returns_none_when_catalog_unavailable(self, mock_fetch):
        assert get_copilot_model_context("gpt-4.1") is None

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=[])
    def test_returns_none_for_empty_catalog(self, mock_fetch):
        assert get_copilot_model_context("gpt-4.1") is None

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=_SAMPLE_CATALOG)
    @patch("hermes_cli.models._resolve_copilot_catalog_api_key_details", return_value=("", 0.0))
    def test_unauthenticated_fallback_still_fetches_catalog(self, mock_resolve, mock_fetch):
        assert get_copilot_model_context("gpt-4.1") == 128_000
        mock_resolve.assert_called_once_with()
        mock_fetch.assert_called_once_with(api_key=None, retry_without_auth=True)

    @patch(
        "hermes_cli.models.fetch_github_model_catalog",
        side_effect=[[
            {
                "id": "claude-opus-4.6-1m",
                "capabilities": {"type": "chat", "limits": {"max_prompt_tokens": 1_000_000}},
            }
        ], _SAMPLE_CATALOG],
    )
    def test_fresh_cache_miss_refetches_catalog(self, mock_fetch):
        import hermes_cli.models as mod

        assert get_copilot_model_context("gpt-4.1", api_key="token-a") is None
        assert get_copilot_model_context("gpt-4.1", api_key="token-a") == 128_000
        assert mock_fetch.call_count == 2
        cache_id = mod._copilot_cache_identity("token-a")
        assert "gpt-4.1" in mod._copilot_context_cache[cache_id]

    @patch(
        "hermes_cli.models.fetch_github_model_catalog",
        side_effect=[[
            {
                "id": "claude-opus-4.6-1m",
                "capabilities": {"type": "chat", "limits": {"max_prompt_tokens": 1_000_000}},
            }
        ], _SAMPLE_CATALOG],
    )
    @patch("hermes_cli.models._resolve_copilot_catalog_api_key_details", return_value=("gh-token", float("inf")))
    def test_no_key_fresh_cache_miss_refetches_catalog(self, mock_resolve, mock_fetch):
        assert get_copilot_model_context("gpt-4.1") is None
        assert get_copilot_model_context("gpt-4.1") == 128_000
        assert mock_resolve.call_count == 1
        assert mock_fetch.call_count == 2


class TestCopilotAuthStateFingerprint:
    def test_uses_gh_config_dir_when_set(self, monkeypatch):
        import hermes_cli.models as mod

        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "")
        monkeypatch.setenv("GH_TOKEN", "")
        monkeypatch.setenv("GITHUB_TOKEN", "")
        monkeypatch.setenv("COPILOT_GH_HOST", "")
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)

        monkeypatch.setenv("GH_CONFIG_DIR", "/tmp/gh-config-a")
        fingerprint_a = mod._copilot_auth_state_fingerprint()
        monkeypatch.setenv("GH_CONFIG_DIR", "/tmp/gh-config-b")
        fingerprint_b = mod._copilot_auth_state_fingerprint()

        assert fingerprint_a != fingerprint_b

    def test_uses_xdg_config_home_when_gh_config_dir_missing(self, monkeypatch):
        import hermes_cli.models as mod

        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "")
        monkeypatch.setenv("GH_TOKEN", "")
        monkeypatch.setenv("GITHUB_TOKEN", "")
        monkeypatch.setenv("COPILOT_GH_HOST", "")
        monkeypatch.delenv("GH_CONFIG_DIR", raising=False)

        monkeypatch.setenv("XDG_CONFIG_HOME", "/tmp/xdg-config-a")
        fingerprint_a = mod._copilot_auth_state_fingerprint()
        monkeypatch.setenv("XDG_CONFIG_HOME", "/tmp/xdg-config-b")
        fingerprint_b = mod._copilot_auth_state_fingerprint()

        assert fingerprint_a != fingerprint_b


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

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=_SAMPLE_CATALOG)
    @patch("hermes_cli.models._resolve_copilot_catalog_api_key_details", return_value=("gh-token", float("inf")))
    def test_copilot_live_context_skips_provider_cache(self, mock_resolve, mock_fetch):
        from agent.model_metadata import get_model_context_length

        with patch("agent.model_metadata.get_cached_context_length") as mock_cache, \
             patch("agent.model_metadata.save_context_length") as mock_save:
            ctx = get_model_context_length(
                "claude-sonnet-4",
                provider="copilot-acp",
                base_url="acp://copilot",
            )

        assert ctx == 200_000
        mock_resolve.assert_called_once_with()
        mock_fetch.assert_called_once_with(api_key="gh-token", retry_without_auth=False)
        mock_cache.assert_not_called()
        mock_save.assert_not_called()

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=None)
    def test_falls_through_when_catalog_unavailable(self, mock_fetch):
        from agent.model_metadata import get_model_context_length

        # Should not raise, should fall through to models.dev or defaults
        ctx = get_model_context_length("gpt-4.1", provider="copilot")
        assert isinstance(ctx, int)
        assert ctx > 0
