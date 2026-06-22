"""Tests for Copilot live /models context-window resolution."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from hermes_cli.models import (
    build_copilot_inventory_snapshot,
    get_copilot_model_context,
)


# Sample catalog items mimicking the Copilot /models API response
_SAMPLE_CATALOG = [
    {
        "id": "claude-opus-4.6-1m",
        "capabilities": {
            "type": "chat",
            "limits": {
                "max_context_window_tokens": 1000000,
                "max_prompt_tokens": 1000000,
                "max_output_tokens": 64000,
                "max_non_streaming_output_tokens": 32000,
            },
        },
    },
    {
        "id": "gpt-4.1",
        "capabilities": {
            "type": "chat",
            "limits": {
                "max_context_window_tokens": 128000,
                "max_prompt_tokens": 128000,
                "max_output_tokens": 32768,
            },
        },
    },
    {
        "id": "claude-sonnet-4",
        "capabilities": {
            "type": "chat",
            "limits": {
                "max_context_window_tokens": 200000,
                "max_prompt_tokens": 200000,
                "max_output_tokens": 64000,
                "max_non_streaming_output_tokens": 32000,
            },
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

_SNAPSHOT_CATALOG = [
    {
        "id": "openai/gpt-4.1-mini",
        "model_picker_enabled": True,
        "supported_endpoints": ["/responses"],
        "capabilities": {
            "type": "chat",
            "supports": {"reasoning_effort": ["low", "medium", "high"]},
        },
    },
    {
        "id": "gpt-4.1",
        "model_picker_enabled": True,
        "supported_endpoints": ["/responses"],
        "capabilities": {
            "type": "chat",
            "supports": {"reasoning_effort": ["low", "medium", "high"]},
        },
    },
    {
        "id": "google/gemini-3.1-pro-preview",
        "model_picker_enabled": True,
        "supported_endpoints": ["/chat/completions"],
        "capabilities": {"type": "chat", "supports": {}},
    },
    {
        "id": "text-embedding-3-small",
        "model_picker_enabled": True,
        "capabilities": {"type": "embedding"},
    },
]


@pytest.fixture(autouse=True)
def _clear_cache():
    """Reset module-level cache before each test."""
    import hermes_cli.models as mod

    mod._copilot_context_cache = {}
    mod._copilot_context_cache_time = 0.0
    yield
    mod._copilot_context_cache = {}
    mod._copilot_context_cache_time = 0.0


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
    def test_cache_expires(self, mock_fetch):
        import hermes_cli.models as mod

        get_copilot_model_context("gpt-4.1")
        assert mock_fetch.call_count == 1

        # Expire the cache
        mod._copilot_context_cache_time = time.time() - 7200
        get_copilot_model_context("gpt-4.1")
        assert mock_fetch.call_count == 2

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=None)
    def test_returns_none_when_catalog_unavailable(self, mock_fetch):
        assert get_copilot_model_context("gpt-4.1") is None

    @patch("hermes_cli.models.fetch_github_model_catalog", return_value=[])
    def test_returns_none_for_empty_catalog(self, mock_fetch):
        assert get_copilot_model_context("gpt-4.1") is None


class TestCopilotInventorySnapshot:
    """Tests for the normalized Copilot inventory snapshot helper."""

    def test_normalizes_aliases_and_per_source_gemini_coverage(self):
        snapshot = build_copilot_inventory_snapshot(
            _SNAPSHOT_CATALOG,
            captured_at=1234.5,
        )

        assert snapshot["freshness"]["state"] == "live"
        assert snapshot["model_ids"] == [
            "gpt-4.1",
            "google/gemini-3.1-pro-preview",
        ]

        gpt = snapshot["models"]["gpt-4.1"]
        assert gpt["first_seen"] == 1234.5
        assert gpt["last_seen"] == 1234.5
        assert set(gpt["raw_aliases"]) == {"gpt-4.1", "openai/gpt-4.1-mini"}
        assert gpt["sources"][0]["google"] is False
        assert gpt["sources"][0]["gemini"] is False
        assert gpt["coverage"]["per_source"]["githubcopilot/models"]["gemini"] is False

        gemini = snapshot["models"]["google/gemini-3.1-pro-preview"]
        assert gemini["sources"][0]["google"] is True
        assert gemini["sources"][0]["gemini"] is True
        assert gemini["coverage"]["google"] is True
        assert gemini["coverage"]["gemini"] is True

        assert any(
            item["raw_id"] == "text-embedding-3-small" and item["included"] is False
            for item in snapshot["raw_evidence"]
        )

    def test_records_separate_limit_fields_with_source_and_freshness(self):
        snapshot = build_copilot_inventory_snapshot(
            _SAMPLE_CATALOG,
            captured_at=1234.5,
        )

        opus_limits = snapshot["models"]["claude-opus-4.6-1m"]["limits"]
        assert opus_limits["prompt_budget"] == 1_000_000
        assert opus_limits["total_context_window"] == 1_000_000
        assert opus_limits["max_output_tokens"] == 64_000
        assert opus_limits["max_non_streaming_output_tokens"] == 32_000
        assert opus_limits["source"] == "githubcopilot/models"
        assert opus_limits["captured_at"] == 1234.5
        assert opus_limits["freshness"]["state"] == "live"
        assert opus_limits["raw_keys"]["total_context_window"] == "max_context_window_tokens"

        gpt_limits = snapshot["models"]["gpt-4.1"]["limits"]
        assert gpt_limits["total_context_window"] == 128_000
        assert gpt_limits["max_non_streaming_output_tokens"] is None
        assert gpt_limits["freshness"]["state"] == "live"

    def test_cached_snapshot_preserves_last_known_good_inventory(
        self, tmp_path, monkeypatch
    ):
        import hermes_cli.models as mod

        monkeypatch.setattr(
            mod,
            "_copilot_inventory_cache_path",
            lambda: tmp_path / "copilot_inventory_cache.json",
        )

        responses = iter([_SNAPSHOT_CATALOG, None])
        monkeypatch.setattr(
            mod,
            "_fetch_github_model_catalog_items",
            lambda api_key=None, timeout=5.0: next(responses),
        )

        first = mod.cached_copilot_inventory_snapshot(force_refresh=True)
        second = mod.cached_copilot_inventory_snapshot(force_refresh=True)

        assert first["model_ids"] == second["model_ids"]
        assert second["freshness"]["state"] == "stale"
        assert second["freshness"]["used_last_known_good"] is True
        assert second["models"]["gpt-4.1"]["raw_aliases"] == first["models"]["gpt-4.1"]["raw_aliases"]
        assert second["models"]["gpt-4.1"]["limits"]["freshness"]["state"] == "stale"


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
