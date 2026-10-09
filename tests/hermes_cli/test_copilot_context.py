"""Tests for lower-owned Copilot live catalogue/context resolution."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

import models.catalog_github as github_catalog
import models.metadata.github as github_metadata
from models.metadata.github import github_model_context_length


_SAMPLE_CATALOG = [
    {
        "id": "claude-opus-4.6-1m",
        "capabilities": {
            "type": "chat",
            "limits": {"max_prompt_tokens": 1_000_000, "max_output_tokens": 64_000},
        },
    },
    {
        "id": "gpt-4.1",
        "capabilities": {
            "type": "chat",
            "limits": {"max_prompt_tokens": 128_000, "max_output_tokens": 32_768},
        },
    },
    {
        "id": "claude-sonnet-4",
        "capabilities": {
            "type": "chat",
            "limits": {"max_prompt_tokens": 200_000, "max_output_tokens": 64_000},
        },
    },
    {"id": "model-without-limits", "capabilities": {"type": "chat"}},
    {
        "id": "model-zero-limit",
        "capabilities": {"type": "chat", "limits": {"max_prompt_tokens": 0}},
    },
]


@pytest.fixture(autouse=True)
def _clear_cache():
    github_catalog.reset_github_model_catalog_cache()
    github_metadata.reset_github_context_cache()
    yield
    github_catalog.reset_github_model_catalog_cache()
    github_metadata.reset_github_context_cache()


class TestGithubModelContext:
    @patch("models.metadata.github.fetch_github_model_catalog", return_value=_SAMPLE_CATALOG)
    def test_returns_max_prompt_tokens(self, mock_fetch):
        assert github_model_context_length("claude-opus-4.6-1m") == 1_000_000
        assert github_model_context_length("gpt-4.1") == 128_000
        assert mock_fetch.call_count == 1

    @patch("models.metadata.github.fetch_github_model_catalog", return_value=_SAMPLE_CATALOG)
    def test_cache_expires(self, mock_fetch):
        github_model_context_length("gpt-4.1")
        assert mock_fetch.call_count == 1
        github_metadata._github_context_cache_time = (
            time.monotonic() - github_metadata._GITHUB_CONTEXT_CACHE_TTL - 1
        )
        github_model_context_length("gpt-4.1")
        assert mock_fetch.call_count == 2

    @patch("models.metadata.github.fetch_github_model_catalog", return_value=[])
    def test_returns_none_for_empty_catalog(self, mock_fetch):
        assert github_model_context_length("gpt-4.1") is None
        mock_fetch.assert_called_once()


def _catalog_payload():
    return {
        "data": [
            {
                "id": "gpt-4.1",
                "model_picker_enabled": True,
                "supported_endpoints": ["/chat/completions"],
            }
        ]
    }


class TestGithubCatalogCache:
    def test_short_lived_cache_returns_independent_copies(self, monkeypatch):
        fetches = []
        monkeypatch.setattr(
            github_catalog,
            "_fetch_json",
            lambda *a, **k: fetches.append((a, k)) or _catalog_payload(),
        )

        first = github_catalog.fetch_github_model_catalog(api_key="token")
        second = github_catalog.fetch_github_model_catalog(api_key="token")

        assert [item["id"] for item in first] == ["gpt-4.1"]
        assert [item["id"] for item in second] == ["gpt-4.1"]
        assert len(fetches) == 1

        second[0]["id"] = "mutated"
        third = github_catalog.fetch_github_model_catalog(api_key="token")
        assert [item["id"] for item in third] == ["gpt-4.1"]
        assert len(fetches) == 1

    def test_cache_expires_after_ttl(self, monkeypatch):
        fetches = []
        monkeypatch.setattr(
            github_catalog,
            "_fetch_json",
            lambda *a, **k: fetches.append((a, k)) or _catalog_payload(),
        )
        github_catalog.fetch_github_model_catalog(api_key="token")
        assert len(fetches) == 1

        github_catalog._github_model_catalog_cache_time = (
            time.monotonic() - github_catalog._GITHUB_MODEL_CATALOG_CACHE_TTL - 1
        )
        github_catalog.fetch_github_model_catalog(api_key="token")
        assert len(fetches) == 2

    def test_cache_misses_on_credential_change(self, monkeypatch):
        fetches = []
        monkeypatch.setattr(
            github_catalog,
            "_fetch_json",
            lambda *a, **k: fetches.append((a, k)) or _catalog_payload(),
        )
        github_catalog.fetch_github_model_catalog(api_key="token-a")
        github_catalog.fetch_github_model_catalog(api_key="token-b")
        assert len(fetches) == 2


class TestModelMetadataCopilotIntegration:
    @patch("models.metadata.github.fetch_github_model_catalog", return_value=_SAMPLE_CATALOG)
    def test_copilot_provider_uses_live_api(self, mock_fetch):
        from models.metadata.context import get_model_context_length

        ctx = get_model_context_length("claude-opus-4.6-1m", provider="copilot")
        assert ctx == 1_000_000
