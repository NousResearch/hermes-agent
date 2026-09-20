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
    def test_cache_expires(self, mock_fetch):
        import hermes_cli.models as mod

        get_copilot_model_context("gpt-4.1")
        assert mock_fetch.call_count == 1

        # Expire the cache
        mod._copilot_context_cache_time = time.time() - 7200
        get_copilot_model_context("gpt-4.1")
        assert mock_fetch.call_count == 2



    @patch("hermes_cli.models._urlopen_model_catalog_request")
    def test_fetch_github_model_catalog_uses_short_lived_cache(self, mock_urlopen):
        import json as _json
        import hermes_cli.models as mod

        mod._github_model_catalog_cache = None
        mod._github_model_catalog_cache_key = None
        mod._github_model_catalog_cache_time = 0.0

        payload = {
            "data": [
                {
                    "id": "gpt-4.1",
                    "model_picker_enabled": True,
                    "supported_endpoints": ["/chat/completions"],
                }
            ]
        }

        class _Resp:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self):
                return _json.dumps(payload).encode()

        mock_urlopen.return_value = _Resp()

        first = mod.fetch_github_model_catalog(api_key="token")
        second = mod.fetch_github_model_catalog(api_key="token")

        assert [item["id"] for item in first] == ["gpt-4.1"]
        assert [item["id"] for item in second] == ["gpt-4.1"]
        assert mock_urlopen.call_count == 1

        # Cached copies are independent — mutating the result must not
        # poison the cache.
        second[0]["id"] = "mutated"
        third = mod.fetch_github_model_catalog(api_key="token")
        assert [item["id"] for item in third] == ["gpt-4.1"]
        assert mock_urlopen.call_count == 1

    @patch("hermes_cli.models._urlopen_model_catalog_request")
    def test_fetch_github_model_catalog_cache_expires_after_ttl(self, mock_urlopen):
        import json as _json
        import time as _time
        import hermes_cli.models as mod

        mod._github_model_catalog_cache = None
        mod._github_model_catalog_cache_key = None
        mod._github_model_catalog_cache_time = 0.0

        payload = {
            "data": [
                {
                    "id": "gpt-4.1",
                    "model_picker_enabled": True,
                    "supported_endpoints": ["/chat/completions"],
                }
            ]
        }

        class _Resp:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self):
                return _json.dumps(payload).encode()

        mock_urlopen.return_value = _Resp()

        mod.fetch_github_model_catalog(api_key="token")
        assert mock_urlopen.call_count == 1

        # Age the entry past the TTL (monotonic clock) — next call re-fetches.
        mod._github_model_catalog_cache_time = (
            _time.monotonic() - mod._GITHUB_MODEL_CATALOG_CACHE_TTL - 1
        )
        mod.fetch_github_model_catalog(api_key="token")
        assert mock_urlopen.call_count == 2

    @patch("hermes_cli.models._urlopen_model_catalog_request")
    def test_fetch_github_model_catalog_cache_misses_on_credential_change(self, mock_urlopen):
        import json as _json
        import hermes_cli.models as mod

        mod._github_model_catalog_cache = None
        mod._github_model_catalog_cache_key = None
        mod._github_model_catalog_cache_time = 0.0

        payload = {
            "data": [
                {
                    "id": "gpt-4.1",
                    "model_picker_enabled": True,
                    "supported_endpoints": ["/chat/completions"],
                }
            ]
        }

        class _Resp:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self):
                return _json.dumps(payload).encode()

        mock_urlopen.return_value = _Resp()

        mod.fetch_github_model_catalog(api_key="token-a")
        assert mock_urlopen.call_count == 1
        # A different token must not be served the previous account's catalog.
        mod.fetch_github_model_catalog(api_key="token-b")
        assert mock_urlopen.call_count == 2

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


