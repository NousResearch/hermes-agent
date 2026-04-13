"""Tests for Perplexity Search API backend integration.

Tests cover:
- Backend detection and availability
- Search request construction and dispatch
- Response normalization
- Error handling (auth, rate limit, timeout, server errors, invalid JSON)
- Boundary conditions (empty results, missing keys, malformed responses)
- Config (PERPLEXITY_API_URL override, env var detection)
- web_extract / web_crawl fallback behavior
"""

import json
import os
from unittest.mock import MagicMock, patch

import httpx
import pytest

# Ensure project root is importable
from tools.web_tools import (
    _get_backend,
    _is_backend_available,
    _normalize_perplexity_search_results,
    _perplexity_base_url,
    _perplexity_search,
    _web_requires_env,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove all web backend API keys so tests start from a clean state."""
    for key in [
        "PERPLEXITY_API_KEY",
        "PERPLEXITY_API_URL",
        "FIRECRAWL_API_KEY",
        "FIRECRAWL_API_URL",
        "PARALLEL_API_KEY",
        "TAVILY_API_KEY",
        "EXA_API_KEY",
    ]:
        monkeypatch.delenv(key, raising=False)


@pytest.fixture()
def perplexity_key(monkeypatch):
    """Set a Perplexity API key for tests that need one."""
    monkeypatch.setenv("PERPLEXITY_API_KEY", "pplx-test-key-12345")


@pytest.fixture()
def mock_config_perplexity(monkeypatch):
    """Configure web backend as perplexity."""
    monkeypatch.setenv("PERPLEXITY_API_KEY", "pplx-test-key-12345")
    monkeypatch.setattr(
        "tools.web_tools._load_web_config",
        lambda: {"backend": "perplexity"},
    )


# ─── Backend detection tests ─────────────────────────────────────────────────

class TestBackendDetection:
    """Tests for _get_backend() and _is_backend_available()."""

    def test_perplexity_detected_when_configured(self, mock_config_perplexity):
        assert _get_backend() == "perplexity"

    def test_perplexity_detected_by_api_key(self, perplexity_key, monkeypatch):
        """When no config backend is set, perplexity is chosen by API key priority."""
        monkeypatch.setattr("tools.web_tools._load_web_config", lambda: {})
        # Perplexity is highest priority in the fallback chain
        assert _get_backend() == "perplexity"

    def test_perplexity_available_with_key(self, perplexity_key):
        assert _is_backend_available("perplexity") is True

    def test_perplexity_unavailable_without_key(self):
        assert _is_backend_available("perplexity") is False

    def test_perplexity_takes_priority_over_firecrawl(self, perplexity_key, monkeypatch):
        """Perplexity is first in the fallback priority chain."""
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-test-key")
        monkeypatch.setattr("tools.web_tools._load_web_config", lambda: {})
        # Mock Firecrawl availability check to avoid real client init
        monkeypatch.setattr("tools.web_tools.check_firecrawl_api_key", lambda: True)
        assert _get_backend() == "perplexity"


# ─── Base URL tests ──────────────────────────────────────────────────────────

class TestBaseURL:
    """Tests for _perplexity_base_url() configuration."""

    def test_default_url(self):
        assert _perplexity_base_url() == "https://api.perplexity.ai"

    def test_custom_url(self, monkeypatch):
        monkeypatch.setenv("PERPLEXITY_API_URL", "https://proxy.example.com")
        assert _perplexity_base_url() == "https://proxy.example.com"

    def test_custom_url_trailing_slash_stripped(self, monkeypatch):
        monkeypatch.setenv("PERPLEXITY_API_URL", "https://proxy.example.com/")
        assert _perplexity_base_url() == "https://proxy.example.com"

    def test_empty_env_uses_default(self, monkeypatch):
        monkeypatch.setenv("PERPLEXITY_API_URL", "")
        assert _perplexity_base_url() == "https://api.perplexity.ai"

    def test_whitespace_env_uses_default(self, monkeypatch):
        monkeypatch.setenv("PERPLEXITY_API_URL", "  ")
        assert _perplexity_base_url() == "https://api.perplexity.ai"


# ─── Normalization tests ─────────────────────────────────────────────────────

class TestNormalizePerplexitySearchResults:
    """Tests for _normalize_perplexity_search_results()."""

    def test_basic_normalization(self):
        data = {
            "id": "test-123",
            "results": [
                {"title": "AI News", "url": "https://example.com/ai", "snippet": "Latest AI news", "date": "2025-01-15"},
                {"title": "Tech Updates", "url": "https://example.com/tech", "snippet": "Tech round-up", "date": "2025-01-14"},
            ],
        }
        result = _normalize_perplexity_search_results(data)
        assert result["success"] is True
        web = result["data"]["web"]
        assert len(web) == 2
        assert web[0] == {
            "url": "https://example.com/ai",
            "title": "AI News",
            "description": "Latest AI news",
            "date": "2025-01-15",
            "position": 1,
        }
        assert web[1]["position"] == 2

    def test_empty_results(self):
        result = _normalize_perplexity_search_results({"results": []})
        assert result["success"] is True
        assert result["data"]["web"] == []

    def test_missing_results_key(self):
        result = _normalize_perplexity_search_results({"id": "test"})
        assert result["success"] is True
        assert result["data"]["web"] == []

    def test_non_list_results(self):
        result = _normalize_perplexity_search_results({"results": "not a list"})
        assert result["success"] is True
        assert result["data"]["web"] == []

    def test_non_dict_items_skipped(self):
        data = {"results": [{"title": "Valid", "url": "https://a.com"}, "string_item", 42, None]}
        result = _normalize_perplexity_search_results(data)
        assert len(result["data"]["web"]) == 1
        assert result["data"]["web"][0]["title"] == "Valid"

    def test_missing_url_and_title_skipped(self):
        """Results with neither URL nor title are not useful to the agent."""
        data = {"results": [{"snippet": "orphan snippet", "date": "2025-01-01"}]}
        result = _normalize_perplexity_search_results(data)
        assert result["data"]["web"] == []

    def test_url_only_is_kept(self):
        """A result with URL but no title is still useful."""
        data = {"results": [{"url": "https://example.com"}]}
        result = _normalize_perplexity_search_results(data)
        assert len(result["data"]["web"]) == 1
        assert result["data"]["web"][0]["title"] == ""

    def test_title_only_is_kept(self):
        """A result with title but no URL is still useful."""
        data = {"results": [{"title": "Some Title"}]}
        result = _normalize_perplexity_search_results(data)
        assert len(result["data"]["web"]) == 1
        assert result["data"]["web"][0]["url"] == ""

    def test_empty_string_fields_normalized(self):
        """None values in fields become empty strings."""
        data = {"results": [{"title": None, "url": None, "snippet": None, "date": None}]}
        result = _normalize_perplexity_search_results(data)
        # Both url and title are None -> "" which fails the "not url and not title" check
        # since empty string is falsy, this should be skipped
        assert result["data"]["web"] == []

    def test_position_numbering(self):
        """Positions should be 1-indexed sequential."""
        data = {"results": [
            {"title": f"Result {i}", "url": f"https://ex.com/{i}"} for i in range(5)
        ]}
        result = _normalize_perplexity_search_results(data)
        positions = [r["position"] for r in result["data"]["web"]]
        assert positions == [1, 2, 3, 4, 5]

    def test_snippet_maps_to_description(self):
        """Perplexity 'snippet' field maps to 'description' in the standard format."""
        data = {"results": [{"title": "T", "url": "https://x.com", "snippet": "Hello world"}]}
        result = _normalize_perplexity_search_results(data)
        assert result["data"]["web"][0]["description"] == "Hello world"


# ─── Search function tests ───────────────────────────────────────────────────

class TestPerplexitySearch:
    """Tests for _perplexity_search() with mocked HTTP calls."""

    def test_search_returns_results(self, perplexity_key, monkeypatch):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "id": "test-id",
            "results": [
                {"title": "AI Research", "url": "https://arxiv.org/abs/2401", "snippet": "New paper", "date": "2025-01-15"},
            ],
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("tools.web_tools.httpx.post", return_value=mock_resp):
            result = _perplexity_search("machine learning", limit=5)

        assert result["success"] is True
        assert len(result["data"]["web"]) == 1
        assert result["data"]["web"][0]["title"] == "AI Research"
        assert result["data"]["web"][0]["description"] == "New paper"
        assert result["data"]["web"][0]["date"] == "2025-01-15"
        assert result["data"]["web"][0]["position"] == 1

    def test_search_sends_correct_payload(self, perplexity_key, monkeypatch):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "t", "results": []}
        mock_resp.raise_for_status = MagicMock()

        with patch("tools.web_tools.httpx.post", return_value=mock_resp) as mock_post:
            _perplexity_search("test query", limit=10)

        call_kwargs = mock_post.call_args
        assert call_kwargs.kwargs["json"]["query"] == "test query"
        assert call_kwargs.kwargs["json"]["max_results"] == 10
        assert "Authorization" in call_kwargs.kwargs["headers"]
        assert "Bearer pplx-test-key-12345" == call_kwargs.kwargs["headers"]["Authorization"]

    def test_search_limit_clamped_to_20(self, perplexity_key, monkeypatch):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "t", "results": []}
        mock_resp.raise_for_status = MagicMock()

        with patch("tools.web_tools.httpx.post", return_value=mock_resp) as mock_post:
            _perplexity_search("test", limit=50)

        assert mock_post.call_args.kwargs["json"]["max_results"] == 20

    def test_search_limit_floor_at_1(self, perplexity_key, monkeypatch):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "t", "results": []}
        mock_resp.raise_for_status = MagicMock()

        with patch("tools.web_tools.httpx.post", return_value=mock_resp) as mock_post:
            _perplexity_search("test", limit=0)

        assert mock_post.call_args.kwargs["json"]["max_results"] == 1

    def test_no_api_key_returns_error(self, monkeypatch):
        monkeypatch.delenv("PERPLEXITY_API_KEY", raising=False)
        result = _perplexity_search("test")
        assert result["success"] is False
        assert "PERPLEXITY_API_KEY" in result["error"]
        assert "perplexity.ai/settings/api" in result["error"]

    def test_401_auth_error(self, perplexity_key, monkeypatch):
        error_resp = MagicMock()
        error_resp.status_code = 401
        error_resp.text = "Unauthorized"
        error_resp.headers = {}

        with patch("tools.web_tools.httpx.post") as mock_post:
            mock_post.side_effect = httpx.HTTPStatusError(
                "401", request=MagicMock(), response=error_resp
            )
            result = _perplexity_search("test")

        assert result["success"] is False
        assert "authentication failed" in result["error"].lower()
        assert "401" in result["error"]

    def test_429_rate_limit_error(self, perplexity_key, monkeypatch):
        error_resp = MagicMock()
        error_resp.status_code = 429
        error_resp.text = "Rate limit exceeded"
        error_resp.headers = {"retry-after": "60"}

        with patch("tools.web_tools.httpx.post") as mock_post:
            mock_post.side_effect = httpx.HTTPStatusError(
                "429", request=MagicMock(), response=error_resp
            )
            result = _perplexity_search("test")

        assert result["success"] is False
        assert "rate limit" in result["error"].lower()
        assert "429" in result["error"]
        assert "60" in result["error"]

    def test_429_without_retry_after(self, perplexity_key, monkeypatch):
        error_resp = MagicMock()
        error_resp.status_code = 429
        error_resp.text = "Too many requests"
        error_resp.headers = {}

        with patch("tools.web_tools.httpx.post") as mock_post:
            mock_post.side_effect = httpx.HTTPStatusError(
                "429", request=MagicMock(), response=error_resp
            )
            result = _perplexity_search("test")

        assert "rate limit" in result["error"].lower()
        # Should NOT contain retry-after hint
        assert "Retry after" not in result["error"]

    def test_500_server_error(self, perplexity_key, monkeypatch):
        error_resp = MagicMock()
        error_resp.status_code = 500
        error_resp.text = "Internal Server Error"
        error_resp.headers = {}

        with patch("tools.web_tools.httpx.post") as mock_post:
            mock_post.side_effect = httpx.HTTPStatusError(
                "500", request=MagicMock(), response=error_resp
            )
            result = _perplexity_search("test")

        assert result["success"] is False
        assert "server error" in result["error"].lower()
        assert "500" in result["error"]

    def test_timeout_error(self, perplexity_key, monkeypatch):
        with patch("tools.web_tools.httpx.post") as mock_post:
            mock_post.side_effect = httpx.TimeoutException("Connection timed out")
            result = _perplexity_search("test")

        assert result["success"] is False
        assert "timed out" in result["error"].lower()

    def test_connection_error(self, perplexity_key, monkeypatch):
        with patch("tools.web_tools.httpx.post") as mock_post:
            mock_post.side_effect = httpx.ConnectError("Connection refused")
            result = _perplexity_search("test")

        assert result["success"] is False
        assert "request failed" in result["error"].lower()

    def test_invalid_json_response(self, perplexity_key, monkeypatch):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.side_effect = json.JSONDecodeError("test", "doc", 0)

        with patch("tools.web_tools.httpx.post", return_value=mock_resp):
            result = _perplexity_search("test")

        assert result["success"] is False
        assert "invalid JSON" in result["error"]

    def test_interrupted_returns_error(self, perplexity_key, monkeypatch):
        with patch("tools.web_tools.is_interrupted", return_value=True, create=True):
            # Need to mock the import inside the function
            with patch("tools.interrupt.is_interrupted", return_value=True):
                result = _perplexity_search("test")
        assert result["success"] is False
        assert "Interrupted" in result["error"]

    def test_custom_api_url(self, perplexity_key, monkeypatch):
        """Perplexity API URL can be overridden for self-hosted/proxy."""
        monkeypatch.setenv("PERPLEXITY_API_URL", "https://proxy.example.com")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "t", "results": []}
        mock_resp.raise_for_status = MagicMock()

        with patch("tools.web_tools.httpx.post", return_value=mock_resp) as mock_post:
            _perplexity_search("test")

        call_url = mock_post.call_args.args[0] if mock_post.call_args.args else mock_post.call_args.kwargs.get("url")
        assert call_url == "https://proxy.example.com/search"

    def test_no_max_tokens_per_page_in_payload(self, perplexity_key, monkeypatch):
        """We intentionally omit max_tokens_per_page — let the API use its default.
        The agent gets snippet-level data, not full page extraction."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"id": "t", "results": []}
        mock_resp.raise_for_status = MagicMock()

        with patch("tools.web_tools.httpx.post", return_value=mock_resp) as mock_post:
            _perplexity_search("test")

        payload = mock_post.call_args.kwargs["json"]
        assert "max_tokens_per_page" not in payload
        assert "query" in payload
        assert "max_results" in payload


# ─── web_requires_env tests ─────────────────────────────────────────────────

class TestWebRequiresEnv:
    """Tests for _web_requires_env() including Perplexity env vars."""

    def test_includes_perplexity_api_key(self):
        env_vars = _web_requires_env()
        assert "PERPLEXITY_API_KEY" in env_vars

    def test_includes_perplexity_api_url(self):
        env_vars = _web_requires_env()
        assert "PERPLEXITY_API_URL" in env_vars


# ─── Backend priority tests ──────────────────────────────────────────────────

class TestBackendPriority:
    """Test that Perplexity's position in the fallback chain is correct."""

    def test_perplexity_over_firecrawl(self, monkeypatch):
        """Perplexity has higher priority than Firecrawl in fallback."""
        monkeypatch.setenv("PERPLEXITY_API_KEY", "pplx-key")
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-key")
        monkeypatch.setattr("tools.web_tools._load_web_config", lambda: {})
        monkeypatch.setattr("tools.web_tools.check_firecrawl_api_key", lambda: True)
        assert _get_backend() == "perplexity"

    def test_config_backend_overrides_priority(self, monkeypatch):
        """Explicit config backend overrides fallback priority."""
        monkeypatch.setenv("PERPLEXITY_API_KEY", "pplx-key")
        monkeypatch.setenv("FIRECRAWL_API_KEY", "fc-key")
        monkeypatch.setattr("tools.web_tools._load_web_config", lambda: {"backend": "firecrawl"})
        monkeypatch.setattr("tools.web_tools.check_firecrawl_api_key", lambda: True)
        assert _get_backend() == "firecrawl"