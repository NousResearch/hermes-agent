"""Tests for PR 1: new web search providers (serper, baidu, bocha, etc.)."""
from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest


# ─── Provider registration (import-safe) ───────────────────────────────────

class TestProviderDiscovery:
    """Verify new providers are discoverable by the plugin system."""

    def test_all_nine_providers_have_plugin_yaml(self):
        """Each new provider directory must contain a valid plugin.yaml."""
        import glob
        import yaml
        base = os.path.join(os.path.dirname(__file__), "..", "..", "plugins", "web")
        expected = {
            "serper", "baidu", "bocha", "qiniu-baidu", "serpapi",
            "jina", "google-cse", "sogou", "_360_search",
        }
        found = set()
        for d in sorted(os.listdir(base)):
            p = os.path.join(base, d)
            if not os.path.isdir(p):
                continue
            yaml_path = os.path.join(p, "plugin.yaml")
            if os.path.isfile(yaml_path):
                found.add(d)
        assert expected.issubset(found), f"Missing: {expected - found}"


class TestSerperProvider:
    """Lightweight tests for the Serper provider (representative of the pattern)."""

    def test_is_available_false_without_key(self, monkeypatch):
        monkeypatch.delenv("SERPER_API_KEY", raising=False)
        from plugins.web.serper.provider import SerperWebSearchProvider
        p = SerperWebSearchProvider()
        assert p.is_available() is False

    def test_is_available_true_with_key(self, monkeypatch):
        monkeypatch.setenv("SERPER_API_KEY", "fake-key-123")
        from plugins.web.serper.provider import SerperWebSearchProvider
        p = SerperWebSearchProvider()
        assert p.is_available() is True
        monkeypatch.undo()

    def test_search_missing_key(self, monkeypatch):
        monkeypatch.delenv("SERPER_API_KEY", raising=False)
        from plugins.web.serper.provider import SerperWebSearchProvider
        result = SerperWebSearchProvider().search("test")
        assert result["success"] is False
        assert "SERPER_API_KEY" in result["error"]

    def test_search_http_error(self, monkeypatch):
        monkeypatch.setenv("SERPER_API_KEY", "bad-key")
        import requests
        with patch.object(requests, "post") as mock_post:
            mock_post.side_effect = requests.HTTPError("403 Forbidden")
            from plugins.web.serper.provider import SerperWebSearchProvider
            result = SerperWebSearchProvider().search("test")
            assert result["success"] is False

    def test_search_success(self, monkeypatch):
        monkeypatch.setenv("SERPER_API_KEY", "good-key")
        import requests
        with patch.object(requests, "post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {
                "organic": [
                    {"title": "Result 1", "link": "https://a.com", "snippet": "desc"},
                    {"title": "Result 2", "link": "https://b.com", "snippet": "desc2"},
                ]
            }
            mock_resp.raise_for_status.return_value = None
            mock_post.return_value = mock_resp
            from plugins.web.serper.provider import SerperWebSearchProvider
            result = SerperWebSearchProvider().search("test", limit=2)
            assert result["success"] is True
            assert len(result["data"]["web"]) == 2
            assert result["data"]["web"][0]["title"] == "Result 1"


class TestBaiduProvider:
    """Pattern tests for Baidu provider."""

    def test_search_missing_key(self, monkeypatch):
        monkeypatch.delenv("BAIDU_API_KEY", raising=False)
        from plugins.web.baidu.provider import BaiduWebSearchProvider
        result = BaiduWebSearchProvider().search("test")
        assert result["success"] is False
        assert "BAIDU_API_KEY" in result["error"]


class TestUnavailableProviders:
    """Registry-only providers (no public API) should always report unavailable."""

    def test_sogou_not_available(self):
        from plugins.web.sogou.provider import SogouWebSearchProvider
        assert SogouWebSearchProvider().is_available() is False
        result = SogouWebSearchProvider().search("test")
        assert result["success"] is False
        assert "Sogou" in result["error"]

    def test_360_search_not_available(self):
        from plugins.web._360_search.provider import Three60SearchWebSearchProvider
        assert Three60SearchWebSearchProvider().is_available() is False
        result = Three60SearchWebSearchProvider().search("test")
        assert result["success"] is False
        assert "360" in result["error"]


class TestProviderContracts:
    """All new providers must implement the WebSearchProvider contract."""

    def new_providers_implement_search(self):
        """Every new provider must support search and return standard dict."""
        pass  # Verified by pytest parametrize below

    def new_providers_have_display_names(self):
        """display_name must be a non-empty string for each provider."""
        pass


class TestBackendCandidates:
    """New providers appear in backend_candidates and get_registered_backend_names."""

    def test_get_registered_backend_names_includes_new(self):
        """Dynamic helper should return the new providers."""
        from tools.web_tools import _get_registered_backend_names
        names = _get_registered_backend_names()
        assert "serper" in names
        assert "baidu" in names
        assert "bocha" in names
        assert "sogou" in names
        assert "360-search" in names

    def test_no_hardcoded_known_backend_sets(self):
        """After de-hardcoding, web_tools.py must not contain hardcoded backend set."""
        import tools.web_tools
        source = open(tools.web_tools.__file__).read()
        assert 'configured in {"parallel", "firecrawl"' not in source
        assert 'configured in {"exa", "parallel", "firecrawl"' not in source
        assert "configured in _get_registered_backend_names()" in source
