#!/usr/bin/env python3
"""
Integration test for the Crawl4ai web extract provider.

This test requires a running Crawl4ai instance with valid credentials.
Set the following environment variables to run:
- CRAWL4AI_URL: URL of your Crawl4ai instance (e.g., http://localhost:11235)
- CRAWL4AI_API_TOKEN: JWT API token for authentication

Run with:
    pytest tests/integration/test_crawl4ai_provider.py -v -s
"""

import os
import pytest
import json
import asyncio

pytestmark = pytest.mark.integration


class TestCrawl4aiIntegration:
    """Integration tests for Crawl4ai provider against a live instance."""

    @pytest.fixture(scope="class")
    def crawl4ai_config(self):
        """Load Crawl4ai configuration from environment."""
        url = os.getenv("CRAWL4AI_URL")
        token = os.getenv("CRAWL4AI_API_TOKEN")

        if not url or not token:
            pytest.skip("CRAWL4AI_URL and CRAWL4AI_API_TOKEN must be set for integration tests")

        return {"url": url.rstrip("/"), "token": token}

    @pytest.fixture(scope="class")
    def provider(self, crawl4ai_config):
        """Create a Crawl4ai provider instance."""
        from plugins.web.crawl4ai.provider import Crawl4aiWebExtractProvider

        provider = Crawl4aiWebExtractProvider()
        # Verify it's available with the provided config
        assert provider.is_available(), "Crawl4ai provider not available with provided config"
        return provider

    @pytest.mark.asyncio
    async def test_extract_single_url(self, provider):
        """Test extracting content from a single URL."""
        result = await provider.extract(["https://example.com"])

        assert len(result) == 1
        assert result[0]["url"] == "https://example.com"
        assert "content" in result[0]
        assert "title" in result[0]
        assert result[0].get("error") is None or "error" not in result[0]
        assert len(result[0]["content"]) > 0

    @pytest.mark.asyncio
    async def test_extract_multiple_urls(self, provider):
        """Test extracting content from multiple URLs."""
        urls = [
            "https://example.com",
            "https://www.python.org"
        ]
        result = await provider.extract(urls)

        assert len(result) == 2
        for i, url in enumerate(urls):
            assert result[i]["url"] == url
            assert "content" in result[i]
            assert "title" in result[i]
            assert result[i].get("error") is None or "error" not in result[i]
            assert len(result[i]["content"]) > 0

    @pytest.mark.asyncio
    async def test_extract_markdown_format(self, provider):
        """Test that extracted content is in markdown format."""
        result = await provider.extract(["https://example.com"])

        assert len(result) == 1
        content = result[0]["content"]
        # Should contain markdown-like content (headers, etc.)
        assert isinstance(content, str)
        assert len(content) > 0

    @pytest.mark.asyncio
    async def test_extract_invalid_url(self, provider):
        """Test handling of invalid/private URLs (SSRF protection)."""
        # Use a private IP that should be blocked
        result = await provider.extract(["http://10.0.0.1/"])

        assert len(result) == 1
        assert result[0]["url"] == "http://10.0.0.1/"
        assert "error" in result[0]
        assert "blocked" in result[0]["error"].lower() or "private" in result[0]["error"].lower()

    def test_supports_extract(self, provider):
        """Test that provider reports extract capability."""
        assert provider.supports_extract() is True

    def test_supports_search_is_false(self, provider):
        """Test that provider correctly reports no search capability."""
        assert provider.supports_search() is False

    def test_provider_name(self, provider):
        """Test provider name is correct."""
        assert provider.name == "crawl4ai"

    def test_display_name(self, provider):
        """Test display name is correct."""
        assert provider.display_name == "Crawl4ai (self-hosted)"

    def test_setup_schema(self, provider):
        """Test setup schema includes required env vars."""
        schema = provider.get_setup_schema()
        assert schema["name"] == "Crawl4ai (self-hosted)"
        assert schema["badge"] == "self-hosted"
        env_var_keys = [v["key"] for v in schema["env_vars"]]
        assert "CRAWL4AI_URL" in env_var_keys
        assert "CRAWL4AI_API_TOKEN" in env_var_keys


class TestCrawl4aiWebToolsIntegration:
    """Integration tests for Crawl4ai via the web_tools module."""

    @pytest.fixture(scope="class")
    def crawl4ai_config(self):
        """Load Crawl4ai configuration from environment."""
        url = os.getenv("CRAWL4AI_URL")
        token = os.getenv("CRAWL4AI_API_TOKEN")

        if not url or not token:
            pytest.skip("CRAWL4AI_URL and CRAWL4AI_API_TOKEN must be set for integration tests")

        return {"url": url.rstrip("/"), "token": token}

    @pytest.fixture(autouse=True)
    def setup_web_backend(self, crawl4ai_config, monkeypatch):
        """Configure web_tools to use crawl4ai backend."""
        from tools import web_tools

        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"backend": "crawl4ai", "extract_backend": "crawl4ai"})
        monkeypatch.setenv("CRAWL4AI_URL", crawl4ai_config["url"])
        monkeypatch.setenv("CRAWL4AI_API_TOKEN", crawl4ai_config["token"])

        # Ensure plugins are loaded
        from tools.web_tools import _ensure_web_plugins_loaded
        _ensure_web_plugins_loaded()

    def test_check_web_api_key_returns_true(self, crawl4ai_config):
        """Test that check_web_api_key returns True with crawl4ai configured."""
        from tools.web_tools import check_web_api_key

        assert check_web_api_key() is True

    def test_is_backend_available_crawl4ai(self, crawl4ai_config):
        """Test that _is_backend_available returns True for crawl4ai."""
        from tools.web_tools import _is_backend_available

        assert _is_backend_available("crawl4ai") is True

    def test_get_backend_returns_crawl4ai(self, crawl4ai_config):
        """Test that _get_backend returns crawl4ai when configured."""
        from tools.web_tools import _get_backend, _get_extract_backend

        assert _get_backend() == "crawl4ai"
        assert _get_extract_backend() == "crawl4ai"

    @pytest.mark.asyncio
    async def test_web_extract_tool_works(self, crawl4ai_config):
        """Test web_extract_tool works with crawl4ai backend."""
        from tools.web_tools import web_extract_tool

        result_json = await web_extract_tool(["https://example.com"], format="markdown")
        result = json.loads(result_json)

        assert "results" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["url"] == "https://example.com"
        assert "content" in result["results"][0]
        assert len(result["results"][0]["content"]) > 0
        assert result["results"][0].get("error") is None

    @pytest.mark.asyncio
    async def test_web_search_tool_returns_error(self, crawl4ai_config):
        """Test web_search_tool returns clear error for extract-only backend."""
        from tools.web_tools import web_search_tool

        result_json = web_search_tool("test query")
        result = json.loads(result_json)

        assert result["success"] is False
        assert "extract-only" in result["error"].lower() or "crawl4ai" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_web_extract_multiple_urls(self, crawl4ai_config):
        """Test web_extract_tool with multiple URLs."""
        from tools.web_tools import web_extract_tool

        urls = ["https://example.com", "https://www.python.org"]
        result_json = await web_extract_tool(urls, format="markdown")
        result = json.loads(result_json)

        assert "results" in result
        assert len(result["results"]) == 2
        for i, url in enumerate(urls):
            assert result["results"][i]["url"] == url
            assert "content" in result["results"][i]
            assert len(result["results"][i]["content"]) > 0

    @pytest.mark.asyncio
    async def test_web_extract_with_char_limit(self, crawl4ai_config):
        """Test web_extract_tool respects char_limit parameter."""
        from tools.web_tools import web_extract_tool

        result_json = await web_extract_tool(
            ["https://www.python.org"],
            format="markdown",
            char_limit=500
        )
        result = json.loads(result_json)

        assert "results" in result
        assert len(result["results"]) == 1
        # Content should be truncated with footer
        content = result["results"][0]["content"]
        assert "[TRUNCATED]" in content or len(content) <= 500


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])