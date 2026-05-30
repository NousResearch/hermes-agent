"""Tests for Firecrawl structured extraction support."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from plugins.web.firecrawl import provider as firecrawl_provider


@pytest.mark.asyncio
async def test_firecrawl_extract_accepts_json_schema(monkeypatch):
    schema = {"type": "object", "properties": {"title": {"type": "string"}}}
    client = Mock()
    client.scrape.return_value = {
        "json": {"title": "LG Gram", "price": "$799"},
        "markdown": "# LG Gram\n$799",
        "metadata": {"title": "LG Gram", "sourceURL": "https://www.amazon.com/dp/B0DZZWMB2L"},
    }
    monkeypatch.setattr(firecrawl_provider, "_get_firecrawl_client", lambda: client)
    monkeypatch.setattr(firecrawl_provider, "check_website_access", lambda url: None)

    result = await firecrawl_provider.FirecrawlWebSearchProvider().extract(
        ["https://www.amazon.com/dp/B0DZZWMB2L"],
        json_schema=schema,
        include_markdown=True,
    )

    client.scrape.assert_called_once_with(
        url="https://www.amazon.com/dp/B0DZZWMB2L",
        formats=[{"type": "json", "schema": schema}, "markdown"],
    )
    assert result[0]["structured_data"] == {"title": "LG Gram", "price": "$799"}
    assert result[0]["content"] == "# LG Gram\n$799"


@pytest.mark.asyncio
async def test_firecrawl_extract_markdown_behavior_unchanged(monkeypatch):
    client = Mock()
    client.scrape.return_value = {
        "markdown": "# Example",
        "metadata": {"title": "Example", "sourceURL": "https://example.com"},
    }
    monkeypatch.setattr(firecrawl_provider, "_get_firecrawl_client", lambda: client)
    monkeypatch.setattr(firecrawl_provider, "check_website_access", lambda url: None)

    result = await firecrawl_provider.FirecrawlWebSearchProvider().extract(
        ["https://example.com"], format="markdown"
    )

    client.scrape.assert_called_once_with(url="https://example.com", formats=["markdown"])
    assert "structured_data" not in result[0]
    assert result[0]["content"] == "# Example"
