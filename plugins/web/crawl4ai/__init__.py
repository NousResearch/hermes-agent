"""Crawl4ai web extract plugin — bundled, auto-loaded."""

from __future__ import annotations


def register(ctx):
    """Register the Crawl4ai provider with the plugin context."""
    from plugins.web.crawl4ai.provider import Crawl4aiWebExtractProvider
    ctx.register_web_search_provider(Crawl4aiWebExtractProvider())