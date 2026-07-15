"""Baidu AI Search (Qianfan) plugin — bundled, auto-loaded.

Provides Chinese-optimized web search via Baidu AI Search Engine.
"""

from __future__ import annotations

from plugins.web.baidu.provider import BaiduWebSearchProvider


def register(ctx) -> None:
    """Register the Baidu Search provider with the plugin context."""
    ctx.register_web_search_provider(BaiduWebSearchProvider())
