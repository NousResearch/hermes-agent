"""Quota-fallback web search plugin — bundled, auto-loaded.

Tries child providers (brave-free, exa, tavily, searxng) in sequence,
falling through on quota exhaustion, rate limits, and transient errors.
Configure via ``web.search_backend: quota-fallback`` in config.yaml.
"""

from __future__ import annotations

from plugins.web.quota_fallback.provider import QuotaFallbackWebSearchProvider


def register(ctx) -> None:
    """Register the quota-fallback provider with the plugin context."""
    ctx.register_web_search_provider(QuotaFallbackWebSearchProvider())
