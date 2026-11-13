"""Perplexity web search + snippets plugin — bundled, auto-loaded.

Backed by the Perplexity Search API over httpx. Both search and extract are
sync; the dispatcher in :mod:`hermes_agent.tools.web_tools` handles the wrap when the
caller is async.
"""

from __future__ import annotations

from hermes_agent.plugins.web.perplexity.provider import PerplexityWebSearchProvider


def register(ctx) -> None:
    """Register the Perplexity provider with the plugin context."""
    ctx.register_web_search_provider(PerplexityWebSearchProvider())
