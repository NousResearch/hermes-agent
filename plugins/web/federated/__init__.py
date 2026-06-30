"""Federated search plugin — bundled, auto-loaded.

Aggregates search results from multiple configured sub-backends and uses an
LLM to rank them by relevance. Configuration lives under ``web.federated``
in config.yaml.
"""

from __future__ import annotations

from plugins.web.federated.provider import FederatedSearchProvider


def register(ctx) -> None:
    """Register the federated search provider with the plugin context."""
    ctx.register_web_search_provider(FederatedSearchProvider())
