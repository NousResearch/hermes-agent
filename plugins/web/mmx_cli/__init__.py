"""MiniMax CLI web search provider — bundled, auto-loaded.

Backed by the `mmx` CLI (npm install -g mmx-cli). Authentication uses the
existing MINIMAX_API_KEY (same key the model gateway uses), charged against
the MiniMax Token Plan weekly quota. This provider only supports search —
mmx-cli does not have a content-extract command, so `web.extract_backend`
should remain set to `tavily` or another extract-capable provider.
"""
from __future__ import annotations

from plugins.web.mmx_cli.provider import MMXCliWebSearchProvider


def register(ctx) -> None:
    """Register the mmx-cli provider with the plugin context."""
    ctx.register_web_search_provider(MMXCliWebSearchProvider())
