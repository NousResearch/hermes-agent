"""Perplexity web plugin — bundled, auto-loaded.

Two capabilities, registered together:

* ``web_search`` backend via Perplexity's Search API (``POST /search``) —
  ranked links. Search only; the dispatcher in :mod:`tools.web_tools`
  handles the async wrap when the caller is async.
* ``perplexity_ask`` tool via Perplexity's Sonar Chat Completions API
  (``POST /chat/completions``) — a finished, citation-grounded answer in one
  call. See :mod:`plugins.web.perplexity.ask`.
"""

from __future__ import annotations

from plugins.web.perplexity.ask import register_ask_tool
from plugins.web.perplexity.provider import PerplexityWebSearchProvider


def register(ctx) -> None:
    """Register the Perplexity search backend and the Sonar ``ask`` tool."""
    ctx.register_web_search_provider(PerplexityWebSearchProvider())
    register_ask_tool(ctx)
