"""Venice web search + extract — bundled, auto-loaded.

Routes :func:`tools.web_tools.web_search` to Venice's ``/augment/search``
(Brave / Google with privacy preservation) and :func:`web_extract` to
``/augment/scrape`` (page → markdown). Same VENICE_API_KEY as the rest
of the Venice multi-modal stack.
"""

from __future__ import annotations

from plugins.web.venice.provider import VeniceWebSearchProvider


def register(ctx) -> None:
    ctx.register_web_search_provider(VeniceWebSearchProvider())
