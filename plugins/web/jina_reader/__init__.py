"""Jina Reader plugin — bundled, auto-loaded."""
from __future__ import annotations

from plugins.web.jina_reader.provider import JinaReaderWebSearchProvider


def register(ctx) -> None:
    """Register the Jina Reader extract provider."""
    ctx.register_web_search_provider(JinaReaderWebSearchProvider())
