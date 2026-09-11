"""MrScraper rendered-page plugin."""

from __future__ import annotations

from plugins.browser.mrscraper.provider import (
    MRSCRAPER_FETCH_RENDERED_HTML_SCHEMA,
    handle_fetch_rendered_html,
)
from plugins.mrscraper_client import is_mrscraper_available


def register(ctx) -> None:
    """Register the rendered-page tool without claiming CDP compatibility."""
    ctx.register_tool(
        name="mrscraper_fetch_rendered_html",
        toolset="mrscraper",
        schema=MRSCRAPER_FETCH_RENDERED_HTML_SCHEMA,
        handler=handle_fetch_rendered_html,
        check_fn=is_mrscraper_available,
        requires_env=["MRSCRAPER_API_TOKEN"],
        emoji="🕸️",
    )
