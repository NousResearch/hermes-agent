"""AIRIES Fetch web provider — non-automated HTTP + HTML parsing."""

from __future__ import annotations

from plugins.web.airies_fetch.provider import AriesFetchProvider


def register(ctx) -> None:
    ctx.register_web_search_provider(AriesFetchProvider())
