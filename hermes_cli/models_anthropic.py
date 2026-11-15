"""Anthropic ``/v1/models`` paging: page URLs and cursor continuation.

Re-imported into ``hermes_cli.models`` for the picker fetcher (``_fetch_anthropic_models``) and
the Anthropic provider plugin's ``fetch_models``.
"""

from __future__ import annotations

import urllib.parse
from typing import Any, Optional


def _anthropic_models_url(base_url: Optional[str] = None, *, after_id: Optional[str] = None) -> str:
    """Anthropic ``/v1/models`` page URL. The endpoint is cursor-paginated with a default page of
    20 (smaller than the live catalog), so every request asks for the maximum page size and
    ``after_id`` continues from a previous page's ``last_id``."""
    endpoint = str(base_url or "https://api.anthropic.com").strip().rstrip("/")
    url = endpoint + ("/models" if endpoint.endswith("/v1") else "/v1/models")
    params = {"limit": "1000"}
    if after_id:
        params["after_id"] = after_id
    return url + ("&" if "?" in url else "?") + urllib.parse.urlencode(params)


_ANTHROPIC_MODELS_MAX_PAGES = 20


def _anthropic_next_cursor(page: Any, seen_cursors: set[str]) -> Optional[str]:
    """``last_id`` to continue from, or None when the page is final or the server repeats a
    cursor (which would otherwise loop forever)."""
    if not isinstance(page, dict) or page.get("has_more") is not True:
        return None
    last_id = page.get("last_id")
    if not isinstance(last_id, str) or not last_id or last_id in seen_cursors:
        return None
    seen_cursors.add(last_id)
    return last_id
