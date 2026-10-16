"""Opt-in, one-shot authenticated retry for Exa's anonymous HTTP 429s."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def retry_after_429(payload: dict[str, Any], timeout: int) -> str | None:
    """Return authenticated content, or None when paid fallback is not authorized.

    Called only after an HTTP 429 from Exa's fixed MCP endpoint. Never infer a
    spending decision from error text (which can echo a query or page content).
    """
    import requests

    from plugins.web._common import provider_env
    from plugins.web.keyless_mcp import EXA_MCP_URL, KeylessMCPError, _parse_mcp_body, _response_text
    from tools.web_tools import _load_web_config

    if _load_web_config().get("exa_paid_fallback") is not True:
        return None
    if payload.get("params", {}).get("name") not in {"web_search_exa", "web_fetch_exa"}:
        return None
    key = provider_env("EXA_API_KEY")
    if not key:
        return None
    logger.warning("Exa anonymous HTTP 429: attempting one authenticated retry (may incur charges)")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "User-Agent": "hermes-agent",
        "x-api-key": key,
    }
    try:
        # Do not forward a credential through a redirect or retry recursively.
        response = requests.post(EXA_MCP_URL, json=payload, headers=headers, timeout=timeout, allow_redirects=False)
        if not 200 <= response.status_code < 300:
            raise KeylessMCPError(f"HTTP {response.status_code}")
        return _parse_mcp_body(_response_text(response))
    except (requests.RequestException, KeylessMCPError) as exc:
        raise KeylessMCPError(f"Authenticated Exa retry failed: {str(exc).replace(key, '[REDACTED]')}") from None
