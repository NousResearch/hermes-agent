"""MiniMax (Token Plan) web search — user plugin.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. Hits the
MiniMax Token Plan ``POST /v1/coding_plan/search`` endpoint and normalizes
the response to Hermes' canonical ``{success, data: {web: [...]}}`` shape.

Verified response shape (live test, 2026-06-23):
    {
      "organic": [
        {"title": str, "link": str, "snippet": str, "date": str}, ...
      ],
      "related_searches": [...]
    }
On API-level errors the response carries ``base_resp.status_code != 0``;
on transport errors the HTTP call raises.

Auth env var (any one of these, first non-empty wins):
    MINIMAX_CODE_PLAN_KEY        # preferred — explicit Token Plan key
    MINIMAX_CODING_API_KEY       # OpenClaw-style alias
    MINIMAX_OAUTH_TOKEN          # OpenClaw-style alias
    MINIMAX_API_KEY              # generic — works if the key is Token-Plan-enabled

Region (must match the key's region, or the API returns 1004 invalid key):
    MINIMAX_API_HOST             # default: https://api.minimax.io  (global)
                                 #   CN:   https://api.minimaxi.com

Reference: https://github.com/MiniMax-AI/MiniMax-Coding-Plan-MCP
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_GLOBAL_HOST = "https://api.minimax.io"
_CN_HOST = "https://api.minimaxi.com"
_SEARCH_PATH = "/v1/coding_plan/search"

# Aliases checked in order; first non-empty value is used. Mirrors OpenClaw's
# minimax-search config so the same .env works in both agents.
_KEY_ALIASES: tuple = (
    "MINIMAX_CODE_PLAN_KEY",
    "MINIMAX_CODING_API_KEY",
    "MINIMAX_OAUTH_TOKEN",
    "MINIMAX_API_KEY",
)


def _resolve_api_key() -> str:
    for k in _KEY_ALIASES:
        v = os.getenv(k, "").strip()
        if v:
            return v
    return ""


def _resolve_host() -> str:
    h = os.getenv("MINIMAX_API_HOST", "").strip()
    if h:
        return h.rstrip("/")
    return _GLOBAL_HOST


class MiniMaxWebSearchProvider(WebSearchProvider):
    """Search-only backend backed by the MiniMax Token Plan search API."""

    @property
    def name(self) -> str:
        # Single token, lowercase, no hyphen — matches the values
        # ``plugins/web/<vendor>/plugin.yaml: provides_web_providers`` lists
        # and the key users set in config.yaml ``web.search_backend``.
        return "minimax"

    @property
    def display_name(self) -> str:
        return "MiniMax (Token Plan)"

    def is_available(self) -> bool:
        # Cheap synchronous check — runs at registration and on every
        # ``hermes tools`` repaint. Per the ABC contract: do NOT make network
        # calls here.
        return bool(_resolve_api_key())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        # Token Plan search returns organic results only; no body extraction.
        # Hermes will route web_extract calls to another configured backend.
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Run a search against ``POST {host}/v1/coding_plan/search``.

        Returns the canonical envelope:
            success:  ``{"success": True,  "data": {"web": [...]}}``
            failure:  ``{"success": False, "error": str}``
        """
        import httpx  # lazy — matches the lazy-dep style of the built-in plugins

        api_key = _resolve_api_key()
        if not api_key:
            return {
                "success": False,
                "error": (
                    "MINIMAX_CODE_PLAN_KEY (or MINIMAX_CODING_API_KEY / "
                    "MINIMAX_OAUTH_TOKEN / MINIMAX_API_KEY) is not set"
                ),
            }

        if not query or not query.strip():
            return {"success": False, "error": "query is required"}

        host = _resolve_host()
        url = f"{host}{_SEARCH_PATH}"

        try:
            resp = httpx.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "MM-API-Source": "hermes-agent-minimax-search",
                },
                json={"q": query.strip()},
                timeout=20,
            )
        except httpx.RequestError as exc:
            logger.warning("MiniMax search transport error: %s", exc)
            return {"success": False, "error": f"Could not reach MiniMax: {exc}"}

        if resp.status_code >= 400:
            # Surface HTTP-level errors with the body so 401/403/region-mismatch
            # are debuggable from the agent's error message.
            body = (resp.text or "")[:300]
            logger.warning("MiniMax search HTTP %s: %s", resp.status_code, body)
            return {
                "success": False,
                "error": f"MiniMax returned HTTP {resp.status_code}: {body}",
            }

        try:
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("MiniMax search JSON parse error: %s", exc)
            return {
                "success": False,
                "error": "Could not parse MiniMax response as JSON",
            }

        # API-level error envelope: base_resp.status_code != 0
        base = payload.get("base_resp") or {}
        if base and base.get("status_code") not in (0, None):
            return {
                "success": False,
                "error": (
                    f"MiniMax API error {base.get('status_code')}: "
                    f"{base.get('status_msg', 'unknown')}"
                ),
            }

        # Success — normalize ``organic[]`` to canonical {title, url, ...}
        organic: List[Dict[str, Any]] = (
            payload.get("organic")
            or payload.get("results")
            or payload.get("web")
            or []
        )
        # Some MiniMax responses use a different shape; fall back gracefully
        # so a single-shape mismatch doesn't break the tool.
        try:
            web = [
                {
                    "title": str(item.get("title") or item.get("name") or ""),
                    "url": str(
                        item.get("link") or item.get("url") or item.get("href") or ""
                    ),
                    "description": str(
                        item.get("snippet")
                        or item.get("description")
                        or item.get("summary")
                        or ""
                    ),
                    "position": i + 1,
                }
                for i, item in enumerate(organic[: max(1, int(limit))])
            ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("MiniMax search normalize error: %s", exc)
            return {
                "success": False,
                "error": f"Could not normalize MiniMax results: {exc}",
            }

        logger.info(
            "MiniMax search '%s': %d results (limit %d)",
            query,
            len(web),
            limit,
        )
        return {"success": True, "data": {"web": web}}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "MiniMax (Token Plan)",
            "badge": "paid",
            "tag": (
                "MiniMax Coding Plan web search. Use a Token Plan key "
                "(sk-cp-...) — chat-only MiniMax API keys may not work."
            ),
            "env_vars": [
                {
                    "key": "MINIMAX_CODE_PLAN_KEY",
                    "prompt": "MiniMax Token Plan API key (sk-cp-...)",
                    "url": "https://platform.minimax.io/",
                },
                {
                    "key": "MINIMAX_API_HOST",
                    "prompt": "MiniMax API host",
                    # default handled in code; shown to user for clarity
                },
            ],
        }
