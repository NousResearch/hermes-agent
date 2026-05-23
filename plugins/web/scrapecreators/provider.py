"""ScrapeCreators Google Search — plugin form.

Routes ``web_search`` tool calls through ScrapeCreators' Google Search API
and normalizes results into Hermes' standard ``{title, url, description,
position}`` rows.

Reference: https://scrapecreators.com/

Config keys this provider responds to::

    web:
      search_backend: "scrapecreators"   # explicit per-capability
      backend: "scrapecreators"          # shared fallback

Optional knobs (under ``web.scrapecreators`` in ``config.yaml``)::

    web:
      scrapecreators:
        base_url: "https://api.scrapecreators.com/v1"
        timeout: 30

Auth env var::

    SCRAPECREATORS_API_KEY=...
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import httpx

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.scrapecreators.com/v1"
DEFAULT_TIMEOUT = 30


def _load_scrapecreators_config() -> Dict[str, Any]:
    """Read ``web.scrapecreators`` from config.yaml (returns {} on miss)."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        web_section = cfg.get("web") if isinstance(cfg, dict) else None
        section = web_section.get("scrapecreators") if isinstance(web_section, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not load web.scrapecreators config: %s", exc)
        return {}


def _coerce_timeout(value: Any) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return float(DEFAULT_TIMEOUT)
    return max(1.0, timeout)


def _normalize_results(payload: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
    """Normalize known ScrapeCreators Google Search payload shapes."""
    candidates = (
        payload.get("results"),
        payload.get("organic_results"),
        (payload.get("data") or {}).get("results") if isinstance(payload.get("data"), dict) else None,
        (payload.get("data") or {}).get("organic_results") if isinstance(payload.get("data"), dict) else None,
    )
    raw_results = next((items for items in candidates if isinstance(items, list)), [])

    web_results: List[Dict[str, Any]] = []
    for i, hit in enumerate(raw_results[:limit]):
        if not isinstance(hit, dict):
            continue
        url = str(hit.get("url") or hit.get("link") or hit.get("href") or "")
        web_results.append(
            {
                "title": str(hit.get("title") or ""),
                "url": url,
                "description": str(hit.get("description") or hit.get("snippet") or hit.get("body") or ""),
                "position": int(hit.get("position") or hit.get("rank") or i + 1),
            }
        )
    return web_results


class ScrapeCreatorsWebSearchProvider(WebSearchProvider):
    """Search-only provider backed by ScrapeCreators' Google Search API."""

    @property
    def name(self) -> str:
        return "scrapecreators"

    @property
    def display_name(self) -> str:
        return "ScrapeCreators Google Search"

    def is_available(self) -> bool:
        """Cheap availability probe — requires ``SCRAPECREATORS_API_KEY``."""
        return bool(os.getenv("SCRAPECREATORS_API_KEY", "").strip())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        api_key = os.getenv("SCRAPECREATORS_API_KEY", "").strip()
        if not api_key:
            return {"success": False, "error": "SCRAPECREATORS_API_KEY is not set"}

        safe_limit = max(1, int(limit or 5))
        cfg = _load_scrapecreators_config()
        base_url = str(cfg.get("base_url") or os.getenv("SCRAPECREATORS_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        timeout = _coerce_timeout(cfg.get("timeout") or os.getenv("SCRAPECREATORS_TIMEOUT"))

        try:
            resp = httpx.get(
                f"{base_url}/google/search",
                params={"query": query},
                headers={
                    "x-api-key": api_key,
                    "accept": "application/json",
                    "user-agent": "Hermes-Agent/1.0",
                },
                timeout=timeout,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("ScrapeCreators HTTP error: %s", exc)
            return {
                "success": False,
                "error": f"ScrapeCreators returned HTTP {exc.response.status_code}",
            }
        except httpx.RequestError as exc:
            logger.warning("ScrapeCreators request error: %s", exc)
            return {"success": False, "error": f"Could not reach ScrapeCreators: {exc}"}

        try:
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("ScrapeCreators response parse error: %s", exc)
            return {"success": False, "error": "Could not parse ScrapeCreators response as JSON"}

        web_results = _normalize_results(payload, safe_limit)
        logger.info("ScrapeCreators search '%s': %d results (limit %d)", query, len(web_results), safe_limit)
        return {"success": True, "data": {"web": web_results}}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "ScrapeCreators Google Search",
            "badge": "search",
            "tag": "Google web search via ScrapeCreators /v1/google/search (search only)",
            "env_vars": [
                {
                    "key": "SCRAPECREATORS_API_KEY",
                    "prompt": "ScrapeCreators API key",
                    "url": "https://scrapecreators.com/",
                },
            ],
            "web_backend": "scrapecreators",
        }
