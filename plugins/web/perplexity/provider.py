"""Perplexity web search — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. Backed by
Perplexity's dedicated Search API (``POST /search``), which returns ranked
web results with title/url/snippet — a clean fit for the ``web_search``
tool contract. Search only; Perplexity exposes no content-extraction
endpoint, so :meth:`supports_extract` stays False.

Config keys this provider responds to::

    web:
      search_backend: "perplexity"   # explicit search backend
      backend: "perplexity"          # shared fallback (search only)

Env vars::

    PERPLEXITY_API_KEY=...           # https://www.perplexity.ai/account/api (required)
    PERPLEXITY_BASE_URL=...          # optional override of https://api.perplexity.ai
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


def _perplexity_search_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST to the Perplexity Search API and return the parsed JSON response.

    Raises ``ValueError`` when ``PERPLEXITY_API_KEY`` is unset; the caller
    catches and surfaces it as a typed error response.
    """
    import httpx

    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        raise ValueError(
            "PERPLEXITY_API_KEY environment variable not set. "
            "Get your API key at https://www.perplexity.ai/account/api"
        )

    base_url = os.getenv("PERPLEXITY_BASE_URL", "https://api.perplexity.ai")
    url = f"{base_url.rstrip('/')}/search"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    logger.info("Perplexity search request to %s", url)

    response = httpx.post(url, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    return response.json()


def _normalize_perplexity_search_results(response: Dict[str, Any]) -> Dict[str, Any]:
    """Map Perplexity ``/search`` response to ``{success, data: {web: [...]}}``."""
    web_results: List[Dict[str, Any]] = []
    for i, result in enumerate(response.get("results", [])):
        web_results.append(
            {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "description": result.get("snippet", ""),
                "position": i + 1,
            }
        )
    return {"success": True, "data": {"web": web_results}}


class PerplexityWebSearchProvider(WebSearchProvider):
    """Perplexity Search API provider (search only)."""

    @property
    def name(self) -> str:
        return "perplexity"

    @property
    def display_name(self) -> str:
        return "Perplexity"

    def is_available(self) -> bool:
        """Return True when ``PERPLEXITY_API_KEY`` is set to a non-empty value."""
        return bool(os.getenv("PERPLEXITY_API_KEY", "").strip())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    @staticmethod
    def _search_filters() -> tuple[str, List[str]]:
        """Read optional ``recency``/``domains`` from ``web.perplexity`` config."""
        try:
            from hermes_cli.config import load_config

            pcfg = (load_config().get("web", {}) or {}).get("perplexity", {}) or {}
        except Exception:  # noqa: BLE001 — config optional
            return "", []
        recency = str(pcfg.get("recency", "") or "").strip().lower()
        if recency not in {"hour", "day", "week", "month", "year"}:
            recency = ""
        domains_raw = pcfg.get("domains") or []
        domains = [d for d in domains_raw if isinstance(d, str) and d.strip()][:10] \
            if isinstance(domains_raw, list) else []
        return recency, domains

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a Perplexity search.

        Optional filters are read from config (``web.perplexity.recency`` and
        ``web.perplexity.domains``) since the ``web_search`` tool contract is
        fixed to ``(query, limit)``. ``recency`` is one of
        hour|day|week|month|year; ``domains`` is an allow/deny list (prefix a
        domain with ``-`` to exclude it).
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}

            payload: Dict[str, Any] = {
                "query": query,
                "max_results": min(limit, 20),
            }
            recency, domains = self._search_filters()
            if recency:
                payload["search_recency_filter"] = recency
            if domains:
                payload["search_domain_filter"] = domains

            logger.info(
                "Perplexity search: '%s' (limit=%d, recency=%s, domains=%s)",
                query, limit, recency or "-", domains or "-",
            )
            raw = _perplexity_search_request(payload)
            return _normalize_perplexity_search_results(raw)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 — including httpx errors
            logger.warning("Perplexity search error: %s", exc)
            return {"success": False, "error": f"Perplexity search failed: {exc}"}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Perplexity",
            "badge": "paid",
            "tag": "AI-native web search with fresh, ranked results.",
            "env_vars": [
                {
                    "key": "PERPLEXITY_API_KEY",
                    "prompt": "Perplexity API key",
                    "url": "https://www.perplexity.ai/account/api",
                },
            ],
        }
