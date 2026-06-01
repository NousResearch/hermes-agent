"""Ollama Cloud web search + content extraction — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. Two
capabilities advertised:

- ``supports_search()``  -> True (Ollama ``/api/web_search``)
- ``supports_extract()`` -> True (Ollama ``/api/web_fetch``)
- ``supports_crawl()``   -> False

Config keys this provider responds to::

    web:
      search_backend: "ollama"      # explicit per-capability
      extract_backend: "ollama"     # explicit per-capability
      backend: "ollama"             # shared fallback for all three

Env vars::

    OLLAMA_API_KEY=...           # https://ollama.com/settings/keys (required)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_OLLAMA_API = "https://ollama.com"


class OllamaWebSearchProvider(WebSearchProvider):
    """Ollama Cloud search + extract provider."""

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def display_name(self) -> str:
        return "Ollama Cloud"

    def is_available(self) -> bool:
        """Return True when ``OLLAMA_API_KEY`` is set to a non-empty value."""
        return bool(os.getenv("OLLAMA_API_KEY", "").strip())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def supports_crawl(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute an Ollama web search."""
        import httpx

        api_key = os.getenv("OLLAMA_API_KEY", "").strip()
        if not api_key:
            return {"success": False, "error": "OLLAMA_API_KEY is not set"}

        count = max(1, min(int(limit), 10))

        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}

            logger.info("Ollama web search: '%s' (limit=%d)", query, count)
            resp = httpx.post(
                f"{_OLLAMA_API}/api/web_search",
                json={"query": query, "max_results": count},
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=60,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("Ollama web search HTTP error: %s", exc)
            return {
                "success": False,
                "error": f"Ollama web search returned HTTP {exc.response.status_code}",
            }
        except httpx.RequestError as exc:
            logger.warning("Ollama web search request error: %s", exc)
            return {"success": False, "error": f"Could not reach Ollama web search: {exc}"}

        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama web search response parse error: %s", exc)
            return {
                "success": False,
                "error": "Could not parse Ollama web search response as JSON",
            }

        raw_results = data.get("results", [])

        web_results = [
            {
                "title": str(r.get("title", "")),
                "url": str(r.get("url", "")),
                "description": str(r.get("content", "")),
                "position": i + 1,
            }
            for i, r in enumerate(raw_results[:limit])
        ]

        logger.info(
            "Ollama web search '%s': %d results (from %d raw, limit %d)",
            query,
            len(web_results),
            len(raw_results),
            limit,
        )

        return {"success": True, "data": {"web": web_results}}

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract content from one or more URLs via Ollama web_fetch.

        Sync — the underlying call is httpx.post(...). Returns the legacy
        list-of-results shape; per-URL failures become items with ``error``.
        """
        import httpx

        api_key = os.getenv("OLLAMA_API_KEY", "").strip()
        if not api_key:
            return [{"url": u, "title": "", "content": "", "error": "OLLAMA_API_KEY is not set"} for u in urls]

        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return [{"url": u, "title": "", "content": "", "error": "Interrupted"} for u in urls]

            logger.info("Ollama web fetch: %d URL(s)", len(urls))
        except Exception:  # noqa: BLE001
            pass

        documents: List[Dict[str, Any]] = []
        for url in urls:
            try:
                resp = httpx.post(
                    f"{_OLLAMA_API}/api/web_fetch",
                    json={"url": url},
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPStatusError as exc:
                logger.warning("Ollama web fetch HTTP error for %s: %s", url, exc)
                documents.append({
                    "url": url,
                    "title": "",
                    "content": "",
                    "error": f"Ollama web fetch returned HTTP {exc.response.status_code}",
                })
            except httpx.RequestError as exc:
                logger.warning("Ollama web fetch request error for %s: %s", url, exc)
                documents.append({
                    "url": url,
                    "title": "",
                    "content": "",
                    "error": f"Could not reach Ollama web fetch: {exc}",
                })
            except Exception as exc:  # noqa: BLE001
                logger.warning("Ollama web fetch error for %s: %s", url, exc)
                documents.append({
                    "url": url,
                    "title": "",
                    "content": "",
                    "error": f"Ollama web fetch failed: {exc}",
                })
            else:
                raw_content = data.get("content", "")
                documents.append({
                    "url": url,
                    "title": data.get("title", ""),
                    "content": raw_content,
                    "raw_content": raw_content,
                    "metadata": {"sourceURL": url},
                })

        return documents

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Ollama Cloud",
            "badge": "free tier · paid upgrade",
            "tag": "Search + extract via Ollama Cloud API. Generous free tier available.",
            "env_vars": [
                {
                    "key": "OLLAMA_API_KEY",
                    "prompt": "Ollama API key",
                    "url": "https://ollama.com/settings/keys",
                },
            ],
        }
