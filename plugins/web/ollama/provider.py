"""Ollama Cloud web search + fetch plugin.

Uses Ollama's public web_search and web_fetch REST endpoints:
  - POST https://ollama.com/api/web_search
  - POST https://ollama.com/api/web_fetch

Backed by :class:`agent.web_search_provider.WebSearchProvider` and auto-loaded
as a bundled ``kind: backend`` plugin under ``plugins/web/ollama``.

Config keys this provider responds to::

    web:
      search_backend: "ollama"
      extract_backend: "ollama"
      backend: "ollama"

Env vars::

    OLLAMA_API_KEY=...                    # required
    OLLAMA_WEB_SEARCH_URL=...             # optional override
    OLLAMA_WEB_FETCH_URL=...              # optional override
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_DEFAULT_SEARCH_URL = "https://ollama.com/api/web_search"
_DEFAULT_FETCH_URL = "https://ollama.com/api/web_fetch"


def _get_api_key() -> Optional[str]:
    """Return the configured Ollama API key, or None if unset."""
    return os.getenv("OLLAMA_API_KEY", "").strip() or None


def _search_url() -> str:
    return os.getenv("OLLAMA_WEB_SEARCH_URL", _DEFAULT_SEARCH_URL).strip().rstrip("/")


def _fetch_url() -> str:
    return os.getenv("OLLAMA_WEB_FETCH_URL", _DEFAULT_FETCH_URL).strip().rstrip("/")


def _http_post_json(url: str, payload: Dict[str, Any], api_key: str, timeout: float = 30.0) -> Any:
    """POST JSON to an Ollama web endpoint and return the parsed response."""
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(body)
        except Exception:
            detail = body
        raise RuntimeError(f"Ollama web API HTTP {exc.code}: {detail}") from exc


class OllamaWebSearchProvider(WebSearchProvider):
    """Ollama Cloud web_search + web_fetch provider."""

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def display_name(self) -> str:
        return "Ollama Cloud"

    def is_available(self) -> bool:
        """Return True when ``OLLAMA_API_KEY`` is set to a non-empty value."""
        return _get_api_key() is not None

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a web search via Ollama Cloud."""
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}

            api_key = _get_api_key()
            if not api_key:
                return {
                    "success": False,
                    "error": "OLLAMA_API_KEY environment variable not set. "
                    "Create an API key at https://ollama.com/settings/keys",
                }

            safe_limit = max(1, int(limit))
            logger.info("Ollama Cloud web search: '%s' (limit=%d)", query, safe_limit)

            raw = _http_post_json(
                _search_url(),
                {"query": query, "max_results": safe_limit},
                api_key,
                timeout=30.0,
            )

            results = []
            for i, hit in enumerate(raw.get("results", [])):
                results.append(
                    {
                        "title": str(hit.get("title", "")),
                        "url": str(hit.get("url", "")),
                        "description": str(hit.get("content", "")),
                        "position": i + 1,
                    }
                )

            logger.info("Ollama Cloud web search: %d results", len(results))
            return {"success": True, "data": {"web": results}}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama Cloud web search error: %s", exc)
            return {"success": False, "error": f"Ollama web search failed: {exc}"}

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Fetch content from one or more URLs via Ollama Cloud web_fetch."""
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return [{"url": u, "error": "Interrupted", "title": ""} for u in urls]

            api_key = _get_api_key()
            if not api_key:
                return [
                    {
                        "url": u,
                        "title": "",
                        "content": "",
                        "error": "OLLAMA_API_KEY environment variable not set. "
                        "Create an API key at https://ollama.com/settings/keys",
                    }
                    for u in urls
                ]

            logger.info("Ollama Cloud web fetch: %d URL(s)", len(urls))
            results: List[Dict[str, Any]] = []
            fetch_url = _fetch_url()

            for url in urls:
                if is_interrupted():
                    results.append({"url": url, "error": "Interrupted", "title": ""})
                    continue

                try:
                    raw = _http_post_json(
                        fetch_url,
                        {"url": url},
                        api_key,
                        timeout=60.0,
                    )
                    content = str(raw.get("content", "") or "")
                    results.append(
                        {
                            "url": url,
                            "title": str(raw.get("title", "")),
                            "content": content,
                            "raw_content": content,
                            "metadata": {"links": raw.get("links", [])},
                        }
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Ollama Cloud web fetch error for %s: %s", url, exc)
                    results.append(
                        {
                            "url": url,
                            "title": "",
                            "content": "",
                            "raw_content": "",
                            "error": f"Ollama web fetch failed: {exc}",
                        }
                    )

            return results
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama Cloud web fetch error: %s", exc)
            return [
                {"url": u, "title": "", "content": "", "error": f"Ollama web fetch failed: {exc}"}
                for u in urls
            ]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Ollama Cloud",
            "badge": "free tier",
            "tag": "Ollama's official web search + fetch APIs. Free tier included with an Ollama account.",
            "env_vars": [
                {
                    "key": "OLLAMA_API_KEY",
                    "prompt": "Ollama API key",
                    "url": "https://ollama.com/settings/keys",
                },
            ],
        }
