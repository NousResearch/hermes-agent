"""Ollama Cloud web search + extract — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. Uses the
Ollama Cloud Web API (``https://ollama.com/api``) with Bearer-token auth.
No SDK dependency — just ``httpx`` which is already a core dep.

Config keys this provider responds to::

    web:
      search_backend: "ollama"      # explicit per-capability
      extract_backend: "ollama"     # explicit per-capability
      backend: "ollama"             # shared fallback for both

Env var::

    OLLAMA_API_KEY=***    # https://ollama.com (included with Ollama subscription)

Search calls ``POST /api/web_search``; extract calls ``POST /api/web_fetch``.
Both are sync — the web_tools dispatcher wraps via ``asyncio.to_thread``
when the caller is async.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List

import httpx

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_OLLAMA_WEB_BASE_URL = "https://ollama.com/api"


class OllamaWebSearchProvider(WebSearchProvider):
    """Ollama Cloud search + extract provider.

    Both methods are sync. The web_extract_tool dispatcher wraps sync
    extracts via ``asyncio.to_thread`` when it needs to keep the event
    loop responsive.
    """

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

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a web search via Ollama Cloud.

        Returns ``{"success": True, "data": {"web": [{...}, ...]}}`` on
        success, ``{"success": False, "error": str}`` on failure.
        """
        try:
            api_key = os.getenv("OLLAMA_API_KEY", "")
            if not api_key:
                return {
                    "success": False,
                    "error": (
                        "OLLAMA_API_KEY environment variable not set. "
                        "Get your API key at https://ollama.com "
                        "(sign in and create API key)"
                    ),
                }

            url = f"{_OLLAMA_WEB_BASE_URL}/web_search"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            payload = {"query": query, "max_results": min(limit, 20)}

            logger.info("Ollama search: '%s' (limit=%d)", query, limit)
            response = httpx.post(url, json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            raw = response.json()

            return _normalize_search_results(raw)
        except httpx.HTTPStatusError as exc:
            logger.warning("Ollama search HTTP error: %s", exc)
            return {
                "success": False,
                "error": f"Ollama Cloud returned HTTP {exc.response.status_code}",
            }
        except httpx.RequestError as exc:
            logger.warning("Ollama search request error: %s", exc)
            return {
                "success": False,
                "error": f"Could not reach Ollama Cloud: {exc}",
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama search error: %s", exc)
            return {"success": False, "error": f"Ollama search failed: {exc}"}

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract content from one or more URLs via Ollama Cloud.

        Ollama's ``web_fetch`` endpoint handles one URL at a time; this
        method iterates and collects results, gracefully handling
        per-URL failures.

        Returns a list of result dicts shaped for the legacy LLM
        post-processing pipeline.
        """
        api_key = os.getenv("OLLAMA_API_KEY", "")
        if not api_key:
            return [
                {
                    "url": u,
                    "title": "",
                    "content": "",
                    "error": (
                        "OLLAMA_API_KEY environment variable not set. "
                        "Get your API key at https://ollama.com"
                    ),
                }
                for u in urls
            ]

        results: List[Dict[str, Any]] = []
        logger.info("Ollama fetch: %d URL(s)", len(urls))
        for url in urls:
            try:
                result = self._fetch_single(url, api_key)
                results.append(result)
            except Exception as exc:
                logger.debug("Ollama fetch failed for %s: %s", url, exc)
                results.append({
                    "url": url,
                    "title": "",
                    "content": "",
                    "error": str(exc),
                })
        return results

    def _fetch_single(self, url: str, api_key: str) -> Dict[str, Any]:
        """Fetch a single URL via Ollama Cloud's web_fetch endpoint."""
        endpoint = f"{_OLLAMA_WEB_BASE_URL}/web_fetch"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {"url": url}

        logger.info("Ollama fetch: %s", url)
        response = httpx.post(endpoint, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        raw = response.json()

        return _normalize_fetch_result(raw, fallback_url=url)

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Ollama Cloud",
            "tag": "Web search and fetch (included with Ollama subscription)",
            "web_backend": "ollama",
            "env_vars": [
                {
                    "key": "OLLAMA_API_KEY",
                    "prompt": "Ollama API key",
                    "url": "https://ollama.com",
                },
            ],
        }


# ─── Response normalizers ─────────────────────────────────────────────────


def _normalize_search_results(response: dict) -> dict:
    """Normalize Ollama web_search response to the standard format.

    The Ollama API may return results in a few shapes:
    - ``{"results": [{...}, ...]}`` — standard
    - ``{"content": "..."}`` — markdown/text content with optional links
    - ``{"content": "{...}"}`` — JSON-encoded content string

    Handles all three cases.
    """
    web_results = []

    if not isinstance(response, dict):
        return {"success": True, "data": {"web": web_results}}

    results = response.get("results", [])

    # Some Ollama responses wrap results inside a "content" field
    if not results and "content" in response:
        content = response.get("content", "")
        if content.startswith("{") or content.startswith("["):
            try:
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    results = parsed
                elif isinstance(parsed, dict):
                    results = parsed.get("results", parsed.get("web", []))
            except json.JSONDecodeError:
                pass
        else:
            # Plain-text / markdown — extract [title](url) links
            links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", content)
            for i, (title, url) in enumerate(links[:10]):
                web_results.append({
                    "title": title,
                    "url": url,
                    "description": "",
                    "position": i + 1,
                })

    for i, result in enumerate(results[:10]):
        if isinstance(result, dict):
            web_results.append({
                "title": result.get("title", result.get("name", "")),
                "url": result.get("url", result.get("link", "")),
                "description": result.get(
                    "description", result.get("snippet", result.get("content", ""))
                ),
                "position": i + 1,
            })
        elif isinstance(result, str):
            web_results.append({
                "title": "",
                "url": result,
                "description": "",
                "position": i + 1,
            })

    return {"success": True, "data": {"web": web_results}}


def _normalize_fetch_result(response: dict, fallback_url: str = "") -> dict:
    """Normalize Ollama web_fetch response to the standard document format."""
    content = ""
    title = ""

    if isinstance(response, dict):
        content = response.get("content", "")
        title = response.get("title", "")
        if not content:
            content = response.get("raw_content", response.get("text", ""))
        metadata = response.get("metadata", {"sourceURL": fallback_url})
        if not isinstance(metadata, dict):
            metadata = {"sourceURL": fallback_url}
        if "sourceURL" not in metadata:
            metadata["sourceURL"] = fallback_url
    else:
        content = str(response) if response else ""
        metadata = {"sourceURL": fallback_url}

    return {
        "url": fallback_url,
        "title": title,
        "content": content,
        "raw_content": content,
        "metadata": metadata,
    }