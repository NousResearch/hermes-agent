"""Jina web search + content extraction — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. Uses
direct HTTP calls via httpx — no Python SDK needed.

Config keys this provider responds to::

    web:
      search_backend: "jina"      # explicit per-capability
      extract_backend: "jina"     # explicit per-capability
      backend: "jina"             # shared fallback

Env vars::

    JINA_API_KEY=***    # https://jina.ai (required)

Endpoints:
  - Search: https://s.jina.ai/ (POST with query body)
  - Extract: https://r.jina.ai/ (GET with URL in path)
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List

import httpx

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_SEARCH_ENDPOINT = "https://s.jina.ai/"
_READ_ENDPOINT = "https://r.jina.ai/"


class JinaWebSearchProvider(WebSearchProvider):
    """Jina Search + Reader provider."""

    @property
    def name(self) -> str:
        return "jina"

    @property
    def display_name(self) -> str:
        return "Jina"

    def is_available(self) -> bool:
        """Return True when ``JINA_API_KEY`` is set to a non-empty value."""
        return bool(os.getenv("JINA_API_KEY", "").strip())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def supports_crawl(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a Jina search via https://s.jina.ai/.

        Returns ``{"success": True, "data": {"web": [...]}}`` on success,
        or ``{"success": False, "error": "..."}`` on failure.
        """
        api_key = os.environ.get("JINA_API_KEY")
        if not api_key:
            return {
                "success": False,
                "error": (
                    "JINA_API_KEY environment variable not set. "
                    "Get your API key at https://jina.ai"
                ),
            }

        headers = {"Authorization": f"Bearer {api_key}"}

        try:
            from urllib.parse import quote_plus
            with httpx.Client(timeout=30) as client:
                resp = client.get(
                    f"{_SEARCH_ENDPOINT}{quote_plus(query)}",
                    headers=headers,
                    params={"numResults": limit},
                )
                resp.raise_for_status()
                text = resp.text
                return _parse_search_response(text, limit)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 — including httpx errors
            logger.warning("Jina search error: %s", exc)
            return {"success": False, "error": f"Jina search failed: {exc}"}

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract content from one or more URLs via https://r.jina.ai/.

        Sync — the underlying call is httpx.get(...). Returns a bare list
        of result dicts; per-URL failures become items with ``error``.

        Connection pooling: one httpx.Client is reused across all URLs.
        """
        api_key = os.environ.get("JINA_API_KEY")
        if not api_key:
            return [
                {"url": u, "title": "", "content": "", "error": "JINA_API_KEY environment variable not set."}
                for u in urls
            ]

        headers = {"Authorization": f"Bearer {api_key}"}

        # One client outside the URL loop for connection pooling (Fix #3).
        results: List[Dict[str, Any]] = []
        try:
            with httpx.Client(timeout=60, follow_redirects=True) as client:
                for url in urls:
                    try:
                        resp = client.get(
                            f"{_READ_ENDPOINT}{url}",
                            headers=headers,
                        )
                        resp.raise_for_status()
                        content = resp.text
                        title = _extract_title(content) or url
                        results.append({
                            "url": url,
                            "title": title,
                            "content": content,
                            "raw_content": content,
                            "metadata": {},
                        })
                    except Exception as exc:  # noqa: BLE001
                        logger.error("Jina extract failed for %s: %s", url, exc)
                        results.append({
                            "url": url,
                            "title": "",
                            "content": "",
                            "raw_content": "",
                            "metadata": {},
                            "error": str(exc),
                        })
        except Exception as exc:  # noqa: BLE001 — client-level failure
            logger.error("Jina extract client error: %s", exc)
            for url in urls:
                results.append({
                    "url": url,
                    "title": "",
                    "content": "",
                    "raw_content": "",
                    "metadata": {},
                    "error": f"Jina extract failed: {exc}",
                })

        return results

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Jina",
            "badge": "paid · search + extract",
            "tag": "Web search with content snippets + URL extraction. Free tier: 100 req/day.",
            "env_vars": [
                {
                    "key": "JINA_API_KEY",
                    "prompt": "Jina API key",
                    "url": "https://jina.ai/",
                },
            ],
        }


def _parse_search_response(text: str, limit: int) -> Dict[str, Any]:
    """Parse Jina search text response into standard shape.

    Jina s.jina.ai returns text (not JSON). Each result is separated by
    ``---`` boundaries and contains a URL, title, and snippet.

    Returns ``{"success": True, "data": {"web": [...]}}`` on success.
    On parse failure, returns ``{"success": False, "error": "..."}``
    so the caller correctly falls back to the next provider (Fix #1).
    """
    try:
        web_results: List[Dict[str, Any]] = []

        # Split by --- separators (result boundaries).
        sections = text.split("---")

        for section in sections:
            section = section.strip()
            if not section:
                continue

            # Extract URL.
            url_match = re.search(r'https?://[^\s<>"\']+|www\.[^\s<>"\']+', section)
            if not url_match:
                continue
            url = url_match.group(0)

            # Extract title — look for "Title:" or "title:" prefix.
            title_match = re.search(r'^(?:Title|title)[\s:]+(.+)$', section, re.MULTILINE)
            if title_match:
                title = title_match.group(1).strip()
            else:
                # Fallback: use URL hostname as title.
                title = url

            # Extract description/snippet — strip any Title: or Snippet:
            # markers that may prefix the text.
            description = section
            description = re.sub(r'^(?:Title|title)[\s:]+.*$', '', description, flags=re.MULTILINE).strip()
            description = re.sub(r'^(?:Snippet|snippet)[\s:]+.*$', '', description, flags=re.MULTILINE).strip()
            # Remove the URL line from description.
            description = re.sub(r'https?://[^\s<>"\']+|www\.[^\s<>"\']+', '', description).strip()
            # Clean up whitespace.
            description = re.sub(r'\n\s*\n', '\n', description).strip()

            if not description:
                description = title  # fallback to title if no snippet

            web_results.append({
                "title": title,
                "url": url,
                "description": description,
                "position": len(web_results) + 1,
            })

        if not web_results:
            return {
                "success": False,
                "error": "Jina search returned no parseable results",
            }

        return {"success": True, "data": {"web": web_results}}

    except Exception as exc:  # noqa: BLE001
        # Exception during parsing — return failure so caller falls back
        # to the next provider (Fix #1 + Fix #5).
        logger.warning("Jina search parse error: %s", exc)
        return {"success": False, "error": f"Jina search parse error: {exc}"}


def _extract_title(content: str) -> str:
    """Extract title from Jina reader response.

    Tries multiple strategies:
    1. <title> tag (HTML-style)
    2. Markdown heading (# Title) — strips # markers (Fix #4)
    3. First non-empty line

    Returns empty string if no title found.
    """
    import re

    # Strategy 1: <title> tag.
    m = re.search(r'<title>(.*?)</title>', content, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()

    # Strategy 2: Markdown heading — strip # markers (Fix #4).
    # Previously this explicitly skipped lines starting with #, so
    # Jina's markdown output titles were missed.
    first_line = content.split('\n')[0].strip()
    if first_line.startswith('#'):
        return re.sub(r'^#{1,6}\s+', '', first_line).strip()

    # Strategy 3: First non-empty line as title.
    for line in content.split('\n'):
        line = line.strip()
        if line and len(line) > 3:
            return line

    return ""
