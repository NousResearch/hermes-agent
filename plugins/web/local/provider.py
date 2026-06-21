"""Local web content extraction provider.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. This provider
implements extraction only: it fetches already-validated public HTTP(S) pages,
strips non-content HTML, and returns readable text. It intentionally avoids
heavy browser/PDF/JavaScript handling; use Firecrawl, Tavily, Exa, or Parallel
for richer extraction.

Config key this provider responds to::

    web:
      extract_backend: "local"

No environment variables are required.
"""

from __future__ import annotations

import html
import logging
import re
from typing import Any, Dict, List

import httpx

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


class LocalWebExtractProvider(WebSearchProvider):
    """No-key extractor for simple public HTML/text pages."""

    @property
    def name(self) -> str:
        return "local"

    @property
    def display_name(self) -> str:
        return "Local HTML extractor"

    def is_available(self) -> bool:
        """Local extraction only needs stdlib + httpx, both already available."""
        return True

    def supports_search(self) -> bool:
        return False

    def supports_extract(self) -> bool:
        return True

    async def extract(
        self, urls: List[str], max_chars: int = 2_000_000, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Extract readable text from public HTML/text URLs.

        The tool wrapper performs secret-in-URL blocking and SSRF/private-network
        validation before dispatching here. This provider is deliberately
        conservative and reports per-URL errors instead of raising.
        """
        results: List[Dict[str, Any]] = []
        timeout = httpx.Timeout(30.0, connect=10.0)
        headers = {
            "User-Agent": "HermesAgent/0.17 web_extract (+https://hermes-agent.nousresearch.com)",
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
        }
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=True, headers=headers
        ) as client:
            for url in urls:
                try:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    ctype = resp.headers.get("content-type", "")
                    if "pdf" in ctype.lower():
                        results.append({
                            "url": url,
                            "title": "",
                            "content": "",
                            "error": (
                                "Local extractor does not handle PDFs; configure "
                                "Firecrawl, Tavily, Exa, or Parallel for PDF extraction."
                            ),
                        })
                        continue
                    raw = resp.text[:max_chars]
                    title = _extract_title(raw)
                    content = _html_to_text(raw)
                    results.append({
                        "url": str(resp.url),
                        "title": title,
                        "content": content,
                        "raw_content": content,
                        "metadata": {"content_type": ctype},
                        "error": "",
                    })
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Local extraction failed for %s: %s", url, exc)
                    results.append({
                        "url": url,
                        "title": "",
                        "content": "",
                        "error": f"Local extraction failed: {exc}",
                    })
        return results

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Local HTML extractor",
            "badge": "free · local · no key",
            "tag": "Fetches simple public HTML/text pages directly; no external extraction service required.",
            "env_vars": [],
        }


def _extract_title(raw: str) -> str:
    title_match = re.search(r"<title[^>]*>(.*?)</title>", raw, flags=re.I | re.S)
    if not title_match:
        return ""
    return html.unescape(re.sub(r"\s+", " ", title_match.group(1)).strip())


def _html_to_text(raw: str) -> str:
    text = re.sub(
        r"(?is)<(script|style|noscript|svg|canvas|template)[^>]*>.*?</\1>",
        " ",
        raw,
    )
    text = re.sub(r"(?is)<!--.*?-->", " ", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(p|div|section|article|header|footer|li|h[1-6]|tr)>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text).replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    content = "\n".join(line for line in lines if line)
    return re.sub(r"\n{3,}", "\n\n", content).strip()
