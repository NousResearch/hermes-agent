"""Jina Reader web-content extraction provider.

The reader is intentionally extract-only. It sends public, SSRF-screened URLs to
Jina's raw Markdown reader; `tools.web_tools.web_extract_tool` performs the
SSRF/credential-like URL screening before dispatching here.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from agent.web_search_provider import WebSearchProvider, get_provider_env

logger = logging.getLogger(__name__)

_READER_PREFIX = "https://r.jina.ai/http://"
_TITLE_RE = re.compile(r"^Title:\s*(.+?)\s*$", re.MULTILINE)


def _reader_url(url: str) -> str:
    """Keep the source URL's scheme explicit for Jina Reader."""
    return f"{_READER_PREFIX}{url}"


def _title_from_markdown(text: str) -> str:
    match = _TITLE_RE.search(text or "")
    return match.group(1).strip() if match else ""


class JinaReaderWebSearchProvider(WebSearchProvider):
    """Public Jina Reader provider for raw Markdown extraction."""

    @property
    def name(self) -> str:
        return "jina-reader"

    @property
    def display_name(self) -> str:
        return "Jina Reader"

    def is_available(self) -> bool:
        # Basic Reader usage is unauthenticated. A JINA_API_KEY, if present,
        # only raises limits; never make a network request during discovery.
        return True

    def supports_search(self) -> bool:
        return False

    def supports_extract(self) -> bool:
        return True

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        import httpx

        headers = {
            "Accept": "text/markdown",
            "X-Return-Format": "markdown",
            "User-Agent": "Hermes-Agent/1.0 (+https://hermes-agent.nousresearch.com)",
        }
        api_key = get_provider_env("JINA_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        results: List[Dict[str, Any]] = []
        for url in urls:
            try:
                response = httpx.get(
                    _reader_url(url),
                    headers=headers,
                    timeout=60,
                    follow_redirects=True,
                )
                response.raise_for_status()
                raw = response.text
                title = _title_from_markdown(raw)
                metadata: Dict[str, Any] = {
                    "sourceURL": url,
                    "title": title,
                    "status_code": response.status_code,
                    "source_provider": self.name,
                }
                cache_hint = response.headers.get("x-cache-warning") or response.headers.get("x-cache")
                if cache_hint:
                    metadata["cache_warning"] = str(cache_hint)[:300]
                results.append(
                    {
                        "url": url,
                        "title": title,
                        "content": raw,
                        "raw_content": raw,
                        "metadata": metadata,
                        "source_provider": self.name,
                    }
                )
            except httpx.HTTPStatusError as exc:
                status = getattr(exc.response, "status_code", None)
                logger.warning("Jina Reader HTTP error for %s: %s", url, exc)
                results.append(
                    {
                        "url": url,
                        "title": "",
                        "content": "",
                        "raw_content": "",
                        "error": f"Jina Reader returned HTTP {status}",
                        "metadata": {"sourceURL": url, "status_code": status, "source_provider": self.name},
                    }
                )
            except httpx.RequestError as exc:
                logger.warning("Jina Reader request error for %s: %s", url, exc)
                results.append(
                    {
                        "url": url,
                        "title": "",
                        "content": "",
                        "raw_content": "",
                        "error": f"Could not reach Jina Reader: {exc}",
                        "metadata": {"sourceURL": url, "source_provider": self.name},
                    }
                )
            except Exception as exc:  # pragma: no cover - defensive boundary
                logger.warning("Jina Reader unexpected error for %s: %s", url, exc)
                results.append(
                    {
                        "url": url,
                        "title": "",
                        "content": "",
                        "raw_content": "",
                        "error": f"Jina Reader extraction failed: {exc}",
                        "metadata": {"sourceURL": url, "source_provider": self.name},
                    }
                )
        return results

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Jina Reader",
            "badge": "free",
            "tag": "Raw Markdown extraction; public URLs only, no key required for basic usage.",
            "env_vars": [
                {
                    "key": "JINA_API_KEY",
                    "prompt": "Optional Jina Reader API key",
                    "url": "https://jina.ai/reader/",
                }
            ],
        }
