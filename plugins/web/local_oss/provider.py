"""Local OSS web providers backed by SearXNG and Crawl4AI.

``local_oss`` is the composite provider: search goes to SearXNG and
extract goes to Crawl4AI. ``crawl4ai`` is registered separately so users
can set ``web.extract_backend: crawl4ai`` alongside any search backend.

Env vars:

    SEARXNG_API_URL=http://localhost:8080  # or legacy SEARXNG_URL
    CRAWL4AI_API_URL=http://localhost:11235
    CRAWL4AI_API_TOKEN=...                 # optional bearer token
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

import httpx

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)


def _has_env(name: str) -> bool:
    value = os.getenv(name)
    return bool(value and value.strip())


def _get_searxng_api_url() -> str:
    """Return the configured SearXNG base URL."""
    url = (
        os.getenv("SEARXNG_API_URL", "").strip()
        or os.getenv("SEARXNG_URL", "").strip()
    ).rstrip("/")
    if not url:
        raise ValueError("SEARXNG_API_URL or SEARXNG_URL environment variable not set.")
    return url


def _get_crawl4ai_api_url() -> str:
    """Return the configured Crawl4AI base URL."""
    url = os.getenv("CRAWL4AI_API_URL", "").strip().rstrip("/")
    if not url:
        raise ValueError("CRAWL4AI_API_URL environment variable not set.")
    return url


def _get_crawl4ai_headers() -> Dict[str, str]:
    """Return headers for Crawl4AI API requests."""
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    token = os.getenv("CRAWL4AI_API_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _normalize_searxng_search_results(
    response: Dict[str, Any],
    limit: int,
) -> Dict[str, Any]:
    """Normalize SearXNG ``/search`` JSON responses to Hermes' schema."""
    raw_results = (response or {}).get("results", [])
    if not isinstance(raw_results, list):
        raw_results = []

    web_results = []
    for idx, result in enumerate(raw_results[:limit]):
        if not isinstance(result, dict):
            continue
        web_results.append(
            {
                "title": result.get("title") or result.get("url", ""),
                "url": result.get("url", ""),
                "description": (
                    result.get("content")
                    or result.get("snippet")
                    or result.get("description")
                    or ""
                ),
                "position": idx + 1,
            }
        )
    return {"success": True, "data": {"web": web_results}}


def _to_plain_object(value: Any) -> Any:
    """Convert SDK/Pydantic-ish objects to plain Python structures."""
    if value is None:
        return None
    if isinstance(value, (dict, list, str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return {k: v for k, v in value.__dict__.items() if not k.startswith("_")}
        except Exception:
            pass
    return value


def _normalize_crawl4ai_document(
    item: Any,
    fallback_url: str = "",
) -> Dict[str, Any]:
    """Normalize one Crawl4AI result entry into Hermes' document schema."""
    plain = _to_plain_object(item)
    if not isinstance(plain, dict):
        return {}

    metadata = _to_plain_object(plain.get("metadata"))
    if not isinstance(metadata, dict):
        metadata = {}

    markdown_payload = _to_plain_object(plain.get("markdown"))
    markdown_content = (
        (
            markdown_payload.get("fit_markdown")
            or markdown_payload.get("raw_markdown")
            or markdown_payload.get("markdown_with_citations")
            or markdown_payload.get("references_markdown")
            or ""
        )
        if isinstance(markdown_payload, dict)
        else (markdown_payload if isinstance(markdown_payload, str) else "")
    )
    content = (
        markdown_content
        or plain.get("fit_markdown")
        or plain.get("raw_markdown")
        or plain.get("extracted_content")
        or plain.get("content")
        or plain.get("cleaned_html")
        or plain.get("html")
        or ""
    )
    url = (
        plain.get("url")
        or plain.get("redirected_url")
        or metadata.get("sourceURL")
        or metadata.get("url")
        or fallback_url
    )
    title = plain.get("title") or metadata.get("title") or ""
    error = None

    if plain.get("success") is False:
        error = (
            plain.get("error")
            or plain.get("error_message")
            or plain.get("message")
            or "crawl failed"
        )
        content = ""
    elif plain.get("error") and not content:
        error = str(plain.get("error"))

    result = {
        "url": url,
        "title": title,
        "content": content,
        "raw_content": content,
        "metadata": metadata,
    }
    if error:
        result["error"] = error
    return result


def _normalize_crawl4ai_documents(
    response: Any,
    fallback_url: str = "",
) -> List[Dict[str, Any]]:
    """Normalize Crawl4AI API response payloads to Hermes' document schema."""
    plain = _to_plain_object(response)
    if not isinstance(plain, dict):
        return []

    raw_results = plain.get("results")
    if not isinstance(raw_results, list) and isinstance(plain.get("result"), list):
        raw_results = plain.get("result")
    if not isinstance(raw_results, list):
        raw_results = []

    documents: List[Dict[str, Any]] = []
    for item in raw_results:
        normalized = _normalize_crawl4ai_document(item, fallback_url=fallback_url)
        if normalized:
            documents.append(normalized)

    if not documents and plain.get("markdown"):
        normalized = _normalize_crawl4ai_document(plain, fallback_url=fallback_url)
        if normalized:
            documents.append(normalized)

    return documents


async def _crawl4ai_post(
    path: str,
    payload: Dict[str, Any],
    timeout: int = 120,
) -> Dict[str, Any]:
    """POST JSON to Crawl4AI and return the parsed payload."""
    url = f"{_get_crawl4ai_api_url()}{path}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=payload, headers=_get_crawl4ai_headers())
        response.raise_for_status()
        return response.json()


async def _crawl4ai_fetch_rendered_document(
    url: str,
    format: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch one rendered Crawl4AI document in the requested format."""
    if format == "html":
        raw = await _crawl4ai_post("/html", {"url": url}, timeout=60)
        return _normalize_crawl4ai_document(raw, fallback_url=url)

    raw = await _crawl4ai_post("/md", {"url": url, "f": "fit"}, timeout=60)
    document = _normalize_crawl4ai_document(raw, fallback_url=url)
    if document.get("content"):
        return document

    raw = await _crawl4ai_post("/md", {"url": url, "f": "raw"}, timeout=60)
    return _normalize_crawl4ai_document(raw, fallback_url=url)


def _crawl4ai_extract_timeout(num_urls: int) -> int:
    return max(60, min(180, 30 * max(num_urls, 1)))


async def _crawl4ai_extract(
    urls: List[str],
    *,
    format: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Extract page content with Crawl4AI and normalize to Hermes' shape."""
    raw = await _crawl4ai_post(
        "/crawl",
        {
            "urls": urls,
            "browser_config": {"headless": True},
            "crawler_config": {"cache_mode": "bypass"},
        },
        timeout=_crawl4ai_extract_timeout(len(urls)),
    )
    results = _normalize_crawl4ai_documents(
        raw,
        fallback_url=urls[0] if urls else "",
    )
    rendered_results = await asyncio.gather(
        *[_crawl4ai_fetch_rendered_document(url, format=format) for url in urls]
    )
    if not results:
        results = [item for item in rendered_results if item]
    rendered_by_url = {
        item.get("url", ""): item
        for item in rendered_results
        if item
    }
    for result in results:
        rendered = rendered_by_url.get(result.get("url", ""))
        if rendered and rendered.get("content"):
            result["content"] = rendered["content"]
            result["raw_content"] = rendered["raw_content"]
        elif rendered and rendered.get("error") and not result.get("error"):
            result["error"] = rendered["error"]
    return results


async def crawl4ai_crawl(
    url: str,
    *,
    depth: str = "basic",
) -> List[Dict[str, Any]]:
    """Crawl a site with Crawl4AI and return normalized documents."""
    max_depth = 2 if depth == "advanced" else 1
    raw = await _crawl4ai_post(
        "/crawl",
        {
            "urls": [url],
            "browser_config": {"headless": True},
            "crawler_config": {
                "cache_mode": "bypass",
                "deep_crawl_strategy": {
                    "type": "BFSDeepCrawlStrategy",
                    "params": {
                        "max_depth": max_depth,
                        "max_pages": 20,
                    },
                },
            },
        },
        timeout=180,
    )
    results = _normalize_crawl4ai_documents(raw, fallback_url=url)
    rendered_results = await asyncio.gather(
        *[
            _crawl4ai_fetch_rendered_document(result.get("url", ""))
            for result in results
            if result.get("url")
        ]
    )
    if not results:
        results = [item for item in rendered_results if item]
    rendered_by_url = {
        item.get("url", ""): item
        for item in rendered_results
        if item
    }
    for result in results:
        rendered = rendered_by_url.get(result.get("url", ""))
        if rendered and rendered.get("content"):
            result["content"] = rendered["content"]
            result["raw_content"] = rendered["raw_content"]
        elif rendered and rendered.get("error") and not result.get("error"):
            result["error"] = rendered["error"]
    return results


class LocalOSSWebSearchProvider(WebSearchProvider):
    """Composite SearXNG search + Crawl4AI extract provider."""

    @property
    def name(self) -> str:
        return "local_oss"

    @property
    def display_name(self) -> str:
        return "Local OSS (SearXNG + Crawl4AI)"

    def is_available(self) -> bool:
        return (
            (_has_env("SEARXNG_API_URL") or _has_env("SEARXNG_URL"))
            and _has_env("CRAWL4AI_API_URL")
        )

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}

            logger.info("SearXNG search via local_oss: '%s' (limit=%d)", query, limit)
            response = httpx.get(
                f"{_get_searxng_api_url()}/search",
                params={"q": query, "format": "json"},
                timeout=30,
            )
            response.raise_for_status()
            return _normalize_searxng_search_results(response.json(), limit)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        except Exception as exc:
            logger.warning("local_oss search error: %s", exc)
            return {"success": False, "error": f"SearXNG search failed: {exc}"}

    async def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return [{"url": u, "error": "Interrupted", "title": ""} for u in urls]

            logger.info("Crawl4AI extract via local_oss: %d URL(s)", len(urls))
            return await _crawl4ai_extract(urls, format=kwargs.get("format"))
        except ValueError as exc:
            return [{"url": u, "title": "", "content": "", "error": str(exc)} for u in urls]
        except Exception as exc:
            logger.warning("local_oss extract error: %s", exc)
            return [
                {
                    "url": u,
                    "title": "",
                    "content": "",
                    "error": f"Crawl4AI extract failed: {exc}",
                }
                for u in urls
            ]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Local OSS",
            "badge": "free · self-hosted",
            "tag": "SearXNG search plus Crawl4AI page extraction.",
            "env_vars": [
                {
                    "key": "SEARXNG_API_URL",
                    "prompt": "SearXNG instance URL",
                    "url": "https://docs.searxng.org/",
                },
                {
                    "key": "CRAWL4AI_API_URL",
                    "prompt": "Crawl4AI API URL",
                    "url": "https://docs.crawl4ai.com/core/self-hosting/",
                },
            ],
        }


class Crawl4AIWebSearchProvider(WebSearchProvider):
    """Crawl4AI extract-only provider."""

    @property
    def name(self) -> str:
        return "crawl4ai"

    @property
    def display_name(self) -> str:
        return "Crawl4AI"

    def is_available(self) -> bool:
        return _has_env("CRAWL4AI_API_URL")

    def supports_search(self) -> bool:
        return False

    def supports_extract(self) -> bool:
        return True

    async def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return [{"url": u, "error": "Interrupted", "title": ""} for u in urls]

            logger.info("Crawl4AI extract: %d URL(s)", len(urls))
            return await _crawl4ai_extract(urls, format=kwargs.get("format"))
        except ValueError as exc:
            return [{"url": u, "title": "", "content": "", "error": str(exc)} for u in urls]
        except Exception as exc:
            logger.warning("Crawl4AI extract error: %s", exc)
            return [
                {
                    "url": u,
                    "title": "",
                    "content": "",
                    "error": f"Crawl4AI extract failed: {exc}",
                }
                for u in urls
            ]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Crawl4AI",
            "badge": "free · self-hosted",
            "tag": "Extract-only backend for pairing with any search provider.",
            "env_vars": [
                {
                    "key": "CRAWL4AI_API_URL",
                    "prompt": "Crawl4AI API URL",
                    "url": "https://docs.crawl4ai.com/core/self-hosting/",
                },
            ],
        }
