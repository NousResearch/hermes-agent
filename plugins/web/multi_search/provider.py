"""Multi-provider web search: Exa + Tavily + web-search-prime fan-out.

Registers as a single ``WebSearchProvider`` named ``multi-search``. When
configured as ``web.backend``, ``web_search`` calls THREE backends in
parallel:

1. **Tavily** — via the registered ``tavily`` provider (keyword search)
2. **Exa** — via the registered ``exa`` provider (neural/semantic search)
3. **web-search-prime** — via the z.ai MCP tool ``mcp_web_search_prime_web_search_prime``
   (Chinese-region optimized search)

Results are interleaved (round-robin), deduplicated by normalized URL, and
returned as a single merged set — maximizing recall and diversity.

For ``web_extract``, Tavily's batch ``/extract`` is primary; Exa
``get_contents`` is the fallback for any URLs Tavily failed on.

Requirements:
    - ``EXA_API_KEY`` and ``TAVILY_API_KEY`` environment variables
    - ``web-search-prime`` MCP server configured in ``mcp_servers``

Config::

    web:
      backend: "multi-search"
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

# MCP tool name for web-search-prime (mcp_{server}_{tool})
_WSP_TOOL_NAME = "mcp_web_search_prime_web_search_prime"


def _normalize_url(url: str) -> str:
    """Normalize URL for dedup: lowercase scheme+host, strip trailing slash."""
    if not url:
        return ""
    u = urlparse(url.strip())
    if not u.scheme or not u.netloc:
        return url.strip().lower()
    path = u.path.rstrip("/")
    return f"{u.scheme.lower()}://{u.netloc.lower()}{path}"


def _search_via_tavily(query: str, limit: int) -> Dict[str, Any]:
    """Call the registered Tavily provider."""
    from agent.web_search_registry import get_provider

    p = get_provider("tavily")
    if p is None:
        return {"success": False, "error": "tavily provider not registered"}
    return p.search(query, limit)


def _search_via_exa(query: str, limit: int) -> Dict[str, Any]:
    """Call the registered Exa provider."""
    from agent.web_search_registry import get_provider

    p = get_provider("exa")
    if p is None:
        return {"success": False, "error": "exa provider not registered"}
    return p.search(query, limit)


def _search_via_wsp(query: str, limit: int) -> Dict[str, Any]:
    """Call web-search-prime via the MCP tool registry dispatch.

    Falls back gracefully if the MCP server is not connected.
    """
    try:
        from tools.registry import registry

        entry = registry.get_entry(_WSP_TOOL_NAME)
        if entry is None:
            return {"success": False, "error": "web-search-prime MCP not connected"}

        result_str = registry.dispatch(
            _WSP_TOOL_NAME,
            {
                "search_query": query,
                "content_size": "medium",
            },
        )
        result = json.loads(result_str) if isinstance(result_str, str) else result_str

        # web-search-prime returns results in a different shape — normalize
        # to {success, data: {web: [{title, url, description, position}]}}
        return _normalize_wsp_result(result, limit)
    except Exception as exc:  # noqa: BLE001
        logger.warning("web-search-prime search error: %s", exc)
        return {"success": False, "error": f"web-search-prime: {exc}"}


def _normalize_wsp_result(result: Any, limit: int) -> Dict[str, Any]:
    """Normalize web-search-prime MCP response to standard search shape.

    The MCP dispatch wraps the tool output as ``{"result": "<json_string>"}``
    where the inner JSON string is a serialized array of items with
    ``title``, ``link``, ``content``, ``refer`` fields. Some wrappers may
    return already-parsed shapes, so we try several known formats.
    """
    # ── Unwrap MCP dispatch envelope ──────────────────────────────────
    # registry.dispatch() returns json string; _search_via_wsp already
    # json.loads it. But the MCP tool itself wraps its payload as
    # {"result": "<another json string>"} — unwrap that here.
    if isinstance(result, dict):
        inner = result.get("result")
        if isinstance(inner, str):
            try:
                result = json.loads(inner)
            except (json.JSONDecodeError, ValueError):
                pass  # might be plain text

    for _ in range(2):
        if not isinstance(result, str):
            break
        try:
            result = json.loads(result)
        except (json.JSONDecodeError, ValueError):
            break

    # Check for error at envelope level
    if isinstance(result, dict) and result.get("error"):
        return {"success": False, "error": f"web-search-prime: {result['error']}"}

    # ── Extract items list from various shapes ────────────────────────
    items: List[Dict[str, Any]] = []

    if isinstance(result, list):
        # Already a list of result items
        items = result
    elif isinstance(result, dict):
        # Shape: {data: {web: [...]}}
        if isinstance(result.get("data"), dict):
            items = result["data"].get("web", [])
        # Shape: {results: [...]}
        if not items:
            items = result.get("results", [])
        # Shape: {searchResults: [...]} or {search_results: [...]}
        if not items:
            items = result.get("searchResults", result.get("search_results", []))
        # Shape: {result: [...]}  (already json.loads'd by caller)
        if not items and isinstance(result.get("result"), list):
            items = result["result"]

    web_results: List[Dict[str, Any]] = []
    for i, item in enumerate(items[:limit]):
        if not isinstance(item, dict):
            continue
        web_results.append(
            {
                "title": item.get("title", item.get("name", "")),
                "url": item.get("url", item.get("link", item.get("href", ""))),
                "description": item.get(
                    "description",
                    item.get("snippet", item.get("summary", item.get("content", ""))),
                ),
                "position": i + 1,
            }
        )

    if not web_results:
        return {"success": False, "error": "web-search-prime: no results parsed"}

    return {"success": True, "data": {"web": web_results}}


class MultiSearchProvider(WebSearchProvider):
    """Triple fan-out search across Tavily + Exa + web-search-prime."""

    @property
    def name(self) -> str:
        return "multi-search"

    @property
    def display_name(self) -> str:
        return "Multi (Tavily+Exa+WSP)"

    def is_available(self) -> bool:
        """True when at least Tavily or Exa is configured.

        web-search-prime availability is checked dynamically per-call
        since MCP connection state changes.
        """
        return bool(
            os.getenv("TAVILY_API_KEY", "").strip()
            or os.getenv("EXA_API_KEY", "").strip()
        )

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Call Tavily, Exa, and web-search-prime in parallel; merge & dedup."""
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}
        except ImportError:
            pass

        per_provider_limit = max(limit, 5)

        tavily_result: Dict[str, Any] = {}
        exa_result: Dict[str, Any] = {}
        wsp_result: Dict[str, Any] = {}

        # ── Parallel triple fan-out ─────────────────────────────────────
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                "tavily": pool.submit(_search_via_tavily, query, per_provider_limit),
                "exa": pool.submit(_search_via_exa, query, per_provider_limit),
                "wsp": pool.submit(_search_via_wsp, query, per_provider_limit),
            }
            for name, fut in futures.items():
                try:
                    result = fut.result(timeout=45)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("%s search raised: %s", name, exc)
                    result = {"success": False, "error": f"{name}: {exc}"}
                if name == "tavily":
                    tavily_result = result
                elif name == "exa":
                    exa_result = result
                else:
                    wsp_result = result

        # ── Collect errors for logging ──────────────────────────────────
        errors: List[str] = []
        sources_data: Dict[str, List[Dict[str, Any]]] = {}

        for name, result in [("tavily", tavily_result), ("exa", exa_result), ("wsp", wsp_result)]:
            if result.get("success"):
                sources_data[name] = result.get("data", {}).get("web", [])
            else:
                errors.append(f"{name}: {result.get('error', 'unknown')}")
                sources_data[name] = []

        # ── Interleave (round-robin) for diversity, dedup by URL ────────
        seen: set[str] = set()
        merged: List[Dict[str, Any]] = []
        source_order = ("tavily", "exa", "wsp")
        max_len = max((len(sources_data[s]) for s in source_order), default=0)

        position = 1
        for i in range(max_len):
            for src in source_order:
                items = sources_data[src]
                if i < len(items):
                    item = dict(items[i])
                    norm = _normalize_url(item.get("url", ""))
                    if norm and norm in seen:
                        continue
                    if norm:
                        seen.add(norm)
                    item["position"] = position
                    item["source"] = src
                    merged.append(item)
                    position += 1

        if not merged:
            return {
                "success": False,
                "error": "; ".join(errors) or "All providers returned no results.",
            }

        if errors:
            logger.warning(
                "Multi-search partial failures: %s (returning %d merged results)",
                "; ".join(errors),
                len(merged),
            )

        logger.info(
            "Multi-search '%s': tavily=%d exa=%d wsp=%d → merged=%d (deduped)",
            query,
            len(sources_data["tavily"]),
            len(sources_data["exa"]),
            len(sources_data["wsp"]),
            len(merged),
        )

        return {"success": True, "data": {"web": merged}}

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Tavily batch extract as primary; Exa fallback for failed URLs."""
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return [{"url": u, "error": "Interrupted", "title": ""} for u in urls]
        except ImportError:
            pass

        from agent.web_search_registry import get_provider

        tavily_p = get_provider("tavily")
        exa_p = get_provider("exa")

        results: List[Dict[str, Any]] = []

        if tavily_p is not None:
            tavily_docs = tavily_p.extract(urls)
            failed_urls: List[str] = []
            for doc, url in zip(tavily_docs, urls):
                if doc.get("error") or not (doc.get("content") or "").strip():
                    failed_urls.append(url)
                else:
                    results.append(doc)

            if failed_urls and exa_p is not None:
                logger.info(
                    "Exa fallback extract for %d/%d URLs", len(failed_urls), len(urls)
                )
                exa_docs = exa_p.extract(failed_urls)
                results.extend(exa_docs)
            elif failed_urls:
                results.extend(
                    d for d, u in zip(tavily_docs, urls) if u in failed_urls
                )
        elif exa_p is not None:
            results = exa_p.extract(urls)
        else:
            results = [
                {"url": u, "title": "", "content": "", "error": "No extract provider"}
                for u in urls
            ]

        return results

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Multi Search",
            "badge": "paid",
            "tag": "Fan-out across Tavily + Exa + web-search-prime for maximum recall.",
            "env_vars": [
                {"key": "TAVILY_API_KEY", "prompt": "Tavily API key", "url": "https://app.tavily.com/home"},
                {"key": "EXA_API_KEY", "prompt": "Exa API key", "url": "https://exa.ai"},
            ],
        }
