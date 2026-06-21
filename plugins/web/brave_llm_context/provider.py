"""Brave Search LLM Context provider.

Search-only provider for Brave's LLM context endpoint. It uses the same
``BRAVE_SEARCH_API_KEY`` credential as ``brave-free`` but returns grounding
results shaped for LLM context.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

_BRAVE_LLM_CONTEXT_ENDPOINT = "https://api.search.brave.com/res/v1/llm/context"


def _env_int(
    name: str,
    default: int,
    *,
    minimum: int = 1,
    maximum: Optional[int] = None,
) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.debug("Ignoring invalid %s=%r; using %d", name, raw, default)
        return default
    value = max(minimum, value)
    if maximum is not None:
        value = min(value, maximum)
    return value


def _compact_snippets(snippets: Any) -> str:
    if isinstance(snippets, str):
        snippets_iter: Iterable[Any] = [snippets]
    elif isinstance(snippets, list):
        snippets_iter = snippets
    else:
        snippets_iter = []

    parts: List[str] = []
    seen: set[str] = set()
    for item in snippets_iter:
        text = " ".join(str(item or "").split())
        if not text or text in seen:
            continue
        seen.add(text)
        parts.append(text)
    return " ".join(parts)


def _age_label(raw_age: Any) -> str:
    """Return the most useful Brave ``sources[url].age`` label.

    Brave documents ``age`` as an array like
    ``["Wednesday, January 15, 2025", "2025-01-15", "392 days ago"]``.
    The final item is the compact human-facing label, while some mocked or
    older responses use a plain string. Handle both without leaking Python
    list reprs into result descriptions.
    """
    if isinstance(raw_age, list):
        for item in reversed(raw_age):
            text = str(item or "").strip()
            if text:
                return text
        return ""
    return str(raw_age or "").strip()


def _source_suffix(source: Mapping[str, Any]) -> str:
    suffix_parts: List[str] = []
    hostname = str(source.get("hostname") or "").strip()
    age = _age_label(source.get("age"))
    if hostname:
        suffix_parts.append(hostname)
    if age:
        suffix_parts.append(age)
    return " | ".join(suffix_parts)


def _description_for(item: Mapping[str, Any], source: Mapping[str, Any]) -> str:
    description = _compact_snippets(item.get("snippets"))
    suffix = _source_suffix(source)
    if suffix and suffix not in description:
        if description:
            return f"{description} ({suffix})"
        return suffix
    return description


def _threshold_mode() -> str:
    mode = (
        os.getenv("BRAVE_LLM_CONTEXT_CONTEXT_THRESHOLD_MODE", "balanced")
        .strip()
        .lower()
    )
    if mode in {"strict", "balanced", "lenient", "disabled"}:
        return mode
    logger.debug(
        "Ignoring invalid BRAVE_LLM_CONTEXT_CONTEXT_THRESHOLD_MODE=%r; using balanced",
        mode,
    )
    return "balanced"


def _title_for(item: Mapping[str, Any], source: Mapping[str, Any]) -> str:
    title = str(item.get("title") or source.get("title") or "").strip()
    if title:
        return title
    hostname = str(source.get("hostname") or "").strip()
    return hostname


def _iter_grounding_items(grounding: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    generic = grounding.get("generic") or []
    if isinstance(generic, list):
        for item in generic:
            if isinstance(item, Mapping):
                yield item

    poi = grounding.get("poi")
    if isinstance(poi, Mapping):
        yield poi

    maps = grounding.get("map") or []
    if isinstance(maps, list):
        for item in maps:
            if isinstance(item, Mapping):
                yield item


class BraveLLMContextWebSearchProvider(WebSearchProvider):
    """Search-only provider using Brave's LLM Context API."""

    @property
    def name(self) -> str:
        return "brave-llm-context"

    @property
    def display_name(self) -> str:
        return "Brave Search LLM Context"

    def is_available(self) -> bool:
        return bool(os.getenv("BRAVE_SEARCH_API_KEY", "").strip())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return False

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        import httpx

        api_key = os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
        if not api_key:
            return {"success": False, "error": "BRAVE_SEARCH_API_KEY is not set"}

        try:
            count = max(1, min(int(limit), 50))
        except (TypeError, ValueError):
            count = 5

        payload = {
            "q": query,
            "count": count,
            "maximum_number_of_urls": count,
            "maximum_number_of_tokens": _env_int(
                "BRAVE_LLM_CONTEXT_MAXIMUM_NUMBER_OF_TOKENS",
                8192,
                minimum=1024,
                maximum=32768,
            ),
            "maximum_number_of_snippets": _env_int(
                "BRAVE_LLM_CONTEXT_MAXIMUM_NUMBER_OF_SNIPPETS",
                50,
                minimum=1,
                maximum=100,
            ),
            "maximum_number_of_tokens_per_url": _env_int(
                "BRAVE_LLM_CONTEXT_MAXIMUM_NUMBER_OF_TOKENS_PER_URL",
                4096,
                minimum=512,
                maximum=8192,
            ),
            "context_threshold_mode": _threshold_mode(),
        }
        timeout = _env_int("BRAVE_LLM_CONTEXT_TIMEOUT_SECONDS", 20, maximum=120)

        try:
            resp = httpx.post(
                _BRAVE_LLM_CONTEXT_ENDPOINT,
                json=payload,
                headers={
                    "X-Subscription-Token": api_key,
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "Content-Type": "application/json",
                },
                timeout=timeout,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("Brave LLM Context HTTP error: %s", exc)
            return {
                "success": False,
                "error": f"Brave LLM Context returned HTTP {exc.response.status_code}",
            }
        except httpx.RequestError as exc:
            logger.warning("Brave LLM Context request error: %s", exc)
            return {"success": False, "error": f"Could not reach Brave LLM Context: {exc}"}

        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Brave LLM Context response parse error: %s", exc)
            return {
                "success": False,
                "error": "Could not parse Brave LLM Context response as JSON",
            }

        grounding = data.get("grounding") or {}
        if not isinstance(grounding, Mapping):
            grounding = {}
        sources = data.get("sources") or {}
        if not isinstance(sources, Mapping):
            sources = {}

        web_results: List[Dict[str, Any]] = []
        seen_urls: set[str] = set()
        for item in _iter_grounding_items(grounding):
            url = str(item.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            raw_source = sources.get(url)
            source = raw_source if isinstance(raw_source, Mapping) else {}
            web_results.append(
                {
                    "title": _title_for(item, source),
                    "url": url,
                    "description": _description_for(item, source),
                    "position": len(web_results) + 1,
                }
            )
            if len(web_results) >= count:
                break

        logger.info(
            "Brave LLM Context '%s': %d results (limit %d)",
            query,
            len(web_results),
            count,
        )

        return {"success": True, "data": {"web": web_results}}

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Brave Search LLM Context",
            "badge": "search-only",
            "tag": "Search-only Brave LLM Context endpoint. Uses BRAVE_SEARCH_API_KEY.",
            "env_vars": [
                {
                    "key": "BRAVE_SEARCH_API_KEY",
                    "prompt": "Brave Search API key",
                    "url": "https://brave.com/search/api/",
                },
            ],
        }
