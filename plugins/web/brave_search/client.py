"""Shared Brave Search API client and response normalization."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.search.brave.com/res/v1"
MAX_RAW_CHARS = 50_000


def get_brave_search_api_key() -> str:
    """Return the configured Brave Search API key, preferring the canonical env var."""
    return (
        os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
        or os.getenv("BRAVE_API_KEY", "").strip()
    )


def is_brave_search_configured() -> bool:
    return bool(get_brave_search_api_key())


def _clamp_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(number, maximum))


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _snippets(item: Dict[str, Any]) -> List[str]:
    snippets = item.get("snippets")
    if isinstance(snippets, list):
        return [_text(s) for s in snippets if _text(s)]
    extra = item.get("extra_snippets")
    values: List[str] = []
    if isinstance(extra, list):
        values.extend(_text(s) for s in extra if _text(s))
    description = _text(item.get("description"))
    if description:
        values.insert(0, description)
    return values


def _result_common(item: Dict[str, Any], position: int) -> Dict[str, Any]:
    return {
        "title": _text(item.get("title")),
        "url": _text(item.get("url")),
        "description": _text(item.get("description")),
        "position": position,
    }


def _parse_result_list(results: Iterable[Any]) -> List[Dict[str, Any]]:
    parsed = []
    for i, item in enumerate(results, start=1):
        if not isinstance(item, dict):
            continue
        entry = _result_common(item, i)
        extra = _snippets(item)
        if extra:
            entry["extra_snippets"] = extra
        parsed.append(entry)
    return parsed


def _parse_web_response(data: Dict[str, Any], *, limit: int) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "web": _parse_result_list(_as_list(_as_dict(data.get("web")).get("results")))[:limit],
        "news": _parse_result_list(_as_list(_as_dict(data.get("news")).get("results")))[:limit],
        "videos": _parse_result_list(_as_list(_as_dict(data.get("videos")).get("results")))[:limit],
        "discussions": _parse_result_list(_as_list(_as_dict(data.get("discussions")).get("results")))[:limit],
        "faq": _parse_result_list(_as_list(_as_dict(data.get("faq")).get("results")))[:limit],
        "locations": _parse_result_list(_as_list(_as_dict(data.get("locations")).get("results")))[:limit],
    }
    infobox = _as_dict(data.get("infobox"))
    infobox_results = _as_list(infobox.get("results"))
    if infobox_results:
        result["infobox"] = _parse_result_list(infobox_results)[:limit]
    elif infobox:
        result["infobox"] = [infobox]
    else:
        result["infobox"] = []
    return result


def _parse_images_response(data: Dict[str, Any], *, limit: int) -> Dict[str, Any]:
    images = []
    for i, item in enumerate(_as_list(_as_dict(data.get("results")).get("results") or data.get("results")), start=1):
        if not isinstance(item, dict):
            continue
        thumbnail = item.get("thumbnail") or {}
        properties = item.get("properties") or {}
        images.append(
            {
                "title": _text(item.get("title")),
                "url": _text(item.get("url") or item.get("image_url")),
                "page_url": _text(item.get("page_url") or item.get("source")),
                "thumbnail_url": _text(
                    thumbnail.get("src") if isinstance(thumbnail, dict) else thumbnail
                ),
                "source": _text(item.get("source")),
                "width": properties.get("width") if isinstance(properties, dict) else None,
                "height": properties.get("height") if isinstance(properties, dict) else None,
                "position": i,
            }
        )
        if len(images) >= limit:
            break
    return {"images": images}


def _parse_llm_context_response(data: Dict[str, Any], *, limit: int) -> Dict[str, Any]:
    grounding = _as_dict(data.get("grounding"))
    generic = _as_list(grounding.get("generic"))
    parsed = []
    for i, item in enumerate(generic, start=1):
        if not isinstance(item, dict):
            continue
        parsed.append(
            {
                "title": _text(item.get("title")),
                "url": _text(item.get("url")),
                "snippets": _snippets(item),
                "position": i,
            }
        )
        if len(parsed) >= limit:
            break
    return {"llm_context": parsed}


def _parse_suggest_response(data: Dict[str, Any]) -> Dict[str, Any]:
    raw = data.get("results") or data.get("suggestions") or data.get("query") or []
    suggestions = []
    for item in _as_list(raw):
        if isinstance(item, dict):
            value = item.get("query") or item.get("text") or item.get("value")
            if value:
                suggestions.append(_text(value))
        elif item:
            suggestions.append(_text(item))
    return {"suggestions": suggestions}


def _bounded_raw(data: Dict[str, Any]) -> Dict[str, Any]:
    raw = json.dumps(data, ensure_ascii=False)
    if len(raw) <= MAX_RAW_CHARS:
        return {"raw": data, "truncated": False}
    return {
        "raw": raw[:MAX_RAW_CHARS],
        "truncated": True,
        "raw_size_chars": len(raw),
    }


class BraveSearchApiClient:
    """Small synchronous client for Brave Search API endpoints."""

    def __init__(self, api_key: Optional[str] = None, timeout: float = 15.0) -> None:
        self.api_key = (api_key or get_brave_search_api_key()).strip()
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        return {
            "X-Subscription-Token": self.api_key,
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
        }

    def _request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.api_key:
            return {
                "success": False,
                "error": "BRAVE_SEARCH_API_KEY or BRAVE_API_KEY is not set",
            }

        try:
            response = httpx.get(
                f"{BASE_URL}{endpoint}",
                params={k: v for k, v in params.items() if v not in (None, "")},
                headers=self._headers(),
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            logger.warning("Brave Search API HTTP error: %s", status)
            detail = f"Brave Search API returned HTTP {status}"
            if status in {402, 403}:
                detail += ". The selected Brave Search API plan may not include this endpoint."
            elif status == 401:
                detail += ". Check BRAVE_SEARCH_API_KEY or BRAVE_API_KEY."
            elif status == 429:
                detail += ". Rate limit or quota was exceeded."
            return {"success": False, "error": detail}
        except httpx.RequestError as exc:
            logger.warning("Brave Search API request error: %s", exc.__class__.__name__)
            return {"success": False, "error": "Could not reach Brave Search API"}

        try:
            return {"success": True, "raw": response.json()}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Brave Search API JSON parse error: %s", exc)
            return {
                "success": False,
                "error": "Could not parse Brave Search API response as JSON",
            }

    def search_web(
        self,
        query: str,
        limit: int = 5,
        *,
        country: str = "ALL",
        search_lang: str = "en",
        safesearch: str = "moderate",
        freshness: Optional[str] = None,
        extra_snippets: bool = True,
        result_filter: Optional[str] = None,
        raw: bool = False,
    ) -> Dict[str, Any]:
        count = _clamp_int(limit, default=5, minimum=1, maximum=20)
        params = {
            "q": query,
            "count": count,
            "country": country,
            "search_lang": search_lang,
            "safesearch": safesearch,
            "freshness": freshness,
            "extra_snippets": "true" if extra_snippets else None,
            "result_filter": result_filter,
        }
        response = self._request("/web/search", params)
        if not response.get("success"):
            return response
        data = _as_dict(response.get("raw"))
        parsed = _parse_web_response(data, limit=count)
        if raw:
            parsed.update(_bounded_raw(data))
        return {"success": True, "data": parsed}

    def llm_context(
        self,
        query: str,
        limit: int = 5,
        *,
        country: str = "ALL",
        search_lang: str = "en",
        safesearch: str = "moderate",
        freshness: Optional[str] = None,
        max_llm_tokens: int = 8192,
        raw: bool = False,
    ) -> Dict[str, Any]:
        count = _clamp_int(limit, default=5, minimum=1, maximum=50)
        tokens = _clamp_int(max_llm_tokens, default=8192, minimum=1024, maximum=32768)
        response = self._request(
            "/llm/context",
            {
                "q": query,
                "count": count,
                "country": country,
                "search_lang": search_lang,
                "safesearch": safesearch,
                "freshness": freshness,
                "maximum_number_of_tokens": tokens,
            },
        )
        if not response.get("success"):
            return response
        data = _as_dict(response.get("raw"))
        parsed = _parse_llm_context_response(data, limit=count)
        if raw:
            parsed.update(_bounded_raw(data))
        return {"success": True, "data": parsed}

    def search_images(
        self,
        query: str,
        limit: int = 5,
        *,
        country: str = "ALL",
        search_lang: str = "en",
        safesearch: str = "strict",
        freshness: Optional[str] = None,
        raw: bool = False,
    ) -> Dict[str, Any]:
        count = _clamp_int(limit, default=5, minimum=1, maximum=20)
        response = self._request(
            "/images/search",
            {
                "q": query,
                "count": count,
                "country": country,
                "search_lang": search_lang,
                "safesearch": "off" if safesearch == "off" else "strict",
                "freshness": freshness,
            },
        )
        if not response.get("success"):
            return response
        data = _as_dict(response.get("raw"))
        parsed = _parse_images_response(data, limit=count)
        if raw:
            parsed.update(_bounded_raw(data))
        return {"success": True, "data": parsed}

    def search_news(self, query: str, limit: int = 5, **kwargs: Any) -> Dict[str, Any]:
        return self._search_filtered("/news/search", query, limit, "news", **kwargs)

    def search_videos(self, query: str, limit: int = 5, **kwargs: Any) -> Dict[str, Any]:
        return self._search_filtered("/videos/search", query, limit, "videos", **kwargs)

    def search_discussions(self, query: str, limit: int = 5, **kwargs: Any) -> Dict[str, Any]:
        return self.search_web(query, limit, result_filter="discussions", **kwargs)

    def _search_filtered(
        self,
        endpoint: str,
        query: str,
        limit: int,
        key: str,
        *,
        country: str = "ALL",
        search_lang: str = "en",
        safesearch: str = "moderate",
        freshness: Optional[str] = None,
        raw: bool = False,
        **_: Any,
    ) -> Dict[str, Any]:
        count = _clamp_int(limit, default=5, minimum=1, maximum=50)
        response = self._request(
            endpoint,
            {
                "q": query,
                "count": count,
                "country": country,
                "search_lang": search_lang,
                "safesearch": safesearch,
                "freshness": freshness,
            },
        )
        if not response.get("success"):
            return response
        data = _as_dict(response.get("raw"))
        section = _as_dict(data.get(key))
        results = data.get("results") or section.get("results") or []
        parsed = _parse_web_response({key: {"results": results}}, limit=count)
        if raw:
            parsed.update(_bounded_raw(data))
        return {"success": True, "data": parsed}

    def suggest(self, query: str, *, country: str = "ALL", raw: bool = False) -> Dict[str, Any]:
        response = self._request("/suggest/search", {"q": query, "country": country})
        if not response.get("success"):
            return response
        data = _as_dict(response.get("raw"))
        parsed = _parse_suggest_response(data)
        if raw:
            parsed.update(_bounded_raw(data))
        return {"success": True, "data": parsed}
