"""iFlow Search web search + content extraction provider.

Subclasses :class:`agent.web_search_provider.WebSearchProvider` and maps Hermes
web tools to the iFlow Search API:

- ``search()``  -> ``POST /api/search/webSearch``
- ``extract()`` -> ``POST /api/search/webFetch``

Config keys this provider responds to::

    web:
      search_backend: "iflow"
      extract_backend: "iflow"
      backend: "iflow"

Env vars::

    IFLOW_API_KEY=...                       # required
    IFLOW_BASE_URL=https://platform.iflow.cn # optional
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from agent.web_search_provider import WebSearchProvider

logger = logging.getLogger(__name__)

DEFAULT_IFLOW_BASE_URL = "https://platform.iflow.cn"


def _missing_key_error() -> str:
    from hermes_constants import display_hermes_home

    return (
        "IFLOW_API_KEY is not set. Run hermes tools and select iFlow Search, "
        f"or set IFLOW_API_KEY in {display_hermes_home()}/.env."
    )


def _env_value(name: str) -> str:
    """Resolve env values through Hermes' config-aware env layer."""
    try:
        from hermes_cli.config import get_env_value

        value = get_env_value(name)
    except Exception:
        value = None
    if value is None:
        value = os.getenv(name, "")
    return (value or "").strip()


def _iflow_api_key() -> str:
    return _env_value("IFLOW_API_KEY")


def _iflow_base_url() -> str:
    return (_env_value("IFLOW_BASE_URL") or DEFAULT_IFLOW_BASE_URL).rstrip("/")


def _clamp_limit(limit: int) -> int:
    try:
        value = int(limit)
    except (TypeError, ValueError):
        value = 5
    return max(1, min(value, 100))


def _headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _http_error_message(status_code: int) -> str:
    if status_code in {401, 403}:
        return "iFlow API key is invalid or unauthorized"
    if status_code == 429:
        return "iFlow rate limit exceeded"
    if 500 <= status_code <= 599:
        return f"iFlow service error: HTTP {status_code}"
    return f"iFlow returned HTTP {status_code}"


def _parse_json_response(response: Any) -> Dict[str, Any]:
    try:
        data = response.json()
    except Exception as exc:  # noqa: BLE001
        raise ValueError("iFlow returned an invalid JSON response") from exc
    if not isinstance(data, dict):
        raise ValueError("iFlow returned an invalid JSON response")
    return data


def _business_error(data: Dict[str, Any]) -> str | None:
    success = data.get("success")
    code = data.get("code")
    if success is False or (code not in {None, "", "200", 200}):
        message = str(data.get("message") or "iFlow request failed")
        if code not in {None, ""}:
            return f"iFlow request failed ({code}): {message}"
        return f"iFlow request failed: {message}"
    return None


def _compact_metadata(result: Dict[str, Any]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for source_key, target_key in (
        ("source", "source"),
        ("date", "published_time"),
        ("published_time", "published_time"),
        ("metadata", "metadata"),
    ):
        value = result.get(source_key)
        if value is not None and value != "":
            metadata[target_key] = value
    return metadata


def _normalize_search_response(data: Dict[str, Any], limit: int) -> Dict[str, Any]:
    error = _business_error(data)
    if error:
        return {"success": False, "error": error}

    payload = data.get("data") or {}
    if not isinstance(payload, dict):
        payload = {}
    raw_results = payload.get("organic") or payload.get("results") or payload.get("web") or []
    if not isinstance(raw_results, list):
        raw_results = []

    web_results = []
    for index, item in enumerate(raw_results[:limit]):
        if not isinstance(item, dict):
            continue
        url = item.get("url") or item.get("link") or ""
        description = item.get("snippet") or item.get("content") or item.get("summary") or ""
        result: Dict[str, Any] = {
            "title": str(item.get("title") or ""),
            "url": str(url or ""),
            "description": str(description or ""),
            "position": int(item.get("position") or index + 1),
        }
        metadata = _compact_metadata(item)
        if metadata:
            result["metadata"] = metadata
        web_results.append(result)

    return {"success": True, "data": {"web": web_results}}


def _normalize_fetch_response(data: Dict[str, Any], fallback_url: str) -> Dict[str, Any]:
    error = _business_error(data)
    if error:
        return {
            "url": fallback_url,
            "title": "",
            "content": "",
            "raw_content": "",
            "error": error,
            "metadata": {"sourceURL": fallback_url},
        }

    payload = data.get("data") or {}
    if not isinstance(payload, dict):
        payload = {}
    url = str(payload.get("url") or fallback_url)
    title = str(payload.get("title") or "")
    content = str(
        payload.get("content")
        or payload.get("markdown")
        or payload.get("text")
        or payload.get("raw_content")
        or ""
    )
    metadata: Dict[str, Any] = {"sourceURL": url, "title": title}
    if "fromCache" in payload:
        metadata["fromCache"] = payload.get("fromCache")
    if "metadata" in payload and isinstance(payload["metadata"], dict):
        metadata["metadata"] = payload["metadata"]

    return {
        "url": url,
        "title": title,
        "content": content,
        "raw_content": content,
        "metadata": metadata,
    }


def _iflow_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST to iFlow and return parsed JSON without exposing credentials."""
    import httpx

    api_key = _iflow_api_key()
    if not api_key:
        raise ValueError(_missing_key_error())

    url = f"{_iflow_base_url()}/{path.lstrip('/')}"
    try:
        response = httpx.post(
            url,
            json=payload,
            headers=_headers(api_key),
            timeout=30,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(_http_error_message(exc.response.status_code)) from exc
    except httpx.TimeoutException as exc:
        raise RuntimeError("iFlow request timeout") from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"Could not reach iFlow Search: {exc}") from exc

    return _parse_json_response(response)


class IFlowWebSearchProvider(WebSearchProvider):
    """iFlow Search provider for Hermes web search and extract tools."""

    @property
    def name(self) -> str:
        return "iflow"

    @property
    def display_name(self) -> str:
        return "iFlow Search"

    def is_available(self) -> bool:
        return bool(_iflow_api_key())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute web search via iFlow ``/api/search/webSearch``."""
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}

            count = _clamp_limit(limit)
            logger.info("iFlow search: '%s' (limit=%d)", query, count)
            data = _iflow_post(
                "/api/search/webSearch",
                {"keywords": query, "num": count},
            )
            return _normalize_search_response(data, count)
        except Exception as exc:  # noqa: BLE001
            logger.warning("iFlow search error: %s", exc)
            return {"success": False, "error": str(exc)}

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract URL content via iFlow ``/api/search/webFetch``."""
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return [
                    {
                        "url": url,
                        "title": "",
                        "content": "",
                        "raw_content": "",
                        "error": "Interrupted",
                    }
                    for url in urls
                ]

            results: List[Dict[str, Any]] = []
            for url in urls:
                try:
                    data = _iflow_post("/api/search/webFetch", {"url": url})
                    results.append(_normalize_fetch_response(data, fallback_url=url))
                except Exception as exc:  # noqa: BLE001
                    logger.warning("iFlow fetch error for %s: %s", url, exc)
                    results.append(
                        {
                            "url": url,
                            "title": "",
                            "content": "",
                            "raw_content": "",
                            "error": str(exc),
                            "metadata": {"sourceURL": url},
                        }
                    )
            return results
        except Exception as exc:  # noqa: BLE001
            logger.warning("iFlow extract error: %s", exc)
            return [
                {
                    "url": url,
                    "title": "",
                    "content": "",
                    "raw_content": "",
                    "error": f"iFlow extract failed: {exc}",
                    "metadata": {"sourceURL": url},
                }
                for url in urls
            ]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "iFlow Search",
            "badge": "paid",
            "tag": "Search + extract in one provider.",
            "env_vars": [
                {
                    "key": "IFLOW_API_KEY",
                    "prompt": "Enter your IFLOW_API_KEY",
                    "url": "https://platform.iflow.cn/docs/",
                },
            ],
        }
