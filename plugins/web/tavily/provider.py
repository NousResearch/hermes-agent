"""Tavily web search + content extraction — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. Two
capabilities advertised:

- ``supports_search()``  -> True (Tavily ``/search``)
- ``supports_extract()`` -> True (Tavily ``/extract``)

Both are sync — the underlying call is ``httpx.post(...)``.

Config keys this provider responds to::

    web:
      search_backend: "tavily"     # explicit per-capability
      extract_backend: "tavily"    # explicit per-capability
      backend: "tavily"            # shared fallback for both

Credentials resolve pool-first: keys added via ``hermes auth add tavily``
are rotated on 401/403/429. When no pool credential is available the
provider falls back to ``TAVILY_API_KEY`` (config-aware; see
:func:`agent.web_search_provider.get_provider_env`).

Env vars::

    TAVILY_API_KEY=...           # https://app.tavily.com/home (required)
    TAVILY_BASE_URL=...          # optional override of https://api.tavily.com
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from agent.credential_pool import CredentialPool, PooledCredential, load_pool
from agent.web_search_provider import WebSearchProvider, get_provider_env

logger = logging.getLogger(__name__)

_POOL_PROVIDER = "tavily"
_ROTATE_STATUSES = frozenset({401, 403, 429})


def _load_tavily_pool() -> Optional[CredentialPool]:
    """Return the Tavily credential pool, or ``None`` if the pool fails to load."""
    try:
        return load_pool(_POOL_PROVIDER)
    except Exception as exc:  # noqa: BLE001 — pool failures must not break web_search
        logger.warning("tavily: failed to load credential pool: %s", exc)
        return None


def _pool_runtime_api_key(entry: Any) -> str:
    if entry is None:
        return ""
    key = getattr(entry, "runtime_api_key", None) or getattr(entry, "access_token", "")
    return str(key or "").strip()


def _resolve_tavily_key() -> Tuple[Optional[str], Optional[CredentialPool], Optional[PooledCredential]]:
    """Resolve a Tavily API key (pool-first, env fallback). ``pool``/``entry`` are ``None`` for env."""
    pool = _load_tavily_pool()
    if pool is not None and pool.has_credentials():
        entry = pool.select()
        if entry is not None:
            api_key = _pool_runtime_api_key(entry)
            if api_key:
                return api_key, pool, entry

    return (get_provider_env("TAVILY_API_KEY") or None), None, None


def _pool_has_entries() -> bool:
    """Cheap probe for ``is_available()`` — no seeding/persisting."""
    try:
        from hermes_cli.auth import read_credential_pool

        entries = read_credential_pool(_POOL_PROVIDER)
    except Exception:
        return False
    return bool(entries)


def _tavily_request(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST to Tavily; on 401/403/429 from a pool key, rotate and retry once.

    Raises ``ValueError`` when no credential is available; the caller
    catches and surfaces it as a typed error response.
    """
    import httpx

    api_key, pool, entry = _resolve_tavily_key()
    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY environment variable not set. "
            "Get your API key at https://app.tavily.com/home"
        )

    base_url = get_provider_env("TAVILY_BASE_URL") or "https://api.tavily.com"
    url = f"{base_url}/{endpoint.lstrip('/')}"
    logger.info("Tavily %s request to %s", endpoint, url)

    for attempt in (1, 2):
        request_payload = dict(payload)  # don't mutate caller's dict
        request_payload["api_key"] = api_key

        response = httpx.post(url, json=request_payload, timeout=60)
        try:
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            status = getattr(exc.response, "status_code", None)
            if (
                attempt == 1
                and pool is not None
                and entry is not None
                and status in _ROTATE_STATUSES
            ):
                rotated = pool.mark_exhausted_and_rotate(
                    status_code=status,
                    error_context={"message": str(exc), "status_code": status},
                )
                if rotated is not None:
                    new_key = _pool_runtime_api_key(rotated)
                    if new_key and new_key != api_key:
                        logger.info(
                            "tavily: rotating credential after HTTP %s and retrying", status
                        )
                        entry = rotated
                        api_key = new_key
                        continue
            raise

    raise RuntimeError("tavily: _tavily_request exited retry loop unexpectedly")


def _normalize_tavily_search_results(response: Dict[str, Any]) -> Dict[str, Any]:
    """Map Tavily ``/search`` response to ``{success, data: {web: [...]}}``."""
    web_results = []
    for i, result in enumerate(response.get("results", [])):
        web_results.append(
            {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "description": result.get("content", ""),
                "position": i + 1,
            }
        )
    return {"success": True, "data": {"web": web_results}}


def _normalize_tavily_documents(
    response: Dict[str, Any], fallback_url: str = ""
) -> List[Dict[str, Any]]:
    """Map Tavily ``/extract`` response to standard documents.

    Documents follow the legacy LLM post-processing shape::

        {"url", "title", "content", "raw_content", "metadata"}

    Failures (``failed_results``, ``failed_urls``) become result entries
    with an ``error`` field rather than raising.
    """
    documents: List[Dict[str, Any]] = []
    for result in response.get("results", []):
        url = result.get("url", fallback_url)
        raw = result.get("raw_content", "") or result.get("content", "")
        documents.append(
            {
                "url": url,
                "title": result.get("title", ""),
                "content": raw,
                "raw_content": raw,
                "metadata": {"sourceURL": url, "title": result.get("title", "")},
            }
        )
    for fail in response.get("failed_results", []):
        documents.append(
            {
                "url": fail.get("url", fallback_url),
                "title": "",
                "content": "",
                "raw_content": "",
                "error": fail.get("error", "extraction failed"),
                "metadata": {"sourceURL": fail.get("url", fallback_url)},
            }
        )
    for fail_url in response.get("failed_urls", []):
        url_str = fail_url if isinstance(fail_url, str) else str(fail_url)
        documents.append(
            {
                "url": url_str,
                "title": "",
                "content": "",
                "raw_content": "",
                "error": "extraction failed",
                "metadata": {"sourceURL": url_str},
            }
        )
    return documents


class TavilyWebSearchProvider(WebSearchProvider):
    """Tavily search + extract provider."""

    @property
    def name(self) -> str:
        return "tavily"

    @property
    def display_name(self) -> str:
        return "Tavily"

    def is_available(self) -> bool:
        """Return True when a Tavily credential is available via env or pool. Must stay cheap."""
        if get_provider_env("TAVILY_API_KEY"):
            return True
        return _pool_has_entries()

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a Tavily search."""
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}

            logger.info("Tavily search: '%s' (limit=%d)", query, limit)
            raw = _tavily_request(
                "search",
                {
                    "query": query,
                    "max_results": min(limit, 20),
                    "include_raw_content": False,
                    "include_images": False,
                },
            )
            return _normalize_tavily_search_results(raw)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 — including httpx errors
            logger.warning("Tavily search error: %s", exc)
            return {"success": False, "error": f"Tavily search failed: {exc}"}

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract content from one or more URLs via Tavily.

        Sync — the underlying call is httpx.post(...). Returns the legacy
        list-of-results shape; per-URL failures become items with ``error``.
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return [
                    {"url": u, "error": "Interrupted", "title": ""} for u in urls
                ]

            logger.info("Tavily extract: %d URL(s)", len(urls))
            raw = _tavily_request(
                "extract",
                {
                    "urls": urls,
                    "include_images": False,
                },
            )
            return _normalize_tavily_documents(
                raw, fallback_url=urls[0] if urls else ""
            )
        except ValueError as exc:
            return [{"url": u, "title": "", "content": "", "error": str(exc)} for u in urls]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tavily extract error: %s", exc)
            return [
                {"url": u, "title": "", "content": "", "error": f"Tavily extract failed: {exc}"}
                for u in urls
            ]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Tavily",
            "badge": "paid",
            "tag": "Search + extract in one provider.",
            "env_vars": [
                {
                    "key": "TAVILY_API_KEY",
                    "prompt": "Tavily API key",
                    "url": "https://app.tavily.com/home",
                },
            ],
        }
