"""LLMLayer web search + scrape provider.

Uses the official ``llmlayer`` Python SDK lazily via ``tools.lazy_deps``.
Only LLMLayer's raw web_search and scrape endpoints are exposed here; the
answer, stream-answer, map, crawl, PDF, and YouTube endpoints are intentionally
out of scope for this backend.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from agent.web_search_provider import WebSearchProvider
from tools.website_policy import check_website_access

logger = logging.getLogger(__name__)

_llmlayer_client: Any = None
_llmlayer_client_config: tuple[str, str] | None = None


def _ensure_llmlayer_sdk_installed() -> None:
    """Trigger lazy install of the LLMLayer SDK when it is missing."""
    try:
        from tools.lazy_deps import ensure as _lazy_ensure

        _lazy_ensure("search.llmlayer", prompt=False)
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        raise ImportError(str(exc))


def _get_llmlayer_client() -> Any:
    """Lazy-load and cache the official LLMLayer SDK client."""
    global _llmlayer_client, _llmlayer_client_config

    api_key = os.getenv("LLMLAYER_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "LLMLAYER_API_KEY environment variable not set. "
            "Get your API key at https://llmlayer.ai"
        )

    base_url = os.getenv("LLMLAYER_BASE_URL", "https://api.llmlayer.dev").strip()
    base_url = base_url.rstrip("/") or "https://api.llmlayer.dev"
    config = (api_key, base_url)
    if _llmlayer_client is not None and _llmlayer_client_config == config:
        return _llmlayer_client

    _ensure_llmlayer_sdk_installed()
    from llmlayer import LLMLayerClient  # noqa: WPS433

    _llmlayer_client = LLMLayerClient(api_key=api_key, base_url=base_url)
    _llmlayer_client_config = config
    return _llmlayer_client


def _reset_client_for_tests() -> None:
    """Drop cached LLMLayer client so tests can re-instantiate cleanly."""
    global _llmlayer_client, _llmlayer_client_config

    _llmlayer_client = None
    _llmlayer_client_config = None


def _to_plain_dict(value: Any) -> Dict[str, Any]:
    """Return a plain dict for SDK models, dicts, or simple objects."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    if hasattr(value, "dict"):
        dumped = value.dict()
        return dumped if isinstance(dumped, dict) else {}
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return {}


def _get_field(value: Any, *names: str, default: Any = "") -> Any:
    """Read the first present field from a dict-like or object-like value."""
    if isinstance(value, dict):
        for name in names:
            if name in value and value[name] is not None:
                return value[name]
        return default
    for name in names:
        if hasattr(value, name):
            found = getattr(value, name)
            if found is not None:
                return found
    return default


def _search_results(response: Any) -> List[Any]:
    """Return the LLMLayer search results list from SDK or dict responses."""
    results = _get_field(response, "results", default=[])
    return results if isinstance(results, list) else []


def _formats_for_extract(format_value: str | None) -> List[str]:
    """Translate Hermes extract format into LLMLayer scrape formats."""
    if format_value == "html":
        return ["html"]
    if format_value == "markdown":
        return ["markdown"]
    return ["markdown", "html"]


def _blocked_result(url: str, blocked: Dict[str, str]) -> Dict[str, Any]:
    """Build a standard per-URL blocked result."""
    return {
        "url": url,
        "title": "",
        "content": "",
        "raw_content": "",
        "error": blocked["message"],
        "blocked_by_policy": {
            "host": blocked["host"],
            "rule": blocked["rule"],
            "source": blocked["source"],
        },
    }


class LLMLayerWebSearchProvider(WebSearchProvider):
    """LLMLayer search + scrape provider."""

    @property
    def name(self) -> str:
        return "llmlayer"

    @property
    def display_name(self) -> str:
        return "LLMLayer"

    def is_available(self) -> bool:
        """Return True when ``LLMLAYER_API_KEY`` is set to a non-empty value."""
        return bool(os.getenv("LLMLAYER_API_KEY", "").strip())

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute an LLMLayer web search."""
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}

            logger.info("LLMLayer search: '%s' (limit=%d)", query, limit)
            response = _get_llmlayer_client().search_web(
                query,
                search_type="general",
                timeout=60,
            )

            web_results = []
            for i, result in enumerate(_search_results(response)[:limit]):
                web_results.append(
                    {
                        "url": _get_field(result, "link", "url"),
                        "title": _get_field(result, "title", "source"),
                        "description": _get_field(
                            result,
                            "snippet",
                            "description",
                            "content",
                        ),
                        "position": i + 1,
                    }
                )

            return {"success": True, "data": {"web": web_results}}
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        except ImportError as exc:
            return {
                "success": False,
                "error": f"LLMLayer SDK not installed: {exc}",
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLMLayer search error: %s", exc)
            return {"success": False, "error": f"LLMLayer search failed: {exc}"}

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract page content from URLs via LLMLayer scrape.

        Uses only the LLMLayer scrape endpoint. Per-URL policy blocks and
        scrape failures are returned as result entries with ``error`` fields.
        """
        from tools.interrupt import is_interrupted

        if is_interrupted():
            return [{"url": u, "error": "Interrupted", "title": ""} for u in urls]

        format_value = kwargs.get("format")
        formats = _formats_for_extract(format_value)
        results: List[Dict[str, Any]] = []

        for url in urls:
            blocked = check_website_access(url)
            if blocked:
                logger.info(
                    "Blocked LLMLayer scrape for %s by rule %s",
                    blocked["host"],
                    blocked["rule"],
                )
                results.append(_blocked_result(url, blocked))
                continue

            try:
                logger.info("LLMLayer scrape: %s", url)
                response = _get_llmlayer_client().scrape(
                    url,
                    formats=formats,
                    include_images=False,
                    include_links=True,
                    timeout=60,
                )
                payload = _to_plain_dict(response)
                metadata = _to_plain_dict(payload.get("metadata"))
                final_url = (
                    payload.get("url")
                    or metadata.get("sourceURL")
                    or metadata.get("url")
                    or url
                )
                final_blocked = check_website_access(final_url)
                if final_blocked:
                    logger.info(
                        "Blocked redirected LLMLayer scrape for %s by rule %s",
                        final_blocked["host"],
                        final_blocked["rule"],
                    )
                    results.append(_blocked_result(final_url, final_blocked))
                    continue

                title = payload.get("title") or metadata.get("title", "")
                if format_value == "html":
                    content = payload.get("html") or payload.get("markdown") or ""
                else:
                    content = payload.get("markdown") or payload.get("html") or ""
                metadata.setdefault("sourceURL", final_url)
                if title:
                    metadata.setdefault("title", title)

                results.append(
                    {
                        "url": final_url,
                        "title": title,
                        "content": content,
                        "raw_content": content,
                        "metadata": metadata,
                    }
                )
            except ValueError as exc:
                results.append(
                    {
                        "url": url,
                        "title": "",
                        "content": "",
                        "raw_content": "",
                        "error": str(exc),
                    }
                )
            except ImportError as exc:
                results.append(
                    {
                        "url": url,
                        "title": "",
                        "content": "",
                        "raw_content": "",
                        "error": f"LLMLayer SDK not installed: {exc}",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLMLayer scrape error for %s: %s", url, exc)
                results.append(
                    {
                        "url": url,
                        "title": "",
                        "content": "",
                        "raw_content": "",
                        "error": f"LLMLayer scrape failed: {exc}",
                    }
                )

        return results

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "LLMLayer",
            "badge": "paid",
            "tag": "Web search and scrape via LLMLayer.",
            "env_vars": [
                {
                    "key": "LLMLAYER_API_KEY",
                    "prompt": "LLMLayer API key",
                    "url": "https://llmlayer.ai",
                },
            ],
        }
