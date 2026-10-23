"""Exa web search + content extraction via the ``exa-py`` SDK (lazy-installed).

Env: ``EXA_API_KEY`` (https://exa.ai). Both methods are sync — Exa's SDK is
sync-only; the dispatcher threads extract when the caller is async.

Optional ``web.exa`` config (keyed path only; unset = Exa's defaults):
``type`` (auto | lite | fast | deep-lite | deep), ``dynamic_highlights`` (bool, research-preview
beta), ``max_age_hours`` (-1..720, Exa's content cache age, applied only when a call reaches Exa;
not sent on ``lite`` searches, which reject it), ``extract_max_characters`` (1..1,000,000, Exa's
``text.maxCharacters`` for extract). The extract cap cuts the page at Exa, before Hermes truncation:
text past it never reaches the head+tail window, the stored copy or a larger ``char_limit``.

``web_search(depth=...)`` on the keyed path: quick sends ``type: lite``, deep sends ``type: deep``,
standard keeps ``web.exa.type``. The keyless tier has no type option and ignores depth.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from plugins.web._common import (
    BaseWebSearchProvider, cached_sdk_client, document, exa_crawl_status, keyless_extract, keyless_search, keyless_variant_schema,
    provider_env, run_extract, run_search, search_ok, use_keyless, vendor_error_entry, web_hit,
)

logger = logging.getLogger(__name__)

_MISSING_KEY = "EXA_API_KEY environment variable not set. Get your API key at https://exa.ai"

_SEARCH_TYPES = ("auto", "lite", "fast", "deep-lite", "deep")
_DEPTH_TYPES = {"quick": "lite", "deep": "deep"}  # web_search depth -> Exa search type
# Exa-Beta token for Dynamic Highlights (https://exa.ai/docs/search/highlights).
_DYNAMIC_HIGHLIGHTS_BETA = "dynamic-highlights-2026-08-28"
_MAX_TEXT_CHARACTERS = 1_000_000  # Exa's ceiling for text.maxCharacters


def _exa_config() -> dict[str, Any]:
    """``web.exa`` from config.yaml ({} on miss)."""
    from tools.web_tools import _load_web_config  # lazy: tests patch tools.web_tools._load_web_config
    cfg = _load_web_config().get("exa")
    return cfg if isinstance(cfg, dict) else {}


def _int_setting(cfg: dict[str, Any], key: str, low: int, high: int) -> int | None:
    """``cfg[key]`` as an int in [low, high]; None when unset, and a warning plus None when invalid."""
    value = cfg.get(key)
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool) and low <= value <= high:
        return value
    logger.warning("Ignoring web.exa.%s=%r: expected an integer from %d to %d", key, value, low, high)
    return None


def _max_age_hours(cfg: dict[str, Any]) -> int | None:
    return _int_setting(cfg, "max_age_hours", -1, 720)


def _search_kwargs(cfg: dict[str, Any], depth: str = "standard") -> dict[str, Any]:
    """Keyword arguments for ``Exa.search`` beyond the query and result count."""
    search_type = _DEPTH_TYPES.get(depth) or str(cfg.get("type") or "auto").strip().lower()
    if search_type not in _SEARCH_TYPES:
        logger.warning("Ignoring web.exa.type=%r: expected one of %s", cfg.get("type"), ", ".join(_SEARCH_TYPES))
        search_type = "auto"
    dynamic = cfg.get("dynamic_highlights") is True
    contents: dict[str, Any] = {"highlights": {"dynamic": True} if dynamic else True}
    max_age = _max_age_hours(cfg)
    if max_age is not None and search_type != "lite":
        contents["max_age_hours"] = max_age
    kwargs: dict[str, Any] = {"contents": contents}
    if search_type != "auto":
        kwargs["type"] = search_type
    if dynamic:
        kwargs["betas"] = [_DYNAMIC_HIGHLIGHTS_BETA]
    return kwargs


def _contents_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Keyword arguments for ``Exa.get_contents`` beyond the URLs."""
    cap = _int_setting(cfg, "extract_max_characters", 1, _MAX_TEXT_CHARACTERS)
    kwargs: dict[str, Any] = {"text": True if cap is None else {"max_characters": cap}}
    max_age = _max_age_hours(cfg)
    if max_age is not None:
        kwargs["max_age_hours"] = max_age
    return kwargs


def _get_exa_client() -> Any:
    def _factory(api_key: str) -> Any:
        from exa_py import Exa  # deliberately lazy
        client = Exa(api_key=api_key)
        client.headers["x-exa-integration"] = "hermes-agent"
        return client

    return cached_sdk_client("_exa_client", "EXA_API_KEY", _MISSING_KEY, "exa", _factory)


def _crawl_status(error: Any) -> Any:
    """The origin status of a failed ``statuses[]`` entry: ``http_status_code``, else the one its tag names."""
    return getattr(error, "http_status_code", None) or exa_crawl_status(getattr(error, "tag", None))


class ExaWebSearchProvider(BaseWebSearchProvider):
    """Exa search + extract provider."""

    NAME = "exa"
    DISPLAY_NAME = "Exa"
    KEY_ENV = "EXA_API_KEY"
    EXTRACT = True
    KEYLESS = True

    def supports_search_depth(self) -> bool:
        return not use_keyless("exa", provider_env("EXA_API_KEY"))

    def search(self, query: str, limit: int = 5, depth: str = "standard") -> dict[str, Any]:
        def _body() -> dict[str, Any]:
            if use_keyless("exa", provider_env("EXA_API_KEY")):
                return keyless_search("Exa", "exa", query, limit, logger)
            logger.info("Exa search: '%s' (limit=%d)", query, limit)
            response = _get_exa_client().search(query, num_results=limit, **_search_kwargs(_exa_config(), depth))
            return search_ok([
                web_hit(r.url or "", r.title or "", " ".join(r.highlights or []), i + 1)
                for i, r in enumerate(response.results or [])
            ])

        return run_search("Exa", logger, _body, sdk=True)

    def extract(self, urls: list[str], **kwargs: Any) -> list[dict[str, Any]]:
        def _body() -> list[dict[str, Any]]:
            if use_keyless("exa", provider_env("EXA_API_KEY")):
                return keyless_extract("Exa", "exa", urls, logger)
            logger.info("Exa extract: %d URL(s)", len(urls))
            response = _get_exa_client().get_contents(urls, **_contents_kwargs(_exa_config()))
            results = [document(r.url or "", r.title or "", r.text or "") for r in response.results or []]
            # A URL Exa could not fetch is absent from ``results``; its reason is in ``statuses``.
            return results + [
                vendor_error_entry(s.id or "", status=_crawl_status(s.error), reason=getattr(s.error, "tag", None))
                for s in getattr(response, "statuses", None) or [] if s.status == "error"
            ]

        return run_extract("Exa", logger, urls, _body, sdk=True)

    def get_setup_schema(self) -> dict[str, Any]:
        return keyless_variant_schema(
            "Exa", "EXA_API_KEY", "https://exa.ai",
            free_tag="Semantic + neural web search with content extraction on Exa's anonymous free tier. Rate-limited under burst load.",
            paid_tag="Semantic + neural web search with content extraction via the Exa SDK. Unthrottled, guaranteed service.",
        )
