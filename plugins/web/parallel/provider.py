"""Parallel.ai web search (sync ``Parallel`` SDK) + async extract (``AsyncParallel``).

Env: ``PARALLEL_API_KEY`` (https://parallel.ai), optional
``PARALLEL_SEARCH_MODE`` = agentic (default) | fast | one-shot.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List

from plugins.web._common import (
    SEARCH_LIMIT_CAP, BaseWebSearchProvider, cached_sdk_client, document, keyless_extract, keyless_search,
    keyless_variant_schema, page_error, provider_env, run_extract_async, run_search, search_ok, use_keyless, web_hit,
)

logger = logging.getLogger(__name__)

_MISSING_KEY = "PARALLEL_API_KEY environment variable not set. Get your API key at https://parallel.ai"


def _client(slot: str, cls_name: str) -> Any:
    def _factory(api_key: str) -> Any:
        import parallel  # deliberately lazy
        return getattr(parallel, cls_name)(api_key=api_key)

    # No eager pm ensure_import here: cached_sdk_client checks the API key FIRST and only
    # then lazy-ensures the extra. Pre-ensuring inverted that order on hosts where the
    # optional extra is absent and lazy installs are disabled, surfacing "Parallel SDK
    # not installed" instead of the missing-key error the keyless-disabled contract
    # (and exa's identical client builder) promises.
    return cached_sdk_client(slot, "PARALLEL_API_KEY", _MISSING_KEY, "parallel-web", _factory)


def _get_sync_client() -> Any:
    return _client("_parallel_client", "Parallel")


def _get_async_client() -> Any:
    return _client("_async_parallel_client", "AsyncParallel")


def _resolve_search_mode() -> str:
    mode = os.getenv("PARALLEL_SEARCH_MODE", "agentic").lower().strip()
    return mode if mode in {"fast", "one-shot", "agentic"} else "agentic"


class ParallelWebSearchProvider(BaseWebSearchProvider):
    """Parallel.ai search + async extract provider."""

    NAME = "parallel"
    DISPLAY_NAME = "Parallel"
    KEY_ENV = "PARALLEL_API_KEY"
    EXTRACT = True
    KEYLESS = True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        def _body() -> Dict[str, Any]:
            if use_keyless("parallel", provider_env("PARALLEL_API_KEY")):
                return keyless_search("Parallel", "parallel", query, limit, logger)
            mode = _resolve_search_mode()
            logger.info("Parallel search: '%s' (mode=%s, limit=%d)", query, mode, limit)
            response = _get_sync_client().beta.search(search_queries=[query], objective=query, mode=mode, max_results=min(limit, SEARCH_LIMIT_CAP))
            return search_ok([
                web_hit(r.url or "", r.title or "", " ".join(r.excerpts or []), i + 1)
                for i, r in enumerate(response.results or [])
            ])

        return run_search("Parallel", logger, _body, sdk=True)

    async def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        async def _body() -> List[Dict[str, Any]]:
            if use_keyless("parallel", provider_env("PARALLEL_API_KEY")):
                # Keyless ring is blocking HTTP — hop off the event loop.
                return await asyncio.to_thread(keyless_extract, "Parallel", "parallel", urls, logger)
            logger.info("Parallel extract: %d URL(s)", len(urls))
            response = await _get_async_client().beta.extract(urls=urls, full_content=True)
            results = [document(r.url or "", r.title or "", r.full_content or "\n\n".join(r.excerpts or [])) for r in response.results or []]
            return results + [
                {**page_error(e.url or "", e.content or e.error_type or "extraction failed"), "metadata": {"sourceURL": e.url or ""}}
                for e in response.errors or []
            ]

        return await run_extract_async("Parallel", logger, urls, _body, sdk=True)

    def get_setup_schema(self) -> Dict[str, Any]:
        return keyless_variant_schema(
            "Parallel", "PARALLEL_API_KEY", "https://parallel.ai",
            free_tag="Objective-tuned search + page extraction on Parallel's anonymous free tier. Rate-limited under burst load.",
            paid_tag="Objective-tuned search + parallel page extraction via the Parallel SDK. Unthrottled, guaranteed service.",
        )
