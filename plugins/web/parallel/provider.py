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

    return cached_sdk_client(slot, "PARALLEL_API_KEY", _MISSING_KEY, "search.parallel", _factory)


def _get_sync_client() -> Any:
    return _client("_parallel_client", "Parallel")


def _get_async_client() -> Any:
    return _client("_async_parallel_client", "AsyncParallel")


def _resolve_search_mode() -> str:
    mode = os.getenv("PARALLEL_SEARCH_MODE", "agentic").lower().strip()
    return mode if mode in {"fast", "one-shot", "agentic"} else "agentic"


class ParallelWebSearchProvider(BaseWebSearchProvider):
    """Parallel.ai search + async extract provider."""

    @property
    def name(self) -> str:
        return "parallel"

    @property
    def display_name(self) -> str:
        return "Parallel"

    def is_available(self) -> bool:
        """Return True when ``PARALLEL_API_KEY`` is set to a non-empty value.

        Deliberately does NOT consider the keyless free tier — that would
        let the legacy preference walk route keyed users of lower-priority
        backends onto Parallel's anonymous tier. Keyless availability is a
        separate, last-resort signal (:meth:`is_keyless_available`).
        """
        from agent.web_search_provider import get_provider_env

        return bool(get_provider_env("PARALLEL_API_KEY"))

    def is_keyless_available(self) -> bool:
        """Parallel serves anonymous free-tier calls via its public MCP endpoint.

        False when the user forced ``web.provider_tier.parallel: paid`` —
        an explicit paid selection must never silently resolve keyless.
        """
        from plugins.web.keyless_mcp import keyless_enabled, provider_tier

        return keyless_enabled() and provider_tier("parallel") != "paid"

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a Parallel search (sync).

        Uses the ``beta.search`` endpoint with the configured mode
        (``PARALLEL_SEARCH_MODE`` env var, default "agentic"). Limit is
        capped at 20 server-side.
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}

            from agent.web_search_provider import get_provider_env

            from plugins.web.keyless_mcp import search_with_failover, use_keyless

            if use_keyless("parallel", get_provider_env("PARALLEL_API_KEY")):
                # Keyless free tier — public MCP endpoint, no SDK needed.
                logger.info(
                    "Parallel keyless search: '%s' (limit=%d)", query, limit
                )
                return search_with_failover("parallel", query, limit)

            mode = _resolve_search_mode()
            logger.info("Parallel search: '%s' (mode=%s, limit=%d)", query, mode, limit)
            response = _get_sync_client().beta.search(search_queries=[query], objective=query, mode=mode, max_results=min(limit, SEARCH_LIMIT_CAP))
            return search_ok([
                web_hit(r.url or "", r.title or "", " ".join(r.excerpts or []), i + 1)
                for i, r in enumerate(response.results or [])
            ])

        return run_search("Parallel", logger, _body, sdk=True)

            from agent.web_search_provider import get_provider_env

            from plugins.web.keyless_mcp import extract_with_failover, use_keyless

            if use_keyless("parallel", get_provider_env("PARALLEL_API_KEY")):
                # Keyless free tier — blocking HTTP, so hop off the loop.
                import asyncio

                logger.info("Parallel keyless extract: %d URL(s)", len(urls))
                return await asyncio.to_thread(
                    extract_with_failover, "parallel", list(urls)
                )

            logger.info("Parallel extract: %d URL(s)", len(urls))
            response = await _get_async_client().beta.extract(urls=urls, full_content=True)
            results = [document(r.url or "", r.title or "", r.full_content or "\n\n".join(r.excerpts or [])) for r in response.results or []]
            return results + [
                {**page_error(e.url or "", e.content or e.error_type or "extraction failed"), "metadata": {"sourceURL": e.url or ""}}
                for e in response.errors or []
            ]

        return await run_extract_async("Parallel", logger, urls, _body, sdk=True)

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Parallel · Free (keyless)",
            "badge": "free · no key",
            "tag": (
                "Objective-tuned search + page extraction on Parallel's "
                "anonymous free tier. Rate-limited under burst load."
            ),
            "env_vars": [],
            "web_tier": "free",
            "variants": [
                {
                    "name": "Parallel · Paid (API key)",
                    "badge": "paid",
                    "tag": (
                        "Objective-tuned search + parallel page extraction "
                        "via the Parallel SDK. Unthrottled, guaranteed service."
                    ),
                    "env_vars": [
                        {
                            "key": "PARALLEL_API_KEY",
                            "prompt": "Parallel API key",
                            "url": "https://parallel.ai",
                        },
                    ],
                    "web_tier": "paid",
                },
            ],
        }
