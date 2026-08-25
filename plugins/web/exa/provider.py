"""Exa web search + content extraction via the ``exa-py`` SDK (lazy-installed).

Env: ``EXA_API_KEY`` (https://exa.ai). Both methods are sync — Exa's SDK is
sync-only; the dispatcher threads extract when the caller is async.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from plugins.web._common import (
    BaseWebSearchProvider, cached_sdk_client, document, keyless_extract, keyless_search, keyless_variant_schema,
    provider_env, run_extract, run_search, search_ok, use_keyless, web_hit,
)

logger = logging.getLogger(__name__)

_MISSING_KEY = "EXA_API_KEY environment variable not set. Get your API key at https://exa.ai"


def _get_exa_client() -> Any:
    def _factory(api_key: str) -> Any:
        from exa_py import Exa  # deliberately lazy
        client = Exa(api_key=api_key)
        client.headers["x-exa-integration"] = "hermes-agent"
        return client

    return cached_sdk_client("_exa_client", "EXA_API_KEY", _MISSING_KEY, "search.exa", _factory)


class ExaWebSearchProvider(BaseWebSearchProvider):
    """Exa search + extract provider."""

    _wt._exa_client = None


class ExaWebSearchProvider(WebSearchProvider):
    """Exa search + extract provider.

    Both methods are sync — Exa's SDK is sync-only. The web_extract_tool
    dispatcher wraps sync extracts via ``asyncio.to_thread`` when it
    needs to keep the event loop responsive.
    """

    @property
    def name(self) -> str:
        return "exa"

    @property
    def display_name(self) -> str:
        return "Exa"

    def is_available(self) -> bool:
        """Return True when ``EXA_API_KEY`` is set to a non-empty value.

        Deliberately does NOT consider the keyless free tier — that would
        let the legacy preference walk route keyed users of lower-priority
        backends onto Exa's anonymous tier. Keyless availability is a
        separate, last-resort signal (:meth:`is_keyless_available`).
        """
        from agent.web_search_provider import get_provider_env

        return bool(get_provider_env("EXA_API_KEY"))

    def is_keyless_available(self) -> bool:
        """Exa serves anonymous free-tier calls via its public MCP endpoint.

        False when the user forced ``web.provider_tier.exa: paid`` — an
        explicit paid selection must never silently resolve keyless.
        """
        from plugins.web.keyless_mcp import keyless_enabled, provider_tier

        return keyless_enabled() and provider_tier("exa") != "paid"

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute an Exa search.

        Returns ``{"success": True, "data": {"web": [{...}, ...]}}`` on
        success, ``{"success": False, "error": str}`` on failure (incl.
        missing API key and SDK install errors).
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"success": False, "error": "Interrupted"}

            from agent.web_search_provider import get_provider_env

            from plugins.web.keyless_mcp import search_with_failover, use_keyless

            if use_keyless("exa", get_provider_env("EXA_API_KEY")):
                # Keyless free tier — public MCP endpoint, no SDK needed.
                logger.info(
                    "Exa keyless search: '%s' (limit=%d)", query, limit
                )
                return search_with_failover("exa", query, limit)

            logger.info("Exa search: '%s' (limit=%d)", query, limit)
            response = _get_exa_client().search(query, num_results=limit, contents={"highlights": True})
            return search_ok([
                web_hit(r.url or "", r.title or "", " ".join(r.highlights or []), i + 1)
                for i, r in enumerate(response.results or [])
            ])

        return run_search("Exa", logger, _body, sdk=True)

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract content from one or more URLs via Exa.

        Returns a list of result dicts shaped for the legacy LLM
        post-processing pipeline. On per-URL or whole-batch failure,
        results carry an ``error`` field rather than raising.
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return [
                    {"url": u, "error": "Interrupted", "title": ""} for u in urls
                ]

            from agent.web_search_provider import get_provider_env

            from plugins.web.keyless_mcp import extract_with_failover, use_keyless

            if use_keyless("exa", get_provider_env("EXA_API_KEY")):
                # Keyless free tier — public MCP endpoint, no SDK needed.
                logger.info("Exa keyless extract: %d URL(s)", len(urls))
                return extract_with_failover("exa", list(urls))

            logger.info("Exa extract: %d URL(s)", len(urls))
            response = _get_exa_client().get_contents(urls, text=True)
            return [document(r.url or "", r.title or "", r.text or "") for r in response.results or []]

        return run_extract("Exa", logger, urls, _body, sdk=True)

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Exa · Free (keyless)",
            "badge": "free · no key",
            "tag": (
                "Semantic + neural web search with content extraction on "
                "Exa's anonymous free tier. Rate-limited under burst load."
            ),
            "env_vars": [],
            "web_tier": "free",
            "variants": [
                {
                    "name": "Exa · Paid (API key)",
                    "badge": "paid",
                    "tag": (
                        "Semantic + neural web search with content extraction "
                        "via the Exa SDK. Unthrottled, guaranteed service."
                    ),
                    "env_vars": [
                        {
                            "key": "EXA_API_KEY",
                            "prompt": "Exa API key",
                            "url": "https://exa.ai",
                        },
                    ],
                    "web_tier": "paid",
                },
            ],
        }
