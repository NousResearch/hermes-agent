"""Exa web search + content extraction via the ``exa-py`` SDK (lazy-installed).

Env: ``EXA_API_KEY`` (https://exa.ai). Both methods are sync — Exa's SDK is
sync-only; the dispatcher threads extract when the caller is async.

Config: ``web.exa_search_type`` picks the search algorithm (``auto``, ``lite``,
``fast``, ``instant``, ``deep``, ``deep-lite``, ``deep-reasoning``, ...). Unset
means the vendor's own default (``auto``) and no argument is sent at all.
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


def _resolve_search_type() -> str:
    """``web.exa_search_type``, lower-cased and stripped — or ``""`` when unset.

    Deliberately not whitelisted: Exa ships search types faster than Hermes ships releases,
    and the SDK types the argument as ``Union[SearchType, str]``, so an unknown value reaches
    the API and its own error names it instead of being silently swapped for a default. A
    config read must never fail a paid search, so any exception here reads as "unset".
    """
    try:
        from hermes_cli.config import load_config
        raw = (load_config().get("web") or {}).get("exa_search_type")
    except Exception as exc:
        logger.debug("Could not read web.exa_search_type: %s", exc, exc_info=True)
        return ""
    return str(raw or "").strip().lower()


def _get_exa_client() -> Any:
    def _factory(api_key: str) -> Any:
        from exa_py import Exa  # deliberately lazy
        client = Exa(api_key=api_key)
        client.headers["x-exa-integration"] = "hermes-agent"
        return client

    return cached_sdk_client("_exa_client", "EXA_API_KEY", _MISSING_KEY, "exa", _factory)


class ExaWebSearchProvider(BaseWebSearchProvider):
    """Exa search + extract provider."""

    NAME = "exa"
    DISPLAY_NAME = "Exa"
    KEY_ENV = "EXA_API_KEY"
    EXTRACT = True
    KEYLESS = True

    def search(self, query: str, limit: int = 5) -> dict[str, Any]:
        def _body() -> dict[str, Any]:
            if use_keyless("exa", provider_env("EXA_API_KEY")):
                return keyless_search("Exa", "exa", query, limit, logger)
            search_type = _resolve_search_type()
            # This line is the audit trail that a given mode actually ran; grep it before
            # believing any cost or latency claim. "auto" is the vendor default.
            logger.info("Exa search: '%s' (limit=%d, type=%s)", query, limit, search_type or "auto")
            # Unset sends no ``type`` at all — byte-identical to the request made before this
            # setting existed, so config absence can never change a user's results.
            type_kwarg = {"type": search_type} if search_type else {}
            response = _get_exa_client().search(
                query, num_results=limit, contents={"highlights": True}, **type_kwarg
            )
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
            response = _get_exa_client().get_contents(urls, text=True)
            return [document(r.url or "", r.title or "", r.text or "") for r in response.results or []]

        return run_extract("Exa", logger, urls, _body, sdk=True)

    def get_setup_schema(self) -> dict[str, Any]:
        return keyless_variant_schema(
            "Exa", "EXA_API_KEY", "https://exa.ai",
            free_tag="Semantic + neural web search with content extraction on Exa's anonymous free tier. Rate-limited under burst load.",
            paid_tag="Semantic + neural web search with content extraction via the Exa SDK. Unthrottled, guaranteed service.",
        )
