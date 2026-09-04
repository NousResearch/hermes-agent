"""Exa web search + content extraction via the ``exa-py`` SDK (lazy-installed).

Env: ``EXA_API_KEY`` (https://exa.ai). Both methods are sync — Exa's SDK is
sync-only; the dispatcher threads extract when the caller is async.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from plugins.web._common import (
    BaseWebSearchProvider, cached_sdk_client, document, keyless_extract, keyless_search, keyless_variant_schema,
    page_error, provider_env, run_extract, run_search, search_ok, use_keyless, web_hit,
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

    NAME = "exa"
    DISPLAY_NAME = "Exa"
    KEY_ENV = "EXA_API_KEY"
    EXTRACT = True
    KEYLESS = True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        def _body() -> Dict[str, Any]:
            if use_keyless("exa", provider_env("EXA_API_KEY")):
                return keyless_search("Exa", "exa", query, limit, logger)
            logger.info("Exa search: '%s' (limit=%d)", query, limit)
            response = _get_exa_client().search(query, num_results=limit, contents={"highlights": True})
            return search_ok([
                web_hit(r.url or "", r.title or "", " ".join(r.highlights or []), i + 1)
                for i, r in enumerate(response.results or [])
            ])

        return run_search("Exa", logger, _body, sdk=True)

    def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        def _body() -> List[Dict[str, Any]]:
            if use_keyless("exa", provider_env("EXA_API_KEY")):
                return keyless_extract("Exa", "exa", urls, logger)
            logger.info("Exa extract: %d URL(s)", len(urls))
            response = _get_exa_client().get_contents(urls, text=True)
            # Exa reports per-URL failures in response.statuses (e.g.
            # CRAWL_NOT_FOUND for a dead link) — NOT in response.results.
            # Without this, a single 404 made the whole call return [] and
            # callers saw a silent empty result instead of an honest per-URL
            # error. Surface each failed status as an error doc so the
            # 1-doc-per-URL contract holds regardless of how many results or
            # statuses the API returns.
            by_url = {
                r.url: r
                for r in response.results or []
                if getattr(r, "url", None)
            }
            status_by_id = {
                s.id: s
                for s in response.statuses or []
                if getattr(s, "id", None)
            }
            docs: List[Dict[str, Any]] = []
            for url in urls:
                result = by_url.get(url)
                if result is not None:
                    docs.append(document(result.url or "", result.title or "", result.text or ""))
                    continue
                status = status_by_id.get(url)
                if status is None:
                    # Status id may not match the raw URL (redirect or
                    # normalization) — fall back to a generic per-URL error
                    # rather than dropping the URL silently.
                    docs.append(page_error(url, f"Exa could not fetch {url}"))
                    continue
                error = (status.status or "").strip()
                docs.append(
                    page_error(
                        url,
                        f"Exa fetch failed ({error})" if error else f"Exa could not fetch {url}",
                    )
                )
            return docs

        return run_extract("Exa", logger, urls, _body, sdk=True)

    def get_setup_schema(self) -> Dict[str, Any]:
        return keyless_variant_schema(
            "Exa", "EXA_API_KEY", "https://exa.ai",
            free_tag="Semantic + neural web search with content extraction on Exa's anonymous free tier. Rate-limited under burst load.",
            paid_tag="Semantic + neural web search with content extraction via the Exa SDK. Unthrottled, guaranteed service.",
        )


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import os  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'WebSearchProvider': ('agent.web_search_provider', 'WebSearchProvider'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
