"""SearXNG search via a user-hosted instance (``/search?format=json``).

Search-only — SearXNG aggregates upstream engines but does not fetch URLs.
Env: ``SEARXNG_URL=http://localhost:8080``. Config: ``web.searxng.auth_provider`` names a
provider whose credential is the bearer, for an instance behind an authenticating gateway.
"""

from __future__ import annotations

import logging
from typing import Any, Dict
from urllib.parse import urlsplit

from plugins.web._common import BaseWebSearchProvider, http_get_json, provider_env, search_fail, search_ok, setup_schema, titled_rows

logger = logging.getLogger(__name__)


def _auth_header(base_url: str) -> dict[str, str]:
    """``Authorization`` with the credential of ``web.searxng.auth_provider``; empty when unset.

    ``select()`` first renews an OAuth token the pool sees expiring. A 401 is not retried with a
    forced refresh: a gateway that keeps refusing the token would force one per search, and a
    refresh that fails takes the credential out of rotation for the provider's own requests.
    Raises ValueError, with the message to show, when the credential cannot be sent.
    """
    # Deferred, as in the sibling backends: discovering this plugin needs neither import.
    from hermes_cli.config import cfg_get, load_config_readonly

    provider = str(cfg_get(load_config_readonly(), "web", "searxng", "auth_provider") or "").strip()
    if not provider:
        return {}
    target = urlsplit(base_url)
    if target.scheme != "https" and target.hostname not in ("127.0.0.1", "::1"):
        raise ValueError(
            f"web.searxng.auth_provider sends the {provider} credential, so SEARXNG_URL must be https "
            "(plain http only to 127.0.0.1 or ::1)"
        )
    from agent.credential_pool import load_pool

    entry = load_pool(provider).select()
    token = entry.runtime_api_key if entry is not None else ""
    if not token:
        raise ValueError(
            f"web.searxng.auth_provider is '{provider}', which has no usable credential: it is not signed in "
            f"(`hermes auth add {provider}`), or its credential is cooling down after an error"
        )
    return {"Authorization": f"Bearer {token}"}


class SearXNGWebSearchProvider(BaseWebSearchProvider):
    """Search via a user-hosted SearXNG instance."""

    NAME = "searxng"
    DISPLAY_NAME = "SearXNG"
    KEY_ENV = "SEARXNG_URL"

    def search(self, query: str, limit: int = 5) -> dict[str, Any]:
        base_url = provider_env("SEARXNG_URL").rstrip("/")
        if not base_url:
            return search_fail("SEARXNG_URL is not set")
        try:
            auth = _auth_header(base_url)
        except ValueError as exc:
            return search_fail(str(exc))
        data, failure = http_get_json(
            "SearXNG", f"{base_url}/search", params={"q": query, "format": "json", "pageno": 1},
            headers={"Accept": "application/json", **auth}, timeout=15, logger=logger, reach_target=f"SearXNG at {base_url}",
        )
        if failure is not None:
            return failure
        raw_results = data.get("results", [])
        # SearXNG may return a score field; sort descending and cap to limit.
        sorted_results = sorted(raw_results, key=lambda r: float(r.get("score", 0)), reverse=True)[:limit]
        web_results = titled_rows(sorted_results, "content")
        logger.info("SearXNG search '%s': %d results (from %d raw, limit %d)", query, len(web_results), len(raw_results), limit)
        return search_ok(web_results)

    def get_setup_schema(self) -> dict[str, Any]:
        return setup_schema(
            "SearXNG", "free · self-hosted", "Free, privacy-respecting metasearch. Point SEARXNG_URL at your instance.",
            "SEARXNG_URL", "SearXNG instance URL (e.g. http://localhost:8080)", "https://searx.space/",
        )
