"""SearXNG search via a user-hosted instance (``/search?format=html``).

SearXNG answers HTTP 403 on ``format=json`` whenever ``server.public_instance``
is false (the self-hosting default) — an intentional restriction that keeps
bots out. This provider therefore asks for ``format=html``, the same rendered
page a browser gets, and parses the result list. That path works on every
instance, private or public, with no ``formats`` config edit and no
``link_token`` handshake.

Search-only — SearXNG aggregates upstream engines but does not fetch URLs.
Env: ``SEARXNG_URL=http://localhost:8080``.
"""

from __future__ import annotations

import html
import logging
import re
from typing import Any, Dict, List

import httpx

from plugins.web._common import (
    BaseWebSearchProvider,
    provider_env,
    search_fail,
    search_ok,
    setup_schema,
    title_hit,
)

logger = logging.getLogger(__name__)

# SearXNG renders each hit as <article class="result ...">; the ``result``
# prefix covers every type variant (result-default, result-images, ...).
_ARTICLE_RE = re.compile(r'<article[^>]*class="result[^"]*"[^>]*>(.*?)</article>', re.DOTALL)
_TITLE_URL_RE = re.compile(r'<h3[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.+?)</a>', re.DOTALL)
# SearXNG renders ``content empty_element`` for a hit whose engine supplied no
# description; its body is a UI placeholder ("This site did not provide any
# description."), so it must not be reported as snippet text.
_SNIPPET_RE = re.compile(r'<p[^>]+class="content(?! empty_element)[^"]*"[^>]*>(.+?)</p>', re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")

_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Hermes SearXNG Provider) AppleWebKit/537.36 Chrome/126.0",
    "Accept": "text/html, application/xhtml+xml, */*",
    "Accept-Language": "de-DE,de;q=0.9,en;q=0.8",
}


def _plain_text(fragment: str) -> str:
    """Strip inline markup, decode HTML entities, collapse runs of whitespace.

    SearXNG renders title and snippet with Jinja's ``|safe``, so engines that
    return entity-encoded text ("&quot;", "&#x27;") reach us verbatim. Tags are
    removed before unescaping so an engine-supplied ``&lt;b&gt;`` survives as
    literal text instead of being stripped as markup.
    """
    text = html.unescape(_TAG_RE.sub("", fragment))
    return re.sub(r"\s+", " ", text).strip()


def _parse_html_results(body: str, limit: int) -> List[Dict[str, Any]]:
    """Parse a SearXNG HTML result page into title-first rows.

    HTML output is already relevance-ordered, so nothing is re-sorted;
    ``limit`` is a hard cap.
    """
    rows: List[Dict[str, Any]] = []
    for block in _ARTICLE_RE.findall(body):
        match = _TITLE_URL_RE.search(block)
        if match is None:
            continue  # container without a parsable title link
        url = match.group(1).strip()
        title = _plain_text(match.group(2))
        if not url or not title:
            continue
        snippet_match = _SNIPPET_RE.search(block)
        snippet = _plain_text(snippet_match.group(1)) if snippet_match else ""
        rows.append(title_hit(title, url, snippet, len(rows) + 1))
        if len(rows) >= limit:
            break
    return rows


class SearXNGWebSearchProvider(BaseWebSearchProvider):
    """Search via a user-hosted SearXNG instance."""

    NAME = "searxng"
    DISPLAY_NAME = "SearXNG"
    KEY_ENV = "SEARXNG_URL"

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        base_url = provider_env("SEARXNG_URL").rstrip("/")
        if not base_url:
            return search_fail("SEARXNG_URL is not set")

        try:
            resp = httpx.get(
                f"{base_url}/search",
                params={"q": query, "format": "html", "pageno": 1},
                headers=_BROWSER_HEADERS,
                timeout=15,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning("SearXNG HTTP error: %s", exc)
            return search_fail(f"SearXNG returned HTTP {exc.response.status_code}")
        except httpx.RequestError as exc:
            logger.warning("SearXNG request error: %s", exc)
            return search_fail(f"Could not reach SearXNG at {base_url}: {exc}")

        web_results = _parse_html_results(resp.text, limit)
        logger.info("SearXNG search '%s': %d results (html, limit %d)", query, len(web_results), limit)
        return search_ok(web_results)

    def get_setup_schema(self) -> Dict[str, Any]:
        return setup_schema(
            "SearXNG", "free · self-hosted", "Free, privacy-respecting metasearch. Point SEARXNG_URL at your instance.",
            "SEARXNG_URL", "SearXNG instance URL (e.g. http://localhost:8080)", "https://searx.space/",
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
