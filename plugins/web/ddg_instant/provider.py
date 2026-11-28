"""DuckDuckGo Instant Answer API — free, no-key, no-deps web search (HER-52).

Wraps https://api.duckduckgo.com/ — the public Instant Answer endpoint
that does not require an API key and has no documented rate limit. Distinct
from ``plugins/web/ddgs`` (which HTML-scrapes via the optional ``ddgs``
package); this provider uses only the Python stdlib so it works on a fresh
install without ``pip install ddgs``.

Result shapes follow the legacy contract documented in
``agent/web_search_provider``::

    success: {"success": True, "data": {"web": [{"title", "url", "description", "position"}, ...]}}
    failure: {"success": False, "error": str}

Coverage note
-------------
DDG Instant Answer is a knowledge-base endpoint, not a full search index.
It returns rich data for **factual / entity** queries (``linux``,
``python``, ``tokyo``) and sparse / empty results for **how-to / tutorial**
queries (``python async tutorial``). When the response has neither an
abstract nor any related topics we surface an explicit, instructive
``search_fail`` so callers can fall back to a full-text backend instead of
silently returning zero hits.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from plugins.web._common import (
    BaseWebSearchProvider,
    search_fail,
    search_ok,
    setup_schema,
    title_hit,
)

logger = logging.getLogger(__name__)

_DDG_ENDPOINT = "https://api.duckduckgo.com/"
_REQUEST_TIMEOUT_SECS = 15
_USER_AGENT = "HermesAgent-DDgInstant/1.0 (+https://github.com/NousResearch/hermes-agent)"


def _http_get_json(url: str, params: Dict[str, str], timeout: int) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Stdlib-only GET + JSON parse. Returns ``(data, None)`` on success,
    ``(None, error_message)`` on failure. Mirrors ``plugins.web._common.http_get_json``
    but uses ``urllib.request`` so this plugin adds no dependencies.
    """
    full = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(full, headers={"User-Agent": _USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        detail = (exc.read().decode("utf-8", errors="replace").strip() if hasattr(exc, "read") else "") or f"HTTP {exc.code}"
        return None, f"DuckDuckGo Instant Answer HTTP error: {detail}"
    except urllib.error.URLError as exc:
        return None, f"Could not reach DuckDuckGo Instant Answer: {exc.reason}"
    except (TimeoutError, OSError) as exc:
        return None, f"DuckDuckGo Instant Answer network error: {exc}"
    try:
        return json.loads(body.decode("utf-8")), None
    except (ValueError, UnicodeDecodeError) as exc:
        return None, f"DuckDuckGo Instant Answer returned non-JSON body: {exc}"


def _coerce_topic(topic: Any) -> Optional[Dict[str, str]]:
    """Normalize a RelatedTopics entry to ``{title, url, description}``.

    DDG nests topics under ``Topics`` (a ``RelatedTopic`` whose ``Topics``
    is itself a list). We flatten one level so a top-level category like
    ``"Linux"`` followed by sub-entries both surface as hits.
    """
    if not isinstance(topic, dict):
        return None
    if topic.get("TopLevelLink") or topic.get("Name"):
        return None  # skip nav-style categories without a destination
    text = (topic.get("Text") or "").strip()
    url = (topic.get("FirstURL") or "").strip()
    if not text or not url:
        return None
    # Convention: "Title — snippet". Split on the first em-dash if present
    # so the rendered title is concise instead of the whole blurb.
    title = text
    for sep in (" — ", " – ", " - "):
        if sep in text:
            title = text.split(sep, 1)[0].strip()
            break
    return {"title": title, "url": url, "description": text}


def _flatten_topics(raw: List[Any]) -> List[Dict[str, str]]:
    """Walk ``RelatedTopics`` and ``Topics`` (nested one level) and return
    leaf entries only. Defensive against DDG changing its shape."""
    out: List[Dict[str, str]] = []
    for entry in raw:
        coerced = _coerce_topic(entry)
        if coerced is not None:
            out.append(coerced)
            continue
        if isinstance(entry, dict) and isinstance(entry.get("Topics"), list):
            for nested in entry["Topics"]:
                coerced_nested = _coerce_topic(nested)
                if coerced_nested is not None:
                    out.append(coerced_nested)
    return out


class DDGInstantWebSearchProvider(BaseWebSearchProvider):
    """DuckDuckGo Instant Answer API provider (no key, no extra deps)."""

    NAME = "ddg-instant"
    DISPLAY_NAME = "DuckDuckGo Instant Answer (free)"
    KEY_ENV = ""  # no key required; presence is determined by endpoint reachability

    # --- availability --------------------------------------------------------
    def is_available(self) -> bool:
        """The Instant Answer API needs no key and no extra package — only
        network access. We avoid a probe ping at registration time because
        this is called on every ``hermes tools`` paint; the first real
        ``search()`` call is where reachability actually matters.
        """
        return True

    # --- search --------------------------------------------------------------
    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        if not (query or "").strip():
            return search_fail("Empty query")
        params = {
            "q": query,
            "format": "json",
            "no_html": "1",
            "skip_disambig": "1",
            "t": "hermes-agent",
        }
        data, failure = _http_get_json(_DDG_ENDPOINT, params, _REQUEST_TIMEOUT_SECS)
        if failure is not None:
            logger.warning("DDG Instant Answer '%s': %s", query, failure)
            return search_fail(failure)

        abstract = (data.get("AbstractText") or data.get("Abstract") or "").strip()
        abstract_url = (data.get("AbstractURL") or "").strip()
        abstract_source = (data.get("AbstractSource") or "").strip()
        hits: List[Dict[str, Any]] = []

        if abstract and abstract_url:
            hits.append(
                title_hit(
                    title=abstract_source or abstract[:80],
                    url=abstract_url,
                    description=abstract,
                    position=1,
                )
            )

        for i, topic in enumerate(_flatten_topics(data.get("RelatedTopics") or []), start=len(hits) + 1):
            if len(hits) >= max(1, int(limit)):
                break
            hits.append(
                title_hit(
                    title=topic["title"],
                    url=topic["url"],
                    description=topic["description"],
                    position=i,
                )
            )

        if not hits:
            # Coverage gap, not a transport error — surface an instructive failure
            # so the caller can fall back to a full-text backend instead of
            # silently returning [].
            logger.info(
                "DDG Instant Answer '%s': no Abstract and no RelatedTopics "
                "(DDG Instant Answer is a knowledge-base endpoint, sparse for how-to/tutorial queries)",
                query,
            )
            return search_fail(
                "DuckDuckGo Instant Answer returned no results — this endpoint is "
                "best for factual/entity queries and is sparse for how-to/tutorial queries. "
                "Try a full-text backend (ddgs, brave, tavily, exa) for that query."
            )

        logger.info(
            "DDG Instant Answer '%s': %d results (limit %d, abstract=%s)",
            query, len(hits), limit, bool(abstract),
        )
        return search_ok(hits)

    # --- setup schema --------------------------------------------------------
    def get_setup_schema(self) -> Dict[str, Any]:
        return setup_schema(
            "DuckDuckGo Instant Answer (free)",
            "free · no key · no extra deps",
            "Free, no-auth Instant Answer API (https://api.duckduckgo.com/). "
            "Knowledge-base endpoint — best for factual/entity queries, sparse for how-to/tutorials.",
        )