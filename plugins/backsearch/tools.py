"""BackSearch tools — search and fetch the web as it was on a given date.

BackSearch (by General Reasoning, https://www.gr.inc) is a point-in-time
web archive with two endpoints:

- ``POST /search`` — hybrid search over a frozen news corpus. Every request
  carries an ``as_of`` date; only documents *crawled* on or before that date
  are returned. The corpus never moves, so the same query with the same
  ``as_of`` returns the same results forever.
- ``POST /fetch`` — point-in-time page fetch. Returns the extracted article
  text from the latest capture on or before the cutoff, not today's bytes.

Both authenticate with an OpenReward API key (``or_...``) in the
``x-api-key`` header — the same credential used for openreward.ai; there is
no separate BackSearch key. Base URL: ``https://search.openreward.ai``
(override with ``OPENREWARD_SEARCH_URL`` for testing/self-routing).

Important semantics baked into the tool descriptions:

- ``as_of`` gates on **crawl_date**, not the article's self-reported publish
  date. A page first archived after the cutoff will not be returned even if
  it claims an earlier publish date. This is what guarantees no post-cutoff
  leakage into a backtest.
- The current preview archive covers **news domains, December 2025 to
  July 2026**. An ``as_of`` outside the archive window returns an empty hit
  list rather than an error, so "no results" on a far-past/future date
  usually means the date is off the edge of the archive.
- Billing is per successful request; a fetch with no capture on or before
  the cutoff returns 404 and costs nothing. An exhausted OpenReward balance
  returns 402.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from tools.registry import tool_error, tool_result

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://search.openreward.ai"

# Preview-archive window (see module docstring). Used only to append a
# helpful hint on empty results — never to reject a request, since the
# archive is expected to widen over time.
_PREVIEW_WINDOW_HINT = (
    "The current BackSearch preview archive covers news domains from "
    "December 2025 to July 2026. An as_of outside that window returns "
    "no hits rather than an error."
)

_AS_OF_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Cap the article text returned by backfetch so a long feature piece can't
# blow out the model context. The full text stays server-side; the model can
# re-fetch with a summarize prompt if it needs the gist of a long article.
_FETCH_TEXT_CAP = 15000


def _get_env(name: str) -> str:
    """Config-aware env lookup (os.environ, then ~/.hermes/.env)."""
    try:
        from hermes_cli.config import get_env_value

        val = get_env_value(name)
    except Exception:
        import os

        val = os.getenv(name)
    return (val or "").strip()


def check_backsearch_available() -> bool:
    """Tools are only exposed when an OpenReward API key is configured."""
    return bool(_get_env("OPENREWARD_API_KEY"))


def _base_url() -> str:
    return (_get_env("OPENREWARD_SEARCH_URL") or DEFAULT_BASE_URL).rstrip("/")


def _request(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST to the BackSearch API and return the parsed JSON response.

    Raises ``ValueError`` with a user-actionable message on missing key,
    payment/auth failures, and no-capture 404s so handlers can surface a
    typed error the model can recover from.
    """
    import httpx

    api_key = _get_env("OPENREWARD_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENREWARD_API_KEY is not set. BackSearch uses your OpenReward "
            "key (or_...) — get one at https://openreward.ai/"
        )

    url = f"{_base_url()}/{endpoint.lstrip('/')}"
    body = {k: v for k, v in payload.items() if v is not None}
    logger.info("BackSearch %s request (as_of=%s)", endpoint, body.get("as_of"))

    response = httpx.post(
        url,
        headers={"x-api-key": api_key, "Content-Type": "application/json"},
        json=body,
        timeout=60,
    )
    if response.status_code == 402:
        raise ValueError(
            "OpenReward balance exhausted (HTTP 402). Top up your prepaid "
            "balance at https://openreward.ai/ to keep using BackSearch."
        )
    if response.status_code in (401, 403):
        raise ValueError(
            "BackSearch rejected the API key (HTTP "
            f"{response.status_code}). Check OPENREWARD_API_KEY."
        )
    if response.status_code == 404 and endpoint.lstrip("/") == "fetch":
        raise ValueError(
            "No capture of this URL exists on or before the as_of date. "
            "Try a later as_of, or a different URL from the search hits."
        )
    response.raise_for_status()
    return response.json()


def _validate_as_of(raw: Any) -> str:
    as_of = str(raw or "").strip()
    if not as_of:
        raise ValueError("as_of is required (YYYY-MM-DD), e.g. '2026-01-15'.")
    if not _AS_OF_RE.match(as_of):
        raise ValueError(
            f"as_of must be an ISO date (YYYY-MM-DD), got: {as_of!r}"
        )
    return as_of


def _as_domain_list(raw: Any) -> Optional[List[str]]:
    if raw is None:
        return None
    if isinstance(raw, str):
        items = [p.strip() for p in raw.split(",")]
    elif isinstance(raw, list):
        items = [str(p).strip() for p in raw]
    else:
        return None
    items = [p for p in items if p]
    return items or None


def _coerce_k(raw: Any, *, default: int = 5, maximum: int = 20) -> int:
    try:
        value = int(raw)
    except Exception:
        value = default
    return max(1, min(maximum, value))


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def handle_backsearch(args: dict, **kw) -> str:
    """Search the frozen archive as of a date."""
    try:
        from tools.interrupt import is_interrupted

        if is_interrupted():
            return tool_error("Interrupted")
    except Exception:
        pass

    try:
        as_of = _validate_as_of(args.get("as_of"))
    except ValueError as exc:
        return tool_error(str(exc))

    query = str(args.get("query") or "").strip()
    if not query:
        return tool_error("query is required.")

    allowed = _as_domain_list(args.get("allowed_domains"))
    blocked = _as_domain_list(args.get("blocked_domains"))
    if allowed and blocked:
        return tool_error(
            "Pass allowed_domains OR blocked_domains, never both."
        )

    try:
        raw = _request(
            "search",
            {
                "query": query,
                "as_of": as_of,
                "k": _coerce_k(args.get("k")),
                "allowed_domains": allowed,
                "blocked_domains": blocked,
            },
        )
    except ValueError as exc:
        return tool_error(str(exc))
    except Exception as exc:  # noqa: BLE001 — httpx errors included
        logger.warning("BackSearch search error: %s", exc)
        return tool_error(f"BackSearch search failed: {exc}")

    hits = raw.get("hits") or []
    results: Dict[str, Any] = {
        "success": True,
        "as_of": as_of,
        "hits": [
            {
                "url": h.get("url", ""),
                "title": h.get("title", ""),
                "snippet": h.get("snippet", ""),
                "host": h.get("host", ""),
                "crawl_date": h.get("crawl_date", ""),
                "publish_date": h.get("publish_date", ""),
            }
            for h in hits
        ],
    }
    if not hits:
        results["note"] = _PREVIEW_WINDOW_HINT
    return tool_result(results)


def handle_backfetch(args: dict, **kw) -> str:
    """Fetch a page as it was archived on or before a date."""
    try:
        from tools.interrupt import is_interrupted

        if is_interrupted():
            return tool_error("Interrupted")
    except Exception:
        pass

    try:
        as_of = _validate_as_of(args.get("as_of"))
    except ValueError as exc:
        return tool_error(str(exc))

    url = str(args.get("url") or "").strip()
    if not url:
        return tool_error("url is required.")

    prompt = str(args.get("prompt") or "").strip() or None

    try:
        raw = _request(
            "fetch",
            {
                "url": url,
                "as_of": as_of,
                "prompt": prompt,
                "summarize": bool(prompt) or None,
            },
        )
    except ValueError as exc:
        return tool_error(str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.warning("BackSearch fetch error: %s", exc)
        return tool_error(f"BackSearch fetch failed: {exc}")

    text = str(raw.get("text") or "")
    truncated = len(text) > _FETCH_TEXT_CAP
    result: Dict[str, Any] = {
        "success": True,
        "as_of": as_of,
        "url": url,
        "text": text[:_FETCH_TEXT_CAP],
    }
    for key in ("title", "crawl_date", "publish_date", "host"):
        if raw.get(key):
            result[key] = raw[key]
    if truncated:
        result["truncated"] = True
        result["note"] = (
            f"Article text truncated to {_FETCH_TEXT_CAP} chars. Re-fetch "
            "with a 'prompt' describing what to extract for a focused summary."
        )
    if not text:
        result["success"] = False
        result["error"] = "No text returned for this capture."
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


BACKSEARCH_SCHEMA = {
    "name": "backsearch",
    "description": (
        "Search the web as it was on a particular date (BackSearch by "
        "General Reasoning). Runs against a FROZEN archive: only documents "
        "crawled on or before as_of are returned, and the same query + "
        "as_of always returns the same results. Use for forecasting "
        "backtests, point-in-time financial research, and any task where "
        "evidence after a cutoff date must not leak in. The cutoff gates "
        "on crawl_date, not the article's self-reported publish date. "
        "Current preview archive: news domains, December 2025 – July 2026; "
        "an as_of outside that window returns zero hits (not an error)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
            "as_of": {
                "type": "string",
                "description": (
                    "Point-in-time cutoff date, ISO format YYYY-MM-DD "
                    "(e.g. '2026-01-15'). Only pages crawled on or before "
                    "this date are searched."
                ),
            },
            "k": {
                "type": "integer",
                "description": "Number of hits to return (1-20, default 5).",
            },
            "allowed_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Restrict the search to these hosts. Mutually exclusive "
                    "with blocked_domains."
                ),
            },
            "blocked_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Exclude these hosts from the search. Mutually exclusive "
                    "with allowed_domains."
                ),
            },
        },
        "required": ["query", "as_of"],
    },
}

BACKFETCH_SCHEMA = {
    "name": "backfetch",
    "description": (
        "Fetch a web page as it was archived on or before a date "
        "(BackSearch by General Reasoning). Returns the extracted article "
        "text from the latest capture on or before as_of — NOT today's "
        "version of the page. Use it to read pages returned by the "
        "backsearch tool. If no capture exists on or before the cutoff, "
        "returns a soft error (try a later as_of or another URL). Pass a "
        "'prompt' to get a focused summary of a long article instead of "
        "the full text."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch from the archive.",
            },
            "as_of": {
                "type": "string",
                "description": (
                    "Point-in-time cutoff date, ISO format YYYY-MM-DD. The "
                    "latest capture on or before this date is returned."
                ),
            },
            "prompt": {
                "type": "string",
                "description": (
                    "Optional: what to extract from the page. When set, the "
                    "archive summarizes the capture against this prompt "
                    "instead of returning the full text."
                ),
            },
        },
        "required": ["url", "as_of"],
    },
}
