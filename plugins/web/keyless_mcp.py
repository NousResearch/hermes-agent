"""Keyless web search/extract via public MCP endpoints.

Exa and Parallel both operate public, anonymous MCP endpoints with a free
tier (the same endpoints the opencode CLI ships as its default search
path):

- Exa:      https://mcp.exa.ai/mcp           (tools: web_search_exa, web_fetch_exa)
- Parallel: https://search.parallel.ai/mcp   (tools: web_search, web_fetch)

This module implements a minimal JSON-RPC ``tools/call`` client for those
two endpoints so a fresh Hermes install with **zero web credentials** still
gets working ``web_search`` / ``web_extract`` tools. The keyless tier is
resolved strictly LAST — after every keyed backend, the managed tool
gateway, ddgs, and custom plugin providers — so it never pre-empts a
deliberate setup (see ``tools.web_tools._get_backend`` and the registry's
``_KEYLESS_PREFERENCE`` walk).

Privacy: requests carry no user identifiers. Parallel's free tier asks for
a ``session_id`` used for rate limiting; we send a random per-process UUID
(rotates every restart, never persisted). Their optional ``model_name``
analytics field is deliberately omitted.

Disable the whole tier with ``web.keyless_fallback: false`` in config.yaml.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

EXA_MCP_URL = "https://mcp.exa.ai/mcp"
PARALLEL_MCP_URL = "https://search.parallel.ai/mcp"

# Free-tier rate-limit correlation id for Parallel — random per process,
# never persisted, not derived from any user/machine identifier.
_SESSION_ID = uuid.uuid4().hex

_TIMEOUT_SECONDS = 30


class KeylessMCPError(RuntimeError):
    """A keyless MCP call failed (transport, rate limit, or tool error)."""


_RATE_LIMIT_MARKERS = (
    "rate limit",
    "rate-limit",
    "ratelimit",
    "too many requests",
    "429",
    "quota exceeded",
    "slow down",
)


def _is_rate_limitish(message: str) -> bool:
    """Heuristic: does an error message look like free-tier throttling?"""
    lowered = (message or "").lower()
    return any(marker in lowered for marker in _RATE_LIMIT_MARKERS)


def keyless_enabled() -> bool:
    """Return True when the keyless fallback tier is enabled.

    Delegates to :func:`agent.web_search_registry._keyless_tier_enabled` so
    the config chokepoint (``web.keyless_fallback``, default on) lives in
    one place alongside the rest of backend resolution.
    """
    try:
        from agent.web_search_registry import _keyless_tier_enabled

        return _keyless_tier_enabled()
    except Exception as exc:  # noqa: BLE001 — resolver optional in stripped envs
        logger.debug("keyless_enabled(): registry helper unavailable: %s", exc)
        return True


def provider_tier(name: str) -> str:
    """Return the user-selected tier for *name*: ``free``, ``paid``, or ``auto``.

    Reads ``web.provider_tier.<name>`` from config.yaml (set by the
    ``hermes tools`` picker's Free/Paid rows). ``free`` forces the keyless
    public endpoint even when the vendor API key is present; ``paid``
    forces the keyed SDK path (missing key surfaces the standard
    "X_API_KEY not set" error instead of silently downgrading to the free
    tier). Anything else — including unset — is ``auto``: key present →
    keyed, otherwise keyless when the tier is enabled.
    """
    try:
        from hermes_cli.config import load_config

        web_cfg = load_config().get("web") or {}
        tiers = web_cfg.get("provider_tier") or {}
        value = str(tiers.get(name, "") or "").lower().strip()
        return value if value in ("free", "paid") else "auto"
    except Exception as exc:  # noqa: BLE001 — config layer optional
        logger.debug("provider_tier(%r) config read failed: %s", name, exc)
        return "auto"


def use_keyless(name: str, api_key: str) -> bool:
    """Decide whether provider *name* should route via the keyless endpoint.

    Single chokepoint shared by the Exa/Parallel search + extract paths so
    tier semantics can't drift between capabilities:

    - tier ``free``  → keyless, even when *api_key* is set
    - tier ``paid``  → keyed, even when *api_key* is missing (the keyed
      path then raises its usual missing-key error)
    - tier ``auto``  → keyed when *api_key* is set; otherwise keyless when
      ``web.keyless_fallback`` is enabled
    """
    tier = provider_tier(name)
    if tier == "free":
        return True
    if tier == "paid":
        return False
    return not api_key and keyless_enabled()


def _parse_mcp_body(body: str) -> str:
    """Extract the first text content item from an MCP tools/call response.

    Handles both plain-JSON bodies and SSE (``data: {...}`` lines) — the
    Exa endpoint answers as an event stream, Parallel as direct JSON.
    Raises :class:`KeylessMCPError` for JSON-RPC errors and ``isError``
    tool results (e.g. Exa's free-tier rate-limit message).
    """

    def _from_payload(payload: str) -> Optional[str]:
        payload = payload.strip()
        if not payload.startswith("{"):
            return None
        data = json.loads(payload)
        err = data.get("error")
        if err:
            raise KeylessMCPError(str(err.get("message") or err))
        result = data.get("result") or {}
        content = result.get("content") or []
        if result.get("isError"):
            texts = [c.get("text", "") for c in content if isinstance(c, dict)]
            raise KeylessMCPError(
                " ".join(t for t in texts if t) or "MCP tool call failed"
            )
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                return str(item["text"])
        return None

    stripped = body.strip()
    if stripped.startswith("{"):
        try:
            text = _from_payload(stripped)
            if text is not None:
                return text
        except json.JSONDecodeError:
            pass

    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        try:
            text = _from_payload(line[len("data: "):])
        except json.JSONDecodeError:
            continue
        if text is not None:
            return text

    raise KeylessMCPError("Unrecognized MCP response shape")


def mcp_call(
    url: str,
    tool: str,
    arguments: Dict[str, Any],
    timeout: int = _TIMEOUT_SECONDS,
) -> str:
    """POST a JSON-RPC ``tools/call`` to *url* and return the text payload.

    Raises :class:`KeylessMCPError` on transport failures, non-2xx
    statuses, JSON-RPC errors, and error-shaped tool results.
    """
    import requests

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": tool, "arguments": arguments},
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "User-Agent": "hermes-agent",
    }
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise KeylessMCPError(f"request failed: {exc}") from exc
    if response.status_code >= 400:
        raise KeylessMCPError(
            f"HTTP {response.status_code}: {response.text[:300]}"
        )
    return _parse_mcp_body(response.text)


# ---------------------------------------------------------------------------
# Parallel (search.parallel.ai) — JSON text payloads
# ---------------------------------------------------------------------------


def parallel_search_keyless(query: str, limit: int = 5) -> Dict[str, Any]:
    """Keyless Parallel web search → legacy search response shape."""
    try:
        text = mcp_call(
            PARALLEL_MCP_URL,
            "web_search",
            {
                "objective": query,
                "search_queries": [query],
                "session_id": _SESSION_ID,
            },
        )
        data = json.loads(text)
        web_results = []
        for i, result in enumerate(data.get("results") or []):
            if limit and i >= limit:
                break
            excerpts = result.get("excerpts") or []
            web_results.append(
                {
                    "url": result.get("url") or "",
                    "title": result.get("title") or "",
                    "description": " ".join(excerpts) if excerpts else "",
                    "position": i + 1,
                }
            )
        return {"success": True, "data": {"web": web_results}}
    except KeylessMCPError as exc:
        return {
            "success": False,
            "error": (
                f"Keyless Parallel search failed: {exc}. "
                "Set PARALLEL_API_KEY (https://parallel.ai) or another web "
                "backend via `hermes tools` for reliable service."
            ),
        }
    except (json.JSONDecodeError, TypeError, KeyError) as exc:
        return {"success": False, "error": f"Keyless Parallel search returned an unexpected payload: {exc}"}


def parallel_extract_keyless(urls: List[str]) -> List[Dict[str, Any]]:
    """Keyless Parallel web fetch → legacy extract result list."""
    try:
        text = mcp_call(
            PARALLEL_MCP_URL,
            "web_fetch",
            {
                "urls": list(urls),
                "objective": "Full page content",
                "session_id": _SESSION_ID,
            },
        )
        data = json.loads(text)
    except (KeylessMCPError, json.JSONDecodeError, TypeError) as exc:
        message = (
            f"Keyless Parallel extract failed: {exc}. "
            "Set PARALLEL_API_KEY (https://parallel.ai) or another web "
            "backend via `hermes tools` for reliable service."
        )
        return [
            {"url": u, "title": "", "content": "", "error": message}
            for u in urls
        ]

    results: List[Dict[str, Any]] = []
    seen = set()
    for result in data.get("results") or []:
        url = result.get("url") or ""
        title = result.get("title") or ""
        content = (
            result.get("full_content")
            or result.get("content")
            or "\n\n".join(result.get("excerpts") or [])
        )
        seen.add(url)
        results.append(
            {
                "url": url,
                "title": title,
                "content": content,
                "raw_content": content,
                "metadata": {"sourceURL": url, "title": title},
            }
        )
    for error in data.get("errors") or []:
        url = error.get("url") or ""
        seen.add(url)
        results.append(
            {
                "url": url,
                "title": "",
                "content": "",
                "error": str(
                    error.get("content") or error.get("error_type") or "extraction failed"
                ),
                "metadata": {"sourceURL": url},
            }
        )
    # Any URL the endpoint silently dropped still gets an error entry so the
    # caller's per-URL contract holds.
    for u in urls:
        if u not in seen:
            results.append(
                {"url": u, "title": "", "content": "", "error": "no content returned"}
            )
    return results


# ---------------------------------------------------------------------------
# Exa (mcp.exa.ai) — formatted plain-text payloads
# ---------------------------------------------------------------------------


def _parse_exa_search_text(text: str, limit: int) -> List[Dict[str, Any]]:
    """Parse Exa's formatted search text into result dicts.

    The payload is blocks separated by ``---`` lines, each shaped like::

        Title: <title>
        URL: <url>
        Published: ...
        Author: ...
        Highlights:
        <free text>
    """
    results: List[Dict[str, Any]] = []
    for block in text.split("\n---\n"):
        title = ""
        url = ""
        highlight_lines: List[str] = []
        in_highlights = False
        for line in block.splitlines():
            stripped = line.strip()
            if stripped.startswith("Title:"):
                title = stripped[len("Title:"):].strip()
                in_highlights = False
            elif stripped.startswith("URL:"):
                url = stripped[len("URL:"):].strip()
                in_highlights = False
            elif stripped.startswith("Highlights:"):
                in_highlights = True
            elif stripped.startswith(("Published:", "Author:")):
                in_highlights = False
            elif in_highlights and stripped:
                highlight_lines.append(stripped)
        if url:
            results.append(
                {
                    "url": url,
                    "title": title,
                    "description": " ".join(highlight_lines),
                    "position": len(results) + 1,
                }
            )
        if limit and len(results) >= limit:
            break
    return results


def exa_search_keyless(query: str, limit: int = 5) -> Dict[str, Any]:
    """Keyless Exa web search → legacy search response shape."""
    try:
        text = mcp_call(
            EXA_MCP_URL,
            "web_search_exa",
            {"query": query, "numResults": max(1, int(limit))},
        )
    except KeylessMCPError as exc:
        return {
            "success": False,
            "error": (
                f"Keyless Exa search failed: {exc}. "
                "Set EXA_API_KEY (https://exa.ai) or another web backend "
                "via `hermes tools` for reliable service."
            ),
        }
    return {"success": True, "data": {"web": _parse_exa_search_text(text, limit)}}


def exa_extract_keyless(urls: List[str]) -> List[Dict[str, Any]]:
    """Keyless Exa web fetch → legacy extract result list.

    ``web_fetch_exa`` takes a ``urls`` array but returns one combined text
    payload; we call it per-URL so each result maps cleanly.
    """
    results: List[Dict[str, Any]] = []
    for url in urls:
        try:
            text = mcp_call(EXA_MCP_URL, "web_fetch_exa", {"urls": [url]})
        except KeylessMCPError as exc:
            results.append(
                {
                    "url": url,
                    "title": "",
                    "content": "",
                    "error": (
                        f"Keyless Exa extract failed: {exc}. "
                        "Set EXA_API_KEY (https://exa.ai) or another web "
                        "backend via `hermes tools` for reliable service."
                    ),
                }
            )
            continue
        title = ""
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                title = stripped[2:].strip()
                break
            if stripped.startswith("Title:"):
                title = stripped[len("Title:"):].strip()
                break
        results.append(
            {
                "url": url,
                "title": title,
                "content": text,
                "raw_content": text,
                "metadata": {"sourceURL": url, "title": title},
            }
        )
    return results



# ---------------------------------------------------------------------------
# Tavily keyless (api.tavily.com — X-Tavily-Access-Mode: keyless)
# ---------------------------------------------------------------------------


TAVILY_API_URL = "https://api.tavily.com"


def _tavily_keyless_post(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST to Tavily with keyless headers; raise KeylessMCPError on failure."""
    import requests

    try:
        response = requests.post(
            f"{TAVILY_API_URL}/{endpoint.lstrip('/')}",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "X-Client-Name": "hermes-agent",
                "X-Tavily-Access-Mode": "keyless",
            },
            timeout=_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        raise KeylessMCPError(f"request failed: {exc}") from exc
    if response.status_code >= 400:
        raise KeylessMCPError(
            (response.text or "").strip() or f"HTTP {response.status_code}"
        )
    return response.json()


def tavily_search_keyless(query: str, limit: int = 5) -> Dict[str, Any]:
    """Keyless Tavily search → legacy search response shape."""
    try:
        data = _tavily_keyless_post(
            "search", {"query": query, "max_results": max(1, int(limit))}
        )
    except KeylessMCPError as exc:
        return {
            "success": False,
            "error": (
                f"Keyless Tavily search failed: {exc}. "
                "Set TAVILY_API_KEY (https://app.tavily.com) or another web "
                "backend via `hermes tools` for reliable service."
            ),
        }
    web_results = []
    for i, result in enumerate(data.get("results") or []):
        web_results.append(
            {
                "url": result.get("url") or "",
                "title": result.get("title") or "",
                "description": result.get("content") or "",
                "position": i + 1,
            }
        )
    return {"success": True, "data": {"web": web_results}}


def tavily_extract_keyless(urls: List[str]) -> List[Dict[str, Any]]:
    """Keyless Tavily extract → legacy extract result list."""
    try:
        data = _tavily_keyless_post("extract", {"urls": list(urls)})
    except KeylessMCPError as exc:
        message = (
            f"Keyless Tavily extract failed: {exc}. "
            "Set TAVILY_API_KEY (https://app.tavily.com) or another web "
            "backend via `hermes tools` for reliable service."
        )
        return [
            {"url": u, "title": "", "content": "", "error": message}
            for u in urls
        ]
    results: List[Dict[str, Any]] = []
    seen = set()
    for result in data.get("results") or []:
        url = result.get("url") or ""
        raw = result.get("raw_content") or result.get("content") or ""
        seen.add(url)
        results.append(
            {
                "url": url,
                "title": result.get("title") or "",
                "content": raw,
                "raw_content": raw,
                "metadata": {"sourceURL": url, "title": result.get("title") or ""},
            }
        )
    for fail in data.get("failed_results") or []:
        url = (fail.get("url") if isinstance(fail, dict) else str(fail)) or ""
        seen.add(url)
        results.append(
            {
                "url": url,
                "title": "",
                "content": "",
                "error": (fail.get("error") if isinstance(fail, dict) else None)
                or "extraction failed",
            }
        )
    for u in urls:
        if u not in seen:
            results.append(
                {"url": u, "title": "", "content": "", "error": "no content returned"}
            )
    return results


# ---------------------------------------------------------------------------
# Firecrawl keyless (public cloud API, no auth header)
# ---------------------------------------------------------------------------


def firecrawl_search_keyless(query: str, limit: int = 5) -> Dict[str, Any]:
    """Keyless Firecrawl cloud search → legacy search response shape."""
    from plugins.web.firecrawl.provider import (
        _KeylessFirecrawlClient,
        _extract_web_search_results,
    )

    try:
        response = _KeylessFirecrawlClient().search(query=query, limit=limit)
        return {"success": True, "data": {"web": _extract_web_search_results(response)}}
    except Exception as exc:  # noqa: BLE001 — normalized below
        return {
            "success": False,
            "error": (
                f"Keyless Firecrawl search failed: {exc}. "
                "Set FIRECRAWL_API_KEY (https://firecrawl.dev) or another web "
                "backend via `hermes tools` for reliable service."
            ),
        }


def firecrawl_extract_keyless(urls: List[str]) -> List[Dict[str, Any]]:
    """Keyless Firecrawl cloud scrape → legacy extract result list."""
    from plugins.web.firecrawl.provider import (
        _KeylessFirecrawlClient,
        _extract_scrape_payload,
    )

    client = _KeylessFirecrawlClient()
    results: List[Dict[str, Any]] = []
    for url in urls:
        try:
            response = client.scrape(url=url, formats=["markdown"])
            payload = _extract_scrape_payload(response) or {}
            metadata = payload.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}
            content = payload.get("markdown") or payload.get("html") or ""
            title = metadata.get("title") or ""
            results.append(
                {
                    "url": url,
                    "title": title,
                    "content": content,
                    "raw_content": content,
                    "metadata": {"sourceURL": url, "title": title},
                }
            )
        except Exception as exc:  # noqa: BLE001 — per-URL error entry
            results.append(
                {
                    "url": url,
                    "title": "",
                    "content": "",
                    "error": (
                        f"Keyless Firecrawl extract failed: {exc}. "
                        "Set FIRECRAWL_API_KEY (https://firecrawl.dev) for "
                        "reliable service."
                    ),
                }
            )
    return results


# ---------------------------------------------------------------------------
# Keenable keyless (api.keenable.ai public endpoints)
# ---------------------------------------------------------------------------


KEENABLE_API_URL = "https://api.keenable.ai"
_KEENABLE_TITLE = "hermes-agent"


def keenable_search_keyless(query: str, limit: int = 5) -> Dict[str, Any]:
    """Keyless Keenable search → legacy search response shape.

    POST /v1/search/public with the mandatory X-Keenable-Title app
    identifier (their keyless tier requires an app name; no user
    identifiers are sent). Response: {results: [{title, url, snippet}]}.
    """
    import requests

    try:
        response = requests.post(
            f"{KEENABLE_API_URL}/v1/search/public",
            json={"query": query, "max_results": max(1, int(limit))},
            headers={
                "Content-Type": "application/json",
                "X-Keenable-Title": _KEENABLE_TITLE,
            },
            timeout=_TIMEOUT_SECONDS,
        )
        if response.status_code >= 400:
            raise KeylessMCPError(
                (response.text or "").strip() or f"HTTP {response.status_code}"
            )
        data = response.json()
    except KeylessMCPError as exc:
        return {
            "success": False,
            "error": (
                f"Keyless Keenable search failed: {exc}. "
                "Set KEENABLE_API_KEY (https://keenable.ai) or another web "
                "backend via `hermes tools` for reliable service."
            ),
        }
    except Exception as exc:  # noqa: BLE001 — transport/JSON errors
        return {
            "success": False,
            "error": f"Keyless Keenable search failed: {exc}.",
        }
    web_results = []
    for i, result in enumerate(data.get("results") or []):
        web_results.append(
            {
                "url": result.get("url") or "",
                "title": result.get("title") or "",
                "description": result.get("snippet")
                or result.get("description")
                or "",
                "position": i + 1,
            }
        )
    return {"success": True, "data": {"web": web_results}}


def keenable_extract_keyless(urls: List[str]) -> List[Dict[str, Any]]:
    """Keyless Keenable page fetch → legacy extract result list.

    GET /v1/fetch/public?url=... returns {url, title, content} (markdown).
    Called per-URL; failures become per-URL error entries.
    """
    import requests

    results: List[Dict[str, Any]] = []
    for url in urls:
        try:
            response = requests.get(
                f"{KEENABLE_API_URL}/v1/fetch/public",
                params={"url": url},
                headers={"X-Keenable-Title": _KEENABLE_TITLE},
                timeout=_TIMEOUT_SECONDS,
            )
            if response.status_code >= 400:
                raise KeylessMCPError(
                    (response.text or "").strip() or f"HTTP {response.status_code}"
                )
            data = response.json()
            content = data.get("content") or ""
            title = data.get("title") or ""
            results.append(
                {
                    "url": data.get("url") or url,
                    "title": title,
                    "content": content,
                    "raw_content": content,
                    "metadata": {"sourceURL": url, "title": title},
                }
            )
        except Exception as exc:  # noqa: BLE001 — per-URL error entry
            results.append(
                {
                    "url": url,
                    "title": "",
                    "content": "",
                    "error": (
                        f"Keyless Keenable extract failed: {exc}. "
                        "Set KEENABLE_API_KEY (https://keenable.ai) for "
                        "reliable service."
                    ),
                }
            )
    return results


# ---------------------------------------------------------------------------
# Round-robin ring + next-in-line failover (rate-limited free tiers)
# ---------------------------------------------------------------------------

_KEYLESS_RING = ("exa", "parallel", "tavily", "firecrawl", "keenable")

_KEYLESS_SEARCHERS = {
    "exa": lambda query, limit: exa_search_keyless(query, limit),
    "parallel": lambda query, limit: parallel_search_keyless(query, limit),
    "tavily": lambda query, limit: tavily_search_keyless(query, limit),
    "firecrawl": lambda query, limit: firecrawl_search_keyless(query, limit),
    "keenable": lambda query, limit: keenable_search_keyless(query, limit),
}

_KEYLESS_EXTRACTORS = {
    "exa": lambda urls: exa_extract_keyless(urls),
    "parallel": lambda urls: parallel_extract_keyless(urls),
    "tavily": lambda urls: tavily_extract_keyless(urls),
    "firecrawl": lambda urls: firecrawl_extract_keyless(urls),
    "keenable": lambda urls: keenable_extract_keyless(urls),
}

# Per-process round-robin cursor, seeded by the random session id so the
# fleet spreads evenly across all five free tiers; advances once per
# unpinned keyless request so a single process also rotates.
_ring_lock = __import__("threading").Lock()
_ring_cursor = int(_SESSION_ID, 16) % len(_KEYLESS_RING)


def _vendor_pinned(name: str) -> bool:
    """True when config explicitly routes web traffic to *name*.

    A pinned vendor starts every keyless request (rotation off); the ring
    is only walked past it on throttle. Pin signals: web.backend /
    web.search_backend / web.extract_backend naming the vendor, or a
    free-tier pin in web.provider_tier.
    """
    if provider_tier(name) == "free":
        return True
    try:
        import tools.web_tools as _wt

        web_cfg = _wt._load_web_config()
        return any(
            (web_cfg.get(key) or "").lower().strip() == name
            for key in ("backend", "search_backend", "extract_backend")
        )
    except Exception as exc:  # noqa: BLE001 — config layer optional
        logger.debug("_vendor_pinned(%r) config read failed: %s", name, exc)
        return False


def _ring_order(name: str) -> List[str]:
    """Return the vendor walk order for a request entering via *name*.

    Pinned vendor → start at it (its position in the ring determines the
    failover succession). Unpinned → true round-robin: start at the next
    cursor position, advancing the cursor per request. Vendors whose tier
    is pinned ``paid`` are excluded entirely (an explicit paid selection
    opts that vendor's free endpoint out).
    """
    global _ring_cursor
    if _vendor_pinned(name):
        start = _KEYLESS_RING.index(name) if name in _KEYLESS_RING else 0
    else:
        with _ring_lock:
            start = _ring_cursor
            _ring_cursor = (_ring_cursor + 1) % len(_KEYLESS_RING)
    ordered = [
        _KEYLESS_RING[(start + i) % len(_KEYLESS_RING)]
        for i in range(len(_KEYLESS_RING))
    ]
    return [v for v in ordered if provider_tier(v) != "paid"]


def search_with_failover(name: str, query: str, limit: int = 5) -> Dict[str, Any]:
    """Keyless search across the vendor ring with next-in-line failover.

    Starts at *name* when the user pinned it, otherwise at the round-robin
    cursor. Rate-limit-shaped errors advance to the next ring vendor;
    non-throttle errors stop the walk (a malformed query fails everywhere).
    The result notes the serving vendor via ``data.served_by`` whenever it
    differs from *name*.
    """
    order = _ring_order(name)
    if not order:
        return {
            "success": False,
            "error": "All keyless web providers are pinned to paid tiers.",
        }
    last: Dict[str, Any] = {}
    for i, vendor in enumerate(order):
        result = _KEYLESS_SEARCHERS[vendor](query, limit)
        if result.get("success"):
            if vendor != name:
                result.setdefault("data", {})["served_by"] = vendor
            return result
        last = result
        if not _is_rate_limitish(result.get("error", "")):
            return result
        nxt = order[i + 1] if i + 1 < len(order) else None
        if nxt:
            logger.info(
                "keyless %s search throttled; failing over to %s", vendor, nxt
            )
    last["error"] = (
        f"{last.get('error', '')} (all keyless vendors throttled: "
        f"{', '.join(order)})"
    )
    return last


def extract_with_failover(name: str, urls: List[str]) -> List[Dict[str, Any]]:
    """Keyless extract across the vendor ring, failing over per-batch.

    Advances to the next ring vendor only when EVERY url in a batch comes
    back with a rate-limit-shaped error — partial failures are page
    problems, not throttling, and return as-is.
    """
    order = _ring_order(name)
    if not order:
        return [
            {"url": u, "title": "", "content": "",
             "error": "All keyless web providers are pinned to paid tiers."}
            for u in urls
        ]
    last: List[Dict[str, Any]] = []
    for i, vendor in enumerate(order):
        results = _KEYLESS_EXTRACTORS[vendor](list(urls))
        errors = [r.get("error", "") for r in results]
        all_throttled = bool(results) and all(
            e and _is_rate_limitish(e) for e in errors
        )
        if not all_throttled:
            return results
        last = results
        nxt = order[i + 1] if i + 1 < len(order) else None
        if nxt:
            logger.info(
                "keyless %s extract throttled; failing over to %s", vendor, nxt
            )
    return last
