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
