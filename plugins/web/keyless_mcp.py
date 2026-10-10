"""Keyless web search/extract via public free-tier endpoints (Exa, Parallel, Firecrawl, Keenable).
Resolved strictly LAST — after every keyed backend, the managed gateway, ddgs and custom plugin
providers — so it never pre-empts a deliberate setup. Privacy: no user identifiers are sent;
Parallel gets a random per-process ``session_id`` (rate limiting only) and its optional
``model_name`` analytics field is deliberately omitted. Disable with ``web.keyless_fallback: false``.
"""

from __future__ import annotations

import json
import inspect
import logging
import re
import threading
import uuid
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field, ValidationError

if TYPE_CHECKING:
    from requests import Response

from plugins.web._common import document as _page, page_error as _page_error, search_fail, search_ok, web_hit as _row
from agent.web_acquisition_errors import WebCancelledError, WebCapabilityUnsupportedError, WebInvalidResponseError, WebResultMissingError, WebUnclassifiedFailure, acquisition_error, failure_from_data, http_failure
from agent.web_acquisition import WebAttempt, WebAttemptFailure, WebAttemptSuccess, WebExtractCapabilities

logger = logging.getLogger(__name__)

EXA_MCP_URL = "https://mcp.exa.ai/mcp"
PARALLEL_MCP_URL = "https://search.parallel.ai/mcp"
KEENABLE_API_URL = "https://api.keenable.ai"
_KEENABLE_TITLE = "hermes-agent"

# Parallel free-tier rate-limit correlation id — random per process, never persisted.
_SESSION_ID = uuid.uuid4().hex

_TIMEOUT_SECONDS = 30


class KeylessMCPError(RuntimeError):
    """A keyless MCP call failed (transport, rate limit, or tool error)."""


class KeylessMCPHttpError(KeylessMCPError):
    """A provider endpoint response with a known HTTP status."""

    def __init__(self, response: Response):
        status = response.status_code
        if type(status) is not int or not 400 <= status <= 599:
            raise WebInvalidResponseError()
        super().__init__(f"HTTP {status}: {_response_text(response)[:300]}")
        self.__cause__ = http_failure(status, response.headers)


class KeylessMCPInvalidResponseError(KeylessMCPError):
    """The upstream response did not contain a valid MCP envelope."""

    def __init__(self, *, cause: BaseException | None = None):
        super().__init__("Unrecognized MCP response shape")
        self.__cause__ = WebInvalidResponseError(cause=cause)


class KeylessMCPNoTextError(KeylessMCPError):
    """The MCP response did not contain usable text content."""

    def __init__(self):
        super().__init__("MCP response contained no text content")
        self.__cause__ = WebInvalidResponseError()


class _MCPContent(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    type: str | None = None
    text: str | None = None


class _MCPResult(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    content: list[_MCPContent] = Field(default_factory=list)
    is_error: bool = Field(default=False, alias="isError")


class _MCPError(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    message: str


class _MCPEnvelope(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    result: _MCPResult | None = None
    error: _MCPError | None = None


_RATE_LIMIT_MARKERS = ("rate limit", "rate-limit", "ratelimit", "too many requests", "429", "quota exceeded", "slow down")

# vendor -> (display label, env key, signup URL) for the standard failure hint.
_VENDOR_HINTS = {
    "exa": ("Exa", "EXA_API_KEY", "https://exa.ai"), "parallel": ("Parallel", "PARALLEL_API_KEY", "https://parallel.ai"),
    "firecrawl": ("Firecrawl", "FIRECRAWL_API_KEY", "https://firecrawl.dev"), "keenable": ("Keenable", "KEENABLE_API_KEY", "https://keenable.ai"),
}


def _is_rate_limitish(message: str) -> bool:
    """Heuristic: does an error message look like free-tier throttling?"""
    return any(marker in (message or "").lower() for marker in _RATE_LIMIT_MARKERS)


# A free tier that refuses this client (401/402/403: anonymous access revoked, IP reputation gate)
# or errors server-side (5xx) fails every query from here, while another vendor may still serve it.
# Anchored to the status the error starts with (after the ``Keyless <Vendor> search failed:`` prefix),
# so a terminal error that echoes the query ("HTTP 400: invalid query 'http 503'") does not match.
_VENDOR_REFUSAL_RE = re.compile(
    r"\s*(?:keyless\s+\w+\s+search\s+failed:\s*)?"
    r"(?:http(?:\s+status)?|status(?:\s+code)?|client\s+error|error\s+code)"
    r"\s*[:=']*\s*(?:40[123]|5\d\d)\b",
    re.IGNORECASE,
)


def _is_search_failover_eligible(message: str) -> bool:
    """Return whether another anonymous vendor may serve the search.

    Rate limits and structured HTTP 401/402/403/5xx vendor refusals are local to
    one free search endpoint. Free-text markers are deliberately ignored:
    vendors may echo the query in an otherwise terminal error.
    """
    return _is_rate_limitish(message) or bool(_VENDOR_REFUSAL_RE.match(message or ""))


def _fail_msg(vendor: str, kind: str, exc: Any, *, other_backends: bool = True) -> str:
    label, env_key, site = _VENDOR_HINTS[vendor]
    alt = " or another web backend via `hermes tools`" if other_backends else ""
    return f"Keyless {label} {kind} failed: {exc}. Set {env_key} ({site}){alt} for reliable service."


def _search(vendor: str, rows: Callable[[], list[dict[str, Any]]], catch: Any = (), fmt: Optional[Callable[[Exception], str]] = None) -> dict[str, Any]:
    """``search_ok(rows())``; :class:`KeylessMCPError` → standard vendor hint, exception
    types in ``catch`` → ``fmt(exc)``; anything else propagates."""
    try:
        return search_ok(rows())
    except KeylessMCPError as exc:
        return search_fail(_fail_msg(vendor, "search", exc), failure=acquisition_error(exc))
    except catch as exc:
        failure = acquisition_error(exc)
        if isinstance(failure, WebUnclassifiedFailure) and isinstance(exc, (TypeError, KeyError)):
            failure = WebInvalidResponseError(cause=exc)
        diagnostic = fmt(exc) if fmt is not None else _fail_msg(vendor, "search", exc)
        return search_fail(diagnostic, failure=failure)


def _per_url(urls: list[str], fetch: Callable[[str], dict[str, Any]], vendor: str, catch: Any = Exception, hint: bool = False) -> list[dict[str, Any]]:
    """Per-URL extract loop: a ``catch`` failure becomes an error entry (``hint`` adds the ``hermes tools`` hint)."""
    def _one(url: str) -> dict[str, Any]:
        try:
            return fetch(url)
        except catch as exc:  # noqa: BLE001 — per-URL error entry
            return _page_error(url, _fail_msg(vendor, "extract", exc, other_backends=hint), failure=acquisition_error(exc))

    return [_one(u) for u in urls]


# --- Tier / config ------------------------------------------------------------
def keyless_enabled() -> bool:
    """Delegates to the registry so the ``web.keyless_fallback`` (default on) chokepoint lives with backend resolution."""
    try:
        from agent.web_search_registry import _keyless_tier_enabled
        return _keyless_tier_enabled()
    except Exception as exc:
        logger.debug("keyless_enabled(): registry helper unavailable: %s", exc)
        return True


_BACKEND_KEYS = ("backend", "search_backend", "extract_backend")


def _web_config_selects(name: str) -> bool:
    """True when any ``web.backend`` / ``search_backend`` / ``extract_backend`` names *name*."""
    import tools.web_tools as _wt
    web_cfg = _wt._load_web_config()
    return any((web_cfg.get(key) or "").lower().strip() == name for key in _BACKEND_KEYS)


def provider_tier(name: str) -> str:
    """``web.provider_tier.<name>`` (``hermes tools`` Free/Paid rows): ``free``, ``paid``, or ``auto`` (anything else/unset)."""
    try:
        from hermes_cli.config import load_config
        tiers = (load_config().get("web") or {}).get("provider_tier") or {}
        value = str(tiers.get(name, "") or "").lower().strip()
        return value if value in ("free", "paid") else "auto"
    except Exception as exc:
        logger.debug("provider_tier(%r) config read failed: %s", name, exc)
        return "auto"


def use_keyless(name: str, api_key: str) -> bool:
    """Single chokepoint for search + extract: ``free`` → keyless even with a key; ``paid`` → keyed even
    without one (the keyed path raises its usual missing-key error); ``auto`` → keyless only when no key + tier enabled."""
    tier = provider_tier(name)
    if tier in ("free", "paid"):
        return tier == "free"
    return not api_key and keyless_enabled()


# --- MCP transport ------------------------------------------------------------
def _parse_mcp_body(body: str) -> str:
    """First text content item from an MCP tools/call response — plain-JSON bodies
    (Parallel) or SSE ``data: {...}`` lines (Exa). Raises :class:`KeylessMCPError` for
    JSON-RPC errors and ``isError`` tool results (e.g. Exa's free-tier rate limit).

    SSE frames are split on the SSE line terminators (CRLF, CR, LF) only:
    ``str.splitlines()`` also breaks on U+0085 / U+2028 / U+2029, which legitimately
    occur inside CJK page text and would cut a ``data:`` line in two. A parsed envelope with no text is reported as such — it is
    the vendor's answer, not an unrecognized shape."""

    def _from_payload(payload: str) -> Optional[str]:
        payload = payload.strip()
        if not payload.startswith("{"):
            return None
        data = json.loads(payload)
        try:
            envelope = _MCPEnvelope.model_validate(data)
        except ValidationError as exc:
            raise KeylessMCPInvalidResponseError(cause=exc)
        if envelope.error is not None:
            raise KeylessMCPError(envelope.error.message)
        result = envelope.result
        if result is None:
            return ""
        texts = [content.text for content in result.content if content.text]
        if result.is_error:
            raise KeylessMCPError(" ".join(t for t in texts if t) or "MCP tool call failed")
        return next(iter(texts), "")

    stripped = body.strip()
    candidates = [stripped] if stripped.startswith("{") else []
    candidates += [line[len("data: "):] for line in re.split(r"\r\n|\r|\n", body) if line.startswith("data: ")]
    envelope_seen = False
    for candidate in candidates:
        try:
            text = _from_payload(candidate)
        except json.JSONDecodeError:
            continue
        if text:
            return text
        envelope_seen = envelope_seen or text == ""
    if envelope_seen:
        raise KeylessMCPNoTextError()
    raise KeylessMCPInvalidResponseError()


def _response_text(response: Any) -> str:
    """Body decoded with the declared charset, else UTF-8. JSON-RPC and SSE bodies are UTF-8 by
    spec, but ``text/event-stream`` carries no charset and ``requests`` then decodes ``.text`` as
    ISO-8859-1 — mojibake for every non-ASCII result and, for CJK, stray U+0085 line breaks that
    made the envelope unparseable."""
    from requests.utils import get_encoding_from_headers
    content_type = response.headers.get("Content-Type", "")
    declared = get_encoding_from_headers({"content-type": content_type}) if "charset=" in content_type.lower() else None
    return response.content.decode(declared or "utf-8", errors="replace")


def mcp_call(url: str, tool: str, arguments: dict[str, Any], timeout: int = _TIMEOUT_SECONDS) -> str:
    """POST a JSON-RPC ``tools/call`` and return the text payload. Raises
    :class:`KeylessMCPError` on transport failures, non-2xx, JSON-RPC and tool errors."""
    import requests
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": tool, "arguments": arguments}}
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream", "User-Agent": "hermes-agent"}
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise KeylessMCPError(f"request failed: {exc}") from exc
    if response.status_code >= 400:
        raise KeylessMCPHttpError(response)
    return _parse_mcp_body(_response_text(response))


# --- Parallel (search.parallel.ai) — JSON text payloads -----------------------
def parallel_search_keyless(query: str, limit: int = 5) -> dict[str, Any]:
    def _rows() -> list[dict[str, Any]]:
        text = mcp_call(PARALLEL_MCP_URL, "web_search", {"objective": query, "search_queries": [query], "session_id": _SESSION_ID})
        from plugins.web._common import search_result_rows
        results = search_result_rows(json.loads(text))
        return [
            _row(r.get("url") or "", r.get("title") or "", " ".join(r.get("excerpts") or []), i + 1)
            for i, r in enumerate(results[:max(limit, 0)] if limit else results)
        ]

    return _search("parallel", _rows, (json.JSONDecodeError, TypeError, KeyError, WebInvalidResponseError), lambda exc: f"Keyless Parallel search returned an unexpected payload: {exc}")


def parallel_extract_keyless(urls: list[str], **kwargs: Any) -> list[dict[str, Any]]:
    try:
        data = json.loads(mcp_call(PARALLEL_MCP_URL, "web_fetch", {"urls": list(urls), "objective": "Full page content", "session_id": _SESSION_ID}))
    except (KeylessMCPError, json.JSONDecodeError, TypeError) as exc:
        message = _fail_msg("parallel", "extract", exc)
        failure = acquisition_error(exc) if isinstance(exc, KeylessMCPError) else WebInvalidResponseError(cause=exc)
        return [_page_error(u, message, failure=failure) for u in urls]
    results = [
        _page(r.get("url") or "", r.get("title") or "", r.get("full_content") or r.get("content") or "\n\n".join(r.get("excerpts") or []), coverage="full" if r.get("full_content") else "partial" if r.get("excerpts") else "unknown")
        for r in data.get("results") or []
    ]
    for error in data.get("errors") or []:
        url = error.get("url") or ""
        results.append({**_page_error(url, str(error.get("content") or error.get("error_type") or "extraction failed")), "metadata": {"sourceURL": url}})
    # URLs the endpoint silently dropped still get an error entry (per-URL contract).
    seen = {r["url"] for r in results}
    results.extend(_page_error(u, "no content returned") for u in urls if u not in seen)
    return results


# --- Exa (mcp.exa.ai) — formatted plain-text payloads -------------------------
def _after(line: str, prefix: str) -> str:
    return line[len(prefix):].strip()


_EXA_LABELS = ("Title:", "URL:", "Highlights:", "Published:", "Author:")


def _parse_exa_search_text(text: str, limit: int) -> list[dict[str, Any]]:
    """Parse Exa's ``---``-separated ``Title:/URL:/Published:/Author:/Highlights:`` blocks."""
    results: list[dict[str, Any]] = []
    for block in text.split("\n---\n"):
        title = url = ""
        highlight_lines: list[str] = []
        in_highlights = False
        for stripped in map(str.strip, block.splitlines()):
            if stripped.startswith("Title:"):
                title = _after(stripped, "Title:")
            elif stripped.startswith("URL:"):
                url = _after(stripped, "URL:")
            elif in_highlights and stripped and not stripped.startswith(_EXA_LABELS):
                highlight_lines.append(stripped)
            # Highlights run until the next labelled field.
            if stripped.startswith(_EXA_LABELS):
                in_highlights = stripped.startswith("Highlights:")
        if url:
            results.append(_row(url, title, " ".join(highlight_lines), len(results) + 1))
        if limit and len(results) >= limit:
            break
    return results


def exa_search_keyless(query: str, limit: int = 5) -> dict[str, Any]:
    return _search("exa", lambda: _parse_exa_search_text(mcp_call(EXA_MCP_URL, "web_search_exa", {"query": query, "numResults": max(1, int(limit))}), limit))


def exa_extract_keyless(urls: list[str], **kwargs: Any) -> list[dict[str, Any]]:
    """Called per-URL; the tool returns one combined text payload."""
    def _fetch(url: str) -> dict[str, Any]:
        arguments = {"urls": [url]}
        if kwargs.get("max_chars") is not None:
            arguments["maxCharacters"] = kwargs["max_chars"]
        text = mcp_call(EXA_MCP_URL, "web_fetch_exa", arguments)
        # Title: first markdown H1 or ``Title:`` line, whichever comes first.
        titles = (_after(s, "# " if s.startswith("# ") else "Title:") for s in map(str.strip, text.splitlines()) if s.startswith(("# ", "Title:")))
        return _page(url, next(titles, ""), text)

    return _per_url(urls, _fetch, "exa", catch=KeylessMCPError, hint=True)


# --- Firecrawl keyless (public cloud API, no auth header) ---------------------
def firecrawl_search_keyless(query: str, limit: int = 5) -> dict[str, Any]:
    from plugins.web.firecrawl.provider import _KeylessFirecrawlClient, _extract_web_search_results
    rows = lambda: _extract_web_search_results(_KeylessFirecrawlClient().search(query=query, limit=limit))
    return _search("firecrawl", rows, Exception, lambda exc: _fail_msg("firecrawl", "search", exc))


def firecrawl_extract_keyless(urls: list[str], **kwargs: Any) -> list[dict[str, Any]]:
    from plugins.web.firecrawl.provider import _KeylessFirecrawlClient, _extract_scrape_payload
    client = _KeylessFirecrawlClient()

    def _fetch(url: str) -> dict[str, Any]:
        format = kwargs.get("format") or "markdown"
        payload = _extract_scrape_payload(client.scrape(url=url, formats=[format])) or {}
        metadata = payload.get("metadata") or {}
        title = metadata.get("title") if isinstance(metadata, dict) else None
        return _page(metadata.get("sourceURL") or url, title or "", payload.get(format) or "", source_url=url, coverage="full")

    return _per_url(urls, _fetch, "firecrawl")


# --- Keenable keyless (api.keenable.ai public endpoints) ----------------------
def _keenable_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    """Call a Keenable public endpoint with the mandatory X-Keenable-Title app id."""
    import requests
    headers = {"X-Keenable-Title": _KEENABLE_TITLE}
    if method == "post":
        headers["Content-Type"] = "application/json"
    response = getattr(requests, method)(f"{KEENABLE_API_URL}{path}", headers=headers, timeout=_TIMEOUT_SECONDS, **kwargs)
    if response.status_code >= 400:
        raise KeylessMCPHttpError(response)
    return response.json()


def keenable_search_keyless(query: str, limit: int = 5) -> dict[str, Any]:
    def _rows() -> list[dict[str, Any]]:
        data = _keenable_request("post", "/v1/search/public", json={"query": query, "max_results": max(1, int(limit))})
        from plugins.web._common import search_result_rows
        return [
            _row(r.get("url") or "", r.get("title") or "", r.get("snippet") or r.get("description") or "", i + 1)
            for i, r in enumerate(search_result_rows(data))
        ]

    return _search("keenable", _rows, Exception, lambda exc: f"Keyless Keenable search failed: {exc}.")


def keenable_extract_keyless(urls: list[str], **kwargs: Any) -> list[dict[str, Any]]:
    def _fetch(url: str) -> dict[str, Any]:
        data = _keenable_request("get", "/v1/fetch/public", params={"url": url})
        return _page(data.get("url") or url, data.get("title") or "", data.get("content") or "", source_url=url, coverage="full")

    return _per_url(urls, _fetch, "keenable")


# --- Round-robin ring + next-in-line failover (rate-limited free tiers) -------
_KEYLESS_RING = ("exa", "parallel", "firecrawl", "keenable")

# Late-bound lookups (not bare references) so ``patch.object(keyless_mcp, "<vendor>_search_keyless")``
# is honored at call time. Tests also ``setitem`` these dicts directly.
_KEYLESS_SEARCHERS: dict[str, Callable[[str, int], dict[str, Any]]] = {v: (lambda query, limit, _v=v: globals()[f"{_v}_search_keyless"](query, limit)) for v in _KEYLESS_RING}
def _keyless_extractor(vendor: str, urls: list[str], **kwargs: Any) -> list[dict[str, Any]]:
    extract = globals()[f"{vendor}_extract_keyless"]
    parameters = inspect.signature(extract).parameters
    supported = kwargs if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()) else {key: value for key, value in kwargs.items() if key in parameters}
    return extract(urls, **supported)


_KEYLESS_EXTRACTORS: dict[str, Callable[..., list[dict[str, Any]]]] = {v: (lambda urls, _v=v, **kwargs: _keyless_extractor(_v, urls, **kwargs)) for v in _KEYLESS_RING}

# Per-process round-robin cursor, seeded by the random session id so the fleet
# spreads across vendors; advances once per unpinned keyless request.
_ring_lock = threading.Lock()
_ring_cursor = int(_SESSION_ID, 16) % len(_KEYLESS_RING)


def _vendor_pinned(name: str) -> bool:
    """True when config explicitly routes web traffic to *name* (backend keys or a
    ``free`` tier pin). A pinned vendor starts every keyless request."""
    if provider_tier(name) == "free":
        return True
    try:
        return _web_config_selects(name)
    except Exception as exc:
        logger.debug("_vendor_pinned(%r) config read failed: %s", name, exc)
        return False


def _ring_order(name: str) -> list[str]:
    """Vendor walk order: pinned → start at *name* (its ring position fixes the failover
    succession); else round-robin from the cursor, advancing it per request. Vendors
    pinned ``paid`` are excluded (explicit paid opts their free endpoint out)."""
    global _ring_cursor
    if _vendor_pinned(name):
        start = _KEYLESS_RING.index(name) if name in _KEYLESS_RING else 0
    else:
        with _ring_lock:
            start = _ring_cursor
            _ring_cursor = (_ring_cursor + 1) % len(_KEYLESS_RING)
    ordered = _KEYLESS_RING[start:] + _KEYLESS_RING[:start]
    return [v for v in ordered if provider_tier(v) != "paid"]


_ALL_PAID_MSG = "All keyless web providers are pinned to paid tiers."


def _walk_ring(name: str, kind: str, call, throttled, *, attempts: list[WebAttempt] | None = None) -> tuple:
    """Call each vendor from :func:`_ring_order` until a result is not ``throttled``.
    Returns ``(order, vendor, result, exhausted)``; ``order`` is empty (result None)
    when every vendor is pinned paid."""
    order = _ring_order(name)
    vendor, result = None, None
    for i, vendor in enumerate(order):
        if attempts is not None:
            from tools.interrupt import is_interrupted
            if is_interrupted():
                return order, vendor, search_fail("Interrupted", failure=WebCancelledError()), False
        if attempts is None:
            result = call(vendor)
        else:
            from tools.web_tools_search import checked_search_response, search_attempts
            try:
                result = call(vendor)
            except Exception as exc:  # noqa: BLE001 — preserve the actual attempted vendor
                result = search_fail(_fail_msg(vendor, kind, exc), failure=acquisition_error(exc))
            result = checked_search_response(result)
            attempts.extend(search_attempts(result, vendor, "keyless"))
            if is_interrupted():
                return order, vendor, search_fail("Interrupted", failure=WebCancelledError()), False
        if not throttled(result):
            return order, vendor, result, False
        if i + 1 < len(order):
            logger.info("keyless %s %s unavailable; failing over to %s", vendor, kind, order[i + 1])
    return order, vendor, result, True


def search_with_failover(name: str, query: str, limit: int = 5) -> dict[str, Any]:
    """Rate limits and vendor refusals (HTTP 401/402/403/5xx) advance to the next
    vendor, other errors stop the walk (a malformed query fails everywhere).
    ``data.served_by`` is set when the serving vendor differs from *name*."""

    def _throttled(result: dict[str, Any]) -> bool:
        return not result.get("success") and _is_search_failover_eligible(result.get("error", ""))

    attempts: list[WebAttempt] = []
    order, vendor, result, exhausted = _walk_ring(name, "search", lambda v: _KEYLESS_SEARCHERS[v](query, limit), _throttled, attempts=attempts)
    if not order:
        return search_fail(_ALL_PAID_MSG, failure=WebUnclassifiedFailure())
    if exhausted:
        result["error"] = f"{result.get('error', '')} (all keyless vendors unavailable: {', '.join(order)})"
    elif result.get("success") and vendor != name:
        result.setdefault("data", {})["served_by"] = vendor
    result["attempts"] = [attempt.model_dump(mode="json", exclude_none=True) for attempt in attempts]
    if result.get("success"):
        result["served_provider"] = vendor
    elif result.get("failure", {}).get("kind") != "cancelled" and attempts and isinstance(attempts[0], WebAttemptFailure):
        result["failure"] = attempts[0].failure
    return result


_KEYLESS_EXTRACT_CAPABILITIES = {
    "exa": WebExtractCapabilities(formats=("markdown",), max_characters=True),
    "parallel": WebExtractCapabilities(formats=("markdown",)),
    "firecrawl": WebExtractCapabilities(formats=("markdown", "html")),
    "keenable": WebExtractCapabilities(formats=("markdown",)),
}


def extract_with_failover(name: str, urls: list[str], **kwargs: Any) -> list[dict[str, Any]]:
    """Preserve per-page receipts through the native keyless ring and cancellation."""
    from tools.interrupt import is_interrupted
    from tools.web_tools_extract import checked_extract_rows

    order = _ring_order(name)
    if not order:
        return [_page_error(url, _ALL_PAID_MSG, failure=WebUnclassifiedFailure()) for url in urls]
    order = [vendor for vendor in order
             if (kwargs.get("format") is None or kwargs["format"] in _KEYLESS_EXTRACT_CAPABILITIES[vendor].formats)
             and (kwargs.get("max_chars") is None or _KEYLESS_EXTRACT_CAPABILITIES[vendor].max_characters)]
    if not order:
        failure = WebCapabilityUnsupportedError("extract-options")
        return [{**_page_error(url, failure.diagnostic, failure=failure), "attempts": []} for url in urls]
    requested = list(dict.fromkeys(urls))
    attempts: dict[str, list[WebAttempt]] = {url: [] for url in requested}
    results: dict[str, dict[str, Any]] = {}

    def cancelled() -> list[dict[str, Any]]:
        return [{**_page_error(url, "Interrupted", failure=WebCancelledError()),
                 "attempts": [attempt.model_dump(mode="json", exclude_none=True) for attempt in attempts[url]]}
                for url in urls]

    for vendor in order:
        if is_interrupted():
            return cancelled()
        try:
            extract = _KEYLESS_EXTRACTORS[vendor]
            parameters = inspect.signature(extract).parameters
            supported = kwargs if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()) else {key: value for key, value in kwargs.items() if key in parameters}
            rows = checked_extract_rows(extract(requested, **supported), requested)
        except Exception as exc:  # noqa: BLE001 — record the attempted vendor before advancing
            failure = acquisition_error(exc)
            rows = checked_extract_rows([_page_error(url, _fail_msg(vendor, "extract", exc), failure=failure) for url in requested], requested)
        throttled = True
        for url in requested:
            row = rows.get(url)
            failure = WebResultMissingError() if row is None else failure_from_data(row.failure) if row.failure is not None else WebUnclassifiedFailure() if row.error else None
            if failure is not None:
                attempts[url].append(WebAttemptFailure(provider=vendor, route="keyless", failure=failure.to_failure()))
                results[url] = _page_error(url, row.error if row is not None and row.error else failure.diagnostic, failure=failure)
                throttled = throttled and failure.scope == "provider" and _is_rate_limitish(results[url]["error"])
                continue
            if row is None:
                raise WebInvalidResponseError()
            attempts[url].append(WebAttemptSuccess(provider=vendor, route="keyless"))
            results[url] = row.model_dump(mode="json", exclude_none=True)
            results[url]["served_provider"] = vendor
            throttled = False
        if is_interrupted():
            return cancelled()
        if not throttled:
            break
    for url, result in results.items():
        first = attempts[url][0]
        if result.get("error") and isinstance(first, WebAttemptFailure):
            result["failure"] = first.failure
        result["attempts"] = [attempt.model_dump(mode="json", exclude_none=True) for attempt in attempts[url]]
    return [results[url] for url in urls]
