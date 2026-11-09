"""Shared plumbing for the bundled web-search provider plugins.

Every helper resolves patched-in-tests collaborators (``get_provider_env``,
``tools.interrupt``, ``plugins.web.keyless_mcp``, ``tools.web_tools`` client
slots) lazily at call time so monkeypatching the source module keeps working.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional

import httpx

from agent.web_search_provider import WebSearchProvider

SEARCH_LIMIT_CAP = 20  # every vendor here caps max_results at 20 server-side


def provider_env(name: str) -> str:
    """Config-aware env lookup (os.environ, then ~/.hermes/.env)."""
    from agent.web_search_provider import get_provider_env
    return get_provider_env(name)


def use_keyless(name: str, api_key: str) -> bool:
    from plugins.web.keyless_mcp import use_keyless as _use_keyless
    return _use_keyless(name, api_key)


def _interrupted() -> bool:
    from tools.interrupt import is_interrupted
    return is_interrupted()


# --- Result shapes (key order is part of the contract — it reaches the model as JSON) ---
def search_ok(web_results: list[dict[str, Any]]) -> dict[str, Any]:
    return {"success": True, "data": {"web": web_results}}


def search_fail(error: str) -> dict[str, Any]:
    return {"success": False, "error": error}


def web_hit(url: str, title: str, description: str, position: int) -> dict[str, Any]:
    return {"url": url, "title": title, "description": description, "position": position}


def title_hit(title: str, url: str, description: str, position: int) -> dict[str, Any]:
    """Title-first row — the historical wire shape of brave/searxng/ddgs/tavily/xai."""
    return {"title": title, "url": url, "description": description, "position": position}


def document(url: str, title: str, content: str, *, source_url: Optional[str] = None) -> dict[str, Any]:
    """Successful extract entry; ``raw_content`` mirrors ``content`` for the legacy pipeline."""
    return {
        "url": url, "title": title, "content": content, "raw_content": content,
        "metadata": {"sourceURL": url if source_url is None else source_url, "title": title},
    }


def page_error(url: str, error: str) -> dict[str, Any]:
    return {"url": url, "title": "", "content": "", "error": error}


def extract_fail(urls: list[str], error: str) -> list[dict[str, Any]]:
    return [page_error(u, error) for u in urls]


# --- Origin HTTP status (the target page's answer, not the vendor API's) ---
# Scraping vendors report the target's own status inside a successful API reply (Firecrawl
# ``metadata.statusCode``, Parallel ``errors[].http_status_code``, Exa ``statuses[].error``).
# Unchecked, a 403/404 page reads as a successful (often empty) extraction: the agent can't tell
# "blocked" from "blank" and the error page is cached as content. The backend did its job, so an
# origin error never triggers the keyless rescue (``is_origin_http_error``).
_ORIGIN_HTTP_ERROR_RE = re.compile(r"HTTP \d{3} from origin(?::|$)")


def is_origin_http_error(error: Any) -> bool:
    """True for an ``error`` string built by ``origin_http_error`` (the target page's own answer)."""
    return isinstance(error, str) and _ORIGIN_HTTP_ERROR_RE.match(error) is not None


def origin_http_error(status: Any, reason: Any = None) -> Optional[str]:
    """``HTTP <status> from origin[: reason]`` when the target answered >= 400, else None."""
    if isinstance(status, bool):
        return None
    try:
        code = int(status)
    except (TypeError, ValueError):
        return None
    if code < 400:
        return None
    detail = str(reason or "").strip()
    return f"HTTP {code} from origin: {detail}" if detail else f"HTTP {code} from origin"


# Vendors that fetch the page themselves also report its status in their own words: Exa
# ``CRAWL_HTTP_<code>`` / ``CRAWL_NOT_FOUND``; Keenable mirrors the target's code as its own HTTP
# status with a message about the target. Read as vendor errors, an origin 429 looked like the
# vendor throttling us and walked the keyless ring.
_EXA_CRAWL_STATUS_RE = re.compile(r"\bCRAWL_(?:HTTP_(\d{3})|(NOT_FOUND))\s*$")
_KEENABLE_ORIGIN_MESSAGE_RE = re.compile(r"the (?:target server|page does not exist)\b", re.IGNORECASE)


def exa_crawl_status(text: Any) -> Optional[int]:
    """The origin status an Exa crawl error tag ends with (``CRAWL_HTTP_429`` → 429, ``CRAWL_NOT_FOUND`` → 404)."""
    match = _EXA_CRAWL_STATUS_RE.search(str(text or ""))
    if match is None:
        return None
    return int(match.group(1)) if match.group(1) else 404


def keenable_origin_error(status: Any, body: str) -> Optional[str]:
    """``HTTP <status> from origin: <message>`` when a Keenable fetch error describes the target's
    answer; None for Keenable's own refusals (its rate limit for this client, auth, 5xx)."""
    try:
        data = json.loads(body or "")
    except ValueError:
        return None
    message = str(data.get("message") or "").strip() if isinstance(data, dict) else ""
    return origin_http_error(status, message) if _KEENABLE_ORIGIN_MESSAGE_RE.match(message) else None


def with_origin_status(entry: dict[str, Any], status: Any, reason: Any = None) -> dict[str, Any]:
    """Set ``error`` on an extract entry whose target answered >= 400; the error page body
    stays in ``content`` (it can still explain the refusal)."""
    if error := origin_http_error(status, reason):
        entry["error"] = error
    return entry


def vendor_error_entry(url: str, *, status: Any = None, reason: Any = None, body: Any = None) -> dict[str, Any]:
    """Per-URL failure from a vendor's own error list. Names the origin status when the vendor
    reports one (body kept as content; ``reason`` is a vendor tag like ``http_error`` that adds
    nothing to the code); otherwise the vendor's body/reason is the error text."""
    if origin := origin_http_error(status):
        entry = {**page_error(url, origin), "content": str(body or "")}
    else:
        entry = page_error(url, str(body or reason or "extraction failed"))
    return {**entry, "metadata": {"sourceURL": url}}


# --- Keyless ring hand-off (shared by exa / parallel / keenable) ---------------
def keyless_search(display: str, name: str, query: str, limit: int, logger: logging.Logger) -> dict[str, Any]:
    from plugins.web.keyless_mcp import search_with_failover
    logger.info("%s keyless search: '%s' (limit=%d)", display, query, limit)
    return search_with_failover(name, query, limit)


def keyless_extract(display: str, name: str, urls: list[str], logger: logging.Logger) -> list[dict[str, Any]]:
    from plugins.web.keyless_mcp import extract_with_failover
    logger.info("%s keyless extract: %d URL(s)", display, len(urls))
    return extract_with_failover(name, list(urls))


# --- Guarded execution: interrupt check + uniform failure classification ---
def _failure_message(vendor: str, kind: str, exc: Exception, logger: logging.Logger, *, sdk: bool, verbatim_value_error: bool) -> str:
    """``verbatim_value_error``: ValueError carries a pre-formatted message (missing key,
    HTTP body) and is returned as-is. ``sdk``: ImportError means the lazily installed
    vendor SDK is missing. Anything else is logged and wrapped."""
    if verbatim_value_error and isinstance(exc, ValueError):
        return str(exc)
    if sdk and isinstance(exc, ImportError):
        return f"{vendor} SDK not installed: {exc}"
    logger.warning("%s %s error: %s", vendor, kind, exc)
    return f"{vendor} {kind} failed: {exc}"


def _guarded(vendor: str, kind: str, logger: logging.Logger, body: Callable[[], Any], interrupted: Any, fail: Callable[[str], Any], sdk: bool, vve: bool) -> Any:
    try:
        if _interrupted():
            return interrupted
        return body()
    except Exception as exc:
        return fail(_failure_message(vendor, kind, exc, logger, sdk=sdk, verbatim_value_error=vve))


def run_search(vendor: str, logger: logging.Logger, body: Callable[[], dict[str, Any]], *, sdk: bool = False, verbatim_value_error: bool = True) -> dict[str, Any]:
    return _guarded(vendor, "search", logger, body, search_fail("Interrupted"), search_fail, sdk, verbatim_value_error)


def _extract_interrupted(urls: list[str]) -> list[dict[str, Any]]:
    return [{"url": u, "error": "Interrupted", "title": ""} for u in urls]


def run_extract(
    vendor: str, logger: logging.Logger, urls: list[str], body: Callable[[], list[dict[str, Any]]],
    *, sdk: bool = False, verbatim_value_error: bool = True,
) -> list[dict[str, Any]]:
    """Per-URL failures are returned as entries with ``error`` — never raised."""
    return _guarded(vendor, "extract", logger, body, _extract_interrupted(urls), lambda m: extract_fail(urls, m), sdk, verbatim_value_error)


async def run_extract_async(
    vendor: str, logger: logging.Logger, urls: list[str], body: Callable[[], Awaitable[list[dict[str, Any]]]],
    *, sdk: bool = False, verbatim_value_error: bool = True,
) -> list[dict[str, Any]]:
    """Async twin of :func:`run_extract` (``body`` is awaited inside the guard)."""
    try:
        if _interrupted():
            return _extract_interrupted(urls)
        return await body()
    except Exception as exc:
        return extract_fail(urls, _failure_message(vendor, "extract", exc, logger, sdk=sdk, verbatim_value_error=verbatim_value_error))


# --- HTTP + SDK client helpers ---
def http_status_detail(response: Any) -> str:
    """Response body text for a >=400 reply, or ``HTTP <code>`` when the body is empty."""
    return (response.text or "").strip() or f"HTTP {response.status_code}"


def http_get_json(
    label: str, url: str, *, params: dict[str, Any], headers: dict[str, str], timeout: int,
    logger: logging.Logger, reach_target: Optional[str] = None,
) -> tuple[Any, Optional[dict[str, Any]]]:
    """GET ``url`` and parse JSON → ``(data, None)`` or ``(None, failure_dict)``.
    ``reach_target`` overrides the "Could not reach ..." subject (SearXNG includes its URL)."""
    try:
        resp = httpx.get(url, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.warning("%s HTTP error: %s", label, exc)
        return None, search_fail(f"{label} returned HTTP {exc.response.status_code}")
    except httpx.RequestError as exc:
        logger.warning("%s request error: %s", label, exc)
        return None, search_fail(f"Could not reach {reach_target or label}: {exc}")
    try:
        return resp.json(), None
    except Exception as exc:
        logger.warning("%s response parse error: %s", label, exc)
        return None, search_fail(f"Could not parse {label} response as JSON")


def titled_rows(raw_results: list[dict[str, Any]], description_key: str) -> list[dict[str, Any]]:
    """Brave/SearXNG row normalizer: ``str()`` every field, 1-based positions."""
    return [
        title_hit(str(r.get("title", "")), str(r.get("url", "")), str(r.get(description_key, "")), i + 1)
        for i, r in enumerate(raw_results)
    ]


def lazy_ensure(extra: str) -> None:
    """Best-effort ``pm.ensure_import``: a missing-but-unavailable extra raises
    ``pm.InstallError``, re-raised as ImportError for the caller's own handling."""
    import pm

    try:
        pm.ensure_import(extra)
    except Exception as exc:
        raise ImportError(str(exc))


def cached_sdk_client(slot: str, env_var: str, missing_key_error: str, feature: str, factory: Callable[[str], Any]) -> Any:
    """Lazy-build + cache a vendor SDK client on ``tools.web_tools.<slot>`` (so tests that
    reset ``tools.web_tools._<vendor>_client = None`` see fresh state). Raises ValueError
    when the key is unset."""
    import tools.web_tools as _wt
    # Resolved before the cache is consulted: the slot is one per process, but the key can change
    # under it (``/reload``, or each multiplexed profile's secret scope resolving its own key).
    api_key = provider_env(env_var)
    if not api_key:
        raise ValueError(missing_key_error)
    # (key, client) as one value so concurrent builds under different keys can never leave one
    # key recorded beside another key's client.
    cached = getattr(_wt, slot, None)
    if cached is not None and cached[0] == api_key:
        return cached[1]
    lazy_ensure(feature)
    client = factory(api_key)
    setattr(_wt, slot, (api_key, client))
    return client


# --- Provider base ---
class BaseWebSearchProvider(WebSearchProvider):
    """Subclasses set ``NAME`` / ``DISPLAY_NAME`` / ``KEY_ENV`` and flip ``EXTRACT`` / ``KEYLESS``.
    ``is_available`` deliberately ignores the keyless tier: otherwise the legacy preference walk
    would route users holding a key for a lower-priority backend onto this vendor's free tier.
    ``is_keyless_available`` is True for keyless vendors unless pinned ``web.provider_tier.<name>: paid``."""

    NAME: str = ""
    DISPLAY_NAME: str = ""
    KEY_ENV: str = ""
    EXTRACT: bool = False
    KEYLESS: bool = False

    name = property(lambda self: self.NAME)
    display_name = property(lambda self: self.DISPLAY_NAME)

    def is_available(self) -> bool:
        return bool(provider_env(self.KEY_ENV))

    def is_keyless_available(self) -> bool:
        from plugins.web.keyless_mcp import keyless_enabled, provider_tier
        return self.KEYLESS and keyless_enabled() and provider_tier(self.NAME) != "paid"

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return self.EXTRACT


def setup_schema(name: str, badge: str, tag: str, key_env: str = "", prompt: str = "", url: str = "", **extra: Any) -> dict[str, Any]:
    """``hermes tools`` picker entry; ``env_vars`` is empty when ``key_env`` is blank."""
    env_vars = [{"key": key_env, "prompt": prompt, "url": url}] if key_env else []
    return {"name": name, "badge": badge, "tag": tag, "env_vars": env_vars, **extra}


def keyless_variant_schema(display: str, key_env: str, key_url: str, *, free_tag: str, paid_tag: str) -> dict[str, Any]:
    """Picker entry for a keyless-ring vendor with a paid variant."""
    paid = setup_schema(f"{display} · Paid (API key)", "paid", paid_tag, key_env, f"{display} API key", key_url, web_tier="paid")
    return setup_schema(f"{display} · Free (keyless)", "free · no key", free_tag, web_tier="free", variants=[paid])
