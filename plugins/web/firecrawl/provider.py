"""Firecrawl web search + extract provider (direct SDK, keyless cloud, or Nous tool-gateway).

Config: ``web.backend`` / ``web.search_backend`` / ``web.extract_backend: firecrawl``.
Env: FIRECRAWL_API_KEY, FIRECRAWL_API_URL (self-hosted), FIRECRAWL_GATEWAY_URL / TOOL_GATEWAY_*.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, NoReturn, Optional, TYPE_CHECKING

import httpx

import httpx

from plugins.web._common import BaseWebSearchProvider, keyless_extract, keyless_search, lazy_ensure, search_fail, search_ok, setup_schema
from tools import managed_tool_gateway as _gateway
from tools import tool_backend_helpers as _backend_helpers
from tools.url_safety import is_safe_url
# Module-level (cheap import) so tests can monkeypatch the policy gate on this module.
from tools.website_policy import check_website_access

logger = logging.getLogger(__name__)

_FIRECRAWL_CLOUD_API_URL = "https://api.firecrawl.dev"


# The SDK costs ~200ms of imports on a cold CLI; defer to first use (tests patch ``Firecrawl`` here).
_FIRECRAWL_CLS_CACHE: Optional[type] = None


def _load_firecrawl_cls() -> type:
    """Import and cache ``firecrawl.Firecrawl`` (lazy_deps install hint → ImportError)."""
    global _FIRECRAWL_CLS_CACHE
    if _FIRECRAWL_CLS_CACHE is None:
        lazy_ensure("search.firecrawl")
        from firecrawl import Firecrawl as _cls
        _FIRECRAWL_CLS_CACHE = _cls
    return _FIRECRAWL_CLS_CACHE


class _FirecrawlProxy:
    """Callable proxy that looks like ``firecrawl.Firecrawl`` but imports lazily."""

    __slots__ = ()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return _load_firecrawl_cls()(*args, **kwargs)

    def __instancecheck__(self, obj: Any) -> bool:
        return isinstance(obj, _load_firecrawl_cls())

    def __repr__(self) -> str:
        return "<lazy firecrawl.Firecrawl proxy>"


Firecrawl = _FirecrawlProxy()

# --- Client construction (direct vs managed-gateway) ---------------------------
def _wt():
    """Client cache slots live on tools.web_tools so tests that reset ``_firecrawl_client`` there see it."""
    import tools.web_tools as _mod
    return _mod


def _env(name: str) -> str:
    from hermes_cli.config import get_env_value
    return (get_env_value(name) or "").strip()


def _get_direct_firecrawl_config() -> Optional[tuple]:
    """Return direct Firecrawl (mode, kwargs, cache key), or None when unavailable.

    ``mode`` is ``"sdk"`` (keyed / self-hosted via the Firecrawl SDK) or
    ``"keyless"`` (explicit Firecrawl selection with no credentials — served
    by :class:`_KeylessFirecrawlClient` against the public cloud API, which
    accepts anonymous rate-limited requests). Keyless requires the explicit
    selection so an unconfigured install never silently routes to it.
    """
    from hermes_cli.config import get_env_value


    if not api_key and not api_url:
        if _is_explicit_firecrawl_selection():
            return (
                "keyless",
                {"api_url": _FIRECRAWL_CLOUD_API_URL},
                ("direct-keyless", _FIRECRAWL_CLOUD_API_URL, None),
            )
        return None


    return "sdk", kwargs, ("direct", api_url or None, api_key or None)


def _is_explicit_firecrawl_selection() -> bool:
    """Return True when config explicitly selects Firecrawl for web tools."""
    import tools.web_tools as _wt

    cfg = _wt._load_web_config()
    return any(
        (cfg.get(key) or "").lower().strip() == "firecrawl"
        for key in ("backend", "search_backend", "extract_backend")
    )


def _use_keyless_ring() -> bool:
    """True when Firecrawl calls should route via the keyless ring.

    Ring dispatch applies when there are no direct credentials, the
    managed Nous gateway isn't the selected path, and the keyless tier
    isn't disabled or pinned paid. Keyed/self-hosted/gateway setups never
    reach the ring.
    """
    from hermes_cli.config import get_env_value

    if (get_env_value("FIRECRAWL_API_KEY") or "").strip():
        return False
    if (get_env_value("FIRECRAWL_API_URL") or "").strip():
        return False
    import tools.web_tools as _wt
    from tools.tool_backend_helpers import NOUS_MANAGED_PROVIDER, read_selection

    try:
        if read_selection("web") == NOUS_MANAGED_PROVIDER:
            return False
    except Exception:  # noqa: BLE001 — selection helpers optional
        pass
    try:
        if _wt._is_tool_gateway_ready() and not _is_explicit_firecrawl_selection():
            return False
    except Exception:  # noqa: BLE001 — probe optional
        pass
    from plugins.web.keyless_mcp import use_keyless

    return use_keyless("firecrawl", "")


class _KeylessFirecrawlClient:
    """Minimal REST client for Firecrawl's keyless cloud mode.

    Duck-types the two SDK methods the provider calls (``search`` /
    ``scrape``) so the rest of the pipeline (result normalizers, caching)
    is unchanged. No Authorization header is ever sent.
    """

    def __init__(self, api_url: str = _FIRECRAWL_CLOUD_API_URL):
        self.api_url = api_url.rstrip("/")

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = httpx.post(
            f"{self.api_url}{path}",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()

    def search(self, *, query: str, limit: int = 5) -> Dict[str, Any]:
        return self._post("/v2/search", {"query": query, "limit": limit})

    def scrape(self, *, url: str, formats: List[str]) -> Dict[str, Any]:
        return self._post("/v2/scrape", {"url": url, "formats": formats})


def _get_firecrawl_gateway_url() -> str:
    return _gateway.build_vendor_gateway_url("firecrawl")


def _is_tool_gateway_ready() -> bool:
    """True when gateway URL + Nous Subscriber token are available."""
    return _gateway.resolve_managed_tool_gateway("firecrawl", token_reader=_gateway.peek_nous_access_token) is not None


def check_firecrawl_api_key() -> bool:
    """Return True when the Firecrawl backend selected via `hermes tools`
    (or, on a never-configured install, either route) is usable.

    Re-exported by :mod:`tools.web_tools` for backward compatibility with
    existing tests and the ``hermes tools`` setup flow.
    """
    from tools.tool_backend_helpers import (
        NOUS_MANAGED_PROVIDER,
        read_selection,
    )

    selected = read_selection("web")
    if selected == NOUS_MANAGED_PROVIDER:
        return _is_tool_gateway_ready()
    if selected is not None:
        return _has_direct_firecrawl_config()
    return _has_direct_firecrawl_config() or _is_tool_gateway_ready()


def _firecrawl_backend_help_suffix() -> str:
    """Return optional managed-gateway guidance for Firecrawl help text."""
    import tools.web_tools as _wt

    if not _wt.managed_nous_tools_enabled():
        return ""
    return (
        ", or use the Nous Tool Gateway via your subscription "
        "(FIRECRAWL_GATEWAY_URL or TOOL_GATEWAY_DOMAIN)"
    )


def _raise_web_backend_configuration_error() -> "NoReturn":
    """Raise a clear error for unsupported web backend configuration."""
    import tools.web_tools as _wt

    message = (
        "Web tools are not configured. "
        "Set FIRECRAWL_API_KEY for cloud Firecrawl or set FIRECRAWL_API_URL "
        "for a self-hosted Firecrawl instance."
    )
    if _wt.managed_nous_tools_enabled():
        message += (
            " With your Nous subscription you can also use the Tool Gateway. "
            "run `hermes tools` and select Nous Subscription as the web provider."
        )
    else:
        message += " " + _wt.nous_tool_gateway_unavailable_message(
            "managed Firecrawl web tools",
        )
    raise ValueError(message)


def _get_firecrawl_client() -> Any:
    """Get or create the cached Firecrawl client.

    Strict selection semantics (switch on the stored ``web`` selection):
    - ``"nous"`` (or legacy ``use_gateway: true``) → managed Tool Gateway
      ONLY; unavailable is a selection-naming error (a present
      FIRECRAWL_API_KEY does not reroute).
    - any other stored web backend → direct Firecrawl ONLY; missing config
      is a selection-naming error — never a silent managed fallback billed
      to Nous.
    - never-configured web section → legacy behavior: direct config when
      present, else the managed gateway.

    Raises ValueError when the resolved path is unusable.

    The cached client is stored on :mod:`tools.web_tools` (as
    ``_firecrawl_client`` and ``_firecrawl_client_config``) rather than on
    this plugin module so that unit tests that reset the cache via
    ``tools.web_tools._firecrawl_client = None`` keep working. Helper
    functions (``resolve_managed_tool_gateway``, ``_read_nous_access_token``,
    ``Firecrawl``) are also looked up via :mod:`tools.web_tools` for the same
    reason — see :func:`_is_tool_gateway_ready`.
    """
    import tools.web_tools as _wt
    from tools.tool_backend_helpers import (
        NOUS_MANAGED_PROVIDER,
        read_selection,
        selection_error,
        selection_exists,
    )

    selected = read_selection("web")

    direct_config = _get_direct_firecrawl_config()

    def _managed_kwargs():
        managed_gateway = _wt.resolve_managed_tool_gateway(
            "firecrawl", token_reader=_wt._read_nous_access_token
        )
        if managed_gateway is None:
            return None
        kwargs = {
            "api_key": managed_gateway.nous_user_token,
            "api_url": managed_gateway.gateway_origin,
        }
        return kwargs, (
            "tool-gateway",
            kwargs["api_url"],
            managed_gateway.nous_user_token,
        )

    if selected == NOUS_MANAGED_PROVIDER:
        managed = _managed_kwargs()
        if managed is None:
            logger.error(
                "Firecrawl client initialization failed: the Nous "
                "Subscription web selection is stored but the tool gateway "
                "is unavailable."
            )
            raise ValueError(selection_error(
                "web",
                NOUS_MANAGED_PROVIDER,
                "the Nous Tool Gateway is not available (not entitled or "
                "unreachable)",
            ))
        kwargs, client_config = managed
        client_mode = "sdk"
    elif selected is not None or selection_exists("web"):
        # Stored vendor selection (or per-capability web keys routing to
        # firecrawl): direct Firecrawl only. With no credentials, the
        # explicit selection unlocks keyless cloud mode instead of erroring.
        if direct_config is None:
            logger.error(
                "Firecrawl client initialization failed: direct Firecrawl "
                "selected but FIRECRAWL_API_KEY/FIRECRAWL_API_URL is not set."
            )
            raise ValueError(selection_error(
                "web",
                selected or "firecrawl",
                "neither FIRECRAWL_API_KEY nor FIRECRAWL_API_URL is set",
            ))
        client_mode, kwargs, client_config = direct_config
    elif direct_config is not None:
        client_mode, kwargs, client_config = direct_config
    else:
        # Never-configured web section: legacy managed fallback.
        managed = _managed_kwargs()
        if managed is None:
            logger.error(
                "Firecrawl client initialization failed: "
                "missing direct config and tool-gateway auth."
            )
            _raise_web_backend_configuration_error()
        kwargs, client_config = managed
        client_mode = "sdk"

    def _unconfigured_message() -> str:
        message = "Web tools are not configured. Set FIRECRAWL_API_KEY for cloud Firecrawl or set FIRECRAWL_API_URL for a self-hosted Firecrawl instance."
        if _backend_helpers.managed_nous_tools_enabled():
            return message + " With your Nous subscription you can also use the Tool Gateway. run `hermes tools` and select Nous Subscription as the web provider."
        return message + " " + _backend_helpers.nous_tool_gateway_unavailable_message("managed Firecrawl web tools")

    # (resolved config, log detail, error message) per selection state; the message is built lazily.
    if selected == NOUS_MANAGED_PROVIDER:
        resolved, log, message = _managed(), "the Nous Subscription web selection is stored but the tool gateway is unavailable.", lambda: selection_error(
            "web", NOUS_MANAGED_PROVIDER, "the Nous Tool Gateway is not available (not entitled or unreachable)")
    elif selected is not None or selection_exists("web"):
        # Stored vendor selection: direct only (no credentials → explicit selection unlocks keyless cloud mode).
        resolved, log, message = direct_config, "direct Firecrawl selected but FIRECRAWL_API_KEY/FIRECRAWL_API_URL is not set.", lambda: selection_error(
            "web", selected or "firecrawl", "neither FIRECRAWL_API_KEY nor FIRECRAWL_API_URL is set")
    elif direct_config is not None:
        resolved = direct_config
    else:  # never-configured web section: legacy managed fallback
        resolved, log, message = _managed(), "missing direct config and tool-gateway auth.", _unconfigured_message
    if resolved is None:
        logger.error("Firecrawl client initialization failed: %s", log)
        raise ValueError(message())
    client_mode, kwargs, client_config = resolved
    cached = getattr(wt, "_firecrawl_client", None)
    if cached is not None and getattr(wt, "_firecrawl_client_config", None) == client_config:
        return cached

    # Construct via the re-exported Firecrawl proxy on tools.web_tools so
    # unit tests patching ``tools.web_tools.Firecrawl`` see their mock.
    if client_mode == "keyless":
        _wt._firecrawl_client = _KeylessFirecrawlClient(api_url=kwargs["api_url"])
    else:
        _wt._firecrawl_client = _wt.Firecrawl(**kwargs)
    _wt._firecrawl_client_config = client_config
    return _wt._firecrawl_client


def _reset_client_for_tests() -> None:
    """Drop the cached Firecrawl client so tests can re-instantiate cleanly.

    Clears the canonical slots on :mod:`tools.web_tools` (where
    :func:`_get_firecrawl_client` reads/writes them).
    """
    import tools.web_tools as _wt

    _wt._firecrawl_client = None
    _wt._firecrawl_client_config = None


# ---------------------------------------------------------------------------
# Response shape normalization (SDK / direct / gateway differ)
# ---------------------------------------------------------------------------


# --- Response shape normalization (SDK / direct / gateway differ) --------------
def _to_plain_object(value: Any) -> Any:
    """SDK objects (pydantic ``model_dump`` / ``__dict__``) → plain data when possible."""
    if value is None or isinstance(value, (dict, list, str, int, float, bool)):
        return value
    for attr, convert in (("model_dump", lambda v: v.model_dump()), ("__dict__", lambda v: {k: x for k, x in v.__dict__.items() if not k.startswith("_")})):
        if hasattr(value, attr):
            try:
                return convert(value)
            except Exception:  # noqa: BLE001
                pass
    return value


def _normalize_result_list(values: Any) -> List[Dict[str, Any]]:
    return [p for p in map(_to_plain_object, values) if isinstance(p, dict)] if isinstance(values, list) else []


def _extract_web_search_results(response: Any) -> List[Dict[str, Any]]:
    """Search results across SDK/direct/gateway response shapes."""
    plain = _to_plain_object(response)
    if isinstance(plain, dict):
        data = plain.get("data")
        if isinstance(data, list):
            return _normalize_result_list(data)
        candidates = [data.get("web"), data.get("results")] if isinstance(data, dict) else []
        for candidate in candidates + [plain.get("web"), plain.get("results")]:
            normalized = _normalize_result_list(candidate)
            if normalized:
                return normalized
    if hasattr(response, "web"):
        return _normalize_result_list(getattr(response, "web", []))
    return []


def _extract_scrape_payload(scrape_result: Any) -> Dict[str, Any]:
    plain = _to_plain_object(scrape_result)
    if not isinstance(plain, dict):
        return {}
    return plain["data"] if isinstance(plain.get("data"), dict) else plain


def _error_entry(url: str, error: str, *, title: str = "", raw: bool = False, blocked: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Per-URL extract failure. ``raw`` adds ``raw_content`` (post-scrape failures carry
    it, pre-scrape ones don't); ``blocked`` adds ``blocked_by_policy``."""
    policy = {"blocked_by_policy": {k: blocked[k] for k in ("host", "rule", "source")}} if blocked else {}
    return {"url": url, "title": title, "content": "", **({"raw_content": ""} if raw else {}), "error": error, **policy}


_SCRAPE_TIMEOUT_MSG = "Scrape timed out after 60s — page may be too large or unresponsive. Try browser_navigate instead."
_UNSAFE_REDIRECT_MSG = "Blocked: URL targets a private or internal network address"


async def _scrape_one(url: str, formats: List[str], format: Optional[str]) -> Dict[str, Any]:
    """Scrape one URL (60s timeout) and re-check SSRF + website policy against the
    post-redirect URL. Never raises for scrape errors; returns an error entry instead."""
    if blocked := check_website_access(url):
        logger.info("Blocked web_extract for %s by rule %s", blocked["host"], blocked["rule"])
        return _error_entry(url, blocked["message"], blocked=blocked)
    try:
        logger.info("Firecrawl scraping: %s", url)
        try:
            scrape_result = await asyncio.wait_for(asyncio.to_thread(_get_firecrawl_client().scrape, url=url, formats=formats), timeout=60)
        except asyncio.TimeoutError:
            logger.warning("Firecrawl scrape timed out for %s", url)
            return _error_entry(url, _SCRAPE_TIMEOUT_MSG)
        payload = _extract_scrape_payload(scrape_result)
        metadata = payload.get("metadata", {})
        # SDK may return a typed object for metadata (raw __dict__ here, unlike _to_plain_object).
        if not isinstance(metadata, dict):
            metadata = metadata.model_dump() if hasattr(metadata, "model_dump") else getattr(metadata, "__dict__", {})
        title, final_url = metadata.get("title", ""), metadata.get("sourceURL", url)
        if not is_safe_url(final_url):
            logger.info("Blocked redirected web_extract for unsafe final URL: %s", final_url)
            return _error_entry(final_url, _UNSAFE_REDIRECT_MSG, title=title, raw=True)
        if final_blocked := check_website_access(final_url):
            logger.info("Blocked redirected web_extract for %s by rule %s", final_blocked["host"], final_blocked["rule"])
            return _error_entry(final_url, final_blocked["message"], title=title, raw=True, blocked=final_blocked)
        markdown, html = payload.get("markdown"), payload.get("html")
        content = markdown if format == "markdown" or (format is None and markdown) else html or markdown or ""
        return {"url": final_url, "title": title, "content": content, "raw_content": content, "metadata": metadata}
    except Exception as scrape_err:  # noqa: BLE001
        logger.debug("Firecrawl scrape failed for %s: %s", url, scrape_err)
        return _error_entry(url, str(scrape_err), raw=True)


class FirecrawlWebSearchProvider(BaseWebSearchProvider):
    """Firecrawl search + extract provider with dual auth paths."""

    NAME = "firecrawl"
    DISPLAY_NAME = "Firecrawl"
    EXTRACT = True
    KEYLESS = True  # default-on ring member unless pinned ``paid``

    def is_available(self) -> bool:
        return check_firecrawl_api_key()

    def is_keyless_available(self) -> bool:
        """Firecrawl serves keyless cloud requests (public API, no auth).

        Default-on ring member of the keyless free tier: fresh installs
        rotate across Exa/Parallel/Tavily/Firecrawl/Keenable. False when
        the user pinned ``web.provider_tier.firecrawl: paid``.
        """
        from plugins.web.keyless_mcp import keyless_enabled, provider_tier

        return keyless_enabled() and provider_tier("firecrawl") != "paid"

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Pre-flight errors (ValueError / ImportError) propagate so the dispatcher emits
        the legacy ``tool_error`` envelope; in-flight errors become failure dicts."""
        from tools.interrupt import is_interrupted
        if is_interrupted():
            return {"success": False, "error": "Interrupted"}

        if _use_keyless_ring():
            # No credentials and no managed gateway: ring dispatch with
            # next-in-line failover on rate limits (default-on free tier).
            from plugins.web.keyless_mcp import search_with_failover

            logger.info(
                "Firecrawl keyless search: '%s' (limit=%d)", query, limit
            )
            return search_with_failover("firecrawl", query, limit)

        logger.info("Firecrawl search: '%s' (limit=%d)", query, limit)
        client = _get_firecrawl_client()
        try:
            web_results = _extract_web_search_results(client.search(query=query, limit=limit))
            logger.info("Firecrawl: found %d search results", len(web_results))
            return search_ok(web_results)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Firecrawl search error: %s", exc)
            return search_fail(f"Firecrawl search failed: {exc}")

    async def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Per-URL scrape; failures become items with an ``error`` field.
        ``format``: "markdown" | "html" | both (markdown preferred)."""
        from tools.interrupt import is_interrupted as _is_interrupted
        if _is_interrupted():
            return [{"url": u, "error": "Interrupted", "title": ""} for u in urls]

        if _use_keyless_ring():
            # No credentials and no managed gateway: ring dispatch with
            # next-in-line failover on rate limits (default-on free tier).
            import asyncio as _asyncio

            from plugins.web.keyless_mcp import extract_with_failover

            logger.info("Firecrawl keyless extract: %d URL(s)", len(urls))
            return await _asyncio.to_thread(
                extract_with_failover, "firecrawl", list(urls)
            )

        format = kwargs.get("format")
        formats = [format] if format in ("markdown", "html") else ["markdown", "html"]
        return [
            {"url": url, "error": "Interrupted", "title": ""} if _is_interrupted() else await _scrape_one(url, formats, format)
            for url in urls
        ]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Firecrawl",
            "badge": "keyless/paid · optional gateway",
            "tag": (
                "Full search + extract; supports keyless cloud, direct API, "
                "and Nous tool-gateway routing."
            ),
            "env_vars": [
                {
                    "key": "FIRECRAWL_API_KEY",
                    "prompt": "Firecrawl API key (optional; blank = keyless cloud or self-hosted)",
                    "url": "https://docs.firecrawl.dev/introduction",
                },
            ],
        }
