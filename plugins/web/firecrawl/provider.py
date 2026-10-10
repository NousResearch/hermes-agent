"""Firecrawl web search + extract provider (direct SDK, keyless cloud, or Nous tool-gateway).

Config: ``web.backend`` / ``web.search_backend`` / ``web.extract_backend: firecrawl``.
Env: FIRECRAWL_API_KEY, FIRECRAWL_API_URL (self-hosted), FIRECRAWL_GATEWAY_URL / TOOL_GATEWAY_*.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

import httpx
from pydantic import ValidationError

from plugins.web._common import BaseWebSearchProvider, keyless_extract, keyless_search, search_fail, search_ok, setup_schema
from agent.web_acquisition_errors import (
    WebAcquisitionError, WebCancelledError, WebPageNotFoundError, WebPrivateAddressDeniedError,
    WebInvalidResponseError, WebPaymentRequiredError, WebTimeoutError, WebUnclassifiedFailure, WebWebsitePolicyDeniedError, WebWebsiteUnsupportedError, acquisition_error,
)
from tools import managed_tool_gateway as _gateway
from tools import tool_backend_helpers as _backend_helpers
from tools.url_safety import is_safe_url
# Module-level (cheap import) so tests can monkeypatch the policy gate on this module.
from tools.website_policy import check_website_access

logger = logging.getLogger(__name__)


def _firecrawl_failure(error: BaseException) -> WebAcquisitionError | None:
    if isinstance(error, ValidationError):
        return WebInvalidResponseError(cause=error)
    try:
        from firecrawl.v2.utils.error_handler import PaymentRequiredError, WebsiteNotSupportedError
    except ImportError:
        return None
    if isinstance(error, PaymentRequiredError):
        return WebPaymentRequiredError(error.status_code)
    if isinstance(error, WebsiteNotSupportedError):
        return WebWebsiteUnsupportedError(error.status_code)
    return None

_FIRECRAWL_CLOUD_API_URL = "https://api.firecrawl.dev"

# The SDK costs ~200ms of imports on a cold CLI; defer to first use (tests patch ``Firecrawl`` here).
_FIRECRAWL_CLS_CACHE: Optional[type] = None


def _load_firecrawl_cls() -> type:
    """Import and cache ``firecrawl.Firecrawl`` (lazy_deps install hint → ImportError)."""
    global _FIRECRAWL_CLS_CACHE
    if _FIRECRAWL_CLS_CACHE is None:
        try:
            from pm import ensure_import as _lazy_ensure

            _lazy_ensure("firecrawl")
        except ImportError:
            pass
        except Exception as exc:
            raise ImportError(str(exc))
        from firecrawl import Firecrawl as _cls  # noqa: WPS433 — deliberately lazy
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
    """Direct Firecrawl ``(mode, kwargs, cache_key)`` or None. ``mode`` is ``"sdk"`` (keyed / self-hosted) or
    ``"keyless"`` (explicit selection + no credentials → anonymous public cloud; the explicit selection is
    required so an unconfigured install never silently routes to it)."""
    api_key = _env("FIRECRAWL_API_KEY")
    api_url = _env("FIRECRAWL_API_URL").rstrip("/")
    if api_key or api_url:
        return "sdk", {k: v for k, v in (("api_key", api_key), ("api_url", api_url)) if v}, ("direct", api_url or None, api_key or None)
    if _is_explicit_firecrawl_selection():
        return "keyless", {"api_url": _FIRECRAWL_CLOUD_API_URL}, ("direct-keyless", _FIRECRAWL_CLOUD_API_URL, None)
    return None


def _is_explicit_firecrawl_selection() -> bool:
    from plugins.web.keyless_mcp import _web_config_selects
    return _web_config_selects("firecrawl")


def _use_keyless_ring(capability: Optional[str] = None) -> bool:
    """Route via the keyless ring only with no direct credentials, when the managed Nous
    gateway isn't the selected path for *capability*, and the keyless tier isn't disabled or pinned paid."""
    if _env("FIRECRAWL_API_KEY") or _env("FIRECRAWL_API_URL"):
        return False
    from tools.tool_backend_helpers import NOUS_MANAGED_PROVIDER, read_web_capability_selection
    from plugins.web.keyless_mcp import use_keyless
    # Both probes are optional layers: a failing probe never blocks the ring.
    for probe in (lambda: read_web_capability_selection(capability) == NOUS_MANAGED_PROVIDER,
                  lambda: _is_tool_gateway_ready() and not _is_explicit_firecrawl_selection()):
        try:
            if probe():
                return False
        except Exception:
            pass
    return use_keyless("firecrawl", "")


class _KeylessFirecrawlClient:
    """Minimal REST client for Firecrawl's keyless cloud mode; duck-types the SDK's
    ``search`` / ``scrape`` and never sends an Authorization header."""

    def __init__(self, api_url: str = _FIRECRAWL_CLOUD_API_URL):
        self.api_url = api_url.rstrip("/")

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = httpx.post(f"{self.api_url}{path}", json=payload, headers={"Content-Type": "application/json"}, timeout=60.0)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise httpx.HTTPStatusError(
                f"HTTP {response.status_code}: {response.text.strip()[:300]}",
                request=exc.request, response=response,
            ) from exc
        result = response.json()
        if not isinstance(result, dict) or type(result.get("success")) is not bool:
            raise WebInvalidResponseError()
        if not result["success"]:
            raise WebUnclassifiedFailure()
        return result

    search = lambda self, *, query, limit=5: self._post("/v2/search", {"query": query, "limit": limit})
    def scrape(self, *, url, formats, timeout=None):
        # _scrape_one passes the SDK's server-side ``timeout`` (ms); the v2 REST payload takes the same field.
        payload = {"url": url, "formats": formats}
        if timeout is not None:
            payload["timeout"] = timeout
        return self._post("/v2/scrape", payload)


def _get_firecrawl_gateway_url() -> str:
    return _gateway.build_vendor_gateway_url("firecrawl")


def _is_tool_gateway_ready() -> bool:
    """True when gateway URL + Nous Subscriber token are available."""
    return _gateway.resolve_managed_tool_gateway("firecrawl", token_reader=_gateway.peek_nous_access_token) is not None


def check_firecrawl_api_key() -> bool:
    """True when the route selected via ``hermes tools`` (or, on a never-configured
    install, either route) is usable."""
    from tools.tool_backend_helpers import NOUS_MANAGED_PROVIDER, read_web_capability_selection

    def _usable(selected) -> bool:
        if selected == NOUS_MANAGED_PROVIDER:
            return _is_tool_gateway_ready()
        return _get_direct_firecrawl_config() is not None or (selected is None and _is_tool_gateway_ready())

    # Search and extract can each pick their route (own key vs gateway); usable if either one is.
    return any(_usable(read_web_capability_selection(cap)) for cap in ("search", "extract"))


def is_managed_route(capability: Optional[str] = None) -> bool:
    """True when Firecrawl calls for *capability* go through the Nous Tool Gateway rather than the user's
    own key / instance — the same decision :func:`_get_firecrawl_client` makes, without building a client."""
    from tools.tool_backend_helpers import NOUS_MANAGED_PROVIDER, read_web_capability_selection
    selected = read_web_capability_selection(capability)
    if selected is not None or _is_explicit_firecrawl_selection():
        return selected == NOUS_MANAGED_PROVIDER
    return _get_direct_firecrawl_config() is None and _is_tool_gateway_ready()


def _firecrawl_backend_help_suffix() -> str:
    return ", or use the Nous Tool Gateway via your subscription (FIRECRAWL_GATEWAY_URL or TOOL_GATEWAY_DOMAIN)" if _backend_helpers.managed_nous_tools_enabled() else ""


def _get_firecrawl_client(capability: Optional[str] = None) -> Any:
    """Get or create the cached Firecrawl client. Strict selection semantics on the stored selection for
    *capability* (``web.<capability>_backend``, else the shared ``web`` selection): ``"nous"`` → managed Tool
    Gateway ONLY; any other stored backend → direct Firecrawl ONLY (never a silent managed fallback billed to
    Nous); never-configured → direct when present, else managed. Raises ValueError when the resolved path is
    unusable."""
    wt = _wt()
    from tools.tool_backend_helpers import NOUS_MANAGED_PROVIDER, read_web_capability_selection, selection_error
    selected = read_web_capability_selection(capability)
    direct_config = _get_direct_firecrawl_config()

    def _managed():
        gw = _gateway.resolve_managed_tool_gateway("firecrawl", token_reader=_gateway.read_nous_access_token)
        if gw is None:
            return None
        return "sdk", {"api_key": gw.nous_user_token, "api_url": gw.gateway_origin}, ("tool-gateway", gw.gateway_origin, gw.nous_user_token)

    def _unconfigured_message() -> str:
        message = "Web tools are not configured. Set FIRECRAWL_API_KEY for cloud Firecrawl or set FIRECRAWL_API_URL for a self-hosted Firecrawl instance."
        if _backend_helpers.managed_nous_tools_enabled():
            return message + " With your Nous subscription you can also use the Tool Gateway. run `hermes tools` and select Nous Subscription as the web provider."
        return message + " " + _backend_helpers.nous_tool_gateway_unavailable_message("managed Firecrawl web tools")

    # (resolved config, log detail, error message) per selection state; the message is built lazily.
    if selected == NOUS_MANAGED_PROVIDER:
        resolved, log, message = _managed(), "the Nous Subscription web selection is stored but the tool gateway is unavailable.", lambda: selection_error(
            "web", NOUS_MANAGED_PROVIDER, "the Nous Tool Gateway is not available (not entitled or unreachable)")
    elif selected is not None or _is_explicit_firecrawl_selection():
        # Stored vendor selection (shared name, or a per-capability key naming firecrawl): direct only (no
        # credentials → explicit selection unlocks keyless cloud mode). A per-capability key naming ANOTHER
        # vendor is not a firecrawl selection: the other capability's ladder reached firecrawl on its own,
        # so it resolves exactly like a never-configured install (#113017).
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
    if cached is not None and cached[0] == client_config:
        return cached[1]
    client = _KeylessFirecrawlClient(api_url=kwargs["api_url"]) if client_mode == "keyless" else Firecrawl(**kwargs)
    wt._firecrawl_client = (client_config, client)
    return client


# --- Response shape normalization (SDK / direct / gateway differ) --------------
def _to_plain_object(value: Any) -> Any:
    """SDK objects (pydantic ``model_dump`` / ``__dict__``) → plain data when possible."""
    if value is None or isinstance(value, (dict, list, str, int, float, bool)):
        return value
    for attr, convert in (("model_dump", lambda v: v.model_dump()), ("__dict__", lambda v: {k: x for k, x in v.__dict__.items() if not k.startswith("_")})):
        if hasattr(value, attr):
            try:
                return convert(value)
            except Exception:
                pass
    return value


def _normalize_result_list(values: Any) -> list[dict[str, Any]]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise WebInvalidResponseError()
    results = []
    for value in values:
        plain = _to_plain_object(value)
        if not isinstance(plain, dict):
            raise WebInvalidResponseError()
        fields = {key: "" if plain.get(key) is None else plain[key] for key in ("title", "description")}
        results.append({**plain, **fields})
    return results


def _extract_web_search_results(response: Any) -> list[dict[str, Any]]:
    """Search results across SDK/direct/gateway response shapes."""
    plain = _to_plain_object(response)
    if isinstance(response, dict):
        data = plain.get("data")
        if isinstance(data, list):
            return _normalize_result_list(data)
        containers = [data, plain] if isinstance(data, dict) else [plain]
        found = False
        for container in containers:
            for key in ("web", "results"):
                if key not in container:
                    continue
                found = True
                if container[key] is None:
                    raise WebInvalidResponseError()
                normalized = _normalize_result_list(container[key])
                if normalized:
                    return normalized
        if found:
            return []
        raise WebInvalidResponseError()
    if hasattr(response, "web"):
        return _normalize_result_list(response.web)
    raise WebInvalidResponseError()


def _extract_scrape_payload(scrape_result: Any) -> dict[str, Any]:
    plain = _to_plain_object(scrape_result)
    if not isinstance(plain, dict):
        return {}
    return plain["data"] if isinstance(plain.get("data"), dict) else plain


def _error_entry(url: str, error: str, *, title: str = "", raw: bool = False, blocked: Optional[dict[str, Any]] = None, failure: Optional[WebAcquisitionError] = None, requested_url: Optional[str] = None) -> dict[str, Any]:
    """Per-URL extract failure. ``raw`` adds ``raw_content`` (post-scrape failures carry
    it, pre-scrape ones don't); ``blocked`` adds ``blocked_by_policy``."""
    policy = {"blocked_by_policy": {k: blocked[k] for k in ("host", "rule", "source")}} if blocked else {}
    result = {"url": url, "title": title, "content": "", **({"raw_content": ""} if raw else {}), "error": error, **policy}
    if failure is not None:
        result["failure"] = failure.to_failure()
    if requested_url is not None:
        result["requested_url"] = requested_url
    return result


_SCRAPE_TIMEOUT_MSG = "Scrape timed out after 60s — page may be too large or unresponsive. Try browser_navigate instead."
_UNSAFE_REDIRECT_MSG = "Blocked: URL targets a private or internal network address"


async def _scrape_one(url: str, formats: list[str], format: Optional[str]) -> dict[str, Any]:
    """Scrape one URL (60s timeout) and re-check SSRF + website policy against the
    post-redirect URL. Never raises for scrape errors; returns an error entry instead."""
    if blocked := check_website_access(url):
        logger.info("Blocked web_extract for %s by rule %s", blocked["host"], blocked["rule"])
        return _error_entry(url, blocked["message"], blocked=blocked, failure=WebWebsitePolicyDeniedError())
    try:
        logger.info("Firecrawl scraping: %s", url)
        try:
            async with asyncio.timeout(60) as deadline:
                scrape_result = await asyncio.to_thread(
                    _get_firecrawl_client("extract").scrape,
                    url=url, formats=formats, timeout=60_000,
                )
        except TimeoutError as exc:
            if not deadline.expired():
                raise
            logger.warning("Firecrawl scrape timed out for %s", url)
            return _error_entry(url, _SCRAPE_TIMEOUT_MSG, failure=WebTimeoutError(60, cause=exc))
        payload = _extract_scrape_payload(scrape_result)
        metadata = payload.get("metadata", {})
        # SDK may return a typed object for metadata (raw __dict__ here, unlike _to_plain_object).
        if not isinstance(metadata, dict):
            metadata = metadata.model_dump() if hasattr(metadata, "model_dump") else getattr(metadata, "__dict__", {})
        title, final_url = metadata.get("title", ""), metadata.get("sourceURL", url)
        if not is_safe_url(final_url):
            logger.info("Blocked redirected web_extract for unsafe final URL: %s", final_url)
            return _error_entry(final_url, _UNSAFE_REDIRECT_MSG, title=title, raw=True, failure=WebPrivateAddressDeniedError(), requested_url=url)
        if final_blocked := check_website_access(final_url):
            logger.info("Blocked redirected web_extract for %s by rule %s", final_blocked["host"], final_blocked["rule"])
            return _error_entry(final_url, final_blocked["message"], title=title, raw=True, blocked=final_blocked, failure=WebWebsitePolicyDeniedError(), requested_url=url)
        if metadata.get("statusCode") in {404, 410}:
            failure = WebPageNotFoundError(metadata["statusCode"])
            return _error_entry(final_url, str(failure), title=title, raw=True, failure=failure, requested_url=url)
        markdown, html = payload.get("markdown"), payload.get("html")
        content = markdown if format == "markdown" or (format is None and markdown) else html or markdown or ""
        return {"url": final_url, "title": title, "content": content, "raw_content": content, "metadata": metadata, "requested_url": url, "coverage": "full"}
    except Exception as scrape_err:  # noqa: BLE001
        logger.debug("Firecrawl scrape failed for %s: %s", url, scrape_err)
        return _error_entry(url, str(scrape_err), raw=True, failure=acquisition_error(scrape_err, classify=_firecrawl_failure))


class FirecrawlWebSearchProvider(BaseWebSearchProvider):
    """Firecrawl search + extract provider with dual auth paths."""

    NAME = "firecrawl"
    DISPLAY_NAME = "Firecrawl"
    EXTRACT = True
    EXTRACT_FORMATS = ("markdown", "html")
    KEYLESS = True  # default-on ring member unless pinned ``paid``

    def is_available(self) -> bool:
        return check_firecrawl_api_key()

    def search(self, query: str, limit: int = 5) -> dict[str, Any]:
        """Pre-flight errors (ValueError / ImportError) propagate so the dispatcher emits
        the legacy ``tool_error`` envelope; in-flight errors become failure dicts."""
        from tools.interrupt import is_interrupted
        if is_interrupted():
            return search_fail("Interrupted", failure=WebCancelledError())
        if _use_keyless_ring("search"):
            return keyless_search("Firecrawl", "firecrawl", query, limit, logger)
        logger.info("Firecrawl search: '%s' (limit=%d)", query, limit)
        client = _get_firecrawl_client("search")
        try:
            web_results = _extract_web_search_results(client.search(query=query, limit=limit))
            logger.info("Firecrawl: found %d search results", len(web_results))
            return search_ok(web_results)
        except Exception as exc:
            logger.warning("Firecrawl search error: %s", exc)
            return search_fail(f"Firecrawl search failed: {exc}", failure=acquisition_error(exc, classify=_firecrawl_failure))

    async def extract(self, urls: list[str], **kwargs: Any) -> list[dict[str, Any]]:
        """Per-URL scrape; failures become items with an ``error`` field.
        ``format``: "markdown" | "html" | both (markdown preferred)."""
        from tools.interrupt import is_interrupted as _is_interrupted
        if _is_interrupted():
            return [{"url": u, "error": "Interrupted", "title": "", "failure": WebCancelledError().to_failure()} for u in urls]
        if _use_keyless_ring("extract"):
            return await asyncio.to_thread(keyless_extract, "Firecrawl", "firecrawl", urls, logger, **kwargs)
        format = kwargs.get("format")
        formats = [format] if format in ("markdown", "html") else ["markdown", "html"]
        return [
            {"url": url, "error": "Interrupted", "title": ""} if _is_interrupted() else await _scrape_one(url, formats, format)
            for url in urls
        ]


    def get_setup_schema(self) -> dict[str, Any]:
        return setup_schema(
            "Firecrawl", "keyless/paid · optional gateway",
            "Full search + extract; supports keyless cloud, direct API, and Nous tool-gateway routing.",
            "FIRECRAWL_API_KEY", "Firecrawl API key (optional; blank = keyless cloud or self-hosted)", "https://docs.firecrawl.dev/introduction",
        )
