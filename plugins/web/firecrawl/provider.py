"""Firecrawl web search + extract — plugin form.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. This is
the largest provider migrated in this PR; it captures the full inline
firecrawl implementation that previously lived in tools/web_tools.py:

  - :data:`Firecrawl` lazy proxy that defers the ~200ms SDK import to
    first use (re-exported by tools.web_tools for backward compat with
    existing tests that mock that name).
  - :func:`_get_firecrawl_client` with direct + managed-gateway dual
    mode, controlled by ``web.use_gateway`` config when both are
    configured.
  - :func:`check_firecrawl_api_key` re-exported (tests + tools_config
    setup hint depend on this name living in tools.web_tools).
  - :func:`_extract_web_search_results` / :func:`_extract_scrape_payload`
    response-shape normalizers that handle SDK / direct API / gateway
    response variants.
  - Per-URL extract loop with 60s timeout, redirect-aware SSRF re-check,
    website-policy gating, and format-aware content selection.

Async note: the underlying SDK is sync. ``extract()`` is declared
``async def`` because it performs per-URL I/O that benefits from
running in an executor; the implementation wraps each scrape in
:func:`asyncio.to_thread` with :func:`asyncio.wait_for(timeout=60)` to
guard against hung fetches.

Config keys this provider responds to::

    web:
      search_backend: "firecrawl"     # explicit per-capability
      extract_backend: "firecrawl"    # explicit per-capability
      backend: "firecrawl"            # shared fallback (default)
      use_gateway: false              # prefer managed gateway when both
                                      # direct + gateway credentials exist

Env vars::

    FIRECRAWL_API_KEY=...            # direct cloud auth
    FIRECRAWL_API_URL=...            # self-hosted Firecrawl
    FIRECRAWL_GATEWAY_URL=...        # Nous tool-gateway (subscribers)
    TOOL_GATEWAY_DOMAIN=...          # alternate gateway env
    TOOL_GATEWAY_SCHEME=...
    TOOL_GATEWAY_USER_TOKEN=...
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from agent.web_search_provider import WebSearchProvider
from tools.url_safety import is_safe_url
from tools.website_policy import check_website_access

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy Firecrawl SDK proxy
# ---------------------------------------------------------------------------
# The firecrawl SDK pulls ~200ms of imports (httpcore, firecrawl.v1/v2 type
# trees) on a cold CLI. We only need it when the backend is actually
# "firecrawl", so defer the import to first use via a callable proxy.
#
# Tests that do ``patch("tools.web_tools.Firecrawl", ...)`` continue to
# work because tools/web_tools.py re-exports ``Firecrawl`` from this
# module — so the patched name still references the same proxy instance.

if TYPE_CHECKING:
    from firecrawl import Firecrawl as FirecrawlSDK  # noqa: F401 — type hints only

_FIRECRAWL_CLS_CACHE: Optional[type] = None


def _load_firecrawl_cls() -> type:
    """Import and cache ``firecrawl.Firecrawl``."""
    global _FIRECRAWL_CLS_CACHE
    if _FIRECRAWL_CLS_CACHE is None:
        try:
            from tools.lazy_deps import ensure as _lazy_ensure

            _lazy_ensure("search.firecrawl", prompt=False)
        except ImportError:
            pass
        except Exception as exc:  # noqa: BLE001 — surface install hint
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


# ---------------------------------------------------------------------------
# Client construction (direct vs managed-gateway)
# ---------------------------------------------------------------------------
#
# The canonical cache slots live on :mod:`tools.web_tools` so tests that do
# ``tools.web_tools._firecrawl_client = None`` between cases see fresh
# state. The plugin reads/writes through that public module — see
# :func:`_get_firecrawl_client` below.


def _get_direct_firecrawl_config() -> Optional[tuple]:
    """Return explicit direct Firecrawl kwargs + cache key, or None when unset."""
    from hermes_cli.config import get_env_value

    api_key = (get_env_value("FIRECRAWL_API_KEY") or "").strip()
    api_url = (get_env_value("FIRECRAWL_API_URL") or "").strip().rstrip("/")

    if not api_key and not api_url:
        return None

    kwargs: Dict[str, str] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if api_url:
        kwargs["api_url"] = api_url

    return kwargs, ("direct", api_url or None, api_key or None)


def _get_firecrawl_gateway_url() -> str:
    """Return the configured Firecrawl gateway URL."""
    import tools.web_tools as _wt

    return _wt.build_vendor_gateway_url("firecrawl")


def _is_tool_gateway_ready() -> bool:
    """Return True when gateway URL + Nous Subscriber token are available.

    Reads ``peek_nous_access_token`` and ``resolve_managed_tool_gateway``
    via :mod:`tools.web_tools` rather than direct imports, so unit tests
    that ``patch("tools.web_tools._peek_nous_access_token", ...)`` see
    their patches honored. The names are re-exported on
    :mod:`tools.web_tools` for exactly this reason.
    """
    import tools.web_tools as _wt

    return _wt.resolve_managed_tool_gateway(
        "firecrawl", token_reader=_wt._peek_nous_access_token
    ) is not None


def _has_direct_firecrawl_config() -> bool:
    """Return True when direct Firecrawl config is explicitly configured."""
    return _get_direct_firecrawl_config() is not None


def check_firecrawl_api_key() -> bool:
    """Return True when Firecrawl backend (direct or gateway) is usable.

    Re-exported by :mod:`tools.web_tools` for backward compatibility with
    existing tests and the ``hermes tools`` setup flow.
    """
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


def _raise_web_backend_configuration_error() -> None:
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

    When ``web.use_gateway`` is set in config, the managed Tool Gateway is
    preferred even if direct Firecrawl credentials are present. Otherwise
    direct Firecrawl takes precedence when explicitly configured.

    Raises ValueError when neither path is usable.

    The cached client is stored on :mod:`tools.web_tools` (as
    ``_firecrawl_client`` and ``_firecrawl_client_config``) rather than on
    this plugin module so that unit tests that reset the cache via
    ``tools.web_tools._firecrawl_client = None`` keep working. Helper
    functions (``prefers_gateway``, ``resolve_managed_tool_gateway``,
    ``_read_nous_access_token``, ``Firecrawl``) are also looked up via
    :mod:`tools.web_tools` for the same reason — see
    :func:`_is_tool_gateway_ready`.
    """
    import tools.web_tools as _wt

    direct_config = _get_direct_firecrawl_config()
    if direct_config is not None and not _wt.prefers_gateway("web"):
        kwargs, client_config = direct_config
    else:
        managed_gateway = _wt.resolve_managed_tool_gateway(
            "firecrawl", token_reader=_wt._read_nous_access_token
        )
        if managed_gateway is None:
            logger.error(
                "Firecrawl client initialization failed: "
                "missing direct config and tool-gateway auth."
            )
            _raise_web_backend_configuration_error()

        kwargs = {
            "api_key": managed_gateway.nous_user_token,
            "api_url": managed_gateway.gateway_origin,
        }
        client_config = (
            "tool-gateway",
            kwargs["api_url"],
            managed_gateway.nous_user_token,
        )

    cached = getattr(_wt, "_firecrawl_client", None)
    cached_config = getattr(_wt, "_firecrawl_client_config", None)
    if cached is not None and cached_config == client_config:
        return cached

    # Construct via the re-exported Firecrawl proxy on tools.web_tools so
    # unit tests patching ``tools.web_tools.Firecrawl`` see their mock.
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


def _to_plain_object(value: Any) -> Any:
    """Convert SDK objects to plain python data structures when possible."""
    if value is None:
        return None

    if isinstance(value, (dict, list, str, int, float, bool)):
        return value

    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:  # noqa: BLE001
            pass

    if hasattr(value, "__dict__"):
        try:
            return {k: v for k, v in value.__dict__.items() if not k.startswith("_")}
        except Exception:  # noqa: BLE001
            pass

    return value


def _normalize_result_list(values: Any) -> List[Dict[str, Any]]:
    """Normalize mixed SDK/list payloads into a list of dicts."""
    if not isinstance(values, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for item in values:
        plain = _to_plain_object(item)
        if isinstance(plain, dict):
            normalized.append(plain)
    return normalized


def _extract_web_search_results(response: Any) -> List[Dict[str, Any]]:
    """Extract Firecrawl search results across SDK/direct/gateway response shapes."""
    response_plain = _to_plain_object(response)

    if isinstance(response_plain, dict):
        data = response_plain.get("data")
        if isinstance(data, list):
            return _normalize_result_list(data)

        if isinstance(data, dict):
            data_web = _normalize_result_list(data.get("web"))
            if data_web:
                return data_web
            data_results = _normalize_result_list(data.get("results"))
            if data_results:
                return data_results

        top_web = _normalize_result_list(response_plain.get("web"))
        if top_web:
            return top_web

        top_results = _normalize_result_list(response_plain.get("results"))
        if top_results:
            return top_results

    if hasattr(response, "web"):
        return _normalize_result_list(getattr(response, "web", []))

    return []


def _extract_scrape_payload(scrape_result: Any) -> Dict[str, Any]:
    """Normalize Firecrawl scrape payload shape across SDK and gateway variants."""
    result_plain = _to_plain_object(scrape_result)
    if not isinstance(result_plain, dict):
        return {}

    nested = result_plain.get("data")
    if isinstance(nested, dict):
        return nested

    return result_plain


# ---------------------------------------------------------------------------
# Direct-HTTP fallback for Firecrawl 4xx responses (local patch 2026-07-22)
# ---------------------------------------------------------------------------
#
# Since 2026-07-05 the Firecrawl API intermittently returns HTTP 400 on
# plain URLs, which broke web_extract wholesale. When Firecrawl rejects a
# scrape with a 4xx, we fall back to a bounded direct HTTP fetch of the
# URL and mark the result with ``metadata.fallback = "direct-http"``.
# The fallback NEVER runs on a successful Firecrawl response, and never
# for non-4xx errors (5xx / timeouts keep their original error surface).

_FALLBACK_TIMEOUT_S = 15.0
_FALLBACK_MAX_BYTES = 2 * 1024 * 1024  # ~2MB cap on fetched content
_FALLBACK_MAX_REDIRECTS = 3
_FALLBACK_USER_AGENT = (
    "Mozilla/5.0 (compatible; Hermes-Agent web_extract direct-http fallback; "
    "+https://github.com/NousResearch/hermes-agent)"
)
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_HTTP_4XX_RE = re.compile(r"\b4\d{2}\b")


def _is_bad_request_error(exc: BaseException) -> bool:
    """Return True when an exception message indicates a Firecrawl 4xx.

    The SDK surfaces API errors as plain exceptions with the status code in
    the message (e.g. ``"400 Bad Request: ..."`` or ``"Bad Request: HTTP
    400"``), so detection is message-based. Deliberately excludes 5xx —
    server-side errors should keep their original error surface.
    """
    return bool(_HTTP_4XX_RE.search(str(exc)))


_BLOCK_TAGS = {
    "p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "ul", "ol",
    "table", "tr", "td", "th", "section", "article", "header", "footer",
    "nav", "main", "aside", "blockquote", "pre", "br", "hr", "form",
}
_SKIP_TAGS = {"script", "style", "noscript", "template"}


class _HTMLTextExtractor(HTMLParser):
    """Minimal stdlib HTML→text: title + block-joined visible text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._title_parts: List[str] = []
        self._in_title = False
        self._skip_depth = 0
        self._blocks: List[str] = []
        self._current: List[str] = []

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in _BLOCK_TAGS:
            self._flush_block()

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag == "title":
            self._in_title = False
        elif tag in _BLOCK_TAGS:
            self._flush_block()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self._title_parts.append(data)
        else:
            self._current.append(data)

    def _flush_block(self) -> None:
        text = " ".join("".join(self._current).split())
        if text:
            self._blocks.append(text)
        self._current = []

    @property
    def title(self) -> str:
        return " ".join("".join(self._title_parts).split())

    @property
    def text(self) -> str:
        self._flush_block()
        return "\n\n".join(self._blocks)


def _direct_http_extract_fallback(url: str) -> Optional[Dict[str, Any]]:
    """Bounded direct HTTP fetch used when Firecrawl returns 4xx.

    Safety bounds: 15s timeout, ~2MB content cap, manual redirect
    following (max 3 hops) with every hop re-checked via
    :func:`is_safe_url` (which rejects non-http(s) schemes such as
    ``file://`` and private/internal IPs) and website-access policy.

    Returns a legacy-shaped result dict with
    ``metadata.fallback = "direct-http"``, or ``None`` when the direct
    fetch is unsafe, blocked, oversized/binary, or fails — in which case
    the caller keeps the original Firecrawl error.
    """
    import httpx

    if not is_safe_url(url):
        logger.info("Direct-HTTP fallback refused unsafe URL: %s", url)
        return None
    if check_website_access(url):
        return None

    try:
        with httpx.Client(
            timeout=_FALLBACK_TIMEOUT_S,
            follow_redirects=False,
            headers={"User-Agent": _FALLBACK_USER_AGENT},
        ) as client:
            current_url = url
            response = None
            for _ in range(_FALLBACK_MAX_REDIRECTS + 1):
                response = client.get(current_url)
                status = getattr(response, "status_code", 200)
                location = response.headers.get("location")
                if status in _REDIRECT_STATUSES and location:
                    next_url = str(httpx.URL(current_url).join(location))
                    if not is_safe_url(next_url) or check_website_access(next_url):
                        logger.info(
                            "Direct-HTTP fallback blocked redirect %s -> %s",
                            current_url,
                            next_url,
                        )
                        return None
                    current_url = next_url
                    continue
                break
            else:
                logger.info("Direct-HTTP fallback: too many redirects for %s", url)
                return None

            response.raise_for_status()

            content_length = response.headers.get("content-length")
            if content_length and content_length.isdigit():
                if int(content_length) > _FALLBACK_MAX_BYTES:
                    logger.info(
                        "Direct-HTTP fallback: %s exceeds %d-byte cap", url,
                        _FALLBACK_MAX_BYTES,
                    )
                    return None

            content_type = (response.headers.get("content-type") or "").lower()
            body = response.text
            if len(body) > _FALLBACK_MAX_BYTES:
                body = body[:_FALLBACK_MAX_BYTES]

            final_url = str(getattr(response, "url", current_url))

            if "html" in content_type:
                extractor = _HTMLTextExtractor()
                extractor.feed(body)
                title = extractor.title
                text = extractor.text
            elif content_type.startswith("text/") or "json" in content_type or "xml" in content_type:
                title = ""
                text = body.strip()
            else:
                logger.info(
                    "Direct-HTTP fallback: unsupported content-type %r for %s",
                    content_type,
                    url,
                )
                return None

            return {
                "url": final_url,
                "title": title,
                "content": text,
                "raw_content": text,
                "metadata": {
                    "sourceURL": final_url,
                    "title": title,
                    "fallback": "direct-http",
                },
            }
    except Exception as exc:  # noqa: BLE001 — fallback is best-effort
        logger.info("Direct-HTTP fallback failed for %s: %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------


class FirecrawlWebSearchProvider(WebSearchProvider):
    """Firecrawl search + extract provider with dual auth paths."""

    @property
    def name(self) -> str:
        return "firecrawl"

    @property
    def display_name(self) -> str:
        return "Firecrawl"

    def is_available(self) -> bool:
        """Return True when direct Firecrawl OR managed-gateway path is configured."""
        return check_firecrawl_api_key()

    def supports_search(self) -> bool:
        return True

    def supports_extract(self) -> bool:
        return True

    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Execute a Firecrawl search.

        Sync; matches the legacy ``_get_firecrawl_client().search(...)``
        call directly. Normalizes the response across SDK/direct/gateway
        shapes via :func:`_extract_web_search_results`.

        Pre-flight errors (``ValueError`` from configuration check,
        ``ImportError`` from missing SDK) propagate to the dispatcher's
        top-level handler, which wraps them as ``tool_error(...)`` —
        matching the legacy ``{"error": "Error searching web: ..."}``
        envelope. Only in-flight errors are caught and surfaced as
        ``{"success": False, "error": ...}``.
        """
        from tools.interrupt import is_interrupted

        if is_interrupted():
            return {"success": False, "error": "Interrupted"}

        logger.info("Firecrawl search: '%s' (limit=%d)", query, limit)
        # _get_firecrawl_client() raises ValueError on unconfigured systems —
        # let it propagate so the dispatcher emits the legacy envelope shape.
        client = _get_firecrawl_client()
        try:
            response = client.search(query=query, limit=limit)
            web_results = _extract_web_search_results(response)
            logger.info("Firecrawl: found %d search results", len(web_results))
            return {"success": True, "data": {"web": web_results}}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Firecrawl search error: %s", exc)
            message = f"Firecrawl search failed: {exc}"
            if _is_bad_request_error(exc):
                message += (
                    " — the Firecrawl API answered 400 Bad Request "
                    "(known-broken since 2026-07-05). The search endpoint "
                    "has no fallback; if you already know a relevant URL, "
                    "use web_extract on it instead — it falls back to a "
                    "direct HTTP fetch on 4xx."
                )
            return {"success": False, "error": message}

    async def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract content from one or more URLs via Firecrawl.

        Async; each URL is scraped in a background thread with a 60s
        timeout. After scraping, the final URL (post-redirect) is
        re-checked against website-access policy.

        Accepted kwargs (others ignored for forward compat):
          - ``format``: ``"markdown"`` or ``"html"``; default is both
            (request both, return markdown when available).

        Returns the legacy per-URL list-of-results shape. Per-URL failures
        (timeout, SSRF block, scrape error, policy block) become items
        with an ``error`` field rather than raising.
        """
        from tools.interrupt import is_interrupted as _is_interrupted

        if _is_interrupted():
            return [{"url": u, "error": "Interrupted", "title": ""} for u in urls]

        format = kwargs.get("format")
        formats: List[str] = []
        if format == "markdown":
            formats = ["markdown"]
        elif format == "html":
            formats = ["html"]
        else:
            formats = ["markdown", "html"]

        # check_website_access is the legacy policy gate; imported at
        # module level (lazy-friendly because the website_policy import is
        # cheap) so monkeypatching it in tests works as expected.

        results: List[Dict[str, Any]] = []

        for url in urls:
            if _is_interrupted():
                results.append({"url": url, "error": "Interrupted", "title": ""})
                continue

            # Pre-scrape website policy gate
            blocked = check_website_access(url)
            if blocked:
                logger.info(
                    "Blocked web_extract for %s by rule %s",
                    blocked["host"],
                    blocked["rule"],
                )
                results.append(
                    {
                        "url": url,
                        "title": "",
                        "content": "",
                        "error": blocked["message"],
                        "blocked_by_policy": {
                            "host": blocked["host"],
                            "rule": blocked["rule"],
                            "source": blocked["source"],
                        },
                    }
                )
                continue

            try:
                logger.info("Firecrawl scraping: %s", url)
                try:
                    scrape_result = await asyncio.wait_for(
                        asyncio.to_thread(
                            _get_firecrawl_client().scrape,
                            url=url,
                            formats=formats,
                        ),
                        timeout=60,
                    )
                except asyncio.TimeoutError:
                    logger.warning("Firecrawl scrape timed out for %s", url)
                    results.append(
                        {
                            "url": url,
                            "title": "",
                            "content": "",
                            "error": (
                                "Scrape timed out after 60s — page may be too large "
                                "or unresponsive. Try browser_navigate instead."
                            ),
                        }
                    )
                    continue

                scrape_payload = _extract_scrape_payload(scrape_result)
                metadata = scrape_payload.get("metadata", {})
                content_markdown = scrape_payload.get("markdown")
                content_html = scrape_payload.get("html")

                # Ensure metadata is a dict (SDK may return a typed object)
                if not isinstance(metadata, dict):
                    if hasattr(metadata, "model_dump"):
                        metadata = metadata.model_dump()
                    elif hasattr(metadata, "__dict__"):
                        metadata = metadata.__dict__
                    else:
                        metadata = {}

                title = metadata.get("title", "")
                final_url = metadata.get("sourceURL", url)

                # Re-check SSRF safety after any redirect reported by Firecrawl.
                if not is_safe_url(final_url):
                    logger.info(
                        "Blocked redirected web_extract for unsafe final URL: %s",
                        final_url,
                    )
                    results.append(
                        {
                            "url": final_url,
                            "title": title,
                            "content": "",
                            "raw_content": "",
                            "error": (
                                "Blocked: URL targets a private or internal "
                                "network address"
                            ),
                        }
                    )
                    continue

                # Re-check website-access policy after any redirect
                final_blocked = check_website_access(final_url)
                if final_blocked:
                    logger.info(
                        "Blocked redirected web_extract for %s by rule %s",
                        final_blocked["host"],
                        final_blocked["rule"],
                    )
                    results.append(
                        {
                            "url": final_url,
                            "title": title,
                            "content": "",
                            "raw_content": "",
                            "error": final_blocked["message"],
                            "blocked_by_policy": {
                                "host": final_blocked["host"],
                                "rule": final_blocked["rule"],
                                "source": final_blocked["source"],
                            },
                        }
                    )
                    continue

                # Choose markdown vs html according to the requested format
                if format == "markdown" or (format is None and content_markdown):
                    chosen_content = content_markdown
                else:
                    chosen_content = content_html or content_markdown or ""

                results.append(
                    {
                        "url": final_url,
                        "title": title,
                        "content": chosen_content,
                        "raw_content": chosen_content,
                        "metadata": metadata,
                    }
                )
            except Exception as scrape_err:  # noqa: BLE001
                logger.debug("Firecrawl scrape failed for %s: %s", url, scrape_err)
                if _is_bad_request_error(scrape_err):
                    # Firecrawl API 4xx (broken for plain URLs since
                    # 2026-07-05): try a bounded direct HTTP fetch instead.
                    fallback_result = _direct_http_extract_fallback(url)
                    if fallback_result is not None:
                        logger.warning(
                            "Firecrawl returned 4xx for %s (%s); "
                            "served content via direct-http fallback",
                            url,
                            scrape_err,
                        )
                        results.append(fallback_result)
                        continue
                results.append(
                    {
                        "url": url,
                        "title": "",
                        "content": "",
                        "raw_content": "",
                        "error": str(scrape_err),
                    }
                )

        return results

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Firecrawl",
            "badge": "paid · optional gateway",
            "tag": (
                "Full search + extract; supports direct API and "
                "Nous tool-gateway routing."
            ),
            "env_vars": [
                {
                    "key": "FIRECRAWL_API_KEY",
                    "prompt": "Firecrawl API key (or leave blank for self-hosted)",
                    "url": "https://docs.firecrawl.dev/introduction",
                },
            ],
        }
