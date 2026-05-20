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
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from agent.web_search_provider import WebSearchProvider
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
    api_key = os.getenv("FIRECRAWL_API_KEY", "").strip()
    api_url = os.getenv("FIRECRAWL_API_URL", "").strip().rstrip("/")

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

    Reads ``read_nous_access_token`` and ``resolve_managed_tool_gateway``
    via :mod:`tools.web_tools` rather than direct imports, so unit tests
    that ``patch("tools.web_tools._read_nous_access_token", ...)`` see
    their patches honored. The names are re-exported on
    :mod:`tools.web_tools` for exactly this reason.
    """
    import tools.web_tools as _wt

    return _wt.resolve_managed_tool_gateway(
        "firecrawl", token_reader=_wt._read_nous_access_token
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
            " With your Nous subscription you can also use the Tool Gateway — "
            "run `hermes tools` and select Nous Subscription as the web provider."
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

    def supports_crawl(self) -> bool:
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
            return {"success": False, "error": f"Firecrawl search failed: {exc}"}

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

    async def crawl(self, url: str, **kwargs: Any) -> Dict[str, Any]:
        """Crawl a seed URL via Firecrawl's ``/crawl`` endpoint.

        Sync SDK call wrapped in ``asyncio.to_thread`` because the dispatcher
        in :func:`tools.web_tools.web_crawl_tool` is async and runs LLM
        post-processing on the response. The dispatcher gates the seed URL
        against SSRF + website-access policy before calling us; this method
        re-checks every crawled page's URL against the policy after the
        crawl returns to catch redirected pages that map to a blocked host.

        Accepted kwargs (others ignored for forward compat):
          - ``instructions``: str — logged then dropped. Firecrawl's /crawl
            endpoint does NOT accept natural-language instructions (that's
            an /extract feature), so we record the value for debugging and
            proceed without it. Tavily's crawl IS instruction-aware; this
            divergence is documented in both plugins' docstrings.
          - ``limit``: int — max pages to crawl (default 20).
          - ``depth``: str — accepted for API parity with Tavily; ignored
            by Firecrawl's crawl endpoint.

        Returns ``{"results": [...]}`` matching the shape that
        :func:`tools.web_tools.web_crawl_tool`'s shared LLM-summarization
        path expects. Per-page failures (policy block on redirected URL,
        bad response shape) are included as items with an ``error`` field
        rather than raising.
        """
        try:
            from tools.interrupt import is_interrupted

            if is_interrupted():
                return {"results": [{"url": url, "title": "", "content": "", "error": "Interrupted"}]}

            instructions = kwargs.get("instructions")
            limit = kwargs.get("limit", 20)

            # Firecrawl's /crawl endpoint does not accept natural-language
            # instructions (that's an /extract feature). Log + drop.
            if instructions:
                logger.info(
                    "Firecrawl crawl: 'instructions' parameter ignored "
                    "(not supported by Firecrawl /crawl)"
                )

            logger.info("Firecrawl crawl: %s (limit=%d)", url, limit)

            crawl_params = {
                "limit": limit,
                "scrape_options": {"formats": ["markdown"]},
            }

            # The SDK call is sync; run in a thread so we don't block the
            # gateway event loop on a multi-page crawl.
            crawl_result = await asyncio.to_thread(
                _get_firecrawl_client().crawl,
                url=url,
                **crawl_params,
            )

            # CrawlJob normalization across SDK + direct + gateway shapes.
            data_list: List[Any] = []
            if hasattr(crawl_result, "data"):
                data_list = crawl_result.data if crawl_result.data else []
                logger.info(
                    "Firecrawl crawl status: %s, %d pages",
                    getattr(crawl_result, "status", "unknown"),
                    len(data_list),
                )
            elif isinstance(crawl_result, dict) and "data" in crawl_result:
                data_list = crawl_result.get("data", []) or []
            else:
                logger.warning(
                    "Firecrawl crawl: unexpected result type %r",
                    type(crawl_result).__name__,
                )

            pages: List[Dict[str, Any]] = []
            for item in data_list:
                # Pydantic model | typed object | dict — handle all shapes.
                content_markdown = None
                content_html = None
                metadata: Any = {}

                if hasattr(item, "model_dump"):
                    item_dict = item.model_dump()
                    content_markdown = item_dict.get("markdown")
                    content_html = item_dict.get("html")
                    metadata = item_dict.get("metadata", {})
                elif hasattr(item, "__dict__"):
                    content_markdown = getattr(item, "markdown", None)
                    content_html = getattr(item, "html", None)
                    metadata_obj = getattr(item, "metadata", {})
                    if hasattr(metadata_obj, "model_dump"):
                        metadata = metadata_obj.model_dump()
                    elif hasattr(metadata_obj, "__dict__"):
                        metadata = metadata_obj.__dict__
                    elif isinstance(metadata_obj, dict):
                        metadata = metadata_obj
                    else:
                        metadata = {}
                elif isinstance(item, dict):
                    content_markdown = item.get("markdown")
                    content_html = item.get("html")
                    metadata = item.get("metadata", {})

                # Ensure metadata is a plain dict.
                if not isinstance(metadata, dict):
                    if hasattr(metadata, "model_dump"):
                        metadata = metadata.model_dump()
                    elif hasattr(metadata, "__dict__"):
                        metadata = metadata.__dict__
                    else:
                        metadata = {}

                page_url = metadata.get(
                    "sourceURL", metadata.get("url", "Unknown URL")
                )
                title = metadata.get("title", "")

                # Per-page policy re-check (catches blocked redirects).
                page_blocked = check_website_access(page_url)
                if page_blocked:
                    logger.info(
                        "Blocked crawled page %s by rule %s",
                        page_blocked["host"],
                        page_blocked["rule"],
                    )
                    pages.append(
                        {
                            "url": page_url,
                            "title": title,
                            "content": "",
                            "raw_content": "",
                            "error": page_blocked["message"],
                            "blocked_by_policy": {
                                "host": page_blocked["host"],
                                "rule": page_blocked["rule"],
                                "source": page_blocked["source"],
                            },
                        }
                    )
                    continue

                content = content_markdown or content_html or ""
                pages.append(
                    {
                        "url": page_url,
                        "title": title,
                        "content": content,
                        "raw_content": content,
                        "metadata": metadata,
                    }
                )

            return {"results": pages}
        except ValueError as exc:
            return {"results": [{"url": url, "title": "", "content": "", "error": str(exc)}]}
        except ImportError as exc:
            return {
                "results": [
                    {
                        "url": url,
                        "title": "",
                        "content": "",
                        "error": f"Firecrawl SDK not installed: {exc}",
                    }
                ]
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Firecrawl crawl error: %s", exc)
            return {
                "results": [
                    {
                        "url": url,
                        "title": "",
                        "content": "",
                        "error": f"Firecrawl crawl failed: {exc}",
                    }
                ]
            }

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Firecrawl",
            "badge": "paid · optional gateway",
            "tag": (
                "Full search + extract + crawl; supports direct API and "
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
