"""Local web extract — direct httpx fetch + trafilatura readability extraction.

Subclasses :class:`agent.web_search_provider.WebSearchProvider`. One
capability advertised:

- ``supports_search()``  -> False (pair with ddgs / searxng / brave-free)
- ``supports_extract()`` -> True  (fetch + readability, no API key)

Unlike the cloud extract providers (Firecrawl, Tavily, Exa, Parallel),
the HTTP request here originates from THIS machine — so the SSRF check
(:func:`tools.url_safety.is_safe_url`) and the website-policy gate
(:func:`tools.website_policy.check_website_access`) are load-bearing and
re-run on EVERY redirect hop, not just on the final URL. Redirects are
followed manually (``follow_redirects=False`` + a bounded hop loop) for
exactly this reason.

Readability extraction uses ``trafilatura`` (Apache-2.0; deps courlan /
htmldate Apache-2.0, jusText BSD-2, lxml BSD, urllib3 MIT — verified
2026-08-16). It is NOT vendored: it installs through the
``tools.lazy_deps`` allowlist (``search.trafilatura``), venv-scoped and
version-pinned, like the other web-backend SDKs. Only trafilatura's
in-memory extraction API is used — its own ``fetch_url`` networking is
never called; all HTTP goes through the SSRF-checked httpx path above.

Config keys this provider responds to::

    web:
      extract_backend: "local"     # explicit per-capability
      backend: "local"             # shared fallback (extract only)

Env vars: none — that is the point.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

from agent.web_search_provider import WebSearchProvider
from tools.url_safety import is_safe_url
from tools.website_policy import check_website_access

logger = logging.getLogger(__name__)

# Per-URL wall clock for fetch + extraction combined (asyncio.wait_for cap
# in extract()); mirrors the firecrawl per-URL 60s guard.
_EXTRACT_TIMEOUT_SECS = 60

# httpx per-request timeout (connect/read/write/pool). The hop loop can
# issue up to _MAX_REDIRECTS + 1 requests, all under _EXTRACT_TIMEOUT_SECS.
_FETCH_TIMEOUT_SECS = 30

_MAX_REDIRECTS = 5

# Body size cap — readable articles are far below this; the cap protects
# against feeding a multi-hundred-MB response to lxml on the agent host.
_MAX_CONTENT_BYTES = 10 * 1024 * 1024

# Content types we hand to trafilatura (or return as-is for text/plain).
# Anything else (PDF, images, archives) gets a typed error suggesting the
# browser tool instead.
_HTMLISH_CONTENT_TYPES = (
    "text/html",
    "application/xhtml+xml",
    "application/xml",
    "text/xml",
)

# Some hosts refuse default library user agents outright; identify as a
# browser engine but keep an honest product token appended.
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) HermesAgent-LocalExtract/1.0"
)

# Test hook — tests inject an ``httpx.MockTransport`` here; production
# leaves it None (httpx builds its default transport).
_transport_for_tests: Optional[Any] = None


def _import_trafilatura() -> Any:
    """Import trafilatura, lazily installing it on first use.

    Follows the firecrawl-SDK pattern: try the plain import first (free
    when already installed), then go through the ``tools.lazy_deps``
    allowlist entry ``search.trafilatura`` (venv-scoped, pinned).
    Raises ``ImportError`` with the remediation hint when lazy installs
    are disabled or the install fails.
    """
    try:
        import trafilatura  # noqa: WPS433 — deliberately lazy

        return trafilatura
    except ImportError:
        pass

    try:
        from tools.lazy_deps import ensure as _lazy_ensure

        _lazy_ensure("search.trafilatura", prompt=False)
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001 — surface install hint
        raise ImportError(str(exc))

    import trafilatura  # noqa: WPS433 — deliberately lazy

    return trafilatura


def _make_client() -> Any:
    """Build the httpx client used by the fetch loop.

    ``follow_redirects=False`` is deliberate — redirects are followed
    manually in :func:`_fetch_readable` so each hop is SSRF- and
    policy-checked before any request is issued.
    """
    import httpx

    return httpx.Client(
        timeout=httpx.Timeout(_FETCH_TIMEOUT_SECS),
        follow_redirects=False,
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;q=0.9,"
                "text/plain;q=0.8,*/*;q=0.5"
            ),
        },
        transport=_transport_for_tests,
    )


def _policy_error_item(url: str, blocked: Dict[str, Any]) -> Dict[str, Any]:
    """Build the standard blocked-by-policy result entry (firecrawl shape)."""
    return {
        "url": url,
        "title": "",
        "content": "",
        "raw_content": "",
        "error": blocked["message"],
        "blocked_by_policy": {
            "host": blocked["host"],
            "rule": blocked["rule"],
            "source": blocked["source"],
        },
    }


def _fetch_readable(url: str) -> Dict[str, Any]:
    """Fetch *url*, following at most ``_MAX_REDIRECTS`` manually.

    Every hop — including the first — is checked with
    :func:`is_safe_url` and :func:`check_website_access` BEFORE the
    request goes out. The body is streamed with a hard byte cap.

    Returns either::

        {"final_url": str, "content": bytes, "content_type": str,
         "encoding": str | None}

    or an error mapping (``final_url`` + ``error`` [+
    ``blocked_by_policy``]) that :func:`_extract_one` converts into a
    per-URL result entry.
    """
    current = url
    with _make_client() as client:
        for _hop in range(_MAX_REDIRECTS + 1):
            if not is_safe_url(current):
                logger.info("Blocked local web_extract for unsafe URL: %s", current)
                return {
                    "final_url": current,
                    "error": (
                        "Blocked: URL targets a private or internal "
                        "network address"
                    ),
                }
            blocked = check_website_access(current)
            if blocked:
                logger.info(
                    "Blocked local web_extract for %s by rule %s",
                    blocked["host"],
                    blocked["rule"],
                )
                return {
                    "final_url": current,
                    "error": blocked["message"],
                    "blocked_by_policy": blocked,
                }

            with client.stream("GET", current) as response:
                if response.is_redirect:
                    location = response.headers.get("location", "")
                    if not location:
                        return {
                            "final_url": current,
                            "error": (
                                f"HTTP {response.status_code} redirect "
                                "without a Location header"
                            ),
                        }
                    current = urljoin(current, location)
                    continue

                if response.status_code >= 400:
                    return {
                        "final_url": current,
                        "error": f"HTTP {response.status_code} fetching URL",
                    }

                content_type = (
                    response.headers.get("content-type", "")
                    .split(";", 1)[0]
                    .strip()
                    .lower()
                )

                supported = (
                    not content_type
                    or content_type in _HTMLISH_CONTENT_TYPES
                    or content_type == "text/plain"
                )
                if not supported:
                    return {
                        "final_url": current,
                        "error": (
                            f"Unsupported content type '{content_type}' — "
                            "the local extract backend handles HTML/XML/plain "
                            "text. Try browser_navigate for other formats."
                        ),
                    }

                body = bytearray()
                for chunk in response.iter_bytes():
                    body.extend(chunk)
                    if len(body) > _MAX_CONTENT_BYTES:
                        return {
                            "final_url": current,
                            "error": (
                                "Response exceeds the "
                                f"{_MAX_CONTENT_BYTES // (1024 * 1024)}MB "
                                "local extract limit"
                            ),
                        }

                encoding: Optional[str] = None
                try:
                    encoding = response.charset_encoding
                except Exception:  # noqa: BLE001 — header parsing quirk
                    encoding = None

                return {
                    "final_url": current,
                    "content": bytes(body),
                    "content_type": content_type,
                    "encoding": encoding,
                }

    # Loop exhausted without a non-redirect response.
    return {
        "final_url": current,
        "error": f"Exceeded {_MAX_REDIRECTS} redirects",
    }


def _metadata_dict(trafilatura: Any, content: bytes, final_url: str) -> Dict[str, Any]:
    """Extract page metadata into the legacy ``metadata`` mapping shape."""
    metadata: Dict[str, Any] = {"sourceURL": final_url, "title": ""}
    try:
        doc = trafilatura.extract_metadata(content, default_url=final_url)
    except Exception:  # noqa: BLE001 — metadata is best-effort
        return metadata
    if doc is None:
        return metadata
    for field in ("title", "author", "description", "sitename", "date"):
        value = getattr(doc, field, None)
        if value:
            metadata[field] = str(value)
    metadata.setdefault("title", "")
    return metadata


def _extract_one(url: str, format: Optional[str]) -> Dict[str, Any]:
    """Fetch + extract a single URL synchronously (runs in a thread).

    Never raises for per-URL conditions the caller should surface as a
    result entry — network errors, policy blocks, unsupported types, and
    empty extractions all come back as ``{"url", ..., "error"}`` items.
    Only programming errors escape.
    """
    import httpx

    try:
        fetched = _fetch_readable(url)
    except httpx.HTTPError as exc:
        return {
            "url": url,
            "title": "",
            "content": "",
            "raw_content": "",
            "error": f"Fetch failed: {exc}",
        }

    final_url = fetched.get("final_url", url)
    if "error" in fetched:
        blocked = fetched.get("blocked_by_policy")
        if blocked:
            return _policy_error_item(final_url, blocked)
        return {
            "url": final_url,
            "title": "",
            "content": "",
            "raw_content": "",
            "error": fetched["error"],
        }

    content: bytes = fetched["content"]
    content_type: str = fetched["content_type"]

    if content_type == "text/plain":
        text = content.decode(fetched.get("encoding") or "utf-8", errors="replace")
        return {
            "url": final_url,
            "title": "",
            "content": text,
            "raw_content": text,
            "metadata": {"sourceURL": final_url, "title": ""},
        }

    try:
        trafilatura = _import_trafilatura()
    except ImportError as exc:
        return {
            "url": final_url,
            "title": "",
            "content": "",
            "raw_content": "",
            "error": (
                "trafilatura is not installed and could not be lazily "
                f"installed: {exc}"
            ),
        }

    output_format = "html" if format == "html" else "markdown"
    try:
        text = trafilatura.extract(
            content,
            url=final_url,
            output_format=output_format,
            include_comments=False,
            include_tables=True,
        )
        if not text:
            # Non-article pages (index/nav-heavy) — fall back to a full
            # text rendering rather than returning nothing.
            text = trafilatura.html2txt(content)
    except Exception as exc:  # noqa: BLE001 — lxml/encoding edge cases
        return {
            "url": final_url,
            "title": "",
            "content": "",
            "raw_content": "",
            "error": f"Extraction failed: {exc}",
        }

    if not text or not text.strip():
        return {
            "url": final_url,
            "title": "",
            "content": "",
            "raw_content": "",
            "error": (
                "No readable content could be extracted from this page — "
                "it may be script-rendered. Try browser_navigate instead."
            ),
        }

    metadata = _metadata_dict(trafilatura, content, final_url)
    return {
        "url": final_url,
        "title": metadata.get("title", ""),
        "content": text,
        "raw_content": text,
        "metadata": metadata,
    }


class LocalWebSearchProvider(WebSearchProvider):
    """Keyless local extract provider (httpx fetch + trafilatura)."""

    @property
    def name(self) -> str:
        return "local"

    @property
    def display_name(self) -> str:
        return "Local (httpx + trafilatura)"

    def is_available(self) -> bool:
        """Return True when trafilatura is importable or lazily installable.

        httpx is a core dependency, so the only gate is the extraction
        package. When it is missing but ``security.allow_lazy_installs``
        permits runtime installs, the provider reports available and the
        first ``extract()`` call performs the venv-scoped install. No
        network I/O here — this runs on every ``hermes tools`` paint.
        """
        try:
            import trafilatura  # noqa: F401

            return True
        except ImportError:
            pass
        try:
            from tools.lazy_deps import _allow_lazy_installs

            return _allow_lazy_installs()
        except Exception:  # noqa: BLE001 — stripped installs
            return False

    def supports_search(self) -> bool:
        return False

    def supports_extract(self) -> bool:
        return True

    async def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract readable content from one or more URLs locally.

        Async; each URL runs fetch + extraction in a background thread
        under a ``_EXTRACT_TIMEOUT_SECS`` cap. Per-URL failures become
        items with an ``error`` field rather than raising.

        Accepted kwargs (others ignored for forward compat):
          - ``format``: ``"markdown"`` (default) or ``"html"`` — mapped to
            trafilatura's output format; both are readability-reduced.
        """
        from tools.interrupt import is_interrupted as _is_interrupted

        if _is_interrupted():
            return [{"url": u, "error": "Interrupted", "title": ""} for u in urls]

        format = kwargs.get("format")
        results: List[Dict[str, Any]] = []

        for url in urls:
            if _is_interrupted():
                results.append({"url": url, "error": "Interrupted", "title": ""})
                continue

            logger.info("Local extract: %s", url)
            try:
                item = await asyncio.wait_for(
                    asyncio.to_thread(_extract_one, url, format),
                    timeout=_EXTRACT_TIMEOUT_SECS,
                )
            except asyncio.TimeoutError:
                logger.warning("Local extract timed out for %s", url)
                item = {
                    "url": url,
                    "title": "",
                    "content": "",
                    "raw_content": "",
                    "error": (
                        f"Extract timed out after {_EXTRACT_TIMEOUT_SECS}s — "
                        "page may be too large or unresponsive. Try "
                        "browser_navigate instead."
                    ),
                }
            except Exception as exc:  # noqa: BLE001 — belt and suspenders
                logger.warning("Local extract failed for %s: %s", url, exc)
                item = {
                    "url": url,
                    "title": "",
                    "content": "",
                    "raw_content": "",
                    "error": str(exc),
                }
            results.append(item)

        return results

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Local (httpx + trafilatura)",
            "badge": "free · no key · extract only",
            "tag": (
                "Fetches pages from this machine and extracts readable "
                "text with trafilatura — pair with any search provider."
            ),
            "env_vars": [],
            # Trigger `_run_post_setup("local_extract")` after the user
            # picks this row so trafilatura gets installed on selection.
            "post_setup": "local_extract",
        }
