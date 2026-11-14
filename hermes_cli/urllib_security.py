"""Security policy for credential-bearing stdlib urllib requests."""

from __future__ import annotations

import copy
import gzip
import logging
import ssl
import urllib.parse
import urllib.request
import zlib
from collections.abc import Callable, Iterable
from typing import Any

logger = logging.getLogger(__name__)

# Headers safe to forward to a different origin. Everything else is dropped:
# custom provider headers routinely carry credentials under arbitrary names.
_CROSS_ORIGIN_SAFE_HEADERS = frozenset({"accept", "user-agent"})
_DEFAULT_PORTS = {"http": 80, "https": 443}


def url_origin(url: str) -> tuple[str, str, int | None]:
    """Return a normalized (scheme, hostname, effective port) origin."""
    parsed = urllib.parse.urlparse(url)
    scheme = (parsed.scheme or "").lower()
    # ``parsed.port`` raises ValueError on malformed ports — let that fail the
    # request closed instead of collapsing it to a default.
    port = parsed.port
    return scheme, (parsed.hostname or "").lower().rstrip("."), port if port is not None else _DEFAULT_PORTS.get(scheme)


def _strip_headers(request, keep: frozenset[str]) -> None:
    """Drop every header on *request* whose lowercased name is not in *keep*."""
    for name, _value in list(request.header_items()):
        if name.lower() not in keep:
            request.remove_header(name)


class SafeCredentialRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Preserve request headers only while redirects stay on one origin."""

    def __init__(
        self, original_url: str, *, cross_origin_safe_headers: Iterable[str] = _CROSS_ORIGIN_SAFE_HEADERS
    ) -> None:
        self._original_origin = url_origin(original_url)
        self._cross_origin_safe_headers = frozenset(str(name).lower() for name in cross_origin_safe_headers)

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Let urllib enforce status/method semantics first (notably 307/308).
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is None:
            return None

        # Allowlist rather than guessing credential header names: normalize_extra_headers
        # permits arbitrary secret-bearing names.
        if url_origin(urllib.parse.urljoin(req.full_url, newurl)) != self._original_origin:
            _strip_headers(redirected, self._cross_origin_safe_headers)
        return redirected


class _CrossOriginRequestSanitizer(urllib.request.BaseHandler):
    """Strip headers after installed request processors have run."""

    # Request processors run in ascending order; infinity keeps this last so an
    # installed cookie/auth/instrumentation processor cannot re-add a secret after
    # the redirect handler sanitized the new Request (stable sort keeps this
    # appended handler after another infinity-ordered one).
    handler_order = float("inf")  # type: ignore[assignment]

    def __init__(self, original_url: str) -> None:
        self._original_origin = url_origin(original_url)

    def _sanitize(self, request: urllib.request.Request):
        if url_origin(request.full_url) != self._original_origin:
            _strip_headers(request, _CROSS_ORIGIN_SAFE_HEADERS)
        return request

    http_request = _sanitize
    https_request = _sanitize


def _resolved_https_context() -> ssl.SSLContext | None:
    """TLS context for Hermes-owned urllib openers.

    None means "use urllib's default", which — with the OS trust store
    installed process-wide — already verifies against the platform's
    certificates. There is nothing to resolve here any more.
    """
    from agent.ssl_verify import install_truststore

    install_truststore()
    return None


def _secure_opener_from_installed_policy(original_url: str, *, ssl_context=None):
    """Clone the installed opener's handlers, replacing redirect policy only.

    When ``ssl_context`` is provided, the cloned HTTPS handler is replaced with
    one bound to that context so per-provider TLS settings (``ssl_ca_cert`` /
    ``ssl_verify``) apply to this request. When it is None, Hermes-owned
    openers verify against the OS trust store; an application-installed
    opener's TLS policy is preserved unchanged.
    """
    installed = getattr(urllib.request, "_opener", None)
    if installed is None:
        context = _resolved_https_context()
        installed = urllib.request.build_opener(*([] if context is None else [urllib.request.HTTPSHandler(context=context)]))

    _https_handler_cls = getattr(urllib.request, "HTTPSHandler", None)
    replace_https = ssl_context is not None and _https_handler_cls is not None
    handlers = [
        copy.copy(handler)
        for handler in getattr(installed, "handlers", ())
        if not isinstance(handler, urllib.request.HTTPRedirectHandler)
        and not (replace_https and isinstance(handler, _https_handler_cls))
    ]
    if replace_https:
        handlers.append(_https_handler_cls(context=ssl_context))
    handlers.append(SafeCredentialRedirectHandler(original_url))
    handlers.append(_CrossOriginRequestSanitizer(original_url))
    secured = urllib.request.build_opener(*handlers)
    # OpenerDirector injects addheaders after request processors (bypassing the
    # sanitizer on redirects), so carry them on the initial request instead.
    secured._hermes_initial_addheaders = list(getattr(installed, "addheaders", ()))
    secured.addheaders = []
    return secured


def decode_response_body(resp: Any, body: bytes) -> bytes:
    """Return *body* with its transport compression undone.

    ``urllib`` never transparently decompresses a response, so a provider whose
    profile asks for gzip (Kimi/Moonshot does, to sidestep httpx's brotli SSE
    decode bug) hands back compressed bytes that ``json.loads`` cannot read.
    The catalog probe swallowed that ``UnicodeDecodeError``, which silently
    degraded the model picker to a provider's stale static list instead of its
    live catalog.

    Decoding follows the response's ``Content-Encoding`` when it advertises
    one, and falls back to sniffing the gzip magic bytes so a server that
    compresses without advertising it is still handled. ``br`` is deliberately
    not decoded: httpx's brotlicffi backend has a streaming decode bug on
    Moonshot's brotli SSE responses (#28043, #48428, #59556).
    """
    if not body:
        return body
    try:
        encoding = (resp.headers.get("Content-Encoding") or "").strip().lower()
    except Exception:
        encoding = ""
    if "gzip" in encoding or body[:2] == b"\x1f\x8b":
        try:
            return gzip.decompress(body)
        except (OSError, EOFError, zlib.error):
            if body[:2] == b"\x1f\x8b":
                # Real gzip bytes we cannot decode: let the caller see the error.
                raise
            # The header claimed gzip but the body is not gzip — fall through.
    if "deflate" in encoding:
        try:
            return zlib.decompress(body)
        except zlib.error:
            return zlib.decompress(body, -zlib.MAX_WBITS)
    return body


def open_credentialed_url(
    request: urllib.request.Request,
    *,
    timeout: float,
    opener_factory: Callable[..., Any] | None = None,
    ssl_context=None,
):
    """Open a request without forwarding credentials across origins.

    The default preserves an application-installed opener's proxy, TLS,
    cookies, custom protocol handlers, and instrumentation while replacing its
    redirect handler. ``opener_factory`` is an explicit test seam; security is
    never disabled based on global ``urlopen`` identity.

    ``ssl_context`` (an ``ssl.SSLContext``) overrides the HTTPS handler's TLS
    policy for this request only. It is used to honor a custom provider's
    ``ssl_ca_cert`` / ``ssl_verify`` on the ``/models`` discovery path, which
    otherwise falls back to the process-wide platform trust store
    (``agent.ssl_verify.install_truststore``).
    """
    if opener_factory is None:
        opener = _secure_opener_from_installed_policy(request.full_url, ssl_context=ssl_context)
        for name, value in getattr(opener, "_hermes_initial_addheaders", ()):
            if not request.has_header(name):
                request.add_header(name, value)
    else:
        opener = opener_factory(SafeCredentialRedirectHandler(request.full_url))
    return opener.open(request, timeout=timeout)


__all__ = [
    "SafeCredentialRedirectHandler",
    "decode_response_body",
    "open_credentialed_url",
    "url_origin",
]
