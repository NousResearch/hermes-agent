"""Security policy for credential-bearing stdlib urllib requests."""

from __future__ import annotations

import copy
import logging
import os
import ssl
import sys
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from utils import file_signature

logger = logging.getLogger(__name__)

# Headers safe to forward to a different origin. Everything else is dropped:
# custom provider headers routinely carry credentials under arbitrary names.
_CROSS_ORIGIN_SAFE_HEADERS = frozenset({"accept", "user-agent"})
_DEFAULT_PORTS = {"http": 80, "https": 443}
_CA_BUNDLE_ENV_VARS = ("HERMES_CA_BUNDLE", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")


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


# Building the context parses the whole CA bundle (~4 ms for certifi) on EVERY Hermes-owned request,
# since no opener is ever installed globally. Memoise on the preferred bundle's ``_bundle_signature``
# (the only file a memoised context was built from) so a rotated/reconfigured bundle is still picked
# up. The context is shared and never mutated by callers. No lock: a race costs one duplicate parse.
_HTTPS_CONTEXT_CACHE: tuple[tuple, ssl.SSLContext] | None = None


def _ca_bundle_candidates() -> tuple[str, ...]:
    """Bundles to try in order; empty keeps the stdlib default.

    Resolving the candidates and loading them are split so the memo below can key on the files that
    are actually going to be read — one precedence rule, not one for the loader and a second for the
    cache key that could drift away from it.
    """
    candidates: list[str] = []
    ca_bundle = next((value for name in _CA_BUNDLE_ENV_VARS if (value := os.getenv(name, "").strip())), "")
    if ca_bundle:
        ca_path = Path(ca_bundle).expanduser()
        if ca_path.is_file():
            candidates.append(str(ca_path))
        else:
            logger.warning("CA bundle path does not exist: %s — falling back to default certificates", ca_bundle)

    # Python on macOS has no usable system root store, so certifi stays the last resort even when a
    # configured bundle was found but turns out to be unloadable.
    if sys.platform == "darwin":
        try:
            import certifi

            candidates.append(certifi.where())
        except ImportError as exc:
            logger.warning(
                "Could not load certifi for urllib HTTPS verification: %s — falling back to default certificates",
                exc,
            )
    return tuple(candidates)


def _bundle_signature(path: str) -> tuple:
    """``(path, *file_signature(stat))`` so an edited or rotated bundle invalidates the memo.

    ``utils.file_signature`` adds inode and ctime to mtime + size, so a timestamp-preserving
    replacement (``cp -p``, ``rsync -t``) of the bundle is caught as well. Unreadable → ``(path, None)``.
    """
    try:
        return (path, *file_signature(Path(path).stat()))
    except OSError:
        return (path, None)


def _resolved_https_context() -> ssl.SSLContext | None:
    """Return the shared explicit-CA context for Hermes-owned urllib openers.

    Memoised on the preferred bundle's signature only: that is the one file a memoised context was
    built from, so a change to a fallback bundle (certifi on macOS) that was never read must not
    invalidate it — and costs no stat per request. The returned context is shared and must not be
    mutated by callers.
    """
    global _HTTPS_CONTEXT_CACHE

    candidates = _ca_bundle_candidates()
    if not candidates:
        return None
    key = _bundle_signature(candidates[0])
    cached = _HTTPS_CONTEXT_CACHE
    if cached is not None and cached[0] == key:
        return cached[1]
    context, used_path = _build_https_context(candidates)
    if used_path == candidates[0]:
        # Only a context built from the preferred bundle is memoised (used_path is None whenever
        # context is None, so this also excludes a failed load). A failed (possibly transient)
        # load — whether it left us with no context or with a fallback bundle (certifi on macOS) —
        # must be retried on the next request, not pinned until the file's mtime/size changes.
        _HTTPS_CONTEXT_CACHE = (key, context)
    return context


def _build_https_context(candidates: tuple[str, ...]) -> tuple[ssl.SSLContext | None, str | None]:
    """Return ``(context, path)`` for the first loadable candidate, or ``(None, None)``."""
    for path in candidates:
        try:
            return ssl.create_default_context(cafile=path), path
        except (OSError, ssl.SSLError) as exc:
            logger.warning(
                "CA bundle could not be loaded from %s: %s — falling back to default certificates", path, exc
            )
    return None, None


def _secure_opener_from_installed_policy(original_url: str, *, ssl_context=None):
    """Clone the installed opener's handlers, replacing redirect policy only.

    ``ssl_context`` rebinds the cloned HTTPS handler so per-provider TLS settings
    (``ssl_ca_cert``/``ssl_verify``) apply; with None a Hermes-owned opener gets the explicit CA
    default from ``_resolved_https_context`` and an application-installed opener keeps its TLS.
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


def open_credentialed_url(
    request: urllib.request.Request,
    *,
    timeout: float,
    opener_factory: Callable[..., Any] | None = None,
    ssl_context=None,
):
    """Open a request without forwarding credentials across origins.

    Preserves an application-installed opener's proxy/TLS/cookies/handlers while replacing its
    redirect handler. ``opener_factory`` is an explicit test seam (security is never disabled
    based on global ``urlopen`` identity); ``ssl_context`` overrides TLS for this request only.
    """
    if opener_factory is None:
        opener = _secure_opener_from_installed_policy(request.full_url, ssl_context=ssl_context)
        for name, value in getattr(opener, "_hermes_initial_addheaders", ()):
            if not request.has_header(name):
                request.add_header(name, value)
    else:
        opener = opener_factory(SafeCredentialRedirectHandler(request.full_url))
    return opener.open(request, timeout=timeout)


__all__ = ["SafeCredentialRedirectHandler", "open_credentialed_url", "url_origin"]
