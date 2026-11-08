"""Bounded retries for PM-owned, idempotent network operations."""

from __future__ import annotations

import errno
import http.client
import logging
import socket
import ssl
import time
import urllib.error
import urllib.parse
from datetime import timezone, UTC
from email.utils import parsedate_to_datetime
from typing import Callable, TypeVar

_T = TypeVar("_T")
_ATTEMPTS = 4
_MAX_DELAY = 30.0
_MAX_RECOVERY_DELAY = 300.0
_HTTP_RETRY = frozenset({408, 429, 500, 502, 503, 504})
_R2_HOSTS = frozenset({"hermes-assets.nousresearch.com"})
_NETWORK_ERRNOS = frozenset({
    errno.ECONNABORTED, errno.ECONNREFUSED, errno.ECONNRESET,
    errno.ENETDOWN, errno.ENETRESET, errno.ENETUNREACH,
    errno.EHOSTUNREACH, errno.EPIPE, errno.ETIMEDOUT,
})


def is_transient(exc: Exception) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in _HTTP_RETRY
    if isinstance(exc, urllib.error.URLError):
        return isinstance(exc.reason, Exception) and is_transient(exc.reason)
    if isinstance(exc, (ssl.SSLEOFError, ssl.SSLZeroReturnError)):
        return True
    if isinstance(exc, ssl.SSLError):
        return False
    if isinstance(exc, socket.gaierror):
        return exc.errno == socket.EAI_AGAIN
    if isinstance(exc, (TimeoutError, ConnectionError, http.client.IncompleteRead)):
        return True
    return isinstance(exc, OSError) and exc.errno in _NETWORK_ERRNOS


def edge_block(exc: BaseException) -> str | None:
    """Explain a 403/429 that a CDN firewall sent instead of the server, else None.

    A bare "HTTP 403" from a regional block reads as a broken install or a missing
    file (#128295). Cloudflare marks challenges with ``cf-mitigated`` and Vercel's
    firewall sets ``x-vercel-mitigated``. Cloudflare's own block page carries
    neither, but on our R2 custom domain a response without ``cf-cache-status``
    never reached the bucket (Cloudflare R2 troubleshooting docs); other hosts
    behind Cloudflare make no such promise, so that fallback is R2-only.
    """
    if not isinstance(exc, urllib.error.HTTPError) or exc.code not in (403, 429):
        return None
    headers = exc.headers or {}
    host = urllib.parse.urlsplit(exc.filename or "").hostname or ""
    if headers.get("x-vercel-mitigated"):
        cdn = "Vercel"
    elif headers.get("cf-mitigated") or (
            host in _R2_HOSTS and (headers.get("server") or "").lower() == "cloudflare"
            and not headers.get("cf-cache-status")):
        cdn = "Cloudflare"
    else:
        return None
    return (f"{host or 'the server'}'s CDN firewall ({cdn}) refused this network. Your install is fine; "
            "retry from another network")


def _delay(exc: Exception, attempt: int) -> float:
    delay = float(2 ** attempt)
    if isinstance(exc, urllib.error.HTTPError):
        value = exc.headers.get("Retry-After", "") if exc.headers else ""
        if value:
            try:
                retry_after = int(value)
            except ValueError:
                try:
                    when = parsedate_to_datetime(value)
                    if when.tzinfo is None:
                        when = when.replace(tzinfo=UTC)
                    retry_after = when.timestamp() - time.time()
                except (TypeError, ValueError, OverflowError):
                    retry_after = 0.0
            delay = max(delay, retry_after)
    return delay


def retry_network(
    operation: Callable[[], _T], *, wait: Callable[[float], None] | None = None,
    progress: Callable[[], int] | None = None, fallback_available: bool = False,
) -> _T:
    """Retry the complete request and body read, never an install or publish.

    The caller resets attempt-local buffers and keeps any resumable bytes.
    A download supplies an interruptible wait so pause does not wait for backoff.
    Downloads supply ``progress``: durable forward progress renews their
    budget. Metadata and non-resumable requests still get four attempts.
    A resumable request yields to an available fallback before a long wait.
    """
    failures_since_progress = 0
    backoff = 0
    saved = progress() if progress is not None else 0
    while True:
        try:
            return operation()
        except Exception as exc:
            if isinstance(exc, urllib.error.HTTPError):
                exc.close()
            if not is_transient(exc):
                raise
            current = progress() if progress is not None else saved
            if current > saved:
                failures_since_progress = 0
                backoff = 0
                saved = current
            else:
                failures_since_progress += 1
            if failures_since_progress >= _ATTEMPTS:
                raise
            delay = _delay(exc, backoff)
            if progress is None:
                delay = min(delay, _MAX_DELAY)
            elif delay > (_MAX_DELAY if fallback_available else _MAX_RECOVERY_DELAY):
                raise
            backoff += 1
            logging.getLogger(__name__).warning(
                "network request failed (%s); retrying in %gs (%d/%d failures without progress)",
                exc, delay, failures_since_progress, _ATTEMPTS,
            )
            (wait or time.sleep)(delay)
