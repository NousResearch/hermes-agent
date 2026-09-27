"""Bounded retries for PM-owned, idempotent network operations."""

from __future__ import annotations

import errno
import http.client
import logging
import socket
import ssl
import time
import urllib.error
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Callable, TypeVar

_T = TypeVar("_T")
_ATTEMPTS = 4
_MAX_DELAY = 30.0
_HTTP_RETRY = frozenset({408, 429, 500, 502, 503, 504})
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
                        when = when.replace(tzinfo=timezone.utc)
                    retry_after = when.timestamp() - time.time()
                except (TypeError, ValueError, OverflowError):
                    retry_after = 0.0
            delay = max(delay, retry_after)
    return min(delay, _MAX_DELAY)


def retry_network(
    operation: Callable[[], _T], *, wait: Callable[[float], None] | None = None,
) -> _T:
    """Retry the complete request and body read, never an install or publish.

    The caller resets attempt-local buffers and keeps any resumable bytes.
    A download supplies an interruptible wait so pause does not wait for backoff.
    """
    attempt = 1
    while True:
        try:
            return operation()
        except Exception as exc:
            if isinstance(exc, urllib.error.HTTPError):
                exc.close()
            if not is_transient(exc) or attempt >= _ATTEMPTS:
                raise
            delay = _delay(exc, attempt - 1)
            attempt += 1
            logging.getLogger(__name__).warning(
                "network request failed (%s); retrying in %gs (attempt %d/%d)",
                exc, delay, attempt, _ATTEMPTS,
            )
            (wait or time.sleep)(delay)
