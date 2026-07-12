"""Pure retry classification for WhatsApp bridge delivery failures.

Retry policy (see docs/plans/2026-07-12-whatsapp-delivery-reliability.md):

- RETRYABLE: the send provably never reached the bridge (connection refused
  before request acceptance) or the bridge explicitly asked for a retry
  (429/502/503/504). Re-sending cannot duplicate a delivered message.
- AMBIGUOUS: the request may have been accepted but the outcome is unknown
  (timeouts). Retrying risks a duplicate send, so callers must stop.
- NON_RETRYABLE: the bridge definitively rejected the request (4xx and any
  other status) or an unknown exception fired. Retrying cannot help.

Categories are sanitized identifiers only — never exception text, message
content, chat ids or tokens — so they are safe for logs and dead-letter
records.
"""

import errno
from dataclasses import dataclass
from typing import Optional

RETRYABLE = "retryable"
NON_RETRYABLE = "non_retryable"
AMBIGUOUS = "ambiguous"

# Only these HTTP statuses are ever retried — everything else is permanent.
_RETRYABLE_STATUSES = frozenset({429, 502, 503, 504})


@dataclass(frozen=True)
class DeliveryFailure:
    """Sanitized classification of a single failed delivery attempt."""

    decision: str  # RETRYABLE | NON_RETRYABLE | AMBIGUOUS
    category: str  # sanitized identifier, e.g. "http_503", "timeout"
    status: Optional[int] = None


def _is_connection_refused(exc: BaseException) -> bool:
    """True when the TCP connection was refused before request acceptance.

    Covers the builtin ``ConnectionRefusedError``, a bare ``OSError`` carrying
    ``ECONNREFUSED``, and aiohttp's ``ClientConnectorError`` which wraps the
    underlying error in an ``os_error`` attribute.
    """
    if isinstance(exc, ConnectionRefusedError):
        return True
    os_error = getattr(exc, "os_error", None)
    if isinstance(os_error, BaseException) and _is_connection_refused(os_error):
        return True
    return isinstance(exc, OSError) and exc.errno == errno.ECONNREFUSED


def classify_delivery_failure(
    status: Optional[int] = None,
    exception: Optional[BaseException] = None,
) -> DeliveryFailure:
    """Classify one failed bridge delivery attempt.

    Pure function: no I/O, no logging. ``exception`` takes precedence over
    ``status`` (an exception means no usable HTTP response). Passing a 2xx
    ``status`` or neither argument is a caller bug and raises ``ValueError``.
    """
    if exception is not None:
        if _is_connection_refused(exception):
            return DeliveryFailure(RETRYABLE, "connection_refused")
        # asyncio.TimeoutError is an alias of TimeoutError on Python 3.11+;
        # aiohttp timeout errors subclass it.
        if isinstance(exception, TimeoutError):
            return DeliveryFailure(AMBIGUOUS, "timeout")
        return DeliveryFailure(NON_RETRYABLE, "unknown_exception")

    if status is None:
        raise ValueError("classify_delivery_failure needs a status or an exception")
    if 200 <= status < 300:
        raise ValueError(f"status {status} is a success, not a failure")

    decision = RETRYABLE if status in _RETRYABLE_STATUSES else NON_RETRYABLE
    return DeliveryFailure(decision, f"http_{status}", status)
