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

import asyncio
import errno
import random
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from plugins.platforms.whatsapp import delivery_ledger as _delivery_ledger

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


# ---------------------------------------------------------------------------
# Bounded delivery attempts
# ---------------------------------------------------------------------------

MAX_DELIVERY_ATTEMPTS = 3

# Delay (seconds) before attempt N, indexed by retry ordinal: attempt 2 waits
# 1s after the first failure, attempt 3 waits 5s.
_BACKOFF_SCHEDULE = (1.0, 5.0)

# Profile-owned delivery policy hook (e.g. Sawi DDD19/30-day outreach rules).
# Upstream Hermes never registers one; profile code opts in via
# set_delivery_policy_hook(). The hook receives a sanitized context dict and
# returns False to veto the delivery before the first attempt is made.
_policy_hook: Optional[Callable[[dict], bool]] = None


def set_delivery_policy_hook(hook: Optional[Callable[[dict], bool]]) -> None:
    """Register (or clear, with ``None``) the profile delivery policy hook."""
    global _policy_hook
    _policy_hook = hook


def retry_backoff_delay(attempt: int, *, jitter: bool = True) -> float:
    """Seconds to wait before ``attempt`` (2-based; attempt 1 never waits).

    With ``jitter`` the delay is stretched by up to 50% so concurrent
    retries don't synchronize against the bridge.
    """
    index = min(max(attempt - 2, 0), len(_BACKOFF_SCHEDULE) - 1)
    delay = _BACKOFF_SCHEDULE[index]
    if jitter:
        delay += random.uniform(0, delay * 0.5)
    return delay


@dataclass
class DeliveryOutcome:
    """Result of one logical delivery (all attempts included)."""

    ok: bool
    attempts: int
    idempotency_key: Optional[str]
    status: Optional[int] = None
    data: Any = None
    error: Optional[str] = None
    failure: Optional[DeliveryFailure] = None
    dead_letter_ref: Optional[str] = None


async def send_with_retries(
    attempt_fn: Callable[[Dict[str, str]], Awaitable[Tuple[int, Any]]],
    *,
    base_headers: Optional[Dict[str, str]] = None,
    max_attempts: int = MAX_DELIVERY_ATTEMPTS,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    jitter: bool = True,
    policy_context: Optional[dict] = None,
    platform: str = "whatsapp",
    route: str = "",
) -> DeliveryOutcome:
    """Run one logical bridge delivery with bounded, classified retries.

    ``attempt_fn(headers)`` performs a single POST and returns
    ``(status, payload)`` — parsed JSON for 2xx, error text otherwise —
    or raises. One ``Idempotency-Key`` is generated per logical delivery
    and reused verbatim on every attempt so the bridge can deduplicate.

    A 2xx ends the loop; a permanent or ambiguous failure stops immediately
    (an ambiguous timeout is NEVER retried — the send may have gone out).
    Only retryable failures re-attempt, up to ``max_attempts``.

    A terminal failure (retries exhausted, ambiguous timeout, or permanent
    rejection) is recorded to the sanitized dead-letter ledger — a no-op
    unless a profile has opted in — and its reference is attached to the
    returned outcome. A policy veto is never dead-lettered: no attempt was
    made, so there is nothing to reconcile.
    """
    idempotency_key = uuid.uuid4().hex
    headers = dict(base_headers or {})
    headers["Idempotency-Key"] = idempotency_key

    if _policy_hook is not None:
        try:
            allowed = _policy_hook(dict(policy_context or {}))
        except Exception:
            # A broken profile hook must not silently bypass the policy it
            # exists to enforce — fail closed.
            allowed = False
        if not allowed:
            return DeliveryOutcome(
                ok=False,
                attempts=0,
                idempotency_key=idempotency_key,
                error="delivery vetoed by policy hook",
                failure=DeliveryFailure(NON_RETRYABLE, "policy_blocked"),
            )

    attempts = 0
    while True:
        attempts += 1
        try:
            status, payload = await attempt_fn(headers)
        except Exception as exc:
            failure = classify_delivery_failure(exception=exc)
            error = str(exc)
            status = None
        else:
            if 200 <= status < 300:
                return DeliveryOutcome(
                    ok=True,
                    attempts=attempts,
                    idempotency_key=idempotency_key,
                    status=status,
                    data=payload,
                )
            failure = classify_delivery_failure(status=status)
            error = payload if isinstance(payload, str) else str(payload)

        if failure.decision != RETRYABLE or attempts >= max_attempts:
            dead_letter_ref = _delivery_ledger.record_dead_letter(
                platform=platform,
                route=route,
                idempotency_key=idempotency_key,
                attempts=attempts,
                category=failure.category,
                status=failure.status,
            )
            return DeliveryOutcome(
                ok=False,
                attempts=attempts,
                idempotency_key=idempotency_key,
                status=status,
                error=error,
                failure=failure,
                dead_letter_ref=dead_letter_ref,
            )
        await sleep(retry_backoff_delay(attempts + 1, jitter=jitter))
