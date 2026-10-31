"""Same-credential backoff for an auxiliary call refused by a rate limit (429, Bedrock throttle).

Auxiliary clients make one attempt per call (OpenAI SDK ``max_retries=0``, botocore
``total_max_attempts: 1``), so the recovery ladder owns the wait. An immediate re-send lands in the
same throttle window and spends the retry, then credential rotation and the provider fallback chain
see a provider that would have answered a moment later. The ladder yields a ``wait`` step; the sync
and async ladder drivers sleep it, so both caller kinds share one schedule.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from agent import retry_utils

logger = logging.getLogger(__name__)

# One same-credential retry, then credential-pool rotation and the provider fallback chain take over.
RATE_LIMIT_RETRIES = 1
# Jittered exponential backoff (seconds): 1-1.5 before the first retry, doubling up to the cap.
RATE_LIMIT_BACKOFF_BASE = 1.0
RATE_LIMIT_BACKOFF_CAP = 4.0
# A provider-declared cooldown longer than this is not worth sitting through inside an auxiliary
# call (compression, titles, summaries): rotate or fall back instead.
RATE_LIMIT_RETRY_AFTER_CAP = 5.0


def rate_limit_retry_after(exc: Exception) -> Optional[float]:
    """Provider-declared cooldown in seconds: ``Retry-After`` / a ``retry_after`` body field, or the
    ``retry-after`` header botocore keeps in ``ResponseMetadata.HTTPHeaders``; None when absent."""
    value = retry_utils.provider_retry_after_seconds(exc)
    response = getattr(exc, "response", None)
    if value is None and isinstance(response, dict):  # botocore ClientError: the parsed response dict
        value = retry_utils.parse_retry_after_seconds(response.get("ResponseMetadata", {}).get("HTTPHeaders"))
    return value if value is not None and value > 0 else None


def rate_limit_wait(exc: Exception, attempt: int) -> Optional[float]:
    """Seconds to wait before same-credential retry ``attempt`` (1-based), or None when the provider
    declared a cooldown longer than ``RATE_LIMIT_RETRY_AFTER_CAP``."""
    declared = rate_limit_retry_after(exc)
    if declared is not None:
        return declared if declared <= RATE_LIMIT_RETRY_AFTER_CAP else None
    return retry_utils.jittered_backoff(
        attempt, base_delay=RATE_LIMIT_BACKOFF_BASE, max_delay=RATE_LIMIT_BACKOFF_CAP)


def rate_limit_retry_rungs(first_err: Exception, client: Any, kwargs: dict[str, Any], label: str):
    """Ladder rungs: wait, then re-send on the same client, up to ``RATE_LIMIT_RETRIES`` times while
    the refusal stays a rate limit. Returns ``(response, None)`` or ``(None, last_err)``."""
    from agent.auxiliary_client import (
        _LadderStep, _credential_rung_accepts, _is_payment_error, _is_rate_limit_error, _rung,
    )

    err = first_err
    for attempt in range(1, RATE_LIMIT_RETRIES + 1):
        wait = rate_limit_wait(err, attempt)
        if wait is None:
            logger.info("Auxiliary %s: provider asked for a %.0fs cooldown; not waiting it out",
                        label, rate_limit_retry_after(err) or 0.0)
            break
        logger.info("Auxiliary %s: rate limited (attempt %d/%d); retrying after %.2fs",
                    label, attempt, RATE_LIMIT_RETRIES, wait)
        yield _LadderStep("wait", (wait,))
        resp, err = yield from _rung(_LadderStep("call", (client, kwargs)), _credential_rung_accepts)
        if err is None:
            return resp, None
        if not _is_rate_limit_error(err) or _is_payment_error(err):
            break
    return None, err
