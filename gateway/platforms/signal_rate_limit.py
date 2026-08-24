"""Signal attachment rate-limit scheduler: process-wide token-bucket simulator mirroring the per-account
attachment rate limit signal-cli/Signal-Server enforce. Producers (``SignalAdapter.send_multiple_images``
and the ``send_message`` tool's Signal path) call ``acquire(n)`` before an attachment send; on a 429 they
call ``feedback(retry_after, n)`` so the model recalibrates from the server's authoritative hint.
Concurrent calls serialize through an ``asyncio.Lock`` — FIFO fairness across sessions sharing one daemon."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Optional

from agent.retry_utils import parse_retry_after_seconds

logger = logging.getLogger(__name__)

SIGNAL_MAX_ATTACHMENTS_PER_MSG = 32  # per-message attachment cap (Signal-{Android,Desktop} source)
SIGNAL_RATE_LIMIT_BUCKET_CAPACITY = 50  # server-side token-bucket capacity for attachments
SIGNAL_RATE_LIMIT_DEFAULT_RETRY_AFTER = 4  # fallback token refill interval for signal-cli < v0.14.3
SIGNAL_RATE_LIMIT_MAX_ATTEMPTS = 2  # initial attempt + 1 retry
SIGNAL_BATCH_PACING_NOTICE_THRESHOLD = 10.0  # estimated wait above this → notify the user
SIGNAL_RPC_ERROR_RATELIMIT = -5  # signal-cli (v0.14.3+) JSON-RPC error code for RateLimitException


class SignalRateLimitError(Exception):
    """Raised by ``SignalAdapter._rpc`` for rate-limit responses when ``raise_on_rate_limit=True``.
    ``retry_after`` is the server's per-token Retry-After in seconds (signal-cli ≥ v0.14.3) or None."""

    def __init__(self, message: str, retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class SignalSchedulerError(Exception):
    pass


# "Retry after 4 seconds" — libsignal-net's RetryLaterException string form, surfaced when 429s hit during
# attachment upload (signal-cli wraps these as AttachmentInvalidException, so the typed path doesn't fire).
_RETRY_AFTER_RE = re.compile(r"Retry after (\d+(?:\.\d+)?)\s*second", re.IGNORECASE)


def _error_message(err: Any) -> str:
    return str(err.get("message", "")) if isinstance(err, dict) else str(err)


def _extract_retry_after_seconds(err: Any) -> Optional[float]:
    """Per-token Retry-After from a signal-cli rate-limit error, or None. Sources, in order:
    ``error.data.response.results[*].retryAfterSeconds`` (signal-cli ≥ v0.14.3), then "Retry after N
    seconds" parsed from the message (RetryLaterException wrapped as AttachmentInvalidException)."""
    if isinstance(err, dict):
        results = ((err.get("data") or {}).get("response") or {}).get("results") or []
        candidates = [parse_retry_after_seconds(r.get("retryAfterSeconds")) for r in results
                      if isinstance(r, dict) and r.get("retryAfterSeconds")]
        if candidates := [c for c in candidates if c is not None]:
            return max(candidates)
    match = _RETRY_AFTER_RE.search(_error_message(err))
    return parse_retry_after_seconds(match.group(1)) if match else None


def _is_signal_rate_limit_error(err: Any) -> bool:
    """True if a signal-cli RPC error reflects a rate-limit failure: the typed ``RATELIMIT_ERROR`` code
    (≥ v0.14.3), legacy ``[429]`` / ``RateLimitException`` substrings, or libsignal-net's
    ``RetryLaterException`` / "Retry after N seconds" leaked through AttachmentInvalidException."""
    if isinstance(err, dict) and err.get("code") == SIGNAL_RPC_ERROR_RATELIMIT:
        return True
    message = _error_message(err)
    msg_lower = message.lower()
    return "[429]" in message or any(s in msg_lower for s in ("ratelimit", "retrylaterexception", "retry after"))


def _format_wait(seconds: float) -> str:
    """Human-friendly wait label for user-facing pacing notices."""
    s = max(0.0, seconds)
    return f"{int(round(s))}s" if s < 90 else f"{max(1, int(round(s / 60)))} min"


def _signal_send_timeout(num_attachments: int) -> float:
    """HTTP timeout for a Signal ``send`` RPC: signal-cli uploads attachments serially inside the call, so
    the default 30s truncates large batches mid-upload (phantom failure). 5s/attachment, 60s floor."""
    return 30.0 if num_attachments <= 0 else max(60.0, 5.0 * num_attachments)


class SignalAttachmentScheduler:
    """Process-wide token-bucket simulator for Signal attachment sends: up to ``capacity`` tokens (default
    50 = Signal's server bucket), one per attachment, refilling at ``refill_rate``/s (calibrated from the
    server's per-token Retry-After once a 429 has been observed; default 1 token / 4s). ``acquire(n)``
    calls serialize through an ``asyncio.Lock`` — FIFO across sessions."""

    def __init__(self, capacity: float = float(SIGNAL_RATE_LIMIT_BUCKET_CAPACITY),
                 default_retry_after: float = float(SIGNAL_RATE_LIMIT_DEFAULT_RETRY_AFTER)) -> None:
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.refill_rate = 1.0 / float(default_retry_after)
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _projected_tokens(self, now: Optional[float] = None) -> float:
        """Tokens the bucket would hold at ``now``, without mutating state."""
        elapsed = (time.monotonic() if now is None else now) - self.last_refill
        if elapsed > 0 and self.tokens < self.capacity:
            return min(self.capacity, self.tokens + elapsed * self.refill_rate)
        return self.tokens

    def _refill(self) -> None:
        now = time.monotonic()
        self.tokens = self._projected_tokens(now)
        self.last_refill = now

    def estimate_wait(self, n: int) -> float:
        """Seconds until ``n`` tokens would be available (lock-free, informational — decides whether to emit
        a pacing notice *before* a possibly-blocking ``acquire``; races vs. concurrent acquires are benign)."""
        deficit = n - self._projected_tokens()
        return 0.0 if deficit <= 0 else deficit / self.refill_rate

    async def acquire(self, n: int) -> float:
        """Block until at least ``n`` tokens are available; return the seconds slept. Does **not** deduct
        tokens — the bucket is a read-only model of server-side capacity; call ``report_rpc_duration()``
        after the RPC to sync. The lock is released during ``asyncio.sleep`` so other callers interleave,
        and the loop re-checks after each sleep in case the deadline was pessimistic. Signal's server is
        ground truth and will 429 (→ requeue) if the model drifts."""
        if n <= 0:
            return 0.0
        if n > self.capacity:
            raise SignalSchedulerError(f"Signal scheduler was called requesting {n} tokens (max is {self.capacity})")
        total_slept, first_pass = 0.0, True
        while True:
            async with self._lock:
                self._refill()
                if self.tokens >= n:
                    if not first_pass or total_slept > 0:
                        logger.debug("Signal scheduler: tokens sufficient for %d (remaining=%.1f, total_slept=%.1fs)",
                                     n, self.tokens, total_slept)
                    return total_slept
                deficit = n - self.tokens
            wait = deficit / self.refill_rate
            if first_pass:
                logger.info("Signal scheduler: pausing %.1fs for %d tokens (available=%.1f, deficit=%.1f, "
                            "refill=%.4f/s ≈ %.1fs/token)", wait, n, self.tokens, deficit, self.refill_rate,
                            1.0 / self.refill_rate)
                first_pass = False
            await asyncio.sleep(wait)
            total_slept += wait

    async def report_rpc_duration(self, rpc_duration: float, n_attachments: int) -> None:
        """Deduct ``n_attachments`` tokens for a completed send RPC. No refill is credited for the upload
        window: Signal's server checks the bucket at RPC start and resumes refill only after the response,
        so crediting it causes cumulative drift that eventually triggers 429s. Advances ``last_refill``."""
        if n_attachments <= 0:
            return
        async with self._lock:
            token_before = self.tokens
            self.tokens = max(0.0, token_before - float(n_attachments))
            self.last_refill = time.monotonic()
        logger.log(logging.INFO if rpc_duration > 10 and n_attachments > 5 else logging.DEBUG,
                   "Signal scheduler: RPC for %d att took %.1fs — tokens %.1f → %.1f (deducted=%d, no upload refill "
                   "credited, refill=%.4fs⁻¹)", n_attachments, rpc_duration, token_before, self.tokens, n_attachments,
                   self.refill_rate)

    def feedback(self, retry_after: Optional[float], n_attempted: int) -> None:
        """Apply server feedback after a 429: empty the bucket and, when ``retry_after`` (per-token refill
        window) is present, calibrate ``refill_rate`` from it."""
        if retry_after and retry_after > 0 and (new_rate := 1.0 / float(retry_after)) != self.refill_rate:
            logger.info("Signal scheduler: calibrating refill_rate to %.4f tokens/sec (server retry_after=%.1fs "
                        "per token)", new_rate, retry_after)
            self.refill_rate = new_rate
        self.tokens = 0.0
        self.last_refill = time.monotonic()

    def state(self) -> dict:
        """Current scheduler state for diagnostic logging (read-only, doesn't advance ``last_refill``)."""
        return {"tokens": round(self._projected_tokens(), 1), "capacity": int(self.capacity),
                "refill_rate": round(self.refill_rate, 4),
                "refill_seconds_per_token": round(1.0 / self.refill_rate, 1) if self.refill_rate > 0 else float("inf")}


_scheduler: Optional[SignalAttachmentScheduler] = None


def get_scheduler() -> SignalAttachmentScheduler:
    """Return the process-wide scheduler, creating it on first access."""
    global _scheduler
    if _scheduler is None:
        _scheduler = SignalAttachmentScheduler()
        logger.info("Signal scheduler: created (capacity=%d tokens, refill=%.4f/s ≈ %.1fs/token)",
                    int(_scheduler.capacity), _scheduler.refill_rate, 1.0 / _scheduler.refill_rate)
    return _scheduler


def _reset_scheduler() -> None:
    """Drop the cached scheduler so the next ``get_scheduler`` builds a fresh one. Test-only."""
    global _scheduler
    _scheduler = None
