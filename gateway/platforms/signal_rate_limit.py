"""
Signal attachment rate-limit scheduler.

Process-wide token-bucket simulator that mirrors the per-account
attachment rate limit signal-cli/Signal-Server enforce. Producers
(``SignalAdapter.send_multiple_images`` and the ``send_message`` tool's
Signal path) call ``acquire(n)`` before an attachment send; on a 429
they call ``feedback(retry_after, n)`` so the model recalibrates from
the server's authoritative hint.

The scheduler serializes concurrent calls through an ``asyncio.Lock``,
giving FIFO fairness across agent sessions sharing one signal-cli
daemon.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIGNAL_MAX_ATTACHMENTS_PER_MSG = 32  # per-message attachment cap (source: Signal-{Android,Desktop} source code)
SIGNAL_RATE_LIMIT_BUCKET_CAPACITY = 50  # server-side token-bucket capacity for attachments rate limiting
SIGNAL_RATE_LIMIT_DEFAULT_RETRY_AFTER = 4  # fallback token refill interval for signal-cli < v0.14.3
SIGNAL_RATE_LIMIT_MAX_ATTEMPTS = 2  # initial attempt + 1 retry
SIGNAL_BATCH_PACING_NOTICE_THRESHOLD = 10.0  # if estimated waiting time > 10s, notify the user about the delay
SIGNAL_RPC_ERROR_RATELIMIT = -5  # signal-cli (v0.14.3+) JSON-RPC error code for RateLimitException


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class SignalRateLimitError(Exception):
    """
    Raised by ``SignalAdapter._rpc`` for rate-limit responses when the
    caller has opted in via ``raise_on_rate_limit=True``.

    Carries the server-supplied per-token Retry-After (in seconds) on
    signal-cli ≥ v0.14.3
    ``retry_after`` is None when the version doesn't expose it.
    """

    def __init__(self, message: str, retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class SignalSchedulerError(Exception):
    pass

# ---------------------------------------------------------------------------
# Detection helpers — used to fish a 429 out of signal-cli's various error
# shapes (typed code, [429] substring, libsignal-net RetryLaterException
# leaked through AttachmentInvalidException).
# ---------------------------------------------------------------------------

# "Retry after 4 seconds" / "retry after 4 second" — libsignal-net's
# RetryLaterException string form, surfaced when 429s hit during
# attachment upload (signal-cli wraps these as AttachmentInvalidException
# rather than RateLimitException, so the typed path doesn't fire).
_RETRY_AFTER_RE = re.compile(r"Retry after (\d+(?:\.\d+)?)\s*second", re.IGNORECASE)


def _extract_retry_after_seconds(err: Any) -> Optional[float]:
    """Pull the per-token Retry-After window from a signal-cli rate-limit error.

    Tries two sources, in order:
    1. ``error.data.response.results[*].retryAfterSeconds`` — the
       structured field signal-cli ≥ v0.14.3 surfaces for plain
       RateLimitException.
    2. ``"Retry after N seconds"`` parsed out of the message — covers
       libsignal-net's RetryLaterException that gets wrapped as
       AttachmentInvalidException during attachment upload, where the
       structured field stays null.

    Returns None when neither yields a value.
    """
    msg = ""
    if isinstance(err, dict):
        data = err.get("data") or {}
        response = data.get("response") or {}
        results = response.get("results") or []
        candidates = [
            r.get("retryAfterSeconds") for r in results
            if isinstance(r, dict) and r.get("retryAfterSeconds")
        ]
        if candidates:
            return float(max(candidates))
        msg = str(err.get("message", ""))
    else:
        msg = str(err)
    match = _RETRY_AFTER_RE.search(msg)
    return float(match.group(1)) if match else None


def _is_signal_rate_limit_error(err: Any) -> bool:
    """True if a signal-cli RPC error reflects a rate-limit failure.

    Matches three layers:
    - typed ``RATELIMIT_ERROR`` code (signal-cli ≥ v0.14.3, plain
      RateLimitException)
    - legacy ``[429] / RateLimitException`` substrings
    - libsignal-net's ``RetryLaterException`` / ``Retry after N seconds``
      surfaced inside ``AttachmentInvalidException`` when the rate
      limit is hit during attachment upload — signal-cli never re-tags
      these as RateLimitException, so substring is the only signal.
    """
    if isinstance(err, dict) and err.get("code") == SIGNAL_RPC_ERROR_RATELIMIT:
        return True

    message = (
        str(err.get("message", ""))
        if isinstance(err, dict)
        else str(err)
    )
    msg_lower = message.lower()
    return (
        "[429]" in message
        or "ratelimit" in msg_lower
        or "retrylaterexception" in msg_lower
        or "retry after" in msg_lower
    )


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def _format_wait(seconds: float) -> str:
    """Human-friendly wait label for user-facing pacing notices."""
    s = max(0.0, seconds)
    if s < 90:
        return f"{int(round(s))}s"
    return f"{max(1, int(round(s / 60)))} min"


def _signal_send_timeout(num_attachments: int) -> float:
    """HTTP timeout for a Signal ``send`` RPC.

    signal-cli uploads attachments serially during the call, so the
    server-side time scales with batch size. Default 30s is fine for
    text-only sends but truncates large attachment batches mid-upload —
    we then log a phantom failure even though signal-cli completes the
    send a few seconds later. Scale at 5s/attachment with a 60s floor.
    """
    if num_attachments <= 0:
        return 30.0
    return max(60.0, 5.0 * num_attachments)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class SignalAttachmentScheduler:
    """Process-wide token-bucket simulator for Signal attachment sends.

    The bucket holds up to ``capacity`` tokens (default 50, matching
    Signal's server-side rate-limit bucket size). Each attachment consumes one
    token. Tokens refill at ``refill_rate`` tokens/second, calibrated
    from the per-token Retry-After hint we get from the server when a
    429 fires. Until we've observed one, we use the documented default
    (1 token / 4 seconds).

    Concurrent ``acquire(n)`` calls serialize through an
    ``asyncio.Lock`` — natural FIFO across agent sessions hitting the
    same daemon.
    """

    def __init__(
        self,
        capacity: float = float(SIGNAL_RATE_LIMIT_BUCKET_CAPACITY),
        default_retry_after: float = float(SIGNAL_RATE_LIMIT_DEFAULT_RETRY_AFTER),
    ) -> None:
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.refill_rate = 1.0 / float(default_retry_after)
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        if elapsed > 0 and self.tokens < self.capacity:
            self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_wait(self, n: int) -> float:
        """Best-effort estimate of the seconds until ``n`` tokens would
        be available. Used to decide whether to emit a user-facing
        pacing notice *before* committing to an ``acquire`` that may
        block silently. Lock-free; small races vs. concurrent acquires
        are benign for an informational notice.
        """
        now = time.monotonic()
        elapsed = now - self.last_refill
        projected = self.tokens
        if elapsed > 0 and projected < self.capacity:
            projected = min(self.capacity, projected + elapsed * self.refill_rate)
        deficit = n - projected
        if deficit <= 0:
            return 0.0
        return deficit / self.refill_rate

    async def acquire(self, n: int) -> float:
        """Block until at least ``n`` tokens are available, return the
        seconds slept.

        Does **not** deduct tokens — the bucket is a read-only model of
        server-side capacity.  Call ``report_rpc_duration()`` after the
        RPC to synchronise the model with the server timeline.

        Not perfect in case lots of coroutines try to acquire for big
        uploads (``report_rpc_duration`` will take a long time to get hit)
        but this is just a simulation. Signal server is ground truth and
        will raise rate-limit exceptions triggering requeues.

        The lock is released during ``asyncio.sleep`` so other callers
        can interleave.  A retry loop re-checks after each sleep in
        case the deadline was pessimistic.
        """
        if n <= 0:
            return 0.0
        if n > self.capacity:
            raise SignalSchedulerError(
                f"Signal scheduler was called requesting {n} tokens "
                f"(max is {self.capacity})",
            )

        total_slept = 0.0
        first_pass = True
        while True:
            async with self._lock:
                self._refill()
                if self.tokens >= n:
                    if not first_pass or total_slept > 0:
                        logger.debug(
                            "Signal scheduler: tokens sufficient for %d "
                            "(remaining=%.1f, total_slept=%.1fs)",
                            n, self.tokens, total_slept,
                        )
                    return total_slept
                deficit = n - self.tokens
            wait = deficit / self.refill_rate
            if first_pass:
                logger.info(
                    "Signal scheduler: pausing %.1fs for %d tokens "
                    "(available=%.1f, deficit=%.1f, refill=%.4f/s ≈ %.1fs/token)",
                    wait, n, self.tokens, deficit,
                    self.refill_rate, 1.0 / self.refill_rate,
                )
                first_pass = False
            await asyncio.sleep(wait)
            total_slept += wait

    async def report_rpc_duration(self, rpc_duration: float, n_attachments: int) -> None:
        """Record an attachment-send RPC that just completed.

        Deducts ``n_attachments`` tokens without crediting refill during
        the upload window. Signal's server checks the bucket at RPC start
        and does *not* refill during request processing — refill resumes
        after the response. Crediting upload-time refill causes cumulative
        drift that eventually triggers 429s.

        Advances ``last_refill`` so the next ``acquire`` / ``_refill``
        starts counting from this point.
        """
        if n_attachments <= 0:
            return

        async with self._lock:
            now = time.monotonic()
            token_before = self.tokens
            self.tokens = max(0.0, token_before - float(n_attachments))
            self.last_refill = now
        logger.log(
            logging.INFO if rpc_duration > 10 and n_attachments > 5 else logging.DEBUG,
            "Signal scheduler: RPC for %d att took %.1fs — "
            "tokens %.1f → %.1f (deducted=%d, no upload refill credited, refill=%.4fs⁻¹)",
            n_attachments, rpc_duration,
            token_before, self.tokens,
            n_attachments, self.refill_rate,
        )

    def feedback(self, retry_after: Optional[float], n_attempted: int) -> None:
        """Apply server feedback after a 429.

        ``retry_after`` is the per-*token* refill window the server
        reports (None when signal-cli is older than v0.14.3 and didn't
        surface it).

        When present we calibrate ``refill_rate`` from it:
        the server is authoritative.
        """
        if retry_after and retry_after > 0:
            new_rate = 1.0 / float(retry_after)
            if new_rate != self.refill_rate:
                logger.info(
                    "Signal scheduler: calibrating refill_rate to %.4f tokens/sec "
                    "(server retry_after=%.1fs per token)",
                    new_rate, retry_after,
                )
                self.refill_rate = new_rate
        self.tokens = 0.0
        self.last_refill = time.monotonic()

    def state(self) -> dict:
        """Return current scheduler state for diagnostic logging (read-only).

        Does not advance ``last_refill`` — safe to call from logging paths
        without perturbing the bucket.
        """
        now = time.monotonic()
        elapsed = now - self.last_refill
        projected = self.tokens
        if elapsed > 0 and projected < self.capacity:
            projected = min(self.capacity, projected + elapsed * self.refill_rate)
        return {
            "tokens": round(projected, 1),
            "capacity": int(self.capacity),
            "refill_rate": round(self.refill_rate, 4),
            "refill_seconds_per_token": round(1.0 / self.refill_rate, 1) if self.refill_rate > 0 else float("inf"),
        }


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------

_scheduler: Optional[SignalAttachmentScheduler] = None


def get_scheduler() -> SignalAttachmentScheduler:
    """Return the process-wide scheduler, creating it on first access."""
    global _scheduler
    if _scheduler is None:
        _scheduler = SignalAttachmentScheduler()
        logger.info(
            "Signal scheduler: created (capacity=%d tokens, refill=%.4f/s ≈ %.1fs/token)",
            int(_scheduler.capacity),
            _scheduler.refill_rate,
            1.0 / _scheduler.refill_rate,
        )
    return _scheduler


def _reset_scheduler() -> None:
    """Drop the cached scheduler so the next ``get_scheduler`` call
    builds a fresh one. Test-only — never call from production paths."""
    global _scheduler
    _scheduler = None
