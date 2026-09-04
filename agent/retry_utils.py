"""Retry utilities — jittered backoff for decorrelated retries.

Replaces fixed exponential backoff with jittered delays to prevent
thundering-herd retry spikes when multiple sessions hit the same
rate-limited provider concurrently.
"""

import random
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Optional

# Monotonic counter for jitter seed uniqueness within the same process.
# Protected by a lock to avoid race conditions in concurrent retry paths
# (e.g. multiple gateway sessions retrying simultaneously).
_jitter_counter = 0
_jitter_lock = threading.Lock()

# Z.AI Coding Plan's GLM-5.2 endpoint often returns HTTP 429 code 1305
# ("The service may be temporarily overloaded...") for otherwise valid
# Hermes requests. Short retries tend to hammer the same overloaded window;
# after a few normal retries, progressively widen the wait window. Keep the
# cap interactive-friendly: a simple TUI message should fail visibly in minutes,
# not sit silent for 20+ minutes.
_ZAI_CODING_OVERLOAD_LONG_BACKOFF = (30.0, 60.0, 90.0, 120.0)

# Number of initial short retries before the adaptive long-backoff tier kicks
# in. Shared by ``adaptive_rate_limit_backoff`` (which walks the long table
# starting at attempt ``short_attempts + 1``) and
# ``zai_coding_overload_retry_ceiling`` (which sizes the retry loop so every
# long-tier entry is reachable). Keeping it a single module constant prevents
# the two from silently desyncing if the short-retry count is ever tuned.
_ZAI_CODING_OVERLOAD_SHORT_ATTEMPTS = 3


def parse_retry_after_seconds(value_or_headers: Any) -> Optional[float]:
    """Parse a ``Retry-After`` value into non-negative seconds.

    Accepts either a raw header value (numeric string / HTTP-date / number)
    or a headers mapping, in which case the ``Retry-After`` key is looked up
    case-insensitively (``.get`` on dict-like objects tries both common
    casings; real HTTP header containers like httpx/requests are already
    case-insensitive).

    Returns:
        Seconds as a ``float`` (negative deltas clamped to ``0.0``), or
        ``None`` when the header is absent or unparseable.
    """
    raw = value_or_headers
    if raw is not None and not isinstance(raw, (str, int, float)):
        # Looks like a headers mapping — pull the header out of it.
        getter = getattr(raw, "get", None)
        if callable(getter):
            try:
                value = getter("Retry-After")
                if value is None:
                    value = getter("retry-after")
            except Exception:
                return None
            raw = value
        else:
            return None
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return max(0.0, float(raw))
    text = str(raw).strip()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except (TypeError, ValueError):
        pass
    # HTTP-date form (RFC 7231): seconds until that instant, clamped at 0.
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


def jittered_backoff(
    attempt: int,
    *,
    base_delay: float = 5.0,
    max_delay: float = 120.0,
    jitter_ratio: float = 0.5,
) -> float:
    """Compute a jittered exponential backoff delay.

    Args:
        attempt: 1-based retry attempt number.
        base_delay: Base delay in seconds for attempt 1.
        max_delay: Maximum delay cap in seconds.
        jitter_ratio: Fraction of computed delay to use as random jitter
            range.  0.5 means jitter is uniform in [0, 0.5 * delay].

    Returns:
        Delay in seconds: min(base * 2^(attempt-1), max_delay) + jitter.

    The jitter decorrelates concurrent retries so multiple sessions
    hitting the same provider don't all retry at the same instant.
    """
    global _jitter_counter
    with _jitter_lock:
        _jitter_counter += 1
        tick = _jitter_counter

    exponent = max(0, attempt - 1)
    if exponent >= 63 or base_delay <= 0:
        delay = max_delay
    else:
        delay = min(base_delay * (2 ** exponent), max_delay)

    # Seed from time + counter for decorrelation even with coarse clocks.
    seed = (time.time_ns() ^ (tick * 0x9E3779B9)) & 0xFFFFFFFF
    rng = random.Random(seed)
    jitter = rng.uniform(0, jitter_ratio * delay)

    return delay + jitter


def _error_text(error: Any) -> str:
    """Best-effort flattened provider error text for retry classification."""
    parts = [
        error,
        getattr(error, "message", None),
        getattr(error, "body", None),
        getattr(error, "response", None),
    ]
    return " ".join(str(part) for part in parts if part is not None).lower()


# B.AI (api.b.ai) account-level 429 code 1302 ("您的账户已达到速率限制，请您控制请求频率").
# Short 2–5s retries deepen the hole; skip them and walk the long table immediately.
# Cap the shared in-process cooldown so concurrent subagents wait instead of
# stampeding the same account.
# Start at 60s so the wait outlasts the credential-pool sole-key 429 TTL
# (EXHAUSTED_TTL_SOLE_CREDENTIAL_SECONDS = 60); a 30s retry would bounce
# off a still-benched key and look like another 429.
_BAI_ACCOUNT_RATE_LIMIT_LONG_BACKOFF = (60.0, 90.0, 120.0, 180.0)
_BAI_COOLDOWN_CAP_SECONDS = 180.0
_bai_cooldown_until = 0.0
_bai_cooldown_lock = threading.Lock()


def is_bai_endpoint(*, base_url: str | None, provider: str | None = None) -> bool:
    """True for the B.AI OpenAI-compatible endpoint used by Hermes ``b-ai``."""
    base = (base_url or "").lower()
    prov = (provider or "").lower()
    return prov == "b-ai" or "api.b.ai" in base


def is_bai_account_rate_limit_error(*, base_url: str | None, model: str | None, error: Any) -> bool:
    """Return True for B.AI account-level 429s (code 1302).

    Unlike Z.AI coding-plan overload (1305, often clears in seconds), 1302 is
    an account frequency cap. Fast retries make it worse, so callers should
    use the long backoff schedule from attempt 1.
    """
    del model  # matcher is endpoint + body, not model family
    status = getattr(error, "status_code", None)
    text = _error_text(error)
    return (
        status == 429
        and is_bai_endpoint(base_url=base_url)
        and ("1302" in text or "达到速率限制" in text)
    )


def record_bai_rate_limit_cooldown(seconds: float) -> None:
    """Extend the in-process B.AI cooldown so sibling subagents wait too."""
    global _bai_cooldown_until
    try:
        wait = float(seconds)
    except (TypeError, ValueError):
        return
    if wait <= 0:
        return
    until = time.time() + min(wait, _BAI_COOLDOWN_CAP_SECONDS)
    with _bai_cooldown_lock:
        if until > _bai_cooldown_until:
            _bai_cooldown_until = until


def bai_rate_limit_cooldown_remaining() -> float:
    """Seconds left on the in-process B.AI cooldown, or 0 if idle."""
    with _bai_cooldown_lock:
        return max(0.0, _bai_cooldown_until - time.time())


def is_zai_coding_overload_error(*, base_url: str | None, model: str | None, error: Any) -> bool:
    """Return True for Z.AI Coding Plan transient overload 429s.

    The coding-plan endpoint reports overload as HTTP 429 with body code 1305
    and message "The service may be temporarily overloaded...". Treat only
    that narrow shape specially so ordinary quota/billing 429s still fail fast
    through the existing classifier.
    """
    base = (base_url or "").lower()
    model_name = (model or "").lower()
    status = getattr(error, "status_code", None)
    text = _error_text(error)
    return (
        status == 429
        and "api.z.ai/api/coding/paas/v4" in base
        and "glm-5.2" in model_name
        and ("1305" in text or "temporarily overloaded" in text)
    )


def _long_backoff_wait(base_delay: float) -> float:
    # A smaller jitter ratio keeps long waits readable while still avoiding
    # synchronized retry storms across concurrent Hermes sessions.
    return jittered_backoff(1, base_delay=base_delay, max_delay=base_delay, jitter_ratio=0.2)


def adaptive_rate_limit_backoff(
    attempt: int,
    *,
    base_url: str | None,
    model: str | None,
    error: Any,
    default_wait: float,
    short_attempts: int = _ZAI_CODING_OVERLOAD_SHORT_ATTEMPTS,
) -> tuple[float, str | None]:
    """Provider-aware rate-limit backoff.

    For most providers this returns ``default_wait`` unchanged.

    * Z.AI Coding Plan GLM-5.2 overloads: keep the first ``short_attempts``
      retries on the normal short exponential schedule, then switch to
      progressively longer waits (30s → 60s → 90s → 120s) plus light jitter.
    * B.AI account 429 (code 1302): skip the short tier — those retries
      deepen the frequency cap — and walk the long table from attempt 1.

    ``attempt`` is 1-based, matching the retry loop's logged attempt number.
    Returns ``(wait_seconds, reason_label)`` where ``reason_label`` is suitable
    for status/log decoration when a provider-specific policy fired.
    """
    if is_bai_account_rate_limit_error(base_url=base_url, model=model, error=error):
        idx = min(max(attempt, 1) - 1, len(_BAI_ACCOUNT_RATE_LIMIT_LONG_BACKOFF) - 1)
        wait = _long_backoff_wait(_BAI_ACCOUNT_RATE_LIMIT_LONG_BACKOFF[idx])
        record_bai_rate_limit_cooldown(wait)
        return wait, "bai_account_rate_limit"

    if not is_zai_coding_overload_error(base_url=base_url, model=model, error=error):
        return default_wait, None
    if attempt <= short_attempts:
        return default_wait, "zai_coding_overload_short"

    idx = min(attempt - short_attempts - 1, len(_ZAI_CODING_OVERLOAD_LONG_BACKOFF) - 1)
    return _long_backoff_wait(_ZAI_CODING_OVERLOAD_LONG_BACKOFF[idx]), "zai_coding_overload_long"


def zai_coding_overload_retry_ceiling(short_attempts: int = _ZAI_CODING_OVERLOAD_SHORT_ATTEMPTS) -> int:
    """Retry-loop ceiling needed for the full Z.AI overload backoff schedule.

    The adaptive policy runs ``short_attempts`` short retries, then walks the
    long-backoff table one entry per subsequent attempt. The retry loop gives
    up as soon as ``retry_count >= ceiling`` — and that check runs *before* the
    attempt's backoff is computed — so the ceiling must sit one past the final
    long-backoff entry for every long tier to actually execute.

    With the default ``api_max_retries`` (3) equal to ``short_attempts`` (3),
    the loop always gave up before reaching the long tier, leaving the whole
    long-backoff schedule as dead code. Callers extend the ceiling to this
    value for Z.AI Coding overload 429s so the 30/60/90/120s waits run.
    """
    return short_attempts + len(_ZAI_CODING_OVERLOAD_LONG_BACKOFF) + 1


def bai_account_rate_limit_retry_ceiling() -> int:
    """Retry-loop ceiling so every B.AI 1302 long-backoff entry can run.

    B.AI skips the short tier, so the ceiling is one past the last long-table
    entry. Same give-up-before-backoff invariant as
    ``zai_coding_overload_retry_ceiling``.
    """
    return len(_BAI_ACCOUNT_RATE_LIMIT_LONG_BACKOFF) + 1
