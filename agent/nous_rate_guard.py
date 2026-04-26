"""Cross-session rate limit guard for Nous Portal.

Writes rate limit state to a shared file so all sessions (CLI, gateway,
cron, auxiliary) can check whether Nous Portal is currently rate-limited
before making requests.  Prevents retry amplification when RPH is tapped.

Each 429 from Nous triggers up to 9 API calls per conversation turn
(3 SDK retries x 3 Hermes retries), and every one of those calls counts
against RPH.  By recording the rate limit state on first 429 and checking
it before subsequent attempts, we eliminate the amplification effect.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

_STATE_SUBDIR = "rate_limits"
_STATE_FILENAME = "nous.json"


def _state_path() -> str:
    """Return the path to the Nous rate limit state file."""
    try:
        from hermes_constants import get_hermes_home
        base = get_hermes_home()
    except ImportError:
        base = os.path.join(os.path.expanduser("~"), ".hermes")
    return os.path.join(base, _STATE_SUBDIR, _STATE_FILENAME)


def _parse_reset_seconds(headers: Optional[Mapping[str, str]]) -> Optional[float]:
    """Extract the best available reset-time estimate from response headers.

    Priority:
      1. x-ratelimit-reset-requests-1h  (hourly RPH window — most useful)
      2. x-ratelimit-reset-requests     (per-minute RPM window)
      3. retry-after                     (generic HTTP header)

    Returns seconds-from-now, or None if no usable header found.
    """
    if not headers:
        return None

    lowered = {k.lower(): v for k, v in headers.items()}

    for key in (
        "x-ratelimit-reset-requests-1h",
        "x-ratelimit-reset-requests",
        "retry-after",
    ):
        raw = lowered.get(key)
        if raw is not None:
            try:
                val = float(raw)
                if val > 0:
                    return val
            except (TypeError, ValueError):
                pass

    return None


def record_nous_rate_limit(
    *,
    headers: Optional[Mapping[str, str]] = None,
    error_context: Optional[dict[str, Any]] = None,
    default_cooldown: float = 300.0,
) -> None:
    """Record that Nous Portal is rate-limited.

    Parses the reset time from response headers or error context.
    Falls back to ``default_cooldown`` (5 minutes) if no reset info
    is available.  Writes to a shared file that all sessions can read.

    Args:
        headers: HTTP response headers from the 429 error.
        error_context: Structured error context from _extract_api_error_context().
        default_cooldown: Fallback cooldown in seconds when no header data.
    """
    now = time.time()
    reset_at = None

    # Try headers first (most accurate)
    header_seconds = _parse_reset_seconds(headers)
    if header_seconds is not None:
        reset_at = now + header_seconds

    # Try error_context reset_at (from body parsing)
    if reset_at is None and isinstance(error_context, dict):
        ctx_reset = error_context.get("reset_at")
        if isinstance(ctx_reset, (int, float)) and ctx_reset > now:
            reset_at = float(ctx_reset)

    # Default cooldown
    if reset_at is None:
        reset_at = now + default_cooldown

    path = _state_path()
    try:
        state_dir = os.path.dirname(path)
        os.makedirs(state_dir, exist_ok=True)

        state = {
            "reset_at": reset_at,
            "recorded_at": now,
            "reset_seconds": reset_at - now,
        }

        # Atomic write: write to temp file + rename
        fd, tmp_path = tempfile.mkstemp(dir=state_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(state, f)
            os.replace(tmp_path, path)
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        logger.info(
            "Nous rate limit recorded: resets in %.0fs (at %.0f)",
            reset_at - now, reset_at,
        )
    except Exception as exc:
        logger.debug("Failed to write Nous rate limit state: %s", exc)


def nous_rate_limit_remaining() -> Optional[float]:
    """Check if Nous Portal is currently rate-limited.

    Returns:
        Seconds remaining until reset, or None if not rate-limited.
    """
    path = _state_path()
    try:
        with open(path) as f:
            state = json.load(f)
        reset_at = state.get("reset_at", 0)
        remaining = reset_at - time.time()
        if remaining > 0:
            return remaining
        # Expired — clean up
        try:
            os.unlink(path)
        except OSError:
            pass
        return None
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
        return None


def clear_nous_rate_limit() -> None:
    """Clear the rate limit state (e.g., after a successful Nous request)."""
    try:
        os.unlink(_state_path())
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.debug("Failed to clear Nous rate limit state: %s", exc)


def format_remaining(seconds: float) -> str:
    """Format seconds remaining into human-readable duration."""
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        m, sec = divmod(s, 60)
        return f"{m}m {sec}s" if sec else f"{m}m"
    h, remainder = divmod(s, 3600)
    m = remainder // 60
    return f"{h}h {m}m" if m else f"{h}h"


# Buckets with reset windows shorter than this are treated as transient
# (upstream jitter, secondary throttling) rather than a genuine quota
# exhaustion worth a cross-session breaker trip.
_MIN_RESET_FOR_BREAKER_SECONDS = 60.0


def is_genuine_nous_rate_limit(
    *,
    headers: Optional[Mapping[str, str]] = None,
    last_known_state: Optional[Any] = None,
) -> bool:
    """Decide whether a 429 from Nous Portal is a real account rate limit.

    Nous Portal multiplexes multiple upstream providers (DeepSeek, Kimi,
    MiMo, Hermes, ...) behind one endpoint.  A 429 can mean either:

      (a) The caller's own RPM / RPH / TPM / TPH bucket on Nous is
          exhausted — a genuine rate limit that will last until the
          bucket resets.
      (b) The upstream provider is out of capacity for a specific model
          — transient, clears in seconds, and has nothing to do with
          the caller's quota on Nous.

    Tripping the cross-session breaker on (b) blocks ALL Nous requests
    (and all models, since Nous is one provider key) for minutes even
    though the caller's account is healthy and a different model would
    have worked.  That's the bug users hit when DeepSeek V4 Pro 429s
    trigger a breaker that then blocks Kimi 2.6 and MiMo V2.5 Pro.

    We tell the two apart by looking at:

      1. The 429 response's own ``x-ratelimit-*`` headers.  Nous emits
         the full suite on every response including 429s.  An exhausted
         bucket (``remaining == 0`` with a reset window >= 60s) is
         proof of (a).
      2. The last-known-good rate-limit state captured by
         ``_capture_rate_limits()`` on the previous successful
         response.  If any bucket there was already near-exhausted with
         a substantial reset window, the current 429 is almost
         certainly (a) continuing from that condition.

    If neither signal fires, we treat the 429 as (b): fail the single
    request, let the retry loop or model-switch proceed, and do NOT
    write the cross-session breaker file.

    Returns True when the evidence points at (a).
    """
    # Signal 1: current 429 response headers.
    state = _parse_buckets_from_headers(headers)
    if _has_exhausted_bucket(state):
        return True

    # Signal 2: last-known-good state from a recent successful response.
    # Accepts either a RateLimitState (dataclass from rate_limit_tracker)
    # or a dict of bucket snapshots.
    if last_known_state is not None and _has_exhausted_bucket_in_object(last_known_state):
        return True

    return False


def _parse_buckets_from_headers(
    headers: Optional[Mapping[str, str]],
) -> dict[str, tuple[Optional[int], Optional[float]]]:
    """Extract (remaining, reset_seconds) per bucket from x-ratelimit-* headers.

    Returns empty dict when no rate-limit headers are present.
    """
    if not headers:
        return {}

    lowered = {k.lower(): v for k, v in headers.items()}
    if not any(k.startswith("x-ratelimit-") for k in lowered):
        return {}

    def _maybe_int(raw: Optional[str]) -> Optional[int]:
        if raw is None:
            return None
        try:
            return int(float(raw))
        except (TypeError, ValueError):
            return None

    def _maybe_float(raw: Optional[str]) -> Optional[float]:
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    result: dict[str, tuple[Optional[int], Optional[float]]] = {}
    for tag in ("requests", "requests-1h", "tokens", "tokens-1h"):
        remaining = _maybe_int(lowered.get(f"x-ratelimit-remaining-{tag}"))
        reset = _maybe_float(lowered.get(f"x-ratelimit-reset-{tag}"))
        if remaining is not None or reset is not None:
            result[tag] = (remaining, reset)
    return result


def _has_exhausted_bucket(
    buckets: Mapping[str, tuple[Optional[int], Optional[float]]],
) -> bool:
    """Return True when any bucket has remaining == 0 AND a meaningful reset window."""
    for remaining, reset in buckets.values():
        if remaining is None or remaining > 0:
            continue
        if reset is None:
            continue
        if reset >= _MIN_RESET_FOR_BREAKER_SECONDS:
            return True
    return False


def _has_exhausted_bucket_in_object(state: Any) -> bool:
    """Check a RateLimitState-like object for an exhausted bucket.

    Accepts the dataclass from ``agent.rate_limit_tracker`` (buckets
    exposed as attributes ``requests_min``, ``requests_hour``,
    ``tokens_min``, ``tokens_hour``) and falls back gracefully for any
    object missing those attributes.
    """
    for attr in ("requests_min", "requests_hour", "tokens_min", "tokens_hour"):
        bucket = getattr(state, attr, None)
        if bucket is None:
            continue
        limit = getattr(bucket, "limit", 0) or 0
        remaining = getattr(bucket, "remaining", 0) or 0
        # Prefer the adjusted "remaining_seconds_now" property when present;
        # fall back to raw reset_seconds.
        reset = getattr(bucket, "remaining_seconds_now", None)
        if reset is None:
            reset = getattr(bucket, "reset_seconds", 0.0) or 0.0
        if limit <= 0:
            continue
        if remaining > 0:
            continue
        if reset >= _MIN_RESET_FOR_BREAKER_SECONDS:
            return True
    return False
