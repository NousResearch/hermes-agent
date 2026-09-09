"""Cross-session rate limit guard for Nous Portal.

Writes rate limit state to a shared file so all sessions (CLI, gateway, cron,
auxiliary) can check whether Nous Portal is currently rate-limited before making
requests. Without it each 429 fans out into up to 9 calls per turn (3 SDK
retries x 3 Hermes retries), all counted against RPH.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from typing import Any, Mapping, Optional
from utils import atomic_write_text
from agent.rate_limit_tracker import (
    _BUCKET_TAGS, _fmt_seconds, _safe_float, _safe_int, has_rate_limit_headers, lower_headers,
)

logger = logging.getLogger(__name__)

# Reset windows shorter than this are transient upstream jitter, not a quota
# exhaustion worth a cross-session breaker trip.
_MIN_RESET_FOR_BREAKER_SECONDS = 60.0

format_remaining = _fmt_seconds


def _state_path() -> str:
    """Path to the Nous rate limit state file."""
    try:
        from hermes_constants import get_hermes_home
        base = get_hermes_home()
    except ImportError:
        base = os.path.join(os.path.expanduser("~"), ".hermes")
    return os.path.join(base, "rate_limits", "nous.json")


def _parse_reset_seconds(headers: Optional[Mapping[str, str]]) -> Optional[float]:
    """Best reset estimate (seconds from now) from hourly, per-minute, then retry-after headers."""
    lowered = lower_headers(headers)
    for key in ("x-ratelimit-reset-requests-1h", "x-ratelimit-reset-requests", "retry-after"):
        val = _safe_float(lowered.get(key), 0.0)
        if val > 0:
            return val
    return None


def record_nous_rate_limit(
    *, headers: Optional[Mapping[str, str]] = None, error_context: Optional[dict[str, Any]] = None,
    default_cooldown: float = 300.0,
) -> None:
    """Record that Nous Portal is rate-limited in the shared state file.

    Reset time comes from headers, then ``error_context["reset_at"]`` (body
    parsing), then ``default_cooldown``.
    """
    now = time.time()
    reset_at = None
    header_seconds = _parse_reset_seconds(headers)
    if header_seconds is not None:
        reset_at = now + header_seconds
    if reset_at is None and isinstance(error_context, dict):
        ctx_reset = error_context.get("reset_at")
        if isinstance(ctx_reset, (int, float)) and ctx_reset > now:
            reset_at = float(ctx_reset)
    if reset_at is None:
        reset_at = now + default_cooldown

    state = {"reset_at": reset_at, "recorded_at": now, "reset_seconds": reset_at - now}
    try:
        atomic_write_text(_state_path(), json.dumps(state))
        logger.info("Nous rate limit recorded: resets in %.0fs (at %.0f)", reset_at - now, reset_at)
    except Exception as exc:
        logger.debug("Failed to write Nous rate limit state: %s", exc)


def nous_rate_limit_remaining() -> Optional[float]:
    """Seconds remaining until reset, or None if not rate-limited (expired state is removed)."""
    path = _state_path()
    try:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        remaining = state.get("reset_at", 0) - time.time()
        if remaining > 0:
            return remaining
        with contextlib.suppress(OSError):
            os.unlink(path)
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


def _is_exhausted(remaining: Optional[int], reset: Optional[float]) -> bool:
    """remaining == 0 AND a reset window long enough to be a real quota exhaustion."""
    return (
        remaining is not None
        and remaining <= 0
        and reset is not None
        and reset >= _MIN_RESET_FOR_BREAKER_SECONDS
    )


def is_genuine_nous_rate_limit(
    *, headers: Optional[Mapping[str, str]] = None, last_known_state: Optional[Any] = None,
) -> bool:
    """Decide whether a 429 from Nous Portal is a real account rate limit.

    Nous multiplexes upstream providers behind one key, so a 429 may be one
    upstream model out of capacity (clears in seconds) rather than our quota;
    tripping the breaker on that would block every Nous model for minutes.
    Only an exhausted bucket (remaining == 0 with reset >= 60s) in the 429's
    own headers, or in the last-known-good ``RateLimitState``, is genuine.
    """
    if _has_exhausted_bucket(_parse_buckets_from_headers(headers)):
        return True
    return last_known_state is not None and _has_exhausted_bucket_in_object(last_known_state)


def _parse_buckets_from_headers(
    headers: Optional[Mapping[str, str]],
) -> dict[str, tuple[Optional[int], Optional[float]]]:
    """(remaining, reset_seconds) per bucket from x-ratelimit-* headers ({} if none)."""
    lowered = lower_headers(headers)
    if not has_rate_limit_headers(lowered):
        return {}
    result: dict[str, tuple[Optional[int], Optional[float]]] = {}
    for _attr, tag in _BUCKET_TAGS:
        remaining = _safe_int(lowered.get(f"x-ratelimit-remaining-{tag}"), None)
        reset = _safe_float(lowered.get(f"x-ratelimit-reset-{tag}"), None)
        if remaining is not None or reset is not None:
            result[tag] = (remaining, reset)
    return result


def _has_exhausted_bucket(buckets: Mapping[str, tuple[Optional[int], Optional[float]]]) -> bool:
    return any(_is_exhausted(remaining, reset) for remaining, reset in buckets.values())


def _has_exhausted_bucket_in_object(state: Any) -> bool:
    """Check a RateLimitState-like object (duck-typed; missing attrs are skipped)."""
    for attr, _tag in _BUCKET_TAGS:
        bucket = getattr(state, attr, None)
        if bucket is None or (getattr(bucket, "limit", 0) or 0) <= 0:
            continue
        remaining = getattr(bucket, "remaining", 0) or 0
        reset = getattr(bucket, "remaining_seconds_now", None)
        if reset is None:
            reset = getattr(bucket, "reset_seconds", 0.0) or 0.0
        if _is_exhausted(remaining, reset):
            return True
    return False


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import tempfile  # noqa: F401,E402


_PLUGIN_COMPAT_LAZY = {
    'atomic_replace': ('utils', 'atomic_replace'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
