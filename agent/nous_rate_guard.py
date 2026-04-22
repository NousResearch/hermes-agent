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
