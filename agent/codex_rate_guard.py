"""Cross-session usage-limit guard for OpenAI Codex OAuth.

Codex OAuth usage limits are account/plan windows, not per-process failures.
When one Hermes session hits ``usage_limit_reached`` and other sessions keep
retrying, every retry wastes allowance and can look like abusive traffic.  This
guard records the reset time in ``$HERMES_HOME/rate_limits/codex.json`` so CLI,
gateway, cron, and auxiliary calls can skip doomed requests until the window
resets.

This is deliberately *not* an evasion layer: it does not rotate IPs, accounts,
or tokens.  It only honors reset/backoff signals and prevents retry storms for
the single Codex OAuth account the user chose to use.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from typing import Any, Mapping, Optional

from utils import atomic_replace

logger = logging.getLogger(__name__)

_STATE_SUBDIR = "rate_limits"
_STATE_FILENAME = "codex.json"
_LANE_STATE_FILENAMES = {
    "all": _STATE_FILENAME,
    "standard": "codex-standard.json",
    "spark": "codex-spark.json",
}

# Short Retry-After windows are usually transient transport/provider jitter.
# Only trip the cross-session breaker for meaningful windows or explicit
# Codex usage-limit payloads.
_MIN_RESET_FOR_BREAKER_SECONDS = 60.0


def _lane_for_model(model: Optional[str] = None) -> str:
    """Return the Codex OAuth usage lane for *model*."""
    if not model:
        return "all"
    return "spark" if "spark" in str(model).lower() else "standard"


def _state_path(lane: str = "all") -> str:
    """Return the path to the Codex rate-limit state file."""
    try:
        from hermes_constants import get_hermes_home

        base = get_hermes_home()
    except ImportError:
        base = os.path.join(os.path.expanduser("~"), ".hermes")
    filename = _LANE_STATE_FILENAMES.get(lane, _STATE_FILENAME)
    return os.path.join(base, _STATE_SUBDIR, filename)


def _coerce_float(value: Any) -> Optional[float]:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_reset_at(value: Any, *, now: float) -> Optional[float]:
    """Parse a reset timestamp or seconds-from-now value."""
    parsed = _coerce_float(value)
    if parsed is None:
        return None
    if parsed > now:
        return parsed
    if parsed > 0:
        return now + parsed
    return None


def _headers_reset_at(headers: Optional[Mapping[str, str]], *, now: float) -> Optional[float]:
    if not headers:
        return None
    lowered = {str(k).lower(): v for k, v in headers.items()}

    # Prefer longer account windows when available.
    for key in (
        "x-ratelimit-reset-requests-1h",
        "x-ratelimit-reset-tokens-1h",
        "x-ratelimit-reset-requests",
        "x-ratelimit-reset-tokens",
        "retry-after",
    ):
        reset_at = _parse_reset_at(lowered.get(key), now=now)
        if reset_at is not None:
            return reset_at
    return None


def _context_reset_at(error_context: Optional[dict[str, Any]], *, now: float) -> Optional[float]:
    if not isinstance(error_context, dict):
        return None
    for key in ("reset_at", "resets_at", "resets_in_seconds", "retry_after"):
        reset_at = _parse_reset_at(error_context.get(key), now=now)
        if reset_at is not None:
            return reset_at
    return None


def is_codex_usage_limit_context(error_context: Optional[dict[str, Any]] = None) -> bool:
    """Return True for Codex account/plan usage-limit payloads."""
    if not isinstance(error_context, dict):
        return False
    reason = str(error_context.get("reason") or "").lower()
    message = str(error_context.get("message") or "").lower()
    combined = f"{reason} {message}"
    return any(
        needle in combined
        for needle in (
            "usage_limit_reached",
            "usage limit reached",
            "usage limit has been reached",
            "agentic usage limit",
        )
    )


def should_record_codex_rate_limit(
    *,
    headers: Optional[Mapping[str, str]] = None,
    error_context: Optional[dict[str, Any]] = None,
    now: Optional[float] = None,
) -> bool:
    """Decide whether a 429 should trip the cross-session breaker."""
    now = time.time() if now is None else now
    if is_codex_usage_limit_context(error_context):
        return True
    reset_at = _headers_reset_at(headers, now=now) or _context_reset_at(error_context, now=now)
    return bool(reset_at and (reset_at - now) >= _MIN_RESET_FOR_BREAKER_SECONDS)


def record_codex_rate_limit(
    *,
    headers: Optional[Mapping[str, str]] = None,
    error_context: Optional[dict[str, Any]] = None,
    model: Optional[str] = None,
    default_cooldown: float = 4 * 3600.0,
) -> Optional[float]:
    """Record Codex OAuth as temporarily unavailable.

    Returns the seconds remaining if a breaker was written, otherwise ``None``.
    The guard records explicit Codex usage-limit errors even when no reset time
    is present, using ``default_cooldown`` as a conservative fallback.
    """
    now = time.time()
    if not should_record_codex_rate_limit(
        headers=headers,
        error_context=error_context,
        now=now,
    ):
        return None

    reset_at = (
        _context_reset_at(error_context, now=now)
        or _headers_reset_at(headers, now=now)
        or (now + default_cooldown)
    )
    remaining = max(0.0, reset_at - now)

    lane = _lane_for_model(model)
    path = _state_path(lane)
    try:
        state_dir = os.path.dirname(path)
        os.makedirs(state_dir, exist_ok=True)
        state = {
            "lane": lane,
            "model": model,
            "reset_at": reset_at,
            "recorded_at": now,
            "reset_seconds": remaining,
            "reason": (error_context or {}).get("reason"),
            "message": (error_context or {}).get("message"),
        }
        fd, tmp_path = tempfile.mkstemp(dir=state_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f)
            atomic_replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        logger.info(
            "Codex OAuth usage guard recorded for %s lane: resets in %.0fs",
            lane,
            remaining,
        )
        return remaining
    except Exception as exc:
        logger.debug("Failed to write Codex rate-limit state: %s", exc)
        return remaining


def extract_error_context_from_exception(exc: BaseException) -> dict[str, Any]:
    """Extract Codex usage-limit context from an SDK/API exception."""
    context: dict[str, Any] = {}
    body = getattr(exc, "body", None)
    payload = None
    if isinstance(body, dict):
        payload = body.get("error") if isinstance(body.get("error"), dict) else body
    if isinstance(payload, dict):
        reason = payload.get("code") or payload.get("type") or payload.get("error")
        if isinstance(reason, str) and reason.strip():
            context["reason"] = reason.strip()
        message = payload.get("message") or payload.get("error_description")
        if isinstance(message, str) and message.strip():
            context["message"] = message.strip()
        for key in ("reset_at", "resets_at", "resets_in_seconds", "retry_after"):
            if payload.get(key) not in {None, ""}:
                context[key] = payload.get(key)
    if "message" not in context:
        raw = str(exc).strip()
        if raw:
            context["message"] = raw[:500]
    return context


def record_codex_rate_limit_from_exception(
    exc: BaseException,
    *,
    model: Optional[str] = None,
    default_cooldown: float = 4 * 3600.0,
) -> Optional[float]:
    """Extract headers/body from an exception and record a Codex breaker."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    return record_codex_rate_limit(
        headers=headers,
        error_context=extract_error_context_from_exception(exc),
        model=model,
        default_cooldown=default_cooldown,
    )


def _remaining_from_path(path: str) -> Optional[float]:
    try:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        reset_at = float(state.get("reset_at", 0))
        remaining = reset_at - time.time()
        if remaining > 0:
            return remaining
        try:
            os.unlink(path)
        except OSError:
            pass
        return None
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def codex_rate_limit_remaining(*, model: Optional[str] = None) -> Optional[float]:
    """Return seconds until Codex OAuth reset, or ``None`` when available."""
    # A global/unknown breaker applies to every Codex lane.
    remaining = _remaining_from_path(_state_path("all"))
    if remaining is not None:
        return remaining
    lane = _lane_for_model(model)
    if lane == "all":
        return None
    return _remaining_from_path(_state_path(lane))


def clear_codex_rate_limit(*, model: Optional[str] = None) -> None:
    """Clear Codex guard state after a successful Codex request."""
    lane = _lane_for_model(model)
    paths = [_state_path("all")]
    if lane != "all":
        paths.append(_state_path(lane))
    for path in paths:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.debug("Failed to clear Codex rate-limit state %s: %s", path, exc)


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
