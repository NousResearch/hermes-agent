"""Cross-session provider cooldown guard.

Records provider usage-cap/reset windows in a small shared state file so
gateway, CLI, cron, and auxiliary sessions can fail over without hammering a
known-capped primary provider.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from typing import Any, Optional

from utils import atomic_replace

logger = logging.getLogger(__name__)

_STATE_SUBDIR = "rate_limits"


def _hermes_home() -> str:
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home()
    except Exception:
        return os.path.join(os.path.expanduser("~"), ".hermes")


def _safe_provider_name(provider: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", (provider or "").strip().lower())
    return safe.strip("-") or "unknown"


def _state_path(provider: str) -> str:
    return os.path.join(_hermes_home(), _STATE_SUBDIR, f"{_safe_provider_name(provider)}.json")


def _parse_reset_at(raw: Any) -> Optional[float]:
    if raw in {None, ""}:
        return None
    now = time.time()
    if isinstance(raw, (int, float)):
        value = float(raw)
        if value > now:
            return value
        if value > 0:
            return now + value
        return None
    if isinstance(raw, str):
        value = raw.strip()
        if not value:
            return None
        try:
            numeric = float(value)
            if numeric > now:
                return numeric
            if numeric > 0:
                return now + numeric
        except ValueError:
            pass
        try:
            from datetime import datetime, timezone

            normalized = value.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(normalized)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            timestamp = parsed.timestamp()
            if timestamp > now:
                return timestamp
        except Exception:
            return None
    return None


def _reset_from_context(error_context: Optional[dict[str, Any]]) -> Optional[float]:
    if not isinstance(error_context, dict):
        return None
    for key in ("reset_at", "resets_at", "retry_after"):
        parsed = _parse_reset_at(error_context.get(key))
        if parsed is not None:
            return parsed
    return None


def record_provider_cooldown(
    provider: str,
    *,
    error_context: Optional[dict[str, Any]] = None,
    default_cooldown: float = 300.0,
) -> None:
    """Record a provider cooldown window.

    ``error_context`` may carry an absolute reset timestamp, ISO timestamp, or
    retry-after seconds. If none is present, a short default cooldown prevents
    immediate retry storms while avoiding a long false-positive outage.
    """

    provider = _safe_provider_name(provider)
    now = time.time()
    reset_at = _reset_from_context(error_context)
    if reset_at is None:
        reset_at = now + default_cooldown

    path = _state_path(provider)
    state_dir = os.path.dirname(path)
    try:
        os.makedirs(state_dir, exist_ok=True)
        state = {
            "provider": provider,
            "reset_at": reset_at,
            "recorded_at": now,
            "reset_seconds": reset_at - now,
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
            "Provider cooldown recorded: provider=%s resets_in=%.0fs",
            provider,
            reset_at - now,
        )
    except Exception as exc:
        logger.debug("Failed to write provider cooldown state for %s: %s", provider, exc)


def provider_cooldown_remaining(provider: str) -> Optional[float]:
    """Return remaining cooldown seconds, or None if provider is available."""

    path = _state_path(provider)
    try:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        remaining = float(state.get("reset_at", 0)) - time.time()
        if remaining > 0:
            return remaining
        try:
            os.unlink(path)
        except OSError:
            pass
        return None
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def clear_provider_cooldown(provider: str) -> None:
    """Clear a provider cooldown after a successful primary-provider request."""

    try:
        os.unlink(_state_path(provider))
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.debug("Failed to clear provider cooldown state for %s: %s", provider, exc)


def is_usage_limit_context(error_context: Optional[dict[str, Any]]) -> bool:
    if not isinstance(error_context, dict):
        return False
    reason = str(error_context.get("reason") or "").lower()
    message = str(error_context.get("message") or "").lower()
    return (
        "usage_limit" in reason
        or "usage limit has been reached" in message
        or "plan_type=plus" in message
    )


__all__ = [
    "clear_provider_cooldown",
    "is_usage_limit_context",
    "provider_cooldown_remaining",
    "record_provider_cooldown",
]
