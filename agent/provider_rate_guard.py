"""Generic cross-session provider rate-limit guard.

Stores provider quota/cooldown state in HERMES_HOME so long-lived gateway,
WebUI, CLI, cron, and background sessions do not all hammer the same exhausted
provider account after the first 429. This is intentionally separate from the
Nous-specific guard: Nous has provider-specific bucket heuristics, while this
module handles explicit reset signals from ordinary provider errors.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)

_STATE_SUBDIR = "rate_limits/providers"
_ACCOUNT_SCOPED_PROVIDERS = frozenset({
    "openai-codex",
    "opencode-go",
    "opencode-zen",
})


def _hermes_home() -> str:
    try:
        from hermes_constants import get_hermes_home

        return str(get_hermes_home())
    except Exception:
        return os.path.join(os.path.expanduser("~"), ".hermes")


def _canonical_provider(provider: str) -> str:
    return (provider or "").strip().lower() or "unknown"


def _canonical_base_url(base_url: str | None) -> str:
    return (base_url or "").strip().rstrip("/").lower()


def _scope_key(provider: str, base_url: str | None, model: str | None = None) -> str:
    provider_norm = _canonical_provider(provider)
    base_norm = _canonical_base_url(base_url)
    model_norm = "" if provider_norm in _ACCOUNT_SCOPED_PROVIDERS else (model or "").strip().lower()
    raw = json.dumps(
        {"provider": provider_norm, "base_url": base_norm, "model": model_norm},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _state_path(provider: str, base_url: str | None = None, model: str | None = None) -> str:
    safe_provider = _canonical_provider(provider).replace("/", "-")
    filename = f"{safe_provider}-{_scope_key(provider, base_url, model)}.json"
    return os.path.join(_hermes_home(), _STATE_SUBDIR, filename)


def _headers_to_dict(headers: Optional[Mapping[str, str]]) -> dict[str, str]:
    if not headers:
        return {}
    try:
        return {str(k).lower(): str(v) for k, v in headers.items()}
    except Exception:
        return {}


def _parse_reset_seconds_from_headers(headers: Optional[Mapping[str, str]]) -> Optional[float]:
    lowered = _headers_to_dict(headers)
    for key in (
        "retry-after",
        "x-ratelimit-reset-requests-1h",
        "x-ratelimit-reset-requests",
        "x-ratelimit-reset-tokens",
        "x-ratelimit-reset",
    ):
        raw = lowered.get(key)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def _parse_reset_seconds_from_context(error_context: Optional[dict[str, Any]]) -> Optional[float]:
    if not isinstance(error_context, dict):
        return None
    now = time.time()
    for key in ("resets_in_seconds", "reset_seconds", "retry_after", "retry_after_seconds"):
        raw = error_context.get(key)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    for key in ("reset_at", "resets_at"):
        raw = error_context.get(key)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value > now:
            return value - now
    return None


def record_provider_rate_limit(
    *,
    provider: str,
    model: str | None = None,
    base_url: str | None = None,
    headers: Optional[Mapping[str, str]] = None,
    error_context: Optional[dict[str, Any]] = None,
    default_cooldown: float = 300.0,
) -> None:
    """Record that a provider/backend is rate-limited.

    Prefer explicit provider reset signals. Fallback to five minutes only when
    the provider says 429 but gives no reset window.
    """
    provider_norm = _canonical_provider(provider)
    if not provider_norm or provider_norm == "unknown":
        return
    now = time.time()
    reset_seconds = (
        _parse_reset_seconds_from_context(error_context)
        or _parse_reset_seconds_from_headers(headers)
        or default_cooldown
    )
    reset_seconds = max(1.0, float(reset_seconds))
    reset_at = now + reset_seconds

    path = _state_path(provider_norm, base_url, model)
    state_dir = os.path.dirname(path)
    try:
        os.makedirs(state_dir, exist_ok=True)
        state = {
            "provider": provider_norm,
            "model": model or "",
            "base_url": base_url or "",
            "reset_at": reset_at,
            "recorded_at": now,
            "reset_seconds": reset_seconds,
        }
        if isinstance(error_context, dict):
            for key in ("plan_type", "code", "reason"):
                if key in error_context:
                    state[key] = error_context[key]
        fd, tmp_path = tempfile.mkstemp(dir=state_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(state, f)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        logger.info(
            "Provider rate limit recorded: provider=%s model=%s resets_in=%.0fs",
            provider_norm,
            model or "",
            reset_seconds,
        )
    except Exception as exc:
        logger.debug("Failed to write provider rate limit state: %s", exc)


def provider_rate_limit_remaining(
    *,
    provider: str,
    model: str | None = None,
    base_url: str | None = None,
) -> Optional[float]:
    """Return seconds remaining for an active provider cooldown, if any."""
    path = _state_path(provider, base_url, model)
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


def clear_provider_rate_limit(
    *,
    provider: str,
    base_url: str | None = None,
    model: str | None = None,
) -> None:
    """Clear a provider cooldown, e.g. after successful reauth/manual reset."""
    try:
        os.unlink(_state_path(provider, base_url, model))
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.debug("Failed to clear provider rate limit state: %s", exc)


def format_remaining(seconds: float) -> str:
    s = max(0, int(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        m, sec = divmod(s, 60)
        return f"{m}m {sec}s" if sec else f"{m}m"
    d, rem = divmod(s, 86400)
    h, rem = divmod(rem, 3600)
    m = rem // 60
    if d:
        return f"{d}d {h}h" if h else f"{d}d"
    return f"{h}h {m}m" if m else f"{h}h"
