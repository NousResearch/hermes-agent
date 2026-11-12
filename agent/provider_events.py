"""Structured provider-health events for API clients (retry / rate limit / usage / fallback).

The CLI and messaging surfaces learn about provider trouble through free-text status lines,
most of which are buffered and only shown once every retry failed. API clients (``/v1/runs``)
need the same facts while the turn is still waiting, in a shape they can render in plain words.
This module builds one small dict per occurrence and hands it to ``agent.provider_event_callback``
(``None`` everywhere except the API server, which turns it into a ``provider.status`` run event):

    {"kind": "retry" | "rate_limited" | "usage_exhausted" | "fallback",
     "provider": str, "model": str, "attempt": int?, "max": int?, "wait_seconds": float?,
     "reset_at": float?  (epoch seconds), "reason": str?, "to_provider": str?, "to_model": str?,
     "locked": bool}

It also records the last failure per ``provider::model`` in a tiny JSON file so a model picker can
warn about a model whose last run failed (``recent_model_failures``). Everything here is best
effort: a callback or disk error never affects the turn.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0", "host.docker.internal"}
_USAGE_TOKENS = ("usage_limit_reached", "usage limit", "gousagelimit", "quota", "insufficient_quota")
# A rate limit that resets further out than this is an exhausted allowance, not a throttle.
_USAGE_RESET_FLOOR_S = 300.0
_HEALTH_FILE_LOCK = threading.Lock()
_FAILURE_TTL_S = 6 * 3600


def is_model_locked(agent: Any) -> bool:
    """True when the client pinned this turn to one model (``require_model_lock``)."""
    return getattr(agent, "_model_locked", False) is True


def is_local_endpoint(agent: Any) -> bool:
    host = (urlparse(str(getattr(agent, "base_url", "") or "")).hostname or "").lower()
    return host in _LOCAL_HOSTS or host.endswith(".local")


def error_class(api_error: Any) -> str:
    """One word for what went wrong: connection | timeout | rate_limit | not_found | server_error | error."""
    status = getattr(api_error, "status_code", None)
    name = type(api_error).__name__
    text = str(api_error).lower()
    if status == 429:
        return "rate_limit"
    if status == 404:
        return "not_found"
    if isinstance(status, int) and status >= 500:
        return "server_error"
    if "Timeout" in name or "timed out" in text:
        return "timeout"
    if "Connection" in name or "ConnectError" in name or "refused" in text or "connection error" in text:
        return "connection"
    return "error"


def parse_reset_at(value: Any) -> Optional[float]:
    """Epoch seconds from an epoch (s or ms), ISO-8601 string or None."""
    if value in (None, ""):
        return None
    try:
        from agent.credential_pool import _parse_absolute_timestamp
        parsed = _parse_absolute_timestamp(value)
        return float(parsed) if parsed else None
    except Exception:
        try:
            v = float(value)
            return v / 1000.0 if v > 1e12 else v
        except (TypeError, ValueError):
            return None


def is_usage_exhausted(classified: Any, error_context: Any) -> bool:
    """An account allowance is used up (Claude plan window, ChatGPT/Codex usage limit, quota, credits),
    as opposed to a short throttle that clears within a retry window."""
    from agent.error_classifier import FailoverReason
    reason = getattr(classified, "reason", None)
    ctx = error_context if isinstance(error_context, dict) else {}
    ctx = {**(getattr(classified, "error_context", None) or {}), **ctx}
    if reason == FailoverReason.billing or ctx.get("usage_window"):
        return True
    if reason not in (FailoverReason.rate_limit, FailoverReason.upstream_rate_limit):
        return False
    text = f"{ctx.get('reason') or ''} {ctx.get('message') or ''}".lower()
    if any(token in text for token in _USAGE_TOKENS):
        return True
    reset_at = parse_reset_at(ctx.get("reset_at"))
    return reset_at is not None and reset_at - time.time() > _USAGE_RESET_FLOOR_S


def emit(agent: Any, kind: str, **fields: Any) -> None:
    """Hand one provider event to ``agent.provider_event_callback`` (if any). Never raises."""
    payload: dict[str, Any] = {
        "kind": kind,
        "provider": str(getattr(agent, "provider", "") or ""),
        "model": str(getattr(agent, "model", "") or ""),
        "locked": is_model_locked(agent),
    }
    for key, value in fields.items():
        if value is None:
            continue
        if key == "reset_at":
            value = parse_reset_at(value)
            if value is None:
                continue
        payload[key] = value
    if kind in ("usage_exhausted", "failed"):
        record_model_failure(payload)
    callback = getattr(agent, "provider_event_callback", None)
    if not callback:
        return
    try:
        callback(payload)
    except Exception:
        logger.debug("provider_event_callback failed", exc_info=True)


def _health_path() -> str:
    try:
        from hermes_constants import get_hermes_home
        home = str(get_hermes_home())
    except Exception:
        home = os.path.expanduser("~/.hermes")
    return os.path.join(home, "cache", "model_health.json")


def _load(path: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _store(path: str, data: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    os.replace(tmp, path)


def _key(provider: str, model: str) -> str:
    return f"{provider}::{model}"


def record_model_failure(event: dict[str, Any], *, path: Optional[str] = None) -> None:
    """Remember the latest failure for ``provider::model`` (kind, when, reset_at, reason)."""
    provider, model = event.get("provider") or "", event.get("model") or ""
    if not provider or not model:
        return
    path = path or _health_path()
    try:
        with _HEALTH_FILE_LOCK:
            data = _load(path)
            now = time.time()
            data = {k: v for k, v in data.items() if isinstance(v, dict) and now - float(v.get("at") or 0) < _FAILURE_TTL_S}
            data[_key(provider, model)] = {
                "kind": event.get("kind"), "at": now, "reset_at": event.get("reset_at"),
                "reason": event.get("reason") or "",
            }
            _store(path, data)
    except Exception:
        logger.debug("model health write failed", exc_info=True)


def record_model_ok(provider: str, model: str, *, path: Optional[str] = None) -> None:
    """A turn on ``provider::model`` succeeded: forget its last failure."""
    if not provider or not model:
        return
    path = path or _health_path()
    try:
        with _HEALTH_FILE_LOCK:
            data = _load(path)
            if data.pop(_key(provider, model), None) is not None:
                _store(path, data)
    except Exception:
        logger.debug("model health write failed", exc_info=True)


def recent_model_failures(*, path: Optional[str] = None, max_age_s: float = 1800.0) -> dict[str, Any]:
    """``provider::model`` -> last failure, for failures newer than *max_age_s* (or not yet reset)."""
    now = time.time()
    out = {}
    for key, value in _load(path or _health_path()).items():
        if not isinstance(value, dict):
            continue
        reset_at = value.get("reset_at")
        fresh = now - float(value.get("at") or 0) < max_age_s
        if fresh or (isinstance(reset_at, (int, float)) and reset_at > now):
            out[key] = value
    return out
