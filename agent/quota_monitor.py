"""Quota monitor for Hermes provider usage-cap state.

Parses 429 response bodies from OpenAI Codex-style providers, persists reset
metadata, and surfaces the reset window in fallback notifications and quota
reports. The state is profile-local via ``get_hermes_home()`` unless tests set
``HERMES_QUOTA_STATE_PATH``.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_DEFAULT_STATE_PATH = get_hermes_home() / "quota_state.json"


def _state_path() -> Path:
    env = os.environ.get("HERMES_QUOTA_STATE_PATH")
    return Path(env) if env else _DEFAULT_STATE_PATH


def read_quota_state() -> Dict[str, Any]:
    """Return the full quota state dict. Empty if never written or unreadable."""
    p = _state_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("quota_monitor: failed to read state: %s", exc)
        return {}


def write_quota_state(state: Dict[str, Any]) -> None:
    """Atomically write quota state."""
    p = _state_path()
    tmp = p.with_suffix(p.suffix + ".tmp") if p.suffix else p.with_name(p.name + ".tmp")
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(p)
    except Exception as exc:
        logger.debug("quota_monitor: failed to write state: %s", exc)


def _coerce_error_dict(error_body: Any) -> Dict[str, Any]:
    body = error_body or {}
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except Exception:
            return {}
    if not isinstance(body, dict):
        return {}

    wrapped = body.get("error")
    if isinstance(wrapped, dict):
        return wrapped
    # OpenAI/Codex SDK paths can expose the inner error object directly:
    # {'type': 'usage_limit_reached', 'resets_at': ...}
    return body


def record_rate_limit(
    provider: str,
    model: str,
    status_code: int,
    error_body: Any,
) -> None:
    """Extract quota metadata from a 429/error body and persist it.

    Accepted shapes:
    - {'error': {'type': 'usage_limit_reached', 'resets_at': <unix>, ...}}
    - {'type': 'usage_limit_reached', 'resets_at': <unix>, ...}
    """
    if status_code != 429:
        return

    error_dict = _coerce_error_dict(error_body)
    resets_at = error_dict.get("resets_at")
    resets_in_seconds = error_dict.get("resets_in_seconds")
    if not resets_at and not resets_in_seconds:
        return

    provider_key = (provider or "unknown").strip() or "unknown"
    state = read_quota_state()
    previous_raw = state.get(provider_key)
    previous = previous_raw if isinstance(previous_raw, dict) else {}
    entry = {
        "provider": provider_key,
        "model": model,
        "status_code": status_code,
        "error_type": error_dict.get("type"),
        "plan_type": error_dict.get("plan_type"),
        "resets_at": resets_at,
        "resets_in_seconds": resets_in_seconds,
        "recorded_at": time.time(),
    }
    # Preserve fallback metadata if fallback was recorded before the error body
    # was parsed successfully.
    for key in ("fallback_activated_at", "fallback_to"):
        if key in previous:
            entry[key] = previous[key]

    state[provider_key] = entry
    state["_last_updated"] = time.time()
    write_quota_state(state)
    logger.info(
        "quota_monitor: recorded %s limit — resets_at=%s (%ss)",
        provider_key,
        resets_at,
        resets_in_seconds,
    )


def record_fallback_notification(provider: str, fallback_provider: str) -> None:
    """Record that fallback was activated away from ``provider``."""
    provider_key = (provider or "unknown").strip() or "unknown"
    state = read_quota_state()
    entry_raw = state.get(provider_key)
    entry = entry_raw if isinstance(entry_raw, dict) else {}
    entry["fallback_activated_at"] = time.time()
    entry["fallback_to"] = fallback_provider
    state[provider_key] = entry
    state["_last_updated"] = time.time()
    write_quota_state(state)


def _fmt_seconds(secs: int | float) -> str:
    s = max(0, int(secs))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        m, sec = divmod(s, 60)
        return f"{m}m {sec}s" if sec else f"{m}m"
    if s < 86400:
        h, rem = divmod(s, 3600)
        m = rem // 60
        return f"{h}h {m}m" if m else f"{h}h"
    d, rem = divmod(s, 86400)
    h = rem // 3600
    return f"{d}d {h}h" if h else f"{d}d"


def format_quota_report(provider: Optional[str] = None) -> str:
    """Human-readable quota report for Discord/CLI."""
    state = read_quota_state()
    if not state:
        return "No quota state recorded yet."

    lines: list[str] = []
    targets = [provider] if provider else list(state.keys())
    now = time.time()

    for key in targets:
        if not key or str(key).startswith("_"):
            continue
        entry = state.get(key)
        if not isinstance(entry, dict):
            continue

        resets_at = entry.get("resets_at")
        resets_in = entry.get("resets_in_seconds")
        recorded = entry.get("recorded_at", 0)
        plan = entry.get("plan_type") or "unknown"
        err_type = entry.get("error_type") or "limit"
        fb_at = entry.get("fallback_activated_at")
        fb_to = entry.get("fallback_to")

        if resets_at:
            remaining = int(float(resets_at) - now)
            status = "reset ✓" if remaining <= 0 else f"resets in {_fmt_seconds(remaining)}"
        elif resets_in:
            stale_remaining = int(float(resets_in) - (now - float(recorded or now)))
            status = (
                "reset ✓ (stale estimate)"
                if stale_remaining <= 0
                else f"~{_fmt_seconds(stale_remaining)} remaining (stale)"
            )
        else:
            status = "unknown"

        header = f"**{key}** ({plan}) — {err_type}"
        lines.append(f"{header}: {status}")
        if fb_at and fb_to:
            fb_ago = int(now - float(fb_at))
            lines.append(f"  → fell back to {fb_to} {_fmt_seconds(fb_ago)} ago")

    return "\n".join(lines) if lines else "No quota entries found."


def get_quota_summary_for_provider(provider: str) -> Optional[str]:
    """One-line quota reset summary for fallback notifications."""
    provider_key = (provider or "unknown").strip() or "unknown"
    state = read_quota_state()
    entry = state.get(provider_key)
    if not isinstance(entry, dict):
        return None

    resets_at = entry.get("resets_at")
    resets_in = entry.get("resets_in_seconds")
    recorded = entry.get("recorded_at", time.time())
    now = time.time()

    if resets_at:
        remaining = int(float(resets_at) - now)
    elif resets_in:
        remaining = int(float(resets_in) - (now - float(recorded or now)))
    else:
        return None

    if remaining <= 0:
        return f"{provider_key} quota has reset."
    return f"{provider_key} quota resets in {_fmt_seconds(remaining)}."
