"""What an unverified ``/api/status`` caller may see of gateway diagnostics (#90700).

``/api/status`` is public; the router calls :func:`redact_status_for_unverified` on the body for
any caller the auth gate would not accept on a gated route.
"""
from __future__ import annotations

import re
from typing import Any

from hermes_agent.gateway.status import normalize_updated_at


# What an unverified caller may read of a platform entry: lifecycle enum, the source-controlled
# error code, the attention flag and timestamps. ``error_message`` is adapter exception text (endpoint
# URLs, response bodies, credential names) and any other key is whatever an adapter chose to write,
# so the anonymous projection is an allowlist, never a denylist (#90700).
_PUBLIC_PLATFORM_STATES = frozenset({
    "connected", "connecting", "disabled", "disconnected", "fatal", "paused", "retrying", "running", "ok"})
# Source-controlled codes (``telegram_polling_conflict``, ``loop_liveness_watchdog``) are snake_case
# identifiers; anything else (spaces, URLs, quotes, ``=``) is prose and does not pass.
_PUBLIC_CODE_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")


def _anonymous_platform_entry(value: Any) -> dict[str, Any]:
    """Allowlisted, value-validated view of one platform entry for an unverified caller."""
    if not isinstance(value, dict):
        return {}
    entry: dict[str, Any] = {}
    state = value.get("state")
    if state is not None:
        entry["state"] = state if isinstance(state, str) and state in _PUBLIC_PLATFORM_STATES else "unknown"
    code = value.get("error_code")
    if isinstance(code, str) and _PUBLIC_CODE_RE.fullmatch(code):
        entry["error_code"] = code
    if isinstance(value.get("needs_attention"), bool):
        entry["needs_attention"] = value["needs_attention"]
    for key in ("retrying_since", "updated_at"):
        stamp = normalize_updated_at(value.get(key)) if isinstance(value.get(key), str) else None
        if stamp is not None:
            entry[key] = stamp
    return entry


def redact_status_for_unverified(status: dict[str, Any]) -> None:
    """Reduce the free-text diagnostics of a ``/api/status`` body to bounded codes in place.
    Full detail stays for a caller the auth gate would accept on a gated route."""
    from agent.monitoring.gateway_health import classify_exit_reason
    status["gateway_platforms"] = {key: _anonymous_platform_entry(value)
                                   for key, value in (status.get("gateway_platforms") or {}).items()}
    reason = status.get("gateway_exit_reason")
    if reason is not None and not (isinstance(reason, str) and _PUBLIC_CODE_RE.fullmatch(reason)):
        status["gateway_exit_reason"] = classify_exit_reason(
            reason, state=status.get("gateway_state"), restart_requested=False)
    # Remediation prose naming the colliding profiles and the credential's env var.
    status["multiplex_standalone_reason"] = None
