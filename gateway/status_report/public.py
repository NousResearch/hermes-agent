"""What any caller may read of a gateway's status: platform entries cut to bounded values, and the
exit reason as a code (#90700).

One projection for the two readers that get no more than that: an unverified ``/api/status``
caller (``hermes_cli.web_routers.status_public``) and the deployment a status report goes to
(``gateway.status_report.payload``).
"""
from __future__ import annotations

import re
from typing import Any

from gateway.status import normalize_updated_at


# What an unverified caller may read of a platform entry: lifecycle enum, the source-controlled
# error code, the attention flag and timestamps. ``error_message`` is adapter exception text (endpoint
# URLs, response bodies, credential names) and any other key is whatever an adapter chose to write,
# so the anonymous projection is an allowlist, never a denylist (#90700).
_PUBLIC_PLATFORM_STATES = frozenset({
    "connected", "connecting", "disabled", "disconnected", "fatal", "paused", "retrying", "running", "ok"})
# Source-controlled codes (``telegram_polling_conflict``, ``loop_liveness_watchdog``) are snake_case
# identifiers; anything else (spaces, URLs, quotes, ``=``) is prose and does not pass.
_PUBLIC_CODE_RE = re.compile(r"[a-z][a-z0-9_]{0,63}")


def public_platform_entry(value: Any) -> dict[str, Any]:
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


def public_exit_reason(reason: Any, *, state: Any) -> Any:
    """*reason* when it is None or already a bounded code, else the code its free text classifies as."""
    if reason is None or (isinstance(reason, str) and _PUBLIC_CODE_RE.fullmatch(reason)):
        return reason
    from agent.monitoring.gateway_health import classify_exit_reason
    return classify_exit_reason(reason, state=state, restart_requested=False)
