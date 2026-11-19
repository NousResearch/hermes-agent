"""What an unverified ``/api/status`` caller may see of gateway diagnostics (#90700).

``/api/status`` is public; the router calls :func:`redact_status_for_unverified` on the body for
any caller the auth gate would not accept on a gated route. The projection itself is
``gateway.status_public``, shared with the status report.
"""
from __future__ import annotations

from typing import Any

from gateway.status_public import public_exit_reason, public_platform_entry


def redact_status_for_unverified(status: dict[str, Any]) -> None:
    """Reduce the free-text diagnostics of a ``/api/status`` body to bounded codes in place.
    Full detail stays for a caller the auth gate would accept on a gated route."""
    status["gateway_platforms"] = {key: public_platform_entry(value)
                                   for key, value in (status.get("gateway_platforms") or {}).items()}
    reason = status.get("gateway_exit_reason")
    if reason is not None:
        status["gateway_exit_reason"] = public_exit_reason(reason, state=status.get("gateway_state"))
    # Remediation prose naming the colliding profiles and the credential's env var.
    status["multiplex_standalone_reason"] = None
