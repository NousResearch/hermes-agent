"""Conversation-loop error classifiers."""

from typing import Optional

def _is_stale_copilot_credential_error(status_code: Optional[int], error_message: str) -> bool:
    """Detect a Copilot 400 that is really a STALE / DEGRADED credential.

    Copilot surfaces a stale or degraded credential as an HTTP 400 rather than a
    clean 401. Two body markers indicate this class:

    - ``model_not_available_for_integrator`` — the request reached the
      restricted ``copilot-language-server`` integrator (the server's fallback
      when it receives a raw OAuth token instead of an exchanged API token),
      whose model allowlist omits enterprise-only models.
    - ``model_not_supported`` / "the requested model is not supported" — the
      cached bearer's Copilot entitlement rotated out from under a long-lived
      process.

    Matched narrowly (status 400 AND a specific marker) so a genuinely wrong
    model name — a real 400 — never triggers the single-shot re-exchange. The
    caller enforces copilot-provider scoping and the single-shot guard.
    """
    lowered = (error_message or "").lower()
    is_400 = status_code == 400 or "error code: 400" in lowered
    if not is_400:
        return False
    return (
        "model_not_available_for_integrator" in lowered
        or "not available for integrator" in lowered
        or "model_not_supported" in lowered
        or "the requested model is not supported" in lowered
    )
