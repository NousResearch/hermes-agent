"""The primary provider's failed login, carried forward when resolution fell back to a backup.

``resolve_runtime_with_fallback`` swaps an expired/missing primary login for the first fallback
entry that resolves. When that backup then fails the turn too (out of credits, its own auth,
any terminal error), the backup's error is all the turn loop knows about, so the user is told
"no credits" and never that the real fix is signing back in to the primary. This module owns
the structured reason and the one place that leads a failed turn's copy with it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

# Private runtime-dict key stamped by the resolution-time fallback walker.
RUNTIME_KEY = "_primary_auth_failure"


@dataclass(frozen=True)
class PrimaryAuthFailure:
    """Why the primary provider was skipped at resolution time and the exact command that fixes it."""

    provider: str
    error: str
    relogin: str
    backup: str = ""


def primary_auth_failure_from_exc(exc: Any, provider: Any, backup: Any = "") -> Optional[PrimaryAuthFailure]:
    """The structured reason for a primary ``AuthError`` that a re-login fixes, else None.

    A quota/rate-limit cap, a retryable refusal and an entitlement (credits/subscription) code
    leave the login valid: telling the user to sign in again would send them the wrong way."""
    from hermes_cli.auth import _ENTITLEMENT_ERROR_CODES, AuthError, is_rate_limited_auth_error

    if not isinstance(exc, AuthError) or is_rate_limited_auth_error(exc) or exc.retryable is True:
        return None
    if exc.code in _ENTITLEMENT_ERROR_CODES or exc.code == "temporarily_unavailable":
        return None
    from agent.turn_failure_copy import relogin_command_hint, short_detail

    slug = str(exc.provider or provider or "").strip().lower()
    slug = "" if slug == "auto" else slug
    return PrimaryAuthFailure(provider=slug, error=short_detail(exc, 300),
                              relogin=relogin_command_hint(slug), backup=str(backup or "").strip())


def attach_primary_auth_failure(agent: Any, failure: Optional[PrimaryAuthFailure]) -> None:
    """Bind (or clear, with None) the reason on an agent built from a fallback runtime."""
    if agent is not None:
        agent._primary_auth_failure = failure if isinstance(failure, PrimaryAuthFailure) else None


# Backup failure reason -> what the backup's problem is, in the user's words.
_BACKUP_PROBLEM = {
    "billing": "is out of credits (or over its usage limit)",
    "billing_unverified": "reported it is out of credits (or over its usage limit)",
    "auth": "rejected its sign-in as well",
    "auth_permanent": "rejected its sign-in as well",
    "rate_limit": "is rate-limiting requests",
    "upstream_rate_limit": "is rate-limiting requests",
    "overloaded": "is overloaded",
    "server_error": "returned a server error",
    "timeout": "did not respond in time",
    "model_not_found": "does not have the configured model",
}


def primary_auth_lead(failure: PrimaryAuthFailure, *, backup: str, failure_reason: Any, error: Any) -> str:
    """Root cause first (re-login the primary, exact command), then the backup's own problem."""
    from agent.redact import redact_sensitive_text
    from agent.turn_failure_copy import provider_label_for, short_detail

    primary = provider_label_for(failure.provider) if failure.provider else "your main AI provider"
    problem = _BACKUP_PROBLEM.get(str(failure_reason or ""))
    if problem is None:
        detail = short_detail(error, 160) if error else "failed"
        problem = f"failed too ({detail})"
    why = redact_sensitive_text(failure.error, force=True)
    return (
        f"⚠️ {primary} needs you to sign in again: run `{failure.relogin}`. Its login failed "
        f"({why}), so Hermes switched to the backup provider ({backup or 'backup'}), and the "
        f"backup {problem}. Re-authenticating {primary} fixes this; topping up the backup only "
        "keeps the fallback running."
    )


def lead_failed_turn_with_primary_auth(agent: Any, result: Any) -> None:
    """Prefix a failed turn's ``final_response`` with the primary re-login (in place).

    Applied once where every turn envelope leaves the loop, so the billing, non-retryable and
    retries-exhausted paths all keep their own backup-specific copy underneath the lead."""
    failure = getattr(agent, "_primary_auth_failure", None)
    if not isinstance(failure, PrimaryAuthFailure) or not isinstance(result, dict):
        return
    if result.get("failed") is not True or result.get("interrupted") or result.get("_primary_auth_led"):
        return
    current = str(getattr(agent, "provider", "") or "").strip()
    if failure.provider and current.lower() == failure.provider:
        return  # the primary itself is serving again: its failure is not this turn's story
    backup = current if current and current.lower() != "custom" else (failure.backup or current)
    lead = primary_auth_lead(failure, backup=backup, failure_reason=result.get("failure_reason"),
                             error=result.get("error"))
    body = str(result.get("final_response") or "").strip()
    result["final_response"] = f"{lead}\n\nBackup provider said: {body}" if body else lead
    result["primary_auth_failure"] = {"provider": failure.provider, "error": failure.error,
                                      "relogin": failure.relogin}
    result["_primary_auth_led"] = True
