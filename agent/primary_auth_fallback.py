"""The primary provider's failed login, carried forward when resolution fell back to a backup.

``resolve_runtime_with_fallback`` swaps an expired/missing primary login for the first fallback
entry that resolves. When that backup then fails the turn too (out of credits, its own auth,
any terminal error), the backup's error is all the turn loop knows about, so the user is told
"no credits" and never that the real fix is signing back in to the primary. This module owns
the structured reason and the one place that leads a failed turn's copy with it.
"""

from __future__ import annotations

import contextvars
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
    backup_base_url: str = ""  # the endpoint resolution fell back to; another endpoint is another route


# AuthError codes whose fix is not a sign-in: an outage, a per-model cooldown, a CLI that is not installed,
# or a configuration error (no provider, an unknown one, a custom route without a base URL).
_NOT_A_LOGIN_PROBLEM = frozenset({"temporarily_unavailable", "model_rate_limited", "missing_external_process_cli",
                                  "no_provider_configured", "invalid_provider", "missing_base_url"})


def primary_auth_failure_from_exc(exc: Any, provider: Any, backup: Any = "",
                                  backup_base_url: Any = "") -> Optional[PrimaryAuthFailure]:
    """The structured reason for a primary ``AuthError`` that a re-login fixes, else None.

    A quota/rate-limit cap, a retryable refusal and an entitlement (credits/subscription) code
    leave the login valid: telling the user to sign in again would send them the wrong way."""
    from hermes_cli.auth import _ENTITLEMENT_ERROR_CODES, AuthError, is_rate_limited_auth_error

    if not isinstance(exc, AuthError) or is_rate_limited_auth_error(exc) or exc.retryable is True:
        return None
    if exc.code in _ENTITLEMENT_ERROR_CODES or exc.code in _NOT_A_LOGIN_PROBLEM:
        return None
    from agent.turn_failure_copy import relogin_command_hint, short_detail

    slug = str(exc.provider or provider or "").strip().lower()
    slug = "" if slug == "auto" else slug
    return PrimaryAuthFailure(provider=slug, error=short_detail(exc, 300),
                              relogin=relogin_command_hint(slug), backup=str(backup or "").strip(),
                              backup_base_url=str(backup_base_url or "").strip())


def attach_primary_auth_failure(agent: Any, failure: Optional[PrimaryAuthFailure]) -> None:
    """Bind (or clear, with None) the reason on an agent built from a fallback runtime.

    The failure belongs to the route that resolution fell back to: when a later step moved the turn
    onto another provider or another endpoint of the same provider (``/model --once``, an API caller's
    provider, a channel override), the backup it names is not the one running, so it is dropped rather
    than blamed on that route."""
    if agent is None:
        return
    if isinstance(failure, PrimaryAuthFailure) and not _runs_the_backup(agent, failure):
        failure = None
    agent._primary_auth_failure = failure if isinstance(failure, PrimaryAuthFailure) else None


def _runs_the_backup(agent: Any, failure: PrimaryAuthFailure) -> bool:
    """Whether ``agent`` runs the route the failure fell back to: same provider and, when both are
    known, the same endpoint (normalized as route identity normalizes it)."""
    from hermes_cli.route_identity import normalize_route_base_url

    running = str(getattr(agent, "provider", "") or "").strip().lower()
    if failure.backup and running and running != failure.backup.strip().lower():
        return False
    running_url = normalize_route_base_url(getattr(agent, "base_url", "") or "")
    backup_url = normalize_route_base_url(failure.backup_base_url)
    return not (running_url and backup_url and running_url != backup_url)


# The failure found while resolving this turn's route. A context variable, not a runner attribute: the
# gateway resolves concurrent chats on separate threads, each with its own copied context.
_PENDING: contextvars.ContextVar[Optional[PrimaryAuthFailure]] = contextvars.ContextVar(
    "pending_primary_auth_failure", default=None)


def set_pending_primary_auth_failure(failure: Optional[PrimaryAuthFailure]) -> None:
    _PENDING.set(failure if isinstance(failure, PrimaryAuthFailure) else None)


def take_pending_primary_auth_failure() -> Optional[PrimaryAuthFailure]:
    """This turn's pending failure, cleared so a later resolution in the same context starts clean."""
    failure = _PENDING.get()
    _PENDING.set(None)
    return failure


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
    if result.get("failed") is not True or result.get("interrupted") or "primary_auth_failure" in result:
        return
    current = str(getattr(agent, "provider", "") or "").strip()
    if failure.provider and current.lower() == failure.provider:
        return  # the primary itself is serving again: its failure is not this turn's story
    backup = current if current and current.lower() != "custom" else (failure.backup or current)
    lead = primary_auth_lead(failure, backup=backup, failure_reason=result.get("failure_reason"),
                             error=result.get("error"))
    body = str(result.get("final_response") or "").strip()
    result["final_response"] = f"{lead}\n\nBackup provider said: {body}" if body else lead
    from agent.redact import redact_sensitive_text

    result["primary_auth_failure"] = {"provider": failure.provider, "relogin": failure.relogin,
                                      "error": redact_sensitive_text(failure.error, force=True)}
