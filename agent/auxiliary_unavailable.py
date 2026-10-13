"""Why an auxiliary client could not be built — and the Nous credential failure behind it.

``_resolve_nous_runtime_api`` must swallow the resolver's ``AuthError`` and return None so the
ladder can still fall back (stored token, fallback_chain). Swallowing it at DEBUG left the goal
loop reporting ``judge error: RuntimeError`` while the real cause was an ``invalid_grant`` refresh
(#42177). This module remembers the latest failure so the ladder's raise and the goal judge can
name it, and warns ONCE per distinct message so operators see it without debug logging.
"""

import contextlib
import logging
import threading
import time
from datetime import datetime
from typing import Any, Optional

from hermes_time import safe_strftime

logger = logging.getLogger(__name__)


class AuxiliaryClientUnavailable(RuntimeError):
    """No auxiliary client could be built for the task (missing credentials / provider)."""


_lock = threading.Lock()
_last_nous_detail: Optional[str] = None
_warned_nous_details: set[str] = set()


def _quarantined_nous_error(exc: BaseException) -> BaseException:
    """Prefer the persisted terminal quarantine marker over *exc*; sentence-terminate either message.

    The pool rung swallows the real ``invalid_grant`` and wipes the dead tokens, so by the time the
    auth-store resolver runs it can only say "No access token found"; the marker it wrote still
    carries the message and code the user needs (#42177). ``format_auth_error`` appends the
    remediation sentence with a space, so an unterminated message reads "Invalid refresh token Run …".
    """
    from hermes_cli.auth import AuthError, get_provider_auth_state
    from hermes_cli.auth_nous import _terminal_quarantine_marker

    with contextlib.suppress(Exception):
        marker = _terminal_quarantine_marker(get_provider_auth_state("nous") or {})
        if marker and marker.get("message"):
            return AuthError(_sentence(marker["message"]), provider="nous", code=marker.get("code"),
                             relogin_required=True)
    if isinstance(exc, AuthError):
        return AuthError(_sentence(exc), provider=exc.provider, code=exc.code,
                         relogin_required=exc.relogin_required, retry_after=exc.retry_after,
                         retryable=exc.retryable)
    return exc


def _sentence(text: object) -> str:
    return str(text).strip().rstrip(".") + "."


def pool_cooldown_message(provider_id: str, *, model: Optional[str] = None) -> Optional[str]:
    """The "all N credentials … are cooling down" error when the provider's pool is fully benched.

    ``resolve_provider_client()`` returns ``None`` both when no credential exists and when every
    pool entry sits in a 429/quota cooldown, so the raise sites could only say "no credentials
    were found. Run hermes auth add …" — wrong on both counts for a valid OAuth grant that is
    merely rate-limited (#56810). Read the persisted pool state (no seeding, no writes) and name
    the cooldown and its reset time instead; ``None`` when the pool is empty or a credential is
    usable (the caller keeps the missing-credential diagnostic).

    *model* mirrors the selection-side predicate: ``_available_entries(model=…)`` benches an
    entry on a credential-wide exhaustion OR an active model-scoped cooldown, so a pool whose
    entries are healthy but model-benched must not be called "no API key" either (#128995).
    ``model=None`` stays conservative exactly like ``model_cooldown_until``: any active model
    cooldown blocks an unscoped route.
    """
    from agent.credential_pool import STATUS_DEAD, PooledCredential
    from agent.credential_pool_cooldowns import _exhausted_until
    from agent.credential_pool_model_cooldowns import model_cooldown_until
    from hermes_cli.auth import read_credential_pool

    entries = []
    with contextlib.suppress(Exception):
        entries = [PooledCredential.from_dict(provider_id, e)
                   for e in read_credential_pool(provider_id) if isinstance(e, dict)]
    live = [e for e in entries if e.last_status != STATUS_DEAD]
    if not live:
        return None
    now = time.time()
    resets: list = []
    scoped_models: set = set()
    credential_wide = False
    for e in live:
        blocking: list = []
        until = _exhausted_until(e, sole_credential=len(live) == 1)
        if until is not None and until > now:
            blocking.append(until)
            credential_wide = True
        scoped_until = model_cooldown_until(e, model)
        if scoped_until is not None:
            blocking.append(scoped_until)
            scoped_models.update(
                name for name, value in (e.model_cooldowns or {}).items()
                if isinstance(value, (int, float)) and value > now and (not model or name == model)
            )
        if not blocking:
            # This entry is usable the way the caller asked: not a cooldown picture.
            return None
        resets.append(min(blocking))
    when = safe_strftime(datetime.fromtimestamp(min(resets)).astimezone(), "%Y-%m-%d %H:%M %Z")
    which = ("its only credential is" if len(live) == 1
             else f"all {len(live)} credentials are")
    if scoped_models and not credential_wide:
        # Healthy credentials benched per-model only: say so instead of implying a key problem.
        names = ", ".join(sorted(scoped_models))
        return (f"Provider '{provider_id}' is set in config.yaml but {which} benched by a "
                f"model-scoped rate-limit cooldown ({names}, until {when}) — the credentials "
                f"themselves are healthy. Wait for the cooldown, clear it with "
                f"`hermes auth reset {provider_id}`, or switch models with `hermes model`.")
    return (f"Provider '{provider_id}' is set in config.yaml but {which} cooling down after a "
            f"rate limit / quota error (429); the next one resets at {when}. Wait for the reset, "
            f"add another credential with `hermes auth add {provider_id}`, or switch to a "
            "different provider with `hermes model`.")


class ProviderCredentialsExhaustedError(RuntimeError):
    """An explicit provider's credential pool is fully exhausted (402 out of credits, quota bench).

    Subclasses ``RuntimeError`` so every existing ``except RuntimeError`` keeps working; the distinct
    type lets a caller (gateway, Desktop) route a billing/quota failure to billing guidance instead
    of the provider-setup flow a plain "No LLM provider configured" would trigger (#94785).
    """

    def __init__(self, message: str, *, provider: Optional[str] = None):
        super().__init__(message)
        self.provider = provider


def pool_billing_message(
    provider_id: str,
    *,
    model: Optional[str] = None,
    pool: Optional[Any] = None,
) -> Optional[str]:
    """The "Provider 'X' … is out of usable credentials (402)" error for a billing-burned pool.

    ``resolve_provider_client()`` returns ``None`` both when no credential exists and when every
    pool entry has been marked exhausted — a 402 out-of-credits account, a 429/quota bench. The
    ``openrouter`` / ``custom`` branch of ``agent_init`` has no provider-specific missing-credentials
    message, so a burned pool there fell through to the generic "No LLM provider configured" setup
    message: actively misleading when a *different* provider is configured and working and only the
    session's override is out of credits (#94785). Name the provider, the session's model, the last
    error code and the provider's own message instead.

    Only a genuine billing verdict — a 402, or the classifier's ``billing`` reason on a 403 spending
    limit — is named here; a plain 429/quota bench keeps the existing cooldown wording (with its
    reset time, #56810), which the caller falls back to when this returns ``None``.

    *pool* is an already-loaded ``CredentialPool`` (preferred — it carries the live in-memory
    exhaustion state); otherwise the persisted rows are read (no seeding, no writes, mirroring
    ``pool_cooldown_message``). ``None`` when the pool is empty, a credential is usable, no entry is
    currently benched, or none carries a billing verdict.
    """
    from agent.credential_pool import (
        FAILURE_REASON_BILLING,
        STATUS_DEAD,
        STATUS_EXHAUSTED,
        PooledCredential,
    )
    from agent.credential_pool_cooldowns import _exhausted_until

    entries: list = []
    with contextlib.suppress(Exception):
        if pool is not None:
            entries = list(pool.entries())
        else:
            from hermes_cli.auth import read_credential_pool

            entries = [PooledCredential.from_dict(provider_id, e)
                       for e in read_credential_pool(provider_id) if isinstance(e, dict)]
    live = [e for e in entries if getattr(e, "last_status", None) != STATUS_DEAD]
    if not live:
        return None
    now = time.time()
    benched = [
        e for e in live
        if getattr(e, "last_status", None) == STATUS_EXHAUSTED
        and (_exhausted_until(e, sole_credential=len(live) == 1) or 0) > now
    ]
    if len(benched) != len(live):
        # A credential is still usable (or merely cooling down on one model): not our case.
        return None
    billing = [
        e for e in benched
        if getattr(e, "last_error_code", None) == 402
        or getattr(e, "failure_reason", None) == FAILURE_REASON_BILLING
    ]
    if not billing:
        return None
    pick = billing[0]
    code = getattr(pick, "last_error_code", None)
    detail = str(getattr(pick, "last_error_message", None) or "").strip()
    which = "its only credential is" if len(live) == 1 else f"all {len(live)} credentials are"
    scope = f" for model '{model}'" if model else ""
    code_text = f" (last error {code}{': ' + detail if detail else ''})" if code else ""
    return (
        f"Provider '{provider_id}'{scope} is out of usable credentials: {which} exhausted"
        f"{code_text}. Add credits or update billing with that provider, then retry, or switch "
        f"to a different provider with `hermes model`."
    )


class ProviderNotConfiguredError(RuntimeError):
    """Agent init found no usable inference provider: nothing is configured, or the explicitly
    configured one has no credentials.

    A distinct type because the remedy is setup, not a retry: clients route it to their provider
    setup / onboarding flow. The wording of the two messages has changed more than once, so a
    client that recognises this failure by matching their text silently stops routing (#119232's
    sibling); the type and the gateway's ``code`` survive rewording.
    """


def missing_provider_credentials_message(provider_id: str, *, model: Optional[str] = None) -> str:
    """The "Provider 'X' is set in config.yaml but …" error for an explicit provider with no credentials.

    The remedy comes from the registry, never from the provider id: ``f"{id.upper()}_API_KEY"``
    invents names nothing reads (alibaba → ALIBABA_API_KEY instead of DASHSCOPE_API_KEY, and the
    unsettable MINIMAX-OAUTH_API_KEY for OAuth ids, #114405 / #78996). OAuth providers have no key
    env var at all, so they are pointed at the sign-in command instead. A pool whose every
    credential is cooling down is not "missing" — that case names the cooldown (#56810), including
    a model-scoped cooldown on otherwise healthy credentials (#128995).

    *model* is passed through to ``pool_cooldown_message`` so the cooldown verdict matches the
    model the caller actually tried.
    """
    cooldown = pool_cooldown_message(provider_id, model=model)
    if cooldown:
        return cooldown
    pconfig = None
    with contextlib.suppress(Exception):
        from hermes_cli.auth import PROVIDER_REGISTRY
        pconfig = PROVIDER_REGISTRY.get(provider_id)
    env_vars = tuple(getattr(pconfig, "api_key_env_vars", None) or ())
    problem, remedy = "no API key was found", ""
    from hermes_cli.local_runtime.endpoint import LLAMACPP_ALIASES
    if provider_id in LLAMACPP_ALIASES:
        problem, remedy = "no local llama.cpp server is running", "Start it in Settings → Providers → Local models"
    elif env_vars:
        remedy = f"Set the {env_vars[0]} environment variable"
    elif pconfig is None:
        remedy = f"Set the {provider_id.upper().replace('-', '_')}_API_KEY environment variable"
    elif str(pconfig.auth_type).startswith("oauth"):
        problem, remedy = "no credentials were found", f"Run `hermes auth add {provider_id}` to sign in"
    else:
        problem = "no credentials were found"
    switch = "switch to a different provider with `hermes model`."
    return (f"Provider '{provider_id}' is set in config.yaml but {problem}. "
            + (f"{remedy}, or {switch}" if remedy else switch.capitalize()))


def _nous_credential_present(exc: BaseException) -> bool:
    """True when a Nous credential exists that failed — a coded error (``invalid_grant``, a quarantine
    marker, ``refresh_failed``) or persisted Nous auth state. The uncoded "not logged into Nous Portal"
    with no stored state is the normal condition for users who never chose Nous; the auto-route walk
    hits it on every discovery pass and must not warn (the ladder already logs its own summary).
    """
    if getattr(exc, "code", None):
        return True
    from hermes_cli.auth import get_provider_auth_state

    with contextlib.suppress(Exception):
        return bool(get_provider_auth_state("nous"))
    return False


def record_nous_credential_failure(exc: BaseException) -> str:
    """Remember *exc* as the latest Nous credential failure.

    Logged once per distinct message: WARNING when a real credential failed, DEBUG when Hermes was
    simply never logged into Nous.
    """
    from hermes_cli.auth import format_auth_error

    exc = _quarantined_nous_error(exc)
    message = format_auth_error(exc) if isinstance(exc, Exception) else str(exc)
    message = message.strip() or type(exc).__name__
    code = getattr(exc, "code", None)
    if code and str(code) not in message:
        message = f"{message} (code: {code})"
    detail = f"Nous Portal runtime credentials unavailable: {message}"
    global _last_nous_detail
    with _lock:
        _last_nous_detail = detail
        first_time = detail not in _warned_nous_details
        _warned_nous_details.add(detail)
    if first_time:
        level = logging.WARNING if _nous_credential_present(exc) else logging.DEBUG
        logger.log(level, "Auxiliary Nous client unavailable: %s", detail)
    return detail


def clear_nous_credential_failure() -> None:
    global _last_nous_detail
    with _lock:
        _last_nous_detail = None


def nous_credential_failure_detail() -> Optional[str]:
    """The latest recorded Nous credential failure, or None when the last resolution succeeded."""
    with _lock:
        return _last_nous_detail
