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
from typing import Optional

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
                         relogin_required=exc.relogin_required)
    return exc


def _sentence(text: object) -> str:
    return str(text).strip().rstrip(".") + "."


def pool_cooldown_message(provider_id: str) -> Optional[str]:
    """The "all N credentials … are cooling down" error when the provider's pool is fully benched.

    ``resolve_provider_client()`` returns ``None`` both when no credential exists and when every
    pool entry sits in a 429/quota cooldown, so the raise sites could only say "no credentials
    were found. Run hermes auth add …" — wrong on both counts for a valid OAuth grant that is
    merely rate-limited (#56810). Read the persisted pool state (no seeding, no writes) and name
    the cooldown and its reset time instead; ``None`` when the pool is empty or a credential is
    usable (the caller keeps the missing-credential diagnostic).
    """
    from agent.credential_pool import STATUS_DEAD, PooledCredential, _exhausted_until
    from hermes_cli.auth import read_credential_pool

    entries = []
    with contextlib.suppress(Exception):
        entries = [PooledCredential.from_dict(provider_id, e)
                   for e in read_credential_pool(provider_id) if isinstance(e, dict)]
    live = [e for e in entries if e.last_status != STATUS_DEAD]
    if not live:
        return None
    now = time.time()
    resets = [until for e in live
              for until in (_exhausted_until(e, sole_credential=len(live) == 1),)
              if until is not None and until > now]
    if len(resets) != len(live):
        return None
    when = time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(min(resets)))
    which = ("its only credential is" if len(live) == 1
             else f"all {len(live)} credentials are")
    return (f"Provider '{provider_id}' is set in config.yaml but {which} cooling down after a "
            f"rate limit / quota error (429); the next one resets at {when}. Wait for the reset, "
            f"add another credential with `hermes auth add {provider_id}`, or switch to a "
            "different provider with `hermes model`.")


def missing_provider_credentials_message(provider_id: str) -> str:
    """The "Provider 'X' is set in config.yaml but …" error for an explicit provider with no credentials.

    The remedy comes from the registry, never from the provider id: ``f"{id.upper()}_API_KEY"``
    invents names nothing reads (alibaba → ALIBABA_API_KEY instead of DASHSCOPE_API_KEY, and the
    unsettable MINIMAX-OAUTH_API_KEY for OAuth ids, #114405 / #78996). OAuth providers have no key
    env var at all, so they are pointed at the sign-in command instead. A pool whose every
    credential is cooling down is not "missing" — that case names the cooldown (#56810).
    """
    cooldown = pool_cooldown_message(provider_id)
    if cooldown:
        return cooldown
    pconfig = None
    with contextlib.suppress(Exception):
        from hermes_cli.auth import PROVIDER_REGISTRY
        pconfig = PROVIDER_REGISTRY.get(provider_id)
    env_vars = tuple(getattr(pconfig, "api_key_env_vars", None) or ())
    problem, remedy = "no API key was found", ""
    if env_vars:
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
