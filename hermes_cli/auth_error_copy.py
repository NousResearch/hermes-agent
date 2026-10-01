"""Plain-language copy for sign-in and provider-setup failures (CLI).

One table of ``(predicate, lead sentence)`` classifies an exception into a sentence a first-time
user can act on; the raw exception is demoted to a ``Details:`` line so nothing is lost for
support. Callers print ``sign_in_failure_lines(...)`` / ``provider_setup_failure_lines(...)``
line by line.
"""

from __future__ import annotations
from auth.oauth import SignInCopyError


from typing import Callable, Sequence, Tuple

# httpx class names and stdlib bases that mean "the request never got a usable answer".
_NETWORK_ERROR_TYPES = frozenset({
    "ConnectError", "ConnectTimeout", "ReadTimeout", "PoolTimeout", "WriteTimeout", "TimeoutException",
    "RemoteProtocolError", "ReadError", "ProxyError", "UnsupportedProtocol", "NetworkError",
})

# OAuth device-flow error codes (RFC 8628 §3.5) -> plain copy. ``{retry}`` is the retry command.


def is_network_error(exc: BaseException) -> bool:
    """True for connection/DNS/timeout failures from httpx, requests or the stdlib."""
    if isinstance(exc, SignInCopyError):
        return False
    names = {cls.__name__ for cls in type(exc).__mro__}
    return bool(names & _NETWORK_ERROR_TYPES) or isinstance(exc, (ConnectionError, TimeoutError))


def is_cancelled(exc: BaseException) -> bool:
    return isinstance(exc, (KeyboardInterrupt, EOFError)) or (
        isinstance(exc, SystemExit) and exc.code in (130, None, 0))


def _details_line(exc: BaseException) -> str:
    text = str(exc).strip() or type(exc).__name__
    return f"  Details: {text}"


_Rule = Tuple[Callable[[BaseException], bool], str]


def _classify(exc: BaseException, rules: Sequence[_Rule], other: str) -> str:
    return next((copy for pred, copy in rules if pred(exc)), other)


def sign_in_failure_lines(
    exc: BaseException, *, service_host: str = "portal.nousresearch.com", retry_command: str = "hermes portal",
) -> list:
    """Lines to print when a device-code / browser sign-in fails for any non-timeout reason."""
    if isinstance(exc, SignInCopyError):
        return str(exc).splitlines()
    rules: Sequence[_Rule] = (
        (is_cancelled, "Sign-in was cancelled. Run `{retry}` when you want to try again."),
        (is_network_error,
         "Could not sign in: Hermes could not reach {host}. Check your internet connection or proxy, "
         "then run `{retry}` again."),
    )
    lead = _classify(
        exc, rules,
        "Could not sign in. Run `{retry}` to try again, or `hermes model` to pick a different provider.")
    lines = [lead.format(host=service_host, retry=retry_command)]
    if not is_cancelled(exc):
        lines.append(_details_line(exc))
    return lines


def provider_setup_failure_lines(exc: BaseException, *, retry_command: str = "hermes model") -> list:
    """Lines to print when the setup wizard's provider step fails: reason, that nothing was saved,
    and how to retry."""
    nothing_saved = (
        "Your provider settings were not changed. Continue the wizard now and run "
        f"`{retry_command}` afterwards to try again.")
    if isinstance(exc, SignInCopyError):
        lead, *details = str(exc).splitlines()
        return [f"Could not finish connecting a provider: {lead[0].lower()}{lead[1:]}", nothing_saved, *details]
    rules: Sequence[_Rule] = (
        (is_cancelled, "sign-in was cancelled"),
        (is_network_error, "no internet connection, or the provider could not be reached"),
    )
    reason = _classify(exc, rules, "something went wrong while talking to the provider")
    lines = [f"Could not finish connecting a provider ({reason}).", nothing_saved]
    if not is_cancelled(exc):
        lines.append(_details_line(exc))
    return lines


from auth.errors import AuthError
from auth.failure_policy import is_rate_limited_auth_error
from auth.providers.nous import _format_nous_entitlement_auth_error
from hermes_cli.config_credentials import credential_pool_environment as _phase6_auth_environment

_GENERIC_ENTITLEMENT_MESSAGES = {
    "subscription_required": "No active paid subscription found. Please purchase/activate a subscription, then retry.",
    "insufficient_credits": "Subscription credits are exhausted. Top up/renew credits, then retry."}
_ENTITLEMENT_ERROR_CODES = frozenset(_GENERIC_ENTITLEMENT_MESSAGES) | {
    "subscription_expired", "no_usable_credits", "account_missing", "member_spend_cap_exceeded"}

def format_auth_error(error: Exception) -> str:
    """Map auth failures to concise user-facing guidance."""
    if not isinstance(error, AuthError) or is_rate_limited_auth_error(error):
        # Rate-limit / quota errors are not credential problems: never append "re-authenticate".
        return str(error)
    if error.relogin_required:
        # Profile-aware: a bare `hermes model` from a named profile re-signs the ROOT store (#114012).
        from hermes_constants import profile_cli_selector

        return f"{error} Run `hermes {profile_cli_selector()}model` to re-authenticate."
    if error.code in _ENTITLEMENT_ERROR_CODES:
        if error.provider == "nous":
            return _format_nous_entitlement_auth_error(error, environment=_phase6_auth_environment())
        generic = _GENERIC_ENTITLEMENT_MESSAGES.get(error.code)
        if generic:
            return generic
    if error.code == "temporarily_unavailable":
        return f"{error} Please retry in a few seconds."
    return str(error)
