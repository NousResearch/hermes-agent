"""Plain-language copy for sign-in and provider-setup failures (CLI).

One table of ``(predicate, lead sentence)`` classifies an exception into a sentence a first-time
user can act on; the raw exception is demoted to a ``Details:`` line so nothing is lost for
support. Callers print ``sign_in_failure_lines(...)`` / ``provider_setup_failure_lines(...)``
line by line.
"""

from __future__ import annotations

from typing import Callable, Sequence, Tuple

# httpx class names and stdlib bases that mean "the request never got a usable answer".
_NETWORK_ERROR_TYPES = frozenset({
    "ConnectError", "ConnectTimeout", "ReadTimeout", "PoolTimeout", "WriteTimeout", "TimeoutException",
    "RemoteProtocolError", "ReadError", "ProxyError", "UnsupportedProtocol", "NetworkError",
})

# OAuth device-flow error codes (RFC 8628 §3.5) -> plain copy. ``{retry}`` is the retry command.
DEVICE_FLOW_ERROR_COPY = {
    "expired_token": (
        "The sign-in code expired before it was approved in the browser. Run `{retry}` to get a new code."),
    "access_denied": (
        "Sign-in was declined in the browser. Run `{retry}` to try again, or `hermes model` to pick a "
        "different provider."),
    "invalid_grant": (
        "The sign-in code was not accepted by the server. Run `{retry}` to get a new code."),
    "invalid_client": (
        "The server did not recognize this copy of Hermes. Run `hermes update`, then `{retry}` again."),
}


class SignInCopyError(RuntimeError):
    """Exception whose ``str()`` is already user copy (lead line + ``Details:`` line)."""

    def __init__(self, message: str, *, oauth_error_code: str = "") -> None:
        super().__init__(message)
        self.oauth_error_code = oauth_error_code


def is_network_error(exc: BaseException) -> bool:
    """True for connection/DNS/timeout failures from httpx, requests or the stdlib."""
    if isinstance(exc, SignInCopyError):
        return False
    names = {cls.__name__ for cls in type(exc).__mro__}
    return bool(names & _NETWORK_ERROR_TYPES) or isinstance(exc, (ConnectionError, TimeoutError))


def is_cancelled(exc: BaseException) -> bool:
    return isinstance(exc, (KeyboardInterrupt, EOFError)) or (
        isinstance(exc, SystemExit) and exc.code in (130, None, 0))


def device_flow_error(code: str, description: str, *, retry_command: str = "hermes portal") -> SignInCopyError:
    """Exception for an OAuth device-flow error code whose text is already user-facing.

    Unknown codes keep the server's description as the lead (it is the only information available)
    but still name the retry command.
    """
    lead = DEVICE_FLOW_ERROR_COPY.get(code, "").format(retry=retry_command)
    if not lead:
        lead = (f"Sign-in did not complete: {description or 'the server rejected the request'}. "
                f"Run `{retry_command}` to try again.")
    details = f"{code}: {description}" if code else description
    return SignInCopyError(f"{lead}\n  Details: {details}" if details else lead, oauth_error_code=code)


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
