"""Where Remote Config may send a credential (the plane bearer, the IdP client secret).

The plane URL and the IdP token URL are operator-selected (D19: the plane defaults to
``https://config-config.nousresearch.com`` and an on-prem deployment points it elsewhere), so this
is a transport policy, not an allow-list of hosts:

- ``https`` only; plain ``http`` only to a loopback host (tests, local development);
- one unambiguous origin: no userinfo, query or fragment in the configured URL;
- a credentialed request never follows a redirect, so the credential reaches only the origin the
  operator configured (a 3xx is refused, not re-authorized hop by hop).
"""
from __future__ import annotations

import ipaddress
import urllib.error
import urllib.parse
import urllib.request

from .credentials import PlaneCredentialError


class CredentialDestinationError(PlaneCredentialError):
    """A credential would go somewhere the transport policy forbids. Never retryable: the
    configuration (or the server's redirect) has to change."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def display_url(url: str) -> str:
    """*url* for messages: any user info (``user:password@``) dropped, so a misconfigured URL
    never echoes a credential back (an unparsable one is cut at the last ``@``)."""
    try:
        parts = urllib.parse.urlsplit(url)
        if parts.username is None and parts.password is None:
            return url
        host = parts.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        port = parts.port
    except ValueError:
        head, sep, tail = url.partition("//")
        return f"{head}{sep}{tail.rpartition('@')[2]}" if sep else url.rpartition("@")[2]
    return urllib.parse.urlunsplit(parts._replace(netloc=f"{host}:{port}" if port is not None else host))


def checked_url(url: str, *, env_name: str) -> str:
    """``url`` when a credential may be sent to it, else :class:`CredentialDestinationError`."""
    try:
        parts = urllib.parse.urlsplit(url)
        host = (parts.hostname or "").lower()
        _ = parts.port  # raises ValueError on a malformed port
    except ValueError as exc:
        raise CredentialDestinationError(f"{env_name}={display_url(url)!r} is not a valid URL: {exc}") from exc
    if not host:
        raise CredentialDestinationError(f"{env_name}={display_url(url)!r} has no host.")
    if parts.username is not None or parts.password is not None or parts.query or parts.fragment:
        raise CredentialDestinationError(
            f"{env_name} must be a plain origin and path (no user info, query or fragment); "
            "Remote Config does not send credentials to an ambiguous destination.")
    scheme = parts.scheme.lower()
    if scheme == "https" or (scheme == "http" and _is_loopback(host)):
        return url
    raise CredentialDestinationError(
        f"{env_name} must be an https:// URL (plain http is accepted only for localhost, "
        f"127.0.0.1 or ::1); Remote Config does not send credentials to {parts.scheme or 'no'}://{host}.")


class _RefuseRedirect(urllib.request.BaseHandler):
    """Every 3xx (308 included, whatever the interpreter's redirect handler knows) is refused."""

    def _refuse(self, req, fp, code, msg, headers):
        fp.close()
        raise CredentialDestinationError(
            f"{req.get_full_url()} answered HTTP {code} redirecting to {display_url(headers.get('Location') or '')!r}; "
            "Remote Config never follows a redirect with credentials. Point the URL at the final "
            "destination.")

    http_error_301 = http_error_302 = http_error_303 = http_error_307 = http_error_308 = _refuse


# build_opener drops its default HTTPRedirectHandler when given a subclass only; give it ours
# AND remove the default explicitly so nothing earlier in the chain follows a 3xx.
_OPENER = urllib.request.OpenerDirector()
for _handler in (urllib.request.ProxyHandler(), urllib.request.UnknownHandler(), urllib.request.HTTPHandler(),
                 urllib.request.HTTPSHandler(), urllib.request.HTTPDefaultErrorHandler(), _RefuseRedirect(),
                 urllib.request.HTTPErrorProcessor()):
    _OPENER.add_handler(_handler)


def open_credentialed(req: urllib.request.Request, *, timeout: float):
    """``urlopen`` for a request carrying a credential: a 3xx raises
    :class:`CredentialDestinationError` instead of being followed."""
    return _OPENER.open(req, timeout=timeout)
