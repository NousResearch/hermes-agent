"""Behind the sign-in gate, a Webapp login lasts only while the browser credential it is bound to does.

On the gateway's Webapp (``app.state.ui_surface == "webapp"``) with the sign-in gate engaged, every
successful cookie login (a password login or an OAuth callback) also sets the cookie
``hermes_webapp_signin_<gateway>``: the hex HMAC-SHA256 of ``"webapp-sign-in"`` keyed with the browser
session token the gateway serves (``web_server._SESSION_TOKEN``, kept in the gateway home by ``gateway.webapp_listener``). The gate treats
a cookie session without a matching value as signed out. Replacing that token (``hermes webapp
--sign-out``, or the CLI replacing the stored token while no gateway runs) therefore signs every browser
out, whichever provider it signed in with. ``<gateway>`` comes from a digest of the gateway's home
(:func:`cookie_gateway`), and the Webapp's listener names every sign-in cookie it sets that way, so two
gateways' Webapps on one hostname keep their own logins and leave the standalone dashboard's alone.

A request that passes records the generation of the credential it was checked against
(``hermes_cli.browser_credential``) on ``request.state.webapp_sign_in``; the socket tickets it mints
carry that generation, so the sockets they open are tracked (``web_session_sockets.track``) and close
at sign-out, and a ticket minted before a sign-out opens nothing after it. The standalone dashboard
never sets or checks the binding.

Such a login's cookies also hold the sign-in provider's access and refresh token. Presented on their own
(``Authorization: Bearer``, a socket's ``?token=``, ``/auth/native/refresh``) they would skip the
binding, and a list of tokens to refuse cannot catch them: a provider may accept many spellings of one
token (the bundled password provider decodes base64 leniently, so the quoted cookie value verifies too).
So on the gateway's Webapp behind the gate a provider token works on its own only when this listener
gave it to a native app's sign-in (RFC 8252: ``/auth/native/token``, then ``/auth/native/refresh``). The
listener keeps a digest of each such token until it expires, in ``RECORD_PATH`` under the gateway home
(``app.state.webapp_home``), so native apps stay signed in across a restart and a sign-out. A browser
login's tokens therefore work only as that login's cookies and end with it. The deployment session
token (``HERMES_DASHBOARD_SESSION_TOKEN``) also works as a bearer or ``?token=`` while the
session-token provider is registered (:func:`access_token_works_alone`): no browser login holds it,
and it has no refresh.

A token Nous Portal minted by RFC 8693 token exchange also works on its own there. The hosted dashboard
signs the user in on its own host, exchanges that sign-in for a token whose audience is this agent,
and relays the browser's requests here with it as a bearer. Portal marks every exchanged token with an
``act`` claim naming the client that asked for it (RFC 8693 §4.1). A browser login on this listener
gets its tokens from the authorization-code grant, which never sets ``act``, and a token cannot gain
the claim without Portal's signature over it. What ``act`` means is Portal's to say, so the verdict is
the provider's: :func:`access_token_works_alone` lets a token through only when a registered provider
that issues exchanged tokens (one with ``is_exchanged_token``, the Nous provider) verifies it as one
of its own, signature, issuer, ``aud``, agent instance and scope included, and its verified claims
name an actor. Another issuer's token carrying an ``act`` claim (a self-hosted OIDC ID token) stays
bound to its login. A cheap shape check comes first, so only a well-formed compact JWS signed with an
asymmetric algorithm is ever sent to a provider; a token of another shape (the password provider's,
whose decoder skips characters it does not know) is never read as one, and the provider turns away
a token whose unverified issuer, audience or agent instance is not its own before it looks up any
signing key, so an anonymous caller cannot make the listener fetch keys for tokens addressed
elsewhere. When the provider cannot be reached the request gets 503, as any other sign-in outage
does (never let through, and never read as a logout), and for a while after it checks such tokens
only against keys it already holds (the Nous provider's backoff), so the listener does not wait on
an unreachable key endpoint once per request. The refresh route keeps :func:`token_works_alone`: an
exchanged token is an access token, and the dashboard host exchanges again rather than refreshing here.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import re
import logging
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import Request
from fastapi.responses import Response

from hermes_cli import browser_credential

_MESSAGE = b"webapp-sign-in"
_log = logging.getLogger(__name__)

#: Where the gateway home keeps the digests of the provider tokens its listener gave native apps.
RECORD_PATH = Path("webapp") / "native-sign-in-tokens.json"
#: How long a refresh token stays recorded. Providers do not say when theirs expire; this is the longest
#: the bundled providers issue (the password provider's 30 days) and how long a browser keeps one.
REFRESH_TOKEN_LIFETIME = 30 * 24 * 60 * 60

_records_lock = threading.Lock()
#: Per gateway home: token digest -> expiry in Unix seconds. Read from the home's file once per
#: process; this process is its only writer.
_records: dict[Optional[Path], dict[str, int]] = {}


def binding(token: str) -> str:
    """The ``hermes_webapp_signin`` value for browser session token *token*."""
    return hmac.new(token.encode(), _MESSAGE, hashlib.sha256).hexdigest()


def applies(request: Request) -> bool:
    """Whether *request* reaches the gateway's Webapp behind the sign-in gate."""
    state = request.app.state
    return getattr(state, "ui_surface", None) == "webapp" and bool(getattr(state, "auth_required", False))


def check(request: Request) -> bool:
    """Whether the cookie login on *request* is bound to the browser credential in use now. When it
    is, records that credential's generation on ``request.state.webapp_sign_in``."""
    from hermes_cli.dashboard_auth.cookies import read_webapp_signin_cookie

    token = browser_credential.current()
    presented = read_webapp_signin_cookie(request, _gateway(request)) or ""
    if not hmac.compare_digest(presented.encode(), binding(token).encode()):
        return False
    request.state.webapp_sign_in = browser_credential.generation_of(token)
    return True


def sign_in_of(request: Request) -> Optional[str]:
    """The browser credential generation *request*'s login was checked against, or None off the
    Webapp's gate (the socket tickets it mints then carry no binding)."""
    return getattr(request.state, "webapp_sign_in", None)


def _set(response: Response, request: Request, token: str) -> None:
    from hermes_cli.dashboard_auth.cookies import detect_https, set_webapp_signin_cookie
    from hermes_cli.dashboard_auth.prefix import prefix_from_request

    set_webapp_signin_cookie(response, value=binding(token), use_https=detect_https(request),
                             prefix=prefix_from_request(request), gateway=_gateway(request))


def bind(response: Response, request: Request) -> None:
    """Bind the login *request* just completed to the browser credential in use now."""
    if applies(request):
        _set(response, request, browser_credential.current())


def renew(response: Response, request: Request) -> None:
    """Keep the binding as long-lived as the session cookies a refresh just rotated, unless the
    credential it was checked against was replaced while the request ran."""
    token = browser_credential.current_if(sign_in_of(request))
    if token is not None:
        _set(response, request, token)


def token_works_alone(request: Request, token: str) -> bool:
    """Whether provider token *token* may be used on its own (a bearer header, a socket's
    ``?token=``, the refresh route) on *request*'s listener: anywhere but the gateway's Webapp behind
    the gate, as on main; there, only when this listener gave it to a native app's sign-in and it has
    not expired."""
    if not applies(request):
        return True
    with _records_lock:
        expires = _loaded(_home(request)).get(_digest(token))
    return expires is not None and expires > time.time()


#: A compact JWS: three base64url segments with no padding (RFC 7515 §7.1).
_COMPACT_JWS = re.compile(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+")
#: Signature algorithms a token-exchange issuer signs with: asymmetric only, so no listener secret
#: could have signed one.
_EXCHANGE_ALGORITHMS = frozenset({"RS256", "RS384", "RS512", "PS256", "PS384", "PS512", "ES256", "ES384",
                                  "ES512", "EdDSA"})


def _segment(part: str) -> object:
    try:
        return json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
    except (binascii.Error, ValueError, RecursionError):
        return None


def _unverified_exchange_claims(token: str) -> Optional[dict]:
    """*token*'s claims, decoded WITHOUT verification, when it is shaped like one minted by RFC 8693
    token exchange: a compact JWS, signed with an asymmetric algorithm, whose claims carry an ``act``
    object naming its actor. Else None. Only decides whether a provider is asked."""
    if not token or len(token) > 16384 or not _COMPACT_JWS.fullmatch(token):
        return None
    header_part, claims_part, _signature = token.split(".")
    header, claims = _segment(header_part), _segment(claims_part)
    algorithm = header.get("alg") if isinstance(header, dict) else None
    if not isinstance(algorithm, str) or algorithm not in _EXCHANGE_ALGORITHMS or not isinstance(claims, dict):
        return None
    actor = claims.get("act")
    return claims if isinstance(actor, dict) and isinstance(actor.get("sub"), str) and actor["sub"] else None


def reads_as_exchanged(token: str) -> bool:
    """Whether *token* is shaped like an exchanged token (:func:`_unverified_exchange_claims`)."""
    return _unverified_exchange_claims(token) is not None


def is_exchanged_token(token: str) -> bool:
    """Whether a registered provider verifies *token* as its own exchanged token for this agent
    (``DashboardAuthProvider.is_exchanged_token``, which says no by default). Raises
    ``ProviderError`` when no provider says yes and one could not tell (its keys cannot be
    fetched), so the caller answers 503 rather than sending the browser to sign in again."""
    if not reads_as_exchanged(token):
        return False
    from hermes_cli.dashboard_auth import ProviderError, list_session_providers

    outage: Optional[ProviderError] = None
    for provider in list_session_providers():
        verdict = getattr(provider, "is_exchanged_token", None)
        if not callable(verdict):  # a provider object that does not subclass the protocol
            continue
        try:
            if verdict(token=token):
                return True
        except ProviderError as exc:
            _log.warning("Could not check an exchanged token with provider %r: %s", provider.name, exc)
            outage = outage or exc
    if outage is not None:
        raise outage
    return False


def access_token_works_alone(request: Request, token: str, *, exchanged: bool = True) -> bool:
    """Whether *token* may be used on its own as an access token (a bearer header, a socket's
    ``?token=``): as :func:`token_works_alone` (looked up at call time, so a patch of it applies), when
    it is an exchanged token (:func:`is_exchanged_token`), which the providers then verify, or when it
    is the deployment session token the registered session-token provider holds. ``exchanged=False``
    skips the exchanged-token check, for a place only the deployment token can open. Raises
    ``ProviderError`` when an exchanged-shaped token cannot be checked (a Portal outage). The refresh
    route keeps :func:`token_works_alone`: neither of those has a refresh."""
    return (token_works_alone(request, token) or (exchanged and is_exchanged_token(token))
            or _is_deployment_token(token))


def _is_deployment_token(token: str) -> bool:
    """Whether *token* is the deployment session token the registered session-token provider read at
    startup. No browser login holds it: that provider has no sign-in."""
    from hermes_cli.dashboard_auth import get_provider

    matches = getattr(get_provider("session-token"), "matches_token", None) if token else None
    return callable(matches) and bool(matches(token))


def gave_native_app(request: Request, session) -> None:
    """Record that this listener just gave *session*'s tokens to a native app's sign-in, so they work
    on their own on the gateway's Webapp behind the gate (:func:`token_works_alone`)."""
    if not applies(request):
        return
    now = int(time.time())
    given = ((session.access_token, int(session.expires_at or 0) or now + REFRESH_TOKEN_LIFETIME),
             (session.refresh_token, now + REFRESH_TOKEN_LIFETIME))
    home = _home(request)
    with _records_lock:
        records = _loaded(home)
        for token, expires in given:
            if token:
                records[_digest(token)] = expires
        for key in [key for key, expires in records.items() if expires <= now]:
            del records[key]
        if home is not None:
            _write(home / RECORD_PATH, records)


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _home(request: Request) -> Optional[Path]:
    home = getattr(request.app.state, "webapp_home", None)
    return Path(home) if home is not None else None


def cookie_gateway(request: Request) -> str:
    """The suffix the gateway's Webapp gives every sign-in cookie it sets (``hermes_session_at_<it>``
    and the rest, ``hermes_cli.dashboard_auth.cookies``): cookies are not scoped to a port. ``""`` off
    the Webapp, where the standalone dashboard keeps main's names."""
    return _gateway(request) if getattr(request.app.state, "ui_surface", None) == "webapp" else ""


def _gateway(request: Request) -> str:
    """This gateway's name in its sign-in cookie: a digest of its home, the same across restarts and
    different for every gateway on one machine (each home has its own); ``""`` without a home."""
    home = _home(request)
    return "" if home is None else hashlib.sha256(str(home.absolute()).encode()).hexdigest()[:16]


def _loaded(home: Optional[Path]) -> dict[str, int]:
    """*home*'s record, read from its file on first use (call with ``_records_lock`` held). Without a
    home it lives in memory only."""
    if home in _records:
        return _records[home]
    stored: object = {}
    if home is not None:
        try:
            stored = json.loads((home / RECORD_PATH).read_text(encoding="utf-8-sig"))
        except FileNotFoundError:
            pass
        except (OSError, ValueError) as exc:  # native apps sign in again
            _log.warning("Could not read the Webapp's record of native sign-ins: %s", exc)
    records = ({key: value for key, value in stored.items() if isinstance(value, int)}
               if isinstance(stored, dict) else {})
    _records[home] = records
    return records


def _write(path: Path, records: dict[str, int]) -> None:
    from utils import atomic_write_text

    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        atomic_write_text(path, json.dumps(records, separators=(",", ":")) + "\n", mode=0o600)
    except OSError as exc:  # this process still accepts them; after a restart native apps sign in again
        _log.warning("Could not save the Webapp's record of native sign-ins: %s", exc)


def _reset_for_tests() -> None:
    """Forget every loaded record, as a new process starts with only the files."""
    with _records_lock:
        _records.clear()
