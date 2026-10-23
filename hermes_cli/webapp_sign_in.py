"""Behind the sign-in gate, a Webapp login lasts only while the browser credential it is bound to does.

On the gateway's Webapp (``app.state.ui_surface == "webapp"``) with the sign-in gate engaged, every
successful cookie login (a password login or an OAuth callback) also sets the cookie
``hermes_webapp_signin_<gateway>`` (``<gateway>`` from a digest of the gateway's home, so two gateways'
Webapps on one hostname keep their own): the
hex HMAC-SHA256 of ``"webapp-sign-in"`` keyed with the browser session token the gateway serves
(``web_server._SESSION_TOKEN``, kept in the gateway home by ``gateway.webapp_listener``). The gate treats
a cookie session without a matching value as signed out. Replacing that token (``hermes webapp
--sign-out``, or the CLI replacing the stored token while no gateway runs) therefore signs every browser
out, whichever provider it signed in with.

A request that passes records the digest of the token it was checked against on
``request.state.webapp_sign_in``; the socket tickets it mints carry that digest, so the sockets they open
are tracked (``web_session_sockets.track``) and close at sign-out, and a ticket minted before a sign-out
opens nothing after it. The standalone dashboard never sets or checks the binding.

Such a login's cookies also hold the sign-in provider's access and refresh token. Presented on their own
(``Authorization: Bearer``, ``/auth/native/refresh``) they would skip the binding, and a list of tokens
to refuse cannot catch them: a provider may accept many spellings of one token (the bundled password
provider decodes base64 leniently, so the quoted cookie value verifies too). So on the gateway's Webapp
behind the gate a provider token works on its own only when this listener gave it to a native app's
sign-in (RFC 8252: ``/auth/native/token``, then ``/auth/native/refresh``). The listener keeps a digest of
each such token until it expires, in ``RECORD_PATH`` under the gateway home (``app.state.webapp_home``),
so native apps stay signed in across a restart and a sign-out. A browser login's tokens therefore work
only as that login's cookies and end with it.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import Request
from fastapi.responses import Response

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


def _current_token() -> str:
    from hermes_cli.web_server import _SESSION_TOKEN
    return _SESSION_TOKEN


def check(request: Request) -> bool:
    """Whether the cookie login on *request* is bound to the browser credential in use now. When it
    is, records that credential's digest on ``request.state.webapp_sign_in``."""
    from hermes_cli.dashboard_auth.cookies import read_webapp_signin_cookie
    from hermes_cli.web_session_sockets import token_digest

    token = _current_token()
    presented = read_webapp_signin_cookie(request, _gateway(request)) or ""
    if not hmac.compare_digest(presented.encode(), binding(token).encode()):
        return False
    request.state.webapp_sign_in = token_digest(token)
    return True


def sign_in_of(request: Request) -> Optional[str]:
    """The browser credential digest *request*'s login was checked against, or None off the Webapp's
    gate (the socket tickets it mints then carry no binding)."""
    return getattr(request.state, "webapp_sign_in", None)


def _set(response: Response, request: Request, token: str) -> None:
    from hermes_cli.dashboard_auth.cookies import detect_https, set_webapp_signin_cookie
    from hermes_cli.dashboard_auth.prefix import prefix_from_request

    set_webapp_signin_cookie(response, value=binding(token), use_https=detect_https(request),
                             prefix=prefix_from_request(request), gateway=_gateway(request))


def bind(response: Response, request: Request) -> None:
    """Bind the login *request* just completed to the browser credential in use now."""
    if applies(request):
        _set(response, request, _current_token())


def renew(response: Response, request: Request) -> None:
    """Keep the binding as long-lived as the session cookies a refresh just rotated, unless the
    credential it was checked against was replaced while the request ran."""
    from hermes_cli.web_session_sockets import token_digest

    signed_in = sign_in_of(request)
    token = _current_token()
    if signed_in is not None and hmac.compare_digest(signed_in, token_digest(token)):
        _set(response, request, token)


def token_works_alone(request: Request, token: str) -> bool:
    """Whether provider token *token* may be used on its own (a bearer header, the refresh route) on
    *request*'s listener: anywhere but the gateway's Webapp behind the gate, as on main; there, only
    when this listener gave it to a native app's sign-in and it has not expired."""
    if not applies(request):
        return True
    with _records_lock:
        expires = _loaded(_home(request)).get(_digest(token))
    return expires is not None and expires > time.time()


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
