"""Server-side RFC 8252 sign-in to an oauth-mode saved gateway, and its token upkeep.

Native Desktop signs in to an oauth-mode remote/cloud entry with its own PKCE pair and a loopback
listener (``electron/native-oauth.ts`` + ``native-oauth-login.ts``). The Webapp has no listener of
its own, so the gateway SERVING it takes that role: it holds the PKCE verifier, hands the browser
the remote's ``/auth/native/authorize`` URL, and its own route
``/webapp/gateway-connections/oauth/callback`` is the loopback redirect. The remote needs no new
code: it is the same native flow Desktop drives (``hermes_agent/hermes_cli/dashboard_auth/routes.py``), which
accepts only a loopback IP-literal redirect, so the callback is addressed to
``127.0.0.1:<serving port>``. A browser on another machine lands on its own loopback instead; it
completes the same flight by pasting that address (:func:`complete_from_url`).

Tokens go to the store's token file (:func:`gateway_connections.write_tokens`), keyed by the
connection id, as the ``NativeTokenSet`` Desktop stores. :func:`ensure_access_token` refreshes them
through the remote's ``/auth/native/refresh`` (single flight per connection, as
``electron/native-access-token.ts``): a 401 there drops the sign-in, anything else keeps it.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from hermes_agent.hermes_cli import gateway_connections as store

NATIVE_FLOW_ID = "native_pkce"
CALLBACK_PATH = "/webapp/gateway-connections/oauth/callback"
#: The remote keeps a pending native authorization this long (``native_flow._PENDING_TTL_SECONDS``).
LOGIN_TTL_S = 600
#: Refresh this early (``native-oauth.ts::tokenNeedsRefresh``, the server's cookie floor).
REFRESH_SKEW_S = 60
_HTTP_TIMEOUT_S = 15.0
_MAX_FLIGHTS = 64


class OAuthSignInError(RuntimeError):
    """A sign-in could not start or complete (the message is safe to show)."""


class NotSignedIn(PermissionError):
    """The oauth-mode connection has no usable sign-in: the user must sign in again."""


@dataclass
class _Flight:
    connection_id: str
    base_url: str
    verifier: str
    redirect_uri: str
    authorize_url: str
    expires_at: float


_flights: dict[str, _Flight] = {}  # state -> flight
_flights_lock = threading.Lock()
_refresh_locks: dict[str, threading.Lock] = {}
_refresh_locks_guard = threading.Lock()


# ── Pure helpers (ports of electron/native-oauth.ts) ─────────────────────────


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def generate_pkce_pair() -> tuple[str, str]:
    """``(verifier, S256 challenge)``: 32 random bytes, RFC 7636 §4."""
    verifier = _b64url(secrets.token_bytes(32))
    return verifier, _b64url(hashlib.sha256(verifier.encode("ascii")).digest())


def _endpoint(base_url: str, path: str) -> str:
    parsed = urlparse(base_url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}{path}"


def build_authorize_url(base_url: str, *, challenge: str, redirect_uri: str, state: str,
                        provider: str = "") -> str:
    query = {"code_challenge": challenge, "code_challenge_method": "S256",
             "redirect_uri": redirect_uri, "state": state}
    if provider:
        query["provider"] = provider
    return f"{_endpoint(base_url, '/auth/native/authorize')}?{urlencode(query)}"


def status_supports_native_flow(status_body: Any) -> bool:
    flows = status_body.get("auth_flows") if isinstance(status_body, dict) else None
    return isinstance(flows, list) and NATIVE_FLOW_ID in flows


def parse_token_response(body: Any) -> dict:
    """A remote ``/auth/native/token|refresh`` body as a ``NativeTokenSet``."""
    body = body if isinstance(body, dict) else {}
    access = str(body.get("access_token") or "")
    if not access:
        raise OAuthSignInError("The gateway's token response has no access token.")
    try:
        expires_at = float(body.get("expires_at"))
    except (TypeError, ValueError):
        expires_at = 0
    return {"accessToken": access, "refreshToken": str(body.get("refresh_token") or ""),
            "expiresAt": int(expires_at), "provider": str(body.get("provider") or ""),
            "userId": str(body.get("user_id") or "")}


def token_needs_refresh(tokens: dict, now: Optional[float] = None) -> bool:
    expires_at = tokens.get("expiresAt") if isinstance(tokens, dict) else None
    if not isinstance(expires_at, (int, float)) or expires_at <= 0:
        return True  # unknown expiry: validate before use
    return (time.time() if now is None else now) >= expires_at - REFRESH_SKEW_S


def loopback_redirect_uri(bound_host: str, bound_port: int) -> str:
    """The remote accepts only an IP-literal loopback redirect (RFC 8252 §8.3)."""
    host = "[::1]" if (bound_host or "").strip("[]") == "::1" else "127.0.0.1"
    return f"http://{host}:{int(bound_port)}{CALLBACK_PATH}"


# ── Remote calls ─────────────────────────────────────────────────────────────


def _client() -> httpx.Client:
    # No proxy settings from the environment and no redirects: the remote's answer is read as is.
    return httpx.Client(timeout=_HTTP_TIMEOUT_S, trust_env=False, follow_redirects=False)


def _connection(connection_id: str) -> tuple[dict, str, dict]:
    """``(entry, base url, extra headers)`` for an oauth-mode remote/cloud *connection_id*, from one read."""
    snap = _snapshot(connection_id)
    return snap["entry"], snap["url"], snap["headers"]


def _post_json(url: str, body: dict, headers: dict) -> httpx.Response:
    with _client() as client:
        return client.post(url, json=body, headers=headers)


def _remote_status(base_url: str, headers: dict) -> dict:
    try:
        with _client() as client:
            response = client.get(_endpoint(base_url, "/api/status"), headers=headers)
    except httpx.HTTPError as exc:
        raise OAuthSignInError(f"Could not reach {base_url}: {exc}") from exc
    if response.status_code != 200:
        raise OAuthSignInError(f"{base_url} answered /api/status with {response.status_code}.")
    try:
        return response.json()
    except ValueError as exc:
        raise OAuthSignInError(f"{base_url} did not answer /api/status with JSON.") from exc


# ── Sign-in flights (one per connection, as electron/oauth-login-flights.ts) ─


def _gc_flights_locked(now: float) -> None:
    for state in [s for s, f in _flights.items() if f.expires_at <= now]:
        _flights.pop(state, None)


def start_sign_in(connection_id: str, *, bound_host: str, bound_port: int, provider: str = "") -> dict:
    """Begin a sign-in for *connection_id*; returns ``{authorizeUrl, redirectUri, expiresAt}``.

    A sign-in already running for the connection is handed back instead of starting a second one:
    the remote keys its PKCE cookie per browser, so a second start would fail the first's callback.
    """
    _entry, base_url, headers = _connection(connection_id)
    redirect_uri = loopback_redirect_uri(bound_host, bound_port)
    now = time.time()
    with _flights_lock:
        _gc_flights_locked(now)
        for flight in _flights.values():
            if (flight.connection_id == connection_id and flight.base_url == base_url
                    and flight.redirect_uri == redirect_uri):
                return _flight_view(flight)
    if not status_supports_native_flow(_remote_status(base_url, headers)):
        raise OAuthSignInError(
            f"{base_url} does not offer native sign-in (no interactive sign-in provider, or an older "
            "Hermes). Update that gateway or connect with a session token.")
    verifier, challenge = generate_pkce_pair()
    state = _b64url(secrets.token_bytes(24))
    flight = _Flight(connection_id=connection_id, base_url=base_url, verifier=verifier,
                     redirect_uri=redirect_uri, expires_at=now + LOGIN_TTL_S,
                     authorize_url=build_authorize_url(base_url, challenge=challenge,
                                                       redirect_uri=redirect_uri, state=state,
                                                       provider=provider))
    with _flights_lock:
        _gc_flights_locked(now)
        if len(_flights) >= _MAX_FLIGHTS:
            raise OAuthSignInError("Too many sign-ins are in progress; finish or wait for one.")
        _flights[state] = flight
    return _flight_view(flight)


def _flight_view(flight: _Flight) -> dict:
    return {"authorizeUrl": flight.authorize_url, "redirectUri": flight.redirect_uri,
            "expiresAt": int(flight.expires_at)}


def pending_sign_in(connection_id: str) -> bool:
    with _flights_lock:
        _gc_flights_locked(time.time())
        return any(f.connection_id == connection_id for f in _flights.values())


def cancel_sign_in(connection_id: str) -> None:
    with _flights_lock:
        for state in [s for s, f in _flights.items() if f.connection_id == connection_id]:
            _flights.pop(state, None)


def complete_sign_in(*, code: str, state: str, error: str = "", error_description: str = "",
                     connection_id: str = "") -> dict:
    """Finish the flight *state* names: redeem *code* with its verifier and store the tokens.

    The flight is consumed on every path (one try per state, as the remote's one-time code), and an
    unknown state is refused before anything else is read (RFC 6749 §10.12). With *connection_id*,
    a flight started for another connection is refused before its code is redeemed, and left alone."""
    with _flights_lock:
        _gc_flights_locked(time.time())
        flight = _flights.get(state) if state else None
        if flight is not None and connection_id and flight.connection_id != connection_id:
            raise store.ConnectionStoreError("That redirect belongs to another connection's sign-in.")
        flight = _flights.pop(state, None) if state else None
    if flight is None:
        raise OAuthSignInError("This sign-in is unknown or expired. Start it again from Settings → Gateways.")
    if error:
        detail = f" ({error_description})" if error_description else ""
        raise OAuthSignInError(f"The gateway refused the sign-in: {error}{detail}")
    if not code:
        raise OAuthSignInError("The sign-in redirect carried no authorization code.")
    try:
        snap = store.dial_snapshot(flight.connection_id)
    except store.NoSuchConnection as exc:
        raise OAuthSignInError("The connection changed while signing in; sign in again.") from exc
    if snap["url"] != flight.base_url:
        # Re-pointed while the browser was away: its headers now belong to the new address.
        raise OAuthSignInError("The connection changed while signing in; sign in again.")
    headers = snap["headers"]
    try:
        response = _post_json(_endpoint(flight.base_url, "/auth/native/token"),
                              {"code": code, "code_verifier": flight.verifier}, headers)
    except httpx.HTTPError as exc:
        raise OAuthSignInError(f"Could not reach {flight.base_url} to finish signing in: {exc}") from exc
    if response.status_code != 200:
        raise OAuthSignInError(f"{flight.base_url} refused the sign-in code ({response.status_code}).")
    tokens = parse_token_response(response.json())
    _store_tokens(flight.connection_id, flight.base_url, tokens)
    return {"connectionId": flight.connection_id, "provider": tokens["provider"], "userId": tokens["userId"]}


def complete_from_url(callback_url: str, *, connection_id: str = "") -> dict:
    """Finish a flight from the address its redirect reached (pasted when that was another
    machine's loopback). Only this gateway's own callback path is read; with *connection_id*, only
    a flight that connection started."""
    parsed = urlparse(str(callback_url or "").strip())
    if parsed.path.rstrip("/") != CALLBACK_PATH:
        raise OAuthSignInError("That address is not a Hermes gateway sign-in redirect.")
    query = {k: v[0] for k, v in parse_qs(parsed.query).items() if v}
    return complete_sign_in(code=query.get("code", ""), state=query.get("state", ""),
                            error=query.get("error", ""), error_description=query.get("error_description", ""),
                            connection_id=connection_id)


def _store_tokens(connection_id: str, base_url: str, tokens: dict) -> None:
    # The entry may have been edited to another URL (or removed) while the browser was away; a
    # sign-in for the old address must not land on the new one. The store re-checks the issuer
    # under its tokens lock, so an edit landing between here and the write is refused too.
    try:
        store.write_tokens(connection_id, tokens, issued_by=base_url)
    except (store.ConnectionStoreError, store.NoSuchConnection) as exc:
        raise OAuthSignInError("The connection changed while signing in; sign in again.") from exc


def sign_out(connection_id: str) -> None:
    cancel_sign_in(connection_id)
    store.write_tokens(connection_id, None)


def sign_in_status(connection_id: str) -> dict:
    """``{signedIn, provider, userId, expiresAt, pending}``; never a token."""
    _connection(connection_id)
    tokens = store.read_tokens(connection_id) or {}
    return {"signedIn": bool(tokens.get("accessToken")), "provider": tokens.get("provider") or "",
            "userId": tokens.get("userId") or "", "expiresAt": tokens.get("expiresAt") or 0,
            "pending": pending_sign_in(connection_id)}


# ── Access tokens (electron/native-access-token.ts) ──────────────────────────


def _refresh_lock(connection_id: str) -> threading.Lock:
    with _refresh_locks_guard:
        return _refresh_locks.setdefault(connection_id, threading.Lock())


@dataclass(frozen=True)
class Grant:
    """An access token with the gateway that issued it and the extra headers read with it: the
    caller sends this token to ``base_url`` only."""
    access_token: str
    base_url: str
    headers: dict


def _snapshot(connection_id: str) -> dict:
    snap = store.dial_snapshot(connection_id)
    entry = snap["entry"]
    if entry.get("kind") not in ("remote", "cloud") or entry.get("authMode") != "oauth" or not snap["url"]:
        raise store.ConnectionStoreError(f'"{entry.get("label")}" does not sign in with OAuth.')
    return snap


def ensure_access_token(connection_id: str, *, rejected_token: str = "") -> Grant:
    """A usable access token for *connection_id*, refreshed when at/near expiry, or when
    *rejected_token* (the one the remote just answered 401 to) is still the stored one.

    The token, its issuer URL and the headers come from one store read, so a re-point landing
    meanwhile never pairs one gateway's address with another's sign-in.

    Raises :class:`NotSignedIn` when there is no sign-in or the remote rejects its refresh token
    (the sign-in is dropped), :class:`OAuthSignInError` when the remote cannot be reached (kept)."""
    with _refresh_lock(connection_id):
        snap = _snapshot(connection_id)
        tokens = snap["tokens"]
        if not tokens:
            raise NotSignedIn("Sign in to this gateway first.")
        forced = bool(rejected_token) and rejected_token == tokens.get("accessToken")
        if not forced and not token_needs_refresh(tokens):
            return _grant(tokens, snap["headers"])
        if not tokens.get("refreshToken"):
            store.write_tokens(connection_id, None)
            raise NotSignedIn("The sign-in expired; sign in again.")
        return _refresh(connection_id, snap["headers"], tokens)


def _grant(tokens: dict, headers: dict) -> Grant:
    return Grant(access_token=tokens["accessToken"], base_url=str(tokens["baseUrl"]).rstrip("/"), headers=dict(headers))


def _refresh(connection_id: str, headers: dict, tokens: dict) -> Grant:
    # The refresh token goes only to the gateway that issued it (the snapshot hands out a set
    # only while the entry still names that URL).
    base_url = str(tokens["baseUrl"]).rstrip("/")
    try:
        response = _post_json(_endpoint(base_url, "/auth/native/refresh"),
                              {"refresh_token": tokens["refreshToken"], "provider": tokens.get("provider") or ""},
                              headers)
    except httpx.HTTPError as exc:
        raise OAuthSignInError(f"Could not reach {base_url} to renew the sign-in: {exc}") from exc
    if response.status_code == 401:
        _drop_if_unchanged(connection_id, tokens)
        raise NotSignedIn("The gateway ended this sign-in; sign in again.")
    if response.status_code != 200:
        raise OAuthSignInError(f"{base_url} could not renew the sign-in ({response.status_code}).")
    rotated = parse_token_response(response.json())
    current = _snapshot(connection_id)
    if not current["tokens"]:
        # Signed out, or re-pointed at another gateway meanwhile: this rotation belongs to neither.
        raise NotSignedIn("The connection changed while renewing its sign-in; sign in again.")
    if current["tokens"].get("accessToken") != tokens.get("accessToken"):
        # A sign-in (Desktop's or a browser's) replaced the tokens meanwhile: it wins, with the
        # address it was issued by.
        return _grant(current["tokens"], current["headers"])
    try:
        store.write_tokens(connection_id, rotated, issued_by=base_url)
    except (store.ConnectionStoreError, store.NoSuchConnection) as exc:
        # Re-pointed or removed while the remote answered: this rotation belongs to neither host.
        raise NotSignedIn("The connection changed while renewing its sign-in; sign in again.") from exc
    return Grant(access_token=rotated["accessToken"], base_url=base_url, headers=dict(headers))


def _drop_if_unchanged(connection_id: str, tokens: dict) -> None:
    current = store.read_tokens(connection_id) or {}
    if current.get("accessToken") == tokens.get("accessToken"):
        store.write_tokens(connection_id, None)
