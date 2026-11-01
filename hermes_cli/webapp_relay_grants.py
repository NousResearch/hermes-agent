"""The spoke side of the gateway relay: grants it issued to hubs, and the requests they sign.

A hub's pairing sends the browser here (``/webapp/relay/authorize``). A user signed in to this
gateway approves it on the consent page, which mints a one-use code bound to that login's identity,
the hub's public key, its callback and a PKCE challenge. The hub exchanges the code
(``/api/webapp/relay/token``, proving the verifier and the key) for a grant, stored here in
``webapp/relay-grants.json`` (user-only): its id, the hub's name, origin and public key, the
identity it acts as, its scopes and the browser credential generation it was approved under.

A relayed request is served as that identity (``serve_relayed``, called by the sign-in gate) once
the signature checks out (``hermes_cli.webapp_relay``), and only while the grant stands: revoking it,
or ``hermes webapp --sign-out`` replacing the browser credential, ends it and closes the sockets it
opened. Scopes narrow what it reaches (``SCOPES``); some routes it never reaches (``refusal``).
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import secrets
import threading
import time
import urllib.parse
import weakref
from pathlib import Path
from typing import Any, Optional

from fastapi import Request
from fastapi.responses import JSONResponse

from hermes_cli import browser_credential
from hermes_cli import webapp_relay as relay

_log = logging.getLogger(__name__)

GRANTS_PATH = Path("webapp") / "relay-grants.json"
CODE_TTL_SECONDS = 120
_MAX_CODES = 64
_MAX_GRANTS = 64

#: Optional scopes a hub may ask for; everything else a signed-in browser reaches (sessions, chat,
#: profiles, settings, cron, skills) comes with every grant.
SCOPES: dict[str, str] = {
    "terminal": "Open terminals on this machine",
    "files": "Read, download and upload files on this machine",
    "display": "View and control bot screens",
    "voice": "Use voice dictation and spoken replies",
}

#: Paths a relayed request never reaches, whatever its scopes: sign-in and its handoffs, Web Access,
#: the relay's own management (except a grant revoking itself) and the relay itself (one hop).
_REFUSED_PREFIXES = ("/api/relay/", "/api/webapp/relay/", "/webapp/", "/auth/", "/login",
                     "/api/auth/", "/api/webapp-access", "/api/webapp/window-ticket")
#: Routes each optional scope opens, matched against the whole path: the Webapp's own surfaces for
#: host files (browsing, transfers, backup archives), terminals (in the page or a desktop window),
#: bot screens and voice.
_SCOPED_ROUTES: tuple[tuple[re.Pattern[str], str], ...] = tuple((re.compile(pattern), scope) for pattern, scope in (
    (r"/api/(fs|files)(/.*)?", "files"), (r"/api/chat/file-upload", "files"),
    (r"/api/ops/(backup/download|import-upload)", "files"),
    (r"/api/(host-terminal|pty|console)(/.*)?", "terminal"), (r"/api/profiles/[^/]+/open-terminal", "terminal"),
    (r"/api/display(/.*)?", "display"), (r"/api/audio(/.*)?", "voice")))
_SOCKET_TICKET_SCOPES = {"host-terminal": "terminal", "speak-stream": "voice", "transcribe-stream": "voice"}
#: What a socket opened with a relayed ticket needs: the upgrade never crosses ``serve_relayed``, so a
#: generic gateway ticket minted through the relay is checked again against the socket it opens.
_SOCKET_PATH_SCOPES = {"/api/pty": "terminal", "/api/console": "terminal", "/api/host-terminal": "terminal",
                       "/api/display/ws": "display", "/api/audio/speak-stream": "voice",
                       "/api/audio/transcribe-stream": "voice"}
#: JSON-RPC namespaces on a relayed ``/api/ws`` each optional scope opens: the gateway's own socket
#: also runs shell and CLI commands, drives the bot screens and records voice (the recorder and the
#: wake word), none of it in a chat turn.
_RPC_SCOPES = {"shell": "terminal", "cli": "terminal", "display": "display", "voice": "voice", "wake": "voice"}
_SELF_REVOKE = re.compile(r"^/api/webapp/relay/grants/([A-Za-z0-9_-]{1,64})$")


# ---- Store ----------------------------------------------------------------------------------

_lock = threading.Lock()
_codes: dict[str, tuple[float, dict[str, Any]]] = {}


def _path(home) -> Path:
    return Path(home) / GRANTS_PATH


def _read(home) -> dict[str, dict[str, Any]]:
    try:
        stored = json.loads(_path(home).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:  # unreadable: no grant stands until the user pairs again
        _log.warning("Gateway relay grants could not be read: %s", exc)
        return {}
    grants = stored.get("grants") if isinstance(stored, dict) else None
    return {gid: grant for gid, grant in (grants or {}).items() if isinstance(grant, dict)}


def _write(home, grants: dict[str, dict[str, Any]]) -> None:
    from gateway.webapp_listener import _close_to_other_users
    from utils import atomic_write_text

    path = _path(home)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _close_to_other_users(path.parent)
    atomic_write_text(path, json.dumps({"grants": grants}, indent=1, sort_keys=True) + "\n", mode=0o600)


def _stands(grant: dict[str, Any]) -> bool:
    """A grant stands while the browser credential it was approved under is the one in use."""
    return browser_credential.is_current(grant.get("credential"))


def grant_for(home, grant_id: str) -> Optional[dict[str, Any]]:
    """The standing grant *grant_id*, with its id, or None."""
    with _lock:
        grant = _read(home).get(grant_id)
    if grant is None or not _stands(grant):
        return None
    return {**grant, "id": grant_id}


def list_grants(home, identity: dict[str, str]) -> list[dict[str, Any]]:
    """The standing grants *identity* approved, without their keys, newest first."""
    with _lock:
        grants = _read(home)
    rows = [{"id": gid, "hub_name": g.get("hub_name", ""), "hub_origin": g.get("hub_origin", ""),
             "scopes": g.get("scopes", []), "created_at": g.get("created_at", 0)}
            for gid, g in grants.items() if _stands(g) and _same_identity(g, identity)]
    return sorted(rows, key=lambda row: row["created_at"], reverse=True)


def revoke(home, grant_id: str) -> bool:
    """End grant *grant_id* and close the sockets it opened; False when there was none."""
    with _lock:
        grants = _read(home)
        removed = grants.pop(grant_id, None)
        if removed is not None:
            _write(home, grants)
    closed = close_relayed_sockets(grant_id)
    if removed is not None:
        _log.info("Gateway relay grant %s for %s revoked; %d sockets closed",
                  grant_id, removed.get("hub_origin"), closed)
    return removed is not None


def _same_identity(grant: dict[str, Any], identity: dict[str, str]) -> bool:
    return (grant.get("subject") == identity.get("user_id") and grant.get("provider") == identity.get("provider")
            and (grant.get("issuer") or "") == (identity.get("issuer") or ""))


def identity_of(session) -> dict[str, str]:
    """The identity fields a grant binds, from a verified dashboard-auth Session."""
    return {"user_id": session.user_id, "provider": session.provider, "issuer": getattr(session, "issuer", "") or ""}


# ---- Authorization ----------------------------------------------------------------------------

class AuthorizeError(ValueError):
    """An authorize request the consent page cannot show; ``str()`` says why."""


def parse_authorize(params) -> dict[str, Any]:
    """The pairing request a hub sent the browser here with, validated. Raises AuthorizeError."""
    redirect_uri = str(params.get("redirect_uri") or "")
    parts = urllib.parse.urlsplit(redirect_uri)
    if (parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password
            or parts.query or parts.fragment or not parts.path.endswith("/webapp/relay/callback")):
        raise AuthorizeError("The gateway asking for access sent an invalid return address.")
    public_key = str(params.get("public_key") or "")
    try:
        relay.load_public_key(public_key)
    except ValueError:
        raise AuthorizeError("The gateway asking for access sent an invalid key.") from None
    challenge = str(params.get("code_challenge") or "")
    if params.get("code_challenge_method") != "S256" or not re.fullmatch(r"[A-Za-z0-9_-]{43}", challenge):
        raise AuthorizeError("The gateway asking for access sent an invalid PKCE challenge.")
    state = str(params.get("state") or "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{16,128}", state):
        raise AuthorizeError("The gateway asking for access sent an invalid state.")
    hub_name = " ".join(str(params.get("hub_name") or "").split())[:64] or parts.hostname
    requested = [s for s in str(params.get("scope") or "").split() if s in SCOPES]
    return {"redirect_uri": redirect_uri, "hub_origin": f"{parts.scheme}://{parts.netloc}",
            "public_key": public_key, "code_challenge": challenge, "state": state, "hub_name": hub_name,
            "scopes": sorted(set(requested))}


def issue_code(request: Request, pairing: dict[str, Any], scopes: list[str]) -> str:
    """The one-use code that approves *pairing* as *request*'s signed-in user, with *scopes*."""
    session = request.state.session
    now = time.monotonic()
    code = secrets.token_urlsafe(32)
    with _lock:
        for stale in [c for c, (until, _p) in _codes.items() if until <= now]:
            del _codes[stale]
        if len(_codes) >= _MAX_CODES:
            raise AuthorizeError("Too many pairings are waiting; try again in a minute.")
        _codes[code] = (now + CODE_TTL_SECONDS, {
            **pairing, "scopes": sorted(set(scopes) & set(pairing["scopes"])),
            "identity": identity_of(session), "credential": browser_credential.generation()})
    return code


def exchange_code(home, *, code: str, verifier: str, redirect_uri: str, audience: str,
                  proof: str) -> dict[str, Any]:
    """Spend *code* for a grant: the verifier must match its challenge, *redirect_uri* its callback
    and *proof* must be signed by its hub's key. *audience* is this gateway's base URL as the hub
    knows it; the hub's requests under the grant are signed for it. Raises PermissionError."""
    with _lock:
        entry = _codes.pop(code, None)
    if entry is None or entry[0] <= time.monotonic():
        raise PermissionError("code unknown, used or expired")
    pairing = entry[1]
    if (pairing["redirect_uri"] != redirect_uri or not relay.pkce_matches(verifier, pairing["code_challenge"])
            or not relay.verify_token_request(pairing["public_key"], code, audience, proof)):
        raise PermissionError("code exchange does not match the pairing")
    if not browser_credential.is_current(pairing["credential"]):
        raise PermissionError("this gateway signed out since the pairing was approved")
    identity = pairing["identity"]
    grant_id = secrets.token_urlsafe(16)
    grant = {"hub_name": pairing["hub_name"], "hub_origin": pairing["hub_origin"],
             "public_key": pairing["public_key"], "audience": audience, "subject": identity["user_id"],
             "provider": identity["provider"], "issuer": identity["issuer"], "scopes": pairing["scopes"],
             "credential": pairing["credential"], "created_at": int(time.time())}
    with _lock:
        grants = {gid: g for gid, g in _read(home).items() if _stands(g)}
        if len(grants) >= _MAX_GRANTS:
            raise PermissionError("this gateway holds too many relay grants; revoke one first")
        grants[grant_id] = grant
        _write(home, grants)
    _log.info("Gateway relay grant %s issued to %s (%s) for %s", grant_id, pairing["hub_name"],
              pairing["hub_origin"], identity["provider"])
    return {"grant_id": grant_id, "subject": identity["user_id"], "provider": identity["provider"],
            "issuer": identity["issuer"], "scopes": pairing["scopes"]}


# ---- Relayed requests -------------------------------------------------------------------------

def refusal(method: str, path: str, query: str, grant: dict[str, Any]) -> Optional[str]:
    """Why a relayed request under *grant* may not reach ``method path?query``, or None."""
    own = _SELF_REVOKE.match(path)
    if own and method == "DELETE" and own.group(1) == grant["id"]:
        return None
    if any(path == p.rstrip("/") or path.startswith(p) for p in _REFUSED_PREFIXES):
        return "this route is not available through a gateway relay"
    needed = next((scope for route, scope in _SCOPED_ROUTES if route.fullmatch(path)), None)
    if path == "/api/webapp/socket-ticket":
        sockets = urllib.parse.parse_qs(query, keep_blank_values=True).get("socket", [""])
        if len(sockets) != 1 or sockets[0] == "relay":
            return "this route is not available through a gateway relay"
        needed = _SOCKET_TICKET_SCOPES.get(sockets[0])
    if needed is not None and needed not in grant.get("scopes", []):
        return f"this gateway relay was not granted {needed} access"
    return None


def _relay_target(request: Request) -> str:
    raw = request.scope.get("raw_path") or request.url.path.encode("utf-8")
    query = request.scope.get("query_string") or b""
    target = raw.decode("latin-1")
    return f"{target}?{query.decode('latin-1')}" if query else target


def _rejected(reason: str, status: int = 401) -> JSONResponse:
    return JSONResponse({"error": "relay_unauthorized" if status == 401 else "relay_forbidden",
                         "detail": reason}, status_code=status)


async def serve_relayed(request: Request, call_next):
    """Serve a request a hub relayed, as the identity its grant names, or refuse it. Called by the
    sign-in gate for any request that carries relay headers."""
    from hermes_cli.dashboard_auth.audit import AuditEvent, audit_log
    from hermes_cli.dashboard_auth.base import Session
    from hermes_cli.dashboard_auth.request_utils import client_ip

    state = request.app.state
    if not relay.relay_enabled(state):
        return _rejected("this gateway does not accept relayed requests")
    if request.headers.get("cookie"):
        return _rejected("a relayed request must not carry cookies")
    try:
        grant_id, subject = relay.signed_grant(request.headers)
        grant = grant_for(state.webapp_home, grant_id)
        if grant is None or grant["subject"] != subject:
            raise relay.RelayAuthError("relay grant unknown or revoked")
        relay.verify_request(grant["public_key"], request.headers, method=request.method,
                             target=_relay_target(request), audience=grant.get("audience", ""))
    except relay.RelayAuthError as exc:
        audit_log(AuditEvent.TOKEN_AUTH_FAILURE, reason=f"relay: {exc}", path=request.url.path,
                  ip=client_ip(request))
        return _rejected(str(exc))
    reason = refusal(request.method.upper(), request.url.path, request.url.query, grant)
    if reason is not None:
        return _rejected(reason, 403)
    request.state.session = Session(
        user_id=grant["subject"], email="", display_name=grant["hub_name"], org_id="", provider=grant["provider"],
        expires_at=int(time.time()) + relay.CLOCK_WINDOW_SECONDS, access_token="", refresh_token="",
        issuer=grant.get("issuer", ""))
    # The grant stands only under the browser credential in use now: its sockets close at sign-out too.
    request.state.webapp_sign_in = grant["credential"]
    request.state.relay_grant = grant
    return await call_next(request)


# ---- Relayed sockets --------------------------------------------------------------------------

_sockets_lock = threading.Lock()
_sockets: dict[int, tuple[weakref.ref, asyncio.AbstractEventLoop, str]] = {}


def admit_relayed_socket(ws, grant_id: str) -> Optional[str]:
    """Why *ws*, opened with a ticket a relayed request minted under *grant_id*, may not open, or
    None: then it is kept, so revoking the grant closes it."""
    home = getattr(getattr(getattr(ws, "app", None), "state", None), "webapp_home", None)
    grant = grant_for(home, grant_id) if home is not None else None
    if grant is None:
        return "ticket minted under a gateway relay grant that has since been revoked"
    needed = _SOCKET_PATH_SCOPES.get(ws.url.path)
    if needed is not None and needed not in grant.get("scopes", []):
        return f"this gateway relay was not granted {needed} access"
    with contextlib.suppress(AttributeError):
        ws._hermes_relay_grant = grant_id
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    with _sockets_lock:
        for key in [key for key, (ref, *_rest) in _sockets.items() if ref() is None]:
            del _sockets[key]
        _sockets[id(ws)] = (weakref.ref(ws), loop, grant_id)
    return None


def relay_grant_of(ws) -> Optional[str]:
    """The relay grant live socket *ws* was opened under, or None. A socket it opens on the hub's
    behalf (a screen viewer) records the same one."""
    return getattr(ws, "_hermes_relay_grant", None) if ws is not None else None


def _profile_name(arg: str) -> bool:
    from hermes_cli.profiles import validate_profile_name
    try:
        validate_profile_name(arg)
    except ValueError:
        return False
    return True


def _any_text(_arg: str) -> bool:
    return True


#: ``cli.exec`` command lines open without terminal access: Bot Mode's profile edits, which the
#: profile and settings routes every grant reaches make as well. A word is itself, a test any argument.
_PROFILE_EDITS = (("profile", "describe", _profile_name, "--text", _any_text),
                  ("profile", "delete", _profile_name, "--yes"),
                  ("--profile", _profile_name, "config", "unset", "model"))


def _profile_edit(argv) -> bool:
    return isinstance(argv, list) and any(
        len(argv) == len(shape) and all(
            isinstance(arg, str) and (arg == want if isinstance(want, str) else want(arg))
            for arg, want in zip(argv, shape))
        for shape in _PROFILE_EDITS)


def relay_rpc_refusal(ws, method, params=None) -> Optional[str]:
    """Why JSON-RPC *method* with *params* may not run on *ws*, a socket another gateway relays here, or
    None (also for a socket nobody relays): its namespace needs a scope the grant lacks, unless it is
    one of Bot Mode's profile edits."""
    grant_id = relay_grant_of(ws)
    needed = _RPC_SCOPES.get(str(method or "").split(".", 1)[0]) if grant_id is not None else None
    if needed is None or (method == "cli.exec" and isinstance(params, dict) and _profile_edit(params.get("argv"))):
        return None
    home = getattr(getattr(getattr(ws, "app", None), "state", None), "webapp_home", None)
    grant = grant_for(home, grant_id) if home is not None else None
    if grant is None:
        return "ticket minted under a gateway relay grant that has since been revoked"
    if needed not in grant.get("scopes", []):
        return f"this gateway relay was not granted {needed} access"
    return None


async def _close(ws) -> None:
    from starlette.websockets import WebSocketDisconnect
    with contextlib.suppress(RuntimeError, OSError, WebSocketDisconnect):
        await ws.close(code=4401, reason="Gateway relay revoked")


def close_relayed_sockets(grant_id: str) -> int:
    """Close every live socket opened under *grant_id*; returns how many were asked to close."""
    with _sockets_lock:
        live = [(key, entry) for key, entry in _sockets.items() if entry[2] == grant_id]
        for key, _entry in live:
            del _sockets[key]
    closing = 0
    for _key, (ref, loop, _gid) in live:
        ws = ref()
        if ws is not None and not loop.is_closed():
            asyncio.run_coroutine_threadsafe(_close(ws), loop)
            closing += 1
    return closing


def _reset_for_tests() -> None:
    with _lock:
        _codes.clear()
    with _sockets_lock:
        _sockets.clear()
