"""Webapp-only routes: the host terminal (loopback or authenticated binds), the
one-use ticket each browser socket dial mints (``/api/webapp/socket-ticket?socket=`` for
``/api/ws`` on this same origin, the host terminal and each voice stream), and the one-use
handoffs (child window, auto-opened launch tab) for the private, loopback Webapp session."""
from __future__ import annotations

import ipaddress
import json
import logging
import os
import secrets
import sys
import threading
import time

from fastapi import APIRouter, HTTPException, Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse

from hermes_cli.dashboard_auth.request_utils import _http_origin
from hermes_cli.web_server_surface import policy, private_launch
from hermes_cli.web_session_sockets import (
    GATEWAY_SOCKET, HOST_TERMINAL_SOCKET, SPEAK_STREAM_SOCKET, TRANSCRIBE_STREAM_SOCKET)

_log = logging.getLogger("hermes_cli.web_server")
router = APIRouter()
WINDOW_TICKET_TTL_SECONDS = 30
#: A sandboxed (snap/Flatpak) browser's cold start can take several seconds to reach the page.
LAUNCH_TICKET_TTL_SECONDS = 60
_MAX_WINDOW_TICKETS = 128
_ticket_lock = threading.Lock()
#: Who a socket opened with the loopback session token is (``chat_ws.gateway_ws`` gives a
#: ``?token=`` socket the same identity, and REST mutations are forwarded as it).
SESSION_TOKEN_OWNER = {"user_id": "legacy-token-owner", "provider": "session-token", "issuer": ""}


def _require_private_webapp(request: Request) -> None:
    if not private_launch(request.app.state):
        raise HTTPException(404, "Private Webapp window handoff is unavailable")


@router.websocket("/api/host-terminal")
async def host_terminal_ws(ws: WebSocket) -> None:
    from hermes_cli import web_host_terminal
    from hermes_cli.web_host_terminal_sessions import host_terminal
    from hermes_cli.web_server_chat import _PTY_BRIDGE_AVAILABLE, PtyUnavailableError, _pty_fail, _ws_gate

    gate = await _ws_gate(ws, HOST_TERMINAL_SOCKET)
    if gate is None:
        return
    peer, mode, cred = gate
    if not web_host_terminal.request_allowed():
        await ws.close(code=4403, reason="host terminal requires authenticated Webapp")
        return
    await ws.accept()
    _log.info("host terminal accepted peer=%s mode=%s cred=%s", peer, mode, cred)
    if not _PTY_BRIDGE_AVAILABLE:
        await _pty_fail(
            ws, PtyUnavailableError("Pseudo-terminal support is not installed on this host."), surface="Terminal")
        return
    await host_terminal(ws)


def _mint_handoff_ticket(state, ttl: float, *, launch: bool) -> str:
    """One-use ticket that ``/webapp/window-session`` exchanges for this start's session token. The
    store keeps ``(expiry, browser credential generation, launch)``, never the credential itself."""
    from hermes_cli import browser_credential

    now = time.monotonic()
    with _ticket_lock:
        tickets = getattr(state, "webapp_window_tickets", {})
        tickets = {key: value for key, value in tickets.items()
                   if value[0] > now and browser_credential.is_current(value[1])}
        state.webapp_window_tickets = tickets
        if len(tickets) >= _MAX_WINDOW_TICKETS:
            raise HTTPException(429, "Too many pending Webapp windows; try again shortly")
        ticket = secrets.token_urlsafe(32)
        tickets[ticket] = (now + ttl, browser_credential.generation(), launch)
    return ticket


def mint_launch_ticket(state) -> str:
    """Ticket for the browser the server opens itself: a launcher's argv carries this, not the token."""
    return _mint_handoff_ticket(state, LAUNCH_TICKET_TTL_SECONDS, launch=True)


def mint_printed_launch_ticket(state) -> str:
    """Ticket for a launch link the user opens by hand where nothing identifies the redeeming
    connection's owner (macOS, the BSDs, WSL): it spends without that check, as a window ticket does, so
    ``hermes webapp`` prints it and never passes it to a browser launcher."""
    return _mint_handoff_ticket(state, LAUNCH_TICKET_TTL_SECONDS, launch=False)


def _loopback_peer_uid(client: tuple[str, int], server: tuple[str, int]) -> int | None:
    """Owner UID of the client end of a local TCP connection, from Linux ``/proc/net/tcp*``.

    Both ends of a loopback connection are listed with their socket owner; the client's is
    the row whose local address is *client* and remote address is *server*. None when absent
    (another network namespace, a closed connection) or unreadable.
    """
    def _normal(host: str | bytes, port: int):
        address = ipaddress.ip_address(host)
        return getattr(address, "ipv4_mapped", None) or address, port

    def _row_address(field: str):
        host_hex, port_hex = field.split(":")
        raw = bytes.fromhex(host_hex)
        # The kernel prints each 32-bit word of the address in host byte order.
        packed = b"".join(int.from_bytes(raw[i:i + 4], "big").to_bytes(4, sys.byteorder)
                          for i in range(0, len(raw), 4))
        return _normal(packed, int(port_hex, 16))

    try:
        wanted = (_normal(*client), _normal(*server))
    except ValueError:
        return None
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(table, encoding="ascii") as stream:
                rows = stream.read().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            fields = row.split()
            try:
                if (_row_address(fields[1]), _row_address(fields[2])) == wanted:
                    return int(fields[7])
            except (IndexError, ValueError):
                continue
    return None


def _peer_is_server_user(request: Request, platform: str = sys.platform) -> bool:
    """Whether a launch ticket that crossed a browser launcher's argv may be spent over this
    request's connection on *platform* (a ``sys.platform`` value, this host's by default).

    Linux checks that the connection belongs to the OS user running the server. Windows checks
    nothing and always says yes: the protection there is that the ticket travels only in the
    launcher's command line, which only its own user, SYSTEM and elevated administrators can read.
    Elsewhere (macOS, the BSDs) every local user can read the launcher's argv and nothing here
    identifies the connection's owner, so an argv-carried launch ticket cannot be redeemed; WSL
    never mints one. Authenticated child-window tickets and the launch tickets ``hermes webapp``
    prints for the user to open by hand (``mint_printed_launch_ticket``) do not use this launcher
    channel.
    """
    if platform == "win32":
        return True
    if platform != "linux":
        return False
    client, server = request.client, request.scope.get("server")
    if client is None or not server:
        return False
    return _loopback_peer_uid((client.host, client.port), (server[0], server[1])) == os.getuid()  # windows-footgun: ok — Linux-only branch


@router.post("/api/webapp/window-ticket")
async def issue_window_ticket(request: Request):
    from hermes_cli.web_server import _require_token

    _require_private_webapp(request)
    _require_token(request)
    ticket = _mint_handoff_ticket(request.app.state, WINDOW_TICKET_TTL_SECONDS, launch=False)
    return JSONResponse({"ticket": ticket}, headers={"Cache-Control": "no-store"})


def _ticket_principal(request: Request) -> tuple[dict, str | None]:
    """``({user_id, provider, issuer}, session_token)`` of the caller: its verified login (no
    token), or on an unauthenticated loopback bind ``SESSION_TOKEN_OWNER`` and the session token
    it presented."""
    session = getattr(request.state, "session", None)
    if session is not None:
        return {"user_id": session.user_id, "provider": session.provider,
                "issuer": getattr(session, "issuer", "") or ""}, None
    if not getattr(request.app.state, "auth_required", False):
        from hermes_cli.web_server import _SESSION_HEADER_NAME, _has_valid_session_token
        presented = (request.headers.get(_SESSION_HEADER_NAME)
                     or request.headers.get("authorization", "").removeprefix("Bearer "))
        if _has_valid_session_token(request):
            return dict(SESSION_TOKEN_OWNER), presented
    raise HTTPException(status_code=401, detail="Unauthorized")


def mint_socket_ticket(identity: dict, socket: str, *, session_token: str | None = None,
                       sign_in: str | None = None, profile_home: str | None = None) -> str:
    """A one-use ticket that opens only *socket* (``web_session_sockets.GATEWAY_SOCKET``,
    ``HOST_TERMINAL_SOCKET`` or one of ``AUDIO_SOCKETS``) as *identity*. Minted with
    *session_token*, or by a login whose sign-in binding was checked against the browser credential
    of generation *sign_in* (``hermes_cli.webapp_sign_in``), it records that credential's generation
    (``hermes_cli.browser_credential``): it opens nothing once that credential is replaced, and the
    socket it opens closes when the browser signs out. With *profile_home* the socket it opens answers
    for that profile (``chat_ws.gateway_ws``)."""
    from hermes_cli.browser_credential import generation_of
    from hermes_cli.dashboard_auth.ws_tickets import mint_ticket

    extra = {"socket": socket}
    if session_token is not None:
        extra["session_token"] = generation_of(session_token)
    if sign_in is not None:
        extra["sign_in"] = sign_in
    if profile_home is not None:
        extra["profile_home"] = profile_home
    return mint_ticket(user_id=identity["user_id"], provider=identity["provider"],
                       issuer=identity.get("issuer", ""), extra=extra)


def _socket_ticket_response(request: Request, identity: dict, socket: str, session_token: str | None,
                            profile_home: str | None = None) -> JSONResponse:
    from hermes_cli.dashboard_auth.ws_tickets import TTL_SECONDS
    from hermes_cli.webapp_sign_in import sign_in_of

    ticket = mint_socket_ticket(identity, socket, session_token=session_token, sign_in=sign_in_of(request),
                                profile_home=profile_home)
    return JSONResponse({"ticket": ticket, "ttl_seconds": TTL_SECONDS}, headers={"Cache-Control": "no-store"})


#: Every socket a Webapp page dials with a one-use ticket from ``POST /api/webapp/socket-ticket``, by
#: its ``web_session_sockets`` name: the ``web_server_surface.SurfacePolicy`` field that lets a surface
#: mint it, and whether the mint checks the profile the page will dial (400/404), so a stale tab learns
#: its profile is gone there rather than from a socket that keeps closing.
SOCKET_TICKETS: dict[str, tuple[str, bool]] = {
    GATEWAY_SOCKET: ("gateway_ticket", True),
    HOST_TERMINAL_SOCKET: ("host_terminal", False),
    SPEAK_STREAM_SOCKET: ("gateway_ticket", False),
    TRANSCRIBE_STREAM_SOCKET: ("gateway_ticket", False),
}


@router.post("/api/webapp/socket-ticket")
async def issue_socket_ticket(request: Request, socket: str = "", profile: str | None = None):
    """The browser's one-use ticket for one dial of *socket* on this same origin: ``gateway`` for
    ``/api/ws`` (also offered as the ``hermes-gateway-ticket.`` subprotocol), ``host-terminal`` for
    ``/api/host-terminal``, ``speak-stream`` and ``transcribe-stream`` for ``/api/audio/<name>``. The
    upgrade gate (``web_server_chat._consume_ws_ticket``) redeems it as this caller's login; it opens
    that socket and nothing else, so the browser mints one per dial. Minted locally: the server that
    serves this page runs the socket's sessions. A ``gateway`` ticket minted with *profile* names
    it: that socket answers a request that names no profile for *profile*, as a native ticket does
    (``chat_ws.gateway_ws``), never for the gateway's launch profile."""
    from hermes_cli.web_server_cron import _cron_profile_home

    if socket not in SOCKET_TICKETS:
        raise HTTPException(400, "Unknown socket")
    surface, checks_profile = SOCKET_TICKETS[socket]
    if not getattr(policy(request.app.state), surface):
        raise HTTPException(404, f"Webapp {socket} tickets are unavailable on this surface")
    identity, session_token = _ticket_principal(request)
    profile_home = None
    if checks_profile:
        _name, home = _cron_profile_home(profile)
        if socket == GATEWAY_SOCKET and profile:
            profile_home = str(home)
    return _socket_ticket_response(request, identity, socket, session_token, profile_home=profile_home)


@router.post("/webapp/window-session")
async def redeem_window_ticket(request: Request):
    from hermes_cli import browser_credential

    _require_private_webapp(request)
    # This endpoint deliberately sits outside /api: the child window or launch
    # tab has no session yet. Only a purpose-specific ticket can exchange for
    # the existing one.
    # Private links use the actual bind origin, which may differ from a
    # configured loopback public_url. Host validation remains in middleware;
    # proxy-normalized ASGI scheme/Host (not raw forwarded headers) are trusted.
    origins = request.headers.getlist("origin")
    origin = _http_origin(origins[0]) if len(origins) == 1 else None
    target_origin = _http_origin(f"{request.url.scheme}://{request.url.netloc}")
    if origin is None or origin != target_origin:
        raise HTTPException(403, "Webapp window handoff requires a same-origin request")
    # Keep the unauthenticated exchange body-free; never parse an arbitrary
    # JSON/multipart payload just to extract a fixed-size capability.
    ticket = request.headers.get("X-Hermes-Window-Ticket", "")
    with _ticket_lock:
        tickets = getattr(request.app.state, "webapp_window_tickets", {})
        entry = tickets.get(ticket)
        # A launch ticket crossed a browser launcher's argv: only a connection owned by this
        # server's user may spend it, and a refused attempt must not burn it for the real browser.
        if entry is not None and entry[2] and not _peer_is_server_user(request):
            entry = None
        elif entry is not None:
            del tickets[ticket]
    token = browser_credential.current_if(entry[1]) if entry is not None and entry[0] > time.monotonic() else None
    if token is None:
        raise HTTPException(403, "Webapp window link expired or was already used; open a new window")
    return JSONResponse({"token": token}, headers={"Cache-Control": "no-store"})


# The credential goes to this tab's sessionStorage, where the Webapp reads it at boot
# (browser-launch-session.ts), never into a URL: the browser keeps every address it opens in its history.
_LAUNCH_SCRIPT = r"""
const basePath = __BASE_PATH__;
const params = new URLSearchParams(location.hash.slice(1));
history.replaceState(null, '', location.pathname);
const ticket = params.get('ticket') || '';
const show = message => { document.getElementById('status').textContent = message; };
const fail = () => show('This launch link was already used or has expired. Run hermes webapp again for a new one.');
if (!/^[A-Za-z0-9_-]{43}$/.test(ticket)) {
  fail();
} else {
  fetch(`${basePath}/webapp/window-session`, {
    method: 'POST', credentials: 'same-origin', headers: {'X-Hermes-Window-Ticket': ticket}
  }).then(response => response.ok ? response.json() : Promise.reject()).then(({token}) => {
    if (typeof token !== 'string' || !/^[A-Za-z0-9_-]{43}$/.test(token)) throw new Error();
    try {
      sessionStorage.setItem(`hermes.webapp.session.v1:${JSON.stringify([location.origin, basePath])}`, token);
    } catch {
      show('This browser does not let the Webapp keep its sign-in in this tab. Allow site data for this '
        + 'address, then run hermes webapp again.');
      return;
    }
    const target = new URL(`${basePath}/`, location.origin);
    target.search = params.get('query') || '';
    location.replace(target.href);
  }).catch(fail);
}
"""


def _handoff_page(request: Request, title: str, status: str, script: str) -> HTMLResponse:
    """Credential-free, uncached page whose only script is *script* (``__BASE_PATH__`` bound)."""
    from hermes_cli.web_server_dashboard import _normalise_prefix

    prefix = _normalise_prefix(request.headers.get("x-forwarded-prefix"))
    nonce = secrets.token_urlsafe(16)
    script = script.replace("__BASE_PATH__", json.dumps(prefix).replace("</", "<\\/"))
    return HTMLResponse(
        '<!doctype html><meta charset="utf-8"><meta name="referrer" content="no-referrer">'
        f'<title>{title}</title><p id="status">{status}</p>'
        f'<script nonce="{nonce}">{script}</script>',
        headers={
            "Cache-Control": "no-store",
            "Referrer-Policy": "no-referrer",
            "Content-Security-Policy": f"default-src 'none'; script-src 'nonce-{nonce}'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
        },
    )


@router.get("/webapp/launch")
async def launch_page(request: Request):
    """Where the server's own browser-open lands: spends the one-use launch ticket from the
    fragment (never sent to the server), keeps the credential it receives in this tab's
    sessionStorage and enters the Webapp at an address that carries neither."""
    _require_private_webapp(request)
    return _handoff_page(request, "Opening Hermes", "Opening Hermes Webapp…", _LAUNCH_SCRIPT)


# No credential appears in this public page or in its URL. The fragment
# names a transient channel; the authorized parent mints a one-use ticket.
_WINDOW_SCRIPT = r"""
const basePath = __BASE_PATH__;
const params = new URLSearchParams(location.hash.slice(1));
history.replaceState(null, '', location.pathname);
const id = params.get('channel') || '';
const status = document.getElementById('status');
if (!/^[a-f0-9-]{36}$/.test(id)) {
  status.textContent = 'Open this window from an authorized Hermes Webapp tab.';
} else {
  const channel = new BroadcastChannel(`hermes.webapp.window:${id}`);
  let receiving = false;
  let finished = false;
  const controller = new AbortController();
  const fail = message => {
    if (finished) return;
    finished = true;
    controller.abort();
    status.textContent = message;
    channel.postMessage({type: 'error'});
    channel.close();
    clearTimeout(timeout);
  };
  const timeout = setTimeout(() => fail('This window link expired. Open a new window from the original tab.'), 30000);
  channel.onmessage = async event => {
    if (receiving) return;
    if (event.data?.error) { fail(event.data.error); return; }
    const ticket = event.data?.ticket;
    if (typeof ticket !== 'string' || !/^[A-Za-z0-9_-]{43}$/.test(ticket)) return;
    receiving = true;
    try {
      const response = await fetch(`${basePath}/webapp/window-session`, {
        method: 'POST', credentials: 'same-origin',
        headers: {'X-Hermes-Window-Ticket': ticket}, signal: controller.signal
      });
      if (!response.ok) throw new Error('Window authorization expired. Open a new window from the original tab.');
      const {token} = await response.json();
      if (finished) return;
      if (typeof token !== 'string' || !/^[A-Za-z0-9_-]{43}$/.test(token)) throw new Error('Invalid window authorization.');
      sessionStorage.setItem(`hermes.webapp.session.v1:${JSON.stringify([location.origin, basePath])}`, token);
      const target = new URL(`${basePath}/`, location.origin);
      target.search = params.get('query') || '';
      target.hash = params.get('route') || '/';
      channel.postMessage({type: 'done'});
      finished = true;
      clearTimeout(timeout);
      channel.close();
      location.replace(target.href);
    } catch (error) { fail(error.message || 'Could not authorize this window. Open it again from the original tab.'); }
  };
  channel.postMessage({type: 'ready'});
}
"""


@router.get("/webapp/window")
async def child_window_page(request: Request):
    _require_private_webapp(request)
    return _handoff_page(request, "Opening Hermes", "Authorizing this Hermes window…", _WINDOW_SCRIPT)
