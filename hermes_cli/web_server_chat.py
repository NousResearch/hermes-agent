"""Chat/terminal WebSocket plumbing: PTY bridge selection and registry, WS
client/origin/auth gates, chat argv resolution, gateway/sidecar URL building.
"""

import logging
import asyncio
import atexit
import concurrent.futures
import contextlib
import functools
import hmac
import os
import re
import sys
import tempfile
import threading
import urllib.request
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from pathlib import Path
from typing import Optional
from hermes_cli.pty_session import PtySessionRegistry

# Same logger the code used before extraction (record parity).
_log = logging.getLogger("hermes_cli.web_server")


# /api/pty spawns ``hermes --tui`` behind a pseudo-terminal and forwards bytes +
# resize escapes to xterm.js.  POSIX uses pty_bridge (fcntl/termios); native
# Windows uses win_pty_bridge (pywinpty/ConPTY); same surface, no handler guards.
try:
    if sys.platform.startswith("win"):
        from hermes_cli.win_pty_bridge import WinPtyBridge as PtyBridge, PtyUnavailableError
    else:
        from hermes_cli.pty_bridge import PtyBridge, PtyUnavailableError
    _PTY_BRIDGE_AVAILABLE = True
except ImportError:  # pragma: no cover - pywinpty / ptyprocess missing
    PtyBridge = None  # type: ignore[assignment]
    _PTY_BRIDGE_AVAILABLE = False

    class PtyUnavailableError(RuntimeError):  # type: ignore[no-redef]
        """Stub when the platform PTY bridge cannot be imported."""
_RESIZE_RE = re.compile(rb"\x1b\[RESIZE:(\d+);(\d+)\]")
_PTY_READ_CHUNK_TIMEOUT = 0.2

# Back-off between idle PTY reads so a quiet terminal does not spin the event
# loop (keeps dashboard idle CPU low).
# A positive sleep lets other coroutines run and keeps dashboard idle CPU low (#42627).
_PTY_IDLE_BACKOFF = 0.05
PTY_REGISTRY = PtySessionRegistry(
    ttl=30 * 60, max_sessions=16, buffer_cap=1 * 1024 * 1024, read_timeout=_PTY_READ_CHUNK_TIMEOUT)


async def _close_stalled_pty_input(ws: WebSocket, *, path: str) -> None:
    """Close only the terminal socket when its child stops accepting input."""
    _log.warning("pty input stalled path=%s; recycling terminal session", path)
    try:
        await ws.close(code=1013, reason="PTY input stalled")
    except Exception:
        pass


async def _legacy_pump(ws: WebSocket, bridge) -> None:
    """Original 1:1 socket<->PTY pump: stream until disconnect, then close the
    bridge. Used when no ``?attach=`` token is supplied (keep-alive opt-in).

    Behavior is identical to the pre-keep-alive ``pty_ws`` body, including the 54028 half-open-socket
    protection (reader EOF → close the WS so the writer's ``ws.receive()`` unparks) and the #53227
    ``to_thread`` offloads for the blocking ``bridge.close()``.
    """
    loop = asyncio.get_running_loop()

    async def pump_pty_to_ws() -> None:
        try:
            while True:
                chunk = await loop.run_in_executor(None, bridge.read, _PTY_READ_CHUNK_TIMEOUT)
                if chunk is None:  # EOF
                    return
                if not chunk:  # no data this tick; yield control and retry
                    await asyncio.sleep(_PTY_IDLE_BACKOFF)
                    continue
                try:
                    await ws.send_bytes(chunk)
                except Exception:
                    return
        finally:
            # Close the WS so the writer's ``ws.receive()`` returns instead of
            # blocking forever on a half-open browser socket (fds would leak and
            # auto-reconnect stacks a fresh PTY on each orphan).  Reap the bridge
            # here too (idempotent): cancelling the handler the instant the WS
            # closes can skip the writer's ``finally``.
            with contextlib.suppress(Exception):
                # The child has exited (EOF) or the send side broke. Closing from the EOF path makes the
                # reap independent of that cancellation race (#54028).
                await asyncio.to_thread(bridge.close)
            with contextlib.suppress(Exception):
                await ws.close()

    reader_task = asyncio.create_task(pump_pty_to_ws())

    try:
        while True:
            try:
                msg = await ws.receive()
            except RuntimeError:
                # ws.receive() after the socket is already disconnected
                # (e.g. closed by the reader task above).
                break
            if msg.get("type") == "websocket.disconnect":
                break
            raw = msg.get("bytes")
            if raw is None:
                text = msg.get("text")
                raw = text.encode("utf-8") if isinstance(text, str) else b""
            if not raw:
                continue
            # Resize escape is consumed locally, never written to the PTY.
            match = _RESIZE_RE.match(raw)
            if match and match.end() == len(raw):
                bridge.resize(cols=int(match.group(1)), rows=int(match.group(2)))
                continue
            if not await bridge.write(raw):
                await _close_stalled_pty_input(ws, path="legacy")
                break
    except WebSocketDisconnect:
        pass
    finally:
        reader_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await reader_task
        await asyncio.to_thread(bridge.close)


from hermes_cli.web_server_boundary import _LOOPBACK_HOSTS


def _ws_client_reason(ws: WebSocket) -> Optional[str]:
    """Return a rejection reason token for the peer IP, or None when allowed
    (:func:`hermes_cli.web_server_boundary.ws_client_reason` on the dashboard app state)."""
    from hermes_cli.web_server import app
    from hermes_cli.web_server_boundary import ws_client_reason
    return ws_client_reason(ws, app.state)


def _ws_client_is_allowed(ws: WebSocket) -> bool:
    """True when the peer IP passes :func:`_ws_client_reason`."""
    return _ws_client_reason(ws) is None


def _ws_host_origin_reason(ws: WebSocket) -> Optional[str]:
    """Return ``host_mismatch …`` / ``origin_mismatch …``, or None when allowed
    (:func:`hermes_cli.web_server_boundary.ws_host_origin_reason` on the dashboard app state)."""
    from hermes_cli.web_server import app
    from hermes_cli.web_server_boundary import ws_host_origin_reason
    return ws_host_origin_reason(ws, app.state)


def _ws_host_origin_is_allowed(ws: WebSocket) -> bool:
    """True when the upgrade passes the dashboard Host/Origin guard."""
    return _ws_host_origin_reason(ws) is None


def _ws_request_is_allowed(ws: WebSocket) -> bool:
    """Return True when the WebSocket upgrade matches dashboard boundaries."""
    return _ws_host_origin_is_allowed(ws) and _ws_client_is_allowed(ws)


from hermes_cli.web_server_boundary import (
    _GATEWAY_WS_PROTOCOL,
    _GATEWAY_WS_TICKET_PROTOCOL_PREFIX,
    _gateway_ws_ticket_from_subprotocol,
)


def _ws_request_view(ws: WebSocket, app) -> Request:
    """A ``Request`` facade over a ``WebSocket`` for the auth helpers.

    ``_verify_access_token`` only touches ``request.headers`` (X-Forwarded-For
    via ``client_ip``) and ``request.client`` on this path (``audit=False``,
    no cookie/redirect work), and ``webapp_sign_in.token_works_alone`` reads
    ``request.app.state``, so a lightweight stand-in carrying *app* avoids
    constructing a real ASGI request inside the upgrade. ``scan_session_providers``
    itself never sees the request — only the provider callbacks do.
    ``webapp_sign_in.token_works_alone`` reads the listener's state from *app*.
    """
    from fastapi import Request

    return Request({
        "type": "http",
        "app": app,
        "headers": [(k.lower().encode("latin-1"), v.encode("latin-1"))
                    for k, v in getattr(ws.headers, "items", dict)()],
        "client": (ws.client.host, 0) if ws.client else None,
        "server": None,
        "scheme": "ws",
        "method": "GET",
        "path": ws.url.path,
        "query_string": b"",
        "root_path": "",
    })


def _ws_auth_reason(ws: WebSocket, *, allow_internal: bool = False,
                    socket: Optional[str] = None) -> tuple[Optional[str], str]:
    """Validate WS-upgrade auth; return ``(reason, credential)``.

    ``reason`` is None when accepted, else a short token (``no_credential``,
    ``token_mismatch``, ``ticket_invalid``, ``internal_invalid``);
    ``credential`` names what was presented so the accept path can log *how*.
    *socket* names the route's socket for its own tickets (``web_session_sockets.GATEWAY_SOCKET``,
    ``HOST_TERMINAL_SOCKET`` or one of ``AUDIO_SOCKETS``): a ticket minted for
    one socket opens no other, and the host terminal opens only with its own ticket.

    Loopback / ``--insecure``: legacy ``?token=`` (constant-time compared); on the
    gateway's Webapp a one-use socket ticket first (``/api/webapp/socket-ticket``).
    Gated: ``?ticket=`` (browser-minted, single-use, 30s TTL) or ``?internal=``
    (process-lifetime, multi-use, only for server-spawned WS clients so the PTY
    child can reconnect; never injected into the SPA; on the gateway's Webapp it
    opens only the sockets that pass *allow_internal*), or ``?token=`` holding a
    user session access token (verified against the dashboard auth session
    providers — the same ``verify_session`` seam the native bearer REST leg
    uses — e.g. a token-mode Remote desktop connection; on the gateway's Webapp
    only a token that listener gave a native app's sign-in, as for the bearer
    header, ``webapp_sign_in.token_works_alone``). The legacy in-process
    ``_SESSION_TOKEN`` is rejected in gated mode: a leaked ``_SESSION_TOKEN``
    must not grant access.

    Either mode: a local Desktop (no static token) presents the private one-use
    ``X-Hermes-Gateway-Ticket`` grant its REST calls use (``authenticate_native_ws``).
    A socket that sends that header is judged by it alone: on a core socket
    (``/api/ws``, ``/api/pty``, the audio streams) the grant is refused
    (``native_invalid``) even when a valid ``token`` query credential rides along.
    That is intended; only the plugin WebSocket gate accepts the grant.
    """
    from hermes_cli.dashboard_auth.native_http import HEADER, authenticate_native_ws
    from hermes_cli.web_server import _SESSION_TOKEN, app
    from hermes_cli.browser_credential import generation_of
    from hermes_cli.web_session_sockets import OWN_TICKET_SOCKETS, track
    auth_required = bool(getattr(app.state, "auth_required", False))
    # The Webapp's own rules (socket tickets on the loopback gate, the server's credential only where
    # its spawned clients dial) apply on its listener; the standalone dashboard and serve keep main's.
    webapp = getattr(app.state, "ui_surface", None) == "webapp"
    if socket in OWN_TICKET_SOCKETS:
        # No standing credential (the session token, a native grant, a login's access token) opens it.
        reject = functools.partial(_audit_ws_rejection, ws) if auth_required else _ignore_rejection
        return _consume_ws_ticket(ws, reject, socket, auth_required=auth_required) or ("no_credential", "none")
    if HEADER in ws.headers:
        grant = authenticate_native_ws(ws)
        if grant is None:
            return "native_invalid", "native"
        ws._hermes_auth_identity = {"user_id": grant["subject"], "provider": "local"}
        return None, "native"
    if auth_required:
        # Lazy import — keeps this function importable in test harnesses
        # that don't bring in the dashboard_auth layer.
        from hermes_cli.dashboard_auth.audit import AuditEvent, audit_log
        from hermes_cli.dashboard_auth.ws_tickets import TicketInvalid, consume_internal_credential

        _reject = functools.partial(_audit_ws_rejection, ws)

        def _stamp_identity(info) -> None:
            # Server-minted {user_id, provider} stamped onto the WS object is the
            # sole identity authority downstream (gateway transport / controller
            # registration); a client can never supply it through RPC params.
            # Only identity fields are carried — bookkeeping such as
            # ``minted_at`` is not part of the identity contract.
            ws._hermes_auth_identity = {
                "user_id": info.get("user_id"), "provider": info.get("provider")}
            if info.get("issuer"):
                ws._hermes_auth_identity["issuer"] = info["issuer"]

        internal = ws.query_params.get("internal", "")
        if internal:
            if webapp and not allow_internal:
                _reject("internal: endpoint not allowed")
                return "internal_not_allowed", "internal"
            try:
                _stamp_identity(consume_internal_credential(internal))
                return None, "internal"
            except TicketInvalid as exc:
                _reject(f"internal: {exc}")
                return "internal_invalid", "internal"

        consumed = _consume_ws_ticket(ws, _reject, socket, auth_required=True)
        if consumed is not None:
            return consumed

        # A user session access token (``?token=``) is verified against the
        # dashboard auth session providers — the SAME ``verify_session`` seam
        # the native bearer REST leg uses (``_verify_access_token``). A
        # token-mode Remote desktop connection bakes its stored session token
        # into the WS URL (buildGatewayWsUrl), so before this leg existed the
        # only credentials gated mode accepted were browser-minted tickets and
        # the WS upgrade 403'd forever. The legacy in-process
        # ``_SESSION_TOKEN`` never reaches this check: gated mode falls
        # through to the explicit rejection below (leaked-constant safety),
        # and a provider-verified token is accepted on identity merit. On the
        # gateway's Webapp it follows the bearer header's rule: only a token this
        # listener gave a native app's sign-in works on its own, so a browser
        # login's tokens work only as its cookies, which sign-out ends.
        token = ws.query_params.get("token", "")
        if token:
            from hermes_cli import webapp_sign_in
            from hermes_cli.dashboard_auth.base import ProviderError
            from hermes_cli.dashboard_auth.middleware import _verify_access_token

            view = _ws_request_view(ws, app)
            try:
                # On the gateway's Webapp a browser login's tokens work only as its cookies, which
                # sign-out ends; only a token this listener gave a native app's sign-in works alone.
                session = (_verify_access_token(view, access_token=token, audit=False)
                           if webapp_sign_in.token_works_alone(view, token) else None)
            except ProviderError as exc:
                # All providers unreachable (IDP outage) — reject without
                # crashing the upgrade; the REST bearer leg answers 503, and
                # the WS accept path has no status channel, only close codes.
                _reject(f"session token verify unavailable: {exc}")
                return "token_unavailable", "token"
            if session is not None:
                _stamp_identity({
                    "user_id": getattr(session, "user_id", None),
                    "provider": getattr(session, "provider", None),
                })
                audit_log(
                    AuditEvent.TOKEN_AUTH_SUCCESS,
                    provider=getattr(session, "provider", None),
                    user_id=getattr(session, "user_id", None),
                    ip=(ws.client.host if ws.client else ""),
                    path=ws.url.path,
                )
                return None, "token"
            audit_log(
                AuditEvent.TOKEN_AUTH_FAILURE,
                reason="session_token_invalid",
                ip=(ws.client.host if ws.client else ""),
                path=ws.url.path,
            )
            return "token_invalid", "token"

        return "no_credential", "none"

    # Loopback: on the Webapp a one-use ticket ``/api/webapp/socket-ticket`` minted for the
    # session-token owner (its sockets on this same origin), else the legacy ``?token=``.
    consumed = _consume_ws_ticket(ws, _ignore_rejection, socket) if webapp else None
    if consumed is not None:
        return consumed
    token = ws.query_params.get("token", "")
    if not token:
        return "no_credential", "none"
    if hmac.compare_digest(token.encode(), _SESSION_TOKEN.encode()):
        track(ws, generation_of(token))
        return None, "token"
    return "token_mismatch", "token"


def _ignore_rejection(reason: str) -> None:
    """The loopback gate keeps no audit record of a refused credential."""


def _audit_ws_rejection(ws: WebSocket, reason: str) -> None:
    """The gated gate's audit record of a refused upgrade credential."""
    from hermes_cli.dashboard_auth.audit import AuditEvent, audit_log
    audit_log(AuditEvent.WS_TICKET_REJECTED, reason=reason, ip=(ws.client.host if ws.client else ""),
              path=ws.url.path)


def _consume_ws_ticket(ws: WebSocket, reject, socket: Optional[str] = None, *,
                       auth_required: bool = False) -> Optional[tuple[Optional[str], str]]:
    """A one-use ticket offered as the ``hermes-gateway-ticket.`` subprotocol or ``?ticket=``:
    ``(reason, credential)`` once judged, or None when the upgrade presents none. An accepted ticket
    stamps the server-minted identity onto the socket; the client never supplies it through RPC
    params. *socket* is the route's socket (``web_session_sockets.ticket_refusal``); a socket a
    session-token ticket, or a Webapp login's ticket, opens is tracked so signing out closes it."""
    from hermes_cli.dashboard_auth.ws_tickets import TicketInvalid, consume_ticket
    from hermes_cli.web_session_sockets import admitted_with, ticket_refusal, track
    protocol_ticket, protocol_reason = _gateway_ws_ticket_from_subprotocol(ws)
    if protocol_reason == "invalid":
        return "ticket_invalid", "ticket-subprotocol"
    ticket = protocol_ticket or ws.query_params.get("ticket", "")
    if not ticket:
        return None
    try:
        info = consume_ticket(ticket)
    except TicketInvalid as exc:
        reject(str(exc))
        return "ticket_invalid", "ticket"
    refusal = ticket_refusal(info, socket, auth_required=auth_required)
    if refusal is not None:
        reject(refusal)
        return "ticket_invalid", "ticket"
    if info.get("provider") == "bot-desktop":
        # A display ticket admits one RFB bridge on /api/display/ws (a watch-only capability handed
        # to a screen viewer); it must not double as a login here.
        reject("display ticket presented as a gateway login")
        return "ticket_invalid", "ticket"
    # Only identity fields are carried; bookkeeping such as ``minted_at`` is not part of it.
    ws._hermes_auth_identity = {"user_id": info.get("user_id"), "provider": info.get("provider")}
    if info.get("issuer"):
        ws._hermes_auth_identity["issuer"] = info["issuer"]
    if info.get("profile_home"):
        ws._hermes_ticket_profile_home = info["profile_home"]
    if (admitted := admitted_with(info)) is not None:
        track(ws, admitted)
    if not protocol_ticket:
        return None, "ticket"
    # Select only the stable public protocol during accept. The ticket-bearing protocol is a
    # credential and must never be reflected back to the browser or retained after admission.
    ws._hermes_ws_subprotocol = _GATEWAY_WS_PROTOCOL
    return None, "ticket-subprotocol"


def _ws_auth_ok(ws: WebSocket, *, allow_internal: bool = False, socket: Optional[str] = None) -> bool:
    """True when the WS-upgrade credential is accepted. See _ws_auth_reason."""
    return _ws_auth_reason(ws, allow_internal=allow_internal, socket=socket)[0] is None


def _ws_auth_mode() -> str:
    """Short label for the active WS auth mode — logged on every connection."""
    from hermes_cli.web_server import app
    if getattr(app.state, "auth_required", False):
        return "gated"
    bound_host = (getattr(app.state, "bound_host", "") or "").strip().lower()
    if bound_host and bound_host not in _LOOPBACK_HOSTS:
        return "insecure"
    return "loopback"


def _ws_close_reason(text: str) -> str:
    """Clamp to RFC 6455's 123-byte close-reason limit (uvicorn raises past it);
    reasons embed an attacker-controlled origin, so truncate rather than crash."""
    encoded = text.encode("utf-8", "replace")
    if len(encoded) <= 123:
        return text
    return encoded[:120].decode("utf-8", "ignore") + "..."


async def _ws_gate(ws: WebSocket, kind: str) -> Optional[tuple[str, str, str]]:
    """Run the pre-accept gates for /api/console, /api/pty and /api/host-terminal.

    Each gate maps to a distinct close code so the log and the browser banner
    agree on the cause: 4404 chat disabled, 4401 bad credential, 4403 host/origin
    mismatch, 4408 peer not allowed. Returns ``(peer, mode, cred)``
    once every gate passes, or None after closing the socket.
    """
    from hermes_cli.web_server import _DASHBOARD_EMBEDDED_CHAT_ENABLED
    peer = ws.client.host if ws.client else "?"
    if not _DASHBOARD_EMBEDDED_CHAT_ENABLED:
        _log.info("%s refused: embedded chat disabled peer=%s", kind, peer)
        await ws.close(code=4404, reason="embedded chat disabled")
        return None

    auth_reason, cred = _ws_auth_reason(ws, socket=kind)
    mode = _ws_auth_mode()
    if auth_reason is not None:
        _log.warning("%s auth rejected reason=%s mode=%s cred=%s peer=%s", kind, auth_reason, mode, cred, peer)
        await ws.close(code=4401, reason=_ws_close_reason(f"auth: {auth_reason}"))
        return None

    host_origin_reason = _ws_host_origin_reason(ws)
    if host_origin_reason is not None:
        _log.warning("%s refused: %s peer=%s", kind, host_origin_reason, peer)
        await ws.close(code=4403, reason=_ws_close_reason(host_origin_reason))
        return None

    client_reason = _ws_client_reason(ws)
    if client_reason is not None:
        _log.warning("%s refused: %s", kind, client_reason)
        await ws.close(code=4408, reason=_ws_close_reason(client_reason))
        return None
    return peer, mode, cred


async def _pty_fail(ws: WebSocket, exc: BaseException, *, surface: str = "Chat") -> None:
    """Tell the user why chat could not start, then close 1011 so the SPA renders
    "Start new session". The raw exception goes to the server log only."""
    from hermes_cli.web_routers.chat_ws_errors import chat_start_failure_message
    _log.warning("pty start failed: %s: %s", type(exc).__name__, exc)
    await ws.send_text(f"\r\n\x1b[31m{chat_start_failure_message(exc, surface=surface)}\x1b[0m\r\n")
    await ws.close(code=1011)


def _resolve_chat_argv(
    resume: Optional[str] = None, sidecar_url: Optional[str] = None, profile: Optional[str] = None,
    active_session_file: Optional[str] = None,
    workspace_cwd: Optional[str] = None) -> tuple[list[str], Optional[str], Optional[dict]]:
    """Resolve the argv + cwd + env for the chat PTY (what ``hermes --tui`` runs).

    Tests monkeypatch this with a tiny fake command.  Env contract: resume goes
    through ``HERMES_TUI_RESUME`` (``ui-tui`` does not parse argv), resolved to
    the newest descendant; ``HERMES_TUI_GATEWAY_URL`` attaches to this process's
    in-memory gateway but is SKIPPED for profile-scoped chats (that gateway runs
    under the dashboard's own profile, so a scoped chat attaches to its profile's gateway);
    ``profile`` scopes the ENTIRE chat by pointing ``HERMES_HOME`` at the profile
    dir, the same propagation ``hermes -p <name>`` performs. ``workspace_cwd``
    (an already-validated host directory, ``chat_workspaces.resolve_chat_cwd``)
    is the workspace the user picked for a FRESH chat: it becomes ``HERMES_CWD``
    (where a self-spawned gateway starts) and ``HERMES_TUI_CWD`` (what the TUI
    passes as the explicit ``cwd`` of ``session.create`` when attached to the
    in-memory gateway, whose own cwd is the dashboard's launch dir).
    """
    from hermes_cli.web_server_profiles import _config_profile_scope, _resolve_profile_dir
    from hermes_cli.web_server_sessions import _open_session_db_for_profile, _session_latest_descendant
    from hermes_cli.main import PROJECT_ROOT
    from hermes_cli.main_tui_launch import _apply_tui_python_env, _make_tui_argv

    profile_dir: Optional[Path] = None
    requested = (profile or "").strip()
    if requested and requested.lower() != "current":
        profile_dir = _resolve_profile_dir(requested)

    argv, cwd = _make_tui_argv(PROJECT_ROOT / "ui-tui", tui_dev=False)
    # Secrets kept — the spawned agent needs provider creds.  An explicit profile
    # scope overrides HERMES_HOME before config is bridged into the env.
    from tools.environments.local import build_subprocess_env
    env = build_subprocess_env(scrub_secrets=False, inherit_profile_home=True)
    if profile_dir is not None:
        env["HERMES_HOME"] = str(profile_dir)
    try:
        from hermes_cli.config import (
            apply_terminal_config_to_env, read_raw_config, terminal_config_owned_env_vars)

        if profile_dir is not None:
            # Drop only the terminal keys the launch profile owns before applying
            # the selected profile; operator exports for other keys stay valid.
            raw_launch_terminal = read_raw_config().get("terminal")
            for env_var in terminal_config_owned_env_vars(raw_launch_terminal):
                env.pop(env_var, None)
            with _config_profile_scope(requested):
                apply_terminal_config_to_env(env=env)
        else:
            apply_terminal_config_to_env(env=env)
    except Exception:
        _log.warning("Failed to apply terminal config bridge for dashboard chat", exc_info=True)
    if workspace_cwd:
        env["HERMES_CWD"] = workspace_cwd
        env["HERMES_TUI_CWD"] = workspace_cwd
    _apply_tui_python_env(env)
    env.setdefault("NODE_ENV", "production")
    # Mouse tracking would swallow wheel events the browser needs for
    # transcript scrolling; disable it for the dashboard PTY only.
    env.setdefault("HERMES_TUI_DISABLE_MOUSE", "1")
    env.setdefault("HERMES_TUI_INLINE", "1")
    # chalk in the child picks its color depth from the SERVER env; hosted
    # deploys have no COLORTERM, so hex colors would snap to the 256 palette.
    env.setdefault("COLORTERM", "truecolor")
    env["HERMES_TUI_DASHBOARD"] = "1"

    if resume:
        _resume_db = _open_session_db_for_profile(
            requested if profile_dir is not None else None, read_only=True)
        try:
            latest_resume, _latest_path = _session_latest_descendant(resume, _resume_db)
        finally:
            _resume_db.close()
        if latest_resume:
            resume = latest_resume
        env["HERMES_TUI_RESUME"] = resume

    if sidecar_url:
        env["HERMES_TUI_SIDECAR_URL"] = sidecar_url

    if active_session_file:
        env["HERMES_TUI_ACTIVE_SESSION_FILE"] = active_session_file

    # Without the attach URL, gatewayClient bootstraps through the profile's own
    # gateway (ui-tui/scripts/gateway_bootstrap.py) under the HERMES_HOME set above;
    # it never spawns a standalone `tui_gateway.entry` writer.
    if profile_dir is None and (gateway_ws_url := _build_gateway_ws_url()):
        env["HERMES_TUI_GATEWAY_URL"] = gateway_ws_url

    return list(argv), str(cwd) if cwd else None, env


# Wildcard bind hosts an in-container client must NOT dial: behind a forward
# proxy (HTTPS_PROXY without 0.0.0.0 in NO_PROXY) the handshake gets MITM'd.
_WILDCARD_HOSTS = frozenset({"0.0.0.0", "::"})


def _resolve_client_ws_host() -> Optional[str]:
    """Host the in-container WS client should dial: ``HERMES_DASHBOARD_WS_HOST``
    wins always; a wildcard bind becomes ``127.0.0.1``; others verbatim."""
    from hermes_cli.web_server import app
    explicit = os.environ.get("HERMES_DASHBOARD_WS_HOST", "").strip()
    if explicit:
        return explicit
    host = getattr(app.state, "bound_host", None)
    if not host:
        return None
    return "127.0.0.1" if host in _WILDCARD_HOSTS else host


def _server_internal_ws_url(path: str, **extra_qs) -> Optional[str]:
    """``ws://<host>:<port><path>?<auth>&<extra>`` for server-spawned WS clients,
    or None when unbound.

    Gated mode uses the process-lifetime internal credential, NOT a single-use
    browser ticket: the child reads the URL once and reuses it on every
    reconnect, and a 30s-TTL ticket can expire before a slow cold boot dials.
    """
    from hermes_cli.web_server import _SESSION_TOKEN, app
    host = _resolve_client_ws_host()
    port = getattr(app.state, "bound_port", None)
    if not host or not port:
        return None
    netloc = f"[{host}]:{port}" if ":" in host and not host.startswith("[") else f"{host}:{port}"
    if getattr(app.state, "auth_required", False):
        from hermes_cli.dashboard_auth.ws_tickets import internal_ws_credential

        auth = {"internal": internal_ws_credential()}
    else:
        auth = {"token": _SESSION_TOKEN}
    return f"ws://{netloc}{path}?{urllib.parse.urlencode({**auth, **extra_qs})}"


def _build_gateway_ws_url() -> Optional[str]:
    """ws:// URL the PTY child attaches to for JSON-RPC gateway traffic."""
    return _server_internal_ws_url("/api/ws")


def _build_sidecar_url(channel: str) -> Optional[str]:
    """ws:// URL the PTY child publishes events to, or None when unbound."""
    return _server_internal_ws_url("/api/pub", channel=channel)


async def _resolve_chat_argv_async(
    resume: Optional[str] = None, sidecar_url: Optional[str] = None, profile: Optional[str] = None,
    active_session_file: Optional[str] = None,
    workspace_cwd: Optional[str] = None) -> tuple[list[str], Optional[str], Optional[dict]]:
    """Resolve chat argv off the event loop (it may run ``npm run build``); the
    async lock keeps one-build-at-a-time without parking worker threads."""
    from hermes_cli.web_server import _get_chat_argv_lock, app
    kwargs = {"resume": resume, "sidecar_url": sidecar_url, "profile": profile}
    if active_session_file is not None:
        kwargs["active_session_file"] = active_session_file
    if workspace_cwd is not None:
        kwargs["workspace_cwd"] = workspace_cwd

    async with _get_chat_argv_lock(app):
        return await asyncio.to_thread(_resolve_chat_argv, **kwargs)


def _active_session_file_for_channel(app: FastAPI, channel: str) -> Path:
    """Return the per-channel file where a dashboard TUI writes its active sid."""
    from hermes_cli.web_server import _get_pty_active_session_files
    files = _get_pty_active_session_files(app)
    if files.get(channel) is None:
        fd, raw_path = tempfile.mkstemp(prefix="hermes-pty-active-", suffix=".json")
        os.close(fd)
        files[channel] = Path(raw_path)
    return files[channel]


# On timeout asyncio cancels the awaitable but the console thread keeps running;
# a small dedicated pool caps the leak instead of exhausting the default pool.
_CONSOLE_EXECUTOR_MAX_WORKERS = 4
_console_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
_console_executor_lock = threading.Lock()


def _get_console_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Lazily create the bounded console worker pool (once per process)."""
    global _console_executor
    if _console_executor is None:
        with _console_executor_lock:
            if _console_executor is None:
                _console_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=_CONSOLE_EXECUTOR_MAX_WORKERS, thread_name_prefix="hermes-console")
                # Tear down on interpreter exit without waiting on in-flight
                # workers: a stuck 60s console command must not block shutdown.
                atexit.register(
                    lambda: _console_executor
                    and _console_executor.shutdown(wait=False, cancel_futures=True))
    return _console_executor
