"""The gateway's local HTTP/WS listener: the native session WebSocket plus the dashboard API.

The caller initializes session authority before starting this listener and owns
all runtime services and signals. This module owns only HTTP resources; it does
not bootstrap an agent, scheduler, hosted room, or a second gateway.

The native session WebSocket (ticketed ``/api/ws`` — every local CLI/TUI/Desktop session) is served
without the dashboard app: importing ``hermes_cli.web_server`` (about 690 routes, ~0.6 s) before the
listener bound stalled every cold start and the control socket's identify. The dashboard is mounted
on its first request (imported off the event loop) or warmed after READY for a supervised gateway
(``warm_gateway_dashboard``); a gated listener (non-loopback bind, ``webapp.public_url`` on the Webapp's,
``dashboard.public_url`` on the private one) and a
process that already imported it mount it before the listener starts, exactly as before.
"""

import asyncio
from dataclasses import dataclass
import logging
import os
import socket
import sys
import types
from typing import Any

logger = logging.getLogger(__name__)

# One listener per process: the dashboard routers share one process-local app.
_ACTIVE_RUNNER: Any = None
# A fixed port (the Webapp's) has to bind again while the last gateway's closed connections sit in
# TIME_WAIT. POSIX allows that with SO_REUSEADDR; on Windows that option lets another process bind
# the same port, and a rebind there does not wait on TIME_WAIT anyway.
_REUSE_FIXED_PORT = os.name != "nt"
# On Windows a fixed port is held for exclusive use instead: otherwise another process may bind it with
# SO_REUSEADDR, and a more specific address (127.0.0.1 under a 0.0.0.0 bind) then takes its connections.
_EXCLUSIVE_FIXED_PORT = os.name == "nt"


@dataclass
class GatewayAPIHandle:
    api_origin: str
    server: Any
    task: asyncio.Task
    app: Any
    socket: socket.socket


class GatewayDashboardMount:
    """ASGI app that imports and mounts the dashboard (``hermes_cli.web_server.app``) on first use.

    ``state`` is the listener's boundary state (``bound_host``, ``bound_port``, ``auth_required``,
    ``trusted_public_hosts``, ``gateway_runner``, ``session_authority``) — what the native WebSocket
    path checks, mirrored onto the dashboard app state once it is mounted. ``state.webapp`` (a
    ``gateway.webapp_listener.WebappMount``) makes the mount serve the browser-hosted Desktop.
    """

    def __init__(self, runner, state):
        self.runner, self._state = runner, state
        self.web = None
        self._lock: asyncio.Lock | None = None
        self._lifespan = None
        self._closed = False
        self._replaced_session_token = None
        self._undo_own_sign_in = None

    @property
    def loaded(self) -> bool:
        return self.web is not None

    @property
    def state(self):
        """Boundary state: the dashboard app's own state once mounted (operators and tests may
        change it there), the listener's pre-mount snapshot before."""
        return self.web.app.state if self.web is not None else self._state

    def _bind_state(self, web) -> None:
        webapp = getattr(self._state, "webapp", None)
        # The browser credential before any app state: one the home cannot store fails the mount
        # with nothing set, and the gateway falls back to its private listener.
        token = None
        if webapp is not None:
            from gateway.webapp_listener import browser_session_token
            token = browser_session_token(webapp.home)
        # Local clients authenticate with tickets minted on the owner-only control socket. The SPA
        # session token is a bearer credential any loopback peer (any OS user) could read from
        # GET /, so the gateway-hosted listener never publishes it.
        web.app.state.withhold_session_token = True
        own_url = _listener_public_url(webapp)
        if own_url is not None:
            web.app.state.listener_public_url = own_url
        elif hasattr(web.app.state, "listener_public_url"):
            delattr(web.app.state, "listener_public_url")
        if webapp is not None:  # before the gate looks for a sign-in provider
            from gateway.webapp_listener import use_own_sign_in
            self._undo_own_sign_in = use_own_sign_in(webapp)
        web._configure_auth_gate(self._state.bound_host, False, None, None)
        web.app.state.gateway_runner = self.runner
        web.app.state.session_authority = getattr(self.runner, "session_authority", None)
        web.app.state.bound_host = self._state.bound_host
        web.app.state.bound_port = self._state.bound_port
        if webapp is not None:
            # The browser-hosted Desktop: its bundle at /, its routes (launch handoff, socket tickets,
            # host terminal) on, and the browser credential kept in the home so tabs survive a restart.
            # The page itself still never carries the credential (the webapp surface withholds it).
            web.app.state.ui_surface = "webapp"
            web.app.state.web_dist = webapp.dist
            web.app.state.webapp_home = webapp.home
            self._replaced_session_token = web._SESSION_TOKEN
            web._SESSION_TOKEN = token

    def _unbind_state(self) -> None:
        undo_own_sign_in, self._undo_own_sign_in = self._undo_own_sign_in, None
        if undo_own_sign_in is not None:  # a listener mounted next signs in with the configured client
            undo_own_sign_in()
        if self.web is not None:
            self.web.app.state.gateway_runner = None
            self.web.app.state.session_authority = None
            self.web.app.state.withhold_session_token = False
            if hasattr(self.web.app.state, "listener_public_url"):
                delattr(self.web.app.state, "listener_public_url")
            if self._replaced_session_token is not None:
                # Removed, not cleared: a listener mounted next (the private one the gateway falls back
                # to) must find the app as main's listener does, with no surface, bundle or home set.
                for name in ("ui_surface", "web_dist", "webapp_home"):
                    if hasattr(self.web.app.state, name):
                        delattr(self.web.app.state, name)
                self.web._SESSION_TOKEN, self._replaced_session_token = self._replaced_session_token, None

    async def ensure(self):
        """Import (off the loop), bind and start the dashboard app once; returns ``web_server``."""
        if self.web is not None:
            return self.web
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self.web is not None:
                return self.web
            if self._closed:
                raise RuntimeError("gateway API stopped")
            import importlib
            web = sys.modules.get("hermes_cli.web_server") or await asyncio.to_thread(
                importlib.import_module, "hermes_cli.web_server")
            if self._closed:  # the listener stopped while the import ran
                raise RuntimeError("gateway API stopped")
            if getattr(web.app.state, "gateway_runner", None) not in (None, self.runner):
                raise RuntimeError("gateway API already started")
            try:
                # A bind that fails partway (the auth gate refusing a gated bind, or the Webapp's
                # browser credential that cannot be read or written) is undone like a failed lifespan:
                # the listener the gateway falls back to finds the app as main's does.
                self._bind_state(web)
                lifespan = web.app.router.lifespan_context(web.app)
                await lifespan.__aenter__()
                if self._closed:  # close() won while the lifespan started; it never saw this mount
                    await lifespan.__aexit__(None, None, None)
                    raise RuntimeError("gateway API stopped")
            except BaseException:
                self.web = web
                self._unbind_state()
                self.web = None
                raise
            self.web, self._lifespan = web, lifespan
            return web

    async def close(self) -> None:
        self._closed = True
        lifespan, self._lifespan = self._lifespan, None
        try:
            if lifespan is not None:
                await lifespan.__aexit__(None, None, None)
        finally:
            self._unbind_state()

    async def __call__(self, scope, receive, send):
        if scope["type"] not in {"http", "websocket"}:
            return
        try:
            web = await self.ensure()
        except Exception:
            logger.exception("Gateway dashboard API failed to load")
            if scope["type"] == "websocket":
                await send({"type": "websocket.close", "code": 1011})
            else:
                from starlette.responses import JSONResponse
                await JSONResponse({"error": "dashboard_unavailable"}, status_code=500)(scope, receive, send)
            return
        await web.app(scope, receive, send)


def _listener_socket(family, kind, proto, port: int, *, dual_stack: bool = False) -> socket.socket:
    """The listener's unbound socket; a fixed *port* may bind again at once (``_REUSE_FIXED_PORT``)
    or is held exclusively (``_EXCLUSIVE_FIXED_PORT``). A *dual_stack* IPv6 socket (the ``::``
    wildcard) takes IPv4 connections too."""
    listener = socket.socket(family, kind, proto)
    if dual_stack:
        # "::" means every address, as Linux binds it by default. Windows binds IPv6 only otherwise:
        # IPv4 clients would never reach the Webapp, and the exclusive hold below would cover the
        # IPv6 port alone, leaving 0.0.0.0 on the same port to another program.
        listener.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
    if port and _REUSE_FIXED_PORT:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    elif port and _EXCLUSIVE_FIXED_PORT:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    return listener


def _listener_public_url(webapp) -> str | None:
    """The public URL the listener answers to: the Webapp's own ``webapp.public_url`` ("" for none),
    never ``dashboard.public_url``, which describes ``hermes dashboard``. None leaves main's listener
    reading ``dashboard.public_url``."""
    return None if webapp is None else webapp.public_url


async def start_gateway_api(runner, *, host: str = "127.0.0.1", port: int = 0, webapp=None) -> GatewayAPIHandle:
    global _ACTIVE_RUNNER
    from gateway.webapp_listener import dial_origin
    from hermes_cli.web_server_boundary import (
        _dashboard_public_hosts, listener_auth_required, public_hosts_engage_gate, public_url_hosts,
        should_require_auth,
    )
    from hermes_cli.web_server_lifecycle import build_listener_server

    # Existing routers/auth helpers share one process-local app. Never rebind it
    # underneath another listener. Each authority is bound to its own profile DB.
    loaded_web = sys.modules.get("hermes_cli.web_server")
    if _ACTIVE_RUNNER is not None or (
            loaded_web is not None and getattr(loaded_web.app.state, "gateway_runner", None) is not None):
        raise RuntimeError("gateway API already started")
    public_url = _listener_public_url(webapp)
    if public_url is None:
        trusted = _dashboard_public_hosts()
        auth_required = listener_auth_required(host, trusted)
    else:
        # What web_server._configure_auth_gate decides for a listener that declares its public URL.
        trusted = public_url_hosts(public_url)
        auth_required = should_require_auth(host) or public_hosts_engage_gate(trusted)
    state = types.SimpleNamespace(
        bound_host=host, bound_port=None, auth_required=auth_required,
        trusted_public_hosts=trusted, gateway_runner=runner,
        session_authority=getattr(runner, "session_authority", None), webapp=webapp)
    mount = GatewayDashboardMount(runner, state)
    # Lifespan off: the dashboard's own lifespan runs when it is mounted (GatewayDashboardMount).
    config, server = build_listener_server(mount, host, port, auth_required=state.auth_required,
                                           timeout_graceful_shutdown=5, lifespan="off")
    family, kind, proto, _, address = socket.getaddrinfo(
        host, port, type=socket.SOCK_STREAM,
    )[0]
    listener = _listener_socket(family, kind, proto, port,
                                dual_stack=family == socket.AF_INET6 and address[0] == "::")
    _ACTIVE_RUNNER = runner
    try:
        listener.bind(address)
        if port:
            # Under SO_REUSEADDR another socket may bind a fixed port too, and the first to listen
            # owns it. Listening now settles that before the dashboard mounts, so a lost race falls
            # back at once instead of after the mount is up.
            listener.listen(config.backlog)
        listener.setblocking(False)
        state.bound_port = listener.getsockname()[1]
        # Loading the ASGI graph may fail too; it owns the same bound socket.
        if not config.loaded:
            config.load()
        config.loaded_app = GatewayRuntimeAPI(config.loaded_app, runner, mount)
        # A gated listener needs the dashboard's auth providers (and refuses to start without one);
        # an already-imported dashboard costs nothing to mount now; the Webapp serves its page and
        # mints launch links from the start.
        if state.auth_required or loaded_web is not None or webapp is not None:
            await mount.ensure()
        server.lifespan = config.lifespan_class(config)
        await server.startup(sockets=[listener])
        if not server.started or server.should_exit:
            raise RuntimeError("gateway API lifespan startup failed")
    except BaseException:
        listener.close()
        _ACTIVE_RUNNER = None
        await mount.close()
        raise

    async def serve():
        global _ACTIVE_RUNNER
        try:
            await server.main_loop()
        finally:
            try:
                await server.shutdown(sockets=[listener])
            finally:
                listener.close()
                try:
                    await mount.close()
                finally:
                    _ACTIVE_RUNNER = None

    task = asyncio.create_task(serve(), name="gateway-api")
    return GatewayAPIHandle(
        # The address the socket bound, never a name: native clients accept only a numeric loopback
        # origin, and ``localhost`` may resolve to either family.
        api_origin=dial_origin(listener.getsockname()[0], state.bound_port),
        server=server, task=task, app=mount, socket=listener,
    )


def warm_gateway_dashboard(runner) -> None:
    """Mount the dashboard in the background once the gateway is READY (supervised gateways, whose
    clients — Desktop, the dashboard SPA — call its routes right away). Never raises."""
    handle = getattr(runner, "session_api", None)
    mount = getattr(handle, "app", None)
    if not isinstance(mount, GatewayDashboardMount) or mount.loaded:
        return

    async def _warm():
        try:
            await mount.ensure()
        except Exception:
            if not mount._closed:
                logger.warning("Gateway dashboard API warm-up failed; it loads on first request", exc_info=True)

    runner._gateway_dashboard_warmup = asyncio.get_running_loop().create_task(
        _warm(), name="gateway-dashboard-warmup")


async def stop_gateway_api(handle: GatewayAPIHandle) -> None:
    """Drain sockets without stopping the session authority or taking signals."""
    handle.server.should_exit = True
    # The bootstrap supervisor reports listener failure. Cleanup must still
    # reach adapter/worker settlement when that listener raised or was cancelled.
    await asyncio.shield(asyncio.gather(handle.task, return_exceptions=True))


def _embedded_chat_enabled() -> bool:
    """``web_server._DASHBOARD_EMBEDDED_CHAT_ENABLED`` without importing the dashboard (always True
    unless a loaded dashboard says otherwise)."""
    web = sys.modules.get("hermes_cli.web_server")
    return True if web is None else bool(getattr(web, "_DASHBOARD_EMBEDDED_CHAT_ENABLED", True))


def _restart_wait_read(runner, scope) -> bool:
    from gateway.drain_control import restart_wait_read
    return scope['type'] == 'http' and restart_wait_read(runner, scope.get('method'), scope.get('path'))


def _restart_wait_socket_ticket(runner, scope) -> bool:
    from gateway.drain_control import restart_wait_socket_ticket
    return scope['type'] == 'http' and restart_wait_socket_ticket(
        runner, scope.get('method'), scope.get('path'), scope.get('query_string', b'').decode('latin-1'))


class GatewayRuntimeAPI:
    """Redeem private local tickets at the existing WS subprotocol boundary.

    Other credentials and HTTP routes retain the complete dashboard gate. Local
    bootstrap never grants an exposure/worker ticket interactive permissions.
    *web_app* is the dashboard mount (or any object whose ``state`` carries the listener's
    ``bound_host`` / ``auth_required`` / ``trusted_public_hosts``).
    """
    def __init__(self, app, runner, web_app):
        self.app, self.runner, self.web_app = app, runner, web_app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'websocket' or scope.get('path') != '/api/ws':
            return await self._serve(scope, receive, send)
        started = accepted = False

        async def send_watched(message):
            nonlocal started, accepted
            started = True
            accepted = accepted or message.get('type') == 'websocket.accept'
            await send(message)

        try:
            return await self._serve(scope, receive, send_watched)
        except Exception as exc:
            # Uvicorn answers an upgrade that raised before accept with HTTP 500 and logs it only to
            # its own logger, never gateway.log: the dialing cron worker or managed worker sees a bare
            # InvalidStatus (#139922). Name the cause here; uvicorn still sends the 500, but only when
            # nothing was sent yet: a handler that had closed the upgrade already got its 403 out.
            if not accepted:
                logger.error('Gateway refused a session WebSocket upgrade from %s %s: %s: %s',
                             scope.get('hermes.gateway_socket_peer') or scope.get('client'),
                             'after its handler had refused it' if started else 'with HTTP 500',
                             type(exc).__name__, exc, exc_info=True)
            raise

    async def _serve(self, scope, receive, send):
        descriptor = getattr(self.runner, 'session_runtime_descriptor', None)
        if scope['type'] not in {'http', 'websocket'} or descriptor is None:
            return await self.app(scope, receive, send)
        draining_socket = (descriptor['state'] in {'ready', 'draining'}
            and (self.runner._draining or descriptor['state'] == 'draining')
            and scope['type'] == 'websocket' and scope['path'] == '/api/ws')
        if ((descriptor['state'] != 'ready' or self.runner._draining) and not draining_socket
                and not _restart_wait_read(self.runner, scope)
                and not _restart_wait_socket_ticket(self.runner, scope)):
            if scope['type'] == 'websocket':
                await send({'type': 'websocket.close', 'code': 1013})
            else:
                from starlette.responses import JSONResponse
                await JSONResponse({'error': 'gateway_not_ready', 'state': descriptor['state']},
                                   status_code=503)(scope, receive, send)
            return
        # Capture before Uvicorn's proxy middleware rewrites scope.client (native grants, both kinds).
        scope['hermes.gateway_socket_peer'] = scope.get('client')
        from gateway.run_idle_exit import attached_client, note_client_activity
        if scope['type'] != 'websocket' or scope['path'] != '/api/ws':
            note_client_activity(self.runner)  # dashboard/API traffic keeps an idle-exit gateway up
            return await self.app(scope, receive, send)
        original_receive = receive
        allow_draining = False
        admitted_in_wait = False
        from gateway.drain_control import restart_wait_open

        async def receive_admitted():
            message = await original_receive()
            # Only a socket admitted INTO the restart wait (to answer a parked prompt) outlives it. One
            # opened before the wait is cut as before: its client reconnects to the next owner, and a
            # held one (a kanban worker polling its card) would otherwise pin the drain to its cap.
            if ((descriptor['state'] != 'ready' or self.runner._draining) and not allow_draining
                    and not (admitted_in_wait and restart_wait_open(self.runner))):
                return {'type': 'websocket.disconnect', 'code': 1013}
            return message

        receive = receive_admitted
        from starlette.websockets import WebSocket
        from hermes_cli.web_server_boundary import (
            _gateway_ws_ticket_from_subprotocol, ws_client_reason, ws_host_origin_reason,
        )
        state = getattr(self.web_app, 'state', None)
        scope['app'] = self.web_app
        ws = WebSocket(scope, receive, send)
        ticket, reason = _gateway_ws_ticket_from_subprotocol(ws)
        if reason == 'none' or ws.headers.get('origin'):
            if draining_socket:
                # Let a Webapp tab's ticket reach its normal upgrade gate during restart wait: only a
                # surface that mints browser tickets redeems one there (elsewhere the loopback gate
                # takes ``?token=``).
                from hermes_cli.web_server_surface import policy
                admitted_in_wait = (reason == 'ok' and 'internal' not in ws.query_params
                                    and policy(state).gateway_ticket
                                    and restart_wait_open(self.runner))
                if not admitted_in_wait:
                    await ws.close(code=1013)
                    return
            return await self.app(scope, receive, send)
        if (reason != 'ok' or not _embedded_chat_enabled()
                or ws_host_origin_reason(ws, state) is not None or ws_client_reason(ws, state) is not None
                or ws.headers.get('origin')
                or not ws.client or ws.client.host not in {'127.0.0.1', '::1'}):
            await ws.close(code=4403)
            return
        operator = True
        try:
            grant = self.runner.session_ticket_store.redeem(ticket, profile_id=None, purpose='interactive')
        except PermissionError:
            operator = False
            try:
                grant = self.runner.session_ticket_store.redeem(ticket, profile_id=None, purpose='worker-adoption')
            except PermissionError:
                # Browser/OAuth tickets have a separate issuer.
                if draining_socket:
                    await ws.close(code=1013)
                    return
                return await self.app(scope, receive, send)
        allow_draining = not operator
        # Decided here, at admission, not at the mint: a restart wait that opened after the client's
        # discovery still refuses the machine poller that asked to stay out of it (a Kanban worker),
        # which would otherwise hold the turn the wait is waiting on.
        admitted_in_wait = restart_wait_open(self.runner) and not grant.get('refuse_restart_wait')
        # A restart's after-turn wait still owns its turns: an interactive client attaches to answer
        # a parked approval/clarify (#139041); stop() closes this window.
        if ((descriptor['state'] != 'ready' or self.runner._draining) and not allow_draining
                and not admitted_in_wait):
            await ws.close(code=1013)
            return
        from gateway.session_authorities import authority_for_profile_id
        authority = authority_for_profile_id(self.runner, grant['profile_id'])
        if authority is None:
            await ws.close(code=4403)
            return
        # The ticket names the served profile; this connection binds to that home's authority.
        scope['hermes.session_authority'] = authority
        from gateway.session_contract import CANONICAL_GATEWAY_PROTOCOL
        from tui_gateway.ws import handle_ws
        from gateway.session_authorities import owner_scope
        with owner_scope(authority), attached_client(self.runner):
            await handle_ws(ws, auth_identity={'user_id': grant['subject'], 'provider': 'local',
                                              'profile_id': grant['profile_id'],
                                              'instance_id': grant['instance_id'],
                                              'capabilities': grant['capabilities'], 'native_bootstrap': True,
                                              'profile_scope': grant.get('scope', 'profile')},
                            subprotocol=CANONICAL_GATEWAY_PROTOCOL, operator=operator)
