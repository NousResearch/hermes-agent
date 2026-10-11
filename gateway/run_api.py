"""The gateway's local HTTP/WS listener: the native session WebSocket plus the dashboard API.

The caller initializes session authority before starting this listener and owns
all runtime services and signals. This module owns only HTTP resources; it does
not bootstrap an agent, scheduler, hosted room, or a second gateway.

The native session WebSocket (ticketed ``/api/ws`` — every local CLI/TUI/Desktop session) is served
without the dashboard app: importing ``hermes_cli.web_server`` (about 690 routes, ~0.6 s) before the
listener bound stalled every cold start and the control socket's identify. The dashboard is mounted
on its first request (imported off the event loop) or warmed after READY for a supervised gateway
(``warm_gateway_dashboard``); a gated listener (non-loopback bind or ``dashboard.public_url``) and a
process that already imported it mount it before the listener starts, exactly as before.
"""

import asyncio
from dataclasses import dataclass
import logging
import socket
import sys
import types
from typing import Any

logger = logging.getLogger(__name__)

# One listener per process: the dashboard routers share one process-local app.
_ACTIVE_RUNNER: Any = None


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
    path checks, mirrored onto the dashboard app state once it is mounted.
    """

    def __init__(self, runner, state):
        self.runner, self._state = runner, state
        self.web = None
        self._lock: asyncio.Lock | None = None
        self._lifespan = None
        self._closed = False

    @property
    def loaded(self) -> bool:
        return self.web is not None

    @property
    def state(self):
        """Boundary state: the dashboard app's own state once mounted (operators and tests may
        change it there), the listener's pre-mount snapshot before."""
        return self.web.app.state if self.web is not None else self._state

    def _bind_state(self, web) -> None:
        # Local clients authenticate with tickets minted on the owner-only control socket. The SPA
        # session token is a bearer credential any loopback peer (any OS user) could read from
        # GET /, so the gateway-hosted listener never publishes it.
        web.app.state.withhold_session_token = True
        web._configure_auth_gate(self._state.bound_host, False, None, None)
        web.app.state.gateway_runner = self.runner
        web.app.state.session_authority = getattr(self.runner, "session_authority", None)
        web.app.state.bound_host = self._state.bound_host
        web.app.state.bound_port = self._state.bound_port

    def _unbind_state(self) -> None:
        if self.web is not None:
            self.web.app.state.gateway_runner = None
            self.web.app.state.session_authority = None
            self.web.app.state.withhold_session_token = False

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
            self._bind_state(web)
            lifespan = web.app.router.lifespan_context(web.app)
            try:
                await lifespan.__aenter__()
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


async def start_gateway_api(runner, *, host: str = "127.0.0.1", port: int = 0) -> GatewayAPIHandle:
    global _ACTIVE_RUNNER
    from hermes_cli.web_server_boundary import _dashboard_public_hosts, listener_auth_required
    from hermes_cli.web_server_lifecycle import build_listener_server

    # Existing routers/auth helpers share one process-local app. Never rebind it
    # underneath another listener. Each authority is bound to its own profile DB.
    loaded_web = sys.modules.get("hermes_cli.web_server")
    if _ACTIVE_RUNNER is not None or (
            loaded_web is not None and getattr(loaded_web.app.state, "gateway_runner", None) is not None):
        raise RuntimeError("gateway API already started")
    trusted = _dashboard_public_hosts()
    state = types.SimpleNamespace(
        bound_host=host, bound_port=None, auth_required=listener_auth_required(host, trusted),
        trusted_public_hosts=trusted, gateway_runner=runner,
        session_authority=getattr(runner, "session_authority", None))
    mount = GatewayDashboardMount(runner, state)
    # Lifespan off: the dashboard's own lifespan runs when it is mounted (GatewayDashboardMount).
    config, server = build_listener_server(mount, host, port, auth_required=state.auth_required,
                                           timeout_graceful_shutdown=5, lifespan="off")
    family, kind, proto, _, address = socket.getaddrinfo(
        host, port, type=socket.SOCK_STREAM,
    )[0]
    listener = socket.socket(family, kind, proto)
    _ACTIVE_RUNNER = runner
    try:
        listener.bind(address)
        listener.setblocking(False)
        state.bound_port = listener.getsockname()[1]
        # Loading the ASGI graph may fail too; it owns the same bound socket.
        if not config.loaded:
            config.load()
        config.loaded_app = GatewayRuntimeAPI(config.loaded_app, runner, mount)
        # A gated listener needs the dashboard's auth providers (and refuses to start without one);
        # an already-imported dashboard costs nothing to mount now.
        if state.auth_required or loaded_web is not None:
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
    origin_host = f"[{host}]" if ":" in host else host
    return GatewayAPIHandle(
        api_origin=f"http://{origin_host}:{state.bound_port}",
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
        descriptor = getattr(self.runner, 'session_runtime_descriptor', None)
        if scope['type'] not in {'http', 'websocket'} or descriptor is None:
            return await self.app(scope, receive, send)
        draining_socket = (descriptor['state'] in {'ready', 'draining'}
            and (self.runner._draining or descriptor['state'] == 'draining')
            and scope['type'] == 'websocket' and scope['path'] == '/api/ws')
        if (descriptor['state'] != 'ready' or self.runner._draining) and not draining_socket:
            if scope['type'] == 'websocket':
                await send({'type': 'websocket.close', 'code': 1013})
            else:
                from starlette.responses import JSONResponse
                await JSONResponse({'error': 'gateway_not_ready', 'state': descriptor['state']},
                                   status_code=503)(scope, receive, send)
            return
        if scope['type'] == 'http':
            # Capture before Uvicorn's proxy middleware rewrites scope.client.
            scope['hermes.gateway_socket_peer'] = scope.get('client')
        from gateway.run_idle_exit import attached_client, note_client_activity
        if scope['type'] != 'websocket' or scope['path'] != '/api/ws':
            note_client_activity(self.runner)  # dashboard/API traffic keeps an idle-exit gateway up
            return await self.app(scope, receive, send)
        original_receive = receive
        allow_draining = False

        async def receive_admitted():
            message = await original_receive()
            if (descriptor['state'] != 'ready' or self.runner._draining) and not allow_draining:
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
        if (descriptor['state'] != 'ready' or self.runner._draining) and not allow_draining:
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
