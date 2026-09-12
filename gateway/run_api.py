"""Compose the existing complete HTTP/WS surface into the gateway event loop.

The caller initializes session authority before starting this listener and owns
all runtime services and signals. This module owns only HTTP resources; it does
not bootstrap an agent, scheduler, hosted room, or a second gateway.
"""

import asyncio
from dataclasses import dataclass
import socket
from typing import Any


@dataclass
class GatewayAPIHandle:
    api_origin: str
    server: Any
    task: asyncio.Task
    app: Any
    socket: socket.socket


async def start_gateway_api(runner, *, host: str = "127.0.0.1", port: int = 0) -> GatewayAPIHandle:
    from hermes_cli import web_server as web

    # Existing routers/auth helpers share one process-local app. Never rebind it
    # underneath another listener. Each authority is bound to its own profile DB.
    if getattr(web.app.state, "gateway_runner", None) is not None:
        raise RuntimeError("gateway API already started")
    web._configure_auth_gate(host, False, None, None)
    config, server = web._build_uvicorn_server(host, port)
    config.timeout_graceful_shutdown = 5
    family, kind, proto, _, address = socket.getaddrinfo(
        host, port, type=socket.SOCK_STREAM,
    )[0]
    listener = socket.socket(family, kind, proto)
    try:
        listener.bind(address)
        listener.setblocking(False)
        # Loading the ASGI graph may fail too; it owns the same bound socket.
        if not config.loaded:
            config.load()
        config.loaded_app = GatewayRuntimeAPI(config.loaded_app, runner, web.app)
    except BaseException:
        listener.close()
        raise

    web.app.state.gateway_runner = runner
    web.app.state.session_authority = getattr(runner, "session_authority", None)
    web.app.state.bound_host = host
    web.app.state.bound_port = listener.getsockname()[1]
    try:
        if not config.loaded:
            config.load()
        server.lifespan = config.lifespan_class(config)
        await server.startup(sockets=[listener])
        if not server.started or server.should_exit:
            raise RuntimeError("gateway API lifespan startup failed")
    except BaseException:
        listener.close()
        web.app.state.gateway_runner = None
        web.app.state.session_authority = None
        if hasattr(server, "lifespan") and not server.lifespan.should_exit:
            await server.lifespan.shutdown()
        raise

    async def serve():
        try:
            await server.main_loop()
        finally:
            try:
                await server.shutdown(sockets=[listener])
            finally:
                listener.close()
                web.app.state.gateway_runner = None
                web.app.state.session_authority = None

    task = asyncio.create_task(serve(), name="gateway-api")
    origin_host = f"[{host}]" if ":" in host else host
    return GatewayAPIHandle(
        api_origin=f"http://{origin_host}:{web.app.state.bound_port}",
        server=server, task=task, app=web.app, socket=listener,
    )


async def stop_gateway_api(handle: GatewayAPIHandle) -> None:
    """Drain sockets without stopping the session authority or taking signals."""
    handle.server.should_exit = True
    # The bootstrap supervisor reports listener failure. Cleanup must still
    # reach adapter/worker settlement when that listener raised or was cancelled.
    await asyncio.shield(asyncio.gather(handle.task, return_exceptions=True))


class GatewayRuntimeAPI:
    """Redeem private local tickets at the existing WS subprotocol boundary.

    Other credentials and HTTP routes retain the complete dashboard gate. Local
    bootstrap never grants an exposure/worker ticket interactive permissions.
    """
    def __init__(self, app, runner, web_app):
        self.app, self.runner, self.web_app = app, runner, web_app

    async def __call__(self, scope, receive, send):
        descriptor = getattr(self.runner, 'session_runtime_descriptor', None)
        if scope['type'] not in {'http', 'websocket'} or descriptor is None:
            return await self.app(scope, receive, send)
        if descriptor['state'] != 'ready' or self.runner._draining:
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
        if scope['type'] != 'websocket' or scope['path'] != '/api/ws':
            return await self.app(scope, receive, send)
        original_receive = receive

        async def receive_admitted():
            message = await original_receive()
            if descriptor['state'] != 'ready' or self.runner._draining:
                return {'type': 'websocket.disconnect', 'code': 1013}
            return message

        receive = receive_admitted
        from starlette.websockets import WebSocket
        from hermes_cli.web_server_chat import (
            _gateway_ws_ticket_from_subprotocol, _ws_request_is_allowed,
        )
        scope['app'] = self.web_app
        ws = WebSocket(scope, receive, send)
        ticket, reason = _gateway_ws_ticket_from_subprotocol(ws)
        if reason == 'none' or ws.headers.get('origin'):
            return await self.app(scope, receive, send)
        from hermes_cli import web_server as web
        if (reason != 'ok' or not web._DASHBOARD_EMBEDDED_CHAT_ENABLED
                or not _ws_request_is_allowed(ws) or ws.headers.get('origin')
                or not ws.client or ws.client.host not in {'127.0.0.1', '::1'}):
            await ws.close(code=4403)
            return
        try:
            grant = self.runner.session_ticket_store.redeem(ticket, profile_id=None, purpose='interactive')
        except PermissionError:
            try:
                grant = self.runner.session_ticket_store.redeem(ticket, profile_id=None, purpose='worker-adoption')
            except PermissionError:
                # Browser/OAuth tickets have a separate issuer.
                return await self.app(scope, receive, send)
        from gateway.session_authorities import authority_for_profile_id
        authority = authority_for_profile_id(self.runner, grant['profile_id'])
        if authority is None:
            await ws.close(code=4403)
            return
        # The ticket names the served profile; this connection binds to that home's authority.
        scope['hermes.session_authority'] = authority
        from tui_gateway.ws import handle_ws
        from gateway.run import _profile_runtime_scope
        from pathlib import Path
        with _profile_runtime_scope(Path(authority.profile_id), hydrate_secrets=False):
            await handle_ws(ws, auth_identity={'user_id': grant['subject'], 'provider': 'local',
                                              'profile_id': grant['profile_id'],
                                              'instance_id': grant['instance_id'],
                                              'capabilities': grant['capabilities'], 'native_bootstrap': True},
                            subprotocol='hermes-gateway-v1')
