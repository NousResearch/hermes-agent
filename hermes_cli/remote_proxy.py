"""``hermes dashboard proxy`` — a hardened API-only reverse proxy for remote access.

The desktop app's remote-gateway mode speaks to the backend over ``/api/*``.
Exposing the whole dashboard server to a tunnel or
reverse proxy therefore over-shares: the SPA HTML embeds the dashboard session
token, and several ``/api`` routes perform machine-lifecycle operations
(update, gateway restart, backup download) that are safe from localhost but
dangerous from a phone on the train.

This proxy is the piece users previously had to hand-roll ("an already-running
Hermes backend ... behind a trusted proxy", per the desktop's remote-gateway
copy). It:

- forwards ONLY ``/api/*`` (HTTP and WebSocket) to the local backend, so the
  SPA HTML — and the session token inlined into it — never crosses the tunnel;
- denies a default set of lifecycle routes outright (403), because a remote
  surface triggering ``hermes update`` or ``gateway stop`` takes the machine's
  gateways down with no local operator to recover them, and a backup download
  is a whole-HERMES_HOME exfiltration in one request;
- audit-logs every denied request to ``logs/remote-proxy-denied.log`` for
  attribution;
- strips hop-by-hop headers in both directions.

It deliberately does NOT duplicate authentication: the loopback dashboard
server keeps enforcing its session token on every protected HTTP and WebSocket
route. Browser OAuth is intentionally unavailable because the proxy does not
forward ``/login`` or ``/auth/callback``. Bind it to loopback, configure a fixed
``HERMES_DASHBOARD_SESSION_TOKEN`` for the backend, and point a tunnel
(Cloudflare, Tailscale funnel, SSH -R, ...) at the proxy.
"""

# NOTE: no `from __future__ import annotations` here. The FastAPI handlers
# below are nested inside create_proxy_app and annotate parameters with
# locally imported types (Request, WebSocket); stringified annotations can't
# be resolved from module globals by FastAPI's dependency system, which would
# silently demote `request` to a query parameter (422s on every call).

import asyncio
import datetime
import logging
from pathlib import Path
from typing import Iterable, Sequence

logger = logging.getLogger(__name__)

# Lifecycle and bulk-data routes a remote surface must not reach by default.
# Rationale per route family:
# - hermes/update: spawns a self-update that restarts backends; remotely
#   triggered updates strand gateways with nobody at the machine.
# - gateway/*: start/stop/restart/drain of the machine's launchd/systemd
#   gateways — same blast radius as update.
# - ops/backup + ops/backup/download: creates, then serves, a zip of the
#   entire HERMES_HOME (sessions, config, credentials) — single-request
#   exfiltration if a tunnel is ever misconfigured.
# - ops/import + ops/import-upload: restores an uploaded archive over the
#   live HERMES_HOME — remote code/data injection.
# - ops/config-migrate: rewrites config on disk.
# Every one of these remains available from localhost surfaces.
DEFAULT_DENY_ROUTES: frozenset[str] = frozenset(
    {
        "/api/hermes/update",
        "/api/gateway/start",
        "/api/gateway/stop",
        "/api/gateway/restart",
        "/api/gateway/drain",
        "/api/ops/backup",
        "/api/ops/backup/download",
        "/api/ops/import",
        "/api/ops/import-upload",
        "/api/ops/config-migrate",
    }
)

# Hop-by-hop headers (RFC 9110 §7.6.1) plus proxy-internal ones. These are
# connection-scoped and must not be forwarded in either direction.
_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
}


def resolve_deny_routes(
    allow: Sequence[str] | None = None,
    deny: Sequence[str] | None = None,
) -> frozenset[str]:
    """Apply ``--allow-route`` / ``--deny-route`` overrides to the default set.

    ``allow`` removes routes from the default deny-list (an explicit operator
    decision to re-enable a lifecycle route remotely); ``deny`` adds extra
    routes. Overrides are exact-path matches, normalized to no trailing slash.
    """
    routes = set(DEFAULT_DENY_ROUTES)
    for path in deny or ():
        routes.add(_normalize(path))
    for path in allow or ():
        routes.discard(_normalize(path))
    return frozenset(routes)


def _normalize(path: str) -> str:
    path = "/" + str(path).strip().lstrip("/")
    return path.rstrip("/") or "/"


def classify_request(path: str, deny_routes: frozenset[str]) -> str:
    """Return ``forward`` | ``deny`` | ``not_found`` for a request path.

    Only ``/api/*`` is ever forwarded; everything else (the SPA, static
    assets, health pages) 404s so nothing outside the JSON-RPC surface can
    leak through the tunnel. Deny matches are exact-path.
    """
    normalized = _normalize(path)
    if normalized != "/api" and not normalized.startswith("/api/"):
        return "not_found"
    if normalized in deny_routes:
        return "deny"
    return "forward"


def filtered_headers(items: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    """Drop hop-by-hop headers; everything else passes through untouched."""
    return [(k, v) for k, v in items if k.lower() not in _HOP_BY_HOP]


def _audit_denied(deny_log: Path, method: str, path: str, client: str) -> None:
    """Append one attribution line per denied request. Best-effort."""
    try:
        deny_log.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        with deny_log.open("a", encoding="utf-8") as fh:
            fh.write(f"{stamp} DENIED {method} {path} client={client}\n")
    except OSError:
        logger.debug("remote-proxy deny log write failed", exc_info=True)


def create_proxy_app(
    *,
    upstream: str,
    deny_routes: frozenset[str] = DEFAULT_DENY_ROUTES,
    deny_log: Path | None = None,
):
    """Build the FastAPI proxy app. Split from serving for testability."""
    import contextlib

    import httpx
    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import PlainTextResponse, StreamingResponse

    upstream = upstream.rstrip("/")
    upstream_ws = "ws" + upstream.removeprefix("http")
    state: dict = {}

    @contextlib.asynccontextmanager
    async def _lifespan(app):
        state["client"] = httpx.AsyncClient(base_url=upstream, timeout=None)
        try:
            yield
        finally:
            client = state.pop("client", None)
            if client is not None:
                await client.aclose()

    app = FastAPI(
        openapi_url=None, docs_url=None, redoc_url=None, lifespan=_lifespan
    )

    async def _forward_http(request: Request, path: str):
        client: httpx.AsyncClient = state["client"]
        url = "/" + path
        if request.url.query:
            url += "?" + request.url.query
        # Buffer the request body rather than streaming it through: JSON-RPC
        # calls are small, the bulk-upload routes are on the deny-list, and a
        # buffered body survives httpx's re-send paths (redirect/auth) that a
        # one-shot generator cannot.
        body = await request.body()
        upstream_request = client.build_request(
            request.method,
            url,
            headers=filtered_headers(request.headers.items()),
            content=body if body else None,
        )
        upstream_response = await client.send(upstream_request, stream=True)

        async def _body():
            try:
                try:
                    async for chunk in upstream_response.aiter_raw():
                        yield chunk
                except httpx.StreamConsumed:
                    # Transports that preload content (MockTransport in tests,
                    # cached/intercepted responses) mark the stream consumed
                    # before we get to iterate; serve the buffered bytes.
                    yield upstream_response.content
            finally:
                await upstream_response.aclose()

        response = StreamingResponse(
            _body(),
            status_code=upstream_response.status_code,
        )
        # Starlette's mapping-style ``headers=`` argument collapses repeated
        # fields. Preserve the raw list so multiple Set-Cookie values and other
        # repeatable end-to-end headers survive the proxy boundary.
        response.raw_headers = [
            (name.lower().encode("latin-1"), value.encode("latin-1"))
            for name, value in filtered_headers(upstream_response.headers.multi_items())
        ]
        return response

    @app.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
    )
    async def _proxy(request: Request, path: str):
        verdict = classify_request("/" + path, deny_routes)
        if verdict == "not_found":
            return PlainTextResponse("not found\n", status_code=404)
        if verdict == "deny":
            client_host = request.client.host if request.client else "?"
            if deny_log is not None:
                _audit_denied(deny_log, request.method, "/" + path, client_host)
            logger.warning(
                "remote-proxy denied %s /%s from %s", request.method, path, client_host
            )
            return PlainTextResponse(
                "route disabled on the remote surface\n", status_code=403
            )
        return await _forward_http(request, path)

    @app.websocket("/{path:path}")
    async def _proxy_ws(websocket: WebSocket, path: str):
        import websockets as ws_client

        verdict = classify_request("/" + path, deny_routes)
        if verdict != "forward":
            if verdict == "deny":
                client_host = websocket.client.host if websocket.client else "?"
                if deny_log is not None:
                    _audit_denied(deny_log, "WS", "/" + path, client_host)
                logger.warning(
                    "remote-proxy denied WS /%s from %s", path, client_host
                )
            # 4403: policy close. Accept first so the close frame is delivered.
            await websocket.accept()
            await websocket.close(code=4403)
            return

        url = f"{upstream_ws}/{path}"
        if websocket.url.query:
            url += "?" + websocket.url.query
        headers = [
            (k, v)
            for k, v in filtered_headers(websocket.headers.items())
            if k.lower() not in {"sec-websocket-key", "sec-websocket-version", "sec-websocket-extensions"}
        ]
        await websocket.accept(subprotocol=websocket.headers.get("sec-websocket-protocol"))
        try:
            async with ws_client.connect(
                url, additional_headers=headers, max_size=None
            ) as upstream_socket:

                async def client_to_upstream() -> None:
                    while True:
                        message = await websocket.receive()
                        if message.get("type") == "websocket.disconnect":
                            await upstream_socket.close()
                            return
                        if message.get("text") is not None:
                            await upstream_socket.send(message["text"])
                        elif message.get("bytes") is not None:
                            await upstream_socket.send(message["bytes"])

                async def upstream_to_client() -> None:
                    async for message in upstream_socket:
                        if isinstance(message, (bytes, bytearray)):
                            await websocket.send_bytes(bytes(message))
                        else:
                            await websocket.send_text(message)

                done, pending = await asyncio.wait(
                    [
                        asyncio.create_task(client_to_upstream()),
                        asyncio.create_task(upstream_to_client()),
                    ],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
        except (WebSocketDisconnect, ConnectionError, OSError):
            pass
        finally:
            try:
                await websocket.close()
            except Exception:
                pass

    return app


def run_remote_proxy(args) -> int:
    """Entry point for ``hermes dashboard proxy`` (handler injected in main)."""
    import uvicorn

    from hermes_constants import get_hermes_home

    deny_routes = resolve_deny_routes(
        allow=getattr(args, "allow_routes", None),
        deny=getattr(args, "extra_deny_routes", None),
    )
    deny_log = get_hermes_home() / "logs" / "remote-proxy-denied.log"
    upstream = str(getattr(args, "upstream", "") or "http://127.0.0.1:9119")

    app = create_proxy_app(
        upstream=upstream, deny_routes=deny_routes, deny_log=deny_log
    )

    host = str(getattr(args, "host", "127.0.0.1") or "127.0.0.1")
    port = int(getattr(args, "port", 9123) or 9123)
    print(f"Hermes remote proxy: {host}:{port} -> {upstream} (API-only)")
    print(f"  Denied routes ({len(deny_routes)}): " + ", ".join(sorted(deny_routes)))
    print(f"  Deny audit log: {deny_log}")
    print("  Point your tunnel at this listener; keep the backend on loopback.")
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0
