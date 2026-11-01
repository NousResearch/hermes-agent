"""The hub's relay: ``/api/relay/<gateway>/<path>`` carries a signed-in browser's REST calls and
sockets to a paired gateway (``hermes_cli.webapp_relay``).

HTTP. The browser's request is signed with this hub's key for the gateway's grant and sent on as
``<gateway url>/<path>``, body and response streamed both ways. Cookies, the browser's own
credentials and forwarding headers never cross. Only headers that describe the body come back, under
a sandboxing content policy, and a redirect only within the gateway: what it serves never acts on
this hub's origin. A 401 from the
gateway means it no longer accepts the grant, so the browser gets a 409 ``relay_reconnect`` instead of
being sent to this hub's sign-in.

Sockets. A relayed socket opens only with this hub's one-use ``relay`` ticket (``?relay_ticket=``,
minted by ``POST /api/webapp/socket-ticket?socket=relay``) for the same identity, and is tracked with
the browser's other sockets, so signing out here closes it. The socket's own credential is the
gateway's: the ticket the browser minted there through this relay, which rides the query or the
subprotocol unchanged. Frames pass through as they are; each side's close reaches the other.

Only the signed-in identity that paired a gateway reaches it.
"""
from __future__ import annotations

import asyncio
import contextlib
import http.cookiejar
import logging
import threading
import urllib.parse
import weakref
from typing import Any, Optional

import httpx
from fastapi import APIRouter, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from hermes_cli import webapp_relay as relay
from hermes_cli import webapp_relay_gateways as hub

_log = logging.getLogger(__name__)
router = APIRouter()

_PREFIX = "/api/relay/"
_HOP_BY_HOP = frozenset({"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
                         "trailer", "transfer-encoding", "upgrade"})
#: Request headers that never cross to the gateway: this hub's credentials and the browser's
#: provenance. The relay signature replaces them.
_DROP_REQUEST = _HOP_BY_HOP | frozenset({
    "host", "cookie", "authorization", "origin", "referer", "x-hermes-session-token", "forwarded",
    "x-real-ip", "x-hermes-gateway-ticket", "x-hermes-window-ticket", *relay.RELAY_HEADERS})
#: Response headers that cross back from the gateway: what describes the body. Anything else would
#: let the gateway act on this hub's origin (cookies, CORS, HSTS, service workers, site-data
#: clearing, refreshes, its own content policy, or caching that outlives the grant).
_KEEP_RESPONSE = frozenset({
    "content-type", "content-length", "content-encoding", "content-language", "content-disposition",
    "content-range", "accept-ranges", "vary", "retry-after"})
#: Whatever the gateway sends is its content, not this hub's page: a document opened from it runs in
#: an opaque origin with no scripts, nothing is sniffed into one, and the browser keeps no copy for
#: after a revoke or for another account signed in here later.
_RESPONSE_POLICY = {"Content-Security-Policy": "sandbox", "X-Content-Type-Options": "nosniff",
                    "Cache-Control": "no-store"}
_METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
#: The most a relayed frame may carry: what the gateway's own socket accepts (file.attach data URLs).
_MAX_FRAME = 384 * 1024 * 1024
#: What a URL parser reads as the "." and ".." path segments (WHATWG URL, any letter case).
_DOT = frozenset({".", "%2e"})
_DOT_DOT = frozenset({"..", ".%2e", "%2e.", "%2e%2e"})


# ---- Shared client ----------------------------------------------------------------------------

def relay_client(state):
    """This listener's HTTP client for relayed requests, one per event loop: no redirects followed,
    no proxy settings from the environment, no read deadline (streams run as long as they run), and
    no cookies kept. A gateway refuses a relayed request that carries one, and a cookie is not scoped
    to a port, so one set by any gateway (or a proxy in front of it) would break later requests."""
    loop = asyncio.get_running_loop()
    held = getattr(state, "webapp_relay_client", None)
    if held is not None and held[0] is loop:
        return held[1]
    no_cookies = http.cookiejar.CookieJar(policy=http.cookiejar.DefaultCookiePolicy(allowed_domains=[]))
    client = httpx.AsyncClient(follow_redirects=False, trust_env=False, cookies=no_cookies,
                               timeout=httpx.Timeout(connect=15.0, read=None, write=60.0, pool=15.0))
    state.webapp_relay_client = (loop, client)
    return client


async def signed_get(state, row: dict[str, Any], target: str):
    """One signed GET of *target* on *row*'s gateway (the Settings connection test)."""
    key = relay.hub_key(state.webapp_home)
    headers = relay.sign_request(key, method="GET", target=target, audience=row["url"], grant_id=row["grant_id"],
                                 subject=row["subject"])
    return await relay_client(state).get(f"{row['url']}{target}", headers=headers, timeout=15.0)


# ---- Routing ----------------------------------------------------------------------------------

def _gateway(state, identity: dict[str, str], gateway_id: str) -> Optional[dict[str, Any]]:
    if not relay.relay_enabled(state):
        return None
    return hub.gateway_for(state.webapp_home, identity, gateway_id)


def _target(scope, gateway_id: str, *, drop_query: frozenset[str] = frozenset()) -> Optional[str]:
    """The raw path and query a relayed request names on the gateway, or None when it is not one
    this relay carries (malformed, or another relay: one hop)."""
    raw = (scope.get("raw_path") or scope.get("path", "").encode("utf-8")).decode("latin-1")
    lead = f"{_PREFIX}{gateway_id}"
    index = raw.find(lead)
    if index < 0:
        return None
    rest = raw[index + len(lead):] or "/"
    if not rest.startswith("/") or rest.startswith("//") or rest.startswith(_PREFIX):
        return None
    query = (scope.get("query_string") or b"").decode("latin-1")
    if drop_query and query:
        pairs = [part for part in query.split("&")
                 if urllib.parse.unquote_plus(part.split("=", 1)[0]) not in drop_query]
        query = "&".join(pairs)
    return f"{rest}?{query}" if query else rest


def _identity(request: Request) -> dict[str, str]:
    from hermes_cli.webapp_relay_grants import identity_of
    session = getattr(request.state, "session", None)
    if session is None or getattr(request.state, "relay_grant", None) is not None:
        raise HTTPException(401, "Unauthorized")
    return identity_of(session)


def _without_dot_segments(path: str) -> str:
    """Absolute *path* with its "." and ".." segments resolved the way a browser resolves them."""
    segments, kept = path.split("/")[1:], []
    for index, segment in enumerate(segments):
        lowered = segment.lower()
        if lowered in _DOT_DOT:
            if kept:
                kept.pop()
        elif lowered not in _DOT:
            kept.append(segment)
            continue
        if index == len(segments) - 1:
            kept.append("")  # a trailing "." or ".." leaves a directory path
    return "/" + "/".join(kept)


def _relayed_location(value: str, row: dict[str, Any], base: str) -> Optional[str]:
    """Where a gateway's redirect points, under this relay; None when it leaves the gateway.

    The browser resolves the Location against this hub, so its dot segments are resolved here first,
    against the gateway: ``/../api/gateway/restart`` names the gateway's route, never this hub's.
    A browser drops tabs and newlines from a URL and reads a backslash as a slash, so a Location
    with those (or any whitespace or control character) is refused rather than reparsed."""
    if value.startswith(f"{row['url']}/") or value == row["url"]:
        value = value[len(row["url"]):] or "/"
    if (not value.startswith("/") or value.startswith("//") or "\\" in value
            or any(ch.isspace() or ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value)):
        return None
    end = min((i for i in (value.find("?"), value.find("#")) if i >= 0), default=len(value))
    return f"{base}{_without_dot_segments(value[:end])}{value[end:]}"


@router.api_route(_PREFIX + "{gateway_id}/{path:path}", methods=_METHODS)
async def relay_http(request: Request, gateway_id: str, path: str):
    from hermes_cli.dashboard_auth.prefix import prefix_from_request

    state = request.app.state
    row = _gateway(state, _identity(request), gateway_id)
    if row is None:
        raise HTTPException(404, "No such gateway")
    target = _target(request.scope, gateway_id)
    if target is None:
        raise HTTPException(400, "Not a route this relay carries")
    method = request.method.upper()
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in _DROP_REQUEST and not k.lower().startswith("x-forwarded-")}
    headers.update(relay.sign_request(relay.hub_key(state.webapp_home), method=method, target=target,
                                      audience=row["url"], grant_id=row["grant_id"], subject=row["subject"]))
    has_body = method not in {"GET", "HEAD", "OPTIONS"} or request.headers.get("content-length") not in (None, "0")
    client = relay_client(state)
    upstream = client.build_request(method, f"{row['url']}{target}", headers=headers,
                                    content=request.stream() if has_body else None)
    try:
        response = await client.send(upstream, stream=True)
    except httpx.HTTPError as exc:  # transport failure: the gateway is unreachable from this hub
        _log.info("Gateway relay to %s failed: %s", row["url"], exc)
        return JSONResponse({"error": "relay_unreachable", "connection_id": gateway_id,
                             "detail": f"{row['label']} is not reachable from this gateway."}, status_code=502)
    if response.status_code == 401:
        await response.aclose()
        return JSONResponse({"error": "relay_reconnect", "connection_id": gateway_id,
                             "detail": f"{row['label']} no longer accepts this gateway's relay. "
                                       "Reconnect it in Settings → Gateways."}, status_code=409)
    base = f"{prefix_from_request(request)}{_PREFIX}{gateway_id}"
    out = {key: value for key, value in response.headers.items() if key.lower() in _KEEP_RESPONSE}
    if "location" in response.headers:
        location = _relayed_location(response.headers["location"], row, base)
        if location is None:
            await response.aclose()
            return JSONResponse({"error": "relay_redirect", "connection_id": gateway_id,
                                 "detail": f"{row['label']} redirected outside itself."}, status_code=502)
        out["Location"] = location
    out.update(_RESPONSE_POLICY)
    return StreamingResponse(response.aiter_raw(), status_code=response.status_code, headers=out,
                             background=BackgroundTask(response.aclose))


# ---- Sockets ----------------------------------------------------------------------------------

_live_lock = threading.Lock()
_live: dict[int, tuple[weakref.ref, asyncio.AbstractEventLoop, str]] = {}


def close_gateway_sockets(gateway_id: str) -> int:
    """Close every relayed socket to *gateway_id* (it was removed); returns how many."""
    with _live_lock:
        live = [(key, entry) for key, entry in _live.items() if entry[2] == gateway_id]
        for key, _entry in live:
            del _live[key]
    count = 0
    for _key, (ref, loop, _gid) in live:
        ws = ref()
        if ws is not None and not loop.is_closed():
            asyncio.run_coroutine_threadsafe(_close(ws, 4404, "Gateway removed"), loop)
            count += 1
    return count


async def _close(ws: WebSocket, code: int, reason: str = "") -> None:
    from starlette.websockets import WebSocketDisconnect
    with contextlib.suppress(RuntimeError, OSError, WebSocketDisconnect):
        await ws.close(code=code, reason=reason[:120])


def _sendable(code: Optional[int]) -> int:
    """A close code one side may send for the other's: 1005/1006 (none/abnormal) are not."""
    if code is None or code in {1005, 1006, 1015} or not 1000 <= code <= 4999:
        return 1011
    return code


def _admit_browser(ws: WebSocket) -> Optional[tuple[dict[str, str], str]]:
    """``(identity, credential generation)`` the browser's relay ticket admits, or None."""
    from hermes_cli.dashboard_auth.ws_tickets import TicketInvalid, consume_ticket
    from hermes_cli.web_server_boundary import ws_host_origin_reason
    from hermes_cli.web_session_sockets import RELAY_SOCKET, admitted_with, ticket_refusal

    state = ws.app.state
    if not ws.headers.get("origin") or ws_host_origin_reason(ws, state) is not None:
        return None
    try:
        info = consume_ticket(ws.query_params.get("relay_ticket", ""))
    except TicketInvalid:
        return None
    admitted = admitted_with(info)
    if ticket_refusal(info, RELAY_SOCKET, auth_required=True) is not None or admitted is None:
        return None
    return {"user_id": info.get("user_id"), "provider": info.get("provider"),
            "issuer": info.get("issuer") or ""}, admitted


def _offered_subprotocols(ws: WebSocket) -> list[str]:
    raw = ws.headers.get("sec-websocket-protocol", "") or ""
    return [value.strip() for value in raw.split(",") if value.strip()]


@router.websocket(_PREFIX + "{gateway_id}/{path:path}")
async def relay_ws(ws: WebSocket, gateway_id: str, path: str) -> None:
    from websockets.asyncio.client import connect
    from websockets.exceptions import WebSocketException
    from websockets.typing import Origin, Subprotocol

    from hermes_cli.web_session_sockets import track

    admitted = _admit_browser(ws)
    row = _gateway(ws.app.state, admitted[0], gateway_id) if admitted is not None else None
    target = _target(ws.scope, gateway_id, drop_query=frozenset({"relay_ticket"})) if row else None
    if admitted is None or row is None or target is None:
        await ws.close(code=4401)
        return
    track(ws, admitted[1])
    parts = urllib.parse.urlsplit(row["url"])
    url = f"{'wss' if parts.scheme == 'https' else 'ws'}://{parts.netloc}{parts.path}{target}"
    offered = [Subprotocol(value) for value in _offered_subprotocols(ws)]
    try:
        upstream = await connect(url, subprotocols=offered or None,
                                 origin=Origin(f"{parts.scheme}://{parts.netloc}"), max_size=_MAX_FRAME,
                                 open_timeout=15, compression=None)
    except (OSError, TimeoutError, WebSocketException) as exc:  # refused (a bad ticket there) or unreachable
        _log.info("Gateway relay socket to %s%s refused: %s", row["url"], path, type(exc).__name__)
        await ws.close(code=4403)
        return
    await ws.accept(subprotocol=upstream.subprotocol)
    with _live_lock:
        _live[id(ws)] = (weakref.ref(ws), asyncio.get_running_loop(), gateway_id)
    try:
        await _pump(ws, upstream)
    finally:
        with _live_lock:
            _live.pop(id(ws), None)


async def _pump(ws: WebSocket, upstream) -> None:
    from websockets.exceptions import ConnectionClosed

    async def browser_to_gateway() -> None:
        while True:
            message = await ws.receive()
            if message["type"] == "websocket.disconnect":
                code = message.get("code") or 1000
                await upstream.close(code=1000 if code in {1005, 1006} else _sendable(code))
                return
            if message.get("bytes") is not None:
                await upstream.send(message["bytes"])
            elif message.get("text") is not None:
                await upstream.send(message["text"])

    async def gateway_to_browser() -> None:
        try:
            async for message in upstream:
                if isinstance(message, bytes):
                    await ws.send_bytes(message)
                else:
                    await ws.send_text(message)
        except ConnectionClosed:
            pass
        await _close(ws, _sendable(upstream.close_code), upstream.close_reason or "")

    tasks = [asyncio.create_task(browser_to_gateway()), asyncio.create_task(gateway_to_browser())]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        with contextlib.suppress(OSError):
            await upstream.close()
        await asyncio.gather(*tasks, return_exceptions=True)
