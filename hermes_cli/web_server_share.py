"""The tailcat share listener: a second loopback socket in front of the same app.

Everything ``tailcat serve`` forwards arrives from 127.0.0.1, so the share must
not reuse the main listener, whose loopback trust (the headless ``GET /`` token
page, loopback-only WS peers) assumes the caller is on this machine. This gate
admits exactly two things:

* ``POST /api/share/pair`` — redeem a one-time connection code for a device token;
* ``/api/...`` requests and WebSockets carrying a paired device's token.

A paired request is re-credentialed with the process session token before it
reaches the app, so every existing route and WS handler authorizes it through
the path it already has. The device token itself never leaves this module.
Revoking a device closes its open WebSockets within one poll interval.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.parse
from typing import Callable, Optional

from hermes_cli import tailcat_share_store as store

log = logging.getLogger(__name__)

PAIR_PATH = "/api/share/pair"
_TOKEN_HEADER = b"x-hermes-session-token"
_STRIPPED_HEADERS = {_TOKEN_HEADER, b"authorization", b"cookie", b"host"}
_STRIPPED_QUERY = {"token", "ticket", "internal"}
_REVOKE_POLL_S = 2.0
_MAX_PAIR_BODY = 4096
WS_CLOSE_REVOKED = 4403


def _header(scope: dict, name: bytes) -> str:
    for key, value in scope.get("headers") or ():
        if key.lower() == name:
            return value.decode("latin-1")
    return ""


def presented_token(scope: dict) -> str:
    """The credential a client presented: header, Bearer, or ``?token=`` (WebSocket)."""
    token = _header(scope, _TOKEN_HEADER)
    if token:
        return token
    auth = _header(scope, b"authorization")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    query = urllib.parse.parse_qs(scope.get("query_string", b"").decode("latin-1"))
    return (query.get("token") or [""])[0]


def recredential(scope: dict, session_token: str, host: str) -> dict:
    """Copy of ``scope`` carrying the process token instead of whatever the client sent."""
    headers = [(k, v) for k, v in scope.get("headers") or () if k.lower() not in _STRIPPED_HEADERS]
    headers.append((b"host", host.encode("latin-1")))
    headers.append((_TOKEN_HEADER, session_token.encode("latin-1")))
    pairs = [(k, v) for k, v in urllib.parse.parse_qsl(
        scope.get("query_string", b"").decode("latin-1"), keep_blank_values=True) if k not in _STRIPPED_QUERY]
    if scope["type"] == "websocket":
        pairs.append(("token", session_token))
    return {**scope, "headers": headers, "query_string": urllib.parse.urlencode(pairs).encode("latin-1")}


async def _json(send, status: int, body: dict) -> None:
    payload = json.dumps(body).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"),
                            (b"cache-control", b"no-store"),
                            (b"content-length", str(len(payload)).encode())]})
    await send({"type": "http.response.body", "body": payload})


async def _read_body(receive, limit: int) -> Optional[bytes]:
    body = b""
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return None
        body += message.get("body", b"")
        if len(body) > limit:
            return None
        if not message.get("more_body"):
            return body


class ShareGate:
    """ASGI app for the share listener. ``session_token`` / ``upstream_host`` are read per request."""

    def __init__(self, app, *, session_token: Callable[[], str], upstream_host: Callable[[], str]):
        self.app = app
        self.session_token = session_token
        self.upstream_host = upstream_host

    async def __call__(self, scope, receive, send):
        kind = scope["type"]
        if kind not in ("http", "websocket"):
            return
        path = scope.get("path", "")
        if kind == "http" and path == PAIR_PATH:
            return await self._pair(scope, receive, send)
        device = None
        if path.startswith("/api/") and not path.startswith("/api/share/"):
            device = store.device_for_token(presented_token(scope))
        if device is None:
            if kind == "websocket":
                await send({"type": "websocket.close", "code": 1008})
            else:
                await _json(send, 401 if path.startswith("/api/") else 404, {"detail": "Unauthorized"})
            return
        store.touch_device(device["id"])
        inner = recredential(scope, self.session_token(), self.upstream_host())
        if kind == "http":
            return await self.app(inner, receive, send)
        return await self._websocket(device["id"], inner, receive, send)

    async def _websocket(self, device_id: str, scope, receive, send) -> None:
        """Run the app's handler, cancelling it once the device is revoked."""
        task = asyncio.ensure_future(self.app(scope, receive, send))
        while not task.done():
            done, _ = await asyncio.wait({task}, timeout=_REVOKE_POLL_S)
            if done:
                break
            if not any(d.get("id") == device_id for d in store.load_devices()):
                log.info("tailcat share: device %s revoked; closing its WebSocket", device_id)
                task.cancel()
                try:
                    await task
                except BaseException:  # noqa: BLE001 — the handler's own teardown already ran
                    pass
                try:
                    await send({"type": "websocket.close", "code": WS_CLOSE_REVOKED})
                except Exception:  # noqa: BLE001 — the socket may already be gone
                    pass
                return
        task.result()

    async def _pair(self, scope, receive, send) -> None:
        if scope.get("method") != "POST":
            return await _json(send, 405, {"detail": "Method not allowed"})
        raw = await _read_body(receive, _MAX_PAIR_BODY)
        try:
            body = json.loads(raw or b"")
        except ValueError:
            body = None
        if not isinstance(body, dict):
            return await _json(send, 400, {"detail": "Expected a JSON object"})
        secret = str(body.get("code") or "")
        parsed = store.parse_code(secret)
        if parsed is not None:
            secret = parsed.secret
        if not secret or not store.consume_code(secret):
            # Uniform answer and a small delay: a code is 192 random bits, so
            # this only keeps a misbehaving client from spinning.
            await asyncio.sleep(0.5)
            return await _json(send, 403, {"detail": "This connection code is invalid, expired, or already used."})
        device, token = store.pair_device(str(body.get("name") or ""))
        log.info("tailcat share: paired device %s (%s)", device["id"], device["name"])
        await _json(send, 200, {"device": device, "token": token, "paired_at": time.time()})
