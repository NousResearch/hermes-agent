"""Transport-only RPC client. Closing a viewer never stops its authority."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
import json
import os
from pathlib import Path
import socket
import time


class GatewayClientError(ValueError):
    pass


class GatewayClient:
    def __init__(self, websocket):
        self.websocket = websocket
        self.events = asyncio.Queue(maxsize=4096)
        self.pending = {}
        self.sequence = 0
        self.reader = None

    async def __aenter__(self):
        self.reader = asyncio.create_task(self._read())
        return self

    async def __aexit__(self, *exc):
        self.reader.cancel()
        with suppress(asyncio.CancelledError):
            await self.reader

    async def _read(self):
        try:
            async for raw in self.websocket:
                frame = json.loads(raw)
                future = self.pending.get(frame.get("id"))
                if future is not None and not future.done():
                    if "error" in frame:
                        # Only bounded authority reason codes, never raw remote diagnostics.
                        reason = frame["error"].get("message", "request_failed")
                        if not isinstance(reason, str) or not reason.replace("_", "").isalnum() or len(reason) > 80:
                            reason = "request_failed"
                        future.set_exception(GatewayClientError(reason))
                    else:
                        future.set_result(frame.get("result", {}))
                elif "method" in frame:
                    self.events.put_nowait(frame)
        except (OSError, ValueError, asyncio.QueueFull):
            pass
        finally:
            error = GatewayClientError("Gateway disconnected; accepted work was not cancelled. Resume the printed session ID.")
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(error)
            # A stalled consumer is failed rather than silently losing terminal events.
            if self.events.full():
                self.events.get_nowait()
            self.events.put_nowait(error)

    async def rpc(self, method, **params):
        self.sequence += 1
        rid = self.sequence
        future = asyncio.get_running_loop().create_future()
        self.pending[rid] = future
        try:
            await self.websocket.send(json.dumps({"jsonrpc": "2.0", "id": rid, "method": method, "params": params}))
            return await asyncio.wait_for(future, 30)
        finally:
            self.pending.pop(rid, None)


def _session_ticket(home: Path, endpoint, *, purpose="interactive") -> str:
    from hermes_cli.gateway_runtime import control_home_for
    from hermes_cli.gateway_runtime_discovery import _socket_path, _identify_response
    # A served secondary's ticket is minted by the multiplexer's socket, bound to the secondary.
    home = control_home_for(home, endpoint)
    request = json.dumps({"protocol": 1, "id": 1, "verb": "session-ticket", "params": {
        "profile_id": endpoint.profile_id, "instance_id": endpoint.instance_id, "purpose": purpose,
    }}).encode() + b"\n"
    if os.name == "nt":
        from gateway.runtime_bootstrap_windows import query_runtime_control
        data = query_runtime_control(home, request, 5)
    else:
        deadline = time.monotonic() + 5
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
            peer.settimeout(5)
            peer.connect(str(_socket_path(home)))
            peer.sendall(request)
            data = bytearray()
            while b"\n" not in data:
                budget = deadline - time.monotonic()
                if budget <= 0:
                    raise GatewayClientError("Gateway bootstrap timed out")
                peer.settimeout(budget)
                chunk = peer.recv(4096)
                if not chunk or len(data) + len(chunk) > 65536:
                    raise GatewayClientError("Invalid gateway bootstrap response")
                data.extend(chunk)
    grant = _identify_response(bytes(data))
    if any(grant.get(k) != v for k, v in {
        "profile_id": endpoint.profile_id, "instance_id": endpoint.instance_id, "runtime_protocol": 1,
    }.items()) or not isinstance(grant.get("ticket"), str) or not grant["ticket"]:
        raise GatewayClientError("Gateway bootstrap identity changed; retry launch")
    return grant["ticket"]


@asynccontextmanager
async def connect_gateway():
    from websockets.asyncio.client import connect
    from hermes_constants import get_hermes_home
    from hermes_cli.gateway_runtime import ensure_gateway_runtime
    from urllib.parse import urlsplit

    remote = os.environ.get("HERMES_TUI_GATEWAY_URL", "").strip()
    protocols = None
    if remote:
        parsed = urlsplit(remote)
        if parsed.scheme not in {"ws", "wss"} or not parsed.hostname:
            raise GatewayClientError("Invalid explicit gateway WebSocket URL; no local fallback")
        url = remote
    else:
        home = get_hermes_home().resolve()
        result = await asyncio.to_thread(ensure_gateway_runtime, home)
        if result.state != "ready" or result.endpoint is None:
            raise GatewayClientError(f"Gateway {result.state}: {result.reason_code or 'not_ready'}")
        endpoint = result.endpoint
        ticket = await asyncio.to_thread(_session_ticket, home, endpoint)
        url = endpoint.api_origin.replace("https:", "wss:").replace("http:", "ws:") + "/api/ws"
        protocols = ["hermes-gateway-v1", "hermes-gateway-ticket." + ticket]
    try:
        async with connect(url, subprotocols=protocols, open_timeout=10, max_size=8 * 1024 * 1024) as ws:
            if protocols and ws.subprotocol != "hermes-gateway-v1":
                raise GatewayClientError("Gateway protocol mismatch; update/restart required")
            async with GatewayClient(ws) as client:
                yield client
    except GatewayClientError:
        raise
    except (OSError, TimeoutError) as exc:
        raise GatewayClientError("Gateway connection failed; no local fallback") from exc
