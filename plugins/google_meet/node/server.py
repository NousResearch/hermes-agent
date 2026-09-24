"""Remote node server — hosts the Meet bot on another machine (``hermes meet node run``).

WebSocket endpoint accepting token-signed RPC requests dispatched to ``process_manager``.
Token: 32 hex chars minted on first boot, persisted at ``$HERMES_HOME/workspace/meetings/
node_token.json`` so approved gateways survive restarts; the operator copies it to the gateway
via ``hermes meet node approve <name> <url> <token>``. ``websockets`` is imported lazily.
"""

from __future__ import annotations

import asyncio
import secrets
import time
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home
from plugins.google_meet._jsonfile import read_json
from plugins.google_meet.node import protocol as _proto
from utils import atomic_json_write

_START_BOT_KEYS = (
    "url",
    "guest_name",
    "duration",
    "headed",
    "persist_after_session",
    "session_id",
    "out_dir",
    "mode",
)


class _RpcError(Exception):
    """Handler-level protocol error; sent verbatim as an error envelope."""


def _rpc_start_bot(payload: Dict[str, Any], pm) -> Dict[str, Any]:
    if "auth_state" in payload:
        raise _RpcError(
            "auth_state is local-only; remote nodes manage their own auth state"
        )
    kwargs = {key: payload[key] for key in _START_BOT_KEYS if key in payload}
    if "url" not in kwargs:
        raise _RpcError("missing 'url' in payload")
    return pm.start(**kwargs)


def _rpc_transcript(payload: Dict[str, Any], pm) -> Dict[str, Any]:
    return pm.transcript(
        last=payload.get("last"),
        include_finished=bool(payload.get("include_finished", False)),
        session_id=payload.get("session_id"),
    )


def _rpc_say(payload: Dict[str, Any], pm) -> Dict[str, Any]:
    return pm.enqueue_say(payload.get("text", ""))


_RPC = {
    "start_bot": _rpc_start_bot,
    "stop": lambda payload, pm: pm.stop(reason=payload.get("reason", "requested")),
    "status": lambda payload, pm: pm.status(),
    "transcript": _rpc_transcript,
    "say": _rpc_say,
}


class NodeServer:
    """WebSocket server that executes meet bot RPCs locally."""

    def __init__(self, host: str = "127.0.0.1", port: int = 18789, token_path: Optional[Path] = None,
                 display_name: str = "hermes-meet-node") -> None:
        self.host = host
        self.port = port
        self.display_name = display_name
        self.token_path = Path(token_path) if token_path is not None else (
            Path(get_hermes_home()) / "workspace" / "meetings" / "node_token.json")
        self._token: Optional[str] = None

    def ensure_token(self) -> str:
        """Return the persisted shared secret, generating one on first use."""
        if self._token:
            return self._token
        data = read_json(self.token_path)
        token = data.get("token") if isinstance(data, dict) else None
        if not (isinstance(token, str) and token):
            token = secrets.token_hex(16)
            atomic_json_write(
                self.token_path,
                {"token": token, "generated_at": time.time()},
                mode=0o600,
            )
        self._token = token
        return token

    async def _handle_request(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Validate + dispatch one decoded request; always returns an envelope, never raises."""
        ok, reason = _proto.validate_request(msg, self.ensure_token())
        if not ok:
            return _proto.make_error(str(msg.get("id") or ""), reason)
        req_id, request_type = msg["id"], msg["type"]
        if request_type == "ping":
            return {
                "type": "pong",
                "id": req_id,
                "payload": {"display_name": self.display_name, "ts": time.time()},
            }
        handler = _RPC.get(request_type)
        if handler is None:
            return _proto.make_error(req_id, f"unhandled type: {request_type!r}")
        from plugins.google_meet import process_manager as pm

        try:
            return _proto.make_response(req_id, handler(msg["payload"], pm))
        except _RpcError as exc:
            return _proto.make_error(req_id, str(exc))
        except Exception as exc:  # noqa: BLE001 — surface any pm crash to client
            return _proto.make_error(req_id, f"{type(exc).__name__}: {exc}")

    async def serve(self) -> None:
        """Run the WebSocket server until cancelled (wrap in ``asyncio.run``)."""
        try:
            import websockets  # type: ignore
        except ImportError as exc:
            raise RuntimeError("NodeServer.serve requires the 'websockets' package. "
                               "Install it with: pip install websockets") from exc
        self.ensure_token()

        async def _handler(ws):
            async for raw in ws:
                try:
                    msg = _proto.decode(raw)
                except ValueError as exc:
                    await ws.send(_proto.encode(_proto.make_error("", f"decode: {exc}")))
                    continue
                await ws.send(_proto.encode(await self._handle_request(msg)))

        async with websockets.serve(_handler, self.host, self.port):
            await asyncio.Future()  # run until cancelled
