"""Remote node server — hosts the Meet bot on another machine (``hermes meet node run``).

WebSocket endpoint accepting token-signed RPC requests dispatched to ``process_manager``.
Token: 32 hex chars minted on first boot, persisted at ``$HERMES_HOME/workspace/meetings/
node_token.json`` so approved gateways survive restarts; the operator copies it to the gateway
via ``hermes meet node approve <name> <url> <token>``. ``websockets`` is imported lazily.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import secrets
import time
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_constants import get_hermes_home
from plugins.google_meet._jsonfile import read_json, write_json_atomic
from plugins.google_meet.node import protocol as _proto

_START_BOT_KEYS = ("url", "guest_name", "duration", "headed", "auth_state", "session_id", "out_dir")


class _RpcError(Exception):
    """Handler-level protocol error; sent verbatim as an error envelope."""


def _rpc_start_bot(payload: Dict[str, Any], pm) -> Dict[str, Any]:
    # Whitelist kwargs we pass through to pm.start.
    kwargs = {k: payload[k] for k in _START_BOT_KEYS if k in payload}
    if "url" not in kwargs:
        raise _RpcError("missing 'url' in payload")
    return pm.start(**kwargs)


def _rpc_say(payload: Dict[str, Any], pm) -> Dict[str, Any]:
    # The bot-side consumer only exists in realtime mode: ok=True means "enqueued", not "spoken".
    text = payload.get("text", "")
    active = pm._read_active()
    enqueued = False
    if active and active.get("out_dir"):
        with contextlib.suppress(OSError):
            queue = Path(active["out_dir"]) / "say_queue.jsonl"
            queue.parent.mkdir(parents=True, exist_ok=True)
            with queue.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"text": text, "ts": time.time()}) + "\n")
            enqueued = True
    return {"ok": True, "enqueued": enqueued, "text": text}


# request type → fn(payload, pm) returning the response payload.
_RPC = {
    "start_bot": _rpc_start_bot,
    "stop": lambda p, pm: pm.stop(reason=p.get("reason", "requested")),
    "status": lambda p, pm: pm.status(),
    "transcript": lambda p, pm: pm.transcript(last=p.get("last")),
    "say": _rpc_say}


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
        tok = data.get("token") if isinstance(data, dict) else None
        if not (isinstance(tok, str) and tok):
            tok = secrets.token_hex(16)  # 32 hex chars
            # Owner-only: the token grants full RPC access to the meet bot.
            write_json_atomic(self.token_path, {"token": tok, "generated_at": time.time()}, mode=0o600)
        self._token = tok
        return tok

    async def _handle_request(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Validate + dispatch one decoded request; always returns an envelope, never raises.
        Envelope ``error`` is for auth/protocol failures and pm crashes; pm's own ``ok``/``error``
        results travel inside a normal response payload."""
        ok, reason = _proto.validate_request(msg, self.ensure_token())
        if not ok:
            return _proto.make_error(str(msg.get("id") or ""), reason)
        req_id, t = msg["id"], msg["type"]
        if t == "ping":
            return {"type": "pong", "id": req_id,
                    "payload": {"display_name": self.display_name, "ts": time.time()}}
        handler = _RPC.get(t)
        if handler is None:
            return _proto.make_error(req_id, f"unhandled type: {t!r}")
        # Import lazily so test mocks can monkeypatch freely.
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
