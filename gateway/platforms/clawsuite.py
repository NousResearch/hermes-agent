"""
ClawSuite platform adapter.

Provides a WebSocket server that allows ClawSuite (an Electron desktop app)
to connect to hermes-agent as a management frontend.

Protocol overview:

  Connect handshake (client → server):
    {"type":"connect","device":{"id":"<id>","nonce":"<nonce>"},"token":"<token>"}

  Challenge (server → client):
    {"type":"event","event":"connect.challenge","payload":{"nonce":"<nonce>"}}

  Confirmed (server → client):
    {"type":"event","event":"connected","payload":{"sessionKey":"<key>","deviceId":"<id>"}}

  Chat (client → server):
    {"type":"chat","message":"hello","sessionKey":"main"}

  Streaming chunk (server → client):
    {"type":"event","event":"agent","payload":{"stream":"chunk","text":"..."}}

  Done (server → client):
    {"type":"event","event":"agent","payload":{"stream":"done","state":"final","message":{...}}}

Configuration (via environment variables or PlatformConfig.extra):
  CLAWSUITE_PORT          WebSocket server port (default: 18789)
  CLAWSUITE_HOST          Bind address (default: 127.0.0.1)
  CLAWSUITE_TOKEN         Shared secret for client auth (optional)
  CLAWSUITE_ALLOW_ALL     Accept any token / no-token connections (default: true when token unset)

Dependencies:
  pip install websockets
"""

import asyncio
import json
import logging
import os
import uuid
from typing import Any, Dict, Optional, Set

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    WebSocketServerProtocol = Any  # type: ignore

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Dependency check
# ──────────────────────────────────────────────────────────────────────────────

def check_clawsuite_requirements() -> bool:
    """Return True when the websockets library is available."""
    return WEBSOCKETS_AVAILABLE


# ──────────────────────────────────────────────────────────────────────────────
# Adapter
# ──────────────────────────────────────────────────────────────────────────────

class ClawSuiteAdapter(BasePlatformAdapter):
    """
    WebSocket server adapter for ClawSuite desktop frontend.

    Starts a local WebSocket server and manages one or more ClawSuite client
    connections.  Each authenticated client is treated as a separate session
    keyed by its device ID.

    Streaming support:
        The adapter patches the message handler result pipeline to intercept
        per-chunk progress events emitted by the agent and forwards them to
        the connected ClawSuite client as ``stream:chunk`` events before the
        final ``stream:done`` payload.
    """

    DEFAULT_PORT = 18789
    DEFAULT_HOST = "127.0.0.1"

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.CLAWSUITE)

        extra = config.extra or {}

        self._port: int = int(
            extra.get("port", os.getenv("CLAWSUITE_PORT", self.DEFAULT_PORT))
        )
        self._host: str = extra.get(
            "host", os.getenv("CLAWSUITE_HOST", self.DEFAULT_HOST)
        )
        self._secret: Optional[str] = extra.get(
            "token", os.getenv("CLAWSUITE_TOKEN")
        )
        # When no token is configured we allow unauthenticated connections
        # (local-only desktop use-case).  Set CLAWSUITE_ALLOW_ALL=false and
        # CLAWSUITE_TOKEN=<secret> to require authentication.
        allow_all_env = os.getenv("CLAWSUITE_ALLOW_ALL", "").lower()
        if allow_all_env in ("false", "0", "no"):
            self._allow_all = False
        else:
            # Default: allow all when no token is configured
            self._allow_all = self._secret is None

        # Active WebSocket connections keyed by device_id
        self._connections: Dict[str, "WebSocketServerProtocol"] = {}
        # Reverse map: ws → device_id (for cleanup)
        self._ws_to_device: Dict[int, str] = {}  # id(ws) → device_id

        # Unauthenticated websockets waiting for their connect message
        self._pending_ws: Set["WebSocketServerProtocol"] = set()

        self._server: Any = None  # websockets.WebSocketServer

        # Per-session streaming sink: session_key → callable that sends a chunk
        # The agent loop calls this (if registered) for each streaming token.
        self._stream_sinks: Dict[str, Any] = {}

    # ──────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────

    async def connect(self) -> bool:
        """Start the WebSocket server and begin accepting connections."""
        if not WEBSOCKETS_AVAILABLE:
            logger.error(
                "[%s] 'websockets' package not installed. "
                "Run: pip install websockets",
                self.name,
            )
            return False

        try:
            self._server = await websockets.serve(
                self._handle_connection,
                self._host,
                self._port,
            )
            self._running = True
            logger.info(
                "[%s] WebSocket server listening on ws://%s:%d",
                self.name, self._host, self._port,
            )
            print(
                f"[{self.name}] Listening on ws://{self._host}:{self._port} "
                f"(auth={'token' if self._secret else 'open'})"
            )
            return True
        except Exception as exc:
            logger.error("[%s] Failed to start WebSocket server: %s", self.name, exc)
            return False

    async def disconnect(self) -> None:
        """Stop the WebSocket server and close all active connections."""
        self._running = False

        # Close all authenticated connections
        for ws in list(self._connections.values()):
            try:
                await ws.close(1001, "Gateway shutting down")
            except Exception:
                pass

        # Close pending unauthenticated connections
        for ws in list(self._pending_ws):
            try:
                await ws.close(1001, "Gateway shutting down")
            except Exception:
                pass

        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        self._connections.clear()
        self._ws_to_device.clear()
        self._pending_ws.clear()
        logger.info("[%s] Disconnected", self.name)

    # ──────────────────────────────────────────────────────────────────────
    # Sending
    # ──────────────────────────────────────────────────────────────────────

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """
        Send a complete text message to a ClawSuite client.

        ``chat_id`` maps to the ClawSuite device ID (also used as session key).
        Streams are sent as a single ``stream:done`` event when content is the
        final agent response, or as individual ``stream:chunk`` events when
        content arrives via :py:meth:`send_stream_chunk`.
        """
        ws = self._connections.get(chat_id)
        if ws is None:
            return SendResult(success=False, error=f"No connection for device '{chat_id}'")

        msg_id = str(uuid.uuid4())
        payload = {
            "type": "event",
            "event": "agent",
            "payload": {
                "stream": "done",
                "state": "final",
                "message": {
                    "id": msg_id,
                    "text": content,
                    "role": "assistant",
                    "replyTo": reply_to,
                    **(metadata or {}),
                },
            },
        }
        try:
            await ws.send(json.dumps(payload))
            return SendResult(success=True, message_id=msg_id)
        except Exception as exc:
            logger.warning("[%s] Failed to send to %s: %s", self.name, chat_id, exc)
            return SendResult(success=False, error=str(exc))

    async def send_stream_chunk(self, chat_id: str, text: str) -> None:
        """
        Send a streaming text chunk to a ClawSuite client.

        Called by the streaming integration layer (see :py:meth:`handle_message`)
        for each token/chunk produced by the agent.
        """
        ws = self._connections.get(chat_id)
        if ws is None:
            return
        payload = {
            "type": "event",
            "event": "agent",
            "payload": {
                "stream": "chunk",
                "text": text,
            },
        }
        try:
            await ws.send(json.dumps(payload))
        except Exception as exc:
            logger.debug("[%s] Chunk send failed for %s: %s", self.name, chat_id, exc)

    async def send_typing(self, chat_id: str) -> None:
        """Send a typing indicator to ClawSuite."""
        ws = self._connections.get(chat_id)
        if ws is None:
            return
        try:
            await ws.send(json.dumps({
                "type": "event",
                "event": "agent.typing",
                "payload": {},
            }))
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────────────
    # Chat info
    # ──────────────────────────────────────────────────────────────────────

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return basic info for a ClawSuite session."""
        connected = chat_id in self._connections
        return {
            "name": f"ClawSuite ({chat_id})",
            "type": "dm",
            "connected": connected,
        }

    # ──────────────────────────────────────────────────────────────────────
    # WebSocket connection handler
    # ──────────────────────────────────────────────────────────────────────

    async def _handle_connection(
        self, ws: "WebSocketServerProtocol", path: str = "/"
    ) -> None:
        """
        Handle the full lifecycle of a single WebSocket connection.

        Phase 1 – handshake: client sends connect message, server validates
                  token and responds with challenge → confirmed.
        Phase 2 – messaging: client sends chat messages, server routes them
                  through the hermes-agent message handler.
        """
        remote = getattr(ws, "remote_address", ("?", 0))
        logger.info("[%s] New connection from %s:%s", self.name, *remote)
        self._pending_ws.add(ws)

        device_id: Optional[str] = None

        try:
            # ── Phase 1: handshake ────────────────────────────────────────
            device_id = await self._do_handshake(ws)
            if device_id is None:
                # Handshake failed — ws already closed inside _do_handshake
                return

            self._pending_ws.discard(ws)
            self._connections[device_id] = ws
            self._ws_to_device[id(ws)] = device_id

            print(f"[{self.name}] Device '{device_id}' authenticated and connected")

            # ── Phase 2: message loop ─────────────────────────────────────
            async for raw in ws:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    await self._send_error(ws, "invalid_json", "Message must be valid JSON")
                    continue

                await self._dispatch_client_message(ws, device_id, data)

        except websockets.exceptions.ConnectionClosedOK:
            logger.info("[%s] Device '%s' disconnected cleanly", self.name, device_id or "?")
        except websockets.exceptions.ConnectionClosedError as exc:
            logger.warning("[%s] Device '%s' connection error: %s", self.name, device_id or "?", exc)
        except Exception as exc:
            logger.exception("[%s] Unexpected error for device '%s': %s", self.name, device_id or "?", exc)
        finally:
            self._pending_ws.discard(ws)
            if device_id and device_id in self._connections:
                del self._connections[device_id]
            if id(ws) in self._ws_to_device:
                del self._ws_to_device[id(ws)]
            logger.info("[%s] Cleaned up connection for device '%s'", self.name, device_id or "?")

    async def _do_handshake(self, ws: "WebSocketServerProtocol") -> Optional[str]:
        """
        Perform the ClawSuite connect handshake.

        Returns the authenticated device_id on success, or None on failure
        (the WebSocket will have been closed already).
        """
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=15.0)
        except asyncio.TimeoutError:
            await ws.close(1008, "Handshake timeout")
            return None
        except Exception as exc:
            logger.warning("[%s] Handshake recv failed: %s", self.name, exc)
            return None

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await ws.close(1007, "Invalid JSON in handshake")
            return None

        if msg.get("type") != "connect":
            await ws.close(1002, "Expected connect message")
            return None

        # Validate auth token
        provided_token = msg.get("token", "")
        if not self._allow_all:
            if not self._secret or provided_token != self._secret:
                await self._send_error(ws, "auth_failed", "Invalid token")
                await ws.close(1008, "Authentication failed")
                return None

        # Extract device info
        device = msg.get("device", {})
        device_id: str = device.get("id") or str(uuid.uuid4())
        client_nonce: str = device.get("nonce", str(uuid.uuid4()))

        # Send challenge
        await ws.send(json.dumps({
            "type": "event",
            "event": "connect.challenge",
            "payload": {"nonce": client_nonce},
        }))

        # Send confirmed
        await ws.send(json.dumps({
            "type": "event",
            "event": "connected",
            "payload": {
                "sessionKey": device_id,
                "deviceId": device_id,
                "server": "hermes-agent",
                "version": "1.0",
            },
        }))

        return device_id

    async def _dispatch_client_message(
        self,
        ws: "WebSocketServerProtocol",
        device_id: str,
        data: Dict[str, Any],
    ) -> None:
        """Route an authenticated client message to the appropriate handler."""
        msg_type = data.get("type", "")

        if msg_type == "chat":
            await self._handle_chat(ws, device_id, data)
        elif msg_type == "ping":
            await ws.send(json.dumps({"type": "pong"}))
        elif msg_type == "disconnect":
            await ws.close(1000, "Client requested disconnect")
        else:
            logger.debug(
                "[%s] Unknown message type '%s' from device '%s'",
                self.name, msg_type, device_id,
            )

    async def _handle_chat(
        self,
        ws: "WebSocketServerProtocol",
        device_id: str,
        data: Dict[str, Any],
    ) -> None:
        """
        Process an incoming chat message from ClawSuite.

        Builds a :class:`~gateway.platforms.base.MessageEvent`, installs a
        streaming sink so chunks are forwarded live, then delegates to the
        base class :py:meth:`handle_message` pipeline (which calls the
        registered hermes-agent message handler and manages interrupts).
        """
        text: str = data.get("message", "").strip()
        session_key: str = data.get("sessionKey", device_id)

        if not text:
            await self._send_error(ws, "empty_message", "Message cannot be empty")
            return

        # Install streaming sink for this session so the agent loop can push chunks
        async def _chunk_sink(chunk_text: str) -> None:
            await self.send_stream_chunk(device_id, chunk_text)

        self._stream_sinks[session_key] = _chunk_sink

        source = self.build_source(
            chat_id=device_id,
            chat_name=f"ClawSuite:{device_id[:8]}",
            chat_type="dm",
            user_id=device_id,
            user_name=f"ClawSuite ({device_id[:8]})",
        )

        event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT if not text.startswith("/") else MessageType.COMMAND,
            source=source,
            raw_message=data,
            message_id=data.get("messageId", str(uuid.uuid4())),
        )

        # Delegate to base class handler (manages interrupts, typing, response send)
        await self.handle_message(event)

    # ──────────────────────────────────────────────────────────────────────
    # Streaming integration
    # ──────────────────────────────────────────────────────────────────────

    def get_stream_sink(self, session_key: str):
        """
        Return the streaming chunk sink for a session.

        The hermes-agent runner can call this to push streaming tokens to
        the connected ClawSuite client as they arrive, giving users live
        feedback during long completions.

        Example (in agent runner)::

            sink = adapter.get_stream_sink(session_key)
            if sink:
                async for chunk in llm_stream:
                    await sink(chunk)
        """
        return self._stream_sinks.get(session_key)

    def clear_stream_sink(self, session_key: str) -> None:
        """Remove the streaming sink for a completed session."""
        self._stream_sinks.pop(session_key, None)

    # ──────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    async def _send_error(
        ws: "WebSocketServerProtocol",
        code: str,
        message: str,
    ) -> None:
        """Send a structured error event to the client."""
        try:
            await ws.send(json.dumps({
                "type": "event",
                "event": "error",
                "payload": {"code": code, "message": message},
            }))
        except Exception:
            pass
