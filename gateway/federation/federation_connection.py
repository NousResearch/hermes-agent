"""Federation WebSocket connection manager.

Manages persistent WebSocket connections to all known peers.
Each peer gets a bidirectional connection with:
- Automatic reconnection with exponential backoff
- Liveness monitoring (ping/pong)
- Message routing (broadcast + directed)
- Auth token verification
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Optional
from gateway.config import FederationConfig
from gateway.federation.federation_protocol import (
    FedMessage,
    MessageType,
    PeerInfo,
)

logger = logging.getLogger(__name__)


class FederationConnectionManager:
    """Manages WebSocket connections to all federation peers."""

    def __init__(
        self,
        device_id: str,
        auth_token: Optional[str] = None,
        ws_port: int = 18765,
        on_message: Optional[Callable[[FedMessage], None]] = None,
        on_peer_join: Optional[Callable[[PeerInfo], None]] = None,
        on_peer_leave: Optional[Callable[[str], None]] = None,
    ):
        self.device_id = device_id
        self.auth_token = auth_token
        self.ws_port = ws_port
        self._on_message = on_message
        self._on_peer_join = on_peer_join
        self._on_peer_leave = on_peer_leave

        # Connection state
        self._ws_connections: dict[str, Any] = {}  # device_id -> websocket
        self._peer_infos: dict[str, PeerInfo] = {}  # device_id -> PeerInfo
        self._reconnect_tasks: dict[str, asyncio.Task] = {}
        self._ping_tasks: dict[str, asyncio.Task] = {}

        # Server state (if we also listen for inbound connections)
        self._server: Optional[Any] = None
        self._running = False

    # ----------------------------------------------------------------
    # Peer management
    # ----------------------------------------------------------------

    def register_peer(self, info: PeerInfo) -> None:
        """Register a peer (from config or mDNS discovery)."""
        if info.device_id == self.device_id:
            return  # Skip self
        self._peer_infos[info.device_id] = info
        logger.info(
            "Federation: registered peer %s (%s) at %s",
            info.device_id, info.hostname, info.ws_url,
        )

    def unregister_peer(self, device_id: str, reason: str = "offline") -> None:
        """Unregister a peer and clean up its connection."""
        if device_id in self._peer_infos:
            del self._peer_infos[device_id]
        self._close_connection(device_id)
        if self._on_peer_leave:
            self._on_peer_leave(device_id)
        logger.info("Federation: peer %s left (%s)", device_id, reason)

    def get_peers(self) -> list[PeerInfo]:
        """Return all currently connected peers."""
        return [
            info for info in self._peer_infos.values()
            if info.device_id in self._ws_connections
        ]

    def get_peer(self, device_id: str) -> Optional[PeerInfo]:
        """Get info for a specific peer."""
        return self._peer_infos.get(device_id)

    def get_online_count(self) -> int:
        """Count of peers with active WebSocket connections."""
        return len(self._ws_connections)

    # ----------------------------------------------------------------
    # Connection lifecycle
    # ----------------------------------------------------------------

    async def start(self, listen: bool = True) -> None:
        """Start connection manager — connect to known peers + optionally listen."""
        self._running = True

        # Connect to configured peers
        for info in self._peer_infos.values():
            if info.ws_url:
                asyncio.create_task(self._connect_to_peer(info))

        # Start ping monitor for all connected peers
        asyncio.create_task(self._ping_monitor_loop())

        # Start listening for inbound connections
        if listen:
            await self._start_server()

        logger.info(
            "Federation: connection manager started (device=%s, port=%d)",
            self.device_id, self.ws_port,
        )

    async def stop(self) -> None:
        """Shut down all connections and stop listening."""
        self._running = False

        # Broadcast PEER_LEAVE
        await self._broadcast(
            FedMessage.peer_leave(self.device_id, reason="shutdown")
        )

        # Cancel all tasks
        for task in list(self._reconnect_tasks.values()):
            task.cancel()
        for task in list(self._ping_tasks.values()):
            task.cancel()

        # Close all connections
        for device_id in list(self._ws_connections):
            self._close_connection(device_id)

        # Stop server
        if self._server:
            self._server.close()
            await self._server.wait_closed()

        logger.info("Federation: connection manager stopped")

    # ----------------------------------------------------------------
    # Message sending
    # ----------------------------------------------------------------

    async def send(self, message: FedMessage) -> bool:
        """Send a message to a specific peer or broadcast."""
        if message.sender_id != self.device_id:
            message.sender_id = self.device_id
        if self.auth_token:
            message.sign(self.auth_token)

        if message.target_id:
            # Directed send
            return await self._send_to_peer(message.target_id, message)
        else:
            # Broadcast
            return await self._broadcast(message)

    async def _send_to_peer(self, device_id: str, message: FedMessage) -> bool:
        """Send a message to a specific peer."""
        ws = self._ws_connections.get(device_id)
        if not ws:
            logger.warning(
                "Federation: no connection to peer %s, message queued",
                device_id,
            )
            return False
        try:
            await ws.send(message.to_json())
            return True
        except Exception as e:
            logger.warning(
                "Federation: failed to send to %s: %s", device_id, e,
            )
            self._close_connection(device_id)
            return False

    async def _broadcast(self, message: FedMessage) -> bool:
        """Broadcast a message to all connected peers."""
        results = []
        for device_id in list(self._ws_connections):
            if device_id == self.device_id:
                continue
            ok = await self._send_to_peer(device_id, message)
            results.append(ok)
        return any(results)

    # ----------------------------------------------------------------
    # Connection internals
    # ----------------------------------------------------------------

    async def _connect_to_peer(self, info: PeerInfo) -> None:
        """Connect to a single peer with automatic reconnection."""
        max_retries = 10
        retry = 0

        while self._running and retry < max_retries:
            try:
                import websockets
                ws = await asyncio.wait_for(
                    websockets.connect(info.ws_url),
                    timeout=10,
                )
                self._ws_connections[info.device_id] = ws
                info.status = "online"
                info.last_seen = time.time()
                logger.info(
                    "Federation: connected to %s (%s)",
                    info.device_id, info.ws_url,
                )

                # Announce ourselves
                join_msg = FedMessage.peer_join(self.device_id, self._get_self_info())
                if self.auth_token:
                    join_msg.sign(self.auth_token)
                await ws.send(join_msg.to_json())

                # Start message receiver
                asyncio.create_task(self._receive_loop(info.device_id, ws))

                # Remove reconnect task
                self._reconnect_tasks.pop(info.device_id, None)
                return

            except Exception as e:
                retry += 1
                delay = min(2 ** retry, 60)
                logger.warning(
                    "Federation: connection to %s failed (attempt %d/%d), retrying in %ds: %s",
                    info.device_id, retry, max_retries, delay, e,
                )
                await asyncio.sleep(delay)

        logger.error(
            "Federation: exhausted retries connecting to %s", info.device_id,
        )

    async def _receive_loop(self, device_id: str, ws: Any) -> None:
        """Receive messages from a peer connection."""
        try:
            async for raw in ws:
                try:
                    msg = FedMessage.from_json(raw)

                    # Verify signature
                    if self.auth_token and not msg.verify(self.auth_token):
                        logger.warning(
                            "Federation: invalid signature from %s", device_id,
                        )
                        continue

                    # Check expiry
                    if msg.is_expired():
                        continue

                    # Update peer liveness
                    if device_id in self._peer_infos:
                        self._peer_infos[device_id].last_seen = time.time()

                    # Handle peer join/leave
                    if msg.msg_type == MessageType.PEER_JOIN.value:
                        peer_data = msg.payload.get("peer_info", {})
                        if peer_data.get("device_id") != self.device_id:
                            info = PeerInfo(**peer_data)
                            info.last_seen = time.time()
                            self._peer_infos[info.device_id] = info
                            if self._on_peer_join:
                                self._on_peer_join(info)
                    elif msg.msg_type == MessageType.PEER_LEAVE.value:
                        self.unregister_peer(device_id, msg.payload.get("reason", "offline"))
                        continue

                    # Route to handler
                    if self._on_message:
                        msg._sender_ws_url = self._peer_infos.get(device_id, PeerInfo(device_id=device_id, hostname="")).ws_url
                        self._on_message(msg)

                except Exception as e:
                    logger.warning(
                        "Federation: failed to parse message from %s: %s",
                        device_id, e,
                    )
        except Exception as e:
            logger.info("Federation: connection to %s closed: %s", device_id, e)
        finally:
            self._close_connection(device_id)
            # Schedule reconnect
            if self._running:
                info = self._peer_infos.get(device_id)
                if info:
                    self._reconnect_tasks[device_id] = asyncio.create_task(
                        self._connect_to_peer(info)
                    )

    def _close_connection(self, device_id: str) -> None:
        """Close a WebSocket connection."""
        ws = self._ws_connections.pop(device_id, None)
        if ws:
            try:
                asyncio.create_task(ws.close())
            except Exception:
                pass

    async def _start_server(self) -> None:
        """Start listening for inbound peer connections."""
        import websockets

        async def handler(ws, path=None):
            # Wait for first message to identify the peer
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=30)
                msg = FedMessage.from_json(raw)

                device_id = msg.sender_id
                if not device_id:
                    await ws.close()
                    return

                if self.auth_token and not msg.verify(self.auth_token):
                    await ws.close()
                    return

                self._ws_connections[device_id] = ws

                # Update peer info
                if msg.msg_type == MessageType.PEER_JOIN.value:
                    peer_data = msg.payload.get("peer_info", {})
                    info = PeerInfo(**peer_data)
                    info.last_seen = time.time()
                    self._peer_infos[device_id] = info
                    if self._on_peer_join:
                        self._on_peer_join(info)

                # Start receive loop from this point
                await self._receive_loop(device_id, ws)

            except Exception as e:
                logger.warning("Federation: inbound handler error: %s", e)
                try:
                    await ws.close()
                except Exception:
                    pass

        try:
            self._server = await websockets.serve(
                handler, "0.0.0.0", self.ws_port,
                ping_interval=30,
                ping_timeout=10,
            )
            logger.info(
                "Federation: listening on port %d", self.ws_port,
            )
        except OSError as e:
            logger.warning(
                "Federation: could not listen on port %d: %s",
                self.ws_port, e,
            )

    async def _ping_monitor_loop(self) -> None:
        """Periodically check peer liveness."""
        while self._running:
            await asyncio.sleep(15)

            now = time.time()
            for device_id, info in list(self._peer_infos.items()):
                if device_id in self._ws_connections:
                    # Connection alive, update last_seen
                    info.last_seen = now
                elif now - info.last_seen > 60:
                    # No connection for 60s, consider offline
                    info.status = "offline"

    def _get_self_info(self) -> PeerInfo:
        """Build PeerInfo for this device."""
        import socket
        import os

        hostname = socket.gethostname()
        ip = ""
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
        except Exception:
            pass

        load_avg = 0.0
        try:
            load_avg = round(os.getloadavg()[0], 2)
        except Exception:
            pass

        return PeerInfo(
            device_id=self.device_id,
            hostname=hostname,
            ws_url=f"ws://{ip}:{self.ws_port}",
            cpu_cores=os.cpu_count() or 0,
            memory_gb=0.0,  # Would need psutil
            load_avg=load_avg,
        )
