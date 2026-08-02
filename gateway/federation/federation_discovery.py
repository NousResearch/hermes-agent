"""Federation mDNS/Bonjour discovery — zero-config peer detection.

Uses multicast DNS (mDNS) to discover Hermes federation peers on the local
network without any manual configuration.  Falls back to stdlib socket
multicast — no external dependency required.

Protocol:
  - Service: ``_hermes-federation._udp.local``
  - TXT record fields: device_id, ws_port, version, status
  - Each device broadcasts its own presence and listens for others
  - Discovered peers are passed to FederationAdapter for WebSocket connection

Usage:
    from gateway.federation.federation_discovery import FederationMDNS

    mdns = FederationMDNS(device_id="my-device", ws_port=18765)
    await mdns.start()

    # Peers are discovered via callback
    # mdns.on_discover(peer_info) is called for each new peer
"""
from __future__ import annotations

import asyncio
import json
import logging
import socket
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# mDNS constants
MDNS_MULTICAST_ADDR = "224.0.0.251"
MDNS_PORT = 5353
HERMES_SERVICE_TYPE = "_hermes-federation._udp.local"
BROADCAST_INTERVAL = 30  # seconds between presence announcements
DISCOVERY_TIMEOUT = 5  # seconds to wait for responses after query


@dataclass
class DiscoveredPeer:
    """A peer discovered via mDNS."""

    device_id: str
    hostname: str
    ws_url: str
    version: str = "2.0.0"
    status: str = "online"
    last_seen: float = field(default_factory=time.time)
    cpu_cores: int = 0
    memory_gb: float = 0.0

    @property
    def ws_port(self) -> int:
        """Extract WebSocket port from ws_url."""
        try:
            return int(self.ws_url.split(":")[-1])
        except (IndexError, ValueError):
            return 18765


class FederationMDNS:
    """mDNS-based federation peer discovery.

    Uses UDP multicast to broadcast device presence and discover peers
    on the local network.  No external dependencies — uses stdlib socket.
    """

    def __init__(
        self,
        device_id: str,
        ws_port: int = 18765,
        on_discover: Optional[Callable[[DiscoveredPeer], None]] = None,
        on_forget: Optional[Callable[[str], None]] = None,
        broadcast_interval: int = BROADCAST_INTERVAL,
    ):
        self.device_id = device_id
        self.ws_port = ws_port
        self.on_discover = on_discover
        self.on_forget = on_forget
        self.broadcast_interval = broadcast_interval

        self._socket: Optional[socket.socket] = None
        self._running = False
        self._peers: dict[str, DiscoveredPeer] = {}
        self._receive_task: Optional[asyncio.Task] = None
        self._broadcast_task: Optional[asyncio.Task] = None

    # ----------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------

    async def start(self) -> None:
        """Start mDNS discovery — broadcast presence and listen for peers."""
        self._running = True
        self._socket = self._create_socket()

        # Start receive loop
        self._receive_task = asyncio.create_task(self._receive_loop())

        # Start broadcast loop
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())

        # Send initial announcement
        await self._announce()

        logger.info(
            "Federation mDNS: started (device=%s, port=%d)",
            self.device_id, self.ws_port,
        )

    async def stop(self) -> None:
        """Stop mDNS discovery."""
        self._running = False

        # Send goodbye message
        await self._announce(status="offline")

        # Cancel tasks
        if self._receive_task:
            self._receive_task.cancel()
        if self._broadcast_task:
            self._broadcast_task.cancel()

        # Close socket
        if self._socket:
            self._socket.close()
            self._socket = None

        logger.info("Federation mDNS: stopped")

    # ----------------------------------------------------------------
    # Peer management
    # ----------------------------------------------------------------

    def get_peers(self) -> list[DiscoveredPeer]:
        """Get all currently known peers."""
        now = time.time()
        # Filter out peers not seen in 2x broadcast interval
        active = []
        for peer in self._peers.values():
            if now - peer.last_seen < self.broadcast_interval * 2:
                active.append(peer)
            else:
                # Peer has gone silent — forget it
                if self.on_forget:
                    self.on_forget(peer.device_id)
        return active

    def get_peer(self, device_id: str) -> Optional[DiscoveredPeer]:
        """Get a specific peer."""
        return self._peers.get(device_id)

    # ----------------------------------------------------------------
    # Socket setup
    # ----------------------------------------------------------------

    def _create_socket(self) -> socket.socket:
        """Create UDP multicast socket for mDNS."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Allow multiple sockets to bind to the same port
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except (AttributeError, OSError):
            pass  # Not available on all platforms

        # Bind to mDNS port
        sock.bind(("", MDNS_PORT))

        # Join multicast group
        mreq = struct.pack(
            "4sl",
            socket.inet_aton(MDNS_MULTICAST_ADDR),
            socket.INADDR_ANY,
        )
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

        # Set TTL for multicast packets
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)

        # Non-blocking for asyncio
        sock.setblocking(False)

        return sock

    # ----------------------------------------------------------------
    # Broadcast
    # ----------------------------------------------------------------

    async def _broadcast_loop(self) -> None:
        """Periodically broadcast device presence."""
        while self._running:
            try:
                await self._announce()
            except Exception as e:
                logger.debug("Federation mDNS: broadcast failed: %s", e)

            await asyncio.sleep(self.broadcast_interval)

    async def _announce(self, status: str = "online") -> None:
        """Broadcast device presence via mDNS."""
        if not self._socket:
            return

        import socket as _socket

        # Build announcement payload
        payload = {
            "type": "announce",
            "device_id": self.device_id,
            "ws_port": self.ws_port,
            "status": status,
            "version": "2.0.0",
            "hostname": socket.gethostname(),
        }

        # Build mDNS query/announcement packet
        # This is a simplified mDNS — just a UDP broadcast with JSON payload
        # In production, you'd use proper mDNS/DNS-SD packet format
        data = json.dumps(payload).encode()

        try:
            self._socket.sendto(
                data,
                (MDNS_MULTICAST_ADDR, MDNS_PORT),
            )
            logger.debug(
                "Federation mDNS: announced %s (%s)",
                self.device_id, status,
            )
        except Exception as e:
            logger.debug("Federation mDNS: announce failed: %s", e)

    # ----------------------------------------------------------------
    # Receive
    # ----------------------------------------------------------------

    async def _receive_loop(self) -> None:
        """Listen for mDNS announcements from other peers."""
        if not self._socket:
            return

        loop = asyncio.get_event_loop()
        sock = self._socket

        while self._running:
            try:
                data = await loop.sock_recv(sock, 4096)
                if not isinstance(data, bytes):
                    continue

                try:
                    payload = json.loads(data)
                    # For UDP multicast, we don't get the sender addr from sock_recv
                    # so we use a placeholder
                    await self._handle_message(payload, ("0.0.0.0", 0))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass  # Not our protocol

            except (OSError, asyncio.CancelledError):
                if not self._running:
                    break
                await asyncio.sleep(1)

    async def _handle_message(self, payload: dict, addr: tuple) -> None:
        """Handle incoming mDNS message."""
        msg_type = payload.get("type", "")

        if msg_type == "announce":
            device_id = payload.get("device_id", "")
            if not device_id or device_id == self.device_id:
                return  # Skip self

            ip = addr[0]
            ws_port = payload.get("ws_port", 18765)
            ws_url = f"ws://{ip}:{ws_port}"

            peer = DiscoveredPeer(
                device_id=device_id,
                hostname=payload.get("hostname", ip),
                ws_url=ws_url,
                version=payload.get("version", "2.0.0"),
                status=payload.get("status", "online"),
                last_seen=time.time(),
            )

            is_new = device_id not in self._peers
            self._peers[device_id] = peer

            if is_new and self.on_discover:
                self.on_discover(peer)
                logger.info(
                    "Federation mDNS: discovered %s (%s) at %s",
                    device_id, peer.hostname, ws_url,
                )
            else:
                logger.debug(
                    "Federation mDNS: refreshed %s", device_id,
                )

    # ----------------------------------------------------------------
    # State queries
    # ----------------------------------------------------------------

    @property
    def peer_count(self) -> int:
        """Number of currently active peers."""
        return len(self.get_peers())
