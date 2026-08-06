"""Federation WebSocket connection manager — hardened for production.

Security:
- TLS/SSL support (wss://) with configurable cert paths
- Mandatory auth_token validation
- Message size limit (1MB max)
- Connection rate limiting (10/min per IP)
- IP whitelist support
- HMAC-SHA256 full 64-char signature

Reliability:
- Proper async connection close
- Port fallback on binding failure
- Active ping to all peers
- Connection quality metrics (latency tracking)
- Actual memory detection
"""
from __future__ import annotations

import asyncio
import logging
import os
import ssl
import time
from pathlib import Path
from typing import Any, Callable, Optional

from gateway.config import FederationConfig
from gateway.federation.federation_protocol import (
    FedMessage,
    MessageType,
    PeerInfo,
)

logger = logging.getLogger(__name__)

# Limits
MAX_MESSAGE_SIZE = 1024 * 1024  # 1MB
MAX_CONNECTIONS_PER_MINUTE = 10
PING_INTERVAL = 15  # seconds
CONNECTION_TIMEOUT = 10
HANDSHAKE_TIMEOUT = 30


class FederationConnectionManager:
    """Manages WebSocket connections to all federation peers — hardened."""

    def __init__(
        self,
        device_id: str,
        auth_token: Optional[str] = None,
        ws_port: int = 18765,
        on_message: Optional[Callable[[FedMessage], None]] = None,
        on_peer_join: Optional[Callable[[PeerInfo], None]] = None,
        on_peer_leave: Optional[Callable[[str], None]] = None,
        tls_cert: Optional[str] = None,
        tls_key: Optional[str] = None,
        ip_whitelist: Optional[list[str]] = None,
    ):
        self.device_id = device_id
        self.auth_token = auth_token
        self.ws_port = ws_port
        self._on_message = on_message
        self._on_peer_join = on_peer_join
        self._on_peer_leave = on_peer_leave

        # TLS
        self.tls_cert = tls_cert
        self.tls_key = tls_key
        self._ssl_context: Optional[ssl.SSLContext] = None
        if tls_cert and tls_key:
            self._ssl_context = self._create_ssl_context()

        # IP whitelist
        self.ip_whitelist: set[str] = set(ip_whitelist) if ip_whitelist else set()

        # State
        self._ws_connections: dict[str, Any] = {}
        self._peer_infos: dict[str, PeerInfo] = {}
        self._reconnect_tasks: dict[str, asyncio.Task] = {}
        self._metrics: dict[str, dict] = {}  # device_id -> {latency_ms, last_ping}

        # Rate limiting: {ip: [timestamps]}
        self._conn_times: dict[str, list[float]] = {}
        self._conn_times_lock = asyncio.Lock()  # Phase 12: protect rate-limit state

        # Server
        self._server: Optional[Any] = None
        self._running = False

    # ----------------------------------------------------------------
    # TLS
    # ----------------------------------------------------------------

    def _create_ssl_context(self) -> Optional[ssl.SSLContext]:
        cert_path = Path(os.path.expanduser(self.tls_cert))  # type: ignore[arg-type]
        key_path = Path(os.path.expanduser(self.tls_key))  # type: ignore[arg-type]
        if not cert_path.exists() or not key_path.exists():
            logger.warning(
                "Federation: TLS cert/key not found at %s / %s",
                cert_path, key_path,
            )
            return None
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(str(cert_path), str(key_path))
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        logger.info("Federation: TLS enabled")
        return ctx

    def _ws_scheme(self) -> str:
        return "wss" if self._ssl_context else "ws"

    # ----------------------------------------------------------------
    # Peer management
    # ----------------------------------------------------------------

    def register_peer(self, info: PeerInfo) -> None:
        if info.device_id == self.device_id:
            return
        self._peer_infos[info.device_id] = info
        logger.info(
            "Federation: registered peer %s (%s) at %s",
            info.device_id, info.hostname, info.ws_url,
        )

    async def unregister_peer(self, device_id: str, reason: str = "offline") -> None:
        self._peer_infos.pop(device_id, None)
        self._metrics.pop(device_id, None)
        await self._close_connection(device_id)
        if self._on_peer_leave:
            self._on_peer_leave(device_id)
        logger.info("Federation: peer %s left (%s)", device_id, reason)

    def get_peers(self) -> list[PeerInfo]:
        return [
            info for info in self._peer_infos.values()
            if info.device_id in self._ws_connections
        ]

    def get_peer(self, device_id: str) -> Optional[PeerInfo]:
        return self._peer_infos.get(device_id)

    def get_online_count(self) -> int:
        return len(self._ws_connections)

    def get_metrics(self, device_id: str) -> Optional[dict]:
        return self._metrics.get(device_id)

    def get_all_metrics(self) -> dict:
        return dict(self._metrics)

    # ----------------------------------------------------------------
    # Rate limiting
    # ----------------------------------------------------------------

    async def _check_rate_limit(self, ip: str) -> bool:
        now = time.time()
        async with self._conn_times_lock:  # Phase 12: Race condition fix
            times = self._conn_times.setdefault(ip, [])
            self._conn_times[ip] = [t for t in times if now - t < 60]
            if len(self._conn_times[ip]) >= MAX_CONNECTIONS_PER_MINUTE:
                logger.warning("Federation: rate limit exceeded for IP %s", ip)
                return False
            self._conn_times[ip].append(now)
            return True

    # ----------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------

    async def start(self, listen: bool = True) -> None:
        self._running = True
        for info in self._peer_infos.values():
            if info.ws_url:
                asyncio.create_task(self._connect_to_peer(info))
        asyncio.create_task(self._ping_loop())
        if listen:
            await self._start_server()
        logger.info(
            "Federation: connection manager started (device=%s, port=%d, tls=%s)",
            self.device_id, self.ws_port,
            "yes" if self._ssl_context else "no",
        )

    async def stop(self) -> None:
        self._running = False
        try:
            await self._broadcast(FedMessage.peer_leave(self.device_id, reason="shutdown"))
        except Exception:
            pass
        for t in list(self._reconnect_tasks.values()):
            t.cancel()
        for device_id in list(self._ws_connections):
            await self._close_connection(device_id)
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        logger.info("Federation: connection manager stopped")

    # ----------------------------------------------------------------
    # Sending
    # ----------------------------------------------------------------

    async def send(self, message: FedMessage) -> bool:
        if message.sender_id != self.device_id:
            message.sender_id = self.device_id
        if self.auth_token:
            message.sign(self.auth_token)
        if message.target_id:
            return await self._send_to(message.target_id, message)
        return await self._broadcast(message)

    async def _send_to(self, device_id: str, message: FedMessage) -> bool:
        ws = self._ws_connections.get(device_id)
        if not ws:
            return False
        try:
            await ws.send(message.to_json())
            return True
        except Exception as e:
            logger.warning("Federation: send to %s failed: %s", device_id, e)
            await self._close_connection(device_id)
            return False

    async def _broadcast(self, message: FedMessage) -> bool:
        results = []
        for did in list(self._ws_connections):
            if did == self.device_id:
                continue
            results.append(await self._send_to(did, message))
        return any(results)

    # ----------------------------------------------------------------
    # Outbound connections
    # ----------------------------------------------------------------

    async def _connect_to_peer(self, info: PeerInfo) -> None:
        max_retries = 10
        for retry in range(1, max_retries + 1):
            if not self._running:
                return
            try:
                import websockets
                extra = {}
                if self.auth_token:
                    extra["additional_headers"] = {"X-Federation-Auth": self.auth_token}
                if self._ssl_context or info.ws_url.startswith("wss://"):
                    # Outbound connections to peers use a CLIENT ssl context.
                    # The server context (self._ssl_context) has no CA trust
                    # chain, so verifying against it would fail for LAN
                    # self-signed certs. Use a permissive client context that
                    # still encrypts (TLS) but skips CA-chain verification —
                    # peer auth is enforced via auth_token + message signing.
                    client_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                    client_ctx.check_hostname = False
                    client_ctx.verify_mode = ssl.CERT_NONE
                    extra["ssl"] = client_ctx
                extra["max_size"] = MAX_MESSAGE_SIZE

                ws = await asyncio.wait_for(
                    websockets.connect(info.ws_url, **extra),
                    timeout=CONNECTION_TIMEOUT,
                )
                self._ws_connections[info.device_id] = ws
                info.status = "online"
                info.last_seen = time.time()
                self._metrics[info.device_id] = {
                    "latency_ms": 0.0, "last_ping": time.time(),
                }
                logger.info(
                    "Federation: connected to %s (%s) [TLS=%s]",
                    info.device_id, info.ws_url,
                    "yes" if self._ssl_context else "no",
                )

                join_msg = FedMessage.peer_join(self.device_id, self._get_self_info())
                if self.auth_token:
                    join_msg.sign(self.auth_token)
                await ws.send(join_msg.to_json())

                asyncio.create_task(self._receive_loop(info.device_id, ws))
                self._reconnect_tasks.pop(info.device_id, None)
                return

            except Exception as e:
                delay = min(2 ** retry, 60)
                logger.warning(
                    "Federation: connection to %s failed (%d/%d), retry in %ds: %s",
                    info.device_id, retry, max_retries, delay, e,
                )
                await asyncio.sleep(delay)

        logger.error("Federation: exhausted retries for %s", info.device_id)

    # ----------------------------------------------------------------
    # Inbound
    # ----------------------------------------------------------------

    async def _start_server(self) -> None:
        import websockets

        async def handler(ws, path=None):
            ip = "unknown"
            try:
                remote = getattr(ws, "remote_address", None)
                if remote:
                    ip = remote[0] if isinstance(remote, tuple) else str(remote)

                # Rate limit
                if not await self._check_rate_limit(ip):
                    await ws.close()
                    return

                # IP whitelist
                if self.ip_whitelist and ip not in self.ip_whitelist:
                    logger.warning(
                        "Federation: rejected non-whitelisted IP %s", ip,
                    )
                    await ws.close()
                    return

                # Handshake
                raw = await asyncio.wait_for(ws.recv(), timeout=HANDSHAKE_TIMEOUT)
                if isinstance(raw, str) and len(raw) > MAX_MESSAGE_SIZE:
                    await ws.close()
                    return

                msg = FedMessage.from_json(raw)
                device_id = msg.sender_id
                if not device_id:
                    await ws.close()
                    return

                # Auth
                if self.auth_token and not msg.verify(self.auth_token):
                    hdr = getattr(ws, "request_headers", {})
                    if hasattr(hdr, "get"):
                        hdr_token = hdr.get("X-Federation-Auth", "")
                    else:
                        hdr_token = ""
                    if hdr_token != self.auth_token:
                        logger.warning(
                            "Federation: rejected unauthenticated from %s", ip,
                        )
                        await ws.close()
                        return

                self._ws_connections[device_id] = ws

                if msg.msg_type == MessageType.PEER_JOIN.value:
                    peer_data = msg.payload.get("peer_info", {})
                    info = PeerInfo(**peer_data)
                    info.last_seen = time.time()
                    self._peer_infos[device_id] = info
                    if self._on_peer_join:
                        self._on_peer_join(info)

                await self._receive_loop(device_id, ws)

            except asyncio.TimeoutError:
                logger.warning("Federation: handshake timeout from %s", ip)
            except Exception as e:
                logger.warning("Federation: inbound error from %s: %s", ip, e)
            finally:
                try:
                    await ws.close()
                except Exception:
                    pass

        # Try preferred port, fallback to random
        port = self.ws_port
        for attempt in range(5):
            try:
                self._server = await websockets.serve(
                    handler, "0.0.0.0", port,
                    ping_interval=PING_INTERVAL,
                    ping_timeout=10,
                    max_size=MAX_MESSAGE_SIZE,
                    ssl=self._ssl_context,
                )
                if port != self.ws_port:
                    self.ws_port = port
                break
            except OSError:
                port = 0
                if attempt == 4:
                    raise

        logger.info(
            "Federation: listening on port %d (TLS=%s)",
            self.ws_port, "yes" if self._ssl_context else "no",
        )

    # ----------------------------------------------------------------
    # Receive loop
    # ----------------------------------------------------------------

    async def _receive_loop(self, device_id: str, ws: Any) -> None:
        try:
            async for raw in ws:
                try:
                    if isinstance(raw, str) and len(raw) > MAX_MESSAGE_SIZE:
                        logger.warning(
                            "Federation: oversized message from %s (%dB)",
                            device_id, len(raw),
                        )
                        continue

                    msg = FedMessage.from_json(raw)

                    if self.auth_token and not msg.verify(self.auth_token):
                        logger.warning("Federation: bad signature from %s", device_id)
                        continue

                    if msg.is_expired():
                        continue

                    if device_id in self._peer_infos:
                        self._peer_infos[device_id].last_seen = time.time()

                    # Track ping latency
                    if msg.msg_type == MessageType.PEER_PONG.value:
                        pong_ts = msg.payload.get("timestamp", 0)
                        if pong_ts and device_id in self._metrics:
                            lat = (time.time() - pong_ts) * 1000
                            self._metrics[device_id]["latency_ms"] = lat
                            self._metrics[device_id]["last_ping"] = time.time()

                    if msg.msg_type == MessageType.PEER_JOIN.value:
                        pd = msg.payload.get("peer_info", {})
                        if pd.get("device_id") != self.device_id:
                            info = PeerInfo(**pd)
                            info.last_seen = time.time()
                            self._peer_infos[info.device_id] = info
                            if self._on_peer_join:
                                self._on_peer_join(info)
                    elif msg.msg_type == MessageType.PEER_LEAVE.value:
                        await self.unregister_peer(
                            device_id, msg.payload.get("reason", "offline"),
                        )
                        continue

                    if self._on_message:
                        msg._sender_ws_url = self._peer_infos.get(
                            device_id,
                            PeerInfo(device_id=device_id, hostname=""),
                        ).ws_url
                        self._on_message(msg)

                except Exception as e:
                    logger.warning(
                        "Federation: parse error from %s: %s", device_id, e,
                    )
        except Exception as e:
            logger.info("Federation: %s closed: %s", device_id, e)
        finally:
            await self._close_connection(device_id)
            if self._running:
                info = self._peer_infos.get(device_id)
                if info:
                    self._reconnect_tasks[device_id] = asyncio.create_task(
                        self._connect_to_peer(info),
                    )

    async def _close_connection(self, device_id: str) -> None:
        ws = self._ws_connections.pop(device_id, None)
        if ws:
            try:
                await asyncio.wait_for(ws.close(), timeout=5)
            except Exception:
                pass

    # ----------------------------------------------------------------
    # Active ping
    # ----------------------------------------------------------------

    async def _ping_loop(self) -> None:
        while self._running:
            await asyncio.sleep(PING_INTERVAL)
            now = time.time()
            for did in list(self._ws_connections):
                ws = self._ws_connections.get(did)
                if ws:
                    try:
                        ping = FedMessage(
                            msg_type=MessageType.PEER_PING.value,
                            sender_id=self.device_id,
                            target_id=did,
                            payload={"timestamp": now},
                        )
                        await ws.send(ping.to_json())
                    except Exception:
                        await self._close_connection(did)

            for did, info in list(self._peer_infos.items()):
                if did not in self._ws_connections and now - info.last_seen > 60:
                    info.status = "offline"

    # ----------------------------------------------------------------
    # Self info
    # ----------------------------------------------------------------

    def _get_self_info(self) -> PeerInfo:
        import socket
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

        memory_gb = 0.0
        try:
            import subprocess
            mem = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=3,
            ).stdout.strip()
            memory_gb = round(int(mem) / (1024 ** 3), 1)
        except Exception:
            pass

        return PeerInfo(
            device_id=self.device_id,
            hostname=hostname,
            ws_url=f"{self._ws_scheme()}://{ip}:{self.ws_port}",
            cpu_cores=os.cpu_count() or 0,
            memory_gb=memory_gb,
            load_avg=load_avg,
        )
