"""Phase 20: Cross-transport relay — HTTP/SSE fallback + transport router.

Provides two capabilities beyond WebSocket:
  1. HttpSSETransport — HTTP/Server-Sent-Events transport as fallback when
     WebSocket is blocked by firewall/NAT. Tasks relay via POST /api/federation/relay
     with SSE stream on GET /api/federation/relay/stream/{peer_id}.
  2. FederationTransportRouter — selects the best available transport per peer
     (WebSocket > HTTP/SSE > shared_db) based on connectivity check.

Architecture:
  - FederationConnectionManager picks the transport per peer
  - HttpSSETransport implements RelayTransport-like interface
  - Router wraps multiple transports, returns first available

Usage:
    # SSE fallback for a specific peer
    sse = HttpSSETransport(peer_id="mac-mini", api_url="https://mac-mini.local:18766")
    ok = await sse.connect()
    await sse.send_outbound({"op": "relay_task", "task_id": "t-1", "payload": {...}})

    # Transport router across all transports
    router = FederationTransportRouter(local_device_id="mac-a")
    await router.register_transport("ws", ws_transport)
    await router.register_transport("sse", http_sse_transport)
    best = await router.get_best_transport(peer_id="mac-b")
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from urllib.parse import urljoin

import aiohttp

from gateway.federation.audit import AuditLog

logger = logging.getLogger(__name__)

# ─── Transport interface ──────────────────────────────────────────────


@runtime_checkable
class RelayTransportLike(Protocol):
    """Minimal transport interface shared by WS and HTTP/SSE."""

    async def connect(self) -> bool:
        ...

    async def disconnect(self) -> None:
        ...

    async def send_outbound(self, action: Dict[str, Any]) -> Dict[str, Any]:
        ...

    @property
    def peer_id(self) -> str:
        ...

    @property
    def is_connected(self) -> bool:
        ...


# ─── HTTP/SSE Transport ───────────────────────────────────────────────


@dataclass
class SSERelayMessage:
    """A single event from an SSE relay stream."""
    event_type: str  # e.g. "task_offer", "task_result", "peer_announce"
    data: Dict[str, Any]
    sequence: int
    timestamp: float


class HttpSSETransport:
    """HTTP/Server-Sent-Events fallback transport for federation relay.

    Used when WebSocket is unavailable (corporate firewall, NAT, etc.).
    Tasks are POSTed; results streamed via SSE.

    Security: all requests require a valid cluster_secret header.
    TLS is enforced unless allow_insecure is explicitly set.
    """

    RELAY_ENDPOINT = "/api/federation/relay"
    STREAM_ENDPOINT = "/api/federation/relay/stream/{peer_id}"

    def __init__(
        self,
        peer_id: str,
        api_url: str,
        cluster_secret: str,
        local_device_id: str,
        timeout_s: float = 10.0,
        allow_insecure: bool = False,
        audit_log: Optional[AuditLog] = None,
    ) -> None:
        self.peer_id = peer_id
        self.api_url = api_url.rstrip("/")
        self.cluster_secret = cluster_secret
        self.local_device_id = local_device_id
        self.timeout_s = timeout_s
        self.allow_insecure = allow_insecure
        self.audit_log = audit_log

        self._session: Optional[aiohttp.ClientSession] = None
        self._stream_task: Optional[asyncio.Task] = None
        self._inbound_queue: asyncio.Queue[SSERelayMessage] = asyncio.Queue()
        self._connected = False
        self._last_seq = 0
        self._lock = asyncio.Lock()

    # ── Lifecycle ────────────────────────────────────────────────────

    async def connect(self) -> bool:
        """Test connectivity by hitting the health endpoint."""
        if self._connected:
            return True
        headers = self._auth_headers()
        timeout = aiohttp.ClientTimeout(total=self.timeout_s)
        self._session = aiohttp.ClientSession(timeout=timeout)
        try:
            async with self._session.get(
                f"{self.api_url}/api/federation/health",
                headers=headers,
                ssl=False if self.allow_insecure else True,
            ) as resp:
                ok = resp.status == 200
                if ok:
                    self._connected = True
                    logger.info("HttpSSE: connected to %s (HTTP/SSE transport)", self.api_url)
                    # Start SSE stream listener
                    self._stream_task = asyncio.create_task(self._sse_reader())
                return ok
        except Exception as exc:
            logger.warning("HttpSSE: failed to connect to %s: %s", self.api_url, exc)
            await self._cleanup()
            return False

    async def disconnect(self) -> None:
        async with self._lock:
            await self._cleanup()

    async def _cleanup(self) -> None:
        self._connected = False
        if self._stream_task:
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass
            self._stream_task = None
        if self._session:
            await self._session.close()
            self._session = None

    # ── Outbound ─────────────────────────────────────────────────────

    async def send_outbound(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """POST a relay action to the peer via HTTP."""
        if not self._connected or not self._session:
            return {"success": False, "error": "not_connected"}

        headers = {
            **self._auth_headers(),
            "Content-Type": "application/json",
        }
        try:
            async with self._session.post(
                f"{self.api_url}{self.RELAY_ENDPOINT}",
                json=action,
                headers=headers,
                ssl=False if self.allow_insecure else True,
            ) as resp:
                data = await resp.json()
                if self.audit_log:
                    from gateway.federation.audit import NodeEvent, TaskEvent
                    # Map SSE relay events to existing audit event types
                    ev = NodeEvent.leave(
                        node_id=self.peer_id,
                        reason=f"sse_sent:{action.get('op')}",
                        peer_id=self.peer_id,
                        op=action.get("op", ""),
                        task_id=action.get("task_id", ""),
                        status_code=str(resp.status),
                    )
                    self.audit_log.append(ev)
                return data
        except asyncio.TimeoutError:
            return {"success": False, "error": "timeout"}
        except Exception as exc:
            logger.warning("HttpSSE: send_outbound failed: %s", exc)
            return {"success": False, "error": str(exc)}

    # ── SSE Reader ───────────────────────────────────────────────────

    async def _sse_reader(self) -> None:
        """Read SSE stream from the peer and push to inbound queue."""
        if not self._session:
            return
        headers = self._auth_headers()
        url = f"{self.api_url}/api/federation/relay/stream/{self.local_device_id}"
        try:
            async with self._session.get(
                url,
                headers=headers,
                ssl=False if self.allow_insecure else True,
            ) as resp:
                async for line in resp.content:
                    if not line.strip():
                        continue
                    if line.startswith(b"data:"):
                        raw = line[5:].strip().decode("utf-8", errors="replace")
                        try:
                            event = json.loads(raw)
                            msg = SSERelayMessage(
                                event_type=event.get("type", "unknown"),
                                data=event.get("data", {}),
                                sequence=event.get("seq", self._last_seq + 1),
                                timestamp=event.get("ts", time.time()),
                            )
                            self._last_seq = msg.sequence
                            await self._inbound_queue.put(msg)
                        except json.JSONDecodeError:
                            pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("HttpSSE: SSE reader error for %s: %s", self.peer_id, exc)

    # ── Inbound ─────────────────────────────────────────────────────

    def set_inbound_handler(self, handler) -> None:
        """Set callback for inbound messages (mirrors RelayTransport interface)."""
        self._inbound_handler = handler
        # Start forwarding queue → handler
        asyncio.create_task(self._forward_inbound())

    async def _forward_inbound(self) -> None:
        while self._connected:
            try:
                msg = await asyncio.wait_for(
                    self._inbound_queue.get(),
                    timeout=5.0,
                )
                if hasattr(self, "_inbound_handler") and self._inbound_handler:
                    await self._inbound_handler(msg.data)
            except asyncio.TimeoutError:
                continue
            except Exception as exc:
                logger.warning("HttpSSE: inbound forward error: %s", exc)

    # ── Helpers ──────────────────────────────────────────────────────

    def _auth_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.cluster_secret}",
            "X-Device-ID": self.local_device_id,
        }

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def health_check(self) -> bool:  # type: ignore[misc]
        """Ping the peer via HTTP to verify liveness."""
        if not self._session:
            return False
        try:
            async with self._session.get(
                f"{self.api_url}/api/federation/health",
                headers=self._auth_headers(),
                ssl=False if self.allow_insecure else True,
            ) as resp:
                return resp.status == 200
        except Exception:
            return False


# ─── Transport Router ─────────────────────────────────────────────────


@dataclass
class TransportScore:
    transport_name: str
    transport: RelayTransportLike
    latency_ms: float
    is_available: bool
    score: float  # higher = better


class FederationTransportRouter:
    """Selects the best available transport per peer.

    Tries transports in priority order:
      1. WebSocket (lowest latency, full-duplex)
      2. HTTP/SSE (firewall-friendly, SSE for inbound)
      3. shared_db (last resort — for completely isolated networks)

    Scores are based on: latency, availability, and transport capability match.
    """

    TRANSPORT_PRIORITY = ["ws", "sse", "shared_db"]

    def __init__(self, local_device_id: str) -> None:
        self.local_device_id = local_device_id
        # name -> (transport, score)
        self._transports: Dict[str, RelayTransportLike] = {}
        self._transport_scores: Dict[str, float] = {}
        self._lock = asyncio.Lock()

    def register_transport(self, name: str, transport: RelayTransportLike) -> None:
        """Register a named transport (ws, sse, shared_db, etc.)."""
        self._transports[name] = transport
        logger.info("TransportRouter: registered transport '%s' for %s", name, self.local_device_id)

    async def get_best_transport(
        self,
        peer_id: str,
        latency_hints: Optional[Dict[str, float]] = None,
    ) -> Optional[TransportScore]:
        """Return the best available transport for a peer.

        Args:
            peer_id: target peer device ID
            latency_hints: {transport_name: latency_ms} for scoring

        Returns:
            TransportScore with the highest score, or None if all unavailable.
        """
        if not self._transports:
            return None

        candidates: List[TransportScore] = []
        hints = latency_hints or {}

        for name, transport in self._transports.items():
            try:
                connected = transport.is_connected if hasattr(transport, "is_connected") else False
                # For connection test, do a quick health check
                health_ok = True
                if hasattr(transport, "health_check"):
                    health_ok = await asyncio.wait_for(
                        transport.health_check(),
                        timeout=3.0,
                    )
                is_available = connected or health_ok
            except Exception:
                is_available = False

            latency = hints.get(name, hints.get(peer_id, 100.0))

            # Score: prefer WS > SSE > shared_db, weighted by latency
            priority = {
                "ws": 3.0,
                "sse": 2.0,
                "shared_db": 1.0,
            }.get(name, 0.5)

            # Latency score: lower is better (0-1 normalized, penalize >500ms)
            latency_score = max(0, 1.0 - (latency / 1000.0))

            total = (priority * 0.6) + (latency_score * 0.4) if is_available else 0.0

            candidates.append(TransportScore(
                transport_name=name,
                transport=transport,
                latency_ms=latency,
                is_available=is_available,
                score=total,
            ))

        available = [c for c in candidates if c.is_available]
        if not available:
            return None

        best = max(available, key=lambda c: c.score)
        logger.debug(
            "TransportRouter: best for %s → %s (score=%.2f, latency=%.0fms)",
            peer_id, best.transport_name, best.score, best.latency_ms,
        )
        return best

    async def relay_task(
        self,
        peer_id: str,
        task_payload: Dict[str, Any],
        latency_hints: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Relay a task to the best available transport for the peer.

        Args:
            peer_id: target peer device ID
            task_payload: {task_id, op, payload, ...}
            latency_hints: optional latency hints per transport

        Returns:
            {success, error?, via_transport?}
        """
        best = await self.get_best_transport(peer_id, latency_hints)
        if not best:
            return {"success": False, "error": "no_transport_available"}

        try:
            result = await best.transport.send_outbound(task_payload)
            result["via_transport"] = best.transport_name
            result["via_latency_ms"] = best.latency_ms
            return result
        except Exception as exc:
            logger.warning("TransportRouter: relay via %s failed: %s", best.transport_name, exc)
            return {"success": False, "error": str(exc), "via_transport": best.transport_name}

    def get_transport_names(self) -> List[str]:
        """Return registered transport names in priority order."""
        return [t for t in self.TRANSPORT_PRIORITY if t in self._transports]


__all__ = [
    "HttpSSETransport",
    "FederationTransportRouter",
    "RelayTransportLike",
    "TransportScore",
    "SSERelayMessage",
]
