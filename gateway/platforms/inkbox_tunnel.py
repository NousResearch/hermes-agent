"""Inkbox tunnel client — agent-side data plane for inkboxwire.com.

Replaces ngrok for the Inkbox adapter. Opens a persistent HTTP/2 connection
to ``{env}.inkboxwire.com:443``, parks a pool of intake streams, and bridges
inbound HTTP + WebSocket traffic to a local aiohttp server.

Edge-mode tunnels only (``tls_mode=edge``): TLS terminates at the Inkbox
NLB; we speak h2c over a TLS-tunneled TCP socket, ALPN ``h2``. Passthrough
mode (CSR / per-tunnel cert) is out of scope.

Public surface:
    InkboxTunnel(
        api_key, base_url, listen_host, listen_port,
        tunnel_name, state_path, allow_create=True,
    )
    .ensure_tunnel()  → looks up or creates the tunnel; returns metadata
    .start()          → opens the persistent connection + intake pool
    .public_url       → ``https://{tunnel_name}.{tunnel_zone}``
    .public_host      → ``{tunnel_name}.{tunnel_zone}``
    .stop()
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import ssl
import struct
import time
from contextlib import suppress
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse
from uuid import UUID, uuid4

try:
    import h2.config
    import h2.connection
    import h2.events
    import h2.exceptions
    import h2.settings

    H2_AVAILABLE = True
except ImportError:
    h2 = None  # type: ignore[assignment]
    H2_AVAILABLE = False

try:
    import aiohttp
    from aiohttp import WSMsgType, web  # noqa: F401  (web only used for typing)

    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None  # type: ignore[assignment]
    WSMsgType = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

import httpx

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------

INKBOX_NAMESPACE_PREFIX = "inkbox-"
INKBOX_FORWARDED_HEADER_PREFIX = "inkbox-h-"

# Fixed inkbox-* meta keys (mirror of TunnelMetaHeader on the server).
META_REQUEST_ID = "inkbox-request-id"
META_METHOD = "inkbox-method"
META_PATH = "inkbox-path"
META_ROUTE_KIND = "inkbox-route-kind"
META_STATUS = "inkbox-status"
META_WS_ID = "inkbox-ws-id"
META_TCP_ID = "inkbox-tcp-id"
META_BODY_URI = "inkbox-body-uri"
META_REASON = "inkbox-reason"

ROUTE_WEBHOOK = "webhook"
ROUTE_WS_UPGRADE = "ws-upgrade"
ROUTE_TCP_STREAM = "tcp-stream"

WS_SUBPROTOCOL = "inkbox-tunnel-ws"

DEFAULT_POOL_SIZE = 8
DEFAULT_BASE_URL = "https://inkbox.ai"
DEFAULT_TUNNEL_PORT = 443
INTAKE_RECONNECT_BACKOFF = (1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 60.0)

# Hop-by-hop response headers we must not replay onto the third-party WS
# accept frame. Matches the server-side _BLOCKED_ACCEPT_HEADERS.
_WS_HOPBYHOP = frozenset({
    "upgrade", "connection", "sec-websocket-key", "sec-websocket-accept",
    "sec-websocket-version", "sec-websocket-protocol",
    "sec-websocket-extensions", "keep-alive", "proxy-authenticate",
    "proxy-authorization", "te", "trailer", "transfer-encoding",
    "content-length",
})


# ----------------------------------------------------------------------------
# REST control plane
# ----------------------------------------------------------------------------

class TunnelControlPlaneError(Exception):
    """Raised for failures hitting /api/v1/tunnels (auth, validation, etc.)."""


def _slug_for_identity(handle: str) -> str:
    """Turn an identity handle into a DNS-safe tunnel name fragment."""
    out = []
    prev_dash = True
    for ch in handle.lower():
        if ch.isalnum():
            out.append(ch)
            prev_dash = False
        elif not prev_dash:
            out.append("-")
            prev_dash = True
    name = "".join(out).strip("-")
    if not name:
        name = "agent"
    # Server enforces 3..63 chars
    if len(name) < 3:
        name = (name + "-tunnel")[:63]
    return name[:63]


def _tunnel_zone_for(base_url: str) -> str:
    """Derive the per-env tunnel zone from the API base URL.

    Production ``inkbox.ai`` → ``inkboxwire.com``; ``beta.inkbox.ai`` →
    ``beta.inkboxwire.com``; ``development.inkbox.ai`` →
    ``development.inkboxwire.com``. Override via ``INKBOX_TUNNEL_ZONE``.
    """
    override = (os.getenv("INKBOX_TUNNEL_ZONE") or "").strip()
    if override:
        return override.lstrip(".").lower()

    host = (urlparse(base_url).hostname or "").lower()
    if not host or host == "inkbox.ai":
        return "inkboxwire.com"
    if host.endswith(".inkbox.ai"):
        env = host[: -len(".inkbox.ai")]
        return f"{env}.inkboxwire.com"
    # Localhost / unknown — fall back to dev zone so the client at least
    # tries something. Operators can override with INKBOX_TUNNEL_ZONE.
    return "development.inkboxwire.com"


class _RestClient:
    """Thin httpx wrapper around /api/v1/tunnels — control plane only."""

    def __init__(self, *, api_key: str, base_url: str) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"X-API-Key": self._api_key},
            timeout=20.0,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def get_tunnel(self, tunnel_id: str) -> Optional[Dict[str, Any]]:
        r = await self._client.get(f"/api/v1/tunnels/{tunnel_id}")
        if r.status_code == 404:
            return None
        if r.status_code >= 400:
            raise TunnelControlPlaneError(
                f"GET /tunnels/{tunnel_id} → {r.status_code}: {r.text}",
            )
        return r.json()

    async def list_tunnels(self) -> List[Dict[str, Any]]:
        r = await self._client.get("/api/v1/tunnels/")
        if r.status_code >= 400:
            raise TunnelControlPlaneError(
                f"GET /tunnels → {r.status_code}: {r.text}",
            )
        return list(r.json().get("tunnels") or [])

    async def create_tunnel(
        self,
        *,
        tunnel_name: str,
        description: str = "",
    ) -> Tuple[Dict[str, Any], str]:
        r = await self._client.post(
            "/api/v1/tunnels/",
            json={
                "tunnel_name": tunnel_name,
                "description": description,
                "tls_mode": "edge",
            },
        )
        if r.status_code == 409:
            raise TunnelControlPlaneError(
                f"tunnel name {tunnel_name!r} already taken (409). "
                "Set INKBOX_TUNNEL_NAME to choose a different one.",
            )
        if r.status_code >= 400:
            raise TunnelControlPlaneError(
                f"POST /tunnels → {r.status_code}: {r.text}",
            )
        body = r.json()
        return body["tunnel"], body["connect_secret"]

    async def rotate_secret(self, tunnel_id: str) -> str:
        r = await self._client.post(
            f"/api/v1/tunnels/{tunnel_id}/rotate-secret",
        )
        if r.status_code >= 400:
            raise TunnelControlPlaneError(
                f"POST /tunnels/{tunnel_id}/rotate-secret → "
                f"{r.status_code}: {r.text}",
            )
        return r.json()["connect_secret"]


# ----------------------------------------------------------------------------
# H2 connection driver
# ----------------------------------------------------------------------------

class _StreamClosed(Exception):
    """Raised on the per-stream queue when the h2 connection ends."""


class _Stream:
    """Per-stream state: response headers, data buffer, end flag."""

    __slots__ = (
        "stream_id", "headers", "data", "ended", "event", "subprotocol", "frame_q",
    )

    def __init__(self, stream_id: int) -> None:
        self.stream_id = stream_id
        self.headers: List[Tuple[str, str]] = []
        self.data: bytearray = bytearray()
        self.ended = False
        # Fired once response HEADERS arrive — for normal POSTs this is
        # near the end; for parked /_system/intake it's the moment the
        # server starts dispatching an envelope; for extended CONNECT
        # (WS bridging) it's the accept handshake.
        self.event = asyncio.Event()
        self.subprotocol: Optional[str] = None
        # For ext-CONNECT WS streams: incremental DATA buffer queue so the
        # bridge pump can drain envelopes as they arrive (instead of waiting
        # for end_stream, which never comes on a long-lived bridge).
        self.frame_q: Optional[asyncio.Queue[Optional[bytes]]] = None


class _H2Driver:
    """Async TCP/TLS h2 connection with per-stream queues.

    Owns one TCP socket, one h2.Connection state machine, one reader task,
    and a writer lock. All public methods are coroutine-safe across the
    intake pool + ws bridges sharing the connection.
    """

    def __init__(self, *, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._h2: Optional[h2.connection.H2Connection] = None
        self._streams: Dict[int, _Stream] = {}
        self._writer_lock = asyncio.Lock()
        self._read_task: Optional[asyncio.Task[None]] = None
        self._closed = asyncio.Event()
        self._ping_task: Optional[asyncio.Task[None]] = None

    @property
    def closed(self) -> asyncio.Event:
        return self._closed

    async def connect(self) -> None:
        ssl_ctx = ssl.create_default_context()
        ssl_ctx.set_alpn_protocols(["h2"])
        self._reader, self._writer = await asyncio.open_connection(
            self._host, self._port, ssl=ssl_ctx,
        )
        sock = self._writer.get_extra_info("ssl_object")
        if sock is not None and sock.selected_alpn_protocol() != "h2":
            raise RuntimeError(
                f"ALPN negotiation failed — expected h2, got "
                f"{sock.selected_alpn_protocol()!r}",
            )

        cfg = h2.config.H2Configuration(
            client_side=True,
            header_encoding="utf-8",
        )
        self._h2 = h2.connection.H2Connection(config=cfg)
        self._h2.initiate_connection()
        # Allow the server to push us inbound payloads on parked streams
        # without waiting for our acknowledgement after every frame.
        self._h2.update_settings({
            h2.settings.SettingCodes.INITIAL_WINDOW_SIZE: 16 * 1024 * 1024,
        })
        await self._flush()
        self._read_task = asyncio.create_task(
            self._read_loop(), name="inkbox-tunnel-h2-reader",
        )
        self._ping_task = asyncio.create_task(
            self._ping_loop(), name="inkbox-tunnel-h2-ping",
        )

    async def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        if self._h2 is not None:
            with suppress(Exception):
                self._h2.close_connection()
                await self._flush()
        if self._writer is not None:
            with suppress(Exception):
                self._writer.close()
                await self._writer.wait_closed()
        for task in (self._read_task, self._ping_task):
            if task is not None and not task.done():
                task.cancel()
                with suppress(Exception):
                    await task
        # Wake any pending stream waiters.
        for s in self._streams.values():
            if s.frame_q is not None:
                with suppress(asyncio.QueueFull):
                    s.frame_q.put_nowait(None)
            s.event.set()
        self._streams.clear()

    async def _flush(self) -> None:
        if self._h2 is None or self._writer is None:
            return
        data = self._h2.data_to_send()
        if not data:
            return
        self._writer.write(data)
        await self._writer.drain()

    async def _ping_loop(self) -> None:
        # h2-level keepalive — distinct from per-stream idle. 20s matches
        # the server's published cadence.
        try:
            while not self._closed.is_set():
                await asyncio.sleep(20.0)
                async with self._writer_lock:
                    if self._h2 is None or self._closed.is_set():
                        return
                    self._h2.ping(b"\x00" * 8)
                    await self._flush()
        except asyncio.CancelledError:
            return
        except Exception:
            logger.debug("[InkboxTunnel] ping loop ended", exc_info=True)

    async def _read_loop(self) -> None:
        assert self._reader is not None and self._h2 is not None
        try:
            while True:
                data = await self._reader.read(65536)
                if not data:
                    return
                events = self._h2.receive_data(data)
                async with self._writer_lock:
                    await self._flush()
                for event in events:
                    self._handle_event(event)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.warning("[InkboxTunnel] h2 read loop crashed: %s", exc)
        finally:
            self._closed.set()
            for s in list(self._streams.values()):
                if s.frame_q is not None:
                    with suppress(asyncio.QueueFull):
                        s.frame_q.put_nowait(None)
                s.event.set()

    def _handle_event(self, event: Any) -> None:
        sid = getattr(event, "stream_id", 0)
        if isinstance(event, h2.events.ResponseReceived):
            s = self._streams.get(sid)
            if s is not None:
                s.headers = list(event.headers)
                s.event.set()
        elif isinstance(event, h2.events.InformationalResponseReceived):
            # 1xx — irrelevant for us, but don't blow up.
            pass
        elif isinstance(event, h2.events.DataReceived):
            s = self._streams.get(sid)
            if s is not None:
                if s.frame_q is not None:
                    s.frame_q.put_nowait(bytes(event.data))
                else:
                    s.data.extend(event.data)
                # Acknowledge so the server keeps sending.
                with suppress(h2.exceptions.ProtocolError):
                    self._h2.acknowledge_received_data(
                        event.flow_controlled_length, sid,
                    )
        elif isinstance(event, h2.events.StreamEnded):
            s = self._streams.get(sid)
            if s is not None:
                s.ended = True
                if s.frame_q is not None:
                    s.frame_q.put_nowait(None)
                s.event.set()
        elif isinstance(event, h2.events.StreamReset):
            s = self._streams.pop(sid, None)
            if s is not None:
                s.ended = True
                if s.frame_q is not None:
                    s.frame_q.put_nowait(None)
                s.event.set()
        elif isinstance(event, h2.events.ConnectionTerminated):
            self._closed.set()
        elif isinstance(event, h2.events.SettingsAcknowledged):
            pass
        # PingReceived / RemoteSettingsChanged / WindowUpdated etc. — h2
        # already handles these internally (acks, flow control).

    # --- public API -----------------------------------------------------

    async def open_post(
        self,
        *,
        path: str,
        headers: Dict[str, str],
        body: bytes = b"",
        stream_response: bool = False,
        end_stream: bool = True,
    ) -> _Stream:
        """Send :method=POST / headers / body. Returns the stream handle.

        ``stream_response=True`` opts the stream into a ``frame_q`` so a
        caller can pull DATA frames as they arrive (used for parked
        intake streams that may sit idle for minutes).
        """
        async with self._writer_lock:
            assert self._h2 is not None
            sid = self._h2.get_next_available_stream_id()
            full_headers = self._build_request_headers(
                method="POST",
                path=path,
                extras=headers,
            )
            self._h2.send_headers(sid, full_headers, end_stream=False)
            if body:
                self._h2.send_data(sid, body, end_stream=end_stream)
            elif end_stream:
                self._h2.end_stream(sid)
            await self._flush()

        s = _Stream(sid)
        if stream_response:
            s.frame_q = asyncio.Queue()
        self._streams[sid] = s
        return s

    async def open_extended_connect(
        self,
        *,
        path: str,
        protocol: str,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> _Stream:
        """Open an HTTP/2 extended-CONNECT stream — ``:protocol`` set.

        The stream is bidirectional and never ends from our side until we
        explicitly close it. The returned stream's ``frame_q`` drains
        inbound DATA frames as they arrive.
        """
        extras = dict(extra_headers or {})
        async with self._writer_lock:
            assert self._h2 is not None
            sid = self._h2.get_next_available_stream_id()
            full_headers = [
                (":method", "CONNECT"),
                (":scheme", "https"),
                (":authority", self._host),
                (":path", path),
                (":protocol", protocol),
            ]
            for k, v in extras.items():
                full_headers.append((k.lower(), v))
            self._h2.send_headers(sid, full_headers, end_stream=False)
            await self._flush()

        s = _Stream(sid)
        s.frame_q = asyncio.Queue()
        self._streams[sid] = s
        return s

    async def send_data(
        self, stream: _Stream, data: bytes, *, end_stream: bool = False,
    ) -> None:
        async with self._writer_lock:
            assert self._h2 is not None
            # h2 enforces flow-control; segment large payloads.
            offset = 0
            while offset < len(data):
                window = min(
                    self._h2.local_flow_control_window(stream.stream_id),
                    self._h2.max_outbound_frame_size,
                )
                if window <= 0:
                    # Out of flow-control credit — release the writer
                    # lock briefly and retry. The reader loop will
                    # call back here once WindowUpdate frames widen
                    # the window.
                    self._writer_lock.release()
                    try:
                        await asyncio.sleep(0.05)
                    finally:
                        await self._writer_lock.acquire()
                    continue
                chunk = data[offset : offset + window]
                self._h2.send_data(stream.stream_id, chunk, end_stream=False)
                offset += len(chunk)
                await self._flush()
            if end_stream:
                self._h2.end_stream(stream.stream_id)
                await self._flush()

    async def end_stream(self, stream: _Stream) -> None:
        async with self._writer_lock:
            assert self._h2 is not None
            with suppress(h2.exceptions.StreamClosedError):
                self._h2.end_stream(stream.stream_id)
                await self._flush()

    async def reset_stream(self, stream: _Stream) -> None:
        async with self._writer_lock:
            assert self._h2 is not None
            with suppress(h2.exceptions.StreamClosedError):
                self._h2.reset_stream(stream.stream_id)
                await self._flush()
        self._streams.pop(stream.stream_id, None)

    def forget_stream(self, stream: _Stream) -> None:
        self._streams.pop(stream.stream_id, None)

    def _build_request_headers(
        self, *, method: str, path: str, extras: Dict[str, str],
    ) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = [
            (":method", method),
            (":scheme", "https"),
            (":authority", self._host),
            (":path", path),
        ]
        for k, v in extras.items():
            out.append((k.lower(), str(v)))
        return out


# ----------------------------------------------------------------------------
# Envelope codec for the WS bridge
# ----------------------------------------------------------------------------

def _encode_envelope(envelope: Dict[str, Any]) -> bytes:
    payload = json.dumps(envelope, separators=(",", ":")).encode("utf-8")
    return struct.pack(">I", len(payload)) + payload


def _decode_envelopes(buf: bytearray) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    while True:
        if len(buf) < 4:
            return out
        (n,) = struct.unpack(">I", bytes(buf[:4]))
        if len(buf) < 4 + n:
            return out
        payload = bytes(buf[4 : 4 + n])
        del buf[: 4 + n]
        out.append(json.loads(payload.decode("utf-8")))


# ----------------------------------------------------------------------------
# Tunnel client
# ----------------------------------------------------------------------------

class InkboxTunnel:
    """Ngrok replacement for the Inkbox adapter.

    Auto-creates (or reuses) an edge-mode tunnel, then opens a persistent
    HTTP/2 connection to inkboxwire.com. Inbound third-party HTTP /
    WebSocket traffic is dispatched to a local aiohttp server at
    ``http://127.0.0.1:{listen_port}``.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        listen_host: str,
        listen_port: int,
        tunnel_name: str,
        state_path: Optional[str] = None,
        identity_handle: str = "",
        pool_size: int = DEFAULT_POOL_SIZE,
    ) -> None:
        if not H2_AVAILABLE:
            raise RuntimeError(
                "Inkbox tunnel needs h2 — `pip install 'hermes-agent[inkbox]'`"
                " or `pip install h2`.",
            )
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError(
                "Inkbox tunnel needs aiohttp — `pip install aiohttp`.",
            )
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._listen_host = listen_host or "127.0.0.1"
        self._listen_port = listen_port
        self._tunnel_name = tunnel_name
        self._identity_handle = identity_handle
        self._pool_size = max(1, int(pool_size))
        self._state_path = state_path
        self._zone = _tunnel_zone_for(self._base_url)

        self._tunnel_id: Optional[str] = None
        self._connect_secret: Optional[str] = None
        self._owner_token: Optional[str] = None
        self._driver: Optional[_H2Driver] = None
        self._intake_tasks: List[asyncio.Task[None]] = []
        self._stop_evt = asyncio.Event()
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._rest: Optional[_RestClient] = None
        self._supervisor_task: Optional[asyncio.Task[None]] = None
        # Stamped each time /_system/hello succeeds; reset to None on
        # disconnect.  The adapter watchdog reads ``connected_seconds``
        # off this for its periodic heartbeat.
        self._connected_at: Optional[float] = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def public_url(self) -> str:
        return f"https://{self.public_host}"

    @property
    def public_host(self) -> str:
        return f"{self._tunnel_name}.{self._zone}"

    @property
    def tunnel_id(self) -> Optional[str]:
        return self._tunnel_id

    @property
    def tunnel_name(self) -> str:
        return self._tunnel_name

    @property
    def connected_seconds(self) -> float:
        """Seconds since the most-recent ``/_system/hello`` succeeded; 0 if never."""
        if self._connected_at is None:
            return 0.0
        return max(0.0, time.time() - self._connected_at)

    def is_alive(self) -> bool:
        """True iff the supervisor task is still running.

        Distinct from ``connected_seconds > 0``: the supervisor handles
        normal disconnect → reconnect cycles itself, so a brief moment
        with ``_connected_at is None`` (between sessions) is healthy.
        Only when the supervisor task itself is gone is the tunnel
        truly dead and in need of external revival.
        """
        if self._stop_evt.is_set():
            return False
        return (
            self._supervisor_task is not None
            and not self._supervisor_task.done()
        )

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    async def ensure_tunnel(self) -> Dict[str, Any]:
        """Resolve the tunnel via REST: reuse saved state, else create.

        Saved state is keyed by tunnel name + identity handle. If the saved
        ID still resolves to an active tunnel on the server, reuse it; if
        the tunnel was deleted server-side, create a new one. Returns the
        tunnel record dict.
        """
        self._rest = _RestClient(api_key=self._api_key, base_url=self._base_url)

        saved = self._load_state()
        if saved.get("tunnel_id") and saved.get("connect_secret"):
            try:
                record = await self._rest.get_tunnel(saved["tunnel_id"])
            except TunnelControlPlaneError as exc:
                logger.warning(
                    "[InkboxTunnel] Could not verify saved tunnel %s: %s",
                    saved["tunnel_id"], exc,
                )
                record = None
            if (
                record is not None
                and record.get("status") == "active"
                and record.get("tunnel_name") == self._tunnel_name
            ):
                self._tunnel_id = record["id"]
                self._connect_secret = saved["connect_secret"]
                logger.info(
                    "[InkboxTunnel] Reusing saved tunnel %s (%s)",
                    record["tunnel_name"], record["id"],
                )
                return record
            logger.info(
                "[InkboxTunnel] Saved tunnel state stale "
                "(status=%s, name=%s); creating fresh tunnel.",
                record.get("status") if record else "missing",
                record.get("tunnel_name") if record else "n/a",
            )

        # Look for an existing org-owned tunnel under this name before
        # creating one — same identity rerunning on a fresh box should
        # land on the same tunnel as long as the secret is recoverable.
        try:
            existing = await self._rest.list_tunnels()
        except TunnelControlPlaneError as exc:
            logger.warning("[InkboxTunnel] list_tunnels failed: %s", exc)
            existing = []
        for record in existing:
            if (
                record.get("tunnel_name") == self._tunnel_name
                and record.get("status") == "active"
            ):
                logger.info(
                    "[InkboxTunnel] Found existing tunnel %s (%s) under our name; "
                    "rotating connect secret.",
                    record["tunnel_name"], record["id"],
                )
                # We can't read the existing secret (it was shown once); the
                # only path back is rotate-secret, which invalidates any
                # prior client. Safe here because we ARE the only client.
                new_secret = await self._rest.rotate_secret(record["id"])
                self._tunnel_id = record["id"]
                self._connect_secret = new_secret
                self._save_state()
                return record

        # Create a fresh edge-mode tunnel.
        description = (
            f"Hermes agent gateway for identity {self._identity_handle!r}"
            if self._identity_handle else "Hermes agent gateway"
        )
        record, secret = await self._rest.create_tunnel(
            tunnel_name=self._tunnel_name,
            description=description,
        )
        self._tunnel_id = record["id"]
        self._connect_secret = secret
        self._save_state()
        logger.info(
            "[InkboxTunnel] Created tunnel %s (%s); status=%s",
            record["tunnel_name"], record["id"], record.get("status"),
        )
        return record

    def _load_state(self) -> Dict[str, Any]:
        if not self._state_path:
            return {}
        try:
            with open(self._state_path, "r", encoding="utf-8") as fp:
                return json.load(fp) or {}
        except FileNotFoundError:
            return {}
        except Exception as exc:
            logger.warning(
                "[InkboxTunnel] Failed to read saved state %s: %s",
                self._state_path, exc,
            )
            return {}

    def _save_state(self) -> None:
        if not self._state_path:
            return
        try:
            os.makedirs(os.path.dirname(self._state_path), exist_ok=True)
            payload = {
                "tunnel_id": self._tunnel_id,
                "tunnel_name": self._tunnel_name,
                "connect_secret": self._connect_secret,
                "zone": self._zone,
                "saved_at": time.time(),
            }
            with open(self._state_path, "w", encoding="utf-8") as fp:
                json.dump(payload, fp, indent=2)
            os.chmod(self._state_path, 0o600)
        except Exception as exc:
            logger.warning(
                "[InkboxTunnel] Failed to write state %s: %s",
                self._state_path, exc,
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> bool:
        """Open the persistent connection and start the intake pool."""
        if not self._tunnel_id or not self._connect_secret:
            await self.ensure_tunnel()
        assert self._tunnel_id and self._connect_secret

        self._http_session = aiohttp.ClientSession()
        self._supervisor_task = asyncio.create_task(
            self._supervisor(), name="inkbox-tunnel-supervisor",
        )
        # Wait briefly for the first hello to land so the caller knows
        # the tunnel is live before connect() returns.
        for _ in range(200):  # ~10s @ 50ms
            if self._owner_token:
                return True
            if self._stop_evt.is_set():
                return False
            await asyncio.sleep(0.05)
        return self._owner_token is not None

    async def stop(self) -> None:
        self._stop_evt.set()
        for t in self._intake_tasks:
            if not t.done():
                t.cancel()
        if self._supervisor_task is not None and not self._supervisor_task.done():
            self._supervisor_task.cancel()
            with suppress(Exception):
                await self._supervisor_task
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
        if self._http_session is not None:
            with suppress(Exception):
                await self._http_session.close()
            self._http_session = None
        if self._rest is not None:
            with suppress(Exception):
                await self._rest.aclose()
            self._rest = None

    async def _supervisor(self) -> None:
        """Open the h2 connection; reconnect with backoff on drop."""
        attempt = 0
        while not self._stop_evt.is_set():
            try:
                await self._run_session()
                attempt = 0  # clean disconnect; restart immediately
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.warning(
                    "[InkboxTunnel] session error: %s; reconnecting",
                    exc, exc_info=False,
                )
            if self._stop_evt.is_set():
                return
            backoff = INTAKE_RECONNECT_BACKOFF[
                min(attempt, len(INTAKE_RECONNECT_BACKOFF) - 1)
            ]
            attempt += 1
            try:
                await asyncio.wait_for(self._stop_evt.wait(), timeout=backoff)
                return
            except asyncio.TimeoutError:
                pass

    async def _run_session(self) -> None:
        """One pass of (connect → hello → park intakes) until the conn drops."""
        driver = _H2Driver(host=self.public_host, port=DEFAULT_TUNNEL_PORT)
        await driver.connect()
        self._driver = driver

        # /_system/hello
        s = await driver.open_post(
            path="/_system/hello",
            headers={
                "x-tunnel-id": self._tunnel_id or "",
                "x-tunnel-secret": self._connect_secret or "",
                "x-pool-size": str(self._pool_size),
                "user-agent": "hermes-inkbox-tunnel/1",
            },
        )
        await self._await_stream_done(s, timeout=15.0)
        status = _h2_status(s.headers)
        if status != 200:
            raise TunnelControlPlaneError(
                f"/_system/hello failed: status={status} body={bytes(s.data)!r}",
            )
        try:
            payload = json.loads(bytes(s.data) or b"{}")
        except json.JSONDecodeError as exc:
            raise TunnelControlPlaneError(f"Bad hello payload: {exc}")
        self._owner_token = payload["owner_token"]
        self._connected_at = time.time()
        pool_size = int(payload.get("default_pool_size") or self._pool_size)
        logger.info(
            "[InkboxTunnel] Connected to %s — tunnel=%s owner=%s pool=%d",
            self.public_host, self._tunnel_id, self._owner_token, pool_size,
        )

        # Park N intake streams.
        self._intake_tasks = [
            asyncio.create_task(
                self._intake_loop(slot=i),
                name=f"inkbox-tunnel-intake-{i}",
            )
            for i in range(pool_size)
        ]

        # Park here until the connection drops.
        await driver.closed.wait()
        # Cancel parkers so they can't try to use a dead connection.
        for t in self._intake_tasks:
            if not t.done():
                t.cancel()
        self._intake_tasks = []
        self._owner_token = None
        self._connected_at = None
        if self._driver is driver:
            self._driver = None

    # ------------------------------------------------------------------
    # Intake pool
    # ------------------------------------------------------------------

    async def _intake_loop(self, *, slot: int) -> None:
        while not self._stop_evt.is_set():
            driver = self._driver
            if driver is None or driver.closed.is_set():
                return
            try:
                await self._intake_once(driver=driver, slot=slot)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.debug(
                    "[InkboxTunnel] intake slot=%d error: %s",
                    slot, exc, exc_info=False,
                )
                # Yield briefly so we don't spin on a permanently-failing
                # condition (e.g. owner_token revoked); the supervisor's
                # reconnect path handles real recovery.
                await asyncio.sleep(0.5)

    async def _intake_once(self, *, driver: _H2Driver, slot: int) -> None:
        s = await driver.open_post(
            path="/_system/intake",
            headers={
                "x-tunnel-id": self._tunnel_id or "",
                "x-owner-token": self._owner_token or "",
                "x-pool-slot": str(slot),
            },
            stream_response=True,
        )
        # First response HEADERS arrive whenever the server dispatches.
        await s.event.wait()
        status = _h2_status(s.headers)
        if status == 408:
            # Idle cap — refill the slot. Drain the empty body if any.
            await self._drain_stream(s)
            return
        if status != 200:
            await self._drain_stream(s)
            raise TunnelControlPlaneError(
                f"intake slot={slot} unexpected status={status}",
            )

        meta = _meta_from_headers(s.headers)
        body = await self._collect_body(s)

        request_id = meta.get(META_REQUEST_ID, "")
        route_kind = meta.get(META_ROUTE_KIND, ROUTE_WEBHOOK)

        if route_kind == ROUTE_WEBHOOK:
            asyncio.create_task(
                self._handle_webhook(request_id=request_id, meta=meta, body=body),
                name=f"inkbox-tunnel-webhook-{request_id}",
            )
        elif route_kind == ROUTE_WS_UPGRADE:
            asyncio.create_task(
                self._handle_ws_upgrade(request_id=request_id, meta=meta),
                name=f"inkbox-tunnel-ws-{request_id}",
            )
        else:
            logger.info(
                "[InkboxTunnel] Ignoring envelope route_kind=%r request_id=%s",
                route_kind, request_id,
            )

    async def _await_stream_done(
        self, s: _Stream, *, timeout: float,
    ) -> None:
        deadline = asyncio.get_event_loop().time() + timeout
        while not s.ended:
            await asyncio.wait_for(
                s.event.wait(),
                timeout=max(0.1, deadline - asyncio.get_event_loop().time()),
            )
            s.event.clear()
        if s.frame_q is not None:
            # Drain any queued data into s.data so callers can parse it.
            while True:
                try:
                    chunk = s.frame_q.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if chunk is None:
                    break
                s.data.extend(chunk)

    async def _drain_stream(self, s: _Stream) -> None:
        if s.frame_q is None:
            # Wait for end_stream.
            while not s.ended:
                await s.event.wait()
                s.event.clear()
            return
        while True:
            chunk = await s.frame_q.get()
            if chunk is None:
                return
            s.data.extend(chunk)

    async def _collect_body(self, s: _Stream) -> bytes:
        """Drain the parked stream until end_stream; return concatenated bytes."""
        if s.frame_q is None:
            return bytes(s.data)
        out = bytearray()
        while True:
            chunk = await s.frame_q.get()
            if chunk is None:
                break
            out.extend(chunk)
        return bytes(out)

    # ------------------------------------------------------------------
    # Webhook dispatch (ROUTE_WEBHOOK)
    # ------------------------------------------------------------------

    async def _handle_webhook(
        self, *, request_id: str, meta: Dict[str, str], body: bytes,
    ) -> None:
        method = meta.get(META_METHOD, "POST")
        path = meta.get(META_PATH, "/")

        # Reconstruct the third-party request headers.
        forwarded: Dict[str, str] = {}
        for k, v in meta.items():
            if k.startswith(INKBOX_FORWARDED_HEADER_PREFIX):
                forwarded[k[len(INKBOX_FORWARDED_HEADER_PREFIX):]] = v

        # Honor the body-offload pointer if present (feature-flagged off
        # by default on the server today; we cover the path anyway).
        body_uri = meta.get(META_BODY_URI, "")
        if body_uri and not body and self._http_session is not None:
            try:
                async with self._http_session.get(body_uri) as r:
                    body = await r.read()
            except Exception as exc:
                logger.warning(
                    "[InkboxTunnel] body-uri fetch failed for %s: %s",
                    request_id, exc,
                )

        local_url = f"http://{self._listen_host}:{self._listen_port}{path}"
        try:
            assert self._http_session is not None
            async with self._http_session.request(
                method=method,
                url=local_url,
                data=body if body else None,
                headers=forwarded,
                allow_redirects=False,
            ) as resp:
                resp_body = await resp.read()
                resp_headers = {k: v for k, v in resp.headers.items()}
                resp_status = resp.status
        except Exception as exc:
            logger.exception(
                "[InkboxTunnel] local dispatch failed for %s %s: %s",
                method, path, exc,
            )
            resp_body = f"local handler error: {exc}".encode("utf-8")
            resp_status = 502
            resp_headers = {"content-type": "text/plain; charset=utf-8"}

        await self._post_response(
            request_id=request_id,
            status=resp_status,
            headers=resp_headers,
            body=resp_body,
        )

    async def _post_response(
        self,
        *,
        request_id: str,
        status: int,
        headers: Dict[str, str],
        body: bytes,
    ) -> None:
        driver = self._driver
        if driver is None or driver.closed.is_set():
            logger.warning(
                "[InkboxTunnel] No live driver to post response for %s",
                request_id,
            )
            return

        meta_headers: Dict[str, str] = {
            META_STATUS: str(status),
        }
        for k, v in headers.items():
            lk = k.lower()
            if lk in _WS_HOPBYHOP and lk != "content-length":
                continue
            if lk == "content-length":
                continue
            meta_headers[f"{INKBOX_FORWARDED_HEADER_PREFIX}{lk}"] = v

        try:
            s = await driver.open_post(
                path=f"/_system/response/{request_id}",
                headers={
                    "x-tunnel-id": self._tunnel_id or "",
                    **meta_headers,
                    "content-length": str(len(body)),
                },
                body=body,
                end_stream=True,
            )
            await self._await_stream_done(s, timeout=10.0)
            driver.forget_stream(s)
        except Exception as exc:
            logger.warning(
                "[InkboxTunnel] post_response failed for %s: %s",
                request_id, exc,
            )

    # ------------------------------------------------------------------
    # WebSocket bridge (ROUTE_WS_UPGRADE)
    # ------------------------------------------------------------------

    async def _handle_ws_upgrade(
        self, *, request_id: str, meta: Dict[str, str],
    ) -> None:
        ws_id = meta.get(META_WS_ID) or request_id
        path = meta.get(META_PATH, "/")
        offered_subprotocols_raw = meta.get(
            f"{INKBOX_FORWARDED_HEADER_PREFIX}sec-websocket-protocol", "",
        )
        offered_subprotocols = [
            p.strip() for p in offered_subprotocols_raw.split(",") if p.strip()
        ]

        forwarded_headers: Dict[str, str] = {}
        for k, v in meta.items():
            if k.startswith(INKBOX_FORWARDED_HEADER_PREFIX):
                forwarded_headers[k[len(INKBOX_FORWARDED_HEADER_PREFIX):]] = v

        local_url = f"ws://{self._listen_host}:{self._listen_port}{path}"
        local_ws: Optional[aiohttp.ClientWebSocketResponse] = None
        bridge_stream: Optional[_Stream] = None
        chosen_subprotocol: Optional[str] = None
        accept_extra_headers: Dict[str, str] = {}

        try:
            assert self._http_session is not None
            # Strip hop-by-hop headers before forwarding to the local WS.
            ws_headers = {
                k: v for k, v in forwarded_headers.items()
                if k.lower() not in _WS_HOPBYHOP
            }
            local_ws = await self._http_session.ws_connect(
                local_url,
                protocols=offered_subprotocols or (),
                headers=ws_headers,
                heartbeat=20.0,
            )
            chosen_subprotocol = local_ws.protocol
            # Forward any non-hop-by-hop response headers from the
            # upstream handshake — the adapter sets a couple of these
            # (x-use-inkbox-text-to-speech, x-use-inkbox-speech-to-text).
            for k, v in (local_ws._response.headers.items()  # type: ignore[attr-defined]
                         if hasattr(local_ws, "_response") else []):
                lk = k.lower()
                if lk in _WS_HOPBYHOP:
                    continue
                accept_extra_headers[lk] = v
        except Exception as exc:
            logger.warning(
                "[InkboxTunnel] local ws connect failed (%s): %s", local_url, exc,
            )
            await self._post_response(
                request_id=ws_id,
                status=502,
                headers={"content-type": "text/plain; charset=utf-8"},
                body=f"local ws handler unavailable: {exc}".encode("utf-8"),
            )
            return

        # Reply to the upgrade envelope (status=101, chosen subproto).
        upgrade_reply_headers: Dict[str, str] = dict(accept_extra_headers)
        if chosen_subprotocol:
            upgrade_reply_headers["sec-websocket-protocol"] = chosen_subprotocol
        await self._post_response(
            request_id=ws_id,
            status=101,
            headers=upgrade_reply_headers,
            body=b"",
        )

        # Open the extended-CONNECT bridge stream.
        driver = self._driver
        if driver is None or driver.closed.is_set():
            with suppress(Exception):
                await local_ws.close()
            return
        try:
            bridge_stream = await driver.open_extended_connect(
                path=f"/_system/ws/{ws_id}",
                protocol=WS_SUBPROTOCOL,
            )
        except Exception as exc:
            logger.warning(
                "[InkboxTunnel] open_extended_connect failed for ws %s: %s",
                ws_id, exc,
            )
            with suppress(Exception):
                await local_ws.close()
            return

        # Wait for the server's ":status: 200" response (CONNECT accept).
        try:
            await asyncio.wait_for(bridge_stream.event.wait(), timeout=15.0)
        except asyncio.TimeoutError:
            logger.warning(
                "[InkboxTunnel] bridge accept timed out for ws %s", ws_id,
            )
            await driver.reset_stream(bridge_stream)
            with suppress(Exception):
                await local_ws.close()
            return
        bridge_stream.event.clear()
        bridge_status = _h2_status(bridge_stream.headers)
        if bridge_status != 200:
            logger.warning(
                "[InkboxTunnel] bridge accept rejected for ws %s: status=%s",
                ws_id, bridge_status,
            )
            with suppress(Exception):
                await local_ws.close()
            driver.forget_stream(bridge_stream)
            return

        # Pump in both directions until either side closes.
        try:
            await asyncio.gather(
                self._pump_local_to_bridge(local_ws=local_ws, stream=bridge_stream),
                self._pump_bridge_to_local(local_ws=local_ws, stream=bridge_stream),
            )
        finally:
            with suppress(Exception):
                await local_ws.close()
            assert self._driver is not None
            with suppress(Exception):
                await self._driver.reset_stream(bridge_stream)

    async def _pump_local_to_bridge(
        self,
        *,
        local_ws: aiohttp.ClientWebSocketResponse,
        stream: _Stream,
    ) -> None:
        driver = self._driver
        assert driver is not None
        try:
            async for msg in local_ws:
                if msg.type == WSMsgType.TEXT:
                    env = {"type": "text", "data": msg.data}
                elif msg.type == WSMsgType.BINARY:
                    env = {
                        "type": "binary",
                        "data": base64.b64encode(msg.data).decode("ascii"),
                    }
                elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.CLOSING):
                    env = {"type": "close", "code": int(local_ws.close_code or 1000)}
                    await driver.send_data(stream, _encode_envelope(env))
                    break
                else:
                    continue
                await driver.send_data(stream, _encode_envelope(env))
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.debug("[InkboxTunnel] local→bridge pump ended: %s", exc)

    async def _pump_bridge_to_local(
        self,
        *,
        local_ws: aiohttp.ClientWebSocketResponse,
        stream: _Stream,
    ) -> None:
        assert stream.frame_q is not None
        buf = bytearray()
        try:
            while True:
                chunk = await stream.frame_q.get()
                if chunk is None:
                    return
                buf.extend(chunk)
                for env in _decode_envelopes(buf):
                    kind = env.get("type")
                    if kind == "text":
                        await local_ws.send_str(env.get("data") or "")
                    elif kind == "binary":
                        try:
                            data = base64.b64decode(env.get("data") or "")
                        except Exception:
                            continue
                        await local_ws.send_bytes(data)
                    elif kind == "close":
                        with suppress(Exception):
                            await local_ws.close(code=int(env.get("code") or 1000))
                        return
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.debug("[InkboxTunnel] bridge→local pump ended: %s", exc)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _h2_status(headers: List[Tuple[str, str]]) -> int:
    for k, v in headers:
        if k == ":status":
            try:
                return int(v)
            except ValueError:
                return 0
    return 0


def _meta_from_headers(headers: List[Tuple[str, str]]) -> Dict[str, str]:
    """Drop pseudo-headers and pass everything else through verbatim."""
    out: Dict[str, str] = {}
    for k, v in headers:
        if k.startswith(":"):
            continue
        out[k.lower()] = v
    return out


def derive_tunnel_name(*, identity_handle: str, override: str = "") -> str:
    """Pick a tunnel name: explicit override wins, else slug of identity."""
    if override:
        return override.strip().lower()
    return _slug_for_identity(identity_handle)


__all__ = [
    "InkboxTunnel",
    "TunnelControlPlaneError",
    "derive_tunnel_name",
    "H2_AVAILABLE",
]
