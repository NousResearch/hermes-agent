"""
Rabbit R1 platform adapter for Hermes Agent (bundled plugin form).

Speaks the OpenClaw/clawdbot-gateway WebSocket protocol so a Rabbit R1
hardware device can talk to Hermes (full memory, skills, crons) from
anywhere - not just the home WiFi.

Unlike the official OpenClaw setup (LAN-only), this adapter runs on a VM or
always-on server behind a tunnel (Tailscale Funnel or Cloudflare Tunnel) so
the R1 works from any network: home, cellular, travelling.

Architecture::

    R1 (anywhere with internet)
        v  wss://yourname.ts.net  (TLS via Tailscale Funnel)
    VM / always-on server
        v
    adapter.py  (this file - BasePlatformAdapter subclass)
        v
    Hermes gateway -> Claude / local model (full memory, skills, crons)

Protocol reference::

    QR payload:  {"type":"clawdbot-gateway","version":1,"ips":[...],
                  "port":18789,"token":"<hex32>","protocol":"ws"}
    Handshake:   connect.challenge -> connect -> node.pair.approved -> connect.ok
    Chat:        chat.send (R1->server) / chat event (server->R1)

Tunnel options (RABBIT_R1_TUNNEL)::

    tailscale    Tailscale Funnel (default, no extra account)
    cloudflare   Cloudflare Quick Tunnel (free account, stable URL)
    none         LAN only (home network)

This adapter registers itself through ``register(ctx)`` and requires **zero
changes to core Hermes code**. Every integration point the older built-in
version patched (platform enum, env overrides, auth maps, cron delivery,
send routing, prompt hint, status/setup/channel-directory) is now supplied
via ``ctx.register_platform()`` keyword arguments and ``plugin.yaml``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import socket
import subprocess
import time
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    WebSocketServerProtocol = Any  # type: ignore

try:
    import qrcode
    QRCODE_AVAILABLE = True
except ImportError:
    QRCODE_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_PORT = 18789
DEFAULT_KEEPALIVE_INTERVAL = 300  # 5 min - well under the R1's ~30 min idle cap
DEFAULT_TUNNEL = "tailscale"

# R1 has no hard message length limit, but keep responses readable on the
# small 2.88-inch screen.
R1_MAX_MESSAGE_LENGTH = 2000

# 64-char hex device IDs - redacted from this adapter's own log output.
_R1_DEVICE_ID_RE = re.compile(r"(?<![A-Za-z0-9])[a-f0-9]{64}(?![A-Za-z0-9])")


def _redact(text: str) -> str:
    """Mask 64-char hex device IDs in log strings (plugin-local redaction)."""
    return _R1_DEVICE_ID_RE.sub("[R1_DEVICE_ID]", str(text))


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

def check_requirements() -> bool:
    """Return True if the required dependencies are available."""
    if not WEBSOCKETS_AVAILABLE:
        logger.warning(
            "Rabbit R1: 'websockets' package not installed. Run: pip install websockets"
        )
        return False
    return True


def validate_config(config) -> bool:
    """Rabbit R1 needs no external credential - the token is auto-generated.

    We just require the websockets dependency to be importable so the
    adapter can actually stand up its server.
    """
    return WEBSOCKETS_AVAILABLE


def is_connected(config) -> bool:
    """Surface in ``hermes status`` / ``get_connected_platforms`` when enabled.

    The platform is considered connected/enabled whenever its dependency is
    present, since it stands up its own server rather than dialing out to a
    credentialed service.
    """
    return WEBSOCKETS_AVAILABLE


# ---------------------------------------------------------------------------
# Tunnel helpers
# ---------------------------------------------------------------------------

def _get_tailscale_funnel_url(port: int) -> Optional[str]:
    """Start a Tailscale Funnel on *port* and return the public wss:// URL.

    Returns None if Tailscale is unavailable or the command fails.
    """
    try:
        # Enable funnel for the port (idempotent - safe to call repeatedly).
        subprocess.run(
            ["tailscale", "funnel", str(port)],
            check=True,
            capture_output=True,
            timeout=15,
        )
        # Get the stable public hostname.
        result = subprocess.run(
            ["tailscale", "status", "--json"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        status = json.loads(result.stdout)
        dns_name = status["Self"]["DNSName"].rstrip(".")
        return f"wss://{dns_name}"
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            KeyError, json.JSONDecodeError, FileNotFoundError):
        return None


def _get_cloudflare_tunnel_url(port: int) -> Optional[str]:
    """Start a Cloudflare Quick Tunnel (trycloudflare.com) and return wss:// URL.

    This is a temporary URL - use Tailscale or a named Cloudflare tunnel for
    stability.
    """
    try:
        proc = subprocess.Popen(
            ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Parse the tunnel URL from cloudflared's stderr output.
        for _ in range(30):
            line = proc.stderr.readline()
            if not line:
                break
            match = re.search(r"https://[a-z0-9\-]+\.trycloudflare\.com", line)
            if match:
                return match.group(0).replace("https://", "wss://")
        return None
    except FileNotFoundError:
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _now_ms() -> int:
    """Current time in milliseconds."""
    return int(time.time() * 1000)


def _get_lan_ip() -> str:
    """Best-effort LAN IP detection for the QR code fallback."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def _resolve_token(config) -> str:
    """Resolve the auth token from env, config, or auto-generate one."""
    extra = getattr(config, "extra", {}) or {}
    return (
        os.getenv("RABBIT_R1_TOKEN")
        or getattr(config, "token", None)
        or extra.get("token")
        or secrets.token_hex(32)
    )


# ---------------------------------------------------------------------------
# Main adapter
# ---------------------------------------------------------------------------

class RabbitR1Adapter(BasePlatformAdapter):
    """
    Rabbit R1 platform adapter.

    Runs a WebSocket server that speaks the clawdbot-gateway protocol. On
    startup it optionally opens a Tailscale Funnel or Cloudflare Tunnel so
    the R1 can reach it from anywhere, then prints a pairing QR code.

    Config env vars:
        RABBIT_R1_TOKEN     - hex32 auth token (auto-generated if not set)
        RABBIT_R1_PORT      - WebSocket server port (default: 18789)
        RABBIT_R1_TUNNEL    - "tailscale" | "cloudflare" | "none"
        RABBIT_R1_PUBLIC_URL - explicit public wss:// URL (overrides tunnel)
        RABBIT_R1_KEEPALIVE_INTERVAL - seconds between heartbeats (default 300)
    """

    MAX_MESSAGE_LENGTH = R1_MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("rabbit_r1"))

        extra = getattr(config, "extra", {}) or {}

        self._port: int = int(
            os.getenv("RABBIT_R1_PORT") or extra.get("port") or DEFAULT_PORT
        )
        self._tunnel_mode: str = str(
            os.getenv("RABBIT_R1_TUNNEL") or extra.get("tunnel") or DEFAULT_TUNNEL
        ).lower()

        self._token: str = _resolve_token(config)

        # Runtime state
        self._server = None
        self._public_url: Optional[str] = None

        # device_id -> websocket for connected R1 devices
        self._clients: Dict[str, WebSocketServerProtocol] = {}

        # Server->R1 keepalive: prevents the R1 from timing out the session
        # due to inactivity. Default 5 minutes keeps us well under the R1's
        # ~30-minute inactivity threshold.
        self._keepalive_interval: int = int(
            os.getenv("RABBIT_R1_KEEPALIVE_INTERVAL")
            or extra.get("keepalive_interval")
            or DEFAULT_KEEPALIVE_INTERVAL
        )
        self._keepalive_tasks: Dict[str, asyncio.Task] = {}

        # Rate limiting: track failed auth attempts per IP.
        self._auth_failures: Dict[str, List[float]] = {}
        self._max_auth_failures = 5      # max failures per window
        self._auth_window_secs = 300.0   # 5-minute window

    # ------------------------------------------------------------------
    # BasePlatformAdapter - required methods
    # ------------------------------------------------------------------

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Start the WebSocket server and (optionally) open the tunnel."""
        if not check_requirements():
            return False

        # Start the tunnel first so we know the public URL for the QR code.
        self._public_url = await self._start_tunnel()

        # Start the WebSocket server.
        try:
            self._server = await websockets.serve(
                self._handle_connection,
                "0.0.0.0",
                self._port,
            )
            logger.info("Rabbit R1: WebSocket server listening on port %s", self._port)
        except OSError as e:
            logger.error("Rabbit R1: Failed to start WebSocket server: %s", e)
            return False

        self._mark_connected()

        # Print pairing instructions + QR code to the console.
        await self._print_pairing_info()
        return True

    async def disconnect(self) -> None:
        """Stop the WebSocket server and cancel keepalive tasks."""
        for device_id in list(self._keepalive_tasks):
            self._stop_keepalive(device_id)
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        self._clients.clear()
        self._mark_disconnected()
        logger.info("Rabbit R1: disconnected")

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a text reply back to the R1 device identified by *chat_id*."""
        ws = self._clients.get(chat_id)
        if not ws:
            return SendResult(success=False, error="Device not connected")

        run_id = str(uuid.uuid4())
        payload = {
            "type": "event",
            "event": "chat",
            "payload": {
                "runId": run_id,
                "sessionKey": "main",
                "seq": 1,
                "state": "final",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": content}],
                    "timestamp": _now_ms(),
                    "stopReason": "stop",
                    "usage": {"input": 0, "output": 0, "totalTokens": 0},
                },
            },
        }
        try:
            await ws.send(json.dumps(payload))
            return SendResult(success=True, message_id=run_id)
        except Exception as e:
            logger.warning("Rabbit R1: send failed for %s: %s", _redact(chat_id), e)
            return SendResult(success=False, error=str(e))

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return metadata about the chat (device) identified by *chat_id*."""
        return {
            "name": "Rabbit R1",
            "type": "dm",
            "chat_id": chat_id,
            "connected": chat_id in self._clients,
        }

    def format_message(self, content: str) -> str:
        """Strip markdown for the R1's small screen (plain-text rendering)."""
        # Bold / italic
        content = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', content)
        content = re.sub(r'_{1,3}(.+?)_{1,3}', r'\1', content)
        # Links [text](url) -> text (url)
        content = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1 (\2)', content)
        # Headers
        content = re.sub(r'^#{1,6}\s+', '', content, flags=re.MULTILINE)
        # Code fences / inline code
        content = re.sub(r'```\w*\n?', '', content)
        content = re.sub(r'`([^`]+)`', r'\1', content)
        return content.strip()

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Send a 'thinking' state to the R1 - shows a loading indicator."""
        ws = self._clients.get(chat_id)
        if not ws:
            return
        payload = {
            "type": "event",
            "event": "chat",
            "payload": {
                "runId": str(uuid.uuid4()),
                "sessionKey": "main",
                "seq": 0,
                "state": "thinking",
                "message": {
                    "role": "assistant",
                    "content": [],
                    "timestamp": _now_ms(),
                },
            },
        }
        try:
            await ws.send(json.dumps(payload))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # WebSocket connection handling
    # ------------------------------------------------------------------

    async def _handle_connection(self, ws, path: str = "/") -> None:
        """Handle a new WebSocket connection from an R1 device."""
        remote = f"{ws.remote_address[0]}:{ws.remote_address[1]}"
        logger.debug("Rabbit R1: new connection from %s", remote)

        # Step 1 - send the challenge immediately.
        nonce = str(uuid.uuid4())
        await self._send(ws, {
            "type": "event",
            "event": "connect.challenge",
            "payload": {"nonce": nonce, "ts": _now_ms()},
        })

        device_id: Optional[str] = None
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("Rabbit R1: invalid JSON from %s", remote)
                    continue

                method = msg.get("method") or msg.get("type", "")

                # Pairing / auth handshake
                if method in ("connect", "gateway.connect"):
                    device_id = await self._handle_connect(ws, msg, remote)
                    if device_id is None:
                        break  # auth failed - connection closed inside handler
                    continue

                # Drop everything until the device is authenticated.
                if device_id is None:
                    logger.warning(
                        "Rabbit R1: unauthenticated message from %s, ignoring", remote
                    )
                    continue

                if method == "chat.send":
                    await self._handle_chat_send(ws, msg, device_id)

                elif method == "system-presence":
                    # Heartbeat - just acknowledge it.
                    await self._send(ws, {
                        "type": "res",
                        "id": msg.get("id"),
                        "ok": True,
                        "payload": {"ts": _now_ms()},
                    })

                elif method == "chat.abort":
                    # Abort an in-progress generation.
                    await self.cancel_background_tasks()
                    await self._send(ws, {"type": "res", "id": msg.get("id"), "ok": True})

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            if device_id:
                self._stop_keepalive(device_id)
                if device_id in self._clients:
                    del self._clients[device_id]
                    logger.info("Rabbit R1: device disconnected: %s", _redact(device_id))

    async def _handle_connect(
        self,
        ws: WebSocketServerProtocol,
        msg: dict,
        remote: str,
    ) -> Optional[str]:
        """Validate the token and complete the pairing handshake.

        Returns the device_id on success, None on failure.
        """
        msg_id = msg.get("id")

        # Rate-limit: reject if too many recent failures from this IP.
        ip = remote.rsplit(":", 1)[0]
        now = time.time()
        failures = [t for t in self._auth_failures.get(ip, [])
                    if now - t < self._auth_window_secs]
        self._auth_failures[ip] = failures
        if len(failures) >= self._max_auth_failures:
            logger.warning(
                "Rabbit R1: rate-limited %s (%d failures in %ss)",
                remote, len(failures), self._auth_window_secs,
            )
            await self._send(ws, {
                "type": "res",
                "id": msg_id,
                "ok": False,
                "error": {"code": 429, "message": "Too many failed attempts"},
            })
            await ws.close()
            return None

        params = msg.get("params", {})

        # Extract token - R1 sends it at params.auth.token.
        client_token = (
            params.get("auth", {}).get("token")
            or params.get("authToken")
            or msg.get("token")
        )
        # Extract device ID.
        device_id = (
            params.get("device", {}).get("id")
            or params.get("deviceId")
            or f"r1-{remote}"
        )

        if not secrets.compare_digest(
            (client_token or "").encode(), self._token.encode()
        ):
            logger.warning("Rabbit R1: auth failed from %s (bad token)", remote)
            self._auth_failures.setdefault(ip, []).append(now)
            await self._send(ws, {
                "type": "res",
                "id": msg_id,
                "ok": False,
                "error": {"code": 401, "message": "Invalid token"},
            })
            await ws.close()
            return None

        # Auth passed - register the device.
        self._clients[device_id] = ws
        self._start_keepalive(device_id, ws)
        logger.info("Rabbit R1: device paired: %s from %s", _redact(device_id), remote)

        await self._send(ws, {
            "type": "event",
            "event": "node.pair.approved",
            "payload": {"deviceId": device_id, "token": str(uuid.uuid4())},
        })
        await self._send(ws, {
            "type": "res",
            "id": msg_id,
            "ok": True,
            "payload": {"status": "paired", "ts": _now_ms()},
        })
        await self._send(ws, {
            "type": "event",
            "event": "connect.ok",
            "payload": {"deviceId": device_id, "ts": _now_ms()},
        })
        return device_id

    async def _handle_chat_send(
        self,
        ws: WebSocketServerProtocol,
        msg: dict,
        device_id: str,
    ) -> None:
        """Route an incoming chat.send message into the Hermes pipeline."""
        params = msg.get("params", {})
        text = (params.get("message") or "").strip()
        if not text:
            return

        # Acknowledge immediately so the R1 doesn't time out.
        await self._send(ws, {"type": "res", "id": msg.get("id"), "ok": True})

        source = self.build_source(
            chat_id=device_id,
            chat_name="Rabbit R1",
            chat_type="dm",
            user_id=device_id,
            user_name="R1 User",
        )
        event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            message_id=params.get("idempotencyKey") or str(uuid.uuid4()),
        )
        await self.handle_message(event)

    # ------------------------------------------------------------------
    # Server->R1 keepalive
    # ------------------------------------------------------------------

    def _start_keepalive(self, device_id: str, ws: WebSocketServerProtocol) -> None:
        """Start a background task that sends periodic heartbeats to the R1."""
        self._stop_keepalive(device_id)  # cancel any existing task
        self._keepalive_tasks[device_id] = asyncio.ensure_future(
            self._keepalive_loop(device_id, ws)
        )

    def _stop_keepalive(self, device_id: str) -> None:
        """Cancel the keepalive task for a device."""
        task = self._keepalive_tasks.pop(device_id, None)
        if task and not task.done():
            task.cancel()

    async def _keepalive_loop(self, device_id: str, ws: WebSocketServerProtocol) -> None:
        """Send system-presence heartbeats every *_keepalive_interval* seconds."""
        try:
            while True:
                await asyncio.sleep(self._keepalive_interval)
                await self._send(ws, {
                    "type": "event",
                    "event": "system-presence",
                    "payload": {"ts": _now_ms(), "deviceId": device_id},
                })
                logger.debug("Rabbit R1: keepalive sent to %s", _redact(device_id))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug("Rabbit R1: keepalive stopped for %s: %s", _redact(device_id), e)

    # ------------------------------------------------------------------
    # Tunnel helpers
    # ------------------------------------------------------------------

    async def _start_tunnel(self) -> Optional[str]:
        """Start the configured tunnel and return the public wss:// URL."""
        # Allow hardcoding the public URL via env var - useful when running
        # as a systemd service where subprocess tunnel detection may fail.
        explicit_url = os.getenv("RABBIT_R1_PUBLIC_URL")
        if explicit_url:
            logger.info("Rabbit R1: using explicit public URL: %s", explicit_url)
            return explicit_url

        if self._tunnel_mode == "none":
            return None

        loop = asyncio.get_event_loop()

        if self._tunnel_mode == "tailscale":
            url = await loop.run_in_executor(None, _get_tailscale_funnel_url, self._port)
            if url:
                logger.info("Rabbit R1: Tailscale Funnel active at %s", url)
            else:
                logger.warning(
                    "Rabbit R1: Tailscale Funnel unavailable - R1 will only work "
                    "on the local network. Run 'tailscale funnel %s' manually or "
                    "set RABBIT_R1_TUNNEL=none.", self._port,
                )
            return url

        if self._tunnel_mode == "cloudflare":
            url = await loop.run_in_executor(None, _get_cloudflare_tunnel_url, self._port)
            if url:
                logger.info("Rabbit R1: Cloudflare Tunnel active at %s", url)
            else:
                logger.warning("Rabbit R1: Cloudflare Tunnel unavailable")
            return url

        logger.warning(
            "Rabbit R1: Unknown tunnel mode %r, skipping tunnel", self._tunnel_mode
        )
        return None

    # ------------------------------------------------------------------
    # QR code / pairing info
    # ------------------------------------------------------------------

    def _build_qr_payload(self) -> str:
        """Build the clawdbot-gateway QR payload JSON string."""
        if self._public_url:
            # Strip scheme - the payload uses host + port separately.
            host = self._public_url.replace("wss://", "").replace("ws://", "")
            port = 443  # Tailscale/Cloudflare terminate TLS on 443
        else:
            host = _get_lan_ip()
            port = self._port
        return json.dumps({
            "type": "clawdbot-gateway",
            "version": 1,
            "ips": [host],
            "port": port,
            "token": self._token,
            "protocol": "wss" if self._public_url else "ws",
        })

    async def _print_pairing_info(self) -> None:
        """Print pairing instructions and save/print the QR code."""
        qr_data = self._build_qr_payload()

        # Save QR code as PNG for easy access (avoids terminal truncation).
        qr_png_path = None
        if QRCODE_AVAILABLE:
            try:
                qr_png_path = os.path.expanduser("~/.hermes/rabbit_r1_qr.png")
                os.makedirs(os.path.dirname(qr_png_path), exist_ok=True)
                qrcode.make(qr_data).save(qr_png_path)
                logger.info("Rabbit R1: QR code saved to %s", qr_png_path)
            except Exception as e:
                logger.warning("Rabbit R1: failed to save QR PNG: %s", e)
                qr_png_path = None

        print("\n" + "=" * 60)
        print("  Rabbit R1 - Hermes Gateway")
        print("=" * 60)
        if self._public_url:
            print(f"  Public URL : {self._public_url}")
            print("  Works from : anywhere (home, cellular, travelling)")
        else:
            host = _get_lan_ip()
            print(f"  Local URL  : ws://{host}:{self._port}")
            print("  Works from : home network only")
        masked = self._token[:6] + "..." + self._token[-4:]
        print(f"  Token      : {masked}  (full token in RABBIT_R1_TOKEN env var)")
        if qr_png_path:
            print(f"  QR image   : {qr_png_path}")
        print()
        print("  Scan the QR code below with your Rabbit R1:")
        print("  (If the QR code is cut off, open the PNG file above instead)")
        print()

        if QRCODE_AVAILABLE:
            qr = qrcode.QRCode(border=1)
            qr.add_data(qr_data)
            qr.make(fit=True)
            qr.print_ascii(invert=True)
        else:
            print(f"  QR payload : {qr_data}")
            print()
            print("  (Install 'qrcode' for a visual QR code: pip install qrcode)")

        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Low-level send
    # ------------------------------------------------------------------

    @staticmethod
    async def _send(ws: WebSocketServerProtocol, data: dict) -> None:
        """Send a JSON message, ignoring closed connections."""
        try:
            await ws.send(json.dumps(data))
        except websockets.exceptions.ConnectionClosed:
            pass


# ---------------------------------------------------------------------------
# Plugin hooks
# ---------------------------------------------------------------------------

def _env_enablement() -> Optional[Dict[str, Any]]:
    """Auto-seed PlatformConfig.extra from env-only setups.

    Lets ``hermes status`` reflect a Rabbit R1 configuration that lives
    entirely in ``.env`` without a ``platforms.rabbit_r1`` block in
    ``config.yaml``. The R1 needs no external credential, so the presence
    of any RABBIT_R1_* var is treated as an intent to enable.
    """
    keys = (
        "RABBIT_R1_TOKEN", "RABBIT_R1_PORT", "RABBIT_R1_TUNNEL",
        "RABBIT_R1_PUBLIC_URL", "RABBIT_R1_KEEPALIVE_INTERVAL",
        "RABBIT_R1_HOME_CHANNEL",
    )
    if not any(os.getenv(k) for k in keys):
        return None

    seeded: Dict[str, Any] = {}
    if os.getenv("RABBIT_R1_TOKEN"):
        seeded["token"] = os.environ["RABBIT_R1_TOKEN"]
    if os.getenv("RABBIT_R1_PORT"):
        try:
            seeded["port"] = int(os.environ["RABBIT_R1_PORT"])
        except ValueError:
            pass
    if os.getenv("RABBIT_R1_TUNNEL"):
        seeded["tunnel"] = os.environ["RABBIT_R1_TUNNEL"]
    if os.getenv("RABBIT_R1_PUBLIC_URL"):
        seeded["public_url"] = os.environ["RABBIT_R1_PUBLIC_URL"]
    if os.getenv("RABBIT_R1_KEEPALIVE_INTERVAL"):
        try:
            seeded["keepalive_interval"] = int(os.environ["RABBIT_R1_KEEPALIVE_INTERVAL"])
        except ValueError:
            pass
    if os.getenv("RABBIT_R1_HOME_CHANNEL"):
        seeded["home_channel"] = os.environ["RABBIT_R1_HOME_CHANNEL"]
    return seeded or {}


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[List[str]] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    """Out-of-process delivery for cron jobs detached from the gateway.

    The Rabbit R1 adapter is a WebSocket *server*: messages can only be
    pushed to a device over a live connection owned by the running gateway.
    A detached cron process holds no such connection, so there is nothing to
    push to. We return a descriptive error instead of silently failing.

    To deliver to an R1 from cron, run the cron job in the gateway process
    (the scheduler routes through the live adapter's ``send()`` there).
    """
    return {
        "error": (
            "Rabbit R1 delivery requires the gateway process (the device "
            "connects to the gateway's WebSocket server). Run the cron job "
            "with the gateway active, or use deliver='origin'."
        )
    }


def interactive_setup() -> None:
    """Minimal stdin wizard for ``hermes setup rabbit_r1``."""
    print()
    print("Rabbit R1 setup")
    print("---------------")
    print("The Rabbit R1 adapter runs a WebSocket server your R1 connects to.")
    print("A tunnel (Tailscale Funnel or Cloudflare) makes it reachable from")
    print("anywhere. On startup a QR code is printed - scan it with your R1 to")
    print("pair. The auth token is auto-generated if you leave it blank.")
    print()

    try:
        from hermes_cli.config import get_env_var, set_env_var
    except ImportError:
        print("hermes_cli.config not available; set RABBIT_R1_* vars manually "
              "in ~/.hermes/.env")
        return

    def _prompt(var: str, prompt: str, *, secret: bool = False) -> None:
        existing = get_env_var(var) if callable(get_env_var) else None
        suffix = " [keep current]" if existing else ""
        try:
            if secret:
                from hermes_cli.secret_prompt import masked_secret_prompt
                value = masked_secret_prompt(f"{prompt}{suffix}: ")
            else:
                value = input(f"{prompt}{suffix}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if value:
            set_env_var(var, value)

    _prompt("RABBIT_R1_TOKEN", "Auth token (blank = auto-generate)", secret=True)
    _prompt("RABBIT_R1_TUNNEL", "Tunnel mode (tailscale/cloudflare/none)")
    _prompt("RABBIT_R1_PORT", "WebSocket port (blank = 18789)")
    _prompt("RABBIT_R1_ALLOWED_USERS", "Allowed device IDs (comma-separated; blank = any)")
    print("Done. Start the gateway and scan the printed QR code with your R1.")


def register(ctx) -> None:
    """Plugin entry point - called by the Hermes plugin system at startup."""
    ctx.register_platform(
        name="rabbit_r1",
        label="Rabbit R1",
        adapter_factory=lambda cfg: RabbitR1Adapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=[],
        install_hint="pip install websockets qrcode",
        setup_fn=interactive_setup,
        # Env-driven auto-configuration: seeds PlatformConfig.extra + home
        # channel from RABBIT_R1_* vars so env-only setups show in status.
        env_enablement_fn=_env_enablement,
        # Cron home-channel delivery target.
        cron_deliver_env_var="RABBIT_R1_HOME_CHANNEL",
        # Out-of-process cron delivery (returns a descriptive error - the R1
        # can only be reached from the live gateway process).
        standalone_sender_fn=_standalone_send,
        # Auth env vars for _is_user_authorized() integration.
        allowed_users_env="RABBIT_R1_ALLOWED_USERS",
        allow_all_env="RABBIT_R1_ALLOW_ALL_USERS",
        # Keep responses readable on the small 2.88-inch screen.
        max_message_length=R1_MAX_MESSAGE_LENGTH,
        # Device IDs are redacted in this adapter's own logs.
        pii_safe=False,
        allow_update_command=True,
        emoji="🐰",
        platform_hint=(
            "The user is on a Rabbit R1 device with a small 2.88-inch "
            "touchscreen. Keep responses concise and conversational - no "
            "markdown, no long lists. The device has voice output so short "
            "spoken-style answers work best. Aim for 1-3 sentences unless the "
            "user asks for detail. If the user asks for the R1 QR code or needs "
            "to reconnect their R1, the pairing QR code PNG is saved at "
            "~/.hermes/rabbit_r1_qr.png - send or share that file. The R1 "
            "auto-reconnects using the same QR code as long as the token has "
            "not changed."
        ),
    )
