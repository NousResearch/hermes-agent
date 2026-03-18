"""
Lattice platform adapter.

Connects to a Lattice push notification server via SSE.
Inbound notifications are routed through the gateway message handler,
which will call agent.interrupt() if an agent is running or start a new conversation.

Requires:
- LATTICE_URL env var (required to enable)
- LATTICE_PRIVATE_KEY_HEX (optional; auto-generated and persisted on first run)
- LATTICE_TOPICS (optional; comma-separated topics to subscribe to)

Lattice auth: Ed25519 keypair. Sign payload ";{unix_timestamp}" for GET requests.
"""

import asyncio
import json
import logging
import os
import random
import time

import httpx

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.session import SessionSource

logger = logging.getLogger(__name__)

SSE_RETRY_DELAY_INITIAL = 2.0
SSE_RETRY_DELAY_MAX = 60.0


def check_lattice_requirements() -> bool:
    """Check if Lattice is configured (LATTICE_URL is set)."""
    return bool(os.getenv("LATTICE_URL"))


def _ensure_lattice_key() -> str:
    """
    Ensure we have a persistent Ed25519 private key.
    If LATTICE_PRIVATE_KEY_HEX is set, use it. Otherwise generate one,
    save to ~/.hermes/.env, and return it.
    """
    privkey = os.getenv("LATTICE_PRIVATE_KEY_HEX", "").strip()
    if privkey and len(privkey) == 64:
        return privkey

    # Generate new keypair and persist
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives import serialization
    except ImportError:
        raise RuntimeError(
            "Lattice requires the cryptography package. Run: pip install cryptography"
        )

    private_key = Ed25519PrivateKey.generate()
    privkey_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    privkey = privkey_bytes.hex()

    # Persist to ~/.hermes/.env
    try:
        from hermes_cli.config import save_env_value
        save_env_value("LATTICE_PRIVATE_KEY_HEX", privkey)
        os.environ["LATTICE_PRIVATE_KEY_HEX"] = privkey
        logger.info("Lattice: generated and persisted new Ed25519 key to ~/.hermes/.env")
    except Exception as e:
        logger.warning("Lattice: could not persist key to .env: %s", e)
        os.environ["LATTICE_PRIVATE_KEY_HEX"] = privkey

    return privkey


def get_lattice_public_key() -> str | None:
    """
    Ensure Lattice Ed25519 key exists (generate if needed), return public key hex.
    Returns None if cryptography is unavailable or key setup fails.
    """
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

        privkey = _ensure_lattice_key()
        key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(privkey))
        pubkey_hex = key.public_key().public_bytes(
            encoding=Encoding.Raw,
            format=PublicFormat.Raw,
        ).hex()
        return pubkey_hex
    except Exception as e:
        logger.warning("Could not get Lattice public key: %s", e)
        return None


def _get_auth_headers(privkey_hex: str) -> dict:
    """Build Lattice auth headers: X-Agent-Pubkey, X-Timestamp, X-Signature."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    privkey_bytes = bytes.fromhex(privkey_hex)
    if len(privkey_bytes) != 32:
        raise ValueError("LATTICE_PRIVATE_KEY_HEX must be 64 hex chars (32 bytes)")

    private_key = Ed25519PrivateKey.from_private_bytes(privkey_bytes)
    pubkey_bytes = private_key.public_key().public_bytes(
        encoding=Encoding.Raw,
        format=PublicFormat.Raw,
    )
    pubkey_hex = pubkey_bytes.hex()

    timestamp = int(time.time())
    payload = f";{timestamp}".encode("utf-8")
    signature = private_key.sign(payload)
    sig_hex = signature.hex()

    return {
        "X-Agent-Pubkey": pubkey_hex,
        "X-Timestamp": str(timestamp),
        "X-Signature": sig_hex,
    }


class LatticeAdapter(BasePlatformAdapter):
    """
    Lattice push notification adapter.

    Maintains a persistent SSE connection to the Lattice server.
    Incoming notifications are converted to MessageEvent and routed
    through the gateway message handler.
    """

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.LATTICE)

        extra = config.extra or {}
        url = extra.get("url") or os.getenv("LATTICE_URL", "")
        self._lattice_url: str = url.rstrip("/")
        topics = extra.get("topics") or os.getenv("LATTICE_TOPICS", "")
        self._topics: str = topics.strip()

        self._privkey_hex: str = ""
        self.client: httpx.AsyncClient | None = None
        self._sse_task: asyncio.Task | None = None
        self._running = False

        logger.info(
            "Lattice adapter initialized: url=%s topics=%s",
            self._lattice_url,
            self._topics or "(none)",
        )

    async def connect(self) -> bool:
        """Connect to Lattice and start SSE listener."""
        if not self._lattice_url:
            logger.error("Lattice: LATTICE_URL is required")
            return False

        try:
            self._privkey_hex = _ensure_lattice_key()
        except Exception as e:
            logger.error("Lattice: key setup failed: %s", e)
            return False

        self.client = httpx.AsyncClient(timeout=30.0)
        self._running = True
        self._sse_task = asyncio.create_task(self._sse_listener())

        logger.info("Lattice: connected to %s", self._lattice_url)
        return True

    async def disconnect(self) -> None:
        """Stop SSE listener and clean up."""
        self._running = False

        if self._sse_task:
            self._sse_task.cancel()
            try:
                await self._sse_task
            except asyncio.CancelledError:
                pass
            self._sse_task = None

        if self.client:
            await self.client.aclose()
            self.client = None

        logger.info("Lattice: disconnected")

    async def _sse_listener(self) -> None:
        """Listen for SSE events from Lattice server."""
        topics_param = f"topics={self._topics}" if self._topics else ""
        path = "/subscribe"
        if topics_param:
            path = f"{path}?{topics_param}"
        url = f"{self._lattice_url}{path}"
        backoff = SSE_RETRY_DELAY_INITIAL

        while self._running:
            try:
                headers = {
                    "Accept": "text/event-stream",
                    **_get_auth_headers(self._privkey_hex),
                }
                logger.debug("Lattice SSE: connecting to %s", url)
                async with self.client.stream(
                    "GET",
                    url,
                    headers=headers,
                    timeout=None,
                ) as response:
                    if response.status_code != 200:
                        body = await response.aread()
                        logger.warning(
                            "Lattice SSE: HTTP %d %s",
                            response.status_code,
                            body[:200] if body else "",
                        )
                        raise httpx.HTTPStatusError(
                            f"HTTP {response.status_code}",
                            request=response.request,
                            response=response,
                        )
                    backoff = SSE_RETRY_DELAY_INITIAL
                    logger.info("Lattice SSE: connected")

                    buffer = ""
                    event_type = ""
                    event_id = ""
                    event_data = ""

                    async for chunk in response.aiter_text():
                        if not self._running:
                            break
                        buffer += chunk
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            line_stripped = line.strip()

                            if line_stripped.startswith("event:"):
                                event_type = line_stripped[6:].strip()
                            elif line_stripped.startswith("id:"):
                                event_id = line_stripped[3:].strip()
                            elif line_stripped.startswith("data:"):
                                # Per SSE spec: multiple data lines are concatenated with newlines
                                chunk = line_stripped[5:].strip()
                                if event_data:
                                    event_data += "\n" + chunk
                                else:
                                    event_data = chunk

                            if line_stripped == "":
                                if event_type or event_data:
                                    await self._dispatch_sse_event(
                                        event_type, event_id, event_data
                                    )
                                event_type = ""
                                event_id = ""
                                event_data = ""

            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._running:
                    logger.warning(
                        "Lattice SSE: error %s (reconnecting in %.0fs)", e, backoff
                    )

            if self._running:
                jitter = backoff * 0.2 * random.random()
                await asyncio.sleep(backoff + jitter)
                backoff = min(backoff * 2, SSE_RETRY_DELAY_MAX)

    async def _dispatch_sse_event(
        self, event_type: str, event_id: str, data_str: str
    ) -> None:
        """Handle a complete SSE event."""
        if event_type == "connected":
            try:
                data = json.loads(data_str) if data_str else {}
                device_token = data.get("deviceToken", "")
                topics = data.get("topics", [])
                logger.info(
                    "Lattice: connected — device token=%s topics=%s",
                    device_token[:16] + "..." if len(device_token) > 16 else device_token,
                    topics,
                )
            except json.JSONDecodeError:
                logger.debug("Lattice: connected event (raw): %s", data_str[:100])
        elif event_type == "notification":
            await self._process_notification(data_str)
        elif event_type:
            logger.debug("Lattice SSE: event=%s id=%s", event_type, event_id)

    async def _process_notification(self, data_str: str) -> None:
        """Parse notification JSON and route to message handler."""
        try:
            data = json.loads(data_str)
        except json.JSONDecodeError:
            logger.debug("Lattice: invalid notification JSON: %s", data_str[:100])
            return

        title = data.get("title", "")
        body = data.get("body", "")
        if title and body:
            text = f"{title}: {body}"
        else:
            text = body or title or "(empty notification)"

        source = SessionSource(
            platform=Platform.LATTICE,
            chat_id="lattice",
            chat_type="dm",
            user_id="lattice",
        )
        event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=data,
        )

        if self._message_handler:
            await self._message_handler(event)
        else:
            logger.warning("Lattice: no message handler set, dropping notification")

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: str | None = None,
        metadata: dict | None = None,
    ) -> SendResult:
        """Lattice is push-in only; log and return success."""
        logger.debug("Lattice send (no-op): chat_id=%s len=%d", chat_id, len(content))
        return SendResult(success=True)

    async def get_chat_info(self, chat_id: str) -> dict:
        """Return minimal chat info."""
        return {"chat_id": chat_id, "type": "dm"}
