"""
Mixin Messenger Platform Adapter for Hermes Agent.

A plugin-based gateway adapter that connects to Mixin Network via the Blaze
WebSocket protocol (wss://blaze.mixin.one, subprotocol Mixin-Blaze-1) for
receiving messages, and the Mixin REST API for sending.

Configuration supports THREE methods (in priority order):

Method 1 — Individual config fields (recommended for clarity):
----------------------------------------------------------------
    gateway:
      platforms:
        mixin:
          enabled: true
          extra:
            app_id: "3a7ab4e9-..."
            session_id: "4faf726f-..."
            session_private_key: "47fbbfb4..."
            server_public_key: "c6557764..."   # optional: only needed for PIN/transfers
            spend_private_key: "f3436eba..."   # optional: only needed for Safe transactions

Method 2 — Keystore file path (for users who downloaded keystore JSON):
----------------------------------------------------------------
    gateway:
      platforms:
        mixin:
          enabled: true
          extra:
            keystore_path: ~/.mixin/mixin-donate.keystore.json

Method 3 — Environment variables (overrides all YAML config):
----------------------------------------------------------------
    MIXIN_APP_ID                    (required)
    MIXIN_SESSION_ID                (required)
    MIXIN_SESSION_PRIVATE_KEY       (required)
    MIXIN_SERVER_PUBLIC_KEY         (optional)
    MIXIN_SPEND_PRIVATE_KEY         (optional)
    MIXIN_KEYSTORE_PATH             (alternative to individual vars)
    MIXIN_HOME_CHAT_ID              (optional: default delivery chat)
    MIXIN_ALLOWED_USERS             (optional: comma-separated user IDs)
    MIXIN_ALLOW_ALL_USERS           (optional: "true" to allow all)

Example .env:
    MIXIN_APP_ID=3a7ab4e9-3d82-4b0f-893f-86aec57e8edf
    MIXIN_SESSION_ID=4faf726f-edc0-4120-b829-02087eeced9f
    MIXIN_SESSION_PRIVATE_KEY=47fbbfb4612092c30aa6c522255aa95a6a057e832364b6c94fd474a87ce24af1
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import struct
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports (gateway libs only available when plugin is loaded by Hermes)
# ---------------------------------------------------------------------------

from gateway.platforms.base import (
    BasePlatformAdapter,
    SendResult,
    MessageEvent,
    MessageType,
)
from gateway.session import SessionSource
from gateway.config import PlatformConfig, Platform

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIXIN_API_BASE = "https://api.mixin.one"
MIXIN_BLAZE_URL = "wss://blaze.mixin.one"
BLAZE_SUBPROTOCOL = "Mixin-Blaze-1"
RECONNECT_BASE_DELAY = 1.0
RECONNECT_MAX_DELAY = 60.0

DEFAULT_KEYSTORE = os.path.expanduser("~/.mixin/mixin-donate.keystore.json")


# ---------------------------------------------------------------------------
# Crypto / JWT helpers
# ---------------------------------------------------------------------------

def _build_jwt(ks: dict, method: str, uri: str, body_str: str = "") -> str:
    """Build a Mixin API JWT signed with EdDSA (Ed25519)."""
    from cryptography.hazmat.primitives.asymmetric import ed25519

    now = int(time.time())
    body_hash = hashlib.sha256(body_str.encode()).hexdigest()

    header = {"alg": "EdDSA", "typ": "JWT"}
    payload = {
        "uid": ks["app_id"],
        "sid": ks["session_id"],
        "iat": now,
        "exp": now + 300,
        "jti": str(uuid.uuid4()),
        "sig": f"{method} {uri} {body_hash}",
        "scp": "FULL",
    }

    def _b64url(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    header_b64 = _b64url(json.dumps(header, separators=(",", ":")).encode())
    payload_b64 = _b64url(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{header_b64}.{payload_b64}".encode()

    private_key = ed25519.Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(ks["session_private_key"])
    )
    sig_b64 = _b64url(private_key.sign(signing_input))

    return f"{header_b64}.{payload_b64}.{sig_b64}"


# ---------------------------------------------------------------------------
# REST API client
# ---------------------------------------------------------------------------

async def _rest_request(method: str, path: str, ks: dict, body: dict = None) -> dict:
    """Make a signed request to the Mixin REST API."""
    import aiohttp

    body_str = json.dumps(body, separators=(",", ":")) if body else ""
    jwt = _build_jwt(ks, method.upper(), path, body_str)

    url = f"{MIXIN_API_BASE}{path}"
    headers = {
        "Authorization": f"Bearer {jwt}",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        async with session.request(method, url, headers=headers, json=body) as resp:
            return await resp.json()


# ---------------------------------------------------------------------------
# Helper to send a message via REST (used by send())
# ---------------------------------------------------------------------------

async def _send_mixin_message(
    ks: dict,
    recipient_id: str,
    category: str,
    data: str,
    conversation_id: str = None,
) -> dict:
    """Send a message via Mixin REST API."""
    payload = {
        "category": category,
        "recipient_id": recipient_id,
        "message_id": str(uuid.uuid4()),
        "data": base64.b64encode(data.encode() if isinstance(data, str) else data).decode(),
    }
    if conversation_id:
        payload["conversation_id"] = conversation_id
    return await _rest_request("POST", "/messages", ks, payload)


# ---------------------------------------------------------------------------
# Mixin Adapter
# ---------------------------------------------------------------------------

class MixinAdapter(BasePlatformAdapter):
    """Async Mixin adapter — Blaze WebSocket for inbound, REST for outbound."""

    MAX_MESSAGE_LENGTH = 65536  # Mixin's ~64KB limit for PLAIN_TEXT/PLAIN_POST

    def __init__(self, config, **kwargs):
        platform = Platform("mixin")
        super().__init__(config=config, platform=platform)

        extra = getattr(config, "extra", {}) or {}

        # --- Load keystore (priority: env vars > individual config > file) ---
        # Individual fields from environment
        env_app_id = os.getenv("MIXIN_APP_ID") or ""
        env_session_id = os.getenv("MIXIN_SESSION_ID") or ""
        env_session_pk = os.getenv("MIXIN_SESSION_PRIVATE_KEY") or ""
        env_server_pk = os.getenv("MIXIN_SERVER_PUBLIC_KEY") or ""
        env_spend_pk = os.getenv("MIXIN_SPEND_PRIVATE_KEY") or ""

        # Individual fields from config.yaml extra
        cfg_app_id = str(extra.get("app_id", "") or "")
        cfg_session_id = str(extra.get("session_id", "") or "")
        cfg_session_pk = str(extra.get("session_private_key", "") or "")
        cfg_server_pk = str(extra.get("server_public_key", "") or "")
        cfg_spend_pk = str(extra.get("spend_private_key", "") or "")

        # Keystore file path (from env or config)
        self.keystore_path = (
            os.getenv("MIXIN_KEYSTORE_PATH")
            or extra.get("keystore_path")
            or ""
        )

        # Build keystore dict — env vars override config, config overrides file
        self._ks = None
        ks_from_fields = self._build_keystore_from_fields(
            app_id=env_app_id or cfg_app_id,
            session_id=env_session_id or cfg_session_id,
            session_private_key=env_session_pk or cfg_session_pk,
            server_public_key=env_server_pk or cfg_server_pk,
            spend_private_key=env_spend_pk or cfg_spend_pk,
        )

        if ks_from_fields:
            self._ks = ks_from_fields
            logger.info("Mixin: keystore loaded from config fields")
        elif self.keystore_path:
            path = Path(self.keystore_path).expanduser()
            if path.exists():
                self._ks = self._load_keystore_from_path(str(path))
                logger.info("Mixin: keystore loaded from: %s", path)
            else:
                logger.warning("Mixin: keystore path set but file not found: %s", path)

        self.home_chat_id = (
            os.getenv("MIXIN_HOME_CHAT_ID")
            or extra.get("home_chat_id")
            or ""
        )

        self._bot_name: str = "Mixin Bot"
        self._creator_id: Optional[str] = None

        # Blaze connection
        self._ws: Optional["aiohttp.ClientWebSocketResponse"] = None  # noqa: F821
        self._blaze_task: Optional[asyncio.Task] = None

        # Track processed message IDs for dedup
        self._processed_ids: set = set()

    @property
    def name(self) -> str:
        return "Mixin"

    # ── Connection lifecycle ──────────────────────────────────────────────

    async def connect(self) -> bool:
        """Load keystore, verify identity, start Blaze WebSocket listener."""
        try:
            self._ensure_keystore()
        except (FileNotFoundError, ValueError) as e:
            logger.error("Mixin: %s", e)
            self._set_fatal_error("keystore_error", str(e), retryable=False)
            return False

        # 1. Verify connection by calling /me
        try:
            me = await _rest_request("GET", "/me", self._ks)
            me_data = me.get("data", {})
            self._bot_name = me_data.get("full_name", "Mixin Bot")
            self._creator_id = me_data.get("app", {}).get("creator_id")
            logger.info("Mixin: connected as '%s'", self._bot_name)
        except Exception as e:
            logger.error("Mixin: /me failed: %s", e)
            self._set_fatal_error("auth_failed", str(e), retryable=True)
            return False

        # 2. Start Blaze listener
        self._blaze_task = asyncio.create_task(self._blaze_loop())
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        """Stop Blaze listener and close WebSocket."""
        if self._blaze_task:
            self._blaze_task.cancel()
            try:
                await self._blaze_task
            except asyncio.CancelledError:
                pass
            self._blaze_task = None

        if self._ws:
            await self._ws.close()
            self._ws = None

        self._mark_disconnected()

    # ── Blaze WebSocket loop (inbound messages) ────────────────────────────

    async def _blaze_loop(self) -> None:
        """Main Blaze WebSocket loop with reconnection."""
        import aiohttp

        delay = RECONNECT_BASE_DELAY
        while self._running or not self.has_fatal_error:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(
                        MIXIN_BLAZE_URL,
                        protocols=[BLAZE_SUBPROTOCOL],
                        heartbeat=30.0,
                    ) as ws:
                        self._ws = ws
                        delay = RECONNECT_BASE_DELAY  # reset on success
                        logger.info("Mixin: Blaze connected")

                        # Send authentication/list message first
                        auth_msg = json.dumps({
                            "id": str(uuid.uuid4()),
                            "action": "LIST_PENDING_MESSAGES",
                        }).encode()
                        await ws.send_bytes(auth_msg)

                        # Listen loop
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.BINARY:
                                await self._handle_blaze_frame(msg.data)
                            elif msg.type == aiohttp.WSMsgType.CLOSED:
                                break
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Mixin: Blaze connection lost: %s", e)
                if not self._running:
                    break

            # Reconnect with exponential backoff
            if self._running:
                logger.info("Mixin: reconnecting in %.1fs...", delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, RECONNECT_MAX_DELAY)

    async def _handle_blaze_frame(self, raw: bytes) -> None:
        """Process a single Blaze binary frame (gzip-compressed JSON)."""
        import gzip

        try:
            decompressed = gzip.decompress(raw)
            data = json.loads(decompressed)
        except Exception:
            logger.warning("Mixin: failed to decode Blaze frame")
            return

        # data can be a list (batch) or a dict (single message)
        items = data if isinstance(data, list) else [data]

        for item in items:
            action = item.get("action", "")
            msg_id = item.get("id", "")
            raw_data = item.get("data", "")

            if action == "CREATE_MESSAGE":
                # data is base64-encoded JSON (either single msg or array)
                try:
                    decoded = base64.b64decode(raw_data)
                    payload = json.loads(decoded)
                except Exception:
                    continue

                # payload can be list or single
                messages = payload if isinstance(payload, list) else [payload]
                for msg in messages:
                    await self._handle_incoming_msg(msg)

                # Ack all messages in this batch
                ack = json.dumps({
                    "id": msg_id,
                    "action": "ACKNOWLEDGE_MESSAGE_RECEIPT",
                }).encode()
                if self._ws:
                    await self._ws.send_bytes(ack)

            elif action == "ACKNOWLEDGE_MESSAGE_RECEIPT":
                # Read receipt from other users — typically ignored by bots
                pass

    async def _handle_incoming_msg(self, msg: dict) -> None:
        """Dispatch a single Mixin message to the gateway handler."""
        message_id = msg.get("message_id", "")
        if message_id in self._processed_ids:
            return  # dedup
        self._processed_ids.add(message_id)
        # Keep set bounded
        if len(self._processed_ids) > 10000:
            self._processed_ids = set(list(self._processed_ids)[-5000:])

        category = msg.get("category", "")
        user_id = msg.get("user_id", "")
        conversation_id = msg.get("conversation_id", "")
        data_b64 = msg.get("data", "") or msg.get("data_base64", "")

        # Skip messages from bots (prevent reply loops)
        # Actually we can't easily check if user is a bot here without an API call.
        # Let the gateway handle it.

        # Skip our own messages
        if user_id == self._ks["app_id"]:
            return

        # Decode data
        text = ""
        try:
            text = base64.b64decode(data_b64).decode("utf-8", errors="replace")
        except Exception:
            pass

        # Map Mixin category to MessageType
        msg_type = MessageType.TEXT
        if category == "PLAIN_IMAGE":
            msg_type = MessageType.PHOTO
        elif category == "PLAIN_VIDEO":
            msg_type = MessageType.VIDEO
        elif category == "PLAIN_AUDIO":
            msg_type = MessageType.AUDIO
        elif category == "PLAIN_DATA":
            msg_type = MessageType.DOCUMENT
        elif category == "PLAIN_STICKER":
            msg_type = MessageType.STICKER
        elif category == "PLAIN_LOCATION":
            msg_type = MessageType.LOCATION
        elif category == "PLAIN_TEXT":
            msg_type = MessageType.TEXT
            # Check for commands
            if text.startswith("/"):
                msg_type = MessageType.COMMAND

        # Build source info
        # In Mixin, each 1:1 conversation is unique; group chats have shared
        # conversation_id. We treat conversation_id as chat_id.
        source = self.build_source(
            chat_id=conversation_id,
            chat_name=conversation_id[:8] + "...",  # truncated UUID
            chat_type="dm",  # simplified — could check for group
            user_id=user_id,
            message_id=message_id,
        )

        event = MessageEvent(
            text=text,
            message_type=msg_type,
            source=source,
            message_id=message_id,
            raw_message=msg,
        )

        await self.handle_message(event)

    # ── Sending ────────────────────────────────────────────────────────────

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a message via Mixin REST API.

        chat_id is the Mixin conversation_id.
        For 1:1 DMs, it's a deterministic conversation ID.
        """
        try:
            ks = self._ensure_keystore()
        except RuntimeError as e:
            return SendResult(success=False, error=str(e))

        try:
            result = await _send_mixin_message(
                ks,
                recipient_id="",
                category="PLAIN_POST",
                data=content,
                conversation_id=chat_id,
            )

            if "error" in result and result["error"]:
                err_desc = result["error"].get("description", str(result["error"]))
                return SendResult(success=False, error=err_desc)

            msg_id = result.get("data", {}).get("message_id", str(uuid.uuid4()))
            return SendResult(success=True, message_id=msg_id)

        except Exception as e:
            return SendResult(success=False, error=str(e), retryable=True)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Mixin doesn't have a typing indicator — no-op."""
        pass

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Get info about a conversation (minimal implementation)."""
        return {
            "name": chat_id[:16],
            "type": "dm",
        }

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
    ) -> SendResult:
        """Send an image via Mixin. image_url should be a public URL.

        Mixin's image sending requires uploading to their CDN first, which
        is complex. For now, we send the URL as a PLAIN_POST text message
        with markdown image.
        """
        if caption:
            text = f"{caption}\n\n![]({image_url})"
        else:
            text = f"![]({image_url})"
        return await self.send(chat_id, text)

    # ── Keystore helpers ──────────────────────────────────────────────────

    @staticmethod
    def _build_keystore_from_fields(
        app_id: str = "",
        session_id: str = "",
        session_private_key: str = "",
        server_public_key: str = "",
        spend_private_key: str = "",
    ) -> Optional[dict]:
        """Build a keystore dict from individual fields.

        Returns None if the required fields (app_id, session_id,
        session_private_key) are not all present.
        """
        app_id = app_id.strip()
        session_id = session_id.strip()
        session_private_key = session_private_key.strip()

        if not app_id or not session_id or not session_private_key:
            return None

        ks = {
            "app_id": app_id,
            "session_id": session_id,
            "session_private_key": session_private_key,
        }
        server_pk = server_public_key.strip() if server_public_key else ""
        spend_pk = spend_private_key.strip() if spend_private_key else ""
        if server_pk:
            ks["server_public_key"] = server_pk
        if spend_pk:
            ks["spend_private_key"] = spend_pk
        return ks

    def _load_keystore_from_path(self, path: str) -> dict:
        """Load and validate Mixin bot keystore from a JSON file."""
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(
                f"Mixin keystore not found: {p}\n\n"
                "Download one from https://developers.mixin.one/dashboard\n"
                "or configure MIXIN_APP_ID + MIXIN_SESSION_ID + "
                "MIXIN_SESSION_PRIVATE_KEY env vars."
            )
        with open(p) as f:
            ks = json.load(f)
        required = ["app_id", "session_id", "session_private_key"]
        missing = [k for k in required if k not in ks]
        if missing:
            raise ValueError(f"Keystore missing fields: {missing}")
        return ks

    def _ensure_keystore(self) -> dict:
        """Return the loaded keystore or raise a helpful error."""
        if self._ks:
            return self._ks
        raise RuntimeError(
            "Mixin bot not configured. "
            "Set MIXIN_APP_ID, MIXIN_SESSION_ID, and "
            "MIXIN_SESSION_PRIVATE_KEY in your .env file, "
            "or add app_id/session_id/session_private_key under "
            "gateway.platforms.mixin.extra in config.yaml, "
            "or point keystore_path to a downloaded keystore JSON."
        )

    # ── Formatting ─────────────────────────────────────────────────────────

    def format_message(self, content: str) -> str:
        """Mixin PLAIN_POST supports standard Markdown — pass through."""
        return content


# ---------------------------------------------------------------------------
# Plugin entry points
# ---------------------------------------------------------------------------


def check_requirements() -> bool:
    """Check if crypto and aiohttp are available."""
    try:
        from cryptography.hazmat.primitives.asymmetric import ed25519  # noqa: F401
        import aiohttp  # noqa: F401
        return True
    except ImportError:
        return False


def validate_config(config) -> bool:
    """Validate that keystore credentials are available.

    Accepts either individual fields (app_id + session_id + session_private_key)
    or a keystore file path pointing to an existing file.
    """
    if not config:
        return False
    extra = getattr(config, "extra", {}) or {}

    # Check individual fields from env
    if os.getenv("MIXIN_APP_ID") and os.getenv("MIXIN_SESSION_ID") and os.getenv("MIXIN_SESSION_PRIVATE_KEY"):
        return True

    # Check individual fields from config
    if extra.get("app_id") and extra.get("session_id") and extra.get("session_private_key"):
        return True

    # Check keystore file path
    ks_path = (
        os.getenv("MIXIN_KEYSTORE_PATH")
        or extra.get("keystore_path")
        or ""
    )
    if ks_path and Path(ks_path).expanduser().exists():
        return True

    return False


def is_connected(config) -> bool:
    """Quick check: do we have keystore credentials?"""
    extra = getattr(config, "extra", {}) if config else {}

    if os.getenv("MIXIN_APP_ID") and os.getenv("MIXIN_SESSION_ID") and os.getenv("MIXIN_SESSION_PRIVATE_KEY"):
        return True
    if extra:
        if extra.get("app_id") and extra.get("session_id") and extra.get("session_private_key"):
            return True
    ks_path = (
        os.getenv("MIXIN_KEYSTORE_PATH")
        or (extra and extra.get("keystore_path"))
        or ""
    )
    return bool(ks_path and Path(ks_path).expanduser().exists())


def register(ctx):
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="mixin",
        label="Mixin Messenger",
        adapter_factory=lambda cfg: MixinAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["MIXIN_APP_ID", "MIXIN_SESSION_ID", "MIXIN_SESSION_PRIVATE_KEY"],
        install_hint=(
            "Requires PyJWT[crypto] and aiohttp (both already Hermes deps).\n\n"
            "Configure via one of:\n"
            "  1. Set MIXIN_APP_ID, MIXIN_SESSION_ID, MIXIN_SESSION_PRIVATE_KEY in .env\n"
            "  2. Add app_id/session_id/session_private_key under "
            "gateway.platforms.mixin.extra in config.yaml\n"
            "  3. Point keystore_path at a downloaded keystore JSON from "
            "https://developers.mixin.one/dashboard\n\n"
            "Only app_id, session_id, and session_private_key are required "
            "for messaging."
        ),
        setup_fn=None,  # no interactive setup yet
        # Auth env vars
        allowed_users_env="MIXIN_ALLOWED_USERS",
        allow_all_env="MIXIN_ALLOW_ALL_USERS",
        max_message_length=65536,
        emoji="🔷",
        pii_safe=True,
        allow_update_command=True,
        platform_hint=(
            "You are chatting via Mixin Messenger. "
            "Mixin supports Markdown formatting via PLAIN_POST — "
            "you can use **bold**, *italic*, `code`, [links](url), "
            "and # headers. Send images as markdown ![](url). "
            "Keep messages under 64KB."
        ),
    )
