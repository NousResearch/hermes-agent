"""
蓝信 (Lanxin) platform adapter for Hermes Agent.

Handles incoming messages via HTTP callback (AES-256-CBC encrypted, HMAC-SHA256
signed) and sends replies through the Lanxin Open Platform REST API.

Requires:
    pip install cryptography aiohttp
    LANXIN_APP_ID, LANXIN_APP_SECRET, LANXIN_API_GW env vars

Configuration in config.yaml:
    platforms:
      lanxin:
        enabled: true
        allowed_users:            # user IDs allowed to talk to the bot; "*" = any
          - "user123"
        require_mention: true     # group chats require @mention
        extra:
          app_id: "your-app-id"        # or LANXIN_APP_ID env var
          app_secret: "your-secret"    # or LANXIN_APP_SECRET env var
          api_gw: "https://api-gw.example.com"  # or LANXIN_API_GW env var
          aes_key: "base64-key"        # or LANXIN_AES_KEY env var
          sign_token: "token"          # or LANXIN_SIGN_TOKEN env var
          bot_port: 8805               # or LANXIN_BOT_PORT env var
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

try:
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from gateway.config import Platform

logger = logging.getLogger(__name__)


# ── Lanxin API Client ────────────────────────────────────────────────────────

class LanxinClient:
    """Lanxin Open Platform API client with token caching."""

    def __init__(self, app_id: str, app_secret: str, api_gw: str):
        self.app_id = app_id
        self.app_secret = app_secret
        self.api_gw = api_gw.rstrip("/")
        self._token: Optional[str] = None
        self._token_expires: float = 0

    async def _request(self, method: str, path: str, data: dict = None) -> dict:
        import aiohttp
        url = f"{self.api_gw}{path}"
        headers = {"Content-Type": "application/json"}
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession() as session:
            if method == "GET":
                async with session.get(url, headers=headers, params=data, timeout=timeout) as resp:
                    return await resp.json()
            else:
                async with session.post(url, headers=headers, json=data, timeout=timeout) as resp:
                    return await resp.json()

    async def get_token(self) -> str:
        if self._token and time.time() < self._token_expires:
            return self._token
        result = await self._request("POST", "/v1/token", {
            "appId": self.app_id,
            "secret": self.app_secret,
        })
        if result.get("errCode") == 0:
            self._token = result["data"]["token"]
            self._token_expires = time.time() + 7000
            return self._token
        raise RuntimeError(f"Failed to get Lanxin token: {result}")

    async def send_message(self, chat_type: str, chat_id: str, content: str) -> dict:
        token = await self.get_token()
        endpoint = "private" if chat_type == "private" else "group"
        return await self._request("POST", f"/v1/message/{endpoint}/send", {
            "receiveId": chat_id,
            "contentType": "text",
            "content": content,
            "type": "text",
        })


# ── AES Decryption ───────────────────────────────────────────────────────────

def decrypt_aes(ciphertext_b64: str, key_b64: str) -> dict:
    """Decrypt Lanxin callback payload (AES-256-CBC, PKCS7)."""
    key_bytes = base64.b64decode(key_b64 + "==")
    iv = key_bytes[:16]
    ciphertext = base64.b64decode(ciphertext_b64)
    cipher = Cipher(algorithms.AES(key_bytes), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    plaintext = decryptor.update(ciphertext) + decryptor.finalize()
    pad_len = plaintext[-1]
    plaintext = plaintext[:-pad_len]
    return json.loads(plaintext.decode("utf-8"))


def verify_signature(token: str, timestamp: str, nonce: str,
                     data_encrypt: str, signature: str) -> bool:
    """Verify callback HMAC-SHA256 signature."""
    sign_str = f"{timestamp}\n{nonce}\n{data_encrypt}\n"
    expected = hmac.new(
        token.encode("utf-8"), sign_str.encode("utf-8"), hashlib.sha256
    ).digest()
    return hmac.compare_digest(base64.b64encode(expected).decode(), signature)


# ── Adapter ──────────────────────────────────────────────────────────────────

class LanxinAdapter(BasePlatformAdapter):
    """Lanxin platform adapter for Hermes."""

    # Lanxin only supports plain text
    supports_code_blocks: bool = False
    supports_async_delivery: bool = True

    def __init__(self, config):
        super().__init__(config=config, platform=Platform("lanxin"))

        extra = getattr(config, "extra", {}) or {}
        self.app_id = extra.get("app_id") or os.getenv("LANXIN_APP_ID", "")
        self.app_secret = extra.get("app_secret") or os.getenv("LANXIN_APP_SECRET", "")
        self.api_gw = extra.get("api_gw") or os.getenv("LANXIN_API_GW", "")
        self.aes_key = extra.get("aes_key") or os.getenv("LANXIN_AES_KEY", "")
        self.sign_token = extra.get("sign_token") or os.getenv("LANXIN_SIGN_TOKEN", "")
        self.bot_port = int(extra.get("bot_port") or os.getenv("LANXIN_BOT_PORT", "8805"))

        # Allowed users
        allowed_raw = extra.get("allowed_users") or os.getenv("LANXIN_ALLOWED_USERS", "")
        if isinstance(allowed_raw, list):
            self._allowed_users: Set[str] = {u.strip() for u in allowed_raw if u.strip()}
        else:
            self._allowed_users = {u.strip() for u in str(allowed_raw).split(",") if u.strip()}
        self._allow_all = (
            os.getenv("LANXIN_ALLOW_ALL_USERS", "").lower() in {"true", "1", "yes"}
            or "*" in self._allowed_users
            or not self._allowed_users
        )

        self.client = LanxinClient(self.app_id, self.app_secret, self.api_gw)
        self._app = None
        self._runner = None
        self._site = None
        # Deduplication
        self._seen_ids: Set[str] = set()

    @property
    def name(self) -> str:
        return "lanxin"

    async def connect(self) -> bool:
        if not AIOHTTP_AVAILABLE:
            logger.warning("[lanxin] aiohttp not installed. Run: pip install aiohttp")
            return False
        if not CRYPTO_AVAILABLE:
            logger.warning("[lanxin] cryptography not installed. Run: pip install cryptography")
            return False
        if not self.app_id or not self.app_secret or not self.api_gw:
            logger.warning("[lanxin] LANXIN_APP_ID, LANXIN_APP_SECRET, LANXIN_API_GW required")
            return False

        try:
            self._app = web.Application()
            self._app.router.add_get("/callback", self._handle_verify)
            self._app.router.add_post("/callback", self._handle_callback)
            self._app.router.add_get("/health", self._handle_health)

            self._runner = web.AppRunner(self._app)
            await self._runner.setup()
            self._site = web.TCPSite(self._runner, "0.0.0.0", self.bot_port)
            await self._site.start()

            self._mark_connected()
            logger.info("[lanxin] Callback server started on port %d", self.bot_port)
            return True
        except Exception as e:
            logger.error("[lanxin] Failed to start: %s", e)
            return False

    async def disconnect(self) -> None:
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()
        logger.info("[lanxin] Disconnected")

    # ── HTTP Handlers ────────────────────────────────────────────────────────

    async def _handle_verify(self, request) -> web.Response:
        """GET challenge verification."""
        challenge = request.query.get("challenge", "")
        return web.Response(text=challenge)

    async def _handle_health(self, request) -> web.Response:
        return web.json_response({"status": "ok", "platform": "lanxin"})

    async def _handle_callback(self, request) -> web.Response:
        """POST callback — decrypt, verify, dispatch."""
        try:
            body = await request.json()
            data_encrypt = body.get("data", "")
            signature = body.get("signature", "")
            timestamp = body.get("timestamp", "")
            nonce = body.get("nonce", "")

            # Signature verification
            if self.sign_token:
                if not verify_signature(self.sign_token, timestamp, nonce, data_encrypt, signature):
                    logger.warning("[lanxin] Signature verification failed")
                    return web.json_response({"errCode": 401, "msg": "Invalid signature"})

            # AES decryption
            if self.aes_key:
                data = decrypt_aes(data_encrypt, self.aes_key)
            else:
                data = json.loads(data_encrypt)

            # Only process message events
            if data.get("type") != "message":
                return web.json_response({"errCode": 0, "msg": "ok"})

            # Dedup
            msg_id = data.get("messageId") or uuid.uuid4().hex
            if msg_id in self._seen_ids:
                return web.json_response({"errCode": 0, "msg": "ok"})
            self._seen_ids.add(msg_id)
            if len(self._seen_ids) > 2000:
                self._seen_ids = set(list(self._seen_ids)[-1000:])

            # Extract fields
            content = data.get("content", "")
            chat_type = data.get("chatType", "private")
            sender_id = data.get("senderId", "")
            chat_id = data.get("chatId", "")
            sender_name = data.get("senderName", sender_id)

            # Parse text
            if isinstance(content, str):
                try:
                    content_obj = json.loads(content)
                    text = content_obj.get("text", content)
                except (json.JSONDecodeError, TypeError):
                    text = content
            else:
                text = str(content) if content else ""

            if not text.strip():
                return web.json_response({"errCode": 0, "msg": "ok"})

            # User allowlist
            if not self._allow_all and sender_id not in self._allowed_users:
                logger.debug("[lanxin] User %s not in allowlist, dropping", sender_id)
                return web.json_response({"errCode": 0, "msg": "ok"})

            # Build source and event
            is_group = chat_type == "group"
            source = self.build_source(
                chat_id=chat_id if is_group else sender_id,
                chat_name=data.get("chatName") if is_group else None,
                chat_type="group" if is_group else "dm",
                user_id=sender_id,
                user_name=sender_name,
            )

            event = MessageEvent(
                text=text,
                message_type=MessageType.TEXT,
                source=source,
                message_id=msg_id,
                raw_message=data,
                timestamp=datetime.now(tz=timezone.utc),
            )

            asyncio.create_task(self._safe_handle(event))
            return web.json_response({"errCode": 0, "msg": "ok"})

        except Exception as e:
            logger.error("[lanxin] Callback error: %s", e, exc_info=True)
            return web.json_response({"errCode": 500, "msg": str(e)})

    async def _safe_handle(self, event: MessageEvent) -> None:
        try:
            await self.handle_message(event)
        except Exception:
            logger.exception("[lanxin] Error processing message")

    # ── Send ─────────────────────────────────────────────────────────────────

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        try:
            meta = metadata or {}
            chat_type = meta.get("chat_type", "")
            # Determine chat_type from chat_id prefix or metadata
            if not chat_type:
                if chat_id.startswith("group:") or ":" not in chat_id:
                    chat_type = "group"
                    actual_id = chat_id.replace("group:", "")
                else:
                    chat_type = "private"
                    actual_id = chat_id.replace("private:", "")
            else:
                actual_id = chat_id

            result = await self.client.send_message(chat_type, actual_id, content)
            if result.get("errCode") == 0:
                return SendResult(success=True)
            return SendResult(success=False, error=result.get("msg", "Unknown error"))
        except Exception as e:
            logger.error("[lanxin] Send failed: %s", e)
            return SendResult(success=False, error=str(e))

    async def send_typing(self, chat_id: str) -> None:
        """Lanxin does not support typing indicators."""
        pass

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        **kwargs,
    ) -> SendResult:
        """Lanxin text-only — send caption as fallback."""
        text = f"[Image] {caption}" if caption else "[Image]"
        return await self.send(chat_id, text, **kwargs)


# ── Plugin Registration ──────────────────────────────────────────────────────

def _check_requirements() -> bool:
    return bool(
        os.getenv("LANXIN_APP_ID")
        and os.getenv("LANXIN_APP_SECRET")
        and os.getenv("LANXIN_API_GW")
    )


def _is_connected(config) -> bool:
    extra = getattr(config, "extra", {}) or {}
    return bool(
        (extra.get("app_id") or os.getenv("LANXIN_APP_ID"))
        and (extra.get("app_secret") or os.getenv("LANXIN_APP_SECRET"))
        and (extra.get("api_gw") or os.getenv("LANXIN_API_GW"))
    )


def _env_enablement() -> Optional[dict]:
    app_id = os.getenv("LANXIN_APP_ID", "").strip()
    app_secret = os.getenv("LANXIN_APP_SECRET", "").strip()
    api_gw = os.getenv("LANXIN_API_GW", "").strip()
    if not app_id or not app_secret or not api_gw:
        return None
    return {
        "app_id": app_id,
        "app_secret": app_secret,
        "api_gw": api_gw,
        "aes_key": os.getenv("LANXIN_AES_KEY", ""),
        "sign_token": os.getenv("LANXIN_SIGN_TOKEN", ""),
        "bot_port": os.getenv("LANXIN_BOT_PORT", "8805"),
    }


def _apply_yaml_config(yaml_cfg: dict, platform_cfg: dict) -> Optional[dict]:
    """Map config.yaml platform.lanxin.* keys to PlatformConfig.extra."""
    extra = {}
    for key in ("app_id", "app_secret", "api_gw", "aes_key", "sign_token", "bot_port"):
        val = platform_cfg.get(key)
        if val:
            env_key = f"LANXIN_{key.upper()}"
            if not os.getenv(env_key):
                os.environ[env_key] = str(val)
            extra[key] = str(val)
    return extra or None


def _build_adapter(config):
    return LanxinAdapter(config)


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="lanxin",
        label="Lanxin (蓝信)",
        adapter_factory=_build_adapter,
        check_fn=_check_requirements,
        is_connected=_is_connected,
        validate_config=_is_connected,
        required_env=["LANXIN_APP_ID", "LANXIN_APP_SECRET", "LANXIN_API_GW"],
        install_hint="pip install cryptography aiohttp",
        env_enablement_fn=_env_enablement,
        apply_yaml_config_fn=_apply_yaml_config,
        allowed_users_env="LANXIN_ALLOWED_USERS",
        allow_all_env="LANXIN_ALLOW_ALL_USERS",
        emoji="🔵",
        allow_update_command=True,
    )
