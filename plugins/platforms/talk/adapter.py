"""
Nextcloud Talk platform adapter for Hermes (plugin path).

Speaks the Nextcloud Talk *bot* protocol natively:

  Inbound  (Talk -> bot):  Nextcloud POSTs an ActivityStreams event to this
                           adapter's webhook.  Headers:
                             X-Nextcloud-Talk-Random     (>= 32 chars)
                             X-Nextcloud-Talk-Signature  = hex( HMAC-SHA256(
                                 secret, random + raw_request_body ) )
                           Verified against the same scheme spreed's
                           ChecksumVerificationService::validateRequest uses.

  Outbound (bot -> Talk):  POST {base}/ocs/v2.php/apps/spreed/api/v1/bot/
                             {roomToken}/message
                           Headers:
                             X-Nextcloud-Talk-Bot-Random     (>= 32 chars)
                             X-Nextcloud-Talk-Bot-Signature  = hex( HMAC-SHA256(
                                 secret, random + message_text ) )
                           (spreed BotController::getBotFromHeaders signs the
                           message TEXT, not the JSON body.)

Config (config.yaml):

  platforms:
    talk:
      enabled: true
      extra:
        base_url: "https://drive.aisyncservices.com"
        secret: "<40-128 char shared secret, same as the registered bot>"
        port: 8646
        webhook_path: "/talk"
        allow_all: true            # PoC: accept any Talk user in bound rooms

  plugins:
    enabled: [talk]                # user-installed platform plugins are gated
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import secrets as _secrets
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:  # aiohttp ships with the gateway (it powers api_server)
    import aiohttp
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except Exception:  # pragma: no cover
    AIOHTTP_AVAILABLE = False

from gateway.config import Platform
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

_DEFAULT_PORT = 8646
_DEFAULT_PATH = "/talk"
_ALLOW_ALL_ENV = "TALK_ALLOW_ALL"
_ALLOWED_USERS_ENV = "TALK_ALLOWED_USERS"


# --------------------------------------------------------------------------- #
# Module-level plugin hooks
# --------------------------------------------------------------------------- #
def check_requirements() -> bool:
    """Dependencies available? (aiohttp is bundled with the gateway.)"""
    return AIOHTTP_AVAILABLE


def _read_conf(config, env_name: str, extra_key: str, default: Any = "") -> Any:
    extra = getattr(config, "extra", {}) or {}
    val = os.getenv(env_name)
    if val is not None and val != "":
        return val
    return extra.get(extra_key, default)


def validate_config(config) -> bool:
    """A base_url and a secret are the minimum to run."""
    return bool(_read_conf(config, "TALK_BASE_URL", "base_url")
                and _read_conf(config, "TALK_BOT_SECRET", "secret"))


def is_connected(config) -> bool:
    return validate_config(config)


def _env_enablement() -> Optional[dict]:
    """Surface an env-only setup in gateway status without instantiating."""
    base_url = os.getenv("TALK_BASE_URL")
    secret = os.getenv("TALK_BOT_SECRET")
    if not (base_url and secret):
        return None
    extra: Dict[str, Any] = {"base_url": base_url, "secret": secret}
    if os.getenv("TALK_PORT"):
        extra["port"] = os.getenv("TALK_PORT")
    return {"extra": extra}


# --------------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------------- #
class TalkAdapter(BasePlatformAdapter):
    """Nextcloud Talk bot adapter."""

    def __init__(self, config, **kwargs):
        platform = Platform("talk")
        super().__init__(config=config, platform=platform)

        self.base_url = str(_read_conf(config, "TALK_BASE_URL", "base_url", "")).rstrip("/")
        self.secret = str(_read_conf(config, "TALK_BOT_SECRET", "secret", ""))
        try:
            self.port = int(_read_conf(config, "TALK_PORT", "port", _DEFAULT_PORT))
        except (ValueError, TypeError):
            self.port = _DEFAULT_PORT
        self.webhook_path = str(_read_conf(config, "TALK_WEBHOOK_PATH", "webhook_path", _DEFAULT_PATH))
        if not self.webhook_path.startswith("/"):
            self.webhook_path = "/" + self.webhook_path

        # Authorization: PoC allow-all, satisfied via the gateway's
        # allow_all_env hook (see register()).  Setting the env here means the
        # authz mixin's _is_user_authorized() short-circuits to True.
        allow_all = bool(_read_conf(config, _ALLOW_ALL_ENV, "allow_all", False))
        if isinstance(allow_all, str):
            allow_all = allow_all.lower() in {"1", "true", "yes"}
        self.allow_all = allow_all
        if allow_all:
            os.environ[_ALLOW_ALL_ENV] = "true"

        self._runner: Optional["web.AppRunner"] = None
        self._site: Optional["web.TCPSite"] = None
        self._running = False

    # ---- lifecycle -------------------------------------------------------- #
    async def connect(self, *, is_reconnect: bool = False, **kwargs) -> bool:
        # ``is_reconnect`` is passed by newer Hermes (>=0.18); older Hermes
        # (0.17) calls connect() with no args. Accept both for portability.
        if not AIOHTTP_AVAILABLE:
            self._set_fatal_error("MISSING_SDK", "aiohttp not installed", retryable=False)
            return False
        if not self.base_url or not self.secret:
            self._set_fatal_error(
                "MISSING_CREDENTIALS",
                "platforms.talk.extra needs base_url and secret",
                retryable=False,
            )
            return False
        try:
            app = web.Application()
            app.router.add_get("/health", self._handle_health)
            app.router.add_post(self.webhook_path, self._handle_webhook)
            self._runner = web.AppRunner(app)
            await self._runner.setup()
            self._site = web.TCPSite(self._runner, "0.0.0.0", self.port)
            await self._site.start()
            self._running = True
            self._mark_connected()
            logger.info(
                "[talk] webhook listening on 0.0.0.0:%d%s -> %s",
                self.port, self.webhook_path, self.base_url,
            )
            return True
        except Exception as e:  # pragma: no cover
            self._set_fatal_error("CONNECT_FAILED", f"Talk connect failed: {e}", retryable=True)
            logger.exception("[talk] connect failed")
            return False

    async def disconnect(self) -> None:
        self._running = False
        try:
            if self._site:
                await self._site.stop()
            if self._runner:
                await self._runner.cleanup()
        except Exception:  # pragma: no cover
            logger.exception("[talk] disconnect error")
        finally:
            self._site = None
            self._runner = None

    # ---- inbound ---------------------------------------------------------- #
    async def _handle_health(self, request: "web.Request") -> "web.Response":
        return web.Response(text="ok")

    def _verify_inbound(self, random_hdr: str, signature_hdr: str, raw_body: bytes) -> bool:
        if not random_hdr or not signature_hdr or len(random_hdr) < 32:
            return False
        digest = hmac.new(
            self.secret.encode("utf-8"),
            random_hdr.encode("utf-8") + raw_body,
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(digest, signature_hdr.lower())

    @staticmethod
    def _extract_text(obj: Dict[str, Any]) -> str:
        """Pull the human-readable message text out of object.content."""
        raw = obj.get("content") or "{}"
        try:
            content = json.loads(raw)
        except Exception:
            return str(raw)
        text = content.get("message") or ""
        params = content.get("parameters") or {}
        # Talk uses {placeholder} tokens for mentions/rich params; best-effort
        # substitute the display names so the agent sees readable text.
        if params and "{" in text:
            for key, val in params.items():
                if isinstance(val, dict):
                    name = val.get("name") or val.get("id")
                    if name is not None:
                        text = text.replace("{" + key + "}", str(name))
        return text

    async def _handle_webhook(self, request: "web.Request") -> "web.Response":
        raw = await request.read()
        random_hdr = request.headers.get("X-Nextcloud-Talk-Random", "")
        signature_hdr = request.headers.get("X-Nextcloud-Talk-Signature", "")
        if not self._verify_inbound(random_hdr, signature_hdr, raw):
            logger.warning("[talk] rejected webhook: bad/absent signature")
            return web.json_response({"error": "invalid signature"}, status=401)

        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            return web.json_response({"error": "bad json"}, status=400)

        # Only react to newly-created chat messages.
        if payload.get("type") != "Create":
            return web.Response(status=200)
        obj = payload.get("object") or {}
        if obj.get("name") != "message":
            return web.Response(status=200)

        actor = payload.get("actor") or {}
        actor_type = str(actor.get("type", ""))
        actor_id = str(actor.get("id", ""))
        # Loop guard: never react to bot-authored messages (incl. our own).
        if actor_type.lower() == "bots" or actor_id.startswith("bots/"):
            return web.Response(status=200)

        target = payload.get("target") or {}
        room_token = str(target.get("id", ""))
        room_name = target.get("name") or ""
        text = self._extract_text(obj)

        if not text.strip() or not room_token:
            return web.Response(status=200)

        source = self.build_source(
            chat_id=room_token,
            chat_name=room_name,
            chat_type="group",
            user_id=actor_id or None,
            user_name=actor.get("name") or None,
            message_id=str(obj.get("id") or ""),
            role_authorized=self.allow_all,
        )
        event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            message_id=str(obj.get("id") or ""),
            raw_message=payload,
        )

        # Ack immediately, run the (multi-second) agent turn in the background
        # so Nextcloud does not time out and flip the bot into an error state.
        asyncio.create_task(self._safe_handle(event))
        return web.Response(status=200)

    async def _safe_handle(self, event: MessageEvent) -> None:
        try:
            await self.handle_message(event)
        except Exception:  # pragma: no cover
            logger.exception("[talk] handle_message failed")

    # ---- outbound --------------------------------------------------------- #
    def _bot_headers(self, message: str) -> Dict[str, str]:
        random_val = _secrets.token_hex(32)  # 64 hex chars, >= 32
        signature = hmac.new(
            self.secret.encode("utf-8"),
            random_val.encode("utf-8") + message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return {
            "X-Nextcloud-Talk-Bot-Random": random_val,
            "X-Nextcloud-Talk-Bot-Signature": signature,
            "OCS-APIRequest": "true",
            "Accept": "application/json",
        }

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not content or not content.strip():
            return SendResult(success=True)  # nothing to say

        url = f"{self.base_url}/ocs/v2.php/apps/spreed/api/v1/bot/{chat_id}/message"
        data: Dict[str, Any] = {"message": content}
        if reply_to:
            try:
                data["replyTo"] = int(reply_to)
            except (ValueError, TypeError):
                pass
        if metadata and isinstance(metadata, dict) and metadata.get("reference_id"):
            data["referenceId"] = str(metadata["reference_id"])

        try:
            timeout = aiohttp.ClientTimeout(total=20)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, data=data, headers=self._bot_headers(content)) as resp:
                    body = await resp.text()
                    if resp.status in (200, 201):
                        return SendResult(success=True, raw_response=body)
                    logger.error("[talk] send HTTP %s: %s", resp.status, body[:300])
                    return SendResult(
                        success=False,
                        error=f"HTTP {resp.status}",
                        retryable=resp.status >= 500,
                        raw_response=body,
                    )
        except Exception as e:  # pragma: no cover
            logger.exception("[talk] send exception")
            return SendResult(success=False, error=str(e), retryable=True)

    async def send_typing(self, chat_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        # Nextcloud Talk has no bot typing indicator over the bot API — no-op.
        return None

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        # The Talk bot message API (v1) sends text; post the URL (+caption).
        text = f"{caption}\n{image_url}" if caption else image_url
        return await self.send(chat_id, text, reply_to=reply_to, metadata=metadata)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        # Minimal: the room token is the id; type is treated as a group room.
        return {"name": chat_id, "type": "group", "chat_id": chat_id}


# --------------------------------------------------------------------------- #
# Plugin entry point
# --------------------------------------------------------------------------- #
def register(ctx) -> None:
    ctx.register_platform(
        name="talk",
        label="Nextcloud Talk",
        adapter_factory=lambda cfg: TalkAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=[],
        install_hint="No extra packages needed (uses the gateway's aiohttp)",
        env_enablement_fn=_env_enablement,
        allowed_users_env=_ALLOWED_USERS_ENV,
        allow_all_env=_ALLOW_ALL_ENV,
        emoji="💬",
        pii_safe=True,
        platform_hint=(
            "You are chatting via Nextcloud Talk. Markdown is supported. "
            "Keep replies concise and conversational; you are talking to the "
            "AiSync team in a shared room."
        ),
    )
