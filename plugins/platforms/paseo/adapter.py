"""Paseo platform adapter (Hermes plugin).

This adapter is intentionally an additive bridge, not a replacement for the
existing Paseo ACP connection. When enabled, Hermes starts a small,
token-protected HTTP service on a separate port (default ``127.0.0.1:8767``;
Paseo's own daemon currently uses ``127.0.0.1:6767``). A Paseo-side client or
bridge can:

* POST inbound user messages to ``/v1/messages``.
* POST interrupts to ``/v1/interrupt`` (or send ``/stop`` as a normal message).
* GET outbound Hermes messages from ``/v1/messages/{chat_id}``.

The plugin is disabled by default. Environment auto-enablement only happens
when ``PASEO_GATEWAY_ENABLED`` is truthy, so installing this plugin cannot
silently perturb an existing Paseo/ACP setup.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import time
import uuid
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional
from urllib.parse import unquote

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover - only exercised in minimal installs
    web = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    is_network_accessible,
)
from gateway.session import SessionSource
from utils import is_truthy_value

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8767
MAX_MESSAGE_LENGTH = 32768
_DEFAULT_BUFFER_SIZE = 200


def _env_truthy(name: str, default: bool = False) -> bool:
    return is_truthy_value(os.getenv(name), default=default)


def _coerce_port(value: Any, default: int = DEFAULT_PORT) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError):
        return default
    if not (0 <= port <= 65535):
        return default
    return port


def check_requirements() -> bool:
    """Return True when aiohttp is importable."""
    return AIOHTTP_AVAILABLE


def _configured_host(config: Any) -> str:
    extra = getattr(config, "extra", {}) or {}
    return str(extra.get("host") or os.getenv("PASEO_GATEWAY_HOST") or DEFAULT_HOST).strip() or DEFAULT_HOST


def _configured_token(config: Any) -> str:
    extra = getattr(config, "extra", {}) or {}
    return str(extra.get("token") or os.getenv("PASEO_GATEWAY_TOKEN") or "").strip()


def _allow_unauthenticated_local(config: Any) -> bool:
    extra = getattr(config, "extra", {}) or {}
    if "allow_unauthenticated_local" in extra:
        return bool(extra.get("allow_unauthenticated_local"))
    return _env_truthy("PASEO_GATEWAY_ALLOW_UNAUTHENTICATED_LOCAL", False)


def validate_config(config: Any) -> bool:
    """Validate the bridge is safe to start.

    The bridge requires a bearer token by default. A tokenless bridge is only
    accepted when explicitly marked ``allow_unauthenticated_local`` AND bound to
    a loopback host. Network-accessible tokenless binds fail closed.
    """
    if not AIOHTTP_AVAILABLE:
        return False
    host = _configured_host(config)
    token = _configured_token(config)
    if token:
        return True
    if _allow_unauthenticated_local(config) and not is_network_accessible(host):
        return True
    return False


def is_connected(config: Any) -> bool:
    """Return whether the Paseo bridge is configured well enough to start."""
    return validate_config(config)


def _env_enablement() -> dict | None:
    """Seed PlatformConfig.extra from env vars only when explicitly enabled."""
    if not _env_truthy("PASEO_GATEWAY_ENABLED", False):
        return None

    host = os.getenv("PASEO_GATEWAY_HOST", DEFAULT_HOST).strip() or DEFAULT_HOST
    seed: dict[str, Any] = {
        "host": host,
        "port": _coerce_port(os.getenv("PASEO_GATEWAY_PORT"), DEFAULT_PORT),
    }
    token = os.getenv("PASEO_GATEWAY_TOKEN", "").strip()
    if token:
        seed["token"] = token
    if _env_truthy("PASEO_GATEWAY_ALLOW_UNAUTHENTICATED_LOCAL", False):
        seed["allow_unauthenticated_local"] = True
    home = os.getenv("PASEO_HOME_CHANNEL", "").strip()
    if home:
        seed["home_channel"] = {"chat_id": home, "name": os.getenv("PASEO_HOME_CHANNEL_NAME", "Paseo")}
    return seed


class PaseoAdapter(BasePlatformAdapter):
    """HTTP bridge adapter for Paseo.

    The adapter keeps outbound responses in bounded per-chat buffers. A Paseo
    client can poll ``GET /v1/messages/{chat_id}`` until a native push channel is
    wired on the Paseo side. This preserves Hermes gateway semantics while
    avoiding any change to the currently running ACP connection.
    """

    supports_code_blocks = True
    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH

    def __init__(self, config: PlatformConfig):
        super().__init__(config=config, platform=Platform("paseo"))
        extra = config.extra or {}
        self.host = _configured_host(config)
        self.port = _coerce_port(extra.get("port") or os.getenv("PASEO_GATEWAY_PORT"), DEFAULT_PORT)
        self.token = _configured_token(config)
        self.allow_unauthenticated_local = _allow_unauthenticated_local(config)
        self.buffer_size = int(extra.get("buffer_size") or _DEFAULT_BUFFER_SIZE)
        self._outbound: Dict[str, Deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=self.buffer_size))
        self._app_runner: Any = None
        self._site: Any = None
        self._message_sequence = 0

    @property
    def name(self) -> str:
        return "Paseo"

    def build_app(self) -> Any:
        if not AIOHTTP_AVAILABLE:
            raise RuntimeError("PaseoAdapter requires aiohttp")

        app = web.Application()
        app.router.add_get("/healthz", self._health)
        app.router.add_post("/v1/messages", self._post_message)
        app.router.add_get("/v1/messages/{chat_id}", self._get_messages)
        app.router.add_post("/v1/interrupt", self._post_interrupt)
        return app

    async def connect(self) -> bool:
        if not validate_config(self.config):
            self._set_fatal_error(
                "config_invalid",
                "Paseo bridge requires aiohttp and a bearer token unless explicitly loopback-only unauthenticated",
                retryable=False,
            )
            return False
        try:
            self._app_runner = web.AppRunner(self.build_app())
            await self._app_runner.setup()
            self._site = web.TCPSite(self._app_runner, self.host, self.port)
            await self._site.start()
            self._mark_connected()
            logger.info("Paseo bridge listening on %s:%s", self.host, self.port)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Paseo bridge failed to start on %s:%s: %s", self.host, self.port, exc)
            self._set_fatal_error("connect_failed", str(exc), retryable=True)
            return False

    async def disconnect(self) -> None:
        self._running = False
        if self._site is not None:
            try:
                await self._site.stop()
            except Exception:  # noqa: BLE001
                logger.debug("Paseo bridge site stop failed", exc_info=True)
            self._site = None
        if self._app_runner is not None:
            try:
                await self._app_runner.cleanup()
            except Exception:  # noqa: BLE001
                logger.debug("Paseo bridge app cleanup failed", exc_info=True)
            self._app_runner = None
        await self.cancel_background_tasks()

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        message_id = self._next_message_id()
        chunks = self.truncate_message(content, self.MAX_MESSAGE_LENGTH)
        payload = {
            "id": message_id,
            "chat_id": str(chat_id),
            "content": "\n".join(chunks),
            "reply_to": reply_to,
            "metadata": metadata or {},
            "created_at": time.time(),
            "role": "assistant",
        }
        self._outbound[str(chat_id)].append(payload)
        return SendResult(success=True, message_id=message_id, raw_response=payload)

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
    ) -> SendResult:
        chat_key = str(chat_id)
        for payload in reversed(self._outbound.get(chat_key, ())):
            if payload.get("id") == message_id:
                payload["content"] = "\n".join(self.truncate_message(content, self.MAX_MESSAGE_LENGTH))
                payload["finalize"] = finalize
                payload["edited_at"] = time.time()
                return SendResult(success=True, message_id=message_id, raw_response=payload)
        return await self.send(chat_id, content, metadata={"edited_from": message_id, "finalize": finalize})

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": str(chat_id), "type": "dm"}

    def _next_message_id(self) -> str:
        self._message_sequence += 1
        return f"paseo-{self._message_sequence}-{uuid.uuid4().hex[:8]}"

    def _authorized(self, request: Any) -> bool:
        if not self.token:
            return self.allow_unauthenticated_local and not is_network_accessible(self.host)
        auth = request.headers.get("Authorization", "")
        if auth.lower().startswith("bearer "):
            supplied = auth.split(" ", 1)[1].strip()
            if secrets.compare_digest(supplied, self.token):
                return True
        supplied = request.headers.get("X-Hermes-Paseo-Token", "").strip()
        return bool(supplied and secrets.compare_digest(supplied, self.token))

    def _unauthorized(self) -> Any:
        return web.json_response({"error": "unauthorized"}, status=401)

    async def _health(self, request: Any) -> Any:
        return web.json_response({"ok": True, "platform": "paseo", "connected": self.is_connected})

    async def _post_message(self, request: Any) -> Any:
        if not self._authorized(request):
            return self._unauthorized()
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON body"}, status=400)
        event = self._event_from_payload(payload)
        if event is None:
            return web.json_response({"error": "text, chat_id, and user_id are required"}, status=400)
        await self.handle_message(event)
        return web.json_response(
            {"ok": True, "chat_id": event.source.chat_id, "message_id": event.message_id},
            status=202,
        )

    async def _post_interrupt(self, request: Any) -> Any:
        if not self._authorized(request):
            return self._unauthorized()
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        chat_id = str(payload.get("chat_id") or "paseo-default")
        user_id = str(payload.get("user_id") or "paseo")
        event = self._event_from_payload({**payload, "text": "/stop", "chat_id": chat_id, "user_id": user_id})
        if event is not None:
            await self.handle_message(event)
        return web.json_response({"ok": True, "chat_id": chat_id}, status=202)

    async def _get_messages(self, request: Any) -> Any:
        if not self._authorized(request):
            return self._unauthorized()
        chat_id = unquote(str(request.match_info.get("chat_id", "")))
        messages = list(self._outbound.get(chat_id, ()))
        after = request.query.get("after")
        if after:
            seen = False
            filtered = []
            for message in messages:
                if seen:
                    filtered.append(message)
                elif message.get("id") == after:
                    seen = True
            messages = filtered if seen else messages
        if request.query.get("clear", "").lower() in {"1", "true", "yes"}:
            self._outbound.pop(chat_id, None)
        return web.json_response({"ok": True, "chat_id": chat_id, "messages": messages})

    def _event_from_payload(self, payload: dict[str, Any]) -> Optional[MessageEvent]:
        text = str(payload.get("text") or "")
        chat_id = str(payload.get("chat_id") or "")
        user_id = str(payload.get("user_id") or "")
        if not (text and chat_id and user_id):
            return None
        source = SessionSource(
            platform=Platform("paseo"),
            chat_id=chat_id,
            chat_name=payload.get("chat_name"),
            chat_type=str(payload.get("chat_type") or "dm"),
            user_id=user_id,
            user_name=payload.get("user_name"),
            thread_id=payload.get("thread_id"),
            message_id=payload.get("message_id"),
        )
        msg_type = MessageType.COMMAND if text.startswith("/") else MessageType.TEXT
        return MessageEvent(
            text=text,
            message_type=msg_type,
            source=source,
            raw_message=payload,
            message_id=str(payload.get("message_id") or uuid.uuid4().hex),
            reply_to_message_id=payload.get("reply_to_message_id"),
        )


async def _standalone_send(
    pconfig: Any,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[List[str]] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    # Cron/send_message can only use this when the live adapter is present; the
    # bridge is intentionally process-local and does not persist outbound queues
    # across a separate cron process.
    return {"error": "paseo standalone send requires the live gateway adapter"}


def register(ctx: Any) -> None:
    """Plugin entry point — called by Hermes plugin discovery."""
    ctx.register_platform(
        name="paseo",
        label="Paseo",
        adapter_factory=lambda cfg: PaseoAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=[],
        install_hint="aiohttp is included with Hermes messaging gateway installs",
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="PASEO_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        allowed_users_env="PASEO_ALLOWED_USERS",
        allow_all_env="PASEO_ALLOW_ALL_USERS",
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji="🛤️",
        pii_safe=True,
        allow_update_command=True,
        platform_hint=(
            "You are communicating through Paseo via the Hermes Paseo gateway bridge. "
            "Slash commands such as /stop, /status, /commands, and /agents are available. "
            "Prototype/local HTML links should use URLs reachable by the Paseo client."
        ),
    )
