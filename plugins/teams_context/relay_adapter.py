"""Power Automate Teams relay platform.

Receives selected Teams messages forwarded by Power Automate and stores them
in the local Teams context database. This path does not use Microsoft Graph
application credentials.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
from types import SimpleNamespace
from typing import Any, Dict, Optional

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from plugins.teams_context.models import TeamsChatMessage
from plugins.teams_context.store import TeamsContextStore

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8647
DEFAULT_PATH = "/teams-relay/messages"
DEFAULT_HEALTH_PATH = "/health"


def check_requirements() -> bool:
    return AIOHTTP_AVAILABLE


def _configured_secret(config: PlatformConfig | None = None) -> str:
    extra = (config.extra if config else {}) or {}
    return str(extra.get("secret") or os.getenv("TEAMS_RELAY_SECRET", "")).strip()


def validate_config(config: PlatformConfig) -> bool:
    return bool(_configured_secret(config))


def is_connected(config: PlatformConfig) -> bool:
    return validate_config(config)


def _env_enablement() -> Optional[dict[str, Any]]:
    if not os.getenv("TEAMS_RELAY_SECRET"):
        return None
    return {
        "host": os.getenv("TEAMS_RELAY_HOST", DEFAULT_HOST),
        "port": int(os.getenv("TEAMS_RELAY_PORT", str(DEFAULT_PORT))),
        "path": os.getenv("TEAMS_RELAY_PATH", DEFAULT_PATH),
        "secret": os.getenv("TEAMS_RELAY_SECRET", ""),
    }


def _normalize_path(value: Any, *, default: str) -> str:
    raw = str(value or default).strip() or default
    return raw if raw.startswith("/") else f"/{raw}"


def _json_response(payload: dict[str, Any], *, status: int):
    if web is not None:
        return web.json_response(payload, status=status)
    return SimpleNamespace(status=status, payload=payload)


class TeamsRelayAdapter(BasePlatformAdapter):
    """HTTP receiver for Power Automate Teams message relays."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("teams_relay"))
        extra = config.extra or {}
        self._host = str(extra.get("host") or os.getenv("TEAMS_RELAY_HOST") or DEFAULT_HOST)
        self._port = int(extra.get("port") or os.getenv("TEAMS_RELAY_PORT") or DEFAULT_PORT)
        self._path = _normalize_path(extra.get("path") or os.getenv("TEAMS_RELAY_PATH"), default=DEFAULT_PATH)
        self._health_path = _normalize_path(extra.get("health_path"), default=DEFAULT_HEALTH_PATH)
        self._secret = _configured_secret(config)
        self._store = TeamsContextStore(extra.get("store_path"))
        self._max_body_bytes = int(extra.get("max_body_bytes") or 262_144)
        self._runner = None

    async def connect(self) -> bool:
        if not self._secret:
            logger.error("[teams_relay] Refusing to start without TEAMS_RELAY_SECRET")
            return False
        if not AIOHTTP_AVAILABLE:
            logger.error("[teams_relay] aiohttp is not installed")
            return False
        app = web.Application()
        app.router.add_get(self._health_path, self._handle_health)
        app.router.add_post(self._path, self._handle_message)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()
        self._mark_connected()
        logger.info("[teams_relay] Listening on %s:%s%s", self._host, self._port, self._path)
        return True

    async def disconnect(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
        self._mark_disconnected()

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        del reply_to, metadata
        logger.info("[teams_relay] Response for %s: %s", chat_id, content[:200])
        return SendResult(success=True)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "teams_relay"}

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        return web.json_response(
            {
                "status": "ok",
                "platform": "teams_relay",
                "path": self._path,
                "store_path": str(self._store.path),
            }
        )

    async def _handle_message(self, request: "web.Request") -> "web.Response":
        if not self._authorized(request):
            return _json_response({"error": "unauthorized"}, status=401)
        content_length = request.content_length or 0
        if content_length > self._max_body_bytes:
            return _json_response({"error": "payload too large"}, status=413)
        try:
            payload = await request.json()
        except Exception:
            return _json_response({"error": "invalid json"}, status=400)
        if not isinstance(payload, dict):
            return _json_response({"error": "payload must be an object"}, status=400)
        try:
            message = TeamsChatMessage.from_relay(payload)
        except ValueError as exc:
            return _json_response({"error": str(exc)}, status=400)
        self._store.upsert_message(message)
        return _json_response(
            {
                "status": "accepted",
                "chat_id": message.chat_id,
                "message_id": message.message_id,
            },
            status=202,
        )

    def _authorized(self, request: "web.Request") -> bool:
        provided = str(
            request.headers.get("X-Hermes-Relay-Secret")
            or request.headers.get("CustomHeader1")
            or ""
        ).strip()
        return bool(provided) and hmac.compare_digest(provided, self._secret)


def register_platform(ctx) -> None:
    ctx.register_platform(
        name="teams_relay",
        label="Teams Relay",
        adapter_factory=lambda cfg: TeamsRelayAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["TEAMS_RELAY_SECRET"],
        install_hint="Set TEAMS_RELAY_SECRET and enable platforms.teams_relay.",
        env_enablement_fn=_env_enablement,
        emoji="🔁",
        allow_update_command=False,
        platform_hint=(
            "Teams Relay is an inbound context-capture endpoint used by "
            "Power Automate. It stores messages locally and does not send "
            "chat replies."
        ),
    )
