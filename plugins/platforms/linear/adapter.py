"""Linear AgentSession platform adapter plugin."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult

from .linear_client import LinearClient

logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8655
DEDUP_TTL_SECONDS = 3600


class LinearAgentSessionAdapter(BasePlatformAdapter):
    """Webhook receiver and responder for Linear AgentSession events."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config=config, platform=Platform("linear"))
        extra = config.extra or {}
        self._host = extra.get("host") or os.getenv("LINEAR_HOST", DEFAULT_HOST)
        self._port = int(extra.get("port") or os.getenv("LINEAR_PORT", DEFAULT_PORT))
        self._webhook_secret = (
            extra.get("webhook_secret")
            or extra.get("secret")
            or os.getenv("LINEAR_WEBHOOK_SECRET", "")
        )
        token = (
            extra.get("token")
            or extra.get("access_token")
            or extra.get("api_key")
            or os.getenv("LINEAR_ACCESS_TOKEN")
            or os.getenv("LINEAR_API_KEY")
        )
        self._token = (token or "").strip()
        self.client = LinearClient(token=token)
        self._runner = None
        self._seen_deliveries: Dict[str, float] = {}

    async def connect(self) -> bool:
        if not AIOHTTP_AVAILABLE:
            logger.warning("[linear] aiohttp not installed")
            return False
        if not self._webhook_secret:
            logger.warning("[linear] LINEAR_WEBHOOK_SECRET not configured")
            return False
        if not self._token:
            logger.warning("[linear] LINEAR_ACCESS_TOKEN or LINEAR_API_KEY not configured")
            return False
        app = web.Application()
        app.router.add_get("/health", self._handle_health)
        app.router.add_post("/linear/agent-sessions", self._handle_agent_session_webhook)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()
        self._mark_connected()
        logger.info("[linear] Listening on %s:%s", self._host, self._port)
        return True

    async def disconnect(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._mark_disconnected()

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        return web.json_response({"status": "ok", "platform": "linear"})

    async def _handle_agent_session_webhook(self, request: "web.Request") -> "web.Response":
        if request.content_length and request.content_length > 1_048_576:
            return web.json_response({"error": "Payload too large"}, status=413)
        body = await request.read()
        if not self._validate_signature(request, body):
            return web.json_response({"error": "Invalid signature"}, status=401)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return web.json_response({"error": "Cannot parse body"}, status=400)

        delivery_id = request.headers.get("Linear-Delivery") or request.headers.get("linear-delivery") or ""
        if not delivery_id:
            delivery_id = payload.get("webhookId") or payload.get("webhook_id") or payload.get("id") or str(int(time.time() * 1000))
        self._prune_seen()
        if delivery_id in self._seen_deliveries:
            return web.json_response({"status": "duplicate", "delivery_id": delivery_id}, status=200)
        self._seen_deliveries[delivery_id] = time.time()

        task = asyncio.create_task(self._process_agent_session_event(payload, delivery_id))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return web.json_response({"status": "accepted", "delivery_id": delivery_id}, status=200)

    def _validate_signature(self, request: "web.Request", body: bytes) -> bool:
        if not self._webhook_secret:
            return False
        signature = request.headers.get("Linear-Signature") or request.headers.get("linear-signature") or ""
        if not signature:
            return False
        expected = hmac.new(self._webhook_secret.encode(), body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(signature, expected)

    def _prune_seen(self) -> None:
        cutoff = time.time() - DEDUP_TTL_SECONDS
        self._seen_deliveries = {k: v for k, v in self._seen_deliveries.items() if v >= cutoff}

    async def _process_agent_session_event(self, payload: Dict[str, Any], delivery_id: str) -> None:
        agent_session = payload.get("agentSession") or payload.get("agent_session") or {}
        agent_session_id = str(agent_session.get("id") or payload.get("agentSessionId") or "")
        if not agent_session_id:
            logger.warning("[linear] AgentSession event missing session id")
            return

        action = payload.get("action") or payload.get("type") or ""
        activity = payload.get("agentActivity") or payload.get("agent_activity") or {}
        signal = (activity.get("signal") or payload.get("signal") or "").lower()

        if signal == "stop":
            text = "/stop"
        elif action == "created":
            text = str(agent_session.get("promptContext") or payload.get("promptContext") or "").strip()
            if not text:
                text = "Linear AgentSession created."
            await self._create_thought(agent_session_id)
        else:
            body = str(activity.get("body") or payload.get("body") or "").strip()
            context = str(agent_session.get("promptContext") or payload.get("promptContext") or "").strip()
            text = body
            if context:
                text = f"{body}\n\nContext:\n{context}" if body else context
            if not text:
                text = "Linear AgentSession prompt."

        message_id = str(activity.get("id") or delivery_id)
        source = self.build_source(
            chat_id=f"agentSession:{agent_session_id}",
            chat_name=f"Linear AgentSession {agent_session_id}",
            chat_type="dm",
            user_id=agent_session_id,
            user_name="Linear",
        )
        event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            message_id=message_id,
            raw_message=payload,
            timestamp=datetime.now(tz=timezone.utc),
        )
        await self.handle_message(event)

    async def _create_thought(self, agent_session_id: str) -> None:
        try:
            await self.client.create_agent_activity(
                agent_session_id=agent_session_id,
                content={"type": "thought", "body": "Hermes received this Linear AgentSession and is thinking…"},
            )
        except Exception as e:
            logger.warning("[linear] Failed to create thought activity: %s", e)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        metadata = metadata or {}
        agent_session_id = chat_id.removeprefix("agentSession:")
        if not agent_session_id:
            return SendResult(success=False, error="Missing Linear AgentSession id")
        content_obj = metadata.get("content")
        if not isinstance(content_obj, dict):
            content_type = metadata.get("content_type") or metadata.get("type") or "response"
            content_obj = {"type": content_type, "body": content}
        try:
            activity = await self.client.create_agent_activity(
                agent_session_id=agent_session_id,
                content=content_obj,
            )
            return SendResult(success=True, message_id=activity.get("id"))
        except Exception as e:
            logger.warning("[linear] Failed to create response activity: %s", e)
            return SendResult(success=False, error="Linear send failed")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "dm"}

    async def wait_background_tasks(self) -> None:
        tasks = [task for task in self._background_tasks if not task.done()]
        if tasks:
            await asyncio.gather(*tasks)


def _env_secret() -> str:
    return os.getenv("LINEAR_WEBHOOK_SECRET", "").strip()


def _env_token() -> str:
    return (os.getenv("LINEAR_ACCESS_TOKEN") or os.getenv("LINEAR_API_KEY") or "").strip()


def _config_secret(extra: dict) -> str:
    return str(extra.get("webhook_secret") or extra.get("secret") or _env_secret()).strip()


def _config_token(extra: dict) -> str:
    return str(
        extra.get("token")
        or extra.get("access_token")
        or extra.get("api_key")
        or _env_token()
    ).strip()


def check_requirements() -> bool:
    # Registry calls check_fn() before validate_config(config).  Keep this
    # dependency-only so config.yaml-provided credentials are not rejected just
    # because the equivalent env vars are absent.
    return AIOHTTP_AVAILABLE


def validate_config(config) -> bool:
    extra = getattr(config, "extra", {}) or {}
    return bool(_config_secret(extra)) and bool(_config_token(extra))


def is_connected(config) -> bool:
    extra = getattr(config, "extra", {}) or {}
    return bool(_config_secret(extra)) and bool(_config_token(extra))


def _env_enablement() -> dict | None:
    secret = _env_secret()
    token = _env_token()
    if not secret or not token:
        return None
    seed = {
        "webhook_secret": secret,
        "host": os.getenv("LINEAR_HOST", DEFAULT_HOST),
        "port": int(os.getenv("LINEAR_PORT", DEFAULT_PORT)),
    }
    seed["token"] = token
    return seed


def register(ctx) -> None:
    ctx.register_platform(
        name="linear",
        label="Linear Agent Sessions",
        adapter_factory=lambda cfg: LinearAgentSessionAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["LINEAR_WEBHOOK_SECRET"],
        env_enablement_fn=_env_enablement,
        install_hint="Set LINEAR_WEBHOOK_SECRET and LINEAR_ACCESS_TOKEN (or LINEAR_API_KEY).",
        emoji="📐",
        pii_safe=True,
        allow_update_command=True,
        platform_hint="You are communicating through Linear Agent Sessions. Keep responses actionable and issue-focused.",
    )
