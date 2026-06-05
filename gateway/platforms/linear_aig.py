"""Linear Agent Interaction webhook adapter.

Receives Linear AgentSessionEvent webhooks, verifies Linear's HMAC
signature, acknowledges new sessions with Agent Activities, and routes the
session prompt into Hermes as a normal gateway task.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
from collections import deque
from typing import Any, Dict, Optional

try:
    import aiohttp
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None  # type: ignore[assignment]
    web = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8647
DEFAULT_WEBHOOK_PATH = "/linear/aig"
DEFAULT_GRAPHQL_URL = "https://api.linear.app/graphql"
DEFAULT_MAX_BODY_BYTES = 1_048_576
DEFAULT_MAX_SEEN_DELIVERIES = 5000
DEFAULT_SIGNATURE_TOLERANCE_SECONDS = 60
DEFAULT_ACCESS_TOKEN_ENV_NAMES = (
    "LINEAR_ACCESS_TOKEN",
    "LINEAR_OAUTH_TOKEN",
    "HERMES_LINEAR_AIG_ACCESS_TOKEN",
    "LINEAR_API_KEY",
)
DEFAULT_WEBHOOK_SECRET_ENV_NAMES = (
    "LINEAR_WEBHOOK_SECRET",
    "LINEAR_AIG_WEBHOOK_SECRET",
    "HERMES_LINEAR_AIG_WEBHOOK_SECRET",
)


def check_linear_aig_requirements() -> bool:
    """Return whether required Linear AIG dependencies are available."""
    return AIOHTTP_AVAILABLE


def _clean_secret_value(value: Any) -> str:
    """Normalize env/config secrets without logging or otherwise exposing them."""
    return str(value or "").strip()


class LinearAIGAdapter(BasePlatformAdapter):
    """Receive Linear AgentSessionEvent webhooks and emit Agent Activities."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.LINEAR_AIG)
        extra = config.extra or {}
        self._host = str(extra.get("host", DEFAULT_HOST))
        self._port = int(extra.get("port", DEFAULT_PORT))
        self._webhook_path = self._normalize_path(
            extra.get("webhook_path", DEFAULT_WEBHOOK_PATH)
        )
        self._health_path = self._normalize_path(extra.get("health_path", "/health"))
        self._graphql_url = str(extra.get("graphql_url", DEFAULT_GRAPHQL_URL))
        self._max_body_bytes = int(extra.get("max_body_bytes", DEFAULT_MAX_BODY_BYTES))
        self._signature_tolerance_seconds = int(
            extra.get(
                "signature_tolerance_seconds",
                DEFAULT_SIGNATURE_TOLERANCE_SECONDS,
            )
        )
        self._webhook_secret_env = str(
            extra.get("webhook_secret_env") or "LINEAR_WEBHOOK_SECRET"
        )
        self._access_token_env = str(
            extra.get("access_token_env") or "LINEAR_ACCESS_TOKEN"
        )
        self._webhook_secret = _clean_secret_value(extra.get("webhook_secret"))
        self._access_token = _clean_secret_value(
            config.token
            or config.api_key
            or extra.get("access_token")
        )
        self._access_token_is_api_key = bool(
            (config.api_key and not config.token)
            or extra.get("auth_scheme") == "api_key"
        )
        self._initial_ack = str(
            extra.get("initial_ack")
            or "Hermes received the task and is starting work."
        )
        self._progress_action = str(extra.get("progress_action") or "Working")
        self._runner = None
        self._session: Optional["aiohttp.ClientSession"] = None
        self._linear_sessions: Dict[str, str] = {}
        self._seen_deliveries: set[str] = set()
        self._seen_delivery_order: deque[str] = deque()

    @staticmethod
    def _normalize_path(value: Any) -> str:
        path = str(value or "").strip() or "/"
        if not path.startswith("/"):
            path = "/" + path
        return path

    def _resolved_webhook_secret(self) -> str:
        if self._webhook_secret:
            return self._webhook_secret
        for env_name in (self._webhook_secret_env, *DEFAULT_WEBHOOK_SECRET_ENV_NAMES):
            value = _clean_secret_value(os.getenv(env_name, ""))
            if value:
                return value
        return ""

    def _resolved_access_token(self) -> str:
        return self._resolved_access_token_with_source()[1]

    def _resolved_access_token_with_source(self) -> tuple[str, str]:
        if self._access_token:
            source = "config.api_key" if self._access_token_is_api_key else "config.token"
            return source, self._access_token
        for env_name in (self._access_token_env, *DEFAULT_ACCESS_TOKEN_ENV_NAMES):
            value = _clean_secret_value(os.getenv(env_name, ""))
            if value:
                return env_name, value
        return "", ""

    def _authorization_header(self) -> str:
        token_source, token = self._resolved_access_token_with_source()
        if not token:
            return ""
        if (
            self._access_token_is_api_key
            or token.startswith("lin_api_")
            or token_source == "LINEAR_API_KEY"
        ):
            return token
        return f"Bearer {token}"

    async def connect(self) -> bool:
        if not AIOHTTP_AVAILABLE:
            logger.warning("[linear_aig] aiohttp is not installed")
            return False
        if not self._resolved_webhook_secret():
            raise ValueError(
                "[linear_aig] Missing Linear webhook signing secret. "
                "Set platforms.linear_aig.extra.webhook_secret or "
                f"{self._webhook_secret_env}/LINEAR_AIG_WEBHOOK_SECRET/"
                "HERMES_LINEAR_AIG_WEBHOOK_SECRET."
            )
        if not self._resolved_access_token():
            raise ValueError(
                "[linear_aig] Missing Linear OAuth access token. "
                "Set platforms.linear_aig.token/api_key, "
                "platforms.linear_aig.extra.access_token, or "
                f"{self._access_token_env}/LINEAR_OAUTH_TOKEN/"
                "HERMES_LINEAR_AIG_ACCESS_TOKEN/LINEAR_API_KEY."
            )

        self._session = aiohttp.ClientSession()
        app = web.Application()
        app.router.add_get(self._health_path, self._handle_health)
        app.router.add_post(self._webhook_path, self._handle_webhook)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()
        self._mark_connected()
        logger.info(
            "[linear_aig] Listening on %s:%d%s",
            self._host,
            self._port,
            self._webhook_path,
        )
        return True

    async def disconnect(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        if self._session:
            await self._session.close()
            self._session = None
        self._mark_disconnected()
        logger.info("[linear_aig] Disconnected")

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": chat_id, "type": "linear_aig"}

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a final Hermes response back as a Linear response activity."""
        agent_session_id = self._linear_sessions.get(chat_id)
        if not agent_session_id:
            return SendResult(
                success=False,
                error=f"Unknown Linear agent session for chat_id: {chat_id}",
            )
        activity_type = (
            "error"
            if (metadata or {}).get("linear_error") or self._looks_like_error(content)
            else "response"
        )
        return await self._create_agent_activity(
            agent_session_id,
            {"type": activity_type, "body": str(content or "")},
        )

    async def send_or_update_status(
        self,
        chat_id: str,
        status_key: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send Hermes progress callbacks as Linear thought/action activities."""
        agent_session_id = self._linear_sessions.get(chat_id)
        if not agent_session_id:
            return SendResult(
                success=False,
                error=f"Unknown Linear agent session for chat_id: {chat_id}",
            )
        text = str(content or "").strip()
        if not text:
            return SendResult(success=True)
        if status_key in {"tool_start", "tool_progress", "tool_result"}:
            payload = {
                "type": "action",
                "action": self._progress_action,
                "parameter": text[:500],
            }
        else:
            payload = {"type": "thought", "body": text}
        return await self._create_agent_activity(agent_session_id, payload)

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        return web.json_response({"status": "ok", "platform": "linear_aig"})

    async def _handle_webhook(self, request: "web.Request") -> "web.Response":
        content_length = request.content_length or 0
        if content_length > self._max_body_bytes:
            return web.json_response({"error": "Payload too large"}, status=413)

        try:
            raw_body = await request.read()
        except Exception:
            logger.exception("[linear_aig] Failed to read webhook body")
            return web.json_response({"error": "Bad request"}, status=400)

        secret = self._resolved_webhook_secret()
        if not self._validate_signature(request, raw_body, secret):
            logger.warning("[linear_aig] Invalid Linear webhook signature")
            return web.json_response({"error": "Invalid signature"}, status=401)

        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            return web.json_response({"error": "Cannot parse body"}, status=400)

        if not self._is_fresh_webhook(payload):
            logger.warning("[linear_aig] Linear webhook timestamp outside tolerance")
            return web.json_response({"error": "Stale webhook timestamp"}, status=401)

        event_type = str(payload.get("type") or "")
        action = str(payload.get("action") or "")
        if event_type and event_type != "AgentSessionEvent":
            return web.json_response({"status": "ignored", "type": event_type})
        if action not in {"created", "prompted"}:
            return web.json_response({"status": "ignored", "action": action})

        agent_session_id = self._extract_agent_session_id(payload)
        if not agent_session_id:
            return web.json_response(
                {"error": "Missing agent session id"},
                status=400,
            )

        delivery_id = self._delivery_id(request, payload, agent_session_id, action)
        if self._is_duplicate(delivery_id):
            return web.json_response(
                {
                    "status": "duplicate",
                    "agent_session_id": agent_session_id,
                    "delivery_id": delivery_id,
                },
                status=200,
            )

        chat_id = f"linear_aig:{agent_session_id}"
        self._linear_sessions[chat_id] = agent_session_id

        if action == "created":
            self._schedule_task(
                self._safe_create_agent_activity(
                    agent_session_id,
                    {"type": "thought", "body": self._initial_ack},
                    label="initial_ack",
                )
            )

        event = MessageEvent(
            text=self._build_prompt(payload, action),
            message_type=MessageType.TEXT,
            source=self.build_source(
                chat_id=chat_id,
                chat_name=f"Linear Agent Session {agent_session_id}",
                chat_type="linear_aig",
                user_id=str(payload.get("actor", {}).get("id") or "linear"),
                user_name=str(payload.get("actor", {}).get("name") or "Linear"),
            ),
            raw_message=payload,
            message_id=delivery_id,
        )

        logger.info(
            "[linear_aig] accepted action=%s session=%s delivery=%s prompt_len=%d",
            action,
            agent_session_id,
            delivery_id,
            len(event.text),
        )

        self._schedule_task(self.handle_message(event))

        return web.json_response(
            {
                "status": "accepted",
                "action": action,
                "agent_session_id": agent_session_id,
                "delivery_id": delivery_id,
            },
            status=202,
        )

    def _schedule_task(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    @staticmethod
    def _looks_like_error(content: str) -> bool:
        text = str(content or "").lstrip().lower()
        return text.startswith("sorry, i encountered an error") or text.startswith(
            "error:"
        )

    async def _safe_create_agent_activity(
        self,
        agent_session_id: str,
        content: dict,
        *,
        label: str,
    ) -> None:
        try:
            await self._create_agent_activity(agent_session_id, content)
        except Exception:
            logger.exception(
                "[linear_aig] Failed to create Linear activity (%s)",
                label,
            )

    def _validate_signature(
        self,
        request: "web.Request",
        body: bytes,
        secret: str,
    ) -> bool:
        if not secret:
            return False
        signature = (
            request.headers.get("Linear-Signature")
            or request.headers.get("linear-signature")
            or ""
        ).strip()
        if not signature:
            return False
        if signature.startswith("sha256="):
            signature = signature.removeprefix("sha256=")
        expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(signature, expected)

    def _is_fresh_webhook(self, payload: dict) -> bool:
        raw_timestamp = payload.get("webhookTimestamp")
        if raw_timestamp is None:
            return True
        try:
            timestamp_ms = int(raw_timestamp)
        except (TypeError, ValueError):
            return False
        observed_ms = int(time.time() * 1000)
        tolerance_ms = max(0, self._signature_tolerance_seconds) * 1000
        return abs(observed_ms - timestamp_ms) <= tolerance_ms

    def _delivery_id(
        self,
        request: "web.Request",
        payload: dict,
        agent_session_id: str,
        action: str,
    ) -> str:
        for header in ("Linear-Delivery", "X-Request-ID", "X-Linear-Delivery"):
            value = request.headers.get(header)
            if value:
                return str(value)
        activity_id = (
            payload.get("agentActivity", {}).get("id")
            if isinstance(payload.get("agentActivity"), dict)
            else None
        )
        created_at = payload.get("createdAt") or int(time.time() * 1000)
        return f"{agent_session_id}:{action}:{activity_id or created_at}"

    def _is_duplicate(self, delivery_id: str) -> bool:
        if delivery_id in self._seen_deliveries:
            return True
        self._seen_deliveries.add(delivery_id)
        self._seen_delivery_order.append(delivery_id)
        while len(self._seen_delivery_order) > DEFAULT_MAX_SEEN_DELIVERIES:
            stale = self._seen_delivery_order.popleft()
            self._seen_deliveries.discard(stale)
        return False

    def _extract_agent_session_id(self, payload: dict) -> str:
        session = payload.get("agentSession")
        if isinstance(session, dict) and session.get("id"):
            return str(session["id"])
        for key in ("agentSessionId", "agent_session_id"):
            if payload.get(key):
                return str(payload[key])
        return ""

    def _build_prompt(self, payload: dict, action: str) -> str:
        if action == "created":
            prompt_context = str(payload.get("promptContext") or "").strip()
            if prompt_context:
                return prompt_context
            session = payload.get("agentSession")
            if isinstance(session, dict):
                nested_context = str(session.get("promptContext") or "").strip()
                if nested_context:
                    return nested_context
            return (
                "Linear created an Agent Session for Hermes. "
                "No promptContext was included in the webhook payload."
            )

        activity = payload.get("agentActivity")
        if isinstance(activity, dict):
            content = activity.get("content")
            if isinstance(content, dict) and content.get("body"):
                return str(content["body"])
            if activity.get("body"):
                return str(activity["body"])
        return "Linear prompted this existing Agent Session."

    async def _create_agent_activity(
        self,
        agent_session_id: str,
        content: dict,
    ) -> SendResult:
        variables = {
            "input": {
                "agentSessionId": agent_session_id,
                "content": content,
            }
        }
        data = await self._graphql(
            """
            mutation AgentActivityCreate($input: AgentActivityCreateInput!) {
              agentActivityCreate(input: $input) {
                success
                agentActivity {
                  id
                }
              }
            }
            """,
            variables,
        )
        result = data.get("agentActivityCreate") or {}
        if result.get("success"):
            activity = result.get("agentActivity") or {}
            return SendResult(success=True, message_id=str(activity.get("id") or ""))
        return SendResult(success=False, error="Linear rejected agent activity")

    async def _graphql(self, query: str, variables: dict) -> dict:
        authorization = self._authorization_header()
        if not authorization:
            raise RuntimeError("Linear access token is not configured")
        body = {"query": query, "variables": variables}
        headers = {
            "Authorization": authorization,
            "Content-Type": "application/json",
        }
        session = self._session
        owns_session = session is None
        if session is None:
            session = aiohttp.ClientSession()
        try:
            async with session.post(
                self._graphql_url,
                json=body,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as response:
                text = await response.text()
                if response.status >= 400:
                    raise RuntimeError(f"Linear GraphQL HTTP {response.status}: {text[:200]}")
                payload = json.loads(text)
        finally:
            if owns_session:
                await session.close()

        if payload.get("errors"):
            raise RuntimeError(f"Linear GraphQL errors: {payload['errors']!r}")
        return payload.get("data") or {}
