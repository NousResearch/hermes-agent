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
from typing import Any, Dict, Mapping, Optional

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    ProcessingOutcome,
    SendResult,
)

from .linear_client import LinearClient

logger = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8655
DEDUP_TTL_SECONDS = 3600
MAX_WEBHOOK_BODY_BYTES = 1_048_576
DEFAULT_AUTH_USER_ID = "linear-agent-session"
DEFAULT_AUTH_USER_NAME = "Linear Agent Session"


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
        self._awaiting_clarify_sessions: Dict[str, float] = {}

    @staticmethod
    def _agent_session_id_from_chat_id(chat_id: str) -> str:
        return str(chat_id or "").removeprefix("agentSession:")

    @staticmethod
    def _prompt_summary(text: str, *, fallback: str = "Linear request") -> str:
        text = " ".join(str(text or "").split())
        if not text:
            return fallback
        return text[:117] + "..." if len(text) > 120 else text

    @staticmethod
    def _select_signal_metadata(choices: list) -> Dict[str, Any]:
        return {
            "options": [
                {"label": str(choice), "value": str(choice)}
                for choice in choices
            ]
        }

    def _mark_awaiting_clarify_reply(self, agent_session_id: str) -> None:
        try:
            from tools.clarify_gateway import get_clarify_timeout
            timeout_seconds = float(get_clarify_timeout())
        except Exception:
            timeout_seconds = 600.0
        self._awaiting_clarify_sessions[agent_session_id] = time.time() + max(timeout_seconds, 0.0) + 60.0

    def _consume_awaiting_clarify_reply(self, agent_session_id: str) -> bool:
        deadline = self._awaiting_clarify_sessions.get(agent_session_id)
        if deadline is None:
            return False
        if deadline < time.time():
            self._awaiting_clarify_sessions.pop(agent_session_id, None)
            return False
        self._awaiting_clarify_sessions.pop(agent_session_id, None)
        return True

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
        app = web.Application(client_max_size=MAX_WEBHOOK_BODY_BYTES)
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
        if request.content_length is not None and request.content_length > MAX_WEBHOOK_BODY_BYTES:
            return web.json_response({"error": "Payload too large"}, status=413)
        body = await request.read()
        status, payload = self._handle_agent_session_body(
            body=body,
            headers=request.headers,
            content_length=request.content_length,
        )
        return web.json_response(payload, status=status)

    def _handle_agent_session_body(
        self,
        *,
        body: bytes,
        headers: Mapping[str, str],
        content_length: Optional[int] = None,
    ) -> tuple[int, Dict[str, Any]]:
        if (
            content_length is not None
            and content_length > MAX_WEBHOOK_BODY_BYTES
        ) or len(body) > MAX_WEBHOOK_BODY_BYTES:
            return 413, {"error": "Payload too large"}
        if not self._validate_signature_headers(headers, body):
            return 401, {"error": "Invalid signature"}
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return 400, {"error": "Cannot parse body"}
        if not isinstance(payload, dict):
            return 400, {"error": "Expected JSON object"}

        delivery_id = headers.get("Linear-Delivery") or headers.get("linear-delivery") or ""
        if not delivery_id:
            delivery_id = (
                payload.get("webhookId")
                or payload.get("webhook_id")
                or payload.get("id")
                or f"body-sha256:{hashlib.sha256(body).hexdigest()}"
            )
        self._prune_seen()
        if delivery_id in self._seen_deliveries:
            return 200, {"status": "duplicate", "delivery_id": delivery_id}
        self._seen_deliveries[delivery_id] = time.time()

        task = asyncio.create_task(self._process_agent_session_event(payload, delivery_id))
        self._track_background_task(task, delivery_id)
        return 200, {"status": "accepted", "delivery_id": delivery_id}

    def _track_background_task(self, task: "asyncio.Task", delivery_id: str) -> None:
        self._background_tasks.add(task)

        def _done(done_task: "asyncio.Task") -> None:
            self._background_tasks.discard(done_task)
            try:
                exc = done_task.exception()
            except asyncio.CancelledError:
                return
            if exc is not None:
                logger.warning(
                    "[linear] AgentSession background task failed delivery=%s: %s",
                    delivery_id,
                    exc,
                    exc_info=exc,
                )

        task.add_done_callback(_done)

    def _validate_signature(self, request: "web.Request", body: bytes) -> bool:
        return self._validate_signature_headers(request.headers, body)

    def _validate_signature_headers(self, headers: Mapping[str, str], body: bytes) -> bool:
        if not self._webhook_secret:
            return False
        signature = headers.get("Linear-Signature") or headers.get("linear-signature") or ""
        if not signature:
            return False
        expected = hmac.new(self._webhook_secret.encode(), body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(signature, expected)

    def _prune_seen(self) -> None:
        cutoff = time.time() - DEDUP_TTL_SECONDS
        self._seen_deliveries = {k: v for k, v in self._seen_deliveries.items() if v >= cutoff}

    def _first_scalar(self, *values: Any) -> str:
        for value in values:
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _first_nested_scalar(self, data: Dict[str, Any], *paths: tuple[str, ...]) -> str:
        for path in paths:
            value: Any = data
            for key in path:
                if not isinstance(value, dict):
                    value = None
                    break
                value = value.get(key)
            text = self._first_scalar(value)
            if text:
                return text
        return ""

    def _agent_activity_text(self, activity: Dict[str, Any], payload: Dict[str, Any]) -> str:
        content = activity.get("content") if isinstance(activity.get("content"), dict) else {}
        return self._first_scalar(
            activity.get("body"),
            content.get("body"),
            content.get("bodyData"),
            payload.get("body"),
        )

    def _linear_actor_identity(
        self,
        payload: Dict[str, Any],
        agent_session: Dict[str, Any],
        activity: Dict[str, Any],
    ) -> tuple[str, str]:
        """Return the stable Linear user identity used for gateway auth.

        AgentSession ids are per-task conversation ids, so using them as
        ``user_id`` makes pairing expire on every new Linear session. Prefer
        actor/user fields from the webhook payload when Linear includes them;
        otherwise use a stable signed-webhook principal.
        """
        user_id = self._first_nested_scalar(
            payload,
            ("actor", "id"),
            ("actor", "email"),
            ("user", "id"),
            ("user", "email"),
            ("creator", "id"),
            ("creator", "email"),
            ("createdBy", "id"),
            ("createdBy", "email"),
            ("organization", "id"),
            ("workspace", "id"),
        ) or self._first_nested_scalar(
            agent_session,
            ("actor", "id"),
            ("actor", "email"),
            ("user", "id"),
            ("user", "email"),
            ("creator", "id"),
            ("creator", "email"),
            ("createdBy", "id"),
            ("createdBy", "email"),
        ) or self._first_nested_scalar(
            activity,
            ("actor", "id"),
            ("actor", "email"),
            ("user", "id"),
            ("user", "email"),
            ("creator", "id"),
            ("creator", "email"),
            ("createdBy", "id"),
            ("createdBy", "email"),
        )

        user_name = self._first_nested_scalar(
            payload,
            ("actor", "name"),
            ("actor", "displayName"),
            ("user", "name"),
            ("user", "displayName"),
            ("creator", "name"),
            ("creator", "displayName"),
            ("createdBy", "name"),
            ("createdBy", "displayName"),
        ) or self._first_nested_scalar(
            agent_session,
            ("actor", "name"),
            ("actor", "displayName"),
            ("user", "name"),
            ("user", "displayName"),
            ("creator", "name"),
            ("creator", "displayName"),
            ("createdBy", "name"),
            ("createdBy", "displayName"),
        ) or self._first_nested_scalar(
            activity,
            ("actor", "name"),
            ("actor", "displayName"),
            ("user", "name"),
            ("user", "displayName"),
            ("creator", "name"),
            ("creator", "displayName"),
            ("createdBy", "name"),
            ("createdBy", "displayName"),
        )

        return (
            user_id or os.getenv("LINEAR_AUTH_USER_ID", DEFAULT_AUTH_USER_ID),
            user_name or os.getenv("LINEAR_AUTH_USER_NAME", DEFAULT_AUTH_USER_NAME),
        )

    async def _process_agent_session_event(self, payload: Dict[str, Any], delivery_id: str) -> None:
        agent_session = payload.get("agentSession") or payload.get("agent_session") or {}
        if not isinstance(agent_session, dict):
            agent_session = {}
        agent_session_id = str(agent_session.get("id") or payload.get("agentSessionId") or "")
        if not agent_session_id:
            logger.warning("[linear] AgentSession event missing session id")
            return

        action = payload.get("action") or payload.get("type") or ""
        activity = payload.get("agentActivity") or payload.get("agent_activity") or {}
        if not isinstance(activity, dict):
            activity = {}
        signal = (activity.get("signal") or payload.get("signal") or "").lower()

        if signal == "stop":
            self._awaiting_clarify_sessions.pop(agent_session_id, None)
            text = "/stop"
        elif action == "created":
            text = str(agent_session.get("promptContext") or payload.get("promptContext") or "").strip()
            if not text:
                text = "Linear AgentSession created."
            await self._create_thought(agent_session_id)
        else:
            body = self._agent_activity_text(activity, payload)
            context = str(agent_session.get("promptContext") or payload.get("promptContext") or "").strip()
            text = body
            awaiting_clarify_reply = bool(body) and self._consume_awaiting_clarify_reply(agent_session_id)
            if awaiting_clarify_reply:
                pass
            elif context and not body.lstrip().startswith("/"):
                text = f"{body}\n\nContext:\n{context}" if body else context
            if not text:
                text = "Linear AgentSession prompt."

        message_id = str(activity.get("id") or delivery_id)
        user_id, user_name = self._linear_actor_identity(payload, agent_session, activity)
        source = self.build_source(
            chat_id=f"agentSession:{agent_session_id}",
            chat_name=f"Linear AgentSession {agent_session_id}",
            chat_type="dm",
            user_id=user_id,
            user_name=user_name,
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
            activity = await self.client.create_agent_activity(
                agent_session_id=agent_session_id,
                content={"type": "thought", "body": "Hermes received this Linear AgentSession and is thinking…"},
            )
            logger.info(
                "[linear] Created initial thought activity for AgentSession %s activity=%s",
                agent_session_id,
                activity.get("id") or "?",
            )
        except Exception as e:
            logger.warning("[linear] Failed to create thought activity: %s", e)

    async def _update_plan(
        self,
        agent_session_id: str,
        plan: list[Dict[str, Any]],
    ) -> None:
        try:
            session = await self.client.update_agent_session(
                agent_session_id=agent_session_id,
                plan=plan,
            )
            logger.info(
                "[linear] Updated AgentSession %s plan steps=%d session_status=%s",
                agent_session_id,
                len(plan),
                session.get("status") or "?",
            )
        except Exception as e:
            logger.warning("[linear] Failed to update AgentSession plan: %s", e)

    async def _prepare_issue_for_agent_work(self, agent_session_id: str) -> None:
        try:
            context = await self.client.get_agent_session_work_context(
                agent_session_id=agent_session_id,
            )
        except Exception as e:
            logger.warning("[linear] Failed to load AgentSession issue context: %s", e)
            return

        agent_session = context.get("agentSession") or {}
        issue = agent_session.get("issue") or {}
        if not isinstance(issue, dict) or not issue.get("id"):
            return

        viewer = context.get("viewer") or {}
        viewer_id = str(viewer.get("id") or "").strip()
        issue_id = str(issue.get("id") or "").strip()
        current_state = issue.get("state") or {}
        state_type = str(current_state.get("type") or "").strip()

        state_id = None
        if state_type not in {"started", "completed", "canceled"}:
            team = issue.get("team") or {}
            states = ((team.get("states") or {}).get("nodes") or [])
            started_states = [
                state for state in states
                if isinstance(state, dict) and state.get("id")
            ]
            if started_states:
                first_started = min(
                    started_states,
                    key=lambda state: float(state.get("position") or 0.0),
                )
                state_id = str(first_started.get("id") or "").strip() or None

        delegate_id = None
        if viewer_id and not issue.get("delegate"):
            delegate_id = viewer_id

        if not delegate_id and not state_id:
            return

        try:
            updated = await self.client.update_issue(
                issue_id=issue_id,
                delegate_id=delegate_id,
                state_id=state_id,
            )
            logger.info(
                "[linear] Prepared issue %s for AgentSession %s delegate_set=%s state_set=%s",
                updated.get("id") or issue_id,
                agent_session_id,
                bool(delegate_id),
                bool(state_id),
            )
        except Exception as e:
            logger.warning("[linear] Failed to prepare issue for AgentSession work: %s", e)

    async def on_processing_start(self, event: MessageEvent) -> None:
        agent_session_id = self._agent_session_id_from_chat_id(event.source.chat_id)
        if not agent_session_id:
            return
        await self._prepare_issue_for_agent_work(agent_session_id)
        summary = self._prompt_summary(event.text)
        try:
            activity = await self.client.create_agent_activity(
                agent_session_id=agent_session_id,
                content={
                    "type": "action",
                    "action": "Processing",
                    "parameter": summary,
                },
                ephemeral=True,
            )
            logger.info(
                "[linear] Created processing action for AgentSession %s activity=%s",
                agent_session_id,
                activity.get("id") or "?",
            )
        except Exception as e:
            logger.warning("[linear] Failed to create processing action: %s", e)
        await self._update_plan(
            agent_session_id,
            plan=[{"content": summary, "status": "inProgress"}],
        )

    async def on_processing_complete(self, event: MessageEvent, outcome: ProcessingOutcome) -> None:
        agent_session_id = self._agent_session_id_from_chat_id(event.source.chat_id)
        if not agent_session_id:
            return
        summary = self._prompt_summary(event.text)
        if outcome == ProcessingOutcome.SUCCESS:
            status = "completed"
            content = summary
        else:
            status = "canceled"
            content = f"Stopped: {summary}"
        await self._update_plan(
            agent_session_id,
            plan=[{"content": content, "status": status}],
        )

    async def send_clarify(
        self,
        chat_id: str,
        question: str,
        choices: Optional[list],
        clarify_id: str,
        session_key: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        agent_session_id = self._agent_session_id_from_chat_id(chat_id)
        if not agent_session_id:
            return SendResult(success=False, error="Missing Linear AgentSession id")
        if choices:
            from tools.clarify_gateway import mark_awaiting_text
            mark_awaiting_text(clarify_id)
        body = str(question or "").strip()
        if not body:
            body = "Please provide more information."
        activity_kwargs: Dict[str, Any] = {
            "agent_session_id": agent_session_id,
            "content": {"type": "elicitation", "body": body},
        }
        if choices:
            activity_kwargs["signal"] = "select"
            activity_kwargs["signal_metadata"] = self._select_signal_metadata(choices)
        try:
            activity = await self.client.create_agent_activity(
                **activity_kwargs,
            )
            logger.info(
                "[linear] Created elicitation activity for AgentSession %s activity=%s",
                agent_session_id,
                activity.get("id") or "?",
            )
        except Exception as e:
            logger.warning("[linear] Failed to create elicitation activity: %s", e)
            return SendResult(success=False, error="Linear elicitation failed")
        self._mark_awaiting_clarify_reply(agent_session_id)
        await self._update_plan(
            agent_session_id,
            plan=[
                {"content": self._prompt_summary(question, fallback="Await user response"), "status": "inProgress"},
                {"content": "Continue after user response", "status": "pending"},
            ],
        )
        return SendResult(success=True, message_id=activity.get("id"))

    async def send_exec_approval(
        self,
        chat_id: str,
        command: str,
        session_key: str,
        description: str = "dangerous command",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        agent_session_id = self._agent_session_id_from_chat_id(chat_id)
        if not agent_session_id:
            return SendResult(success=False, error="Missing Linear AgentSession id")
        cmd_preview = str(command or "")
        if len(cmd_preview) > 500:
            cmd_preview = cmd_preview[:497] + "..."
        reason = str(description or "dangerous command").strip()
        body = (
            "Dangerous command requires approval:\n\n"
            f"```\n{cmd_preview}\n```\n"
            f"Reason: {reason}"
        )
        try:
            activity = await self.client.create_agent_activity(
                agent_session_id=agent_session_id,
                content={"type": "elicitation", "body": body},
                signal="select",
                signal_metadata={
                    "options": [
                        {"label": "Allow once", "value": "/approve"},
                        {"label": "Allow for session", "value": "/approve session"},
                        {"label": "Always allow", "value": "/approve always"},
                        {"label": "Deny", "value": "/deny"},
                    ]
                },
            )
            logger.info(
                "[linear] Created approval elicitation for AgentSession %s activity=%s",
                agent_session_id,
                activity.get("id") or "?",
            )
        except Exception as e:
            logger.warning("[linear] Failed to create approval elicitation: %s", e)
            return SendResult(success=False, error="Linear approval prompt failed")
        await self._update_plan(
            agent_session_id,
            plan=[
                {"content": "Await command approval", "status": "inProgress"},
                {"content": self._prompt_summary(reason, fallback="Continue after approval"), "status": "pending"},
            ],
        )
        return SendResult(success=True, message_id=activity.get("id"))

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        metadata = metadata or {}
        agent_session_id = self._agent_session_id_from_chat_id(chat_id)
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
            await asyncio.gather(*tasks, return_exceptions=True)


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
        allowed_users_env="LINEAR_ALLOWED_USERS",
        allow_all_env="LINEAR_ALLOW_ALL_USERS",
        emoji="📐",
        pii_safe=True,
        allow_update_command=True,
        suppress_home_channel_prompt=True,
        platform_hint="You are communicating through Linear Agent Sessions. Keep responses actionable and issue-focused.",
    )
