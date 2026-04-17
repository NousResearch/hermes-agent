"""Linear Agent platform adapter.

Provides first-class Linear Agent Session support instead of treating Linear as a
stateless generic webhook source.

Endpoints:
- GET  /health
- GET  /linear/oauth/authorize
- GET  /linear/oauth/callback
- POST /linear/webhook

Configuration lives under ``platforms.linear.extra`` and/or env vars loaded by
``gateway.config._apply_env_overrides``.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import secrets
import socket as _socket
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency gate exercised elsewhere
    AIOHTTP_AVAILABLE = False
    web = None  # type: ignore[assignment]

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8646
DEFAULT_WEBHOOK_PATH = "/linear/webhook"
DEFAULT_AUTHORIZE_PATH = "/linear/oauth/authorize"
DEFAULT_CALLBACK_PATH = "/linear/oauth/callback"
DEFAULT_SCOPES = ["read", "write", "app:mentionable", "app:assignable"]
STATE_TTL_SECONDS = 1800
TOKEN_REFRESH_SKEW_SECONDS = 300
MAX_BODY_BYTES = 1_048_576
_TOKEN_STORE_FILENAME="linear...json"
_STATE_STORE_FILENAME = "linear_oauth_states.json"
_LINEAR_APP_LOCK_SCOPE = "linear_app"
DEFAULT_MAX_CONCURRENT_SESSIONS = 3
DEFAULT_EXECUTION_MODE = "autonomous_with_testing"
SUPPORTED_EXECUTION_MODES = {
    "autonomous_dev",
    "autonomous_with_testing",
    "human_gate",
    "manual_only",
}
DEFAULT_SUPPORTED_TASK_TYPES = ["engineering", "ops", "research", "product", "admin"]
DEFAULT_TASK_TYPE = "engineering"
DEFAULT_TASK_TYPE_LABEL_PREFIX = "type:"
DEFAULT_EXECUTION_MODE_LABEL_PREFIX = "mode:"
DEFAULT_IN_PROGRESS_STATE_NAME = "In Progress"
DEFAULT_BLOCKED_STATE_NAME = "Blocked"
DEFAULT_TESTING_STATE_NAME = "Testing"
DEFAULT_TESTING_FALLBACK_STATE_NAME = "In Review"
DEFAULT_IN_REVIEW_STATE_NAME = "In Review"
DEFAULT_DONE_STATE_NAME = "Done"


def check_linear_requirements() -> bool:
    """Check if Linear adapter dependencies are available."""
    return AIOHTTP_AVAILABLE


class LinearAdapter(BasePlatformAdapter):
    """Native Linear Agent Session adapter."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.LINEAR)
        extra = config.extra or {}
        self._host = str(extra.get("host") or DEFAULT_HOST)
        self._port = int(extra.get("port") or DEFAULT_PORT)
        self._public_base_url = str(extra.get("public_base_url") or "").rstrip("/")
        self._webhook_path = str(extra.get("webhook_path") or DEFAULT_WEBHOOK_PATH)
        self._authorize_path = str(extra.get("authorize_path") or DEFAULT_AUTHORIZE_PATH)
        self._callback_path = str(extra.get("callback_path") or DEFAULT_CALLBACK_PATH)
        self._client_id = str(extra.get("client_id") or "")
        self._client_secret = str(extra.get("client_secret") or "")
        self._webhook_secret = str(extra.get("webhook_secret") or "")
        raw_scopes = extra.get("scopes") or DEFAULT_SCOPES
        if isinstance(raw_scopes, str):
            self._scopes = [p.strip() for p in raw_scopes.replace(" ", ",").split(",") if p.strip()]
        else:
            self._scopes = [str(p).strip() for p in raw_scopes if str(p).strip()]
        self._runner: Optional["web.AppRunner"] = None
        self._site: Optional["web.TCPSite"] = None
        self._max_body_bytes = int(extra.get("max_body_bytes") or MAX_BODY_BYTES)
        self._max_concurrent_sessions = max(1, int(extra.get("max_concurrent_sessions") or DEFAULT_MAX_CONCURRENT_SESSIONS))
        self._default_execution_mode = self._normalize_execution_mode(extra.get("default_execution_mode"))
        raw_project_modes = extra.get("project_execution_modes") or {}
        self._project_execution_modes = {
            str(key).strip(): self._normalize_execution_mode(value)
            for key, value in raw_project_modes.items()
            if str(key).strip()
        } if isinstance(raw_project_modes, dict) else {}
        raw_supported_task_types = extra.get("supported_task_types") or DEFAULT_SUPPORTED_TASK_TYPES
        if isinstance(raw_supported_task_types, str):
            self._supported_task_types = {
                task_type.strip().lower()
                for task_type in raw_supported_task_types.replace(" ", ",").split(",")
                if task_type.strip()
            }
        else:
            self._supported_task_types = {
                str(task_type).strip().lower()
                for task_type in raw_supported_task_types
                if str(task_type).strip()
            }
        if not self._supported_task_types:
            self._supported_task_types = {DEFAULT_TASK_TYPE}
        self._task_type_label_prefix = str(extra.get("task_type_label_prefix") or DEFAULT_TASK_TYPE_LABEL_PREFIX).strip().lower()
        self._execution_mode_label_prefix = str(extra.get("execution_mode_label_prefix") or DEFAULT_EXECUTION_MODE_LABEL_PREFIX).strip().lower()
        self._in_progress_state_name = str(extra.get("in_progress_state_name") or DEFAULT_IN_PROGRESS_STATE_NAME)
        self._blocked_state_name = str(extra.get("blocked_state_name") or DEFAULT_BLOCKED_STATE_NAME)
        self._testing_state_name = str(extra.get("testing_state_name") or DEFAULT_TESTING_STATE_NAME)
        self._testing_fallback_state_name = str(extra.get("testing_fallback_state_name") or DEFAULT_TESTING_FALLBACK_STATE_NAME)
        self._in_review_state_name = str(extra.get("in_review_state_name") or DEFAULT_IN_REVIEW_STATE_NAME)
        self._done_state_name = str(extra.get("done_state_name") or DEFAULT_DONE_STATE_NAME)
        self._session_semaphore = asyncio.Semaphore(self._max_concurrent_sessions)
        self._session_counter_lock = asyncio.Lock()
        self._running_session_count = 0
        self._queued_session_count = 0
        self._session_info: Dict[str, Dict[str, Any]] = {}
        self._issue_state_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._tokens_path = get_hermes_home() / _TOKEN_STORE_FILENAME
        self._states_path = get_hermes_home() / _STATE_STORE_FILENAME

    @staticmethod
    def _normalize_execution_mode(value: Any) -> str:
        mode = str(value or DEFAULT_EXECUTION_MODE).strip().lower()
        return mode if mode in SUPPORTED_EXECUTION_MODES else DEFAULT_EXECUTION_MODE

    def _extract_label_names(self, issue: Dict[str, Any]) -> List[str]:
        labels = issue.get("labels") or issue.get("labelIds") or []
        if isinstance(labels, dict):
            labels = labels.get("nodes") or labels.get("items") or []
        names: List[str] = []
        for label in labels:
            if isinstance(label, dict):
                name = str(label.get("name") or label.get("label") or "").strip()
            else:
                name = str(label).strip()
            if name:
                names.append(name)
        return names

    def _derive_task_type(self, issue: Dict[str, Any]) -> str:
        for label in self._extract_label_names(issue):
            lowered = label.lower()
            if lowered.startswith(self._task_type_label_prefix):
                task_type = lowered[len(self._task_type_label_prefix):].strip()
                if task_type:
                    return task_type
        return DEFAULT_TASK_TYPE

    def _derive_execution_mode(self, issue: Dict[str, Any]) -> str:
        for label in self._extract_label_names(issue):
            lowered = label.lower()
            if lowered.startswith(self._execution_mode_label_prefix):
                mode = lowered[len(self._execution_mode_label_prefix):].strip()
                return self._normalize_execution_mode(mode)

        project = issue.get("project") or {}
        project_name = str(project.get("name") or "").strip()
        project_id = str(project.get("id") or "").strip()
        if project_id and project_id in self._project_execution_modes:
            return self._project_execution_modes[project_id]
        if project_name and project_name in self._project_execution_modes:
            return self._project_execution_modes[project_name]
        return self._default_execution_mode

    def _build_execution_policy(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        task_type = self._derive_task_type(issue)
        execution_mode = self._derive_execution_mode(issue)
        can_execute = execution_mode != "manual_only" and task_type in self._supported_task_types
        block_reason = None
        if execution_mode == "manual_only":
            block_reason = "Project is configured for manual_only execution mode."
        elif task_type not in self._supported_task_types:
            block_reason = f"Task type '{task_type}' is not executable by the current Jax executor."
        return {
            "task_type": task_type,
            "execution_mode": execution_mode,
            "can_execute": can_execute,
            "block_reason": block_reason,
        }

    def _store_session_metadata(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        agent_session = payload.get("agentSession") or {}
        issue = agent_session.get("issue") or {}
        chat_id = f"linear:{agent_session.get('id') or ''}"
        policy = self._build_execution_policy(issue)
        project = issue.get("project") or {}
        creator = agent_session.get("creator") or {}
        assignee = issue.get("assignee") or {}
        session = {
            "agent_session_id": str(agent_session.get("id") or ""),
            "app_user_id": str(payload.get("appUserId") or agent_session.get("appUserId") or ""),
            "organization_id": str(payload.get("organizationId") or agent_session.get("organizationId") or ""),
            "chat_name": issue.get("identifier") or issue.get("title") or agent_session.get("url") or chat_id,
            "issue_id": str(issue.get("id") or issue.get("identifier") or ""),
            "issue_identifier": str(issue.get("identifier") or issue.get("id") or ""),
            "issue_title": str(issue.get("title") or ""),
            "team_id": str((issue.get("team") or {}).get("id") or issue.get("teamId") or ""),
            "team_name": str((issue.get("team") or {}).get("name") or issue.get("team") or ""),
            "project_id": str(project.get("id") or ""),
            "project_name": str(project.get("name") or ""),
            "label_names": self._extract_label_names(issue),
            "creator_id": str(agent_session.get("creatorId") or creator.get("id") or ""),
            "creator_name": str(creator.get("name") or ""),
            "current_assignee_id": str(assignee.get("id") or issue.get("assigneeId") or "") or None,
            "current_assignee_name": str(assignee.get("name") or issue.get("assignee") or "") or None,
            "updated_at": time.time(),
            **policy,
        }
        self._session_info[chat_id] = session
        return session

    def _determine_handoff_assignee(self, session: Dict[str, Any]) -> Optional[str]:
        current_assignee = session.get("current_assignee_id")
        if current_assignee and current_assignee != session.get("app_user_id"):
            return current_assignee
        creator_id = session.get("creator_id")
        return creator_id or None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        missing = []
        if not self._client_id:
            missing.append("client_id")
        if not self._client_secret:
            missing.append("client_secret")
        if not self._webhook_secret:
            missing.append("webhook_secret")
        if missing:
            logger.error("[linear] Missing required config: %s", ", ".join(missing))
            return False

        if not self._acquire_platform_lock(
            _LINEAR_APP_LOCK_SCOPE,
            self._client_id,
            "Linear app credentials",
        ):
            return False

        app = web.Application()
        app.router.add_get("/health", self._handle_health)
        app.router.add_get(self._authorize_path, self._handle_authorize)
        app.router.add_get(self._callback_path, self._handle_callback)
        app.router.add_post(self._webhook_path, self._handle_webhook)

        try:
            try:
                with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as sock:
                    sock.settimeout(1)
                    sock.connect(("127.0.0.1", self._port))
                logger.error("[linear] Port %d already in use", self._port)
                self._release_platform_lock()
                return False
            except (ConnectionRefusedError, OSError):
                pass

            self._runner = web.AppRunner(app)
            await self._runner.setup()
            self._site = web.TCPSite(self._runner, self._host, self._port)
            await self._site.start()
            self._mark_connected()
            logger.info(
                "[linear] Listening on %s:%d (authorize=%s callback=%s webhook=%s)",
                self._host,
                self._port,
                self._authorize_path,
                self._callback_path,
                self._webhook_path,
            )
            return True
        except Exception:
            self._release_platform_lock()
            raise

    async def disconnect(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
            self._site = None
        self._release_platform_lock()
        self._mark_disconnected()
        logger.info("[linear] Disconnected")

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        session = self._session_info.get(chat_id)
        if not session:
            return SendResult(success=False, error=f"Unknown Linear session: {chat_id}")

        activity_type = str((metadata or {}).get("linear_activity_type") or "response")
        ephemeral = bool((metadata or {}).get("ephemeral", False))
        signal = (metadata or {}).get("signal")
        try:
            result = await self._create_activity(
                app_user_id=session["app_user_id"],
                agent_session_id=session["agent_session_id"],
                activity_type=activity_type,
                body=content,
                ephemeral=ephemeral,
                signal=signal,
            )
            activity = ((result or {}).get("agentActivityCreate") or {}).get("agentActivity") or {}
            return SendResult(success=True, message_id=activity.get("id"))
        except Exception as exc:
            logger.error("[linear] Failed to send activity for %s: %s", chat_id, exc)
            return SendResult(success=False, error=str(exc))

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        info = self._session_info.get(chat_id, {})
        return {
            "name": info.get("chat_name") or chat_id,
            "type": "linear",
            "chat_id": chat_id,
        }

    # ------------------------------------------------------------------
    # HTTP handlers
    # ------------------------------------------------------------------

    async def _handle_health(self, request: "web.Request") -> "web.Response":
        return web.json_response({"status": "ok", "platform": "linear"})

    async def _handle_authorize(self, request: "web.Request") -> "web.Response":
        state = secrets.token_urlsafe(32)
        states = self._load_json(self._states_path)
        now = time.time()
        states = {
            key: value for key, value in states.items()
            if isinstance(value, dict) and now - float(value.get("created_at", 0)) < STATE_TTL_SECONDS
        }
        states[state] = {"created_at": now}
        self._save_json(self._states_path, states)
        raise web.HTTPFound(self._build_authorize_url(state))

    async def _handle_callback(self, request: "web.Request") -> "web.Response":
        code = request.query.get("code", "")
        state = request.query.get("state", "")
        error = request.query.get("error", "")
        if error:
            return web.Response(status=400, text=f"Linear authorization failed: {error}\n")
        if not code or not state:
            return web.Response(status=400, text="Missing code/state query parameters.\n")

        states = self._load_json(self._states_path)
        state_entry = states.pop(state, None)
        self._save_json(self._states_path, states)
        if not state_entry:
            return web.Response(status=400, text="Invalid or expired OAuth state.\n")
        created_at = float((state_entry or {}).get("created_at") or 0)
        if not created_at or time.time() - created_at >= STATE_TTL_SECONDS:
            return web.Response(status=400, text="Invalid or expired OAuth state.\n")

        try:
            token_data = await self._exchange_code_for_token(code)
            viewer = await self._query_viewer(token_data["access_token"])
            app_user_id = str(viewer["id"])
            stored = self._load_json(self._tokens_path)
            token_data["app_user_id"] = app_user_id
            token_data["viewer_name"] = viewer.get("name") or "Linear App"
            token_data["stored_at"] = time.time()
            stored[app_user_id] = token_data
            self._save_json(self._tokens_path, stored)
        except Exception as exc:
            logger.error("[linear] OAuth callback failed: %s", exc)
            return web.Response(status=400, text=f"Linear OAuth setup failed: {exc}\n")

        return web.Response(
            text=(
                "Linear OAuth setup complete.\n\n"
                f"App user ID: {app_user_id}\n"
                f"Webhook URL: {self._public_url(self._webhook_path)}\n"
                "Enable Linear Agent Session events and point them at the webhook URL above.\n"
            )
        )

    async def _handle_webhook(self, request: "web.Request") -> "web.Response":
        content_length = request.content_length or 0
        if content_length > self._max_body_bytes:
            return web.json_response({"error": "Payload too large"}, status=413)

        try:
            raw_body = await request.read()
        except Exception:
            return web.json_response({"error": "Failed to read body"}, status=400)

        if not self._validate_signature(raw_body, request.headers.get("Linear-Signature", "")):
            return web.json_response({"error": "Invalid signature"}, status=401)

        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        action = str(payload.get("action") or "")
        if action not in {"created", "prompted"}:
            return web.json_response({"status": "ignored", "action": action or "unknown"}, status=200)

        agent_session = payload.get("agentSession") or {}
        agent_session_id = str(agent_session.get("id") or "")
        app_user_id = str(payload.get("appUserId") or agent_session.get("appUserId") or "")
        if not agent_session_id or not app_user_id:
            return web.json_response({"error": "Missing agent session/app user IDs"}, status=400)

        try:
            await self._ensure_access_token(app_user_id)
        except Exception as exc:
            logger.error("[linear] No usable OAuth token for app user %s: %s", app_user_id, exc)
            return web.json_response({"error": f"OAuth token unavailable: {exc}"}, status=503)

        chat_id = f"linear:{agent_session_id}"
        issue = agent_session.get("issue") or {}
        chat_name = issue.get("identifier") or issue.get("title") or agent_session.get("url") or chat_id
        session = self._store_session_metadata(payload)
        session["chat_name"] = chat_name

        try:
            await self._create_activity(
                app_user_id=app_user_id,
                agent_session_id=agent_session_id,
                activity_type="thought",
                body="Jax is looking into this…",
                ephemeral=True,
            )
        except Exception:
            logger.debug("[linear] Initial ephemeral acknowledgement failed", exc_info=True)

        prompt = self._build_prompt(payload)
        creator = agent_session.get("creator") or {}
        source = self.build_source(
            chat_id=chat_id,
            chat_name=chat_name,
            chat_type="thread",
            user_id=str(agent_session.get("creatorId") or f"linear:{app_user_id}"),
            user_name=str(creator.get("name") or "Linear user"),
        )
        message_id = str(payload.get("webhookId") or f"{agent_session_id}:{action}:{int(time.time() * 1000)}")
        event = MessageEvent(
            text=prompt,
            message_type=MessageType.TEXT,
            source=source,
            raw_message=payload,
            message_id=message_id,
        )

        task = asyncio.create_task(self.handle_message(event))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return web.json_response(
            {
                "status": "accepted",
                "platform": "linear",
                "action": action,
                "agent_session_id": agent_session_id,
            },
            status=202,
        )

    async def _process_message_background(self, event: MessageEvent, session_key: str) -> None:
        session = self._session_info.get(event.source.chat_id) or {}
        if session and not session.get("can_execute", True):
            reason = session.get("block_reason") or "Task is not executable automatically."
            assignee_id = self._determine_handoff_assignee(session)
            await self._transition_issue_for_session(
                session,
                self._blocked_state_name,
                assignee_id=assignee_id,
                comment=f"Jax cannot execute this issue automatically: {reason}",
            )
            await self._maybe_send_queue_activity(
                event.source.chat_id,
                f"Jax cannot execute this task automatically and moved it to {self._blocked_state_name} for Pablo. Reason: {reason}",
            )
            return

        queue_notice_needed = False

        async with self._session_counter_lock:
            if self._running_session_count >= self._max_concurrent_sessions:
                self._queued_session_count += 1
                queue_notice_needed = True

        if queue_notice_needed:
            await self._maybe_send_queue_activity(
                event.source.chat_id,
                f"Jax queued this session and will pick it up once one of the {self._max_concurrent_sessions} active slots frees up.",
            )

        acquired = False
        counted_running = False
        queued_count_decremented = not queue_notice_needed
        try:
            await self._session_semaphore.acquire()
            acquired = True

            async with self._session_counter_lock:
                if queue_notice_needed and self._queued_session_count > 0:
                    self._queued_session_count -= 1
                    queued_count_decremented = True
                self._running_session_count += 1
                counted_running = True

            if queue_notice_needed:
                await self._maybe_send_queue_activity(
                    event.source.chat_id,
                    "Jax is starting work on this session now.",
                )

            await super()._process_message_background(event, session_key)
        finally:
            async with self._session_counter_lock:
                if not queued_count_decremented and self._queued_session_count > 0:
                    self._queued_session_count -= 1
                if counted_running and self._running_session_count > 0:
                    self._running_session_count -= 1
            if acquired:
                self._session_semaphore.release()

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------

    def _build_prompt(self, payload: Dict[str, Any]) -> str:
        action = str(payload.get("action") or "")
        agent_session = payload.get("agentSession") or {}
        issue = agent_session.get("issue") or {}
        session_url = agent_session.get("url") or ""
        issue_identifier = issue.get("identifier") or ""
        issue_title = issue.get("title") or ""
        guidance = payload.get("guidance") or []
        guidance_text = json.dumps(guidance, indent=2)[:3000] if guidance else "[]"
        session = self._session_info.get(f"linear:{agent_session.get('id') or ''}", {})
        flow_context = (
            f"Task type: {session.get('task_type', DEFAULT_TASK_TYPE)}\n"
            f"Project execution mode: {session.get('execution_mode', self._default_execution_mode)}\n"
            f"Auto-executable by current Jax executor: {'yes' if session.get('can_execute', True) else 'no'}\n"
        )

        if action == "created":
            prompt_context = payload.get("promptContext") or ""
            return (
                "A new Linear Agent Session was created for you. "
                "Reply as Jax in the Linear session.\n\n"
                f"Session URL: {session_url or '(unknown)'}\n"
                f"Issue: {issue_identifier} {issue_title}\n"
                f"{flow_context}"
                f"Guidance:\n```json\n{guidance_text}\n```\n\n"
                "Use the following Linear-provided promptContext as the authoritative context:\n\n"
                f"```text\n{prompt_context[:12000]}\n```"
            )

        activity = payload.get("agentActivity") or {}
        content = activity.get("content") or {}
        body = content.get("body") or json.dumps(content, indent=2)[:4000]
        signal = activity.get("signal")
        return (
            "A user added a follow-up prompt to an existing Linear Agent Session.\n\n"
            f"Session URL: {session_url or '(unknown)'}\n"
            f"Issue: {issue_identifier} {issue_title}\n"
            f"Signal: {signal or '(none)'}\n"
            f"User message:\n\n{body}"
        )

    # ------------------------------------------------------------------
    # Linear API helpers
    # ------------------------------------------------------------------

    def _validate_signature(self, body: bytes, signature: str) -> bool:
        if not signature:
            return False
        expected = hmac.new(self._webhook_secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(signature, expected)

    def _build_authorize_url(self, state: str) -> str:
        params = urllib.parse.urlencode(
            {
                "client_id": self._client_id,
                "redirect_uri": self._public_url(self._callback_path),
                "response_type": "code",
                "scope": ",".join(self._scopes),
                "state": state,
                "actor": "app",
                "prompt": "consent",
            }
        )
        return f"https://linear.app/oauth/authorize?{params}"

    def _public_url(self, path: str) -> str:
        if self._public_base_url:
            return f"{self._public_base_url}{path}"
        host = self._host
        display_host = "127.0.0.1" if host == "0.0.0.0" else host
        return f"http://{display_host}:{self._port}{path}"

    async def _exchange_code_for_token(self, code: str) -> Dict[str, Any]:
        form = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self._public_url(self._callback_path),
            "client_id": self._client_id,
            "client_secret": self._client_secret,
        }
        token_data = await asyncio.to_thread(
            self._http_form,
            "https://api.linear.app/oauth/token",
            form,
            {"Content-Type": "application/x-www-form-urlencoded"},
        )
        token_data["obtained_at"] = time.time()
        expires_in = int(token_data.get("expires_in") or 0)
        if expires_in > 0:
            token_data["expires_at"] = token_data["obtained_at"] + expires_in
        return token_data

    async def _refresh_token(self, app_user_id: str, refresh_token: str) -> Dict[str, Any]:
        form = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self._client_id,
            "client_secret": self._client_secret,
        }
        token_data = await asyncio.to_thread(
            self._http_form,
            "https://api.linear.app/oauth/token",
            form,
            {"Content-Type": "application/x-www-form-urlencoded"},
        )
        stored = self._load_json(self._tokens_path)
        existing = stored.get(app_user_id, {})
        token_data["obtained_at"] = time.time()
        expires_in = int(token_data.get("expires_in") or 0)
        if expires_in > 0:
            token_data["expires_at"] = token_data["obtained_at"] + expires_in
        if not token_data.get("refresh_token"):
            token_data["refresh_token"] = refresh_token
        token_data["app_user_id"] = app_user_id
        token_data["viewer_name"] = existing.get("viewer_name") or existing.get("name")
        token_data["stored_at"] = time.time()
        stored[app_user_id] = token_data
        self._save_json(self._tokens_path, stored)
        return token_data

    async def _ensure_access_token(self, app_user_id: str) -> str:
        stored = self._load_json(self._tokens_path)
        token = stored.get(app_user_id)
        if not token:
            raise RuntimeError(f"No stored OAuth token for app user {app_user_id}")
        access_token = str(token.get("access_token") or "")
        expires_at = float(token.get("expires_at") or 0)
        refresh_token = str(token.get("refresh_token") or "")
        if access_token and (not expires_at or expires_at - time.time() > TOKEN_REFRESH_SKEW_SECONDS):
            return access_token
        if not refresh_token:
            if access_token:
                return access_token
            raise RuntimeError(f"Stored token for {app_user_id} has expired and no refresh token is available")
        refreshed = await self._refresh_token(app_user_id, refresh_token)
        return str(refreshed.get("access_token") or "")

    async def _query_viewer(self, access_token: str) -> Dict[str, Any]:
        payload = {
            "query": "query { viewer { id name } }",
        }
        result = await asyncio.to_thread(
            self._http_json,
            "https://api.linear.app/graphql",
            payload,
            {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
        )
        viewer = ((result.get("data") or {}).get("viewer") or {})
        if not viewer.get("id"):
            raise RuntimeError(f"Viewer query returned no app user ID: {result}")
        return viewer

    async def _create_activity(
        self,
        *,
        app_user_id: str,
        agent_session_id: str,
        activity_type: str,
        body: str,
        ephemeral: bool = False,
        signal: Optional[str] = None,
    ) -> Dict[str, Any]:
        access_token = await self._ensure_access_token(app_user_id)
        content = {
            "type": activity_type,
            "body": body,
        }
        payload = {
            "query": (
                "mutation($input: AgentActivityCreateInput!) { "
                "agentActivityCreate(input: $input) { success agentActivity { id } } }"
            ),
            "variables": {
                "input": {
                    "agentSessionId": agent_session_id,
                    "content": content,
                    "ephemeral": ephemeral,
                }
            },
        }
        if signal:
            payload["variables"]["input"]["signal"] = signal
        result = await asyncio.to_thread(
            self._http_json,
            "https://api.linear.app/graphql",
            payload,
            {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
        )
        errors = result.get("errors") or []
        if errors:
            raise RuntimeError(errors[0].get("message") or str(errors[0]))
        create_payload = ((result.get("data") or {}).get("agentActivityCreate") or {})
        if not create_payload.get("success"):
            raise RuntimeError(f"agentActivityCreate failed: {result}")
        return (result.get("data") or {})

    async def _maybe_send_queue_activity(self, chat_id: str, body: str) -> None:
        session = self._session_info.get(chat_id)
        if not session:
            return
        try:
            await self._create_activity(
                app_user_id=session["app_user_id"],
                agent_session_id=session["agent_session_id"],
                activity_type="thought",
                body=body,
                ephemeral=True,
            )
        except Exception:
            logger.debug("[linear] Queue status activity failed for %s", chat_id, exc_info=True)

    async def on_processing_start(self, event: MessageEvent) -> None:
        session = self._session_info.get(event.source.chat_id)
        if not session or not session.get("can_execute", True):
            return
        await self._transition_issue_for_session(
            session,
            self._in_progress_state_name,
            assignee_id=session.get("current_assignee_id"),
            comment=(
                "Jax started working on this issue automatically "
                f"(task type: {session.get('task_type', DEFAULT_TASK_TYPE)}, "
                f"mode: {session.get('execution_mode', self._default_execution_mode)})."
            ),
        )

    async def on_processing_complete(self, event: MessageEvent, outcome) -> None:
        session = self._session_info.get(event.source.chat_id)
        if not session or not session.get("can_execute", True):
            return

        assignee_id = session.get("current_assignee_id")
        execution_mode = session.get("execution_mode", self._default_execution_mode)
        if outcome.name == "SUCCESS":
            if execution_mode == "autonomous_dev":
                await self._transition_issue_for_session(
                    session,
                    self._done_state_name,
                    assignee_id=assignee_id,
                    comment="Jax finished implementation work and marked this issue Done automatically.",
                )
                return

            if execution_mode == "human_gate":
                await self._transition_issue_for_session(
                    session,
                    self._in_review_state_name,
                    assignee_id=assignee_id,
                    comment="Jax finished implementation work and left this issue in In Review for human approval.",
                )
                return

            target_state = self._testing_state_name
            comment = (
                "Jax finished implementation work and moved this issue to "
                f"{self._testing_state_name} (task type: {session.get('task_type', DEFAULT_TASK_TYPE)}, "
                f"mode: {execution_mode})."
            )
            if not await self._state_name_exists(session.get("team_id"), self._testing_state_name, session.get("app_user_id")):
                target_state = self._testing_fallback_state_name
                comment = (
                    "Jax finished implementation work and moved this issue to "
                    f"{self._testing_fallback_state_name} because the team has no {self._testing_state_name} state configured."
                )
            await self._transition_issue_for_session(
                session,
                target_state,
                assignee_id=assignee_id,
                comment=comment,
            )
            return

        await self._transition_issue_for_session(
            session,
            self._blocked_state_name,
            assignee_id=assignee_id,
            comment="Jax could not finish this issue and moved it to Blocked for follow-up.",
        )

    async def _transition_issue_for_session(
        self,
        session: Dict[str, Any],
        target_state: str,
        *,
        assignee_id: Optional[str] = None,
        comment: Optional[str] = None,
    ) -> None:
        issue_id = str(session.get("issue_id") or session.get("issue_identifier") or "")
        team_id = str(session.get("team_id") or "")
        app_user_id = str(session.get("app_user_id") or "")
        if issue_id and team_id and target_state:
            await self._update_issue_state(issue_id, team_id, app_user_id, target_state, assignee_id=assignee_id)
        if comment and session.get("agent_session_id"):
            try:
                await self._create_activity(
                    app_user_id=app_user_id,
                    agent_session_id=str(session.get("agent_session_id") or ""),
                    activity_type="thought",
                    body=comment,
                    ephemeral=False,
                )
            except Exception:
                logger.debug("[linear] Issue transition activity failed for %s", issue_id, exc_info=True)

    async def _state_name_exists(self, team_id: str, state_name: str, app_user_id: Optional[str] = None) -> bool:
        if not team_id or not app_user_id:
            return False
        states = await self._list_issue_states(team_id, app_user_id)
        target = state_name.strip().lower()
        return any(str(state.get("name") or "").strip().lower() == target for state in states)

    async def _list_issue_states(self, team_id: str, app_user_id: str) -> List[Dict[str, Any]]:
        if not team_id:
            return []
        if team_id in self._issue_state_cache:
            return self._issue_state_cache[team_id]
        access_token = await self._ensure_access_token(app_user_id)
        payload = {
            "query": (
                "query { workflowStates(filter: { team: { id: { eq: \""
                + team_id
                + "\" } } }) { nodes { id name type } } }"
            )
        }
        result = await asyncio.to_thread(
            self._http_json,
            "https://api.linear.app/graphql",
            payload,
            {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
        )
        states = ((result.get("data") or {}).get("workflowStates") or {}).get("nodes") or []
        self._issue_state_cache[team_id] = states
        return states

    async def _update_issue_state(
        self,
        issue_id: str,
        team_id: str,
        app_user_id: str,
        state_name: str,
        *,
        assignee_id: Optional[str] = None,
    ) -> None:
        states = await self._list_issue_states(team_id, app_user_id)
        target_state = next(
            (state for state in states if str(state.get("name") or "").strip().lower() == state_name.strip().lower()),
            None,
        )
        if not target_state:
            raise RuntimeError(f"Linear state '{state_name}' not found for team {team_id}")
        access_token = await self._ensure_access_token(app_user_id)
        input_parts = [f'stateId: "{target_state["id"]}"']
        if assignee_id:
            input_parts.append(f'assigneeId: "{assignee_id}"')
        payload = {
            "query": (
                "mutation { issueUpdate(id: \""
                + issue_id
                + "\", input: { "
                + ", ".join(input_parts)
                + " }) { success issue { id identifier state { name type } assignee { id name } } } }"
            )
        }
        result = await asyncio.to_thread(
            self._http_json,
            "https://api.linear.app/graphql",
            payload,
            {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
        )
        errors = result.get("errors") or []
        if errors:
            raise RuntimeError(errors[0].get("message") or str(errors[0]))
        update_payload = ((result.get("data") or {}).get("issueUpdate") or {})
        if not update_payload.get("success"):
            raise RuntimeError(f"issueUpdate failed: {result}")

    async def _revoke_token(self, access_token: str) -> None:
        if not access_token:
            return
        await asyncio.to_thread(
            self._http_form,
            "https://api.linear.app/oauth/revoke",
            {},
            {"Authorization": f"Bearer {access_token}"},
        )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _load_json(self, path: Path) -> Dict[str, Any]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except FileNotFoundError:
            return {}
        except Exception:
            logger.warning("[linear] Failed to load JSON store %s", path)
            return {}

    def _save_json(self, path: Path, data: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

    # ------------------------------------------------------------------
    # Blocking HTTP helpers (run via asyncio.to_thread)
    # ------------------------------------------------------------------

    def _http_json(self, url: str, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=body, method="POST")
        for key, value in headers.items():
            request.add_header(key, value)
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read().decode("utf-8")
        return json.loads(data) if data else {}

    def _http_form(self, url: str, form: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        body = urllib.parse.urlencode({k: str(v) for k, v in form.items()}).encode("utf-8")
        request = urllib.request.Request(url, data=body, method="POST")
        for key, value in headers.items():
            request.add_header(key, value)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            payload = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {payload}") from exc
        return json.loads(data) if data else {}
