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
from typing import Any, Dict, Optional

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
DEFAULT_SCOPES = ["read", "comments:create", "app:mentionable", "app:assignable"]
STATE_TTL_SECONDS = 1800
TOKEN_REFRESH_SKEW_SECONDS = 60
MAX_BODY_BYTES = 1_048_576
_TOKEN_STORE_FILENAME = "linear_oauth_tokens.json"
_STATE_STORE_FILENAME = "linear_oauth_states.json"


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
        self._session_info: Dict[str, Dict[str, Any]] = {}
        self._tokens_path = get_hermes_home() / _TOKEN_STORE_FILENAME
        self._states_path = get_hermes_home() / _STATE_STORE_FILENAME

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

        app = web.Application()
        app.router.add_get("/health", self._handle_health)
        app.router.add_get(self._authorize_path, self._handle_authorize)
        app.router.add_get(self._callback_path, self._handle_callback)
        app.router.add_post(self._webhook_path, self._handle_webhook)

        try:
            with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                sock.connect(("127.0.0.1", self._port))
            logger.error("[linear] Port %d already in use", self._port)
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

    async def disconnect(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None
            self._site = None
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
        self._session_info[chat_id] = {
            "agent_session_id": agent_session_id,
            "app_user_id": app_user_id,
            "organization_id": str(payload.get("organizationId") or agent_session.get("organizationId") or ""),
            "chat_name": chat_name,
            "updated_at": time.time(),
        }

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

        if action == "created":
            prompt_context = payload.get("promptContext") or ""
            return (
                "A new Linear Agent Session was created for you. "
                "Reply as Jax in the Linear session.\n\n"
                f"Session URL: {session_url or '(unknown)'}\n"
                f"Issue: {issue_identifier} {issue_title}\n"
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
