"""Mattermost gateway adapter.

Connects to a self-hosted (or cloud) Mattermost instance via its REST API
(v4) and WebSocket for real-time events.  No external Mattermost library
required — uses aiohttp which is already a Hermes dependency.

Environment variables:
    MATTERMOST_URL              Server URL (e.g. https://mm.example.com)
    MATTERMOST_TOKEN            Bot token or personal-access token
    MATTERMOST_ALLOWED_USERS    Comma-separated user IDs
    MATTERMOST_HOME_CHANNEL     Channel ID for cron/notification delivery
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gateway.config import Platform, PlatformConfig
from gateway.platforms.helpers import MessageDeduplicator
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

logger = logging.getLogger(__name__)

# Mattermost post size limit (server default is 16383, but 4000 is the
# practical limit for readable messages — matching OpenClaw's choice).
MAX_POST_LENGTH = 4000

# Channel type codes returned by the Mattermost API.
_CHANNEL_TYPE_MAP = {
    "D": "dm",
    "G": "group",
    "P": "group",   # private channel → treat as group
    "O": "channel",
}

# Reconnect parameters (exponential backoff).
_RECONNECT_BASE_DELAY = 2.0
_RECONNECT_MAX_DELAY = 60.0
_RECONNECT_JITTER = 0.2

# Mattermost action callback defaults for interactive clarify buttons.
_DEFAULT_ACTIONS_HOST = "0.0.0.0"
_DEFAULT_ACTIONS_PORT = 8769
_ACTIONS_PATH = "/mattermost/actions"
_MATTERMOST_ACTION_ID_RE = re.compile(r"[^A-Za-z0-9]")


def _truthy(value: Any, default: bool = False) -> bool:
    """Coerce common config/env truthy strings."""
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def check_mattermost_requirements() -> bool:
    """Return True if the Mattermost adapter can be used."""
    token = os.getenv("MATTERMOST_TOKEN", "")
    url = os.getenv("MATTERMOST_URL", "")
    if not token:
        logger.debug("Mattermost: MATTERMOST_TOKEN not set")
        return False
    if not url:
        logger.warning("Mattermost: MATTERMOST_URL not set")
        return False
    try:
        import aiohttp  # noqa: F401
        return True
    except ImportError:
        logger.warning("Mattermost: aiohttp not installed")
        return False


class MattermostAdapter(BasePlatformAdapter):
    """Gateway adapter for Mattermost (self-hosted or cloud)."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.MATTERMOST)

        self._base_url: str = (
            config.extra.get("url", "")
            or os.getenv("MATTERMOST_URL", "")
        ).rstrip("/")
        self._token: str = config.token or os.getenv("MATTERMOST_TOKEN", "")

        self._bot_user_id: str = ""
        self._bot_username: str = ""

        # aiohttp session + websocket handle
        self._session: Any = None  # aiohttp.ClientSession
        self._ws: Any = None       # aiohttp.ClientWebSocketResponse
        self._ws_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        self._closing = False

        # Reply mode: "thread" to nest replies, "off" for flat messages.
        self._reply_mode: str = (
            config.extra.get("reply_mode", "")
            or os.getenv("MATTERMOST_REPLY_MODE", "off")
        ).lower()

        # Dedup cache (prevent reprocessing)
        self._dedup = MessageDeduplicator()

        # Optional interactive-message callback server for clarify buttons.
        # Mattermost action buttons POST back to an integration URL; when
        # configured, Hermes renders clarify choices as real buttons and
        # resolves them through this lightweight aiohttp endpoint.
        actions_url = (
            config.extra.get("actions_url", "")
            or os.getenv("MATTERMOST_ACTIONS_URL", "")
        ).strip()
        self._actions_url: str = actions_url.rstrip("/")
        self._actions_enabled: bool = _truthy(
            config.extra.get("actions_enabled", os.getenv("MATTERMOST_ACTIONS_ENABLED", "")),
            default=bool(self._actions_url),
        )
        self._actions_host: str = (
            config.extra.get("actions_host", "")
            or os.getenv("MATTERMOST_ACTIONS_HOST", _DEFAULT_ACTIONS_HOST)
        )
        self._actions_port: int = int(
            config.extra.get("actions_port", "")
            or os.getenv("MATTERMOST_ACTIONS_PORT", str(_DEFAULT_ACTIONS_PORT))
        )
        self._actions_runner: Any = None
        self._actions_server_ready: bool = False
        # One-time action tokens keyed by the opaque value embedded in the
        # Mattermost button context. Do not put reusable secrets in message
        # props: post props may be visible via clients/API export.
        self._clarify_action_tokens: Dict[str, Dict[str, Any]] = {}
        self._approval_action_tokens: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

    async def _api_get(self, path: str) -> Dict[str, Any]:
        """GET /api/v4/{path}."""
        import aiohttp
        url = f"{self._base_url}/api/v4/{path.lstrip('/')}"
        try:
            async with self._session.get(url, headers=self._headers(), timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    logger.error("MM API GET %s → %s: %s", path, resp.status, body[:200])
                    return {}
                return await resp.json()
        except aiohttp.ClientError as exc:
            logger.error("MM API GET %s network error: %s", path, exc)
            return {}

    async def _api_post(
        self, path: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """POST /api/v4/{path} with JSON body."""
        import aiohttp
        url = f"{self._base_url}/api/v4/{path.lstrip('/')}"
        try:
            async with self._session.post(
                url, headers=self._headers(), json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    logger.error("MM API POST %s → %s: %s", path, resp.status, body[:200])
                    return {}
                return await resp.json()
        except aiohttp.ClientError as exc:
            logger.error("MM API POST %s network error: %s", path, exc)
            return {}

    async def _api_put(
        self, path: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """PUT /api/v4/{path} with JSON body."""
        import aiohttp
        url = f"{self._base_url}/api/v4/{path.lstrip('/')}"
        try:
            async with self._session.put(
                url, headers=self._headers(), json=payload
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    logger.error("MM API PUT %s → %s: %s", path, resp.status, body[:200])
                    return {}
                return await resp.json()
        except aiohttp.ClientError as exc:
            logger.error("MM API PUT %s network error: %s", path, exc)
            return {}

    async def _upload_file(
        self, channel_id: str, file_data: bytes, filename: str, content_type: str = "application/octet-stream"
    ) -> Optional[str]:
        """Upload a file and return its file ID, or None on failure."""
        import aiohttp

        url = f"{self._base_url}/api/v4/files"
        form = aiohttp.FormData()
        form.add_field("channel_id", channel_id)
        form.add_field(
            "files",
            file_data,
            filename=filename,
            content_type=content_type,
        )
        headers = {"Authorization": f"Bearer {self._token}"}
        async with self._session.post(url, headers=headers, data=form, timeout=aiohttp.ClientTimeout(total=60)) as resp:
            if resp.status >= 400:
                body = await resp.text()
                logger.error("MM file upload → %s: %s", resp.status, body[:200])
                return None
            data = await resp.json()
            infos = data.get("file_infos", [])
            return infos[0]["id"] if infos else None

    # ------------------------------------------------------------------
    # Required overrides
    # ------------------------------------------------------------------

    async def connect(self) -> bool:
        """Connect to Mattermost and start the WebSocket listener."""
        import aiohttp

        if not self._base_url or not self._token:
            logger.error("Mattermost: URL or token not configured")
            return False

        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        self._closing = False

        # Verify credentials and fetch bot identity.
        me = await self._api_get("users/me")
        if not me or "id" not in me:
            logger.error("Mattermost: failed to authenticate — check MATTERMOST_TOKEN and MATTERMOST_URL")
            await self._session.close()
            return False

        self._bot_user_id = me["id"]
        self._bot_username = me.get("username", "")
        logger.info(
            "Mattermost: authenticated as @%s (%s) on %s",
            self._bot_username,
            self._bot_user_id,
            self._base_url,
        )

        # Start WebSocket in background.
        self._ws_task = asyncio.create_task(self._ws_loop())
        await self._start_actions_server()
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        """Disconnect from Mattermost."""
        self._closing = True

        if self._ws_task and not self._ws_task.done():
            self._ws_task.cancel()
            try:
                await self._ws_task
            except (asyncio.CancelledError, Exception):
                pass

        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()

        if self._ws:
            await self._ws.close()
            self._ws = None

        await self._stop_actions_server()

        if self._session and not self._session.closed:
            await self._session.close()

        logger.info("Mattermost: disconnected")


    async def _resolve_root_id(self, post_id: str) -> str:
        """Resolve a post_id to the thread root_id for Mattermost.

        Mattermost requires root_id to be the *root* post of a thread.
        If the post is a reply (has its own root_id), we must use that
        root_id instead.  Using a reply's own ID as root_id causes
        "Invalid RootId parameter" errors.
        """
        if not post_id:
            return post_id
        # Check if this post has a root_id (meaning it's a reply)
        data = await self._api_get(f"posts/{post_id}")
        if data and data.get("root_id"):
            return data["root_id"]
        return post_id

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a message (or multiple chunks) to a channel."""
        if not content:
            return SendResult(success=True)

        formatted = self.format_message(content)
        chunks = self.truncate_message(formatted, MAX_POST_LENGTH)

        last_id = None
        for chunk in chunks:
            payload: Dict[str, Any] = {
                "channel_id": chat_id,
                "message": chunk,
            }
            # Thread support: reply_to is the root post ID.
            if reply_to and self._reply_mode == "thread":
                # Ensure root_id points to the thread root, not a reply.
                # Mattermost rejects non-root post IDs as root_id.
                resolved_root = await self._resolve_root_id(reply_to)
                payload["root_id"] = resolved_root

            data = await self._api_post("posts", payload)
            if not data or "id" not in data:
                return SendResult(success=False, error="Failed to create post")
            last_id = data["id"]

        return SendResult(success=True, message_id=last_id)

    def _actions_endpoint_url(self, kind: str = "clarify") -> str:
        """Return the Mattermost integration callback URL for action buttons."""
        if not self._actions_url:
            return ""
        suffix = (kind or "clarify").strip("/")
        base = self._actions_url.rstrip("/")
        for known in ("/clarify", "/approval"):
            if base.endswith(known):
                base = base[: -len(known)]
                break
        return f"{base}/{suffix}"

    async def send_exec_approval(
        self,
        chat_id: str,
        command: str,
        session_key: str,
        description: str = "dangerous command",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Ask for dangerous-command approval with Mattermost action buttons."""
        endpoint = self._actions_endpoint_url("approval")
        if not self._actions_enabled or not endpoint or not self._actions_server_ready:
            return SendResult(success=False, error="Mattermost action callback server is not ready")

        expected_user_id = ""
        thread_id = ""
        if isinstance(metadata, dict):
            expected_user_id = str(metadata.get("user_id") or "")
            thread_id = str(metadata.get("thread_id") or "")

        prompt_id = secrets.token_hex(6)

        def _register_approval_token(choice: str) -> str:
            token = secrets.token_urlsafe(24)
            self._approval_action_tokens[token] = {
                "session_key": session_key,
                "chat_id": chat_id,
                "thread_id": thread_id,
                "expected_user_id": expected_user_id,
                "choice": choice,
            }
            return token

        button_defs = [
            ("once", "Allow Once", "success"),
            ("session", "Allow Session", "primary"),
            ("always", "Always Allow", "primary"),
            ("deny", "Deny", "danger"),
        ]
        actions: List[Dict[str, Any]] = []
        for choice, label, style in button_defs:
            token = _register_approval_token(choice)
            actions.append({
                # Mattermost action IDs must match [A-Za-z0-9]+ or the click
                # 404s before reaching the integration callback.
                "id": f"approval{prompt_id}{choice}",
                "name": label,
                "type": "button",
                "style": style,
                "integration": {
                    "url": endpoint,
                    "context": {
                        "kind": "approval",
                        "token": token,
                    },
                },
            })

        cmd_preview = command if len(command) <= 3200 else f"{command[:3197]}..."
        payload: Dict[str, Any] = {
            "channel_id": chat_id,
            "message": "⚠️ **Dangerous command requires approval:**",
            "props": {
                "attachments": [{
                    "text": f"```\n{cmd_preview}\n```\nReason: {description}",
                    "actions": actions,
                }],
            },
        }
        if thread_id and self._reply_mode == "thread":
            payload["root_id"] = await self._resolve_root_id(thread_id)

        data = await self._api_post("posts", payload)
        if not data or "id" not in data:
            self._clear_approval_action_tokens(session_key)
            return SendResult(success=False, error="Failed to create Mattermost approval post")
        return SendResult(success=True, message_id=data.get("id"))

    def _clear_approval_action_tokens(self, session_key: str) -> None:
        """Drop all one-time Mattermost action tokens for an approval session."""
        stale = [
            token for token, state in self._approval_action_tokens.items()
            if state.get("session_key") == session_key
        ]
        for token in stale:
            self._approval_action_tokens.pop(token, None)

    async def send_clarify(
        self,
        chat_id: str,
        question: str,
        choices: Optional[List[str]],
        clarify_id: str,
        session_key: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Ask a clarify question with Mattermost action buttons when possible."""
        from tools import clarify_gateway as cm

        # Open-ended questions still use text capture; there is no button UI to render.
        if not choices:
            cm.mark_awaiting_text(clarify_id)
            return await self.send(chat_id, f"❓ {question}", metadata=metadata)

        endpoint = self._actions_endpoint_url("clarify")
        if not self._actions_enabled or not endpoint or not self._actions_server_ready:
            # No live callback endpoint configured; preserve the BasePlatform numbered-text fallback.
            return await super().send_clarify(chat_id, question, choices, clarify_id, session_key, metadata=metadata)

        expected_user_id = ""
        thread_id = ""
        if isinstance(metadata, dict):
            expected_user_id = str(metadata.get("user_id") or "")
            thread_id = str(metadata.get("thread_id") or "")

        actions: List[Dict[str, Any]] = []

        def _register_action_token(choice_value: str, choice_index: Optional[int] = None) -> str:
            token = secrets.token_urlsafe(24)
            self._clarify_action_tokens[token] = {
                "clarify_id": clarify_id,
                "session_key": session_key,
                "chat_id": chat_id,
                "thread_id": thread_id,
                "expected_user_id": expected_user_id,
                "choice_index": choice_index,
                "choice": choice_value,
            }
            return token

        safe_clarify_id = _MATTERMOST_ACTION_ID_RE.sub("", str(clarify_id)) or "prompt"
        for idx, choice in enumerate(choices):
            label = str(choice)
            display = label if len(label) <= 64 else f"{label[:61]}..."
            token = _register_action_token(label, idx)
            actions.append({
                # Mattermost routes post actions through
                # /api/v4/posts/{post_id}/actions/{action_id:[A-Za-z0-9]+};
                # hyphens render fine but make clicks 404 before they reach
                # our integration callback.
                "id": f"clarify{safe_clarify_id}{idx}",
                "name": f"{idx + 1}. {display}",
                "type": "button",
                "style": "default",
                "integration": {
                    "url": endpoint,
                    "context": {
                        "kind": "clarify",
                        "token": token,
                    },
                },
            })

        other_token = _register_action_token("__other__", None)
        actions.append({
            "id": f"clarify{safe_clarify_id}other",
            "name": "Other",
            "type": "button",
            "style": "primary",
            "integration": {
                "url": endpoint,
                "context": {
                    "kind": "clarify",
                    "token": other_token,
                },
            },
        })

        payload: Dict[str, Any] = {
            "channel_id": chat_id,
            "message": f"❓ {question}",
            "props": {
                "attachments": [{
                    "text": "Choose one of the options below, or click Other to type a custom answer.",
                    "actions": actions,
                }]
            },
        }
        data = await self._api_post("posts", payload)
        if not data or "id" not in data:
            logger.warning("Mattermost clarify buttons failed; falling back to numbered text prompt")
            self._clear_clarify_action_tokens(clarify_id)
            return await super().send_clarify(chat_id, question, choices, clarify_id, session_key, metadata=metadata)
        return SendResult(success=True, message_id=data.get("id"))

    def _clear_clarify_action_tokens(self, clarify_id: str) -> None:
        """Drop all one-time Mattermost action tokens for a clarify prompt."""
        stale = [
            token for token, state in self._clarify_action_tokens.items()
            if state.get("clarify_id") == clarify_id
        ]
        for token in stale:
            self._clarify_action_tokens.pop(token, None)

    async def _start_actions_server(self) -> None:
        """Start the optional Mattermost action callback server."""
        if not self._actions_enabled or not self._actions_url or self._actions_runner:
            return
        try:
            from aiohttp import web
        except ImportError:
            logger.warning("Mattermost action buttons disabled: aiohttp.web unavailable")
            return

        app = web.Application()
        app.router.add_get(f"{_ACTIONS_PATH}/health", self._handle_actions_health)
        app.router.add_post(f"{_ACTIONS_PATH}/clarify", self._handle_clarify_action_request)
        app.router.add_post(f"{_ACTIONS_PATH}/approval", self._handle_approval_action_request)
        try:
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, self._actions_host, self._actions_port)
            await site.start()
        except Exception as exc:
            logger.warning(
                "Mattermost action buttons disabled: failed to listen on %s:%s (%s)",
                self._actions_host,
                self._actions_port,
                exc,
            )
            try:
                await runner.cleanup()  # type: ignore[possibly-undefined]
            except Exception:
                pass
            return

        self._actions_runner = runner
        self._actions_server_ready = True
        logger.info(
            "Mattermost action callbacks listening on %s:%s%s (public URL: %s)",
            self._actions_host,
            self._actions_port,
            _ACTIONS_PATH,
            self._actions_url,
        )

    async def _stop_actions_server(self) -> None:
        """Stop the optional Mattermost action callback server."""
        runner = self._actions_runner
        if not runner:
            self._actions_server_ready = False
            return
        self._actions_runner = None
        self._actions_server_ready = False
        try:
            await runner.cleanup()
        except Exception:
            logger.debug("Mattermost action callback cleanup failed", exc_info=True)

    async def _handle_actions_health(self, request: Any) -> Any:
        from aiohttp import web
        return web.json_response({"ok": True, "platform": "mattermost"})

    async def _handle_clarify_action_request(self, request: Any) -> Any:
        from aiohttp import web
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": {"message": "Invalid JSON payload."}}, status=400)
        response = await self._handle_clarify_action_payload(payload)
        if "error" in response:
            # Mattermost renders non-2xx action callback responses as the
            # generic "Sorry, we could not find the page."  Return a normal
            # interactive-message response with an ephemeral error instead so
            # stale/timed-out buttons explain themselves without looking like a
            # broken route.
            error = response.get("error") or {}
            message = error.get("message") if isinstance(error, dict) else ""
            return web.json_response({"ephemeral_text": message or "This action is no longer available."})
        return web.json_response(response)

    async def _handle_approval_action_request(self, request: Any) -> Any:
        from aiohttp import web
        try:
            payload = await request.json()
        except Exception:
            return web.json_response({"error": {"message": "Invalid JSON payload."}}, status=400)
        response = await self._handle_approval_action_payload(payload)
        if "error" in response:
            error = response.get("error") or {}
            message = error.get("message") if isinstance(error, dict) else ""
            return web.json_response({"ephemeral_text": message or "This approval is no longer available."})
        return web.json_response(response)

    async def _handle_approval_action_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve a Mattermost dangerous-command approval button payload."""
        from tools.approval import has_blocking_approval, resolve_gateway_approval

        if not isinstance(payload, dict):
            return {"error": {"message": "Invalid Mattermost action payload."}}
        context = payload.get("context") or {}
        if not isinstance(context, dict):
            return {"error": {"message": "Invalid Mattermost action context."}}
        if context.get("kind") != "approval":
            return {"error": {"message": "Unsupported Mattermost action."}}

        token = str(context.get("token") or "")
        if not token:
            return {"error": {"message": "Missing Mattermost action token."}}
        state = self._approval_action_tokens.get(token)
        if not state:
            return {"error": {"message": "This approval is no longer waiting for an answer."}}

        payload_channel_id = str(payload.get("channel_id") or "")
        expected_channel_id = str(state.get("chat_id") or "")
        if expected_channel_id and payload_channel_id and payload_channel_id != expected_channel_id:
            return {"error": {"message": "This approval belongs to a different Mattermost channel."}}

        expected_user_id = str(state.get("expected_user_id") or "")
        payload_user_id = str(payload.get("user_id") or "")
        if expected_user_id and payload_user_id != expected_user_id:
            return {"error": {"message": "Only the user who was asked can approve this command."}}

        session_key = str(state.get("session_key") or "")
        choice = str(state.get("choice") or "")
        if choice not in {"once", "session", "always", "deny"}:
            self._approval_action_tokens.pop(token, None)
            return {"error": {"message": "Invalid approval choice."}}
        if not session_key or not has_blocking_approval(session_key):
            self._clear_approval_action_tokens(session_key)
            return {"error": {"message": "This approval is no longer waiting for an answer."}}

        count = resolve_gateway_approval(session_key, choice)
        self._clear_approval_action_tokens(session_key)
        if not count:
            return {"error": {"message": "This approval is no longer waiting for an answer."}}

        if choice == "deny":
            status = "❌ Denied"
            ephemeral = "Denied. The command will not run."
        elif choice == "session":
            status = "✅ Approved for this session"
            ephemeral = "Approved for this session."
        elif choice == "always":
            status = "✅ Approved permanently"
            ephemeral = "Approved permanently."
        else:
            status = "✅ Approved once"
            ephemeral = "Approved once."
        return {
            "update": {
                "message": f"⚠️ **Dangerous command approval**\n\n{status}",
                "props": {},
            },
            "ephemeral_text": ephemeral,
        }

    async def _handle_clarify_action_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve a Mattermost clarify button action payload."""
        from tools import clarify_gateway as cm

        if not isinstance(payload, dict):
            return {"error": {"message": "Invalid Mattermost action payload."}}
        context = payload.get("context") or {}
        if not isinstance(context, dict):
            return {"error": {"message": "Invalid Mattermost action context."}}
        if context.get("kind") != "clarify":
            return {"error": {"message": "Unsupported Mattermost action."}}

        token = str(context.get("token") or "")
        if not token:
            return {"error": {"message": "Missing Mattermost action token."}}
        state = self._clarify_action_tokens.get(token)
        if not state:
            return {"error": {"message": "This question is no longer waiting for an answer."}}

        payload_channel_id = str(payload.get("channel_id") or "")
        expected_channel_id = str(state.get("chat_id") or "")
        if expected_channel_id and payload_channel_id and payload_channel_id != expected_channel_id:
            return {"error": {"message": "This action belongs to a different Mattermost channel."}}

        expected_user_id = str(state.get("expected_user_id") or "")
        payload_user_id = str(payload.get("user_id") or "")
        if expected_user_id and payload_user_id != expected_user_id:
            return {"error": {"message": "Only the user who was asked can answer this question."}}

        clarify_id = str(state.get("clarify_id") or "")
        if not clarify_id:
            self._clarify_action_tokens.pop(token, None)
            return {"error": {"message": "Missing clarify id."}}

        choice = state.get("choice")
        if choice == "__other__":
            if not cm.mark_awaiting_text(clarify_id):
                self._clear_clarify_action_tokens(clarify_id)
                return {"error": {"message": "This question is no longer waiting for an answer."}}
            question = ""
            with cm._lock:
                entry = cm._entries.get(clarify_id)
                if entry is not None:
                    question = entry.question
            self._clear_clarify_action_tokens(clarify_id)
            return {
                "update": {
                    "message": f"❓ {question}\n\n✍️ Waiting for a typed answer..." if question else "✍️ Waiting for a typed answer...",
                    "props": {},
                },
                "ephemeral_text": "Type your answer as a reply in this thread/channel.",
            }

        # Prefer the canonical choice from the pending clarify entry when an
        # index is present, so button payload tampering cannot substitute text.
        try:
            raw_choice_index = state.get("choice_index")
            choice_index = int(raw_choice_index) if raw_choice_index is not None else -1
        except (TypeError, ValueError):
            choice_index = -1
        with cm._lock:
            entry = cm._entries.get(clarify_id)
            if entry and entry.choices and 0 <= choice_index < len(entry.choices):
                choice = entry.choices[choice_index]
            question = entry.question if entry else ""
        if choice is None:
            self._clarify_action_tokens.pop(token, None)
            return {"error": {"message": "Missing choice."}}

        resolved = cm.resolve_gateway_clarify(clarify_id, str(choice))
        self._clear_clarify_action_tokens(clarify_id)
        if not resolved:
            return {"error": {"message": "This question is no longer waiting for an answer."}}
        selected = str(choice)
        return {
            "update": {
                "message": f"❓ {question}\n\n✅ Selected: {selected}" if question else f"✅ Selected: {selected}",
                "props": {},
            },
            "ephemeral_text": f"Got it: {selected}",
        }

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Return channel name and type."""
        data = await self._api_get(f"channels/{chat_id}")
        if not data:
            return {"name": chat_id, "type": "channel"}

        ch_type = _CHANNEL_TYPE_MAP.get(data.get("type", "O"), "channel")
        display_name = data.get("display_name") or data.get("name") or chat_id
        return {"name": display_name, "type": ch_type}

    # ------------------------------------------------------------------
    # Optional overrides
    # ------------------------------------------------------------------

    async def send_typing(
        self, chat_id: str, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Send a typing indicator."""
        await self._api_post(
            f"users/{self._bot_user_id}/typing",
            {"channel_id": chat_id},
        )

    async def edit_message(
        self, chat_id: str, message_id: str, content: str, *, finalize: bool = False
    ) -> SendResult:
        """Edit an existing post."""
        formatted = self.format_message(content)
        data = await self._api_put(
            f"posts/{message_id}/patch",
            {"message": formatted},
        )
        if not data or "id" not in data:
            return SendResult(success=False, error="Failed to edit post")
        return SendResult(success=True, message_id=data["id"])

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Download an image and upload it as a file attachment."""
        return await self._send_url_as_file(
            chat_id, image_url, caption, reply_to, "image"
        )

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Upload a local image file."""
        return await self._send_local_file(
            chat_id, image_path, caption, reply_to
        )

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Upload a local file as a document."""
        return await self._send_local_file(
            chat_id, file_path, caption, reply_to, file_name
        )

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Upload an audio file."""
        return await self._send_local_file(
            chat_id, audio_path, caption, reply_to
        )

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Upload a video file."""
        return await self._send_local_file(
            chat_id, video_path, caption, reply_to
        )

    def format_message(self, content: str) -> str:
        """Mattermost uses standard Markdown — mostly pass through.

        Strip image markdown into plain links (files are uploaded separately).
        """
        # Convert ![alt](url) to just the URL — Mattermost renders
        # image URLs as inline previews automatically.
        content = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"\2", content)
        return content

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------

    async def _send_url_as_file(
        self,
        chat_id: str,
        url: str,
        caption: Optional[str],
        reply_to: Optional[str],
        kind: str = "file",
    ) -> SendResult:
        """Download a URL and upload it as a file attachment."""
        from tools.url_safety import is_safe_url
        if not is_safe_url(url):
            logger.warning("Mattermost: blocked unsafe URL (SSRF protection)")
            return await self.send(chat_id, f"{caption or ''}\n{url}".strip(), reply_to)

        import aiohttp

        file_data = None
        ct = "application/octet-stream"
        fname = url.rsplit("/", 1)[-1].split("?")[0] or f"{kind}.png"

        for attempt in range(3):
            try:
                async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status >= 500 or resp.status == 429:
                        if attempt < 2:
                            logger.debug("Mattermost download retry %d/2 for %s (status %d)",
                                         attempt + 1, url[:80], resp.status)
                            await asyncio.sleep(1.5 * (attempt + 1))
                            continue
                    if resp.status >= 400:
                        return await self.send(chat_id, f"{caption or ''}\n{url}".strip(), reply_to)
                    file_data = await resp.read()
                    ct = resp.content_type or "application/octet-stream"
                    break
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                logger.warning("Mattermost: failed to download %s after %d attempts: %s", url, attempt + 1, exc)
                return await self.send(chat_id, f"{caption or ''}\n{url}".strip(), reply_to)

        if file_data is None:
            logger.warning("Mattermost: download returned no data for %s", url)
            return await self.send(chat_id, f"{caption or ''}\n{url}".strip(), reply_to)

        file_id = await self._upload_file(chat_id, file_data, fname, ct)
        if not file_id:
            return await self.send(chat_id, f"{caption or ''}\n{url}".strip(), reply_to)

        payload: Dict[str, Any] = {
            "channel_id": chat_id,
            "message": caption or "",
            "file_ids": [file_id],
        }
        if reply_to and self._reply_mode == "thread":
            payload["root_id"] = await self._resolve_root_id(reply_to)

        data = await self._api_post("posts", payload)
        if not data or "id" not in data:
            return SendResult(success=False, error="Failed to post with file")
        return SendResult(success=True, message_id=data["id"])

    async def _send_local_file(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str],
        reply_to: Optional[str],
        file_name: Optional[str] = None,
    ) -> SendResult:
        """Upload a local file and attach it to a post."""
        import mimetypes

        p = Path(file_path)
        if not p.exists():
            logger.warning(
                "Mattermost: local file not found, skipping: %s", file_path
            )
            return SendResult(success=True, message_id=None)

        fname = file_name or p.name
        ct = mimetypes.guess_type(fname)[0] or "application/octet-stream"
        file_data = p.read_bytes()

        file_id = await self._upload_file(chat_id, file_data, fname, ct)
        if not file_id:
            return SendResult(success=False, error="File upload failed")

        payload: Dict[str, Any] = {
            "channel_id": chat_id,
            "message": caption or "",
            "file_ids": [file_id],
        }
        if reply_to and self._reply_mode == "thread":
            payload["root_id"] = await self._resolve_root_id(reply_to)

        data = await self._api_post("posts", payload)
        if not data or "id" not in data:
            return SendResult(success=False, error="Failed to post with file")
        return SendResult(success=True, message_id=data["id"])

    async def send_multiple_images(
        self,
        chat_id: str,
        images: List[Tuple[str, str]],
        metadata: Optional[Dict[str, Any]] = None,
        human_delay: float = 0.0,
    ) -> None:
        """Send a batch of images as a single Mattermost post with multiple attachments.

        Mattermost supports up to 5 ``file_ids`` per post. Each image is
        uploaded individually (Mattermost's file API is one-at-a-time),
        then a single post is created referencing all uploaded file_ids
        at once. Batches larger than 5 are chunked. Falls back to the
        base per-image loop on total failure.
        """
        if not images:
            return

        import mimetypes
        import aiohttp
        from urllib.parse import unquote as _unquote

        CHUNK = 5  # Mattermost post file_ids cap
        chunks = [images[i:i + CHUNK] for i in range(0, len(images), CHUNK)]

        for chunk_idx, chunk in enumerate(chunks):
            if human_delay > 0 and chunk_idx > 0:
                await asyncio.sleep(human_delay)

            file_ids: List[str] = []
            caption_parts: List[str] = []
            try:
                for image_url, alt_text in chunk:
                    if alt_text:
                        caption_parts.append(alt_text)

                    if image_url.startswith("file://"):
                        local_path = _unquote(image_url[7:])
                        p = Path(local_path)
                        if not p.exists():
                            logger.warning("Mattermost: skipping missing image %s", local_path)
                            continue
                        fname = p.name
                        ct = mimetypes.guess_type(fname)[0] or "image/png"
                        file_data = p.read_bytes()
                    else:
                        from tools.url_safety import is_safe_url
                        if not is_safe_url(image_url):
                            logger.warning("Mattermost: blocked unsafe image URL in batch")
                            continue
                        try:
                            async with self._session.get(
                                image_url, timeout=aiohttp.ClientTimeout(total=30)
                            ) as resp:
                                if resp.status >= 400:
                                    logger.warning(
                                        "Mattermost: failed to download image (HTTP %d): %s",
                                        resp.status, image_url[:80],
                                    )
                                    continue
                                file_data = await resp.read()
                                ct = resp.content_type or "image/png"
                        except Exception as dl_err:
                            logger.warning("Mattermost: download failed for %s: %s", image_url[:80], dl_err)
                            continue
                        fname = image_url.rsplit("/", 1)[-1].split("?")[0] or f"image_{len(file_ids)}.png"

                    fid = await self._upload_file(chat_id, file_data, fname, ct)
                    if fid:
                        file_ids.append(fid)

                if not file_ids:
                    continue

                payload: Dict[str, Any] = {
                    "channel_id": chat_id,
                    "message": "\n".join(caption_parts),
                    "file_ids": file_ids,
                }
                logger.info(
                    "Mattermost: sending %d image(s) as single post (chunk %d/%d)",
                    len(file_ids), chunk_idx + 1, len(chunks),
                )
                data = await self._api_post("posts", payload)
                if not data or "id" not in data:
                    logger.warning("Mattermost: multi-image post failed, falling back")
                    await super().send_multiple_images(chat_id, chunk, metadata, human_delay=human_delay)
            except Exception as e:
                logger.warning(
                    "Mattermost: multi-image send failed (chunk %d/%d), falling back: %s",
                    chunk_idx + 1, len(chunks), e, exc_info=True,
                )
                await super().send_multiple_images(chat_id, chunk, metadata, human_delay=human_delay)

    # ------------------------------------------------------------------
    # WebSocket
    # ------------------------------------------------------------------

    async def _ws_loop(self) -> None:
        """Connect to the WebSocket and listen for events, reconnecting on failure."""
        delay = _RECONNECT_BASE_DELAY
        while not self._closing:
            try:
                await self._ws_connect_and_listen()
                # Clean disconnect — reset delay.
                delay = _RECONNECT_BASE_DELAY
            except asyncio.CancelledError:
                return
            except Exception as exc:
                if self._closing:
                    return
                # Detect permanent auth/permission failures that will never
                # succeed on retry — stop reconnecting instead of looping forever.
                import aiohttp
                err_str = str(exc).lower()
                if isinstance(exc, aiohttp.WSServerHandshakeError) and exc.status in {401, 403}:
                    logger.error("Mattermost WS auth failed (HTTP %d) — stopping reconnect", exc.status)
                    return
                if "401" in err_str or "403" in err_str or "unauthorized" in err_str:
                    logger.error("Mattermost WS permanent error: %s — stopping reconnect", exc)
                    return
                logger.warning("Mattermost WS error: %s — reconnecting in %.0fs", exc, delay)

            if self._closing:
                return

            # Exponential backoff with jitter.
            import random
            jitter = delay * _RECONNECT_JITTER * random.random()
            await asyncio.sleep(delay + jitter)
            delay = min(delay * 2, _RECONNECT_MAX_DELAY)

    async def _ws_connect_and_listen(self) -> None:
        """Single WebSocket session: connect, authenticate, process events."""
        # Build WS URL: https:// → wss://, http:// → ws://
        ws_url = re.sub(r"^http", "ws", self._base_url) + "/api/v4/websocket"
        logger.info("Mattermost: connecting to %s", ws_url)

        self._ws = await self._session.ws_connect(ws_url, heartbeat=30.0)

        # Authenticate via the WebSocket.
        auth_msg = {
            "seq": 1,
            "action": "authentication_challenge",
            "data": {"token": self._token},
        }
        await self._ws.send_json(auth_msg)
        logger.info("Mattermost: WebSocket connected and authenticated")

        async for raw_msg in self._ws:
            if self._closing:
                return

            if raw_msg.type in {
                raw_msg.type.TEXT,
                raw_msg.type.BINARY,
            }:
                try:
                    event = json.loads(raw_msg.data)
                except (json.JSONDecodeError, TypeError):
                    continue
                await self._handle_ws_event(event)
            elif raw_msg.type in {
                raw_msg.type.ERROR,
                raw_msg.type.CLOSE,
                raw_msg.type.CLOSING,
                raw_msg.type.CLOSED,
            }:
                logger.info("Mattermost: WebSocket closed (%s)", raw_msg.type)
                break

    async def _handle_ws_event(self, event: Dict[str, Any]) -> None:
        """Process a single WebSocket event."""
        event_type = event.get("event")
        if event_type != "posted":
            return

        data = event.get("data", {})
        raw_post_str = data.get("post")
        if not raw_post_str:
            return

        try:
            post = json.loads(raw_post_str)
        except (json.JSONDecodeError, TypeError):
            return

        # Ignore own messages.
        if post.get("user_id") == self._bot_user_id:
            return

        # Ignore system posts.
        if post.get("type"):
            return

        post_id = post.get("id", "")

        # Dedup.
        if self._dedup.is_duplicate(post_id):
            return

        # Build message event.
        channel_id = post.get("channel_id", "")
        channel_type_raw = data.get("channel_type", "O")
        chat_type = _CHANNEL_TYPE_MAP.get(channel_type_raw, "channel")

        # For DMs, user_id is sufficient.  For channels, check for @mention.
        message_text = post.get("message", "")

        # Mention-gating for non-DM channels.
        # Config (config.yaml `mattermost.*` with env-var fallback):
        #   require_mention / MATTERMOST_REQUIRE_MENTION: Require @mention in channels (default: true)
        #   free_response_channels / MATTERMOST_FREE_RESPONSE_CHANNELS: Channel IDs where bot responds without mention
        #   allowed_channels / MATTERMOST_ALLOWED_CHANNELS: If set, bot ONLY responds in these channels (whitelist)
        if channel_type_raw != "D":
            # allowed_channels check (whitelist — must pass before other gating).
            # When set, messages from channels NOT in this list are silently
            # ignored, even if @mentioned.  DMs are already excluded above.
            allowed_raw = self.config.extra.get("allowed_channels") if self.config.extra else None
            if allowed_raw is None:
                allowed_raw = os.getenv("MATTERMOST_ALLOWED_CHANNELS", "")
            if isinstance(allowed_raw, list):
                allowed_channels = {str(c).strip() for c in allowed_raw if str(c).strip()}
            else:
                allowed_channels = {
                    c.strip() for c in str(allowed_raw).split(",") if c.strip()
                }
            if allowed_channels and channel_id not in allowed_channels:
                logger.debug(
                    "Mattermost: ignoring message in non-allowed channel: %s",
                    channel_id,
                )
                return

            require_mention = os.getenv(
                "MATTERMOST_REQUIRE_MENTION", "true"
            ).lower() not in {"false", "0", "no"}

            free_channels_raw = os.getenv("MATTERMOST_FREE_RESPONSE_CHANNELS", "")
            free_channels = {ch.strip() for ch in free_channels_raw.split(",") if ch.strip()}
            is_free_channel = channel_id in free_channels

            mention_patterns = [
                f"@{self._bot_username}",
                f"@{self._bot_user_id}",
            ]
            has_mention = any(
                pattern.lower() in message_text.lower()
                for pattern in mention_patterns
            )

            if require_mention and not is_free_channel and not has_mention:
                logger.debug(
                    "Mattermost: skipping non-DM message without @mention (channel=%s)",
                    channel_id,
                )
                return

            # Strip @mention from the message text so the agent sees clean input.
            if has_mention:
                for pattern in mention_patterns:
                    message_text = re.sub(
                        re.escape(pattern), "", message_text, flags=re.IGNORECASE
                    ).strip()

        # Resolve sender info.
        sender_id = post.get("user_id", "")
        sender_name = data.get("sender_name", "").lstrip("@") or sender_id

        # Thread support: if the post is in a thread, use root_id.
        thread_id = post.get("root_id") or None

        # Determine message type.
        file_ids = post.get("file_ids") or []
        msg_type = MessageType.TEXT
        if message_text.startswith("/"):
            msg_type = MessageType.COMMAND

        # Download file attachments immediately (URLs require auth headers
        # that downstream tools won't have).
        media_urls: List[str] = []
        media_types: List[str] = []
        for fid in file_ids:
            try:
                file_info = await self._api_get(f"files/{fid}/info")
                fname = file_info.get("name", f"file_{fid}")
                ext = Path(fname).suffix or ""
                mime = file_info.get("mime_type", "application/octet-stream")

                import aiohttp
                dl_url = f"{self._base_url}/api/v4/files/{fid}"
                async with self._session.get(
                    dl_url,
                    headers={"Authorization": f"Bearer {self._token}"},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status < 400:
                        file_data = await resp.read()
                        from gateway.platforms.base import cache_image_from_bytes, cache_document_from_bytes
                        if mime.startswith("image/"):
                            local_path = cache_image_from_bytes(file_data, ext or ".png")
                            media_urls.append(local_path)
                            media_types.append(mime)
                        elif mime.startswith("audio/"):
                            from gateway.platforms.base import cache_audio_from_bytes
                            local_path = cache_audio_from_bytes(file_data, ext or ".ogg")
                            media_urls.append(local_path)
                            media_types.append(mime)
                        else:
                            local_path = cache_document_from_bytes(file_data, fname)
                            media_urls.append(local_path)
                            media_types.append(mime)
                    else:
                        logger.warning("Mattermost: failed to download file %s: HTTP %s", fid, resp.status)
            except Exception as exc:
                logger.warning("Mattermost: error downloading file %s: %s", fid, exc)

        # Set message type based on downloaded media types.
        if media_types and msg_type == MessageType.TEXT:
            if any(m.startswith("image/") for m in media_types):
                msg_type = MessageType.PHOTO
            elif any(m.startswith("audio/") for m in media_types):
                msg_type = MessageType.VOICE
            elif media_types:
                msg_type = MessageType.DOCUMENT

        source = self.build_source(
            chat_id=channel_id,
            chat_type=chat_type,
            user_id=sender_id,
            user_name=sender_name,
            thread_id=thread_id,
        )

        # Per-channel ephemeral prompt
        from gateway.platforms.base import resolve_channel_prompt
        _channel_prompt = resolve_channel_prompt(
            self.config.extra, channel_id, None,
        )

        msg_event = MessageEvent(
            text=message_text,
            message_type=msg_type,
            source=source,
            raw_message=post,
            message_id=post_id,
            media_urls=media_urls if media_urls else None,
            media_types=media_types if media_types else None,
            channel_prompt=_channel_prompt,
        )

        await self.handle_message(msg_event)


