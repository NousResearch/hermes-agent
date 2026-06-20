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
import hmac
import json
import logging
import os
import re
import time
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from gateway.config import Platform, PlatformConfig
from gateway.platforms.helpers import MessageDeduplicator
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    http_request,
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

        # Interaction callback server (Sprint 2)
        self._interaction_app: Any = None
        self._interaction_runner: Any = None
        self._interaction_site: Any = None

        # Interaction config
        self._callback_base_url: str = ""
        self._hmac_secret: str = ""
        self._listen_host: str = "127.0.0.1"
        self._listen_port: int = 8391

        # Inbound approval state: approval_id -> {session_key, chat_id, message_id}
        self._approval_state: Dict[str, dict] = {}
        self._approval_counter: int = 0

        # Slash-confirm state: confirm_id -> {session_key, chat_id, message_id}
        self._slash_confirm_state: Dict[str, dict] = {}

        # User info cache: user_id -> {username, display_name, first_name, last_name}
        self._user_cache: Dict[str, dict] = {}

        self._last_post_status: Optional[int] = None
        self._last_post_error: str = ""

        # Dedup cache (prevent reprocessing)
        self._dedup = MessageDeduplicator()

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
            resp = await http_request(self._session, "get", url, headers=self._headers())
            async with resp:
                if resp.status >= 400:
                    body = await resp.text()
                    logger.error("MM API GET %s → %s: %s", path, resp.status, body[:200])
                    return {}
                return await resp.json()
        except asyncio.TimeoutError:
            logger.error("MM API GET %s timeout", path)
            return {}
        except aiohttp.ClientError as exc:
            logger.error("MM API GET %s network error: %s", path, exc)
            return {}

    async def _api_post(
        self, path: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """POST /api/v4/{path} with JSON body."""
        import aiohttp
        url = f"{self._base_url}/api/v4/{path.lstrip('/')}"
        self._last_post_status = None
        self._last_post_error = ""
        try:
            resp = await http_request(self._session, "post", url, headers=self._headers(), json=payload)
            async with resp:
                self._last_post_status = resp.status
                if resp.status >= 400:
                    body = await resp.text()
                    self._last_post_error = body or ""
                    logger.error("MM API POST %s → %s: %s", path, resp.status, body[:200])
                    return {}
                return await resp.json()
        except asyncio.TimeoutError:
            logger.error("MM API POST %s timeout", path)
            return {}
        except aiohttp.ClientError as exc:
            self._last_post_error = str(exc)
            logger.error("MM API POST %s network error: %s", path, exc)
            return {}

    async def _thread_root_for_send(
        self,
        reply_to: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        """Resolve the Mattermost root_id from reply_to or metadata."""
        if self._reply_mode != "thread":
            return None
        candidate = reply_to
        if not candidate and isinstance(metadata, dict):
            candidate = metadata.get("thread_id") or metadata.get("root_id")
        if not candidate:
            return None
        return await self._resolve_root_id(str(candidate))

    def _last_post_failure_is_broken_thread_root(self) -> bool:
        """Return True only for clear invalid/missing Mattermost thread roots."""
        if self._last_post_status not in {400, 404}:
            return False
        body = (self._last_post_error or "").lower()
        if not body:
            return False
        rootish = any(marker in body for marker in ("root_id", "rootid", "root id", "thread", "post"))
        broken = any(marker in body for marker in ("invalid", "not found", "does not exist", "missing"))
        return rootish and broken

    async def _post_preserving_thread(
        self,
        chat_id: str,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Post once, optionally falling back flat for final notify content."""
        data = await self._api_post("posts", payload)
        if data or "root_id" not in payload:
            return data
        if not (isinstance(metadata, dict) and metadata.get("notify")):
            return data
        if not self._last_post_failure_is_broken_thread_root():
            return data

        flat_payload = dict(payload)
        flat_payload.pop("root_id", None)
        original = str(flat_payload.get("message") or "")
        flat_payload["message"] = (
            "⚠️ Mattermost thread delivery failed; posting final reply in channel.\n\n"
            + original
        ).strip()
        logger.warning(
            "Mattermost: falling back to flat channel delivery for notify-worthy post in %s",
            chat_id,
        )
        return await self._api_post("posts", flat_payload)

    async def _api_put(
        self, path: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """PUT /api/v4/{path} with JSON body."""
        import aiohttp
        url = f"{self._base_url}/api/v4/{path.lstrip('/')}"
        try:
            resp = await http_request(self._session, "put", url, headers=self._headers(), json=payload)
            async with resp:
                if resp.status >= 400:
                    body = await resp.text()
                    logger.error("MM API PUT %s → %s: %s", path, resp.status, body[:200])
                    return {}
                return await resp.json()
        except asyncio.TimeoutError:
            logger.error("MM API PUT %s timeout", path)
            return {}
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
        try:
            resp = await http_request(self._session, "post", url, headers=headers, data=form, timeout=60)
            async with resp:
                if resp.status >= 400:
                    body = await resp.text()
                    logger.error("MM file upload → %s: %s", resp.status, body[:200])
                    return None
                data = await resp.json()
                infos = data.get("file_infos", [])
                return infos[0]["id"] if infos else None
        except (asyncio.TimeoutError, aiohttp.ClientError):
            logger.exception("MM file upload failed")
            return None

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
            timeout=None
        )
        self._closing = False

        # Read interaction config
        interactions_cfg = self.config.extra.get("interactions") or {}
        self._callback_base_url = str(
            interactions_cfg.get("callback_url", "")
        ).rstrip("/")
        self._listen_host = str(interactions_cfg.get("listen_host", "127.0.0.1"))
        self._listen_port = int(interactions_cfg.get("listen_port", 8391))
        self._hmac_secret = (
            os.getenv("MATTERMOST_INTERACTIONS_HMAC_SECRET")
            or str(interactions_cfg.get("hmac_secret", ""))
        )

        # Start the inbound interaction server
        if self._callback_base_url and self._hmac_secret:
            if len(self._hmac_secret.encode("utf-8")) < 32:
                logger.warning(
                    "Mattermost interaction HMAC secret is too short (%d bytes, "
                    "minimum 32). Interactive buttons disabled.",
                    len(self._hmac_secret.encode("utf-8")),
                )
                self._callback_base_url = ""
            else:
                await self._start_interaction_server()

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
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        """Disconnect from Mattermost."""
        self._closing = True

        # Stop the inbound interaction server.
        if self._interaction_site:
            try:
                await self._interaction_site.stop()
            except Exception as exc:
                logger.debug("Mattermost interaction site stop failed: %s", exc)
        if self._interaction_runner:
            try:
                await self._interaction_runner.cleanup()
            except Exception as exc:
                logger.debug("Mattermost interaction runner cleanup failed: %s", exc)
        self._approval_state.clear()
        self._slash_confirm_state.clear()

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
            # Thread support: reply_to or metadata["thread_id"] is the root post ID.
            resolved_root = await self._thread_root_for_send(reply_to, metadata)
            if resolved_root:
                payload["root_id"] = resolved_root

            data = await self._post_preserving_thread(chat_id, payload, metadata)
            if not data or "id" not in data:
                return SendResult(success=False, error="Failed to create post")
            last_id = data["id"]

        return SendResult(success=True, message_id=last_id)

    # ------------------------------------------------------------------
    # User info resolution
    # ------------------------------------------------------------------

    async def _resolve_user_info(self, user_id: str) -> dict:
        """Fetch user info from the Mattermost API, caching the result.

        Returns a dict with keys: username, display_name, first_name,
        last_name, nickname.  Returns an empty dict on failure.
        """
        if not user_id or not self._session:
            return {}
        if user_id in self._user_cache:
            return self._user_cache[user_id]
        data = await self._api_get(f"users/{user_id}")
        if not data or "id" not in data:
            logger.debug("Mattermost: failed to resolve user info for %s", user_id)
            return {}
        info = {
            "username": data.get("username", ""),
            "first_name": data.get("first_name", ""),
            "last_name": data.get("last_name", ""),
            "nickname": data.get("nickname", ""),
        }
        # Build display name: nickname > first_name + last_name > username
        nickname = info["nickname"].strip()
        if nickname:
            info["display_name"] = nickname
        else:
            first = info["first_name"].strip()
            last = info["last_name"].strip()
            if first or last:
                info["display_name"] = f"{first} {last}".strip()
            else:
                info["display_name"] = info["username"]
        self._user_cache[user_id] = info
        return info

    def _get_display_name_for_sender(
        self,
        sender_id: str,
        sender_name: str,
        user_info: dict,
    ) -> str:
        """Return the best available display name for a message sender.

        Priority:
        1. ``display_name`` from the resolved API user info (nickname,
           first+last name, or username).
        2. ``sender_name`` (the ``sender_name`` from the WebSocket event,
           i.e. the Mattermost username).
        3. ``sender_id`` (the raw user ID as a last resort).
        """
        if user_info:
            dn = user_info.get("display_name", "").strip()
            if dn:
                return dn
        if sender_name:
            return sender_name
        return sender_id

    # ------------------------------------------------------------------
    # Interaction token signing
    # ------------------------------------------------------------------

    _INTERACTION_TOKEN_TTL: int = 3600  # 1 hour

    def _sign_interaction_token(self, payload: str) -> str:
        """Sign *payload* with HMAC-SHA256 using ``_hmac_secret``."""
        secret_bytes = self._hmac_secret.encode("utf-8")
        return hmac.new(
            secret_bytes,
            payload.encode("utf-8"),
            sha256,
        ).hexdigest()

    def _make_interaction_token(self, kind: str, ref_id: str, choice: str) -> str:
        """Produce a signed interaction token.

        Format: <sig>.<rand_hex>.<ts>
        """
        rand_hex = os.urandom(8).hex()
        ts = str(int(time.time()))
        canonical = f"{kind}:{ref_id}:{choice}:{rand_hex}:{ts}"
        sig = self._sign_interaction_token(canonical)
        return f"{sig}.{rand_hex}.{ts}"

    def _verify_interaction_token(
        self,
        kind: str,
        ref_id: str,
        choice: str,
        token: str,
    ) -> bool:
        """Validate an interaction token.

        Returns True only when all checks pass.
        """
        parts = token.rsplit(".", 2)
        if len(parts) != 3:
            return False

        sig, rand_hex, ts_str = parts

        # Validate hex component
        try:
            int(rand_hex, 16)
        except ValueError:
            return False

        # TTL check
        try:
            token_ts = int(ts_str)
        except ValueError:
            return False
        if time.time() - token_ts > self._INTERACTION_TOKEN_TTL:
            return False

        # Constant-time HMAC compare
        canonical = f"{kind}:{ref_id}:{choice}:{rand_hex}:{ts_str}"
        expected_sig = self._sign_interaction_token(canonical)
        return hmac.compare_digest(expected_sig, sig)

    # ------------------------------------------------------------------
    # Interaction HTTP server
    # ------------------------------------------------------------------

    def _is_interactive_user_authorized(
        self,
        user_id: str,
        *,
        channel_id: str = "",
    ) -> bool:
        """Return whether a Mattermost user may click approval buttons."""
        if not user_id:
            return False
        if os.getenv("MATTERMOST_ALLOW_ALL_USERS", "").lower() in (
            "true", "1", "yes",
        ):
            return True
        allowed_raw = os.getenv("MATTERMOST_ALLOWED_USERS", "")
        if not allowed_raw:
            return True
        allowed = {u.strip() for u in allowed_raw.split(",") if u.strip()}
        return user_id in allowed

    async def _start_interaction_server(self) -> None:
        """Start the inbound interaction server."""
        from aiohttp import web

        self._interaction_app = web.Application(client_max_size=65536)
        self._interaction_app.router.add_post(
            "/mattermost/interactions", self._handle_interaction_request
        )
        self._interaction_runner = web.AppRunner(self._interaction_app)
        await self._interaction_runner.setup()
        self._interaction_site = web.TCPSite(
            self._interaction_runner, self._listen_host, self._listen_port
        )
        try:
            await self._interaction_site.start()
            logger.info(
                "Mattermost: interaction server started on %s:%d",
                self._listen_host, self._listen_port,
            )
        except OSError as exc:
            logger.warning(
                "Mattermost interaction server failed to start on %s:%d (%s). "
                "Falling back to text-based /approve.",
                self._listen_host, self._listen_port, exc,
            )
            await self._interaction_runner.cleanup()
            self._interaction_app = None
            self._interaction_runner = None
            self._interaction_site = None
            self._callback_base_url = ""

    async def _handle_interaction_request(
        self, request: Any,
    ) -> "web.Response":
        """Handle a button-click callback from the Mattermost server."""
        from aiohttp import web as _web

        raw = await request.read()
        if len(raw) > 65536:
            return _web.Response(status=413)

        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return _web.Response(status=400)

        if not isinstance(payload, dict):
            return _web.Response(status=400)

        context = (
            (payload.get("data") or {}).get("context")
            or payload.get("context")
            or {}
        )
        user_id = payload.get("user_id", "")
        channel_id = payload.get("channel_id", "")

        kind = context.get("kind", "")
        ref_id = context.get("approval_id") or context.get("confirm_id") or ""
        choice = context.get("choice", "")
        token = context.get("token", "")

        if not self._verify_interaction_token(
            kind, ref_id, choice, token
        ):
            logger.warning(
                "Mattermost interaction: invalid or expired token "
                "(kind=%s, ref_id=%s)", kind, ref_id,
            )
            return _web.json_response(
                {"text": "Request expired or invalid. Please resend the prompt."},
                status=200,
            )

        if not self._is_interactive_user_authorized(user_id):
            logger.warning(
                "Mattermost interaction: unauthorized user %s for kind %s",
                user_id, kind,
            )
            return _web.json_response(
                {"text": "You are not authorized to approve commands."},
                status=200,
            )

        if kind == "approval":
            return await self._resolve_approval_interaction(
                ref_id, choice, user_id, channel_id,
            )
        elif kind == "confirm":
            return await self._resolve_slash_confirm_interaction(
                ref_id, choice, user_id, channel_id,
            )

        return _web.Response(status=200)

    async def _resolve_approval_interaction(
        self,
        approval_id: str,
        choice: str,
        user_id: str,
        channel_id: str,
    ) -> "web.Response":
        """Resolve a pending approval from a button click."""
        from aiohttp import web as _web

        state = self._approval_state.pop(approval_id, None)
        if not state:
            logger.info(
                "Mattermost: approval %s already resolved or unknown",
                approval_id,
            )
            return _web.json_response(
                {"text": "This approval has already been resolved."},
                status=200,
            )

        if state.get("chat_id") and channel_id \
                and state["chat_id"] != channel_id:
            logger.warning(
                "Mattermost: approval %s channel mismatch (expected=%s, got=%s)",
                approval_id, state["chat_id"], channel_id,
            )
            self._approval_state[approval_id] = state
            return _web.json_response(
                {"text": "Channel mismatch — approval not resolved."},
                status=200,
            )

        requester_user_id = state.get("requester_user_id", "") or ""
        if requester_user_id and user_id and requester_user_id != user_id:
            logger.warning(
                "Mattermost: approval %s user mismatch", approval_id,
            )
            self._approval_state[approval_id] = state
            return _web.json_response(
                {
                    "text": (
                        "Only the user who requested this command can approve "
                        "or deny it via the buttons. Use /approve or /deny in "
                        "the chat instead."
                    ),
                },
                status=200,
            )

        try:
            from tools.approval import resolve_gateway_approval
            count = resolve_gateway_approval(state["session_key"], choice)
            logger.info(
                "Mattermost button resolved %d approval(s) for session %s "
                "(choice=%s, user=%s)",
                count, state["session_key"], choice, user_id,
            )
        except Exception as exc:
            logger.error("Mattermost: resolve_gateway_approval failed: %s", exc)
            return _web.json_response(
                {"text": "Internal error resolving approval. Please try /approve in text."},
                status=200,
            )

        label = {
            "once": "Allowed once",
            "session": "Allowed for session",
            "always": "Allowed permanently",
            "deny": "Denied",
        }
        return _web.json_response(
            {
                "update": {
                    "message": (
                        f"{'✅' if choice != 'deny' else '❌'} {label.get(choice, choice)}"
                    ),
                },
            },
            status=200,
        )

    async def _resolve_slash_confirm_interaction(
        self,
        confirm_id: str,
        choice: str,
        user_id: str,
        channel_id: str,
    ) -> "web.Response":
        """Resolve a pending slash-command confirmation from a button click."""
        from aiohttp import web as _web

        state = self._slash_confirm_state.pop(confirm_id, None)
        if not state:
            return _web.json_response(
                {"text": "This confirmation has already been resolved."},
                status=200,
            )

        try:
            from tools.slash_confirm import resolve as resolve_slash_confirm
            result = await resolve_slash_confirm(
                state["session_key"], confirm_id, choice,
            )
            logger.info(
                "Mattermost slash-confirm resolved for session %s (choice=%s, user=%s)",
                state["session_key"], choice, user_id,
            )
        except Exception as exc:
            logger.error("Mattermost: resolve_slash_confirm failed: %s", exc)
            return _web.json_response(
                {"text": "Internal error resolving confirmation."},
                status=200,
            )

        label = {
            "once": "Approved once",
            "always": "Always approved",
            "cancel": "Cancelled",
        }
        msg = f"{'✅' if choice != 'cancel' else '❌'} {label.get(choice, choice)}"
        if result:
            msg += f"\n{result}"
        return _web.json_response(
            {"update": {"message": msg}},
            status=200,
        )

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
            chat_id, image_url, caption, reply_to, "image", metadata
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
            chat_id, image_path, caption, reply_to, metadata=metadata
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
            chat_id, file_path, caption, reply_to, file_name, metadata
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
            chat_id, audio_path, caption, reply_to, metadata=metadata
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
            chat_id, video_path, caption, reply_to, metadata=metadata
        )

    # ------------------------------------------------------------------
    # Interactive approval / slash-confirm buttons
    # ------------------------------------------------------------------

    async def send_exec_approval(
        self,
        chat_id: str,
        command: str,
        session_key: str,
        description: str = "dangerous command",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an interactive button approval prompt via Mattermost Post Actions."""
        if not self._callback_base_url or not self._hmac_secret:
            return SendResult(
                success=False,
                error="Interaction server not configured",
            )

        self._approval_counter += 1
        approval_id = sha256(
            f"{session_key}:{self._approval_counter}".encode()
        ).hexdigest()[:12]

        cmd_preview = (
            command[:MAX_POST_LENGTH] + "..."
            if len(command) > MAX_POST_LENGTH
            else command
        )
        callback_url = f"{self._callback_base_url}/mattermost/interactions"

        def _button(name: str, choice: str, style: str) -> dict:
            action_id = f"appr{approval_id}{choice}"
            token = self._make_interaction_token("approval", approval_id, choice)
            return {
                "id": action_id,
                "name": name,
                "type": "button",
                "style": style,
                "integration": {
                    "url": callback_url,
                    "context": {
                        "kind": "approval",
                        "approval_id": approval_id,
                        "choice": choice,
                        "token": token,
                    },
                },
            }

        payload: Dict[str, Any] = {
            "channel_id": chat_id,
            "props": {
                "attachments": [
                    {
                        "pretext": "⚠️ **Command Approval Required**",
                        "text": f"```\n{cmd_preview}\n```\n**Reason:** {description}",
                        "actions": [
                            _button("✅ Allow Once", "once", "primary"),
                            _button("✅ Allow Session", "session", "default"),
                            _button("✅ Always Allow", "always", "default"),
                            _button("❌ Deny", "deny", "danger"),
                        ],
                    }
                ]
            },
        }

        meta = metadata or {}
        reply_to_mid = meta.get("reply_to_message_id") if meta else None
        if self._reply_mode == "thread" and reply_to_mid:
            root_id = await self._resolve_root_id(reply_to_mid)
            if root_id:
                payload["root_id"] = root_id

        data = await self._api_post("posts", payload)
        if not data or "id" not in data:
            return SendResult(
                success=False,
                error="Failed to post approval prompt",
            )

        self._approval_state[approval_id] = {
            "session_key": session_key,
            "chat_id": chat_id,
            "message_id": data["id"],
            "requester_user_id": (metadata or {}).get("requester_user_id", ""),
        }

        return SendResult(success=True, message_id=data["id"])

    async def send_slash_confirm(
        self,
        chat_id: str,
        title: str,
        message: str,
        session_key: str,
        confirm_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a 3-button slash-command confirmation prompt."""
        if not self._callback_base_url or not self._hmac_secret:
            return SendResult(
                success=False,
                error="Interaction server not configured",
            )

        callback_url = f"{self._callback_base_url}/mattermost/interactions"

        def _button(name: str, choice: str, style: str) -> dict:
            action_id = f"sc{confirm_id}{choice}"
            token = self._make_interaction_token("confirm", confirm_id, choice)
            return {
                "id": action_id,
                "name": name,
                "type": "button",
                "style": style,
                "integration": {
                    "url": callback_url,
                    "context": {
                        "kind": "confirm",
                        "confirm_id": confirm_id,
                        "choice": choice,
                        "token": token,
                    },
                },
            }

        payload: Dict[str, Any] = {
            "channel_id": chat_id,
            "message": f"⚠️ {title}\n\n{message}",
            "props": {
                "attachments": [
                    {
                        "actions": [
                            _button("✅ Approve Once", "once", "primary"),
                            _button("✅ Always Approve", "always", "default"),
                            _button("❌ Cancel", "cancel", "danger"),
                        ],
                    }
                ]
            },
        }

        meta = metadata or {}
        reply_to_mid = meta.get("reply_to_message_id") if meta else None
        if self._reply_mode == "thread" and reply_to_mid:
            root_id = await self._resolve_root_id(reply_to_mid)
            if root_id:
                payload["root_id"] = root_id

        data = await self._api_post("posts", payload)
        if not data or "id" not in data:
            return SendResult(
                success=False,
                error="Failed to post slash-confirm prompt",
            )

        self._slash_confirm_state[confirm_id] = {
            "session_key": session_key,
            "chat_id": chat_id,
            "message_id": data["id"],
        }

        return SendResult(success=True, message_id=data["id"])

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
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Download a URL and upload it as a file attachment."""
        from tools.url_safety import is_safe_url
        if not is_safe_url(url):
            logger.warning("Mattermost: blocked unsafe URL (SSRF protection)")
            return await self.send(chat_id, f"{caption or ''}\n{url}".strip(), reply_to, metadata=metadata)

        import aiohttp

        file_data = None
        ct = "application/octet-stream"
        fname = url.rsplit("/", 1)[-1].split("?")[0] or f"{kind}.png"

        for attempt in range(3):
            try:
                resp = await http_request(self._session, "get", url)
                async with resp:
                    if resp.status >= 500 or resp.status == 429:
                        if attempt < 2:
                            logger.debug("Mattermost download retry %d/2 for %s (status %d)",
                                         attempt + 1, url[:80], resp.status)
                            await asyncio.sleep(1.5 * (attempt + 1))
                            continue
                        return await self.send(chat_id, f"{caption or ''}\n{url}".strip(), reply_to)
                    if resp.status >= 400:
                        return await self.send(chat_id, f"{caption or ''}\n{url}".strip(), reply_to, metadata=metadata)
                    file_data = await resp.read()
                    ct = resp.content_type or "application/octet-stream"
                    break
            except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
                if attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                logger.warning("Mattermost: failed to download %s after %d attempts: %s", url, attempt + 1, exc)
                return await self.send(chat_id, f"{caption or ''}\n{url}".strip(), reply_to, metadata=metadata)

        if file_data is None:
            logger.warning("Mattermost: download returned no data for %s", url)
            return await self.send(chat_id, f"{caption or ''}\n{url}".strip(), reply_to, metadata=metadata)

        file_id = await self._upload_file(chat_id, file_data, fname, ct)
        if not file_id:
            return await self.send(chat_id, f"{caption or ''}\n{url}".strip(), reply_to, metadata=metadata)

        payload: Dict[str, Any] = {
            "channel_id": chat_id,
            "message": caption or "",
            "file_ids": [file_id],
        }
        resolved_root = await self._thread_root_for_send(reply_to, metadata)
        if resolved_root:
            payload["root_id"] = resolved_root

        data = await self._post_preserving_thread(chat_id, payload, metadata)
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
        metadata: Optional[Dict[str, Any]] = None,
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
        resolved_root = await self._thread_root_for_send(reply_to, metadata)
        if resolved_root:
            payload["root_id"] = resolved_root

        data = await self._post_preserving_thread(chat_id, payload, metadata)
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
                            resp = await http_request(self._session, "get", image_url)
                            async with resp:
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
                resolved_root = await self._thread_root_for_send(None, metadata)
                if resolved_root:
                    payload["root_id"] = resolved_root
                logger.info(
                    "Mattermost: sending %d image(s) as single post (chunk %d/%d)",
                    len(file_ids), chunk_idx + 1, len(chunks),
                )
                data = await self._post_preserving_thread(chat_id, payload, metadata)
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
        sender_name = data.get("sender_name", "").lstrip("@")

        # When the WebSocket event includes sender_name (or it's empty),
        # resolve the sender's display name from the API.  This ensures
        # the user's display name (nickname / first+last / username) is
        # preferred over the raw username when available.
        user_info = {}
        if sender_id:
            user_info = await self._resolve_user_info(sender_id)
            sender_name = self._get_display_name_for_sender(
                sender_id, sender_name, user_info,
            )
        else:
            sender_name = sender_name or ""

        # Thread support: if the post is in a thread, use root_id. In
        # thread mode, top-level channel posts are valid roots for progress.
        thread_id = post.get("root_id") or None
        if (
            not thread_id
            and self._reply_mode == "thread"
            and channel_type_raw != "D"
            and post_id
        ):
            thread_id = post_id

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
                resp = await http_request(
                    self._session, "get", dl_url,
                    headers={"Authorization": f"Bearer {self._token}"},
                )
                async with resp:
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
            message_id=post_id,
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




# ---------------------------------------------------------------------------
# Plugin standalone-send (out-of-process cron delivery via Mattermost REST)
# ---------------------------------------------------------------------------


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[list] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    """Send via the Mattermost v4 REST API without a live gateway adapter.

    Used by ``tools/send_message_tool._send_via_adapter`` when the gateway
    runner is not in this process (typical for cron jobs running out-of-process).
    Reads ``MATTERMOST_TOKEN`` from ``pconfig.token`` (set by the gateway
    config loader from env) and falls back to the ``MATTERMOST_TOKEN`` env
    var.  Server URL comes from ``pconfig.extra["url"]`` (set by the YAML
    bridge / env loader) or the ``MATTERMOST_URL`` env var.

    Thread replies (Mattermost CRT) are supported via the ``root_id`` field
    on the ``POST /posts`` payload — pass ``thread_id`` when threading is
    desired.  ``media_files`` are uploaded via ``POST /files``
    (multipart/form-data), then their returned ``file_id`` values are
    attached to the post.

    ``force_document`` is accepted for signature parity with other
    standalone senders but unused — Mattermost stores every uploaded file
    as a generic attachment regardless.
    """
    try:
        import aiohttp
    except ImportError:
        return {"error": "aiohttp not installed. Run: pip install aiohttp"}

    base_url = (
        (getattr(pconfig, "extra", {}) or {}).get("url")
        or os.getenv("MATTERMOST_URL", "")
    ).rstrip("/")
    token = (getattr(pconfig, "token", None) or os.getenv("MATTERMOST_TOKEN", "")).strip()
    if not base_url or not token:
        return {
            "error": (
                "Mattermost standalone send: MATTERMOST_URL and "
                "MATTERMOST_TOKEN must both be set"
            )
        }

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    upload_headers = {"Authorization": f"Bearer {token}"}

    media_files = media_files or []

    try:
        # Resolve proxy + session kwargs once so a single ClientSession can
        # cover the optional file uploads + final post.
        from gateway.platforms.base import resolve_proxy_url, proxy_kwargs_for_aiohttp
        _proxy = resolve_proxy_url(platform_env_var="MATTERMOST_PROXY")
        _sess_kw, _req_kw = proxy_kwargs_for_aiohttp(_proxy)

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
            **_sess_kw,
        ) as session:
            # 1. Upload media (if any) and collect file_ids.
            file_ids: List[str] = []
            for media in media_files:
                file_path = media.get("path") if isinstance(media, dict) else media
                if not file_path or not os.path.exists(file_path):
                    continue
                form = aiohttp.FormData()
                # Mattermost requires channel_id on file uploads so the
                # server can attribute them.
                form.add_field("channel_id", chat_id)
                with open(file_path, "rb") as fh:
                    form.add_field(
                        "files",
                        fh.read(),
                        filename=os.path.basename(file_path),
                    )
                async with session.post(
                    f"{base_url}/api/v4/files",
                    data=form,
                    headers=upload_headers,
                    **_req_kw,
                ) as upload_resp:
                    if upload_resp.status not in {200, 201}:
                        body = await upload_resp.text()
                        return {
                            "error": (
                                f"Mattermost file upload failed "
                                f"({upload_resp.status}): {body[:400]}"
                            )
                        }
                    upload_data = await upload_resp.json()
                    for info in upload_data.get("file_infos", []):
                        if info.get("id"):
                            file_ids.append(info["id"])

            # 2. Post the message (with thread root + attached file_ids).
            payload: Dict[str, Any] = {
                "channel_id": chat_id,
                "message": message,
            }
            if thread_id:
                payload["root_id"] = thread_id
            if file_ids:
                payload["file_ids"] = file_ids
            async with session.post(
                f"{base_url}/api/v4/posts",
                headers=headers,
                json=payload,
                **_req_kw,
            ) as resp:
                if resp.status not in {200, 201}:
                    body = await resp.text()
                    return {
                        "error": (
                            f"Mattermost API error ({resp.status}): "
                            f"{body[:400]}"
                        )
                    }
                data = await resp.json()
            return {
                "success": True,
                "platform": "mattermost",
                "chat_id": chat_id,
                "message_id": data.get("id"),
            }
    except aiohttp.ClientError as exc:
        return {"error": f"Mattermost send failed (network): {exc}"}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Mattermost send failed: {exc}"}


# ---------------------------------------------------------------------------
# Interactive setup wizard
# ---------------------------------------------------------------------------


def interactive_setup() -> None:
    """Guide the user through Mattermost bot setup.

    Mirrors Discord/Teams' ``interactive_setup`` shape: lazy-imports CLI
    helpers so the plugin's import surface stays small, prompts for the
    server URL + bot token, captures an allowlist, and offers to set a
    home channel.  Replaces the central
    ``hermes_cli/setup.py::_setup_mattermost`` function this migration
    removes.
    """
    from hermes_cli.config import get_env_value, save_env_value
    from hermes_cli.cli_output import (
        prompt,
        prompt_yes_no,
        print_header,
        print_info,
        print_success,
    )

    print_header("Mattermost")
    existing = get_env_value("MATTERMOST_TOKEN")
    if existing:
        print_info("Mattermost: already configured")
        if not prompt_yes_no("Reconfigure Mattermost?", False):
            return

    print_info("Works with any self-hosted Mattermost instance.")
    print_info("   1. In Mattermost: Integrations → Bot Accounts → Add Bot Account")
    print_info("   2. Copy the bot token")
    print()
    mm_url = prompt("Mattermost server URL (e.g. https://mm.example.com)")
    if mm_url:
        save_env_value("MATTERMOST_URL", mm_url.rstrip("/"))
    token = prompt("Bot token", password=True)
    if not token:
        return
    save_env_value("MATTERMOST_TOKEN", token)
    print_success("Mattermost token saved")

    print()
    print_info("🔒 Security: Restrict who can use your bot")
    print_info("   To find your user ID: click your avatar → Profile")
    print_info("   or use the API: GET /api/v4/users/me")
    print()
    allowed_users = prompt("Allowed user IDs (comma-separated, leave empty for open access)")
    if allowed_users:
        save_env_value("MATTERMOST_ALLOWED_USERS", allowed_users.replace(" ", ""))
        print_success("Mattermost allowlist configured")
    else:
        print_info("⚠️  No allowlist set - anyone who can message the bot can use it!")

    print()
    print_info("📬 Home Channel: where Hermes delivers cron job results and notifications.")
    print_info("   To get a channel ID: click channel name → View Info → copy the ID")
    print_info("   You can also set this later by typing /set-home in a Mattermost channel.")
    home_channel = prompt("Home channel ID (leave empty to set later with /set-home)")
    if home_channel:
        save_env_value("MATTERMOST_HOME_CHANNEL", home_channel)

    # ------------------------------------------------------------------
    # Interactive Approvals (optional — button-based /approve)
    # ------------------------------------------------------------------

    print()
    print_header("Interactive Approvals — (optional — button-based /approve)")
    print_info("Requires a URL that your Mattermost server can reach.")
    print_info(
        "   For self-hosted Mattermost behind a firewall, add the callback host"
    )
    print_info(
        "   to ServiceSettings.AllowedUntrustedInternalConnections in config.xml."
    )
    if not prompt_yes_no("Enable interactive approval buttons?", False):
        print_info("Text-based /approve will be used instead.")
        return

    callback_url = prompt(
        "Interaction callback URL (e.g. https://gateway.example.com:8391)"
    )
    if not callback_url:
        print_info("No callback URL set — text-based /approve will be used.")
        _no_url_port = prompt("Listen port [8391]", default="8391")
        try:
            _no_url_port_int = int(_no_url_port or "8391")
        except ValueError:
            print_info(f"Invalid port '{_no_url_port}', using 8391")
            _no_url_port_int = 8391
        _write_interactions_config(
            "",
            prompt("Listen host [127.0.0.1]", default="127.0.0.1"),
            _no_url_port_int,
            "",
        )
        return

    listen_host = prompt("Listen host", default="127.0.0.1")
    if not listen_host:
        listen_host = "127.0.0.1"
    listen_port_str = prompt("Listen port", default="8391")
    try:
        listen_port = int(listen_port_str or "8391")
    except ValueError:
        print_info(f"Invalid port '{listen_port_str}', using 8391")
        listen_port = 8391

    hmac_secret = prompt("HMAC secret (openssl rand -hex 32)", password=True)
    if not hmac_secret or len(hmac_secret) < 32:
        print_info(
            "⚠️  HMAC secret must be ≥32 bytes. Skipping interactive buttons —"
        )
        print_info("   text-based /approve will still work.")

    _write_interactions_config(callback_url, listen_host, listen_port, hmac_secret)


def _write_interactions_config(
    callback_url: str,
    listen_host: str,
    listen_port: int,
    hmac_secret: str,
) -> None:
    """Write interaction config to .env (secret) and config.yaml (operational)."""
    from hermes_cli.config import save_env_value
    from hermes_cli.cli_output import print_success
    from cli import save_config_value

    if hmac_secret and len(hmac_secret) >= 32:
        save_env_value("MATTERMOST_INTERACTIONS_HMAC_SECRET", hmac_secret)
        print_success("HMAC secret saved to .env")

    if callback_url or listen_host != "127.0.0.1" or listen_port != 8391:
        save_config_value(
            "mattermost.interactions.callback_url",
            callback_url.rstrip("/") if isinstance(callback_url, str) else "",
        )
        save_config_value(
            "mattermost.interactions.listen_host", listen_host
        )
        save_config_value(
            "mattermost.interactions.listen_port", listen_port
        )
        print_success(
            f"Interaction config saved: {callback_url} on {listen_host}:{listen_port}",
        )


# ---------------------------------------------------------------------------
# YAML → env config bridge (apply_yaml_config_fn, #25443)
# ---------------------------------------------------------------------------


def _apply_yaml_config(yaml_cfg: dict, mattermost_cfg: dict) -> dict | None:
    """Translate ``config.yaml`` ``mattermost:`` keys into env vars.

    Implements the ``apply_yaml_config_fn`` contract (#24836 / #25443).
    Mirrors the legacy ``mattermost_cfg`` block that used to live in
    ``gateway/config.py::load_gateway_config()`` before this migration.

    The MattermostAdapter reads its runtime configuration via
    ``os.getenv()`` for ``MATTERMOST_REQUIRE_MENTION``,
    ``MATTERMOST_FREE_RESPONSE_CHANNELS``, and
    ``MATTERMOST_ALLOWED_CHANNELS``.  Rather than rewrite those call sites
    to read from ``PlatformConfig.extra``, this hook keeps the env-driven
    model and merely owns the YAML→env translation here, next to the
    adapter that consumes it.

    Env vars take precedence over YAML — every assignment is guarded
    by ``not os.getenv(...)`` so an explicit env var survives a config.yaml
    update.  Returns ``None`` because no extras are seeded into
    ``PlatformConfig.extra`` directly (everything flows through env).
    """
    if "require_mention" in mattermost_cfg and not os.getenv("MATTERMOST_REQUIRE_MENTION"):
        os.environ["MATTERMOST_REQUIRE_MENTION"] = str(mattermost_cfg["require_mention"]).lower()
    frc = mattermost_cfg.get("free_response_channels")
    if frc is not None and not os.getenv("MATTERMOST_FREE_RESPONSE_CHANNELS"):
        if isinstance(frc, list):
            frc = ",".join(str(v) for v in frc)
        os.environ["MATTERMOST_FREE_RESPONSE_CHANNELS"] = str(frc)
    # allowed_channels: if set, bot ONLY responds in these channels (whitelist)
    ac = mattermost_cfg.get("allowed_channels")
    if ac is not None and not os.getenv("MATTERMOST_ALLOWED_CHANNELS"):
        if isinstance(ac, list):
            ac = ",".join(str(v) for v in ac)
        os.environ["MATTERMOST_ALLOWED_CHANNELS"] = str(ac)

    # Interactive approvals — HMAC secret bridge + extras return.
    interactions = mattermost_cfg.get("interactions")
    if interactions and not os.getenv("MATTERMOST_INTERACTIONS_HMAC_SECRET"):
        hmac_secret = interactions.get("hmac_secret", "")
        if hmac_secret:
            os.environ["MATTERMOST_INTERACTIONS_HMAC_SECRET"] = str(hmac_secret)

    return {"interactions": interactions} if interactions else None


# ---------------------------------------------------------------------------
# is_connected probe
# ---------------------------------------------------------------------------


def _is_connected(config) -> bool:
    """Mattermost is considered connected when BOTH MATTERMOST_TOKEN and
    MATTERMOST_URL are set.

    Looks up via ``hermes_cli.gateway.get_env_value`` at call time (not via
    the plugin's own bound import) so tests that patch
    ``gateway_mod.get_env_value`` can suppress ambient env vars.  Matches
    what the legacy connected-platforms check did before this migration.
    """
    import hermes_cli.gateway as gateway_mod
    return bool(
        (gateway_mod.get_env_value("MATTERMOST_TOKEN") or "").strip()
        and (gateway_mod.get_env_value("MATTERMOST_URL") or "").strip()
    )


# ---------------------------------------------------------------------------
# Plugin registration entry point
# ---------------------------------------------------------------------------


def _build_adapter(config):
    """Factory wrapper that constructs MattermostAdapter from a PlatformConfig."""
    return MattermostAdapter(config)


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="mattermost",
        label="Mattermost",
        adapter_factory=_build_adapter,
        check_fn=check_mattermost_requirements,
        is_connected=_is_connected,
        required_env=["MATTERMOST_URL", "MATTERMOST_TOKEN"],
        install_hint="pip install aiohttp",
        # Interactive setup wizard — replaces the central
        # hermes_cli/setup.py::_setup_mattermost function.
        setup_fn=interactive_setup,
        # YAML→env config bridge — owns the translation of
        # ``config.yaml`` ``mattermost:`` keys (require_mention,
        # free_response_channels, allowed_channels) into ``MATTERMOST_*``
        # env vars that the adapter reads via ``os.getenv()``.  Replaces
        # the hardcoded block that used to live in ``gateway/config.py``.
        # Hook contract: #24836 / #25443.
        apply_yaml_config_fn=_apply_yaml_config,
        # Auth env vars for _is_user_authorized() integration.
        allowed_users_env="MATTERMOST_ALLOWED_USERS",
        allow_all_env="MATTERMOST_ALLOW_ALL_USERS",
        # Cron home-channel delivery.
        cron_deliver_env_var="MATTERMOST_HOME_CHANNEL",
        # Out-of-process cron delivery via Mattermost REST API.  Without
        # this hook, ``deliver=mattermost`` cron jobs fail with "No live
        # adapter" when cron runs separately from the gateway.  Mirrors
        # the Discord / Teams pattern.
        standalone_sender_fn=_standalone_send,
        # Mattermost practical post-length limit (server default is 16383
        # but 4000 is the readable threshold the adapter has used since
        # day one).
        max_message_length=MAX_POST_LENGTH,
        # Display
        emoji="💬",
        allow_update_command=True,
    )
