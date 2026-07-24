"""Mattermost gateway adapter.

Connects to a self-hosted (or cloud) Mattermost instance via its REST API
(v4) and WebSocket for real-time events.  No external Mattermost library
required — uses aiohttp which is already a Hermes dependency.

Environment variables:
    MATTERMOST_URL              Server URL (e.g. https://mm.example.com)
    MATTERMOST_TOKEN            Bot token or personal-access token
    MATTERMOST_ALLOWED_USERS    Comma-separated user IDs
    MATTERMOST_HOME_CHANNEL     Channel ID for cron/notification delivery
    MATTERMOST_OBSERVE_UNMENTIONED_CHANNEL_MESSAGES
                                Observe authorized unmentioned posts in allowlisted channels
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import hmac
import ipaddress
import json
import logging
import os
import re
import secrets
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

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

_MATTERMOST_DISABLE_MENTIONS_PROPS = {"disable_mentions": True}

# Reconnect parameters (exponential backoff).
_RECONNECT_BASE_DELAY = 2.0
_RECONNECT_MAX_DELAY = 60.0
_RECONNECT_JITTER = 0.2


def _with_mentions_disabled(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a post payload that prevents Mattermost from firing mentions."""
    props = payload.get("props")
    if isinstance(props, dict):
        payload["props"] = {**props, **_MATTERMOST_DISABLE_MENTIONS_PROPS}
    else:
        payload["props"] = dict(_MATTERMOST_DISABLE_MENTIONS_PROPS)
    return payload


def check_mattermost_requirements() -> bool:
    """Return True if the Mattermost adapter runtime dependency is available."""
    try:
        import aiohttp  # noqa: F401
        return True
    except ImportError:
        logger.warning("Mattermost: aiohttp not installed")
        return False


def validate_mattermost_config(config: PlatformConfig) -> bool:
    """Return True when Mattermost has enough config to connect."""
    extra = getattr(config, "extra", {}) or {}
    token = (getattr(config, "token", None) or os.getenv("MATTERMOST_TOKEN", "")).strip()
    url = (extra.get("url", "") or os.getenv("MATTERMOST_URL", "")).strip()
    if not token:
        logger.debug("Mattermost: MATTERMOST_TOKEN not set")
        return False
    if not url:
        logger.warning("Mattermost: MATTERMOST_URL not set")
        return False
    return True


class MattermostAdapter(BasePlatformAdapter):
    """Gateway adapter for Mattermost (self-hosted or cloud)."""

    splits_long_messages = True  # send() chunks via truncate_message(MAX_POST_LENGTH)

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.MATTERMOST)

        self._base_url: str = (
            config.extra.get("url", "")
            or os.getenv("MATTERMOST_URL", "")
        ).rstrip("/")
        self._token: str = config.token or os.getenv("MATTERMOST_TOKEN", "")

        self._bot_user_id: str = ""
        self._bot_username: str = ""
        # Passive observation is restricted to human senders. Cache successful
        # Mattermost user lookups so ordinary channel chatter does not require
        # one API request per post. Lookup failures are deliberately not cached.
        self._sender_is_bot_cache: Dict[str, bool] = {}

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

        self._last_post_status: Optional[int] = None
        self._last_post_error: str = ""
        self._interaction_url = str(config.extra.get("interaction_url") or "").strip()
        parsed_interaction_url = urlparse(self._interaction_url)
        interaction_hostname = parsed_interaction_url.hostname or ""
        try:
            interaction_is_loopback = ipaddress.ip_address(
                interaction_hostname
            ).is_loopback
        except ValueError:
            interaction_is_loopback = interaction_hostname == "localhost" or (
                interaction_hostname.endswith(".localhost")
            )
        interaction_transport_safe = parsed_interaction_url.scheme == "https" or (
            parsed_interaction_url.scheme == "http" and interaction_is_loopback
        )
        self._interaction_url_valid = bool(
            interaction_transport_safe
            and interaction_hostname
            and not parsed_interaction_url.username
            and not parsed_interaction_url.password
        )
        self._interaction_host = str(
            config.extra.get("interaction_host") or "127.0.0.1"
        ).strip()
        try:
            self._interaction_port = int(config.extra.get("interaction_port", 8789))
        except (TypeError, ValueError):
            self._interaction_port = 8789
        self._interaction_path = parsed_interaction_url.path or "/mattermost/actions"
        trusted_raw = config.extra.get("interaction_allowed_cidrs", ())
        if isinstance(trusted_raw, str):
            trusted_values = trusted_raw.split(",")
        elif isinstance(trusted_raw, (list, tuple, set)):
            trusted_values = trusted_raw
        else:
            trusted_values = ()
        self._interaction_allowed_networks = []
        for value in trusted_values:
            try:
                self._interaction_allowed_networks.append(
                    ipaddress.ip_network(str(value).strip(), strict=False)
                )
            except ValueError:
                logger.warning(
                    "Mattermost: ignoring invalid interaction_allowed_cidrs entry: %r",
                    value,
                )
        try:
            self._interaction_timeout_seconds = max(
                30,
                min(3600, int(config.extra.get("interaction_timeout_seconds", 300))),
            )
        except (TypeError, ValueError):
            self._interaction_timeout_seconds = 300
        self._interaction_runner: Any = None
        self._interaction_start_failed = False
        self._pending_interactions: Dict[str, Dict[str, Any]] = {}
        self._rich_posts = str(config.extra.get("rich_posts", "false")).lower() in {
            "true", "1", "yes", "on",
        }
        self._feedback_buttons = str(
            config.extra.get("feedback_buttons", "false")
        ).lower() in {"true", "1", "yes", "on"}

        # Dedup cache (prevent reprocessing)
        self._dedup = MessageDeduplicator()

    @property
    def REQUIRES_EDIT_FINALIZE(self) -> bool:  # noqa: N802
        """Rich posts need a final edit even when streamed text is unchanged."""
        return self._rich_posts

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
        if ".." in path:
            logger.error("MM API path traversal blocked: %s", path)
            return {}
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
        if ".." in path:
            logger.error("MM API path traversal blocked: %s", path)
            return {}
        url = f"{self._base_url}/api/v4/{path.lstrip('/')}"
        self._last_post_status = None
        self._last_post_error = ""
        try:
            async with self._session.post(
                url, headers=self._headers(), json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                self._last_post_status = resp.status
                if resp.status >= 400:
                    body = await resp.text()
                    self._last_post_error = body or ""
                    logger.error("MM API POST %s → %s: %s", path, resp.status, body[:200])
                    return {}
                return await resp.json()
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
        if ".." in path:
            logger.error("MM API path traversal blocked: %s", path)
            return {}
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

    async def _api_delete(self, path: str) -> bool:
        """DELETE /api/v4/{path}."""
        import aiohttp

        if ".." in path:
            logger.error("MM API path traversal blocked: %s", path)
            return False
        url = f"{self._base_url}/api/v4/{path.lstrip('/')}"
        try:
            async with self._session.delete(
                url,
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    logger.error(
                        "MM API DELETE %s → %s: %s",
                        path,
                        resp.status,
                        body[:200],
                    )
                    return False
                return True
        except aiohttp.ClientError as exc:
            logger.error("MM API DELETE %s network error: %s", path, exc)
            return False

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

    async def connect(self, *, is_reconnect: bool = False) -> bool:
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

        if self._interactions_available():
            try:
                await self._start_interaction_server()
            except Exception as exc:
                self._interaction_start_failed = True
                logger.error("Mattermost: failed to start interaction endpoint: %s", exc)

        # Start WebSocket in background.
        self._ws_task = asyncio.create_task(self._ws_loop())
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

        await self._stop_interaction_server()

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
        """Send messages with optional Mattermost-native rich attachments."""
        if not content:
            return SendResult(success=True)

        formatted = self.format_message(content)
        chunks = self.truncate_message(formatted, MAX_POST_LENGTH)
        last_id = None
        for chunk in chunks:
            payload: Dict[str, Any] = {"channel_id": chat_id, "message": chunk}
            nonce = ""
            force_plain = bool(
                metadata
                and (metadata.get("expect_edits") or metadata.get("multi_chunk"))
            )
            if self._rich_posts and len(chunks) == 1 and not force_plain:
                actions = None
                if self._feedback_buttons and self._interactions_available():
                    nonce, actions = self._build_feedback_actions(chat_id)
                try:
                    from plugins.platforms.mattermost.rich_posts import (
                        render_rich_post,
                    )

                    rendered = render_rich_post(chunk, actions=actions)
                except Exception:
                    logger.warning(
                        "Mattermost: rich-post rendering failed; using Markdown",
                        exc_info=True,
                    )
                    rendered = None
                if rendered:
                    payload.update(rendered)
                    if nonce:
                        attachment = payload["props"]["attachments"][0]
                        remembered = self._remember_interaction(
                            nonce,
                            {
                                "kind": "feedback",
                                "channel_id": chat_id,
                                "post_id": "",
                                "expires_at": attachment["actions"][0]["integration"]["context"][
                                    "expires_at"
                                ],
                                "choices": ["helpful", "not_helpful"],
                                "attachment": attachment,
                            },
                        )
                        if not remembered:
                            attachment.pop("actions", None)
                            nonce = ""

            payload = _with_mentions_disabled(payload)
            # Thread support: reply_to or metadata["thread_id"] is the root post ID.
            resolved_root = await self._thread_root_for_send(reply_to, metadata)
            if resolved_root:
                payload["root_id"] = resolved_root

            data = await self._post_preserving_thread(chat_id, payload, metadata)
            if not data and (payload.get("props") or {}).get("attachments"):
                if nonce:
                    self._pending_interactions.pop(nonce, None)
                plain_payload = dict(payload)
                plain_payload.pop("props", None)
                plain_payload["message"] = chunk
                plain_payload = _with_mentions_disabled(plain_payload)
                logger.warning(
                    "Mattermost: rich post failed; retrying with plain Markdown"
                )
                data = await self._post_preserving_thread(
                    chat_id, plain_payload, metadata
                )
            if not data or "id" not in data:
                if nonce:
                    self._pending_interactions.pop(nonce, None)
                return SendResult(success=False, error="Failed to create post")
            last_id = data["id"]
            if nonce and nonce in self._pending_interactions:
                self._pending_interactions[nonce]["post_id"] = last_id

        return SendResult(success=True, message_id=last_id)

    # ------------------------------------------------------------------
    # Native interactive approval support
    # ------------------------------------------------------------------

    @staticmethod
    def _secret_value(name: str) -> str:
        """Read a secret through the authoritative profile-aware resolver."""
        try:
            from agent.secret_scope import get_secret
        except ImportError:
            # Defensive compatibility for loading this plugin outside the full
            # Hermes package. In normal gateway operation the resolver exists.
            return (os.getenv(name) or "").strip()

        try:
            value = get_secret(name)
        except Exception:
            # In multiplex mode an unscoped read is a security boundary, not a
            # reason to fall back to another profile's process environment.
            logger.warning("Mattermost: scoped secret lookup failed for %s", name)
            return ""
        return str(value).strip() if value is not None else ""

    def _interaction_secret(self) -> str:
        return self._secret_value("MATTERMOST_INTERACTION_SECRET")

    def _interactions_available(self) -> bool:
        """Return whether signed Mattermost callbacks are configured."""
        return bool(
            self._interaction_url_valid
            and self._interaction_secret()
            and self._interaction_allowed_networks
            and not self._interaction_start_failed
        )

    def _interaction_source_is_trusted(self, remote: Any) -> bool:
        """Accept callbacks only from explicitly configured network peers."""
        try:
            address = ipaddress.ip_address(str(remote or ""))
        except ValueError:
            return False
        if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
            address = address.ipv4_mapped
        return any(address in network for network in self._interaction_allowed_networks)

    async def _handle_interaction_request(self, request: Any) -> Any:
        """Handle Mattermost's bounded JSON action callback."""
        from aiohttp import web

        if not self._interaction_source_is_trusted(getattr(request, "remote", None)):
            return web.json_response(
                {"ephemeral_text": "Untrusted interaction source."}, status=403
            )
        if request.content_length is not None and request.content_length > 65_536:
            return web.json_response(
                {"ephemeral_text": "Interaction payload is too large."}, status=413
            )
        try:
            payload = await request.json()
        except web.HTTPRequestEntityTooLarge:
            return web.json_response(
                {"ephemeral_text": "Interaction payload is too large."}, status=413
            )
        except Exception:
            return web.json_response(
                {"ephemeral_text": "Invalid interaction payload."}, status=400
            )
        if not isinstance(payload, dict):
            return web.json_response(
                {"ephemeral_text": "Invalid interaction payload."}, status=400
            )
        body, status = await self._dispatch_interaction(payload)
        return web.json_response(body, status=status)

    async def _handle_interaction_health(self, _request: Any) -> Any:
        from aiohttp import web

        return web.json_response({"status": "ok", "platform": "mattermost"})

    async def _start_interaction_server(self) -> None:
        """Start the private callback listener used by Mattermost actions."""
        if self._interaction_runner is not None or not self._interactions_available():
            return
        from aiohttp import web

        app = web.Application(client_max_size=65_536)
        app.router.add_post(self._interaction_path, self._handle_interaction_request)
        app.router.add_get(
            f"{self._interaction_path.rstrip('/')}/health",
            self._handle_interaction_health,
        )
        runner = web.AppRunner(app)
        await runner.setup()
        try:
            site = web.TCPSite(
                runner, self._interaction_host, self._interaction_port
            )
            await site.start()
        except Exception:
            await runner.cleanup()
            raise
        self._interaction_runner = runner
        logger.info(
            "Mattermost: interaction endpoint listening on http://%s:%d%s",
            self._interaction_host,
            self._interaction_port,
            self._interaction_path,
        )

    async def _stop_interaction_server(self) -> None:
        """Stop the Mattermost callback listener, if it was started."""
        runner = self._interaction_runner
        self._interaction_runner = None
        if runner is not None:
            await runner.cleanup()

    def _interaction_signature(
        self,
        *,
        nonce: str,
        kind: str,
        choice: str,
        channel_id: str,
        expires_at: int,
    ) -> str:
        message = "\x00".join(
            (nonce, kind, choice, channel_id, str(expires_at))
        ).encode()
        return hmac.new(
            self._interaction_secret().encode(), message, hashlib.sha256
        ).hexdigest()

    def _build_interaction_action(
        self,
        *,
        nonce: str,
        kind: str,
        choice: str,
        channel_id: str,
        name: str,
        style: str = "default",
        expires_at: Optional[int] = None,
    ) -> Dict[str, Any]:
        if expires_at is None:
            expires_at = int(time.time()) + self._interaction_timeout_seconds
        context = {
            "nonce": nonce,
            "kind": kind,
            "choice": choice,
            "expires_at": expires_at,
            "signature": self._interaction_signature(
                nonce=nonce,
                kind=kind,
                choice=choice,
                channel_id=channel_id,
                expires_at=expires_at,
            ),
        }
        return {
            "id": f"hermes_{kind}_{choice}",
            "type": "button",
            "name": name,
            "style": style,
            "integration": {"url": self._interaction_url, "context": context},
        }

    def _build_feedback_actions(
        self, channel_id: str
    ) -> Tuple[str, List[Dict[str, Any]]]:
        nonce = secrets.token_urlsafe(18)
        expires_at = int(time.time()) + self._interaction_timeout_seconds
        actions = [
            self._build_interaction_action(
                nonce=nonce,
                kind="feedback",
                choice="helpful",
                channel_id=channel_id,
                name="Helpful",
                style="success",
                expires_at=expires_at,
            ),
            self._build_interaction_action(
                nonce=nonce,
                kind="feedback",
                choice="not_helpful",
                channel_id=channel_id,
                name="Not helpful",
                style="default",
                expires_at=expires_at,
            ),
        ]
        return nonce, actions

    def _remember_interaction(self, nonce: str, state: Dict[str, Any]) -> bool:
        """Remember bounded, single-use callback state without evicting approvals for feedback."""
        now = int(time.time())
        for pending_nonce, pending_state in list(self._pending_interactions.items()):
            if int(pending_state.get("expires_at") or 0) <= now:
                self._pending_interactions.pop(pending_nonce, None)

        if len(self._pending_interactions) >= 256:
            feedback_nonce = next(
                (
                    pending_nonce
                    for pending_nonce, pending_state in self._pending_interactions.items()
                    if pending_state.get("kind") == "feedback"
                ),
                None,
            )
            if feedback_nonce is not None:
                self._pending_interactions.pop(feedback_nonce, None)
            elif state.get("kind") == "feedback":
                return False
            else:
                self._pending_interactions.pop(next(iter(self._pending_interactions)))

        self._pending_interactions[nonce] = state
        return True

    async def send_exec_approval(
        self,
        chat_id: str,
        command: str,
        session_key: str,
        description: str = "dangerous command",
        metadata: Optional[Dict[str, Any]] = None,
        allow_permanent: bool = True,
        allow_session: bool = True,
        smart_denied: bool = False,
        approval_id: Optional[str] = None,
    ) -> SendResult:
        """Send a signed Mattermost attachment with approval buttons."""
        if not self._interactions_available():
            return SendResult(
                success=False,
                error=(
                    "Mattermost interactions require a valid interaction_url, "
                    "MATTERMOST_INTERACTION_SECRET, trusted interaction_allowed_cidrs, "
                    "and a healthy callback listener"
                ),
            )
        if not approval_id:
            return SendResult(
                success=False,
                error="Mattermost interactive approvals require an exact approval ID",
            )

        nonce = secrets.token_urlsafe(18)
        expires_at = int(time.time()) + self._interaction_timeout_seconds
        actions = [
            self._build_interaction_action(
                nonce=nonce,
                kind="approval",
                choice="once",
                channel_id=chat_id,
                name="Allow Once",
                style="primary",
                expires_at=expires_at,
            )
        ]
        if not smart_denied and allow_session:
            actions.append(
                self._build_interaction_action(
                    nonce=nonce,
                    kind="approval",
                    choice="session",
                    channel_id=chat_id,
                    name="Allow Session",
                    expires_at=expires_at,
                )
            )
        if not smart_denied and allow_permanent:
            actions.append(
                self._build_interaction_action(
                    nonce=nonce,
                    kind="approval",
                    choice="always",
                    channel_id=chat_id,
                    name="Always Allow",
                    expires_at=expires_at,
                )
            )
        actions.append(
            self._build_interaction_action(
                nonce=nonce,
                kind="approval",
                choice="deny",
                channel_id=chat_id,
                name="Deny",
                style="danger",
                expires_at=expires_at,
            )
        )

        header = "**⚠️ Command Approval Required**"
        if smart_denied:
            header += "\n\n**Smart DENY:** owner override applies to this operation only."
        command_preview = command[:6000] + ("…" if len(command) > 6000 else "")
        text = (
            f"{header}\n\n```\n{command_preview}\n```\n\n"
            f"Reason: {description[:1000]}"
        )
        fallback = f"Command approval required: {command[:160]}"
        attachment = {
            "fallback": fallback,
            "color": "warning",
            "text": text,
            "actions": actions,
        }
        payload: Dict[str, Any] = {
            "channel_id": chat_id,
            "message": fallback,
            "props": {"attachments": [attachment]},
        }
        root_id = await self._thread_root_for_send(None, metadata)
        if root_id:
            payload["root_id"] = root_id

        self._remember_interaction(
            nonce,
            {
                "kind": "approval",
                "channel_id": chat_id,
                "post_id": "",
                "session_key": session_key,
                "approval_id": approval_id,
                "expires_at": expires_at,
                "choices": [
                    action["integration"]["context"]["choice"] for action in actions
                ],
                "attachment": attachment,
            },
        )
        data = await self._post_preserving_thread(chat_id, payload, metadata)
        post_id = str(data.get("id") or "") if data else ""
        if not post_id:
            self._pending_interactions.pop(nonce, None)
            return SendResult(success=False, error="Failed to post Mattermost approval")
        self._pending_interactions[nonce]["post_id"] = post_id
        return SendResult(success=True, message_id=post_id, raw_response=data)

    def _is_interactive_user_authorized(
        self,
        user_id: str,
        *,
        channel_id: str = "",
        user_name: Optional[str] = None,
        team_id: str = "",
    ) -> bool:
        """Apply the profile-bound gateway policy to action callbacks."""
        normalized_user_id = str(user_id or "").strip()
        if not normalized_user_id:
            return False
        # GatewayRunner installs this callback with its bound profile name, so
        # it consults the correct profile's allowlist and pairing store. Never
        # reconstruct a profile-less SessionSource or fall back to global env
        # authorization for a command-approval callback.
        return (
            self._is_sender_authorized(
                normalized_user_id,
                "group",
                str(channel_id or normalized_user_id),
            )
            is True
        )

    async def _dispatch_interaction(
        self, payload: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], int]:
        """Validate and execute one Mattermost interaction callback."""
        context = payload.get("context")
        if not isinstance(context, dict):
            return {"ephemeral_text": "Invalid or expired Hermes action."}, 401
        nonce = str(context.get("nonce") or "")
        kind = str(context.get("kind") or "")
        choice = str(context.get("choice") or "")
        supplied_signature = str(context.get("signature") or "")
        raw_expires_at = context.get("expires_at")
        if not isinstance(raw_expires_at, (int, str)):
            return {"ephemeral_text": "Invalid Hermes action."}, 401
        try:
            expires_at = int(raw_expires_at)
        except (TypeError, ValueError):
            return {"ephemeral_text": "Invalid Hermes action."}, 401
        state = self._pending_interactions.get(nonce)
        if not state:
            return {"ephemeral_text": "This Hermes action has expired."}, 200

        channel_id = str(payload.get("channel_id") or "")
        post_id = str(payload.get("post_id") or "")
        expected_signature = self._interaction_signature(
            nonce=nonce,
            kind=kind,
            choice=choice,
            channel_id=channel_id,
            expires_at=expires_at,
        )
        if not supplied_signature or not hmac.compare_digest(
            supplied_signature, expected_signature
        ):
            return {"ephemeral_text": "Invalid Hermes action."}, 401
        if (
            kind != state.get("kind")
            or choice not in state.get("choices", ())
            or expires_at != state.get("expires_at")
            or channel_id != state.get("channel_id")
            or post_id != state.get("post_id")
        ):
            return {"ephemeral_text": "Invalid Hermes action target."}, 401
        if time.time() > expires_at:
            self._pending_interactions.pop(nonce, None)
            return {"ephemeral_text": "This Hermes action has expired."}, 200

        user_id = str(payload.get("user_id") or "")
        user_name = str(payload.get("user_name") or user_id or "unknown")
        if not self._is_interactive_user_authorized(
            user_id,
            channel_id=channel_id,
            user_name=user_name,
            team_id=str(payload.get("team_id") or ""),
        ):
            return {"ephemeral_text": "You are not authorized for this action."}, 403

        if kind == "feedback" and choice in {"helpful", "not_helpful"}:
            state = self._pending_interactions.pop(nonce, None)
            if not state:
                return {"ephemeral_text": "This Hermes action has expired."}, 200
            logger.info(
                "Mattermost feedback clicked: value=%s user=%s channel=%s post=%s",
                choice,
                user_id,
                channel_id,
                post_id,
            )
            resolved_attachment = dict(state["attachment"])
            resolved_attachment.pop("actions", None)
            return {
                "ephemeral_text": "Thanks for the feedback.",
                "update": {
                    "message": str(resolved_attachment.get("fallback") or ""),
                    "props": {"attachments": [resolved_attachment]},
                },
            }, 200

        if kind != "approval" or choice not in {"once", "session", "always", "deny"}:
            return {"ephemeral_text": "Invalid Hermes action."}, 400

        state = self._pending_interactions.pop(nonce, None)
        if not state:
            return {"ephemeral_text": "This Hermes action has expired."}, 200
        try:
            from tools.approval import resolve_gateway_approval

            count = resolve_gateway_approval(
                str(state["session_key"]),
                choice,
                approval_id=str(state["approval_id"]),
            )
        except Exception:
            logger.exception("Mattermost: failed to resolve approval callback")
            count = 0

        label_map = {
            "once": f"✅ Approved once by {user_name}",
            "session": f"✅ Approved for session by {user_name}",
            "always": f"✅ Approved permanently by {user_name}",
            "deny": f"❌ Denied by {user_name}",
        }
        decision = label_map[choice]
        if not count:
            decision = "⌛ Approval expired — command was not run."
        else:
            self.resume_typing_for_chat(channel_id)
        resolved_attachment = dict(state["attachment"])
        resolved_attachment.pop("actions", None)
        return {
            "update": {
                "message": decision,
                "props": {"attachments": [resolved_attachment]},
            }
        }, 200

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
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Edit a post; render rich structure only for a single final chunk."""
        formatted = self.format_message(content)
        payload: Dict[str, Any] = {"message": formatted}
        nonce = ""
        multi_chunk = bool((metadata or {}).get("multi_chunk"))
        if finalize and self._rich_posts and not multi_chunk:
            actions = None
            if self._feedback_buttons and self._interactions_available():
                nonce, actions = self._build_feedback_actions(chat_id)
            try:
                from plugins.platforms.mattermost.rich_posts import (
                    render_rich_post,
                )

                rendered = render_rich_post(formatted, actions=actions)
            except Exception:
                logger.warning(
                    "Mattermost: final rich-post edit failed to render",
                    exc_info=True,
                )
                rendered = None
            if rendered:
                payload.update(rendered)
                if nonce:
                    attachment = payload["props"]["attachments"][0]
                    remembered = self._remember_interaction(
                        nonce,
                        {
                            "kind": "feedback",
                            "channel_id": chat_id,
                            "post_id": message_id,
                            "expires_at": attachment["actions"][0]["integration"][
                                "context"
                            ]["expires_at"],
                            "choices": ["helpful", "not_helpful"],
                            "attachment": attachment,
                        },
                    )
                    if not remembered:
                        attachment.pop("actions", None)
                        nonce = ""

        payload = _with_mentions_disabled(payload)
        data = await self._api_put(f"posts/{message_id}/patch", payload)
        if not data and (payload.get("props") or {}).get("attachments"):
            if nonce:
                self._pending_interactions.pop(nonce, None)
            data = await self._api_put(
                f"posts/{message_id}/patch",
                _with_mentions_disabled({"message": formatted, "props": {}}),
            )
        if not data or "id" not in data:
            if nonce:
                self._pending_interactions.pop(nonce, None)
            return SendResult(success=False, error="Failed to edit post")
        return SendResult(success=True, message_id=data["id"])

    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        """Delete a Mattermost post used as an obsolete streaming chunk."""
        del chat_id  # Mattermost deletes posts by globally unique post ID.
        post_id = str(message_id or "")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", post_id):
            logger.warning("Mattermost: refusing malformed post ID for deletion")
            return False
        return await self._api_delete(f"posts/{post_id}")

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
                async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status >= 500 or resp.status == 429:
                        if attempt < 2:
                            logger.debug("Mattermost download retry %d/2 for %s (status %d)",
                                         attempt + 1, url[:80], resp.status)
                            await asyncio.sleep(1.5 * (attempt + 1))
                            continue
                    if resp.status >= 400:
                        return await self.send(chat_id, f"{caption or ''}\n{url}".strip(), reply_to, metadata=metadata)
                    file_data = await resp.read()
                    ct = resp.content_type or "application/octet-stream"
                    break
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
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

        payload: Dict[str, Any] = _with_mentions_disabled({
            "channel_id": chat_id,
            "message": caption or "",
            "file_ids": [file_id],
        })
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

        payload: Dict[str, Any] = _with_mentions_disabled({
            "channel_id": chat_id,
            "message": caption or "",
            "file_ids": [file_id],
        })
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

                payload: Dict[str, Any] = _with_mentions_disabled({
                    "channel_id": chat_id,
                    "message": "\n".join(caption_parts),
                    "file_ids": file_ids,
                })
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

    @staticmethod
    def _mattermost_bool(value: Any, *, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "on"}
        return bool(value)

    def _mattermost_config_value(self, key: str, env_name: str) -> Any:
        env_value = os.getenv(env_name)
        if env_value not in {None, ""}:
            return env_value
        return (self.config.extra or {}).get(key)

    def _mattermost_channel_set(self, key: str, env_name: str) -> set[str]:
        raw = self._mattermost_config_value(key, env_name)
        if isinstance(raw, (list, tuple, set)):
            return {str(value).strip() for value in raw if str(value).strip()}
        return {value.strip() for value in str(raw or "").split(",") if value.strip()}

    def _mattermost_require_mention(self) -> bool:
        return self._mattermost_bool(
            self._mattermost_config_value("require_mention", "MATTERMOST_REQUIRE_MENTION"),
            default=True,
        )

    def _mattermost_observe_unmentioned_channel_messages(self) -> bool:
        """Whether eligible unmentioned channel posts are persisted as context."""
        configured = self._mattermost_config_value(
            "observe_unmentioned_channel_messages",
            "MATTERMOST_OBSERVE_UNMENTIONED_CHANNEL_MESSAGES",
        )
        return self._mattermost_bool(configured)

    def _mattermost_observation_enabled(
        self, channel_id: str, allowed_channels: set[str]
    ) -> bool:
        """Fail closed unless passive observation has an explicit channel scope."""
        return bool(
            self._mattermost_observe_unmentioned_channel_messages()
            and allowed_channels
            and channel_id in allowed_channels
        )

    @staticmethod
    def _mattermost_post_has_automation_marker(post: Dict[str, Any]) -> bool:
        """Return True for incoming-webhook or explicitly bot-marked posts."""
        props = post.get("props") or {}
        if isinstance(props, str):
            try:
                props = json.loads(props)
            except (TypeError, ValueError):
                # Malformed automation metadata is not trustworthy enough for
                # passive persistence. Fail closed.
                return True
        if not isinstance(props, dict):
            return True
        return any(
            MattermostAdapter._mattermost_bool(props.get(key))
            for key in ("from_webhook", "from_bot")
        )

    async def _mattermost_observation_sender_is_human(
        self, post: Dict[str, Any]
    ) -> bool:
        """Verify that a passive-observation candidate came from a human.

        Authorization and human-vs-automation are separate gates: an allowlisted
        bot or webhook must still never enter shared passive context.
        """
        if self._mattermost_post_has_automation_marker(post):
            return False
        sender_id = str(post.get("user_id") or "").strip()
        if not sender_id:
            return False
        cached = self._sender_is_bot_cache.get(sender_id)
        if cached is not None:
            return not cached
        try:
            user = await self._api_get(f"users/{sender_id}")
        except Exception:
            logger.warning(
                "Mattermost: sender lookup failed; skipping passive observation "
                "(user=%s)",
                sender_id,
                exc_info=True,
            )
            return False
        if (
            not isinstance(user, dict)
            or type(user.get("is_bot")) is not bool
        ):
            logger.warning(
                "Mattermost: sender lookup lacked a boolean is_bot; skipping passive "
                "observation (user=%s)",
                sender_id,
            )
            return False
        is_bot = user["is_bot"]
        if len(self._sender_is_bot_cache) >= 4096:
            self._sender_is_bot_cache.pop(next(iter(self._sender_is_bot_cache)))
        self._sender_is_bot_cache[sender_id] = is_bot
        return not is_bot

    def _mattermost_thread_id(self, post: Dict[str, Any], chat_type: str) -> Optional[str]:
        thread_id = post.get("root_id") or None
        if not thread_id and self._reply_mode == "thread" and chat_type != "dm":
            thread_id = post.get("id") or None
        return thread_id

    @staticmethod
    def _mattermost_observed_attributed_text(
        sender_name: str, sender_id: str, message_text: str
    ) -> str:
        sender = sender_name or sender_id or "unknown"
        user_id = sender_id or "unknown"
        return f"[{sender}|{user_id}]\n{message_text}"

    def _mattermost_observed_channel_prompt(self) -> str:
        return (
            "You are handling a Mattermost channel message.\n"
            f"- Your identity: user_id={self._bot_user_id}, @-mention name "
            f"in this workspace=@{self._bot_username}.\n"
            "- observed Mattermost channel context may be provided in a separate "
            "context-only block before the current message; it is not necessarily "
            "addressed to you.\n"
            "- Treat only the current addressed message as a request, and use "
            "observed context only when that message asks for it."
        )

    def _observe_unmentioned_channel_message(
        self,
        *,
        post: Dict[str, Any],
        channel_id: str,
        chat_type: str,
        sender_id: str,
        sender_name: str,
        message_text: str,
    ) -> None:
        """Persist authorized Mattermost chatter without dispatching the agent."""
        store = getattr(self, "_session_store", None)
        if not store:
            return
        try:
            file_ids = post.get("file_ids") or []
            observed_text = message_text
            if file_ids:
                suffix = (
                    f"[Observed Mattermost post includes {len(file_ids)} attachment(s); "
                    "passive observation does not download attachments.]"
                )
                observed_text = f"{observed_text}\n{suffix}" if observed_text else suffix
            if not observed_text:
                return
            source = self.build_source(
                chat_id=channel_id,
                chat_type=chat_type,
                user_id=None,
                user_name=None,
                thread_id=self._mattermost_thread_id(post, chat_type),
                message_id=post.get("id", ""),
                role_authorized=True,
            )
            session_entry = store.get_or_create_session(source)
            entry: Dict[str, Any] = {
                "role": "user",
                "content": self._mattermost_observed_attributed_text(
                    sender_name, sender_id, observed_text
                ),
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "observed": True,
            }
            if post.get("id"):
                entry["message_id"] = str(post["id"])
            store.append_to_transcript(session_entry.session_id, entry)
            logger.info(
                "Mattermost: channel message observed (no bot trigger): channel=%s from=%s",
                channel_id,
                sender_id or "unknown",
            )
        except Exception as exc:
            logger.warning("Mattermost: failed to observe channel message: %s", exc)

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

        # Resolve sender before mention gating. Passive observation must make a
        # full authorization decision before anything reaches the transcript.
        sender_id = post.get("user_id", "")
        sender_name = data.get("sender_name", "").lstrip("@") or sender_id

        # For DMs, user_id is sufficient.  For channels, check for @mention.
        message_text = post.get("message", "")

        # Mention-gating for non-DM channels.
        observation_active = False
        # Config (config.yaml `mattermost.*` with env-var fallback):
        #   require_mention / MATTERMOST_REQUIRE_MENTION: Require @mention in channels (default: true)
        #   free_response_channels / MATTERMOST_FREE_RESPONSE_CHANNELS: Channel IDs where bot responds without mention
        #   allowed_channels / MATTERMOST_ALLOWED_CHANNELS: If set, bot ONLY responds in these channels (whitelist)
        if channel_type_raw != "D":
            # allowed_channels check (whitelist — must pass before other gating).
            # When set, messages from channels NOT in this list are silently
            # ignored, even if @mentioned.  DMs are already excluded above.
            allowed_channels = self._mattermost_channel_set(
                "allowed_channels", "MATTERMOST_ALLOWED_CHANNELS"
            )
            if allowed_channels and channel_id not in allowed_channels:
                logger.debug(
                    "Mattermost: ignoring message in non-allowed channel: %s",
                    channel_id,
                )
                return

            require_mention = self._mattermost_require_mention()
            free_channels = self._mattermost_channel_set(
                "free_response_channels", "MATTERMOST_FREE_RESPONSE_CHANNELS"
            )
            is_free_channel = channel_id in free_channels
            observation_active = (
                require_mention
                and not is_free_channel
                and self._mattermost_observation_enabled(channel_id, allowed_channels)
            )

            mention_patterns = [
                f"@{self._bot_username}",
                f"@{self._bot_user_id}",
            ]
            has_mention = any(
                pattern.lower() in message_text.lower()
                for pattern in mention_patterns
            )

            if require_mention and not is_free_channel and not has_mention:
                if (
                    observation_active
                    and not message_text.lstrip().startswith("/")
                    and self._is_sender_authorized(sender_id, chat_type, channel_id)
                    is True
                    and await self._mattermost_observation_sender_is_human(post)
                ):
                    self._observe_unmentioned_channel_message(
                        post=post,
                        channel_id=channel_id,
                        chat_type=chat_type,
                        sender_id=sender_id,
                        sender_name=sender_name,
                        message_text=message_text,
                    )
                logger.debug(
                    "Mattermost: skipping non-DM message without @mention (channel=%s)",
                    channel_id,
                )
                return

            # Addressed turns in observed channels share the same channel/thread
            # session as passive rows. Authenticate before removing per-sender
            # routing identity from the event source.
            if observation_active:
                if self._is_sender_authorized(
                    sender_id, chat_type, channel_id
                ) is not True:
                    logger.debug(
                        "Mattermost: ignoring unauthorized sender in observed channel: %s",
                        sender_id,
                    )
                    return
                if not await self._mattermost_observation_sender_is_human(post):
                    logger.debug(
                        "Mattermost: ignoring automated sender in observed channel: %s",
                        sender_id,
                    )
                    return

            # Strip @mention from the message text so the agent sees clean input.
            if has_mention:
                for pattern in mention_patterns:
                    message_text = re.sub(
                        re.escape(pattern), "", message_text, flags=re.IGNORECASE
                    ).strip()

        # Thread support: replies use root_id; in thread reply mode, a top-level
        # channel post becomes the root for the session and response.
        thread_id = self._mattermost_thread_id(post, chat_type)

        # Determine message type.
        file_ids = post.get("file_ids") or []
        msg_type = MessageType.TEXT
        if message_text[:1].isspace() and message_text.lstrip().startswith("/"):
            message_text = message_text.lstrip()
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
            message_id=post_id,
        )

        # Per-channel ephemeral prompt
        from gateway.platforms.base import resolve_channel_prompt
        _channel_prompt = resolve_channel_prompt(
            self.config.extra, channel_id, None,
        )

        if observation_active:
            source = dataclasses.replace(
                source,
                user_id=None,
                user_name=None,
                user_id_alt=None,
                # The adapter completed the gateway's full authorization
                # callback before converting this into a shared routing source.
                role_authorized=True,
            )
            if msg_type != MessageType.COMMAND:
                message_text = self._mattermost_observed_attributed_text(
                    sender_name, sender_id, message_text
                )
            observe_prompt = self._mattermost_observed_channel_prompt()
            _channel_prompt = (
                f"{_channel_prompt}\n\n{observe_prompt}"
                if _channel_prompt
                else observe_prompt
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
    multi_chunk: bool = False,
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
            rich_posts = str(
                (getattr(pconfig, "extra", {}) or {}).get("rich_posts", "false")
            ).lower() in {"true", "1", "yes", "on"}
            if (
                rich_posts
                and not multi_chunk
                and len(message) <= MAX_POST_LENGTH
            ):
                try:
                    from plugins.platforms.mattermost.rich_posts import (
                        render_rich_post,
                    )

                    rendered = render_rich_post(message)
                except Exception:
                    rendered = None
                if rendered:
                    payload.update(rendered)
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
                if resp.status in {200, 201}:
                    data = await resp.json()
                elif "props" in payload:
                    plain_payload = dict(payload)
                    plain_payload.pop("props", None)
                    plain_payload["message"] = message
                    async with session.post(
                        f"{base_url}/api/v4/posts",
                        headers=headers,
                        json=plain_payload,
                        **_req_kw,
                    ) as plain_resp:
                        if plain_resp.status not in {200, 201}:
                            body = await plain_resp.text()
                            return {
                                "error": (
                                    f"Mattermost API error ({plain_resp.status}): "
                                    f"{body[:400]}"
                                )
                            }
                        data = await plain_resp.json()
                else:
                    body = await resp.text()
                    return {
                        "error": (
                            f"Mattermost API error ({resp.status}): "
                            f"{body[:400]}"
                        )
                    }
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
    from hermes_cli.config import get_env_value, remove_env_value, save_env_value
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
    home_channel = prompt("Home channel ID (leave empty to set later with /set-home)").strip()
    if home_channel:
        save_env_value("MATTERMOST_HOME_CHANNEL", home_channel)
    else:
        if remove_env_value("MATTERMOST_HOME_CHANNEL"):
            print_info("Home channel cleared.")
    print_info("   Open config in your editor:  hermes config edit")


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
    update. Rich-post settings are returned as ``PlatformConfig.extra``
    values because they are adapter-native rather than legacy env settings.
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
    nested_extra = mattermost_cfg.get("extra")
    extras = dict(nested_extra) if isinstance(nested_extra, dict) else {}
    for key in (
        "rich_posts",
        "feedback_buttons",
        "interaction_url",
        "interaction_host",
        "interaction_port",
        "interaction_timeout_seconds",
        "interaction_allowed_cidrs",
        "observe_unmentioned_channel_messages",
    ):
        if key in mattermost_cfg:
            extras[key] = mattermost_cfg[key]
    return extras or None


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
        validate_config=validate_mattermost_config,
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
