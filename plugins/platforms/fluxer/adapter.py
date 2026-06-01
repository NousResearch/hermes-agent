"""Fluxer platform plugin for Hermes Agent.

Text-first adapter:
- REST `POST /channels/:id/messages` for outbound messages.
- Fluxer Gateway websocket `MESSAGE_CREATE` events for inbound messages.

Fluxer self-hosting is still moving, so this adapter intentionally keeps the
surface conservative and easy to test. Media/rich embeds can layer on once the
API settles.
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import re
import subprocess
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urljoin, urlparse

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    SUPPORTED_IMAGE_DOCUMENT_TYPES,
    cache_audio_from_url,
    cache_document_from_bytes,
    cache_image_from_url,
    safe_url_for_log,
)

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 4000
_DEFAULT_BASE_URL = "https://api.fluxer.app/v1"
_GATEWAY_VERSION = 1
_VOICE_MESSAGE_FLAG = 1 << 13
_RECONNECT_BASE_DELAY = 2.0
_RECONNECT_MAX_DELAY = 60.0
_HEARTBEAT_ACK_TIMEOUT_FACTOR = 2.5
_DEFAULT_BACKLOG_LIMIT = 25
_DEFAULT_BACKLOG_BOOTSTRAP_SECONDS = 120
_EXEC_APPROVAL_REACTIONS = {
    "✅": "once",
    "☑️": "once",
    "🔁": "session",
    "♾": "always",
    "♾️": "always",
    "❌": "deny",
    "✖️": "deny",
    "✖": "deny",
}
_EXEC_APPROVAL_REACTION_ORDER = ("✅", "🔁", "♾️", "❌")
_MENTION_EVERYONE_RE = re.compile(r"@(everyone|here)\b", re.IGNORECASE)
_MENTION_ROLE_RE = re.compile(r"<@&\d+>")
_MENTION_USER_RE = re.compile(r"<@!?\d+>")


def _strip_slash(url: str) -> str:
    return (url or "").strip().rstrip("/")


def _api_base(base_url: str) -> str:
    """Normalize a user-provided Fluxer URL to the REST API base.

    The official hosted API is already scoped as ``https://api.fluxer.app/v1``.
    Self-hosted Fluxer may expose either an already-scoped API URL or a plain
    web origin; preserve scoped URLs and append ``/api`` only for a plain origin.
    """
    base = _strip_slash(base_url)
    if not base:
        return ""
    if base.endswith("/api") or base.endswith("/api/v1") or base.endswith("/v1") or "/api/" in base:
        return base
    if base == "https://api.fluxer.app":
        return f"{base}/v1"
    return f"{base}/api"


def _build_identify_payload(bot_token: str) -> Dict[str, Any]:
    return {
        "op": 2,
        "d": {
            "token": bot_token,
            "properties": {
                "os": "linux",
                "browser": "hermes",
                "device": "hermes",
            },
        },
    }


def _headers(bot_token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bot {bot_token}",
        "Content-Type": "application/json",
        "User-Agent": "Hermes-Fluxer/0.1",
    }


def _auth_headers(bot_token: str) -> Dict[str, str]:
    """Headers for requests where httpx must set Content-Type itself."""
    return {
        "Authorization": f"Bot {bot_token}",
        "User-Agent": "Hermes-Fluxer/0.1",
    }


def _event_seq(payload: Dict[str, Any]) -> Optional[int]:
    seq = payload.get("s")
    try:
        return int(seq) if seq is not None else None
    except (TypeError, ValueError):
        return None


def _author_name(author: Dict[str, Any]) -> Optional[str]:
    return (
        author.get("global_name")
        or author.get("display_name")
        or author.get("username")
        or author.get("name")
    )


def _chat_type(raw: Any) -> str:
    if isinstance(raw, str):
        lowered = raw.lower()
        if lowered in {"dm", "direct", "private"}:
            return "dm"
        if lowered in {"group_dm", "group", "group-dm"}:
            return "group"
        if lowered in {"thread", "public_thread", "private_thread", "news_thread"}:
            return "thread"
        if lowered in {"forum", "guild_forum"}:
            return "forum"
        try:
            raw = int(lowered)
        except ValueError:
            return "channel"
    # Discord-like channel types in several codebases:
    # 1 = DM, 3 = group DM, 10/11/12 = thread, 15 = forum.
    if raw == 1:
        return "dm"
    if raw == 3:
        return "group"
    if raw in {10, 11, 12}:
        return "thread"
    if raw == 15:
        return "forum"
    return "channel"


def _quote_id(value: Any) -> str:
    """Quote a Fluxer path segment without letting IDs alter REST routes."""
    return quote(str(value or ""), safe="")


def _attachment_url(att: Dict[str, Any]) -> str:
    return str(att.get("url") or att.get("proxy_url") or "").strip()


def _attachment_filename(att: Dict[str, Any]) -> str:
    return str(att.get("filename") or att.get("title") or att.get("name") or "attachment").strip()


def _extension_for_attachment(att: Dict[str, Any], content_type: str, default: str = ".bin") -> str:
    filename = _attachment_filename(att)
    suffix = Path(filename).suffix.lower()
    if suffix:
        return suffix
    subtype = (content_type or "").split("/", 1)[-1].split(";", 1)[0].lower()
    aliases = {"jpeg": ".jpg", "plain": ".txt", "mpeg": ".mp3", "quicktime": ".mov"}
    if subtype:
        return aliases.get(subtype, f".{subtype}")
    return default


def _message_type_for_media(media_types: List[str]) -> MessageType:
    if not media_types:
        return MessageType.TEXT
    if any(m.startswith("image/") for m in media_types):
        return MessageType.PHOTO
    if any(m.startswith("audio/") for m in media_types):
        return MessageType.AUDIO
    if any(m.startswith("video/") for m in media_types):
        return MessageType.VIDEO
    return MessageType.DOCUMENT


def _is_voice_message(data: Dict[str, Any]) -> bool:
    try:
        flags = int(data.get("flags") or 0)
    except (TypeError, ValueError):
        flags = 0
    if flags & _VOICE_MESSAGE_FLAG:
        return True
    if str(data.get("message_type") or data.get("type") or "").lower() in {"voice", "voice_message"}:
        return True
    attachments = data.get("attachments") or []
    if isinstance(attachments, list):
        for att in attachments:
            if not isinstance(att, dict):
                continue
            if bool(att.get("is_voice_message") or att.get("voice") or att.get("voice_message")):
                return True
    return False


def _audio_duration_seconds(path: Path) -> int:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nk=1:nw=1", str(path)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
        duration = float((result.stdout or "").strip())
        if duration > 0:
            return max(1, int(round(duration)))
    except Exception:
        pass
    return 1


def _coerce_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "y", "*"}:
            return True
        if normalized in {"0", "false", "no", "off", "n", ""}:
            return False
    return default


def _split_ids(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    return {part.strip() for part in str(value).split(",") if part.strip()}


def _normalize_emoji_name(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("name") or value.get("id") or ""
    text = str(value or "").strip()
    # Variation selector makes some emoji arrive as either ♾ or ♾️.
    if text == "♾️":
        return "♾"
    return text


def _parse_fluxer_timestamp(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _extract_message_list(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("messages", "items", "data", "results"):
            items = payload.get(key)
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
    return []


def _message_content(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    content = data.get("content")
    return content if isinstance(content, str) else ""


def _extract_reply_message(data: Dict[str, Any]) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Return the replied-to message id and any embedded referenced payload.

    Fluxer deployments vary while the API is young: some
    events embed ``referenced_message``, some use ``message_reference`` only,
    and older bridge payloads may call it ``reply_to``. Normalize all obvious
    forms so Hermes gets reply context whenever the information is available.
    """
    candidates = (
        data.get("referenced_message"),
        data.get("referencedMessage"),
        data.get("reply_to_message"),
        data.get("replyToMessage"),
        data.get("reply_to"),
    )
    for candidate in candidates:
        if isinstance(candidate, dict):
            msg_id = candidate.get("id") or candidate.get("message_id")
            if msg_id:
                return str(msg_id), candidate
        elif candidate:
            return str(candidate), None

    reference = data.get("message_reference") or data.get("messageReference") or {}
    if isinstance(reference, dict):
        msg_id = reference.get("message_id") or reference.get("messageId") or reference.get("id")
        if msg_id:
            return str(msg_id), None
    return None, None


def _fluxer_component_row(buttons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"type": 1, "components": buttons}]


def _fluxer_button(*, custom_id: str, label: str, style: int) -> Dict[str, Any]:
    return {"type": 2, "style": style, "custom_id": custom_id[:100], "label": label, "disabled": False}


def _directory_channel_type(raw: Any) -> Optional[str]:
    if raw in (0, "0", "text", "guild_text"):
        return "channel"
    if raw in (5, "5", "announcement", "news"):
        return "channel"
    if raw in (10, 11, 12, "10", "11", "12", "thread", "public_thread", "private_thread"):
        return "thread"
    if raw in (15, "15", "forum", "guild_forum"):
        return "forum"
    if raw in (1, "1", "dm", "direct"):
        return "dm"
    if raw in (3, "3", "group_dm", "group"):
        return "group"
    return None


class FluxerAdapter(BasePlatformAdapter):
    """Fluxer adapter using bot REST + Gateway websocket APIs."""

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("fluxer"))
        extra = getattr(config, "extra", {}) or {}
        self.base_url = _strip_slash(
            os.getenv("FLUXER_BASE_URL") or extra.get("base_url") or _DEFAULT_BASE_URL
        )
        self.api_base_url = _api_base(self.base_url)
        self.bot_token = (
            os.getenv("FLUXER_BOT_TOKEN") or extra.get("bot_token") or ""
        ).strip()
        self.gateway_url = _strip_slash(
            os.getenv("FLUXER_GATEWAY_URL") or extra.get("gateway_url") or ""
        )
        self.bot_user_id: Optional[str] = str(extra.get("bot_user_id")) if extra.get("bot_user_id") else None
        self._allowed_user_ids = _split_ids(os.getenv("FLUXER_ALLOWED_USERS") or extra.get("allowed_users"))
        self._allow_all_users = _coerce_bool(os.getenv("FLUXER_ALLOW_ALL_USERS", extra.get("allow_all_users")), False)
        self._backlog_enabled = str(
            os.getenv("FLUXER_BACKLOG_RECOVERY", extra.get("backlog_recovery", "true"))
        ).strip().lower() not in {"0", "false", "no", "off"}
        self._backlog_limit = min(
            _coerce_int(os.getenv("FLUXER_BACKLOG_LIMIT") or extra.get("backlog_limit"), _DEFAULT_BACKLOG_LIMIT),
            100,
        )
        self._backlog_bootstrap_seconds = min(
            _coerce_int(
                os.getenv("FLUXER_BACKLOG_BOOTSTRAP_SECONDS") or extra.get("backlog_bootstrap_seconds"),
                _DEFAULT_BACKLOG_BOOTSTRAP_SECONDS,
            ),
            900,
        )
        self._delivery_verification_enabled = str(
            os.getenv("FLUXER_DELIVERY_VERIFICATION", extra.get("delivery_verification", "true"))
        ).strip().lower() not in {"0", "false", "no", "off"}
        self._allow_mention_everyone = _coerce_bool(
            os.getenv("FLUXER_ALLOW_MENTION_EVERYONE", extra.get("allow_mention_everyone")),
            False,
        )
        self._allow_mention_roles = _coerce_bool(
            os.getenv("FLUXER_ALLOW_MENTION_ROLES", extra.get("allow_mention_roles")),
            False,
        )
        self._allow_mention_users = _coerce_bool(
            os.getenv("FLUXER_ALLOW_MENTION_USERS", extra.get("allow_mention_users")),
            True,
        )
        self._allow_mention_replied_user = _coerce_bool(
            os.getenv("FLUXER_ALLOW_MENTION_REPLIED_USER", extra.get("allow_mention_replied_user")),
            True,
        )
        self._require_mention = _coerce_bool(
            os.getenv("FLUXER_REQUIRE_MENTION", extra.get("require_mention")),
            True,
        )
        self._strict_mention = _coerce_bool(
            os.getenv("FLUXER_STRICT_MENTION", extra.get("strict_mention")),
            False,
        )
        self._free_response_channels = _split_ids(
            os.getenv("FLUXER_FREE_RESPONSE_CHANNELS") or extra.get("free_response_channels")
        )
        self._allowed_channel_ids = _split_ids(
            os.getenv("FLUXER_ALLOWED_CHANNELS") or extra.get("allowed_channels")
        )
        self._mention_patterns = tuple(
            part.strip()
            for part in _split_ids(os.getenv("FLUXER_MENTION_PATTERNS") or extra.get("mention_patterns"))
            if part.strip()
        )
        self._register_native_commands_on_connect = _coerce_bool(
            os.getenv("FLUXER_REGISTER_NATIVE_COMMANDS", extra.get("register_native_commands")),
            False,
        )
        self._application_id = str(os.getenv("FLUXER_APPLICATION_ID") or extra.get("application_id") or "").strip()
        self._native_command_guild_ids = _split_ids(
            os.getenv("FLUXER_NATIVE_COMMAND_GUILDS") or extra.get("native_command_guilds")
        )
        self._mentioned_threads: set[str] = set()
        self._mentioned_threads_max = 5000
        self._home_channel_ids: set[str] = set()
        self._known_channel_ids: set[str] = set()
        home_channel = getattr(config, "home_channel", None)
        if home_channel and getattr(home_channel, "chat_id", None):
            self._home_channel_ids.add(str(home_channel.chat_id))
            self._known_channel_ids.add(str(home_channel.chat_id))
        home_extra = extra.get("home_channel")
        if isinstance(home_extra, dict) and home_extra.get("chat_id"):
            self._home_channel_ids.add(str(home_extra["chat_id"]))
            self._known_channel_ids.add(str(home_extra["chat_id"]))
        home_env = os.getenv("FLUXER_HOME_CHANNEL", "").strip()
        if home_env:
            self._home_channel_ids.add(home_env)
            self._known_channel_ids.add(home_env)
        self._last_disconnect_at: Optional[datetime] = None
        self._ws = None
        self._listener_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        self._last_seq: Optional[int] = None
        self._last_heartbeat_sent_at: Optional[float] = None
        self._last_heartbeat_ack_at: Optional[float] = None
        self._awaiting_heartbeat_ack = False
        self._closing = False
        self._seen_message_ids: set[str] = set()
        self._typing_tasks: Dict[str, asyncio.Task] = {}
        self._pending_exec_approvals: Dict[str, Dict[str, Any]] = {}
        self._pending_component_actions: Dict[str, Dict[str, Any]] = {}

    async def connect(self) -> bool:
        if not self.base_url or not self.bot_token:
            self._set_fatal_error(
                "missing_config",
                "Fluxer requires FLUXER_BOT_TOKEN; FLUXER_BASE_URL is optional and defaults to https://api.fluxer.app/v1",
                retryable=False,
            )
            return False
        try:
            self._closing = False
            self._mark_connected()
            await self._connect_gateway_once()
            await self._maybe_register_native_commands()
            return True
        except Exception as exc:
            self._mark_disconnected()
            logger.warning("Fluxer connect failed: %s", exc)
            self._set_fatal_error("connect_failed", f"Fluxer connect failed: {exc}", retryable=True)
            return False

    async def disconnect(self) -> None:
        self._closing = True
        self._running = False
        for task in (self._heartbeat_task, self._listener_task, self._reconnect_task):
            if task and not task.done():
                task.cancel()
        for task in list(self._typing_tasks.values()):
            if task and not task.done():
                task.cancel()
        self._typing_tasks.clear()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None
        self._heartbeat_task = None
        self._listener_task = None
        self._reconnect_task = None
        self._mark_disconnected()

    async def _connect_gateway_once(self) -> None:
        if not self.gateway_url:
            info = await self._request("GET", "/gateway/bot")
            self.gateway_url = _strip_slash(str(info.get("url") or ""))
        if not self.gateway_url:
            raise RuntimeError("Fluxer gateway URL missing from /gateway/bot")

        import websockets

        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
        current = asyncio.current_task()
        if self._listener_task and self._listener_task is not current and not self._listener_task.done():
            self._listener_task.cancel()
        self._ws = None
        self._heartbeat_task = None
        self._listener_task = None
        self._awaiting_heartbeat_ack = False
        self._last_heartbeat_sent_at = None
        self._last_heartbeat_ack_at = None

        sep = "&" if "?" in self.gateway_url else "?"
        ws_url = f"{self.gateway_url}{sep}v={_GATEWAY_VERSION}&encoding=json"
        reconnect_cutoff = self._last_disconnect_at or (
            datetime.now(tz=timezone.utc) - timedelta(seconds=self._backlog_bootstrap_seconds)
        )
        self._ws = await websockets.connect(ws_url, open_timeout=15, close_timeout=5, max_size=None)
        self._listener_task = asyncio.create_task(self._listen_loop(), name="fluxer-listen")
        logger.info("Fluxer gateway websocket connected")
        await self._recover_backlog(cutoff=reconnect_cutoff)

    def _schedule_reconnect(self, reason: str) -> None:
        if self._closing:
            return
        if self._reconnect_task and not self._reconnect_task.done():
            return
        logger.warning("Fluxer gateway disconnected (%s); scheduling reconnect", reason)
        self._reconnect_task = asyncio.create_task(self._reconnect_loop(reason), name="fluxer-reconnect")

    async def _reconnect_loop(self, reason: str) -> None:
        delay = _RECONNECT_BASE_DELAY
        while not self._closing:
            try:
                await asyncio.sleep(delay)
                await self._connect_gateway_once()
                self._mark_connected()
                logger.info("Fluxer gateway reconnected after %s", reason)
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Fluxer reconnect failed: %s; retrying in %.1fs", exc, delay)
                delay = min(delay * 2, _RECONNECT_MAX_DELAY)

    def _sanitize_outbound_mentions(self, content: str) -> str:
        """Fail closed for global/role pings in model-generated Fluxer output.

        Fluxer message readback exposes mention fields on current deployments,
        but the hosted API is still young. We send an ``allowed_mentions``
        object for compatible servers and also neutralize dangerous ping syntax
        in the content itself so unsupported deployments cannot accidentally
        @everyone a channel from LLM output or echoed user text.
        """
        sanitized = content
        if not self._allow_mention_everyone:
            sanitized = _MENTION_EVERYONE_RE.sub(lambda m: "@\u200b" + m.group(1), sanitized)
        if not self._allow_mention_roles:
            sanitized = _MENTION_ROLE_RE.sub(lambda m: m.group(0).replace("<@&", "<@\u200b&", 1), sanitized)
        if not self._allow_mention_users:
            sanitized = _MENTION_USER_RE.sub(lambda m: m.group(0).replace("<@", "<@\u200b", 1), sanitized)
        return sanitized

    def _allowed_mentions_payload(self) -> Dict[str, Any]:
        parse: List[str] = []
        if self._allow_mention_everyone:
            parse.append("everyone")
        if self._allow_mention_roles:
            parse.append("roles")
        if self._allow_mention_users:
            parse.append("users")
        return {"parse": parse, "replied_user": self._allow_mention_replied_user}

    def _outbound_message_payload(self, content: str, **extra: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "content": self._sanitize_outbound_mentions(content),
            "allowed_mentions": self._allowed_mentions_payload(),
        }
        payload.update(extra)
        return payload

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        payload: Dict[str, Any] = {}
        if reply_to:
            payload["message_reference"] = {"message_id": str(reply_to)}
        if metadata:
            thread_id = metadata.get("thread_id")
            if thread_id and "message_reference" not in payload:
                # Fluxer thread semantics are still stabilizing; keep this as
                # metadata only when callers explicitly provide it.
                payload["message_reference"] = {"message_id": str(thread_id)}
        self._known_channel_ids.add(str(chat_id))

        try:
            formatted = self.format_message(content)
            chunks = self.truncate_message(formatted, MAX_MESSAGE_LENGTH)
            message_ids: List[str] = []
            responses: List[Dict[str, Any]] = []

            for index, chunk in enumerate(chunks):
                chunk_payload = self._outbound_message_payload(chunk, **payload)
                if index > 0:
                    # Reply/reference metadata only belongs on the first split
                    # chunk; applying the same reference to every continuation
                    # creates noisy threads and can make partial retries nastier.
                    chunk_payload.pop("message_reference", None)

                data = await self._request(
                    "POST",
                    f"/channels/{_quote_id(chat_id)}/messages",
                    json=chunk_payload,
                )
                responses.append(data)
                if data.get("id"):
                    message_id = str(data["id"])
                    message_ids.append(message_id)
                    verified = await self._verify_delivery(
                        chat_id,
                        message_id,
                        expected_content=chunk_payload["content"],
                    )
                    if verified is not None:
                        responses[-1] = {**data, "delivery_verified": True, "delivery_readback": verified}
                    else:
                        responses[-1] = {**data, "delivery_verified": False}

            return SendResult(
                success=True,
                message_id=message_ids[0] if message_ids else None,
                raw_response={"message_ids": message_ids, "responses": responses},
            )
        except Exception as exc:
            logger.warning("Fluxer send failed: %s", exc)
            return SendResult(success=False, error=str(exc), retryable=True)

    async def send_exec_approval(
        self,
        chat_id: str,
        command: str,
        session_key: str,
        description: str = "dangerous command",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a native-button dangerous-command approval prompt."""
        target_id = str((metadata or {}).get("thread_id") or chat_id)
        cmd_display = command if len(command) <= 3200 else command[:3197] + "..."
        content = (
            "⚠️ **Command approval required**\n"
            f"```\n{cmd_display}\n```\n"
            f"Reason: {description}\n\n"
            "Use the buttons below, or text fallback: `/approve`, `/approve session`, `/approve always`, or `/deny`."
        )
        nonce = uuid.uuid4().hex[:12]
        button_specs = (
            ("once", "Allow once", 3),
            ("session", "Session", 1),
            ("always", "Always", 2),
            ("deny", "Deny", 4),
        )
        custom_ids = {choice: f"hermes_exec_{nonce}_{choice}" for choice, _label, _style in button_specs}
        components = _fluxer_component_row([
            _fluxer_button(custom_id=custom_ids[choice], label=label, style=style)
            for choice, label, style in button_specs
        ])
        try:
            self._known_channel_ids.add(target_id)
            data = await self._request(
                "POST",
                f"/channels/{_quote_id(target_id)}/messages",
                json=self._outbound_message_payload(self.format_message(content), components=components),
            )
            message_id = str(data.get("id") or "")
            if not message_id:
                return SendResult(success=False, error="Fluxer approval message missing id", retryable=True)

            pending = {
                "session_key": session_key,
                "channel_id": target_id,
                "created_at": time.time(),
                "content": content,
                "resolved": False,
            }
            self._pending_exec_approvals[message_id] = pending
            for choice, custom_id in custom_ids.items():
                self._pending_component_actions[custom_id] = {
                    "kind": "exec_approval",
                    "message_id": message_id,
                    "session_key": session_key,
                    "channel_id": target_id,
                    "choice": choice,
                }
            return SendResult(success=True, message_id=message_id, raw_response=data)
        except Exception as exc:
            logger.warning("Fluxer exec approval prompt failed: %s", exc)
            return SendResult(success=False, error=str(exc), retryable=True)

    async def send_slash_confirm(
        self,
        chat_id: str,
        title: str,
        message: str,
        session_key: str,
        confirm_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a native-button slash-command confirmation prompt."""
        target_id = str((metadata or {}).get("thread_id") or chat_id)
        nonce = uuid.uuid4().hex[:12]
        specs = (
            ("once", "Approve once", 3),
            ("always", "Always approve", 2),
            ("cancel", "Cancel", 4),
        )
        custom_ids = {choice: f"hermes_confirm_{nonce}_{choice}" for choice, _label, _style in specs}
        components = _fluxer_component_row([
            _fluxer_button(custom_id=custom_ids[choice], label=label, style=style)
            for choice, label, style in specs
        ])
        content = f"**{title}**\n{message}"
        try:
            data = await self._request(
                "POST",
                f"/channels/{_quote_id(target_id)}/messages",
                json=self._outbound_message_payload(self.format_message(content), components=components),
            )
            message_id = str(data.get("id") or "")
            for choice, custom_id in custom_ids.items():
                self._pending_component_actions[custom_id] = {
                    "kind": "slash_confirm",
                    "message_id": message_id,
                    "session_key": session_key,
                    "confirm_id": confirm_id,
                    "channel_id": target_id,
                    "choice": choice,
                }
            return SendResult(success=True, message_id=message_id or None, raw_response=data)
        except Exception as exc:
            logger.warning("Fluxer slash confirm prompt failed: %s", exc)
            return SendResult(success=False, error=str(exc), retryable=True)

    async def _add_message_reaction(self, chat_id: str, message_id: str, emoji: str) -> None:
        await self._request(
            "PUT",
            f"/channels/{_quote_id(chat_id)}/messages/{_quote_id(message_id)}/reactions/{quote(emoji, safe='')}/@me",
        )

    def _reaction_user_allowed(self, user_id: str) -> bool:
        if not user_id:
            return False
        if self.bot_user_id and user_id == str(self.bot_user_id):
            return False
        if self._allow_all_users:
            return True
        return user_id in self._allowed_user_ids

    async def _handle_reaction_add(self, data: Dict[str, Any]) -> None:
        message_id = str(data.get("message_id") or "")
        pending = self._pending_exec_approvals.get(message_id)
        if not pending or pending.get("resolved"):
            return

        user_id = str(data.get("user_id") or ((data.get("member") or {}).get("user") or {}).get("id") or "")
        member_user = ((data.get("member") or {}).get("user") or {})
        if member_user.get("bot") or not self._reaction_user_allowed(user_id):
            logger.info("Fluxer approval reaction ignored for unauthorized/bot user %s", user_id or "<missing>")
            return

        emoji_name = _normalize_emoji_name(data.get("emoji"))
        choice = _EXEC_APPROVAL_REACTIONS.get(emoji_name)
        if choice is None:
            return

        pending["resolved"] = True
        self._pending_exec_approvals.pop(message_id, None)
        session_key = str(pending.get("session_key") or "")
        try:
            from tools.approval import resolve_gateway_approval

            count = resolve_gateway_approval(session_key, choice)
            logger.info(
                "Fluxer reaction resolved %d approval(s) for session %s (choice=%s user=%s)",
                count,
                session_key,
                choice,
                user_id,
            )
        except Exception as exc:
            logger.error("Fluxer reaction approval resolve failed: %s", exc)

        label = {
            "once": "approved once",
            "session": "approved for session",
            "always": "approved permanently",
            "deny": "denied",
        }.get(choice, choice)
        try:
            await self.edit_message(
                str(pending.get("channel_id") or data.get("channel_id") or ""),
                message_id,
                f"{pending.get('content') or '⚠️ Command approval required'}\n\nResolved: {label} by <@{user_id}>.",
                finalize=True,
            )
        except Exception:
            pass

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Edit a previously sent Fluxer message.

        Gateway tool-progress bubbles are only enabled for adapters that
        override ``BasePlatformAdapter.edit_message``. Fluxer exposes a PATCH
        message endpoint, so implement it to unlock visible tool
        calling/progress in chat instead of only final responses and status
        callbacks.
        """
        try:
            formatted = self.format_message(content)
            if len(formatted) > MAX_MESSAGE_LENGTH:
                formatted = formatted[: MAX_MESSAGE_LENGTH - 3] + "..."
            data = await self._request(
                "PATCH",
                f"/channels/{_quote_id(chat_id)}/messages/{_quote_id(message_id)}",
                json=self._outbound_message_payload(formatted),
            )
            verified = await self._verify_delivery(
                chat_id,
                str(data.get("id") or message_id),
                expected_content=self._sanitize_outbound_mentions(formatted),
            )
            raw_response = dict(data)
            raw_response["delivery_verified"] = verified is not None
            if verified is not None:
                raw_response["delivery_readback"] = verified
            return SendResult(
                success=True,
                message_id=str(data.get("id") or message_id),
                raw_response=raw_response,
            )
        except Exception as exc:
            logger.warning("Fluxer edit failed for message %s: %s", message_id, exc)
            return SendResult(success=False, error=str(exc), retryable=True)

    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        """Delete a Fluxer message when gateway cleanup/ephemeral flows ask."""
        try:
            await self._request("DELETE", f"/channels/{_quote_id(chat_id)}/messages/{_quote_id(message_id)}")
            return True
        except Exception as exc:
            logger.debug("Fluxer delete failed for message %s: %s", message_id, exc)
            return False

    async def list_pinned_messages(self, chat_id: str) -> List[Dict[str, Any]]:
        """Return pinned messages for a Fluxer channel.

        Fluxer's pin routes are exposed at:
        ``/channels/{channel_id}/messages/pins``. The hosted API currently
        returns ``{"items": [...], "has_more": bool}``, while some deployments
        may return a bare list, so accept both.
        """
        payload = await self._request("GET", f"/channels/{_quote_id(chat_id)}/messages/pins")
        if isinstance(payload, dict):
            items = payload.get("items") or payload.get("messages") or []
            return list(items) if isinstance(items, list) else []
        if isinstance(payload, list):
            return payload
        return []

    async def pin_message(self, chat_id: str, message_id: str) -> bool:
        """Pin a Fluxer message when the server supports message pins."""
        try:
            await self._request("PUT", f"/channels/{_quote_id(chat_id)}/messages/pins/{_quote_id(message_id)}")
            return True
        except Exception as exc:
            logger.debug("Fluxer pin failed for message %s: %s", message_id, exc)
            return False

    async def unpin_message(self, chat_id: str, message_id: str) -> bool:
        """Unpin a Fluxer message when the server supports message pins."""
        try:
            await self._request("DELETE", f"/channels/{_quote_id(chat_id)}/messages/pins/{_quote_id(message_id)}")
            return True
        except Exception as exc:
            logger.debug("Fluxer unpin failed for message %s: %s", message_id, exc)
            return False

    async def _maybe_register_native_commands(self) -> None:
        if not self._register_native_commands_on_connect:
            return
        application_id = self._application_id or self.bot_token.split(".", 1)[0]
        if not application_id:
            logger.warning("Fluxer native command registration enabled but no application id is available")
            return
        guild_ids = sorted(self._native_command_guild_ids)
        try:
            if guild_ids:
                for guild_id in guild_ids:
                    await self.register_native_commands(application_id, guild_id=guild_id)
            else:
                await self.register_native_commands(application_id)
        except Exception as exc:
            logger.warning("Fluxer native command registration failed: %s", exc)

    def _native_command_payload(self, command: Any) -> Dict[str, Any]:
        description = str(getattr(command, "description", "Run Hermes command") or "Run Hermes command")[:100]
        payload: Dict[str, Any] = {
            "name": str(getattr(command, "name", "")).replace("_", "-")[:32],
            "description": description,
            "type": 1,
        }
        args_hint = str(getattr(command, "args_hint", "") or "").strip()
        if args_hint:
            payload["options"] = [{"type": 3, "name": "args", "description": args_hint[:100], "required": False}]
        return payload

    async def register_native_commands(
        self,
        application_id: str,
        *,
        guild_id: Optional[str] = None,
        commands: Optional[List[Any]] = None,
    ) -> Any:
        """Bulk upsert Fluxer application commands using Discord/JFA routes."""
        if commands is None:
            from hermes_cli.commands import COMMAND_REGISTRY

            commands = [cmd for cmd in COMMAND_REGISTRY if not getattr(cmd, "cli_only", False)]
        payloads = [self._native_command_payload(command) for command in commands if getattr(command, "name", None)]
        if guild_id:
            path = f"/applications/{_quote_id(application_id)}/guilds/{_quote_id(guild_id)}/commands"
        else:
            path = f"/applications/{_quote_id(application_id)}/commands"
        return await self._request("PUT", path, json=payloads)

    async def list_channels(self) -> List[Dict[str, Any]]:
        """Enumerate reachable Fluxer guild channels and active threads."""
        try:
            guilds_payload = await self._request("GET", "/users/@me/guilds", params={"limit": 200, "with_counts": True})
        except Exception as exc:
            logger.debug("Fluxer guild enumeration failed: %s", exc)
            return []
        guilds = guilds_payload if isinstance(guilds_payload, list) else (guilds_payload.get("guilds") or guilds_payload.get("items") or [])
        channels: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for guild in guilds:
            if not isinstance(guild, dict):
                continue
            guild_id = str(guild.get("id") or "")
            if not guild_id:
                continue
            guild_name = str(guild.get("name") or guild_id)
            try:
                guild_channels = await self._request("GET", f"/guilds/{_quote_id(guild_id)}/channels")
            except Exception as exc:
                logger.debug("Fluxer channel enumeration failed for guild %s: %s", guild_id, exc)
                guild_channels = []
            for item in guild_channels if isinstance(guild_channels, list) else []:
                if not isinstance(item, dict):
                    continue
                channel_id = str(item.get("id") or "")
                name = str(item.get("name") or "")
                channel_type = _directory_channel_type(item.get("type"))
                if not channel_id or not name or channel_type not in {"channel", "forum", "thread"} or channel_id in seen:
                    continue
                seen.add(channel_id)
                entry: Dict[str, Any] = {"id": channel_id, "name": name, "guild": guild_name, "guild_id": guild_id, "type": channel_type}
                if item.get("parent_id"):
                    entry["parent_id"] = str(item["parent_id"])
                channels.append(entry)
            try:
                threads_payload = await self._request("GET", f"/guilds/{_quote_id(guild_id)}/threads/active", warn_on_error=False)
            except Exception as exc:
                logger.debug("Fluxer active thread enumeration unavailable for guild %s: %s", guild_id, exc)
                threads_payload = {}
            threads = threads_payload.get("threads") if isinstance(threads_payload, dict) else []
            for thread in threads if isinstance(threads, list) else []:
                if not isinstance(thread, dict):
                    continue
                thread_id = str(thread.get("id") or "")
                name = str(thread.get("name") or "")
                if not thread_id or not name or thread_id in seen:
                    continue
                seen.add(thread_id)
                entry = {"id": thread_id, "name": name, "guild": guild_name, "guild_id": guild_id, "type": "thread"}
                if thread.get("parent_id"):
                    entry["parent_id"] = str(thread["parent_id"])
                channels.append(entry)
        for channel_id in self._known_channel_ids:
            if channel_id not in seen:
                channels.append({"id": channel_id, "name": channel_id, "type": "channel"})
        return channels

    async def create_thread(
        self,
        chat_id: str,
        name: str,
        *,
        message_id: Optional[str] = None,
        auto_archive_duration: int = 1440,
        thread_type: int = 11,
    ) -> Dict[str, Any]:
        """Create a Fluxer thread, optionally from an existing message."""
        if message_id:
            payload = {"name": name, "auto_archive_duration": auto_archive_duration, "rate_limit_per_user": 0}
            return await self._request(
                "POST",
                f"/channels/{_quote_id(chat_id)}/messages/{_quote_id(message_id)}/threads",
                json=payload,
            )
        payload = {
            "name": name,
            "type": thread_type,
            "auto_archive_duration": auto_archive_duration,
            "rate_limit_per_user": 0,
            "invitable": True,
        }
        return await self._request("POST", f"/channels/{_quote_id(chat_id)}/threads", json=payload)

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Start a persistent Fluxer typing indicator loop.

        Fluxer typing indicators expire after a short window. Hermes calls
        ``send_typing`` when a turn starts, so keep refreshing until the runner
        calls ``stop_typing`` after the final response. This preserves a
        clean typing-only "I am working" mode when ``display.tool_progress`` is
        set to ``off``.
        """
        chat_key = str(chat_id)
        if chat_key in self._typing_tasks:
            return

        async def _typing_loop() -> None:
            try:
                while True:
                    try:
                        await self._request("POST", f"/channels/{_quote_id(chat_key)}/typing", json={})
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.debug("Fluxer typing indicator failed for %s: %s", chat_key, exc)
                        return
                    await asyncio.sleep(8)
            except asyncio.CancelledError:
                pass
            finally:
                self._typing_tasks.pop(chat_key, None)

        self._typing_tasks[chat_key] = asyncio.create_task(_typing_loop(), name=f"fluxer-typing-{chat_key}")

    async def stop_typing(self, chat_id: str) -> None:
        chat_key = str(chat_id)
        task = self._typing_tasks.pop(chat_key, None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        try:
            cached = await cache_image_from_url(image_url, Path(image_url.split("?", 1)[0]).suffix or ".jpg")
            return await self.send_image_file(chat_id, cached, caption=caption, reply_to=reply_to, metadata=metadata)
        except Exception as exc:
            logger.warning("Fluxer image URL upload failed: %s", exc)
            return SendResult(success=False, error=str(exc), retryable=True)

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        return await self._send_file_message(
            chat_id,
            image_path,
            caption=caption,
            reply_to=reply_to,
            metadata=metadata,
            title=kwargs.get("title"),
        )

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        return await self._send_file_message(
            chat_id,
            file_path,
            caption=caption,
            file_name=file_name,
            reply_to=reply_to,
            metadata=metadata,
            title=kwargs.get("title"),
        )

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        return await self._send_file_message(
            chat_id,
            video_path,
            caption=caption,
            reply_to=reply_to,
            metadata=metadata,
            title=kwargs.get("title"),
        )

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        return await self._send_file_message(
            chat_id,
            audio_path,
            caption=caption,
            reply_to=reply_to,
            metadata=metadata,
            flags=_VOICE_MESSAGE_FLAG,
            is_voice=True,
            duration=kwargs.get("duration"),
            waveform=kwargs.get("waveform"),
            title=kwargs.get("title"),
        )

    async def _send_file_message(
        self,
        chat_id: str,
        file_path: str,
        *,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        flags: int = 0,
        is_voice: bool = False,
        duration: Optional[int] = None,
        waveform: Optional[str] = None,
        title: Optional[str] = None,
    ) -> SendResult:
        resolved = self.validate_media_delivery_path(file_path)
        if not resolved:
            return SendResult(success=False, error=f"Unsafe or missing file path: {file_path}", retryable=False)

        path = Path(resolved)
        filename = file_name or path.name
        payload: Dict[str, Any] = {
            "nonce": str(int(time.time() * 1000)),
            "allowed_mentions": self._allowed_mentions_payload(),
        }
        if caption and not is_voice:
            payload["content"] = self._sanitize_outbound_mentions(caption)
        if reply_to:
            payload["message_reference"] = {"message_id": str(reply_to)}
        if metadata:
            thread_id = metadata.get("thread_id")
            if thread_id and "message_reference" not in payload:
                payload["message_reference"] = {"message_id": str(thread_id)}
        if flags:
            payload["flags"] = flags
        self._known_channel_ids.add(str(chat_id))

        attachment: Dict[str, Any] = {"id": 0, "filename": filename, "title": title or filename}
        if is_voice:
            # Fluxer's schema requires duration + waveform for VOICE_MESSAGE uploads.
            attachment["duration"] = int(duration) if duration is not None else _audio_duration_seconds(path)
            attachment["waveform"] = str(waveform or "AAAA")
        payload["attachments"] = [attachment]

        try:
            data = await self._multipart_request(
                "POST",
                f"/channels/{_quote_id(chat_id)}/messages",
                payload=payload,
                files=[("files[0]", path, filename)],
            )
            message_id = str(data.get("id")) if data.get("id") else None
            verified = await self._verify_delivery(
                chat_id,
                message_id,
                expected_attachment_count=1,
            )
            raw_response = dict(data)
            raw_response["delivery_verified"] = verified is not None
            if verified is not None:
                raw_response["delivery_readback"] = verified
            return SendResult(success=True, message_id=message_id, raw_response=raw_response)
        except Exception as exc:
            logger.warning("Fluxer file upload failed: %s", exc)
            return SendResult(success=False, error=str(exc), retryable=True)

    async def _verify_delivery(
        self,
        chat_id: str,
        message_id: Optional[str],
        *,
        expected_content: Optional[str] = None,
        expected_attachment_count: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Read back a sent/edited message and validate the visible payload.

        Verification is deliberately non-fatal for the caller: a REST timeout or
        stale read after a successful POST may still mean the message landed,
        and retrying could duplicate messages. We attach verification state to
        ``raw_response`` and log mismatches for diagnostics.
        """
        if not self._delivery_verification_enabled or not message_id:
            return None
        try:
            data = await self._request("GET", f"/channels/{_quote_id(chat_id)}/messages/{_quote_id(message_id)}")
        except Exception as exc:
            logger.warning("Fluxer delivery read-back failed for message %s: %s", message_id, exc)
            return None

        problems: List[str] = []
        if str(data.get("id") or "") != str(message_id):
            problems.append("id_mismatch")
        if expected_content is not None and str(data.get("content") or "") != expected_content:
            problems.append("content_mismatch")
        if expected_attachment_count is not None:
            attachments = data.get("attachments") or []
            actual_count = len(attachments) if isinstance(attachments, list) else 0
            if actual_count < expected_attachment_count:
                problems.append(f"attachment_count_mismatch:{actual_count}<{expected_attachment_count}")
        if self.bot_user_id:
            author = data.get("author") or {}
            author_id = str(author.get("id") or data.get("author_id") or "")
            if author_id and author_id != self.bot_user_id:
                problems.append("author_mismatch")
        if problems:
            logger.warning("Fluxer delivery verification warning for %s: %s", message_id, ",".join(problems))
            data = dict(data)
            data["delivery_verification_warnings"] = problems
        return data

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        try:
            data = await self._request("GET", f"/channels/{_quote_id(chat_id)}")
            return {
                "id": str(data.get("id") or chat_id),
                "name": data.get("name") or str(chat_id),
                "type": _chat_type(data.get("type")),
                "raw": data,
            }
        except Exception:
            return {"id": str(chat_id), "name": str(chat_id), "type": "channel"}

    async def _request(self, method: str, path: str, *, warn_on_error: bool = True, **kwargs) -> Dict[str, Any]:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx is required for Fluxer adapter") from exc

        url = urljoin(self.api_base_url + "/", path.lstrip("/"))
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.request(method, url, headers=_headers(self.bot_token), **kwargs)
            if response.status_code >= 400 and warn_on_error:
                logger.warning(
                    "Fluxer REST %s %s failed: status=%s body=%s",
                    method,
                    path,
                    response.status_code,
                    response.text[:500],
                )
            response.raise_for_status()
            if not response.content:
                return {}
            return response.json()

    async def _multipart_request(
        self,
        method: str,
        path: str,
        *,
        payload: Dict[str, Any],
        files: List[tuple[str, Path, str]],
    ) -> Dict[str, Any]:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx is required for Fluxer adapter") from exc

        url = urljoin(self.api_base_url + "/", path.lstrip("/"))
        multipart_files = []
        handles = []
        try:
            for field_name, file_path, filename in files:
                handle = file_path.open("rb")
                handles.append(handle)
                content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
                multipart_files.append((field_name, (filename, handle, content_type)))
            data = {"payload_json": json.dumps(payload, separators=(",", ":"))}
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.request(
                    method,
                    url,
                    headers=_auth_headers(self.bot_token),
                    data=data,
                    files=multipart_files,
                )
                response.raise_for_status()
                if not response.content:
                    return {}
                return response.json()
        finally:
            for handle in handles:
                try:
                    handle.close()
                except Exception:
                    pass

    async def _download_attachment_bytes(self, url: str) -> bytes:
        try:
            import httpx
            from tools.url_safety import is_safe_url
        except ImportError as exc:
            raise RuntimeError("httpx and url safety helpers are required for Fluxer attachments") from exc

        if not is_safe_url(url):
            raise ValueError(f"Blocked unsafe Fluxer attachment URL: {safe_url_for_log(url)}")
        headers = {"User-Agent": "Hermes-Fluxer/0.1", "Accept": "*/*"}
        attachment_host = urlparse(url).netloc.lower()
        api_host = urlparse(self.api_base_url).netloc.lower()
        if attachment_host and api_host and attachment_host == api_host:
            headers["Authorization"] = f"Bot {self.bot_token}"
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            return response.content

    async def _cache_attachment(self, att: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
        url = _attachment_url(att)
        if not url:
            return None, None
        try:
            from tools.url_safety import is_safe_url
        except ImportError as exc:
            raise RuntimeError("url safety helpers are required for Fluxer attachments") from exc
        if not is_safe_url(url):
            logger.warning("Fluxer blocked unsafe attachment URL: %s", safe_url_for_log(url))
            return None, None
        content_type = str(att.get("content_type") or att.get("contentType") or "application/octet-stream").split(";", 1)[0].lower()
        filename = _attachment_filename(att)
        ext = _extension_for_attachment(att, content_type)

        try:
            if content_type.startswith("image/") or ext in SUPPORTED_IMAGE_DOCUMENT_TYPES:
                image_type = content_type if content_type.startswith("image/") else SUPPORTED_IMAGE_DOCUMENT_TYPES.get(ext, "image/jpeg")
                image_ext = ext if ext in SUPPORTED_IMAGE_DOCUMENT_TYPES else _extension_for_attachment(att, image_type, ".jpg")
                return await cache_image_from_url(url, image_ext), image_type
            if content_type.startswith("audio/"):
                return await cache_audio_from_url(url, ext if ext != ".bin" else ".ogg"), content_type

            data = await self._download_attachment_bytes(url)
            return cache_document_from_bytes(data, filename), content_type or "application/octet-stream"
        except Exception as exc:
            logger.warning("Fluxer failed to cache attachment %s; dropping attachment: %s", filename, exc)
            return None, None

    async def _extract_attachments(self, data: Dict[str, Any]) -> tuple[List[str], List[str]]:
        media_urls: List[str] = []
        media_types: List[str] = []
        attachments = data.get("attachments") or []
        if not isinstance(attachments, list):
            return media_urls, media_types
        for att in attachments:
            if not isinstance(att, dict):
                continue
            cached, mtype = await self._cache_attachment(att)
            if cached:
                media_urls.append(cached)
                media_types.append(mtype or "application/octet-stream")
        return media_urls, media_types

    async def _resolve_reply_context(
        self,
        data: Dict[str, Any],
        channel_id: str,
    ) -> tuple[Optional[str], Optional[str]]:
        reply_to_message_id, referenced = _extract_reply_message(data)
        if not reply_to_message_id:
            return None, None

        reply_to_text = _message_content(referenced)
        if reply_to_text:
            return reply_to_message_id, reply_to_text

        # Some gateway events only include ``message_reference``. One REST
        # lookup recovers the text for context injection; failures are non-fatal
        # because the triggering message itself is still valid.
        try:
            payload = await self._request("GET", f"/channels/{_quote_id(channel_id)}/messages/{_quote_id(reply_to_message_id)}")
            reply_to_text = _message_content(payload) or None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("Fluxer reply context lookup failed for %s: %s", reply_to_message_id, exc)
            reply_to_text = None
        return reply_to_message_id, reply_to_text

    def _message_mentions_bot(self, data: Dict[str, Any], text: str) -> bool:
        if self.bot_user_id:
            uid = re.escape(str(self.bot_user_id))
            if re.search(rf"<@!?{uid}>", text or ""):
                return True
            mentions = data.get("mentions") or []
            if isinstance(mentions, list):
                for mention in mentions:
                    if isinstance(mention, dict) and str(mention.get("id") or "") == str(self.bot_user_id):
                        return True
                    if str(mention) == str(self.bot_user_id):
                        return True

        # Human-friendly direct-address fallback for Fluxer instances that do
        # not expose bot mention markup yet. Keep it anchored to the start so
        # "Hermes said..." in the middle of a discussion doesn't wake the bot.
        default_patterns = (r"^\s*@?hermes\b[,:]?",)
        for pattern in (*default_patterns, *self._mention_patterns):
            try:
                if re.search(pattern, text or "", flags=re.IGNORECASE):
                    return True
            except re.error:
                logger.debug("Ignoring invalid Fluxer mention pattern: %s", pattern)
        return False

    def _strip_bot_mention(self, text: str) -> str:
        stripped = text or ""
        if self.bot_user_id:
            stripped = re.sub(rf"<@!?{re.escape(str(self.bot_user_id))}>\s*", "", stripped).strip()
        stripped = re.sub(r"^\s*@?hermes\b[,:]?\s*", "", stripped, flags=re.IGNORECASE)
        for pattern in self._mention_patterns:
            try:
                stripped = re.sub(pattern, "", stripped, count=1, flags=re.IGNORECASE).strip()
            except re.error:
                pass
        return stripped

    def _should_process_message(
        self,
        *,
        channel_id: str,
        chat_type: str,
        text: str,
        data: Dict[str, Any],
        reply_to_message_id: Optional[str],
    ) -> tuple[bool, str]:
        if chat_type == "dm":
            return True, text
        if self._allowed_channel_ids and channel_id not in self._allowed_channel_ids:
            logger.debug("Fluxer ignoring message in non-allowed channel %s", channel_id)
            return False, text
        if channel_id in self._free_response_channels or channel_id in self._home_channel_ids:
            return True, text
        if not self._require_mention:
            return True, text

        is_mentioned = self._message_mentions_bot(data, text)
        thread_key = str(data.get("thread_id") or (channel_id if chat_type == "thread" else "") or reply_to_message_id or "")
        in_mentioned_thread = bool(thread_key and thread_key in self._mentioned_threads)
        if self._strict_mention and not is_mentioned:
            return False, text
        if not is_mentioned and not in_mentioned_thread:
            return False, text
        if is_mentioned:
            text = self._strip_bot_mention(text)
            if thread_key:
                self._mentioned_threads.add(thread_key)
                if len(self._mentioned_threads) > self._mentioned_threads_max:
                    for old in list(self._mentioned_threads)[: self._mentioned_threads_max // 2]:
                        self._mentioned_threads.discard(old)
        return True, text

    async def _recover_backlog(self, *, cutoff: datetime) -> None:
        if not self._backlog_enabled or not self._known_channel_ids:
            return
        cutoff = cutoff.astimezone(timezone.utc)
        for channel_id in sorted(self._known_channel_ids):
            try:
                payload = await self._request(
                    "GET",
                    f"/channels/{_quote_id(channel_id)}/messages",
                    params={"limit": self._backlog_limit},
                )
                messages = _extract_message_list(payload)
                recovered = 0
                for message in reversed(messages):
                    ts = _parse_fluxer_timestamp(message.get("timestamp") or message.get("created_at"))
                    if ts is not None and ts < cutoff:
                        continue
                    if not message.get("channel_id"):
                        message = {**message, "channel_id": channel_id}
                    msg_id = str(message.get("id") or "")
                    if msg_id and msg_id in self._seen_message_ids:
                        continue
                    await self._handle_message_create(
                        message,
                        {"op": 0, "t": "MESSAGE_CREATE", "d": message, "backfill": True},
                    )
                    recovered += 1
                if recovered:
                    logger.info("Fluxer recovered %d backlog message(s) for channel %s", recovered, channel_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Fluxer backlog recovery failed for channel %s: %s", channel_id, exc)

    async def _listen_loop(self) -> None:
        assert self._ws is not None
        reason = "closed"
        try:
            async for raw in self._ws:
                payload = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode("utf-8"))
                await self._handle_gateway_dispatch(payload)
            reason = "websocket closed"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._running:
                logger.warning("Fluxer listener stopped: %s", exc)
            reason = str(exc)
        finally:
            if not self._closing:
                self._last_disconnect_at = datetime.now(tz=timezone.utc)
                self._mark_disconnected()
                self._schedule_reconnect(reason)

    async def _send_heartbeat(self) -> None:
        if self._ws is None:
            return
        self._last_heartbeat_sent_at = time.monotonic()
        self._awaiting_heartbeat_ack = True
        await self._ws.send(json.dumps({"op": 1, "d": self._last_seq}))

    async def _heartbeat_loop(self, interval_ms: int) -> None:
        interval_s = max(interval_ms, 1000) / 1000
        ack_timeout_s = interval_s * _HEARTBEAT_ACK_TIMEOUT_FACTOR
        try:
            while not self._closing and self._ws is not None:
                await asyncio.sleep(interval_s)
                if (
                    self._awaiting_heartbeat_ack
                    and self._last_heartbeat_sent_at is not None
                    and time.monotonic() - self._last_heartbeat_sent_at > ack_timeout_s
                ):
                    logger.warning("Fluxer heartbeat ACK timed out; closing websocket to trigger reconnect")
                    try:
                        await self._ws.close()
                    except Exception:
                        pass
                    return
                await self._send_heartbeat()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("Fluxer heartbeat stopped: %s", exc)

    def _interaction_user_id(self, data: Dict[str, Any]) -> str:
        user = data.get("user") or ((data.get("member") or {}).get("user") or {})
        return str(user.get("id") or "")

    async def _respond_to_interaction(self, data: Dict[str, Any], *, content: str, components: Optional[List[Dict[str, Any]]] = None) -> None:
        interaction_id = str(data.get("id") or "")
        token = str(data.get("token") or "")
        if not interaction_id or not token:
            return
        payload: Dict[str, Any] = {"type": 7, "data": self._outbound_message_payload(content)}
        if components is not None:
            payload["data"]["components"] = components
        await self._request("POST", f"/interactions/{_quote_id(interaction_id)}/{_quote_id(token)}/callback", json=payload)

    async def _handle_interaction_create(self, data: Dict[str, Any]) -> None:
        interaction_type = data.get("type")
        interaction_data = data.get("data") or {}
        if interaction_type == 3 or interaction_data.get("custom_id"):
            await self._handle_component_interaction(data)
            return
        if interaction_type == 2:
            await self._handle_application_command_interaction(data)

    async def _handle_component_interaction(self, data: Dict[str, Any]) -> None:
        interaction_data = data.get("data") or {}
        custom_id = str(interaction_data.get("custom_id") or "")
        action = self._pending_component_actions.get(custom_id)
        if not action:
            return
        user_id = self._interaction_user_id(data)
        if not self._reaction_user_allowed(user_id):
            logger.info("Fluxer component action ignored for unauthorized user %s", user_id or "<missing>")
            return
        interaction_message_id = str(((data.get("message") or {}).get("id")) or "")
        action_message_id = str(action.get("message_id") or "")
        if action_message_id and interaction_message_id and interaction_message_id != action_message_id:
            logger.warning("Fluxer component action %s ignored for mismatched message %s", custom_id, interaction_message_id)
            return
        interaction_channel_id = str(data.get("channel_id") or ((data.get("channel") or {}).get("id")) or "")
        action_channel_id = str(action.get("channel_id") or "")
        if action_channel_id and interaction_channel_id and interaction_channel_id != action_channel_id:
            logger.warning("Fluxer component action %s ignored for mismatched channel %s", custom_id, interaction_channel_id)
            return
        self._pending_component_actions.pop(custom_id, None)
        if action.get("kind") == "exec_approval":
            await self._resolve_exec_approval_action(action, user_id=user_id, interaction=data)
        elif action.get("kind") == "slash_confirm":
            await self._resolve_slash_confirm_action(action, user_id=user_id, interaction=data)

    async def _resolve_exec_approval_action(self, action: Dict[str, Any], *, user_id: str, interaction: Dict[str, Any]) -> None:
        message_id = str(action.get("message_id") or ((interaction.get("message") or {}).get("id")) or "")
        pending = self._pending_exec_approvals.pop(message_id, None)
        if not pending or pending.get("resolved"):
            return
        pending["resolved"] = True
        session_key = str(action.get("session_key") or pending.get("session_key") or "")
        choice = str(action.get("choice") or "deny")
        try:
            from tools.approval import resolve_gateway_approval

            count = resolve_gateway_approval(session_key, choice)
            logger.info("Fluxer component resolved %d approval(s) for session %s (choice=%s user=%s)", count, session_key, choice, user_id)
        except Exception as exc:
            logger.error("Fluxer component approval resolve failed: %s", exc)
        for cid, state in list(self._pending_component_actions.items()):
            if state.get("message_id") == message_id:
                self._pending_component_actions.pop(cid, None)
        label = {"once": "approved once", "session": "approved for session", "always": "approved permanently", "deny": "denied"}.get(choice, choice)
        try:
            await self._respond_to_interaction(interaction, content=f"{pending.get('content') or '⚠️ Command approval required'}\n\nResolved: {label} by <@{user_id}>.", components=[])
        except Exception as exc:
            logger.debug("Fluxer component approval response failed: %s", exc)

    async def _resolve_slash_confirm_action(self, action: Dict[str, Any], *, user_id: str, interaction: Dict[str, Any]) -> None:
        session_key = str(action.get("session_key") or "")
        confirm_id = str(action.get("confirm_id") or "")
        choice = str(action.get("choice") or "cancel")
        try:
            from tools.slash_confirm import resolve

            await resolve(session_key, confirm_id, choice)
        except Exception as exc:
            logger.error("Fluxer slash confirm resolve failed: %s", exc)
        try:
            await self._respond_to_interaction(interaction, content=f"Slash confirmation: {choice} by <@{user_id}>.", components=[])
        except Exception:
            pass

    async def _handle_application_command_interaction(self, data: Dict[str, Any]) -> None:
        interaction_data = data.get("data") or {}
        name = str(interaction_data.get("name") or "").strip()
        if not name:
            return
        options = interaction_data.get("options") or []
        parts = [f"/{name}"]
        if isinstance(options, list):
            for option in options:
                if isinstance(option, dict) and option.get("value") is not None:
                    parts.append(str(option.get("value")))
        text = " ".join(parts).strip()
        user = data.get("user") or ((data.get("member") or {}).get("user") or {})
        channel = data.get("channel") or {}
        channel_id = str(data.get("channel_id") or channel.get("id") or "")
        if not channel_id:
            return
        source = self.build_source(
            chat_id=channel_id,
            chat_name=channel.get("name"),
            chat_type=_chat_type(channel.get("type")),
            user_id=str(user.get("id") or "") or None,
            user_name=_author_name(user),
            guild_id=data.get("guild_id"),
            message_id=str(data.get("id") or "") or None,
        )
        await self._request("POST", f"/interactions/{_quote_id(data.get('id'))}/{_quote_id(data.get('token'))}/callback", json={"type": 5, "data": {"flags": 64}})
        await self.handle_message(MessageEvent(text=text, message_type=MessageType.TEXT, source=source, raw_message={"interaction": data}, message_id=str(data.get("id") or "") or None))

    async def _handle_gateway_dispatch(self, payload: Dict[str, Any]) -> None:
        op = payload.get("op")
        self._last_seq = _event_seq(payload) or self._last_seq

        if op == 10:  # HELLO
            interval = int(((payload.get("d") or {}).get("heartbeat_interval") or 41250))
            if self._ws is not None:
                await self._ws.send(json.dumps(_build_identify_payload(self.bot_token)))
                # Fluxer's hosted gateway also tolerates/expects an
                # immediate heartbeat; waiting a full interval can trip hosted
                # gateway 4009 heartbeat-timeout closes during dogfood sessions.
                await self._send_heartbeat()
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(interval), name="fluxer-heartbeat")
            return
        if op == 1:  # server heartbeat request
            await self._send_heartbeat()
            return
        if op == 11:  # HEARTBEAT_ACK
            self._awaiting_heartbeat_ack = False
            self._last_heartbeat_ack_at = time.monotonic()
            return
        if op in {7, 9, 12}:  # RECONNECT / INVALID_SESSION / GATEWAY_ERROR
            logger.warning("Fluxer gateway requested reconnect: op=%s payload=%s", op, payload.get("d"))
            if self._ws is not None:
                await self._ws.close()
            return
        if op != 0:  # not DISPATCH
            return

        event_name = payload.get("t")
        data = payload.get("d") or {}
        if event_name == "READY":
            user = data.get("user") or data.get("bot") or {}
            if user.get("id"):
                self.bot_user_id = str(user["id"])
            return
        if event_name == "MESSAGE_REACTION_ADD":
            await self._handle_reaction_add(data)
            return
        if event_name == "INTERACTION_CREATE":
            await self._handle_interaction_create(data)
            return
        if event_name in {"THREAD_CREATE", "THREAD_UPDATE", "CHANNEL_CREATE", "CHANNEL_UPDATE"}:
            channel_id = str(data.get("id") or "")
            if channel_id:
                self._known_channel_ids.add(channel_id)
            return
        if event_name == "MESSAGE_DELETE":
            await self._handle_message_delete(data)
            return
        if event_name == "MESSAGE_UPDATE":
            await self._handle_message_update(data)
            return
        if event_name != "MESSAGE_CREATE":
            return
        await self._handle_message_create(data, payload)

    async def _handle_message_update(self, data: Dict[str, Any]) -> None:
        """Track Fluxer edits that affect adapter-owned state.

        We intentionally do not re-dispatch edited user messages as fresh user
        turns; that would surprise users and can duplicate agent runs. For now
        MESSAGE_UPDATE only keeps pending approval bookkeeping in sync so a
        later reaction edits the current prompt text rather than stale content.
        """
        message_id = str(data.get("id") or data.get("message_id") or "")
        if not message_id:
            return
        pending = self._pending_exec_approvals.get(message_id)
        if pending is None:
            return
        content = data.get("content")
        if isinstance(content, str) and content:
            pending["content"] = content

    async def _handle_message_delete(self, data: Dict[str, Any]) -> None:
        """Fail closed if an outstanding approval prompt is deleted."""
        message_id = str(data.get("id") or data.get("message_id") or "")
        if not message_id:
            return
        self._seen_message_ids.discard(message_id)
        pending = self._pending_exec_approvals.pop(message_id, None)
        if not pending or pending.get("resolved"):
            return
        pending["resolved"] = True
        session_key = str(pending.get("session_key") or "")
        try:
            from tools.approval import resolve_gateway_approval

            count = resolve_gateway_approval(session_key, "deny")
            logger.info(
                "Fluxer deleted approval prompt denied %d approval(s) for session %s",
                count,
                session_key,
            )
        except Exception as exc:
            logger.error("Fluxer deleted approval prompt deny failed: %s", exc)

    async def _handle_message_create(self, data: Dict[str, Any], raw_payload: Dict[str, Any]) -> None:
        msg_id = str(data.get("id") or "")
        if msg_id:
            if msg_id in self._seen_message_ids:
                return
            self._seen_message_ids.add(msg_id)
            if len(self._seen_message_ids) > 2000:
                self._seen_message_ids = set(list(self._seen_message_ids)[-1000:])

        author = data.get("author") or data.get("user") or {}
        author_id = str(author.get("id") or data.get("author_id") or "")
        if author.get("bot") or (self.bot_user_id and author_id == str(self.bot_user_id)):
            return

        text = data.get("content") or ""
        media_urls, media_types = await self._extract_attachments(data)
        if not text and not media_urls:
            return

        channel_id = str(data.get("channel_id") or data.get("channel", {}).get("id") or "")
        if not channel_id:
            return
        self._known_channel_ids.add(channel_id)
        reply_to_message_id, reply_to_text = await self._resolve_reply_context(data, channel_id)
        chat_type = _chat_type(data.get("channel_type") or (data.get("channel") or {}).get("type"))
        should_process, text = self._should_process_message(
            channel_id=channel_id,
            chat_type=chat_type,
            text=text,
            data=data,
            reply_to_message_id=reply_to_message_id,
        )
        if not should_process:
            return
        source = self.build_source(
            chat_id=channel_id,
            chat_name=(data.get("channel") or {}).get("name"),
            chat_type=chat_type,
            user_id=author_id or None,
            user_name=_author_name(author),
            guild_id=data.get("guild_id"),
            message_id=msg_id or None,
        )

        timestamp = datetime.now(tz=timezone.utc)
        ts_raw = data.get("timestamp") or data.get("created_at")
        if isinstance(ts_raw, str):
            try:
                timestamp = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except ValueError:
                pass

        event = MessageEvent(
            text=text,
            message_type=MessageType.VOICE if _is_voice_message(data) else _message_type_for_media(media_types),
            source=source,
            raw_message=raw_payload,
            message_id=msg_id or None,
            media_urls=media_urls,
            media_types=media_types,
            reply_to_message_id=reply_to_message_id,
            reply_to_text=reply_to_text,
            timestamp=timestamp,
        )
        await self.handle_message(event)


def check_requirements() -> bool:
    if not os.getenv("FLUXER_BOT_TOKEN"):
        return False
    try:
        import httpx  # noqa: F401
        import websockets  # noqa: F401
    except ImportError:
        return False
    return True


def validate_config(config) -> bool:
    extra = getattr(config, "extra", {}) or {}
    token = os.getenv("FLUXER_BOT_TOKEN") or extra.get("bot_token", "")
    return bool(str(token).strip())


def is_connected(config) -> bool:
    return validate_config(config)


def _env_enablement() -> dict | None:
    base_url = os.getenv("FLUXER_BASE_URL", "").strip()
    token = os.getenv("FLUXER_BOT_TOKEN", "").strip()
    if not token:
        return None
    seed: dict = {"bot_token": token}
    if base_url:
        seed["base_url"] = base_url
    gateway_url = os.getenv("FLUXER_GATEWAY_URL", "").strip()
    if gateway_url:
        seed["gateway_url"] = gateway_url
    home = os.getenv("FLUXER_HOME_CHANNEL", "").strip()
    if home:
        seed["home_channel"] = {
            "chat_id": home,
            "name": os.getenv("FLUXER_HOME_CHANNEL_NAME", "").strip() or home,
        }
    return seed


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[List[str]] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    adapter = FluxerAdapter(pconfig)
    metadata = {"thread_id": thread_id} if thread_id else None
    try:
        last: Optional[SendResult] = None
        if message:
            last = await adapter.send(chat_id, message, metadata=metadata)
            if not last.success:
                return {"error": last.error or "Fluxer send failed"}
        for media_item in media_files or []:
            if isinstance(media_item, (tuple, list)):
                media_path = str(media_item[0])
                is_voice_directive = bool(media_item[1]) if len(media_item) > 1 else False
            else:
                media_path = str(media_item)
                is_voice_directive = False
            ext = Path(media_path).suffix.lower()
            if not force_document and ext in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
                last = await adapter.send_image_file(chat_id, media_path, metadata=metadata)
            elif not force_document and ext in {".mp4", ".mov", ".webm", ".mkv", ".avi"}:
                last = await adapter.send_video(chat_id, media_path, metadata=metadata)
            elif not force_document and (is_voice_directive or ext in {".mp3", ".m4a", ".ogg", ".opus", ".wav", ".flac", ".aac"}):
                last = await adapter.send_voice(chat_id, media_path, metadata=metadata)
            else:
                last = await adapter.send_document(chat_id, media_path, metadata=metadata)
            if not last.success:
                return {"error": last.error or "Fluxer media send failed"}
        if last and last.success:
            return {"success": True, "platform": "fluxer", "chat_id": chat_id, "message_id": last.message_id}
        return {"error": "Fluxer send failed: empty message and no media"}
    except Exception as exc:
        return {"error": str(exc)}


def interactive_setup() -> None:
    print("Fluxer platform setup")
    print("Set FLUXER_BOT_TOKEN in ~/.hermes/.env, then restart the gateway.")
    print("Optional: set FLUXER_BASE_URL for self-hosted Fluxer; official hosted defaults to https://api.fluxer.app/v1.")


def register(ctx) -> None:
    ctx.register_platform(
        name="fluxer",
        label="Fluxer",
        adapter_factory=lambda cfg: FluxerAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["FLUXER_BOT_TOKEN"],
        install_hint="pip install httpx websockets   # Fluxer adapter dependencies",
        setup_fn=interactive_setup,
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="FLUXER_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        allowed_users_env="FLUXER_ALLOWED_USERS",
        allow_all_env="FLUXER_ALLOW_ALL_USERS",
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji="⚡",
        pii_safe=False,
        allow_update_command=True,
        platform_hint=(
            "You are chatting via Fluxer, a Discord-like open-source chat "
            "platform. Fluxer supports rich Markdown, channels, DMs, files, "
            "and voice/video. Prefer normal Markdown for structured replies."
        ),
    )
