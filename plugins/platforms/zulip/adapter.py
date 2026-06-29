"""Zulip gateway adapter.

Connects Hermes Agent to Zulip through the official REST API and event queue.
Zulip topics are treated as Hermes threads. A channel mention activates the
current Zulip topic, so the bot can continue there without requiring a new
@mention for every follow-up.

Environment variables:
    ZULIP_URL                   Server URL (e.g. https://chat.example.com)
    ZULIP_BOT_EMAIL             Bot email address
    ZULIP_API_KEY               Bot API key
    ZULIP_ALLOWED_USERS         Comma-separated user IDs or emails
    ZULIP_ALLOWED_GROUPS        Comma-separated user group names or IDs
    ZULIP_HOME_CHANNEL          Stream/channel name for cron/notifications
    ZULIP_HOME_TOPIC            Topic for cron/notifications
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import ssl
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    resolve_channel_prompt,
)
from gateway.platforms.helpers import MessageDeduplicator

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 9000
_EVENT_BACKOFF_SECONDS = 3.0
_GROUP_CACHE_TTL_SECONDS = 60.0

_APPROVAL_REACTIONS = {
    "check": "once",
    "clock": "session",
    "hourglass": "session",
    "infinity": "always",
    "x": "deny",
    "cross_mark": "deny",
}


@dataclass
class _ZulipApprovalPrompt:
    session_key: str
    chat_id: str
    message_id: str
    topic: Optional[str] = None
    requester_user_id: Optional[str] = None
    expires_at: float = 0.0
    resolved: bool = False
    bot_reactions: list[str] = field(default_factory=list)


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


def _csv_set(value: Any) -> set[str]:
    if isinstance(value, list):
        return {str(v).strip() for v in value if str(v).strip()}
    return {v.strip() for v in str(value or "").split(",") if v.strip()}


def _stream_name(message: Dict[str, Any]) -> str:
    recipient = message.get("display_recipient")
    if isinstance(recipient, str) and recipient:
        return recipient
    return str(message.get("stream") or message.get("stream_id") or "")


def _message_topic(message: Dict[str, Any]) -> Optional[str]:
    topic = message.get("topic")
    if topic is None:
        topic = message.get("subject")
    topic = str(topic or "").strip()
    return topic or None


def check_zulip_requirements() -> bool:
    """Return True if the Zulip adapter can be used."""
    if not os.getenv("ZULIP_URL", "").strip():
        logger.warning("Zulip: ZULIP_URL not set")
        return False
    if not os.getenv("ZULIP_BOT_EMAIL", "").strip():
        logger.warning("Zulip: ZULIP_BOT_EMAIL not set")
        return False
    if not os.getenv("ZULIP_API_KEY", "").strip():
        logger.debug("Zulip: ZULIP_API_KEY not set")
        return False
    try:
        import aiohttp  # noqa: F401
        return True
    except ImportError:
        logger.warning("Zulip: aiohttp not installed")
        return False


class ZulipAdapter(BasePlatformAdapter):
    """Gateway adapter for Zulip."""

    splits_long_messages = True

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform("zulip"))
        self._base_url = (
            config.extra.get("url", "") if config.extra else ""
        ) or os.getenv("ZULIP_URL", "")
        self._base_url = self._base_url.rstrip("/")
        self._bot_email = (
            config.extra.get("bot_email", "") if config.extra else ""
        ) or os.getenv("ZULIP_BOT_EMAIL", "")
        self._api_key = config.token or os.getenv("ZULIP_API_KEY", "")

        self._session: Any = None
        self._auth: Any = None
        self._ssl_context: Any = None
        self._queue_id = ""
        self._last_event_id = -1
        self._event_task: Optional[asyncio.Task] = None
        self._closing = False

        self._bot_user_id = ""
        self._bot_full_name = ""
        self._bot_mention_ids: set[str] = set()
        self._stream_ids_by_name: dict[str, str] = {}
        self._active_channel_topics: set[tuple[str, str]] = set()
        self._allowed_groups = _csv_set(
            config.extra.get("allowed_groups") if config.extra else None
        ) or _csv_set(os.getenv("ZULIP_ALLOWED_GROUPS", ""))
        self._allowed_group_user_ids: set[str] = set()
        self._allowed_groups_loaded_at = 0.0

        self._dedup = MessageDeduplicator()
        self._approval_timeout_seconds = float(
            os.getenv("ZULIP_APPROVAL_TIMEOUT_SECONDS", "300") or "300"
        )
        self._approval_require_sender = _truthy(
            os.getenv("ZULIP_APPROVAL_REQUIRE_SENDER", "true"), True
        )
        self._approval_prompts_by_message: dict[str, _ZulipApprovalPrompt] = {}
        self._approval_prompt_by_session: dict[str, str] = {}

    def _build_ssl_context(self) -> Any:
        ca_cert = os.getenv("ZULIP_CA_CERT", "").strip()
        if not ca_cert:
            return None
        context = ssl.create_default_context(cafile=ca_cert)
        return context

    async def _api_request(
        self,
        method: str,
        path: str,
        *,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        import aiohttp

        if ".." in path:
            logger.error("Zulip API path traversal blocked: %s", path)
            return {}
        url = f"{self._base_url}/api/v1/{path.lstrip('/')}"
        try:
            async with self._session.request(
                method,
                url,
                auth=self._auth,
                data=data,
                params=params,
                ssl=self._ssl_context,
                timeout=aiohttp.ClientTimeout(total=90),
            ) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    logger.error("Zulip API %s %s -> %s: %s", method, path, resp.status, text[:400])
                    return {}
                if not text:
                    return {}
                return json.loads(text)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.warning("Zulip API %s %s network error: %s", method, path, exc)
            return {}
        except json.JSONDecodeError as exc:
            logger.warning("Zulip API %s %s invalid JSON: %s", method, path, exc)
            return {}

    async def _api_get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return await self._api_request("GET", path, params=params)

    async def _api_post(self, path: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return await self._api_request("POST", path, data=data)

    async def _register_queue(self) -> bool:
        data = {
            "event_types": json.dumps(["message", "reaction"]),
            "apply_markdown": "false",
            "client_gravatar": "false",
            "all_public_streams": str(_truthy(os.getenv("ZULIP_ALL_PUBLIC_STREAMS", "true"), True)).lower(),
        }
        result = await self._api_post("register", data=data)
        if result.get("result") != "success":
            return False
        self._queue_id = str(result.get("queue_id") or "")
        self._last_event_id = int(result.get("last_event_id", -1))
        return bool(self._queue_id)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Authenticate and start Zulip event polling."""
        import aiohttp

        if not self._base_url or not self._bot_email or not self._api_key:
            logger.error("Zulip: ZULIP_URL, ZULIP_BOT_EMAIL, and ZULIP_API_KEY are required")
            return False

        self._auth = aiohttp.BasicAuth(self._bot_email, self._api_key)
        self._ssl_context = self._build_ssl_context()
        self._session = aiohttp.ClientSession()
        self._closing = False

        me = await self._api_get("users/me")
        if me.get("result") != "success":
            logger.error("Zulip: authentication failed")
            await self._session.close()
            return False

        self._bot_user_id = str(me.get("user_id") or "")
        self._bot_full_name = str(me.get("full_name") or "Hermes")
        self._bot_mention_ids = {self._bot_user_id, self._bot_email.lower(), self._bot_full_name.lower()}

        if not await self._register_queue():
            logger.error("Zulip: failed to register event queue")
            await self._session.close()
            return False

        self._event_task = asyncio.create_task(self._event_loop())
        self._mark_connected()
        logger.info("Zulip: connected as %s (%s)", self._bot_email, self._bot_user_id)
        return True

    async def _stream_id_for_name(self, stream_name: str) -> Optional[str]:
        if not stream_name:
            return None
        cached = self._stream_ids_by_name.get(stream_name)
        if cached:
            return cached
        result = await self._api_get("users/me/subscriptions")
        if result.get("result") != "success":
            return None
        for sub in result.get("subscriptions") or []:
            name = str(sub.get("name") or "")
            stream_id = sub.get("stream_id")
            if name and stream_id is not None:
                self._stream_ids_by_name[name] = str(stream_id)
        return self._stream_ids_by_name.get(stream_name)

    async def _refresh_allowed_group_members(self, *, force: bool = False) -> None:
        if not self._allowed_groups:
            self._allowed_group_user_ids = set()
            return
        now = time.monotonic()
        if (
            not force
            and self._allowed_groups_loaded_at
            and now - self._allowed_groups_loaded_at < _GROUP_CACHE_TTL_SECONDS
        ):
            return
        result = await self._api_get("user_groups")
        if result.get("result") != "success":
            logger.warning("Zulip: failed to refresh allowed user groups")
            self._allowed_group_user_ids = set()
            self._allowed_groups_loaded_at = now
            return

        groups = result.get("user_groups") or []
        by_id = {str(g.get("id")): g for g in groups if g.get("id") is not None}
        by_name = {str(g.get("name") or "").strip().lower(): g for g in groups if g.get("name")}
        wanted = {str(g).strip() for g in self._allowed_groups if str(g).strip()}
        wanted_lower = {g.lower() for g in wanted}

        selected_ids: set[str] = set()
        resolved_entries: set[str] = set()
        resolved_lower_entries: set[str] = set()
        for raw in wanted:
            if raw in by_id:
                selected_ids.add(raw)
                resolved_entries.add(raw)
        for raw in wanted_lower:
            group = by_name.get(raw)
            if group and group.get("id") is not None:
                selected_ids.add(str(group["id"]))
                resolved_lower_entries.add(raw)

        resolved_users: set[str] = set()
        seen_group_ids: set[str] = set()

        def collect(group_id: str) -> None:
            if group_id in seen_group_ids:
                return
            seen_group_ids.add(group_id)
            group = by_id.get(group_id)
            if not group:
                return
            for member_id in group.get("members") or []:
                resolved_users.add(str(member_id))
            for subgroup_id in group.get("direct_subgroup_ids") or []:
                collect(str(subgroup_id))

        for group_id in selected_ids:
            collect(group_id)

        missing = sorted(
            raw for raw in wanted
            if raw not in resolved_entries and raw.lower() not in resolved_lower_entries
        )
        if missing:
            logger.warning("Zulip: allowed user group(s) not found: %s", ", ".join(missing))
        self._allowed_group_user_ids = resolved_users
        self._allowed_groups_loaded_at = now

    async def _sender_in_allowed_group(self, sender_id: str) -> bool:
        if not self._allowed_groups or not sender_id:
            return False
        await self._refresh_allowed_group_members()
        return str(sender_id) in self._allowed_group_user_ids

    async def disconnect(self) -> None:
        """Disconnect from Zulip."""
        self._closing = True
        if self._event_task and not self._event_task.done():
            self._event_task.cancel()
            try:
                await self._event_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._queue_id:
            try:
                await self._api_request("DELETE", f"events?queue_id={self._queue_id}")
            except Exception:
                pass
        if self._session and not self._session.closed:
            await self._session.close()

    async def _event_loop(self) -> None:
        while not self._closing:
            params = {
                "queue_id": self._queue_id,
                "last_event_id": str(self._last_event_id),
                "dont_block": "false",
            }
            result = await self._api_get("events", params=params)
            if not result:
                await asyncio.sleep(_EVENT_BACKOFF_SECONDS)
                continue
            if result.get("code") == "BAD_EVENT_QUEUE_ID":
                if not await self._register_queue():
                    await asyncio.sleep(_EVENT_BACKOFF_SECONDS)
                continue
            for event in result.get("events", []):
                try:
                    self._last_event_id = max(self._last_event_id, int(event.get("id", self._last_event_id)))
                    await self._handle_event(event)
                except Exception as exc:
                    logger.warning("Zulip: failed to handle event: %s", exc, exc_info=True)

    async def _handle_event(self, event: Dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "message":
            await self._handle_message_event(event.get("message") or {})
        elif event_type == "reaction":
            await self._handle_reaction_event(event)

    def _message_has_mention(self, message: Dict[str, Any], text: str) -> bool:
        flags = set(message.get("flags") or [])
        if "mentioned" in flags or "wildcard_mentioned" in flags:
            return True
        lowered = text.lower()
        patterns = [
            rf"@\*\*{re.escape(self._bot_full_name.lower())}\*\*",
            rf"@{re.escape(self._bot_full_name.lower())}",
            rf"@{re.escape(self._bot_email.lower())}",
            rf"@hermes",
        ]
        return any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in patterns)

    def _strip_mentions(self, text: str) -> str:
        cleaned = text
        mention_patterns = [
            rf"@\*\*{re.escape(self._bot_full_name)}\*\*",
            rf"@{re.escape(self._bot_full_name)}",
            rf"@{re.escape(self._bot_email)}",
            r"@Hermes",
        ]
        for pattern in mention_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def _topic_key(self, chat_id: str, topic: Optional[str]) -> tuple[str, str]:
        return (str(chat_id), str(topic or ""))

    async def _handle_message_event(self, message: Dict[str, Any]) -> None:
        message_id = str(message.get("id") or "")
        if not message_id or self._dedup.is_duplicate(message_id):
            return
        if str(message.get("sender_id") or "") == self._bot_user_id:
            return
        if message.get("sender_is_bot"):
            return

        text = str(message.get("content") or "").strip()
        msg_type_raw = message.get("type")
        sender_id = str(message.get("sender_id") or "")
        sender_email = str(message.get("sender_email") or "").lower()
        sender_name = str(message.get("sender_full_name") or sender_email or sender_id)
        role_authorized = await self._sender_in_allowed_group(sender_id)

        if msg_type_raw == "private":
            chat_id = f"dm:{sender_id}"
            chat_type = "dm"
            topic = None
        else:
            chat_id = _stream_name(message)
            chat_type = "channel"
            topic = _message_topic(message)

            allowed_raw = self.config.extra.get("allowed_channels") if self.config.extra else None
            if allowed_raw is None:
                allowed_raw = os.getenv("ZULIP_ALLOWED_CHANNELS", "")
            allowed_channels = _csv_set(allowed_raw)
            if allowed_channels and chat_id not in allowed_channels:
                return

            free_raw = self.config.extra.get("free_response_channels") if self.config.extra else None
            if free_raw is None:
                free_raw = os.getenv("ZULIP_FREE_RESPONSE_CHANNELS", "")
            free_channels = _csv_set(free_raw)
            require_mention = _truthy(os.getenv("ZULIP_REQUIRE_MENTION", "true"), True)
            has_mention = self._message_has_mention(message, text)
            is_gateway_command = text.startswith("/")
            topic_is_active = self._topic_key(chat_id, topic) in self._active_channel_topics
            if require_mention and chat_id not in free_channels and not has_mention and not topic_is_active and not is_gateway_command:
                return
            if has_mention:
                text = self._strip_mentions(text)
                self._active_channel_topics.add(self._topic_key(chat_id, topic))

        if not text:
            return

        source = self.build_source(
            chat_id=chat_id,
            chat_name=chat_id,
            chat_type=chat_type,
            user_id=sender_id,
            user_name=sender_name,
            user_id_alt=sender_email or None,
            thread_id=topic,
            chat_topic=topic,
            message_id=message_id,
            role_authorized=role_authorized,
        )

        channel_prompt = resolve_channel_prompt(self.config.extra, chat_id, topic)
        msg_type = MessageType.COMMAND if text.startswith("/") else MessageType.TEXT
        await self.handle_message(
            MessageEvent(
                text=text,
                message_type=msg_type,
                source=source,
                raw_message=message,
                message_id=message_id,
                channel_prompt=channel_prompt,
            )
        )

    def _send_target(self, chat_id: str) -> tuple[str, str]:
        if str(chat_id).startswith("dm:"):
            return "private", json.dumps([int(str(chat_id).split(":", 1)[1])])
        return "stream", str(chat_id)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        if not content:
            return SendResult(success=True)

        target_type, target = self._send_target(chat_id)
        topic = None
        if isinstance(metadata, dict):
            topic = metadata.get("thread_id") or metadata.get("topic")
        if not topic:
            topic = os.getenv("ZULIP_HOME_TOPIC", "Hermes")

        last_id = None
        chunks = self.truncate_message(self.format_message(content), MAX_MESSAGE_LENGTH)
        for chunk in chunks:
            payload = {
                "type": target_type,
                "to": target,
                "content": chunk,
            }
            if target_type == "stream":
                payload["topic"] = str(topic)
            result = await self._api_post("messages", data=payload)
            if result.get("result") != "success":
                return SendResult(success=False, error=result.get("msg") or "Failed to send Zulip message")
            last_id = str(result.get("id") or "")
        return SendResult(success=True, message_id=last_id)

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
    ) -> SendResult:
        result = await self._api_request(
            "PATCH",
            f"messages/{message_id}",
            data={"content": self.format_message(content)},
        )
        if result.get("result") != "success":
            return SendResult(success=False, error=result.get("msg") or "Failed to edit Zulip message")
        return SendResult(success=True, message_id=message_id)

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        if str(chat_id).startswith("dm:"):
            return {"name": chat_id, "type": "dm"}
        return {"name": chat_id, "type": "channel"}

    async def send_typing(self, chat_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        target_type, target = self._send_target(chat_id)
        payload = {"op": "start", "type": target_type}
        topic = (metadata or {}).get("thread_id") if metadata else None
        if target_type == "stream":
            stream_id = await self._stream_id_for_name(target)
            if not stream_id:
                return
            payload["stream_id"] = stream_id
            if topic:
                payload["topic"] = str(topic)
        else:
            payload["to"] = target
        await self._api_post("typing", data=payload)

    async def _add_reaction(self, message_id: str, emoji_name: str) -> None:
        await self._api_post(
            f"messages/{message_id}/reactions",
            data={"emoji_name": emoji_name},
        )

    async def _handle_reaction_event(self, event: Dict[str, Any]) -> None:
        if event.get("op") != "add":
            return
        message_id = str(event.get("message_id") or "")
        prompt = self._approval_prompts_by_message.get(message_id)
        if not prompt or prompt.resolved:
            return
        if prompt.expires_at and time.monotonic() > prompt.expires_at:
            prompt.resolved = True
            return
        user_id = str(event.get("user_id") or "")
        if self._approval_require_sender and prompt.requester_user_id and user_id != prompt.requester_user_id:
            return

        emoji_name = str(event.get("emoji_name") or "")
        choice = _APPROVAL_REACTIONS.get(emoji_name)
        if not choice:
            return

        try:
            from tools.approval import resolve_gateway_approval

            resolve_gateway_approval(prompt.session_key, choice)
            prompt.resolved = True
            await self.edit_message(
                prompt.chat_id,
                prompt.message_id,
                f"Command approval resolved: `{choice}`",
                finalize=True,
            )
        except Exception as exc:
            logger.warning("Zulip: failed to resolve approval: %s", exc, exc_info=True)

    async def send_exec_approval(
        self,
        chat_id: str,
        command: str,
        session_key: str,
        description: str = "dangerous command",
        metadata: Optional[dict] = None,
    ) -> SendResult:
        requester_user_id = str((metadata or {}).get("requester_user_id") or "") or None
        cmd_preview = command[:2000] + "..." if len(command) > 2000 else command
        text = (
            "**Dangerous command requires approval**\n"
            f"```\n{cmd_preview}\n```\n"
            f"Reason: {description}\n\n"
            "React to this message:\n"
            "- check mark = approve once\n"
            "- clock = approve for this session\n"
            "- infinity = approve always\n"
            "- x = deny"
        )

        result = await self.send(chat_id, text, metadata=metadata)
        if not result.success or not result.message_id:
            return result

        old_message_id = self._approval_prompt_by_session.get(session_key)
        if old_message_id:
            self._approval_prompts_by_message.pop(old_message_id, None)
        prompt = _ZulipApprovalPrompt(
            session_key=session_key,
            chat_id=chat_id,
            message_id=result.message_id,
            topic=(metadata or {}).get("thread_id") if metadata else None,
            requester_user_id=requester_user_id,
            expires_at=time.monotonic() + max(self._approval_timeout_seconds, 0),
        )
        self._approval_prompts_by_message[result.message_id] = prompt
        self._approval_prompt_by_session[session_key] = result.message_id

        for emoji in ("check", "clock", "infinity", "x"):
            try:
                await self._add_reaction(result.message_id, emoji)
                prompt.bot_reactions.append(emoji)
            except Exception as exc:
                logger.debug("Zulip: failed to add approval reaction %s: %s", emoji, exc)

        return result

    def format_message(self, content: str) -> str:
        return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"\2", content)


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[list] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    try:
        import aiohttp
    except ImportError:
        return {"error": "aiohttp not installed. Run: pip install aiohttp"}

    base_url = ((getattr(pconfig, "extra", {}) or {}).get("url") or os.getenv("ZULIP_URL", "")).rstrip("/")
    bot_email = ((getattr(pconfig, "extra", {}) or {}).get("bot_email") or os.getenv("ZULIP_BOT_EMAIL", ""))
    api_key = (getattr(pconfig, "token", None) or os.getenv("ZULIP_API_KEY", "")).strip()
    ca_cert = os.getenv("ZULIP_CA_CERT", "").strip()
    if not base_url or not bot_email or not api_key:
        return {"error": "Zulip standalone send: ZULIP_URL, ZULIP_BOT_EMAIL, and ZULIP_API_KEY are required"}
    target_type = "private" if str(chat_id).startswith("dm:") else "stream"
    target = json.dumps([int(str(chat_id).split(":", 1)[1])]) if target_type == "private" else str(chat_id)
    payload = {"type": target_type, "to": target, "content": message}
    if target_type == "stream":
        payload["topic"] = thread_id or os.getenv("ZULIP_HOME_TOPIC", "Hermes")

    try:
        ssl_context = ssl.create_default_context(cafile=ca_cert) if ca_cert else None
        async with aiohttp.ClientSession(auth=aiohttp.BasicAuth(bot_email, api_key)) as session:
            async with session.post(f"{base_url}/api/v1/messages", data=payload, ssl=ssl_context) as resp:
                data = await resp.json()
                if resp.status >= 400 or data.get("result") != "success":
                    return {"error": f"Zulip API error ({resp.status}): {data}"}
                return {"success": True, "platform": "zulip", "chat_id": chat_id, "message_id": data.get("id")}
    except Exception as exc:
        return {"error": f"Zulip send failed: {exc}"}


def interactive_setup() -> None:
    from hermes_cli.config import get_env_value, save_env_value
    from hermes_cli.cli_output import (
        print_header,
        print_info,
        print_success,
        prompt,
        prompt_yes_no,
    )

    print_header("Zulip")
    existing = get_env_value("ZULIP_API_KEY")
    if existing:
        print_info("Zulip: already configured")
        if not prompt_yes_no("Reconfigure Zulip?", False):
            return

    print_info("Create a Zulip bot, then paste its email and API key.")
    url = prompt("Zulip server URL (e.g. https://chat.example.com)")
    if url:
        save_env_value("ZULIP_URL", url.rstrip("/"))
    email = prompt("Zulip bot email")
    if email:
        save_env_value("ZULIP_BOT_EMAIL", email.strip())
    api_key = prompt("Zulip bot API key", password=True)
    if api_key:
        save_env_value("ZULIP_API_KEY", api_key.strip())
    allowed_users = prompt("Allowed Zulip user IDs/emails (comma-separated, leave empty to set manually)")
    if allowed_users:
        save_env_value("ZULIP_ALLOWED_USERS", allowed_users.replace(" ", ""))
    allowed_groups = prompt("Allowed Zulip user groups (comma-separated, leave empty to skip)")
    if allowed_groups:
        save_env_value("ZULIP_ALLOWED_GROUPS", allowed_groups.strip())
    home_channel = prompt("Home channel/stream name (leave empty to set later)")
    if home_channel:
        save_env_value("ZULIP_HOME_CHANNEL", home_channel.strip())
    home_topic = prompt("Home topic (default Hermes)")
    if home_topic:
        save_env_value("ZULIP_HOME_TOPIC", home_topic.strip())
    print_success("Zulip configuration saved")


def _apply_yaml_config(yaml_cfg: dict, zulip_cfg: dict) -> dict | None:
    extras: dict[str, Any] = {}
    if "url" in zulip_cfg and not os.getenv("ZULIP_URL"):
        os.environ["ZULIP_URL"] = str(zulip_cfg["url"]).rstrip("/")
    if "bot_email" in zulip_cfg and not os.getenv("ZULIP_BOT_EMAIL"):
        os.environ["ZULIP_BOT_EMAIL"] = str(zulip_cfg["bot_email"])
    if "require_mention" in zulip_cfg and not os.getenv("ZULIP_REQUIRE_MENTION"):
        os.environ["ZULIP_REQUIRE_MENTION"] = str(zulip_cfg["require_mention"]).lower()
    allowed_groups = zulip_cfg.get("allowed_groups")
    if allowed_groups is not None and not os.getenv("ZULIP_ALLOWED_GROUPS"):
        if isinstance(allowed_groups, list):
            allowed_groups = ",".join(str(v) for v in allowed_groups)
        os.environ["ZULIP_ALLOWED_GROUPS"] = str(allowed_groups)
    for yaml_key, env_key in (
        ("free_response_channels", "ZULIP_FREE_RESPONSE_CHANNELS"),
        ("allowed_channels", "ZULIP_ALLOWED_CHANNELS"),
    ):
        value = zulip_cfg.get(yaml_key)
        if value is not None and not os.getenv(env_key):
            if isinstance(value, list):
                value = ",".join(str(v) for v in value)
            os.environ[env_key] = str(value)
    for key in (
        "url",
        "bot_email",
        "allowed_channels",
        "allowed_groups",
        "free_response_channels",
    ):
        if key in zulip_cfg:
            extras[key] = zulip_cfg[key]
    return extras or None


def _env_enablement() -> dict[str, Any] | None:
    if not (os.getenv("ZULIP_URL") and os.getenv("ZULIP_BOT_EMAIL") and os.getenv("ZULIP_API_KEY")):
        return None
    extra: dict[str, Any] = {
        "url": os.getenv("ZULIP_URL", "").rstrip("/"),
        "bot_email": os.getenv("ZULIP_BOT_EMAIL", ""),
    }
    home = os.getenv("ZULIP_HOME_CHANNEL")
    if home:
        extra["home_channel"] = home
    return extra


def _is_connected(config) -> bool:
    import hermes_cli.gateway as gateway_mod

    return bool(
        (gateway_mod.get_env_value("ZULIP_URL") or "").strip()
        and (gateway_mod.get_env_value("ZULIP_BOT_EMAIL") or "").strip()
        and (gateway_mod.get_env_value("ZULIP_API_KEY") or "").strip()
    )


def _build_adapter(config):
    return ZulipAdapter(config)


def register(ctx) -> None:
    ctx.register_platform(
        name="zulip",
        label="Zulip",
        adapter_factory=_build_adapter,
        check_fn=check_zulip_requirements,
        is_connected=_is_connected,
        required_env=["ZULIP_URL", "ZULIP_BOT_EMAIL", "ZULIP_API_KEY"],
        install_hint="pip install aiohttp",
        setup_fn=interactive_setup,
        apply_yaml_config_fn=_apply_yaml_config,
        env_enablement_fn=_env_enablement,
        allowed_users_env="ZULIP_ALLOWED_USERS",
        allow_all_env="ZULIP_ALLOW_ALL_USERS",
        allowlist_envs=["ZULIP_ALLOWED_GROUPS"],
        cron_deliver_env_var="ZULIP_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji="💬",
        platform_hint="You are chatting in Zulip. Keep replies concise and use the current topic as the active thread.",
        allow_update_command=True,
    )
