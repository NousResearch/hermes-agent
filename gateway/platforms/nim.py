from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Optional

from nim_bot_py.bridge import NimBridgeError
from nim_bot_py.bridge import NodeBridge as PackagedNodeBridge

from gateway.config import (
    NimResolvedConfig,
    Platform,
    PlatformConfig,
    _default_nim_bridge_command,
    decode_nim_chat_id,
    encode_nim_chat_id,
    load_nim_config,
    load_nim_instances,
)
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.platforms.nim_bridge import NodeBridgeProcess

logger = logging.getLogger(__name__)


def check_nim_requirements(config: PlatformConfig | None = None) -> bool:
    instances = load_nim_instances(config or PlatformConfig(enabled=True))
    if not instances:
        return False
    return any(_check_nim_instance_requirements(resolved) for resolved in instances)


def _check_nim_instance_requirements(resolved: NimResolvedConfig) -> bool:
    command = list(resolved.bridge_command or [])
    if not command:
        return False
    executable = command[0]
    if os.path.isabs(executable) or "/" in executable:
        executable_ok = Path(executable).exists()
    else:
        executable_ok = shutil.which(executable) is not None
    if not executable_ok:
        return False

    default_command = _default_nim_bridge_command()
    if command == default_command:
        try:
            PackagedNodeBridge(command=command).ensure_runtime()
        except NimBridgeError as exc:
            logger.warning("[nim] nim-bot-py runtime is unavailable: %s", exc)
            return False
        return True

    if len(command) >= 2 and executable.endswith("node"):
        return Path(command[1]).exists()

    return True


class NimAdapter(BasePlatformAdapter):
    MAX_MESSAGE_LENGTH = 4000

    def __init__(
        self,
        config: PlatformConfig,
        *,
        bridge: NodeBridgeProcess | Any | None = None,
        resolved: NimResolvedConfig | None = None,
        event_sink: Any | None = None,
    ) -> None:
        super().__init__(config=config, platform=Platform.NIM)
        self.resolved: NimResolvedConfig = resolved or load_nim_config(config)
        self._bridge = bridge or NodeBridgeProcess(self.resolved.bridge_command)
        self._chat_cache: dict[str, dict[str, str]] = {}
        self._event_sink = event_sink

    async def connect(self) -> bool:
        if not self.resolved.configured():
            self._mark_disconnected()
            return False
        await self._bridge.start(self.resolved, event_handler=self._on_bridge_event)
        self._mark_connected()
        return True

    async def disconnect(self) -> None:
        await self._bridge.stop()
        self._mark_disconnected()

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        routed_chat_id = self._strip_route_prefix(chat_id)
        session_type = self._infer_session_type(routed_chat_id, metadata)
        reply_to_id = reply_to or (metadata or {}).get("reply_to")
        result: dict[str, Any] | None = None
        for chunk in self._split_content(content):
            result = await self._bridge.send_text(
                chat_id=routed_chat_id,
                text=chunk,
                session_type=session_type,
                reply_to=reply_to_id,
            )
        return SendResult(
            success=True,
            message_id=str((result or {}).get("message_id") or (result or {}).get("client_message_id") or ""),
            raw_response=result or {},
        )

    def _split_content(self, content: str) -> list[str]:
        chunk_limit = max(1, int(self.resolved.text_chunk_limit or self.MAX_MESSAGE_LENGTH))
        if len(content) <= chunk_limit:
            return [content]

        chunks: list[str] = []
        remaining = content
        while remaining:
            if len(remaining) <= chunk_limit:
                chunks.append(remaining)
                break
            split_at = remaining.rfind("\n", 0, chunk_limit)
            if split_at > 0:
                chunks.append(remaining[:split_at + 1])
                remaining = remaining[split_at + 1 :]
                continue
            chunks.append(remaining[:chunk_limit])
            remaining = remaining[chunk_limit:]
        return chunks

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        routed_chat_id = self._strip_route_prefix(chat_id)
        cached = self._chat_cache.get(routed_chat_id)
        if cached is not None:
            return dict(cached)
        if routed_chat_id.startswith(("team:", "superTeam:", "qchat:")):
            return {"name": routed_chat_id, "type": "group"}
        return {"name": routed_chat_id, "type": "dm"}

    async def health(self) -> dict[str, Any]:
        return await self._bridge.health()

    async def _on_bridge_event(self, envelope: dict[str, Any]) -> None:
        if envelope.get("event") != "message":
            return
        payload = dict(envelope.get("payload") or {})
        ignore_reason = self._ignore_reason(payload)
        self._log_inbound_debug(payload, ignore_reason)
        if ignore_reason is not None:
            return
        event = self._to_message_event(payload)
        routed_chat_id = self._strip_route_prefix(event.source.chat_id)
        self._chat_cache[routed_chat_id] = {
            "name": event.source.chat_name or event.source.chat_id,
            "type": event.source.chat_type,
        }
        if self._event_sink is not None:
            await self._event_sink(event)
            return
        await self.handle_message(event)

    def _ignore_reason(self, payload: dict[str, Any]) -> str | None:
        if payload.get("from_self"):
            return "self_message"
        message_source = payload.get("message_source")
        if message_source not in (None, "", 1, "1"):
            return f"non_online_message_source:{message_source}"
        session_type = str(payload.get("session_type") or "p2p")
        sender_id = str(payload.get("sender_id") or "")
        if session_type == "p2p":
            if not self._is_allowed_direct_sender(sender_id):
                return "dm_sender_not_allowed"
            return None
        if session_type in {"team", "superTeam"}:
            if not self._is_allowed_team_message(
                group_id=str(payload.get("target_id") or ""),
                sender_id=sender_id,
                session_type=session_type,
            ):
                return "group_not_allowed"
            if not self._is_mentioned(payload):
                return "group_not_mentioned"
            return None
        if session_type == "qchat":
            if not self._is_allowed_qchat_message(
                server_id=str(payload.get("server_id") or ""),
                channel_id=str(payload.get("channel_id") or ""),
                sender_id=sender_id,
            ):
                return "qchat_not_allowed"
            if not self._is_mentioned(payload):
                return "qchat_not_mentioned"
            return None
        return f"unsupported_session_type:{session_type}"

    def _log_inbound_debug(self, payload: dict[str, Any], ignore_reason: str | None) -> None:
        if not self.resolved.debug:
            return
        session_type = str(payload.get("session_type") or "p2p")
        if session_type == "p2p" and ignore_reason is None:
            return
        force_push_ids = [str(item) for item in payload.get("force_push_account_ids") or []]
        logger.info(
            "[nim:%s] inbound debug session_type=%s sender=%s target=%s mentioned=%s mention_all=%s "
            "force_push_ids=%s ignore_reason=%s text=%r",
            self.resolved.instance_name,
            session_type,
            str(payload.get("sender_id") or ""),
            str(payload.get("target_id") or ""),
            bool(payload.get("mentioned")),
            bool(payload.get("mention_all")),
            force_push_ids,
            ignore_reason or "accepted",
            str(payload.get("text") or "")[:200],
        )

    def _is_allowed_direct_sender(self, sender_id: str) -> bool:
        policy = self.resolved.p2p_policy
        if policy == "disabled":
            return False
        if policy == "open":
            return True
        if not self.resolved.p2p_allow_from:
            return False
        normalized_sender = sender_id.lower()
        return any(entry.lower() == normalized_sender for entry in self.resolved.p2p_allow_from)

    def _is_allowed_team_message(self, *, group_id: str, sender_id: str, session_type: str) -> bool:
        policy = self.resolved.team_policy
        if policy == "disabled":
            return False
        if policy == "open":
            return True
        if not self.resolved.team_allow_from:
            return False

        normalized_group = group_id.lower()
        normalized_sender = sender_id.lower()
        normalized_type = "superTeam" if session_type == "superTeam" else "team"

        for raw_entry in self.resolved.team_allow_from:
            parts = [part.strip() for part in str(raw_entry).split("|")]
            first = (parts[0] if parts else "").lower()
            entry_type: str | None = None

            if first in {"1", "2"}:
                entry_type = "team" if first == "1" else "superTeam"
                entry_group = (parts[1] if len(parts) > 1 else "").strip().lower()
                entry_sender = (parts[2] if len(parts) > 2 else "").strip().lower()
            else:
                entry_group = first
                entry_sender = (parts[1] if len(parts) > 1 else "").strip().lower()

            if entry_type is not None and normalized_type != entry_type:
                continue
            if entry_group != normalized_group:
                continue
            if entry_sender and entry_sender != normalized_sender:
                continue
            return True

        return False

    def _is_allowed_qchat_message(self, *, server_id: str, channel_id: str, sender_id: str) -> bool:
        policy = self.resolved.qchat_policy
        if policy == "disabled":
            return False
        if policy == "open":
            return True
        if not self.resolved.qchat_allow_from:
            return False

        normalized_server = server_id.lower()
        normalized_channel = channel_id.lower()
        normalized_sender = sender_id.lower()

        for raw_entry in self.resolved.qchat_allow_from:
            parts = [part.strip() for part in str(raw_entry).split("|")]
            entry_server = (parts[0] if parts else "").lower()
            entry_channel = (parts[1] if len(parts) > 1 else "").strip().lower()
            entry_sender = (parts[2] if len(parts) > 2 else "").strip().lower()

            if entry_server != normalized_server:
                continue
            if entry_channel and entry_channel != normalized_channel:
                continue
            if entry_sender and entry_sender != normalized_sender:
                continue
            return True

        return False

    def _is_mentioned(self, payload: dict[str, Any]) -> bool:
        if payload.get("mentioned") or payload.get("mention_all"):
            return True
        force_push_ids = {str(item) for item in payload.get("force_push_account_ids") or []}
        account = self.resolved.credentials.account if self.resolved.credentials else ""
        return bool(account and account in force_push_ids)

    def _to_message_event(self, payload: dict[str, Any]) -> MessageEvent:
        session_type = str(payload.get("session_type") or "p2p")
        sender_id = str(payload.get("sender_id") or "")
        target_id = str(payload.get("target_id") or "")
        chat_type = "dm" if session_type == "p2p" else "group"
        if session_type == "p2p":
            raw_chat_id = f"user:{sender_id}"
        elif session_type == "qchat":
            server_id = str(payload.get("server_id") or "")
            channel_id = str(payload.get("channel_id") or "")
            target = target_id or f"{server_id}:{channel_id}".strip(":")
            raw_chat_id = f"qchat:{target}"
        elif session_type == "superTeam":
            raw_chat_id = f"superTeam:{target_id}"
        else:
            raw_chat_id = f"team:{target_id}"
        chat_id = self._apply_route_prefix(raw_chat_id)
        source = self.build_source(
            chat_id=chat_id,
            chat_type=chat_type,
            chat_name=payload.get("conversation_name"),
            user_id=sender_id,
            user_name=payload.get("sender_name"),
        )
        return MessageEvent(
            text=str(payload.get("text") or ""),
            message_type=self._to_message_type(str(payload.get("message_type") or "text")),
            source=source,
            raw_message=payload,
            message_id=str(payload.get("message_id") or payload.get("client_message_id") or ""),
            reply_to_message_id=str(payload.get("reply_to") or "") or None,
        )

    def _infer_session_type(self, chat_id: str, metadata: dict[str, Any] | None) -> str:
        if metadata and metadata.get("session_type"):
            return str(metadata["session_type"])
        if chat_id.startswith("qchat:"):
            return "qchat"
        if chat_id.startswith("superTeam:"):
            return "superTeam"
        if chat_id.startswith("team:"):
            return "team"
        return "p2p"

    def _to_message_type(self, value: str) -> MessageType:
        mapping = {
            "text": MessageType.TEXT,
            "image": MessageType.PHOTO,
            "audio": MessageType.AUDIO,
            "video": MessageType.VIDEO,
            "file": MessageType.DOCUMENT,
        }
        return mapping.get(value, MessageType.TEXT)

    def _apply_route_prefix(self, chat_id: str) -> str:
        return encode_nim_chat_id(self.resolved.instance_name, chat_id) if self.resolved.route_prefix else chat_id

    def _strip_route_prefix(self, chat_id: str) -> str:
        if not self.resolved.route_prefix:
            return str(chat_id)
        instance_name, routed = decode_nim_chat_id(chat_id)
        if instance_name == self.resolved.instance_name and routed:
            return routed
        return str(chat_id)


class MultiNimAdapter(BasePlatformAdapter):
    _QCHAT_DEDUPE_TTL_SECONDS = 15.0

    def __init__(
        self,
        config: PlatformConfig,
        *,
        resolved_instances: list[NimResolvedConfig] | None = None,
        bridge_factory: Any | None = None,
    ) -> None:
        super().__init__(config=config, platform=Platform.NIM)
        self._resolved_instances = resolved_instances or load_nim_instances(config)
        self._bridge_factory = bridge_factory
        self._instances: dict[str, NimAdapter] = {}
        self._default_instance_name: str | None = None
        self._recent_qchat_events: dict[str, float] = {}

    async def connect(self) -> bool:
        if not self._resolved_instances:
            self._mark_disconnected()
            return False

        connected = 0
        self._instances = {}
        self._default_instance_name = None
        for resolved in self._resolved_instances:
            bridge = self._bridge_factory(resolved) if self._bridge_factory else None
            adapter = NimAdapter(
                self.config,
                bridge=bridge,
                resolved=resolved,
                event_sink=self.handle_message,
            )
            self._instances[resolved.instance_name] = adapter
            if self._default_instance_name is None:
                self._default_instance_name = resolved.instance_name
            try:
                success = await adapter.connect()
            except Exception:
                self._instances.pop(resolved.instance_name, None)
                if self._default_instance_name == resolved.instance_name:
                    self._default_instance_name = next(iter(self._instances), None)
                logger.exception("[nim:%s] connect failed", resolved.instance_name)
                continue
            if not success:
                self._instances.pop(resolved.instance_name, None)
                if self._default_instance_name == resolved.instance_name:
                    self._default_instance_name = next(iter(self._instances), None)
                continue
            connected += 1

        if connected:
            self._mark_connected()
            return True

        self._mark_disconnected()
        return False

    async def disconnect(self) -> None:
        for adapter in list(self._instances.values()):
            try:
                await adapter.disconnect()
            except Exception:
                logger.debug("[nim] failed to disconnect child adapter", exc_info=True)
        self._instances.clear()
        self._default_instance_name = None
        self._mark_disconnected()

    async def handle_message(self, event: MessageEvent) -> None:
        if self._should_drop_duplicate_qchat_event(event):
            return
        await super().handle_message(event)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> SendResult:
        adapter, routed_chat_id = self._resolve_adapter(chat_id, metadata)
        if adapter is None:
            raise RuntimeError("NIM instance is unavailable for chat target")
        return await adapter.send(routed_chat_id, content, reply_to=reply_to, metadata=metadata)

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        adapter, routed_chat_id = self._resolve_adapter(chat_id, None)
        if adapter is None:
            return {"name": chat_id, "type": "dm"}
        return await adapter.get_chat_info(routed_chat_id)

    async def health(self) -> dict[str, Any]:
        children = {}
        for name, adapter in self._instances.items():
            try:
                children[name] = await adapter.health()
            except Exception as exc:
                children[name] = {"connected": False, "error": str(exc)}
        return {
            "connected": bool(self._instances),
            "instances": children,
        }

    def _resolve_adapter(
        self,
        chat_id: str,
        metadata: dict[str, Any] | None,
    ) -> tuple[NimAdapter | None, str]:
        requested_instance = None
        if metadata:
            requested_instance = str(metadata.get("nim_instance") or "").strip() or None
        routed_instance, routed_chat_id = decode_nim_chat_id(chat_id)
        instance_name = requested_instance or routed_instance or self._default_instance_name
        if instance_name and instance_name in self._instances:
            return self._instances[instance_name], routed_chat_id if routed_instance else str(chat_id)
        if requested_instance or routed_instance:
            return None, routed_chat_id if routed_instance else str(chat_id)
        if self._default_instance_name and self._default_instance_name in self._instances:
            return self._instances[self._default_instance_name], str(chat_id)
        return None, str(chat_id)

    def _should_drop_duplicate_qchat_event(self, event: MessageEvent) -> bool:
        raw_message = event.raw_message if isinstance(event.raw_message, dict) else {}
        session_type = str(raw_message.get("session_type") or "")
        if session_type != "qchat":
            return False

        server_id = str(raw_message.get("server_id") or raw_message.get("serverId") or "")
        channel_id = str(raw_message.get("channel_id") or raw_message.get("channelId") or "")
        sender_id = str(raw_message.get("sender_id") or raw_message.get("senderId") or event.source.user_id or "")
        message_id = str(
            event.message_id
            or raw_message.get("message_id")
            or raw_message.get("msgIdServer")
            or raw_message.get("msg_server_id")
            or raw_message.get("client_message_id")
            or raw_message.get("msgIdClient")
            or raw_message.get("msg_client_id")
            or ""
        )
        if not message_id:
            message_id = f"{sender_id}|{server_id}|{channel_id}|{event.text}"
        dedupe_key = f"{sender_id}|{server_id}|{channel_id}|{message_id}"

        now = time.monotonic()
        cutoff = now - self._QCHAT_DEDUPE_TTL_SECONDS
        expired = [key for key, ts in self._recent_qchat_events.items() if ts < cutoff]
        for key in expired:
            self._recent_qchat_events.pop(key, None)

        if dedupe_key in self._recent_qchat_events:
            logger.info("[nim] dropping duplicate qchat event: %s", dedupe_key)
            return True

        self._recent_qchat_events[dedupe_key] = now
        return False


PlatformAdapter = NimAdapter
