"""Durable ingest helpers for Discord Native Multi-Bot Protocol v2.

Slice 2.3 normalized Discord messages into deterministic topics and message-map
rows. Slice 3.1 wires those observations into the deterministic routing engine.
This module still never invokes Hermes or sends replies; it only persists durable
routing state for later workers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from gateway.discord_identity_registry import DiscordIdentityRegistry
from gateway.discord_protocol_v2_routing import DiscordProtocolV2Router
from gateway.discord_protocol_v2_store import DiscordProtocolV2Store
from gateway.secret_refs import redact_sensitive_data


@dataclass(frozen=True)
class NormalizedDiscordMessage:
    discord_message_id: str
    guild_id: str
    channel_id: str
    thread_id: str | None
    parent_channel_id: str | None
    topic_id: str
    title: str
    author_id: str
    author_kind: str
    author_bot_user_id: str | None
    source_client_agent_id: str
    reply_to_message_id: str | None
    mentions: list[str]
    payload: dict[str, Any]


@dataclass(frozen=True)
class DiscordIngestResult:
    ignored: bool
    reason: str | None
    normalized: NormalizedDiscordMessage | None
    topic: dict[str, Any] | None
    message: dict[str, Any] | None
    deliveries: list[dict[str, Any]]
    route_decisions: list[dict[str, Any]]


class DiscordProtocolV2Ingestor:
    """Normalize and persist observed Discord messages for protocol v2."""

    def __init__(
        self,
        *,
        store: DiscordProtocolV2Store,
        identity_registry: DiscordIdentityRegistry,
        default_intake_agent_id: str | None = None,
        guild_allowlist: Iterable[str] | None = None,
        mode: str = "listen_only",
    ) -> None:
        self.store = store
        self.identity_registry = identity_registry
        self.default_intake_agent_id = default_intake_agent_id
        self.guild_allowlist = {str(item) for item in guild_allowlist or []}
        self.mode = str(mode or "listen_only")
        self.router = DiscordProtocolV2Router(
            store=store,
            identity_registry=identity_registry,
            default_intake_agent_id=default_intake_agent_id,
        )

    def ingest_message(
        self,
        *,
        source_client_agent_id: str,
        message: Any,
    ) -> DiscordIngestResult:
        normalized = self.normalize_message(
            source_client_agent_id=source_client_agent_id,
            message=message,
        )
        if self.guild_allowlist and normalized.guild_id not in self.guild_allowlist:
            return DiscordIngestResult(
                ignored=True,
                reason="guild_not_allowed",
                normalized=normalized,
                topic=None,
                message=None,
                deliveries=[],
                route_decisions=[],
            )

        topic = self.store.upsert_topic(
            topic_id=normalized.topic_id,
            guild_id=normalized.guild_id,
            channel_id=normalized.channel_id,
            thread_id=normalized.thread_id,
            parent_channel_id=normalized.parent_channel_id,
            title=normalized.title,
            state={"mode": self.mode, "source": "discord_native_multibot"},
        )

        if self.mode == "shadow":
            plan = self.router.plan_message(normalized)
            self.store.upsert_message_map(
                discord_message_id=normalized.discord_message_id,
                guild_id=normalized.guild_id,
                channel_id=normalized.channel_id,
                thread_id=normalized.thread_id,
                parent_channel_id=normalized.parent_channel_id,
                direction="inbound",
                author_id=normalized.author_id,
                author_kind=normalized.author_kind,
                author_bot_user_id=normalized.author_bot_user_id,
                source_client_agent_id=normalized.source_client_agent_id,
                mentions=normalized.mentions,
                payload={
                    **plan.payload,
                    "shadow_replay": True,
                    "shadow_target_agent_ids": plan.target_agent_ids,
                },
            )
            self.store.record_route_decision(
                source_type="discord_message",
                source_id=normalized.discord_message_id,
                topic_id=normalized.topic_id,
                author_kind=normalized.author_kind,
                decision=plan.decision,
                target_agent_ids=plan.target_agent_ids,
                reason=plan.reason,
                payload={
                    **plan.payload,
                    "shadow_replay": True,
                    "shadow_no_delivery": True,
                },
            )
            message_row = self.store.get_message_map(normalized.discord_message_id)
            route_decisions = self.store.route_decisions_for(
                "discord_message", normalized.discord_message_id
            )
            return DiscordIngestResult(
                ignored=False,
                reason=None,
                normalized=normalized,
                topic=topic,
                message=message_row,
                deliveries=[],
                route_decisions=route_decisions,
            )

        deliveries = self.router.route_message(normalized)
        message_row = self.store.get_message_map(normalized.discord_message_id)
        route_decisions = self.store.route_decisions_for(
            "discord_message", normalized.discord_message_id
        )
        return DiscordIngestResult(
            ignored=False,
            reason=None,
            normalized=normalized,
            topic=topic,
            message=message_row,
            deliveries=deliveries,
            route_decisions=route_decisions,
        )

    def normalize_message(
        self,
        *,
        source_client_agent_id: str,
        message: Any,
    ) -> NormalizedDiscordMessage:
        guild = _attr(message, "guild")
        guild_id = _string_id(_attr(guild, "id"), "dm")
        channel = _attr(message, "channel")
        raw_channel_id = _string_id(_attr(channel, "id"), "unknown")
        parent_channel_id = _channel_parent_id(channel)
        thread_id = raw_channel_id if parent_channel_id else None
        channel_id = parent_channel_id or raw_channel_id
        topic_id = normalize_topic_id(
            guild_id=guild_id,
            channel_id=channel_id,
            thread_id=thread_id,
        )
        author = _attr(message, "author")
        author_id = _string_id(_attr(author, "id"), "unknown")
        author_kind = classify_author_kind(
            message=message,
            author_id=author_id,
            identity_registry=self.identity_registry,
        )
        author_bot_user_id = author_id if author_kind in {"registered_bot", "external_bot"} else None
        reply_to_message_id = _reference_message_id(message)
        mention_objects = list(_attr(message, "mentions", []) or [])
        mentions = [_string_id(_attr(mention, "id")) for mention in mention_objects]
        mention_users = [
            {"id": _string_id(_attr(mention, "id")), "bot": bool(_attr(mention, "bot", False))}
            for mention in mention_objects
        ]
        content = _string_id(_attr(message, "content"))
        payload = redact_sensitive_data(
            {
                "discord_message_id": _string_id(_attr(message, "id")),
                "guild_id": guild_id,
                "channel_id": channel_id,
                "thread_id": thread_id,
                "parent_channel_id": parent_channel_id,
                "source_client_agent_id": source_client_agent_id,
                "reply_to_message_id": reply_to_message_id,
                "content": content,
                "content_length": len(content),
                "mention_ids": mentions,
                "mention_users": mention_users,
                "attachments_count": len(_attr(message, "attachments", []) or []),
            }
        )
        return NormalizedDiscordMessage(
            discord_message_id=_string_id(_attr(message, "id")),
            guild_id=guild_id,
            channel_id=channel_id,
            thread_id=thread_id,
            parent_channel_id=parent_channel_id,
            topic_id=topic_id,
            title=_string_id(_attr(channel, "name"), channel_id),
            author_id=author_id,
            author_kind=author_kind,
            author_bot_user_id=author_bot_user_id,
            source_client_agent_id=source_client_agent_id,
            reply_to_message_id=reply_to_message_id,
            mentions=mentions,
            payload=payload,
        )

def normalize_topic_id(*, guild_id: str, channel_id: str, thread_id: str | None) -> str:
    """Return the deterministic Discord topic key: guild/channel/thread-or-root."""

    return f"{guild_id}/{channel_id}/{thread_id or 'root'}"


def classify_author_kind(
    *,
    message: Any,
    author_id: str,
    identity_registry: DiscordIdentityRegistry,
) -> str:
    """Classify a Discord message author for bot-loop-safe ingest."""

    if _attr(message, "webhook_id"):
        return "webhook"
    message_type = _attr(message, "type")
    message_type_name = _attr(message_type, "name", message_type)
    if message_type_name not in (None, "default", "reply"):
        return "system"
    registered_bot_ids = {
        identity.discord_bot_user_id for identity in identity_registry.identities.values()
    }
    if author_id in registered_bot_ids:
        return "registered_bot"
    author = _attr(message, "author")
    if bool(_attr(author, "bot", False)):
        return "external_bot"
    return "human"



def _channel_parent_id(channel: Any) -> str | None:
    parent_id = _attr(channel, "parent_id")
    if parent_id is None:
        parent = _attr(channel, "parent")
        parent_id = _attr(parent, "id")
    return _string_id(parent_id) if parent_id is not None else None


def _reference_message_id(message: Any) -> str | None:
    reference = _attr(message, "reference")
    if reference is not None:
        reference_id = _string_id(_attr(reference, "message_id"), "")
        if reference_id:
            return reference_id

    referenced_message = _attr(message, "referenced_message")
    if referenced_message is not None:
        reference_id = _string_id(_attr(referenced_message, "id"), "")
        if reference_id:
            return reference_id

    reference_message_id = _string_id(_attr(message, "reference_message_id"), "")
    return reference_message_id or None


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def _string_id(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)
