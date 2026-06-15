"""Deterministic routing engine for Discord Native Multi-Bot Protocol v2.

Slice 3.1 keeps routing independent from Discord adapter side effects.  It consumes
already-normalized/persisted Discord observations, applies mention/reply/default
priority, and persists durable inbound deliveries plus route diagnostics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from gateway.discord_identity_registry import DiscordIdentityRegistry
from gateway.discord_protocol_v2_store import DISCORD_SOURCE_TYPE, DiscordProtocolV2Store


@dataclass(frozen=True)
class DiscordRoutePlan:
    decision: str
    target_agent_ids: list[str]
    reason: str
    payload: dict[str, Any]


class DiscordProtocolV2Router:
    """Route normalized Discord observations to Hermes agent identities."""

    def __init__(
        self,
        *,
        store: DiscordProtocolV2Store,
        identity_registry: DiscordIdentityRegistry,
        default_intake_agent_id: str | None = None,
        allow_default_intake: bool = True,
    ) -> None:
        self.store = store
        self.identity_registry = identity_registry
        self.default_intake_agent_id = default_intake_agent_id
        self.allow_default_intake = allow_default_intake

    def route_message(self, message: Any) -> list[dict[str, Any]]:
        """Persist a route decision and return created/reused inbound deliveries."""

        plan = self.plan_message(message)
        if plan.target_agent_ids:
            return self.store.create_discord_inbound_deliveries(
                discord_message_id=message.discord_message_id,
                guild_id=message.guild_id,
                channel_id=message.channel_id,
                topic_id=message.topic_id,
                author_id=message.author_id,
                author_kind=message.author_kind,
                target_agent_ids=plan.target_agent_ids,
                route_reason=plan.reason,
                thread_id=message.thread_id,
                parent_channel_id=message.parent_channel_id,
                author_bot_user_id=message.author_bot_user_id,
                source_client_agent_id=message.source_client_agent_id,
                mentions=message.mentions,
                payload=plan.payload,
            )

        # Keep zero-delivery diagnostics explicit.  For non-human author kinds we
        # intentionally use the store helper because it also enforces the guard
        # that Discord-originated deliveries can only be human-authored.
        if message.author_kind != "human":
            existing = self.store.get_message_map(message.discord_message_id)
            if existing is not None and existing.get("direction") == "projection":
                self.store.upsert_message_map(
                    discord_message_id=message.discord_message_id,
                    guild_id=message.guild_id,
                    channel_id=message.channel_id,
                    thread_id=message.thread_id,
                    parent_channel_id=message.parent_channel_id,
                    direction="projection",
                    agent_id=existing.get("agent_id"),
                    delivery_key=existing.get("delivery_key"),
                    outbox_delivery_id=existing.get("outbox_delivery_id"),
                    agent_event_id=existing.get("agent_event_id"),
                    author_id=message.author_id,
                    author_kind=message.author_kind,
                    author_bot_user_id=message.author_bot_user_id,
                    source_client_agent_id=existing.get("source_client_agent_id"),
                    mentions=message.mentions,
                    payload=plan.payload,
                )
                self.store.record_route_decision(
                    source_type=DISCORD_SOURCE_TYPE,
                    source_id=message.discord_message_id,
                    topic_id=message.topic_id,
                    author_kind=message.author_kind,
                    decision=plan.decision,
                    target_agent_ids=[],
                    reason=plan.reason,
                    payload=plan.payload,
                )
                return []
            return self.store.create_discord_inbound_deliveries(
                discord_message_id=message.discord_message_id,
                guild_id=message.guild_id,
                channel_id=message.channel_id,
                topic_id=message.topic_id,
                author_id=message.author_id,
                author_kind=message.author_kind,
                target_agent_ids=[],
                route_reason=plan.reason,
                thread_id=message.thread_id,
                parent_channel_id=message.parent_channel_id,
                author_bot_user_id=message.author_bot_user_id,
                source_client_agent_id=message.source_client_agent_id,
                mentions=message.mentions,
                payload=plan.payload,
            )

        self.store.upsert_message_map(
            discord_message_id=message.discord_message_id,
            guild_id=message.guild_id,
            channel_id=message.channel_id,
            thread_id=message.thread_id,
            parent_channel_id=message.parent_channel_id,
            direction="inbound",
            author_id=message.author_id,
            author_kind=message.author_kind,
            author_bot_user_id=message.author_bot_user_id,
            source_client_agent_id=message.source_client_agent_id,
            mentions=message.mentions,
            payload=plan.payload,
        )
        self.store.record_route_decision(
            source_type=DISCORD_SOURCE_TYPE,
            source_id=message.discord_message_id,
            topic_id=message.topic_id,
            author_kind=message.author_kind,
            decision=plan.decision,
            target_agent_ids=[],
            reason=plan.reason,
            payload=plan.payload,
        )
        return []

    def plan_message(self, message: Any) -> DiscordRoutePlan:
        """Apply route priority without mutating durable state."""

        payload = dict(message.payload or {})
        if message.author_kind != "human":
            return DiscordRoutePlan(
                decision="zero_delivery",
                target_agent_ids=[],
                reason=f"non_human_author:{message.author_kind}",
                payload={
                    **payload,
                    "route_reason": f"non_human_author:{message.author_kind}",
                    "suppressed_discord_origin": True,
                },
            )

        mentioned_agent_ids = self.mentioned_agent_ids(message.mentions)
        if mentioned_agent_ids:
            return DiscordRoutePlan(
                decision="delivered",
                target_agent_ids=mentioned_agent_ids,
                reason="explicit_mention",
                payload={
                    **payload,
                    "route_reason": "explicit_mention",
                    "mentioned_agent_ids": mentioned_agent_ids,
                },
            )

        unknown_bot_mentions = self.unknown_bot_mentions(payload)
        if unknown_bot_mentions:
            return DiscordRoutePlan(
                decision="policy_failed",
                target_agent_ids=[],
                reason="unknown_bot_mention",
                payload={
                    **payload,
                    "route_reason": "unknown_bot_mention",
                    "unknown_bot_mentions": unknown_bot_mentions,
                },
            )

        reply_agent_id = self.reply_target_agent_id(message.reply_to_message_id)
        if reply_agent_id:
            return DiscordRoutePlan(
                decision="delivered",
                target_agent_ids=[reply_agent_id],
                reason="reply_to_agent",
                payload={
                    **payload,
                    "route_reason": "reply_to_agent",
                    "reply_to_message_id": message.reply_to_message_id,
                    "reply_target_agent_id": reply_agent_id,
                },
            )

        default_agent_id = self.default_target_agent_id()
        if default_agent_id:
            return DiscordRoutePlan(
                decision="delivered",
                target_agent_ids=[default_agent_id],
                reason="default_intake",
                payload={
                    **payload,
                    "route_reason": "default_intake",
                    "default_intake_agent_id": default_agent_id,
                },
            )

        return DiscordRoutePlan(
            decision="policy_failed",
            target_agent_ids=[],
            reason="default_intake_disallowed",
            payload={**payload, "route_reason": "default_intake_disallowed"},
        )

    def mentioned_agent_ids(self, mention_ids: list[str]) -> list[str]:
        bot_user_to_agent = self._active_bot_user_to_agent()
        targets: list[str] = []
        for mention_id in mention_ids:
            agent_id = bot_user_to_agent.get(str(mention_id))
            if agent_id and agent_id not in targets:
                targets.append(agent_id)
        return targets

    def unknown_bot_mentions(self, payload: dict[str, Any]) -> list[str]:
        bot_user_to_agent = self._active_bot_user_to_agent()
        unknown: list[str] = []
        for item in payload.get("mention_users", []) or []:
            if not isinstance(item, dict):
                continue
            mention_id = str(item.get("id") or "")
            if not mention_id or mention_id in bot_user_to_agent:
                continue
            if bool(item.get("bot")) and mention_id not in unknown:
                unknown.append(mention_id)
        return unknown

    def reply_target_agent_id(self, reply_to_message_id: str | None) -> str | None:
        if not reply_to_message_id:
            return None
        row = self.store.get_message_map(reply_to_message_id)
        if row is None:
            return None
        if row.get("direction") not in {"outbound", "projection"}:
            return None
        if row.get("author_kind") != "registered_bot":
            return None

        for candidate in (row.get("agent_id"), row.get("source_client_agent_id")):
            if isinstance(candidate, str) and self._is_active_agent(candidate):
                return candidate

        author_bot_user_id = row.get("author_bot_user_id")
        if isinstance(author_bot_user_id, str):
            return self._active_bot_user_to_agent().get(author_bot_user_id)
        return None

    def default_target_agent_id(self) -> str | None:
        if not self.allow_default_intake or not self.default_intake_agent_id:
            return None
        if self._is_active_agent(self.default_intake_agent_id):
            return self.default_intake_agent_id
        return None

    def _active_bot_user_to_agent(self) -> dict[str, str]:
        result: dict[str, str] = {}
        for agent_id in self.identity_registry.active_agent_ids:
            identity = self.identity_registry.get_identity(agent_id, include_disabled=False)
            if identity is not None:
                result[str(identity.discord_bot_user_id)] = identity.agent_id
        return result

    def _is_active_agent(self, agent_id: str) -> bool:
        return self.identity_registry.get_identity(agent_id, include_disabled=False) is not None


def route_decision_targets(decision: dict[str, Any]) -> list[str]:
    """Small test/debug helper for decoding stored route_decisions rows."""

    return list(json.loads(decision.get("target_agent_ids_json") or "[]"))
