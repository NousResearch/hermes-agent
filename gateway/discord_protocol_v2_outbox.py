"""Durable native Discord outbox sender for protocol v2.

This module owns the transport side of ``outbox_deliveries``.  It is designed
for tests and the native multibot adapter to inject the per-agent Discord client
resolver, so no live Discord connection is required at import time.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from gateway.discord_protocol_v2_store import (
    DiscordProtocolV2Store,
    projection_idempotency_key,
    response_idempotency_key,
)
from gateway.platforms.base import BasePlatformAdapter

logger = logging.getLogger(__name__)

DISCORD_MAX_MESSAGE_LENGTH = 2000


class DiscordProtocolV2OutboxError(Exception):
    """Base class for protocol-v2 outbox errors."""


class MissingDiscordClientError(DiscordProtocolV2OutboxError):
    """Raised internally when no live client exists for the target agent."""


class PossibleSendCommittedError(DiscordProtocolV2OutboxError):
    """Send raised after the Discord network call may already have committed.

    Reconciliation, not blind retry, must decide whether a Discord message was
    created.  The sender catches this exception class and marks the delivery
    ``uncertain``.
    """


@dataclass(frozen=True)
class DiscordProtocolV2ClientBinding:
    """Resolved native Discord client for one target agent."""

    client: Any
    source_client_agent_id: str
    author_bot_user_id: str | None = None


@dataclass(frozen=True)
class DiscordProtocolV2OutboxResult:
    """Result of one outbox delivery attempt."""

    outbox_delivery_id: str
    status: str
    sent_parts: int = 0
    skipped_parts: int = 0
    error: str | None = None


ClientResolver = Callable[[str], Any]
SendPartCallable = Callable[..., Any]


def split_discord_message(content: str) -> list[str]:
    """Split response text using the same splitter used by DiscordAdapter.send()."""

    text = str(content or "")
    if not text.strip():
        return []
    chunks = BasePlatformAdapter.truncate_message(text, DISCORD_MAX_MESSAGE_LENGTH)
    for index, chunk in enumerate(chunks):
        if len(chunk) > DISCORD_MAX_MESSAGE_LENGTH:
            raise ValueError(
                f"Discord outbox chunk {index} exceeds {DISCORD_MAX_MESSAGE_LENGTH} chars"
            )
    return chunks


def create_response_outbox_delivery(
    store: DiscordProtocolV2Store,
    *,
    inbound_delivery_key: str,
    target_agent_id: str,
    topic_id: str,
    channel_id: str,
    content: str,
    thread_id: str | None = None,
    mentions: Sequence[str] | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create or return the stable response outbox row for one inbound delivery."""

    body = dict(payload or {})
    body.setdefault("content", content)
    body.setdefault("target_agent_id", target_agent_id)
    body.setdefault("source_inbound_delivery_key", inbound_delivery_key)
    if mentions is not None:
        body.setdefault("mentions", list(mentions))
    return store.create_outbox_delivery(
        idempotency_key=response_idempotency_key(inbound_delivery_key, target_agent_id),
        target_agent_id=target_agent_id,
        topic_id=topic_id,
        channel_id=channel_id,
        thread_id=thread_id,
        delivery_kind="response",
        source_inbound_delivery_key=inbound_delivery_key,
        payload=body,
    )


def create_projection_outbox_delivery(
    store: DiscordProtocolV2Store,
    *,
    agent_event_id: str,
    target_agent_id: str,
    topic_id: str,
    channel_id: str,
    content: str,
    thread_id: str | None = None,
    mentions: Sequence[str] | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create or return the stable projection outbox row for one agent event."""

    body = dict(payload or {})
    body.setdefault("content", content)
    body.setdefault("target_agent_id", target_agent_id)
    body.setdefault("source_agent_event_id", agent_event_id)
    if mentions is not None:
        body.setdefault("mentions", list(mentions))
    return store.create_outbox_delivery(
        idempotency_key=projection_idempotency_key(agent_event_id, target_agent_id),
        target_agent_id=target_agent_id,
        topic_id=topic_id,
        channel_id=channel_id,
        thread_id=thread_id,
        delivery_kind="projection",
        source_agent_event_id=agent_event_id,
        payload=body,
    )


class DiscordProtocolV2OutboxSender:
    """Lease pending outbox rows and send them through target-agent clients."""

    def __init__(
        self,
        *,
        store: DiscordProtocolV2Store,
        client_resolver: ClientResolver,
        send_part: SendPartCallable | None = None,
        worker_id: str | None = None,
        lease_seconds: int = 60,
    ) -> None:
        self.store = store
        self.client_resolver = client_resolver
        self.send_part = send_part or default_send_discord_part
        self.worker_id = worker_id or f"discord-v2-outbox:{uuid.uuid4().hex[:12]}"
        self.lease_seconds = int(lease_seconds)

    async def run_once(self) -> DiscordProtocolV2OutboxResult | None:
        """Lease and deliver at most one retryable outbox row."""

        outbox = self.store.lease_next_outbox(
            lease_owner=self.worker_id,
            lease_seconds=self.lease_seconds,
        )
        if outbox is None:
            return None
        return await self.deliver_outbox(outbox)

    async def deliver_outbox(self, outbox: dict[str, Any]) -> DiscordProtocolV2OutboxResult:
        """Deliver an already selected outbox row idempotently."""

        outbox_id = str(outbox["outbox_delivery_id"])
        current = self.store.get_outbox_delivery(outbox_id)
        if current is None:
            raise KeyError(outbox_id)
        if current["status"] in {"sent", "acked", "reconciled"}:
            return DiscordProtocolV2OutboxResult(outbox_id, "already_sent")

        if not self.store.has_fresh_outbox_lease(outbox_id, self.worker_id):
            return DiscordProtocolV2OutboxResult(outbox_id, "lease_lost")

        try:
            binding = await self._resolve_client(str(current["target_agent_id"]))
        except MissingDiscordClientError as exc:
            logger.warning(
                "Discord v2 outbox %s has no client for target agent %s",
                outbox_id,
                current.get("target_agent_id"),
            )
            return DiscordProtocolV2OutboxResult(
                outbox_id,
                "missing_client",
                error=str(exc),
            )

        if current["status"] != "sending":
            current = self.store.mark_outbox_sending_if_leased_by(outbox_id, self.worker_id)
            if current is None:
                return DiscordProtocolV2OutboxResult(outbox_id, "lease_lost")

        payload = _decode_payload(current.get("payload_json"))
        chunks = split_discord_message(_extract_content(payload))
        if not chunks:
            if self.store.mark_outbox_sent_if_leased_by(outbox_id, self.worker_id) is None:
                return DiscordProtocolV2OutboxResult(outbox_id, "lease_lost")
            return DiscordProtocolV2OutboxResult(outbox_id, "empty")

        sent_parts = 0
        skipped_parts = 0
        try:
            for part_index, chunk in enumerate(chunks):
                part = _get_outbox_part(self.store, outbox_id, part_index)
                if part and part.get("discord_message_id"):
                    skipped_parts += 1
                    continue
                if part is None:
                    if not _insert_outbox_part_if_missing(
                        self.store,
                        outbox_id,
                        part_index,
                        lease_owner=self.worker_id,
                    ):
                        return DiscordProtocolV2OutboxResult(
                            outbox_id,
                            "lease_lost",
                            sent_parts=sent_parts,
                            skipped_parts=skipped_parts,
                        )

                discord_message_id = await self._send_one_part(
                    binding=binding,
                    outbox=current,
                    payload=payload,
                    part_index=part_index,
                    content=chunk,
                )
                if not self._owns_outbox_lease(outbox_id):
                    return DiscordProtocolV2OutboxResult(
                        outbox_id,
                        "lease_lost",
                        sent_parts=sent_parts,
                        skipped_parts=skipped_parts,
                    )
                self.store.add_outbox_part(
                    outbox_delivery_id=outbox_id,
                    part_index=part_index,
                    status="sent",
                    discord_message_id=discord_message_id,
                )
                record_outbox_message_map(
                    store=self.store,
                    outbox=current,
                    payload=payload,
                    part_index=part_index,
                    content=chunk,
                    discord_message_id=discord_message_id,
                    source_client_agent_id=binding.source_client_agent_id,
                    author_bot_user_id=binding.author_bot_user_id,
                )
                sent_parts += 1
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Discord v2 outbox %s send result uncertain: %s",
                outbox_id,
                type(exc).__name__,
            )
            self.store.mark_outbox_uncertain_if_leased_by(outbox_id, self.worker_id)
            return DiscordProtocolV2OutboxResult(
                outbox_id,
                "uncertain",
                sent_parts=sent_parts,
                skipped_parts=skipped_parts,
                error=type(exc).__name__,
            )

        if self.store.mark_outbox_sent_if_leased_by(outbox_id, self.worker_id) is None:
            return DiscordProtocolV2OutboxResult(
                outbox_id,
                "lease_lost",
                sent_parts=sent_parts,
                skipped_parts=skipped_parts,
            )
        return DiscordProtocolV2OutboxResult(
            outbox_id,
            "sent",
            sent_parts=sent_parts,
            skipped_parts=skipped_parts,
        )

    def _owns_outbox_lease(self, outbox_delivery_id: str) -> bool:
        current = self.store.get_outbox_delivery(outbox_delivery_id)
        if current is None:
            raise KeyError(outbox_delivery_id)
        return self.store.has_fresh_outbox_lease(outbox_delivery_id, self.worker_id, status="sending")

    async def _resolve_client(self, target_agent_id: str) -> DiscordProtocolV2ClientBinding:
        resolved = await _maybe_await(self.client_resolver(target_agent_id))
        identity = self.store.get_identity(target_agent_id) or {}
        binding = _normalize_client_binding(resolved, target_agent_id, identity)
        if binding is None or binding.client is None:
            raise MissingDiscordClientError(target_agent_id)
        if str(binding.source_client_agent_id) != target_agent_id:
            raise MissingDiscordClientError(
                "resolved Discord client for agent "
                f"{binding.source_client_agent_id!r} does not match target agent "
                f"{target_agent_id!r}"
            )
        return binding

    async def _send_one_part(
        self,
        *,
        binding: DiscordProtocolV2ClientBinding,
        outbox: dict[str, Any],
        payload: dict[str, Any],
        part_index: int,
        content: str,
    ) -> str:
        message = await _maybe_await(
            self.send_part(
                binding.client,
                channel_id=str(outbox["channel_id"]),
                thread_id=outbox.get("thread_id"),
                content=content,
                part_index=part_index,
                outbox_delivery=outbox,
                payload=payload,
            )
        )
        message_id = _extract_discord_message_id(message)
        if not message_id:
            raise PossibleSendCommittedError("send returned no Discord message id")
        return message_id

async def default_send_discord_part(
    client: Any,
    *,
    channel_id: str,
    thread_id: str | None = None,
    content: str,
    **_kwargs: Any,
) -> Any:
    """Send one already-split Discord message part with a native client."""

    target_id = thread_id or channel_id
    channel = None
    get_channel = getattr(client, "get_channel", None)
    if callable(get_channel):
        channel = get_channel(int(target_id))
    if channel is None:
        fetch_channel = getattr(client, "fetch_channel", None)
        if not callable(fetch_channel):
            raise MissingDiscordClientError(f"client cannot fetch channel {target_id}")
        channel = await _maybe_await(fetch_channel(int(target_id)))
    if channel is None:
        raise MissingDiscordClientError(f"channel {target_id} not found")
    send = getattr(channel, "send", None)
    if not callable(send):
        raise MissingDiscordClientError(f"channel {target_id} cannot send")
    return await _maybe_await(send(content=content, reference=None))


def record_outbox_message_map(
    *,
    store: DiscordProtocolV2Store,
    outbox: dict[str, Any],
    payload: dict[str, Any],
    part_index: int,
    content: str,
    discord_message_id: str,
    source_client_agent_id: str,
    author_bot_user_id: str | None = None,
) -> None:
    """Persist the Discord message map row for a sent/reconciled outbox part."""

    topic = store.get_topic(str(outbox["topic_id"]))
    if topic is None:
        raise KeyError(f"unknown Discord v2 topic {outbox['topic_id']!r}")
    target_agent_id = str(outbox["target_agent_id"])
    author_bot_user_id = author_bot_user_id or _identity_bot_user_id(
        store,
        target_agent_id,
    )
    direction = "projection" if outbox["delivery_kind"] == "projection" else "outbound"
    store.upsert_message_map(
        discord_message_id=discord_message_id,
        guild_id=str(topic["guild_id"]),
        channel_id=str(outbox["channel_id"]),
        thread_id=outbox.get("thread_id"),
        parent_channel_id=topic.get("parent_channel_id"),
        direction=direction,
        agent_id=target_agent_id,
        delivery_key=outbox.get("source_inbound_delivery_key"),
        outbox_delivery_id=str(outbox["outbox_delivery_id"]),
        agent_event_id=outbox.get("source_agent_event_id"),
        author_id=str(author_bot_user_id or target_agent_id),
        author_kind="registered_bot",
        author_bot_user_id=author_bot_user_id,
        source_client_agent_id=source_client_agent_id,
        mentions=payload.get("mentions") or [],
        payload={
            "part_index": part_index,
            "content": content,
            "outbox_delivery_id": str(outbox["outbox_delivery_id"]),
        },
    )


def _normalize_client_binding(
    value: Any,
    target_agent_id: str,
    identity: dict[str, Any],
) -> DiscordProtocolV2ClientBinding | None:
    if value is None:
        return None
    if isinstance(value, DiscordProtocolV2ClientBinding):
        return value
    if isinstance(value, dict):
        client = value.get("client")
        return DiscordProtocolV2ClientBinding(
            client=client,
            source_client_agent_id=str(value.get("source_client_agent_id") or target_agent_id),
            author_bot_user_id=value.get("author_bot_user_id")
            or identity.get("discord_bot_user_id"),
        )
    client = getattr(value, "client", None)
    if client is not None:
        return DiscordProtocolV2ClientBinding(
            client=client,
            source_client_agent_id=str(
                getattr(value, "source_client_agent_id", None)
                or getattr(value, "agent_id", None)
                or target_agent_id
            ),
            author_bot_user_id=getattr(value, "author_bot_user_id", None)
            or getattr(value, "bot_user_id", None)
            or identity.get("discord_bot_user_id"),
        )
    return DiscordProtocolV2ClientBinding(
        client=value,
        source_client_agent_id=target_agent_id,
        author_bot_user_id=identity.get("discord_bot_user_id"),
    )


def _extract_discord_message_id(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str | int):
        return str(value)
    if isinstance(value, dict):
        for key in ("message_id", "id"):
            if value.get(key) is not None:
                return str(value[key])
        ids = value.get("message_ids")
        if isinstance(ids, list) and ids:
            return str(ids[0])
    for attr in ("message_id", "id"):
        candidate = getattr(value, attr, None)
        if candidate is not None:
            return str(candidate)
    raw_response = getattr(value, "raw_response", None)
    if isinstance(raw_response, dict):
        ids = raw_response.get("message_ids")
        if isinstance(ids, list) and ids:
            return str(ids[0])
    return None


def _extract_content(payload: dict[str, Any]) -> str:
    for key in ("content", "final_response", "text", "message"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return ""


def _decode_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not value:
        return {}
    try:
        decoded = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _get_outbox_part(
    store: DiscordProtocolV2Store,
    outbox_delivery_id: str,
    part_index: int,
) -> dict[str, Any] | None:
    row = store.conn.execute(
        """
        SELECT * FROM outbox_parts
        WHERE outbox_delivery_id = ? AND part_index = ?
        """,
        (outbox_delivery_id, part_index),
    ).fetchone()
    return dict(row) if row is not None else None


def _insert_outbox_part_if_missing(
    store: DiscordProtocolV2Store,
    outbox_delivery_id: str,
    part_index: int,
    *,
    lease_owner: str,
) -> bool:
    cursor = store.conn.execute(
        """
        INSERT OR IGNORE INTO outbox_parts (outbox_delivery_id, part_index, status, discord_message_id)
        SELECT ?, ?, 'pending', NULL
        WHERE EXISTS (
            SELECT 1 FROM outbox_deliveries
            WHERE outbox_delivery_id = ? AND status = 'sending'
              AND lease_owner = ? AND lease_until > ?
        )
        """,
        (outbox_delivery_id, part_index, outbox_delivery_id, lease_owner, _store_now(store)),
    )
    store.conn.commit()
    return (
        cursor.rowcount == 1
        or store.has_fresh_outbox_lease(outbox_delivery_id, lease_owner, status="sending")
        and _get_outbox_part(store, outbox_delivery_id, part_index) is not None
    )


def _store_now(store: DiscordProtocolV2Store) -> str:
    return store.conn.execute("SELECT strftime('%Y-%m-%dT%H:%M:%S+00:00', 'now')").fetchone()[0]


def _identity_bot_user_id(store: DiscordProtocolV2Store, target_agent_id: str) -> str | None:
    identity = store.get_identity(target_agent_id)
    if identity is None:
        return None
    return identity.get("discord_bot_user_id")
