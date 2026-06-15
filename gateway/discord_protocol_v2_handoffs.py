"""Internal handoff/consult/review API for Discord protocol v2.

These helpers are the only local API that should fan out agent-agent
collaboration into target-agent ``inbound_deliveries``.  Discord projections are
represented only as durable outbox rows; the sender slice owns any external
Discord I/O.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from gateway.discord_protocol_v2_events import (
    HANDOFF_STATES,
    REQUEST_EVENT_TYPES,
    AgentEventEnvelope,
    create_agent_event_envelope,
    new_agent_event_id,
    safe_event_payload,
)
from gateway.discord_protocol_v2_outbox import create_projection_outbox_delivery
from gateway.discord_protocol_v2_store import (
    INTERNAL_SOURCE_TYPE,
    DiscordProtocolV2Store,
    projection_idempotency_key,
)

_REQUEST_KIND_TO_EVENT_TYPE = {
    "handoff": "handoff.requested",
    "consult": "consult.requested",
    "review": "review.requested",
}
_STATE_TO_EVENT_TYPE = {
    "accepted": "handoff.accepted",
    "declined": "handoff.declined",
    "completed": "handoff.completed",
    "cancelled": "handoff.cancelled",
}


@dataclass(frozen=True)
class InternalEventResult:
    event: dict[str, Any]
    delivery: dict[str, Any] | None
    outbox_delivery: dict[str, Any] | None
    handoff: dict[str, Any] | None = None


def request_handoff(
    store: DiscordProtocolV2Store,
    *,
    source_agent_id: str | None,
    target_agent_id: str,
    topic_id: str,
    payload: Mapping[str, Any] | None = None,
    agent_event_id: str | None = None,
    idempotency_seed: str | None = None,
    channel_id: str | None = None,
    thread_id: str | None = None,
    projection_content: str | None = None,
) -> InternalEventResult:
    return request_internal_collaboration(
        store,
        kind="handoff",
        source_agent_id=source_agent_id,
        target_agent_id=target_agent_id,
        topic_id=topic_id,
        payload=payload,
        agent_event_id=agent_event_id,
        idempotency_seed=idempotency_seed,
        channel_id=channel_id,
        thread_id=thread_id,
        projection_content=projection_content,
    )


def request_consult(
    store: DiscordProtocolV2Store,
    *,
    source_agent_id: str | None,
    target_agent_id: str,
    topic_id: str,
    payload: Mapping[str, Any] | None = None,
    agent_event_id: str | None = None,
    idempotency_seed: str | None = None,
    channel_id: str | None = None,
    thread_id: str | None = None,
    projection_content: str | None = None,
) -> InternalEventResult:
    return request_internal_collaboration(
        store,
        kind="consult",
        source_agent_id=source_agent_id,
        target_agent_id=target_agent_id,
        topic_id=topic_id,
        payload=payload,
        agent_event_id=agent_event_id,
        idempotency_seed=idempotency_seed,
        channel_id=channel_id,
        thread_id=thread_id,
        projection_content=projection_content,
    )


def request_review(
    store: DiscordProtocolV2Store,
    *,
    source_agent_id: str | None,
    target_agent_id: str,
    topic_id: str,
    payload: Mapping[str, Any] | None = None,
    agent_event_id: str | None = None,
    idempotency_seed: str | None = None,
    channel_id: str | None = None,
    thread_id: str | None = None,
    projection_content: str | None = None,
) -> InternalEventResult:
    return request_internal_collaboration(
        store,
        kind="review",
        source_agent_id=source_agent_id,
        target_agent_id=target_agent_id,
        topic_id=topic_id,
        payload=payload,
        agent_event_id=agent_event_id,
        idempotency_seed=idempotency_seed,
        channel_id=channel_id,
        thread_id=thread_id,
        projection_content=projection_content,
    )


def request_internal_collaboration(
    store: DiscordProtocolV2Store,
    *,
    kind: str,
    source_agent_id: str | None,
    target_agent_id: str,
    topic_id: str,
    payload: Mapping[str, Any] | None = None,
    agent_event_id: str | None = None,
    idempotency_seed: str | None = None,
    channel_id: str | None = None,
    thread_id: str | None = None,
    projection_content: str | None = None,
) -> InternalEventResult:
    """Persist a requested handoff/consult/review and enqueue local fan-out."""

    try:
        event_type = _REQUEST_KIND_TO_EVENT_TYPE[str(kind)]
    except KeyError as exc:
        raise ValueError("kind must be handoff, consult, or review") from exc
    envelope = create_agent_event_envelope(
        event_type=event_type,
        source_agent_id=source_agent_id,
        target_agent_id=target_agent_id,
        topic_id=topic_id,
        payload=payload,
        agent_event_id=agent_event_id,
        idempotency_seed=idempotency_seed,
    )
    return _persist_requested_event(
        store,
        envelope=envelope,
        kind=str(kind),
        channel_id=channel_id,
        thread_id=thread_id,
        projection_content=projection_content,
    )


def accept_handoff(
    store: DiscordProtocolV2Store,
    *,
    handoff_id: str | None = None,
    agent_event_id: str | None = None,
    actor_agent_id: str | None = None,
    payload: Mapping[str, Any] | None = None,
) -> InternalEventResult:
    return transition_handoff(
        store,
        status="accepted",
        handoff_id=handoff_id,
        agent_event_id=agent_event_id,
        actor_agent_id=actor_agent_id,
        payload=payload,
    )


def decline_handoff(
    store: DiscordProtocolV2Store,
    *,
    handoff_id: str | None = None,
    agent_event_id: str | None = None,
    actor_agent_id: str | None = None,
    payload: Mapping[str, Any] | None = None,
) -> InternalEventResult:
    return transition_handoff(
        store,
        status="declined",
        handoff_id=handoff_id,
        agent_event_id=agent_event_id,
        actor_agent_id=actor_agent_id,
        payload=payload,
    )


def complete_handoff(
    store: DiscordProtocolV2Store,
    *,
    handoff_id: str | None = None,
    agent_event_id: str | None = None,
    actor_agent_id: str | None = None,
    payload: Mapping[str, Any] | None = None,
) -> InternalEventResult:
    return transition_handoff(
        store,
        status="completed",
        handoff_id=handoff_id,
        agent_event_id=agent_event_id,
        actor_agent_id=actor_agent_id,
        payload=payload,
    )


def cancel_handoff(
    store: DiscordProtocolV2Store,
    *,
    handoff_id: str | None = None,
    agent_event_id: str | None = None,
    actor_agent_id: str | None = None,
    payload: Mapping[str, Any] | None = None,
) -> InternalEventResult:
    return transition_handoff(
        store,
        status="cancelled",
        handoff_id=handoff_id,
        agent_event_id=agent_event_id,
        actor_agent_id=actor_agent_id,
        payload=payload,
    )


def transition_handoff(
    store: DiscordProtocolV2Store,
    *,
    status: str,
    handoff_id: str | None = None,
    agent_event_id: str | None = None,
    actor_agent_id: str | None = None,
    payload: Mapping[str, Any] | None = None,
) -> InternalEventResult:
    """Persist a durable handoff state transition idempotently."""

    if status not in HANDOFF_STATES or status == "requested":
        raise ValueError("status must be accepted, declined, completed, or cancelled")
    handoff = _get_handoff(store, handoff_id=handoff_id, agent_event_id=agent_event_id)
    if handoff is None:
        raise KeyError("handoff not found")

    request_event_id = str(handoff["agent_event_id"])
    transition_event_id = new_agent_event_id(f"handoff:{request_event_id}:{status}")
    event_payload = safe_event_payload(
        {
            **dict(payload or {}),
            "handoff_id": handoff["handoff_id"],
            "request_agent_event_id": request_event_id,
            "status": status,
        }
    )
    event = store.create_agent_event(
        agent_event_id=transition_event_id,
        event_type=_STATE_TO_EVENT_TYPE[status],
        source_agent_id=actor_agent_id or handoff.get("target_agent_id"),
        target_agent_id=str(handoff.get("source_agent_id") or handoff["target_agent_id"]),
        topic_id=str(handoff["topic_id"]),
        payload=event_payload,
        status=status,
    )
    updated = store.update_handoff_status(
        handoff_id=str(handoff["handoff_id"]),
        status=status,
        payload={**_json_load(handoff.get("payload_json"), {}), **event_payload},
    )
    return InternalEventResult(event=event, delivery=None, outbox_delivery=None, handoff=updated)


def _persist_requested_event(
    store: DiscordProtocolV2Store,
    *,
    envelope: AgentEventEnvelope,
    kind: str,
    channel_id: str | None,
    thread_id: str | None,
    projection_content: str | None,
) -> InternalEventResult:
    if envelope.event_type not in REQUEST_EVENT_TYPES:
        raise ValueError("requested event type required")
    event = store.create_agent_event(
        agent_event_id=envelope.agent_event_id,
        event_type=envelope.event_type,
        source_agent_id=envelope.source_agent_id,
        target_agent_id=envelope.target_agent_id,
        topic_id=envelope.topic_id,
        payload=envelope.payload,
        status="requested",
    )
    delivery = store.create_internal_event_delivery(
        agent_event_id=envelope.agent_event_id,
        target_agent_id=envelope.target_agent_id,
        topic_id=envelope.topic_id,
        route_reason=envelope.event_type,
        payload={**envelope.payload, "agent_event_id": envelope.agent_event_id},
    )
    handoff = None
    if envelope.event_type == "handoff.requested":
        handoff = store.upsert_handoff(
            handoff_id=f"handoff:{envelope.agent_event_id}",
            agent_event_id=envelope.agent_event_id,
            source_agent_id=envelope.source_agent_id,
            target_agent_id=envelope.target_agent_id,
            topic_id=envelope.topic_id,
            status="requested",
            payload=envelope.payload,
        )
    store.record_route_decision(
        source_type=INTERNAL_SOURCE_TYPE,
        source_id=envelope.agent_event_id,
        topic_id=envelope.topic_id,
        author_kind="registered_bot",
        decision="delivered",
        target_agent_ids=[envelope.target_agent_id],
        reason=envelope.event_type,
        payload=envelope.payload,
    )
    outbox = _create_projection(
        store,
        envelope=envelope,
        kind=kind,
        channel_id=channel_id,
        thread_id=thread_id,
        projection_content=projection_content,
    )
    return InternalEventResult(event=event, delivery=delivery, outbox_delivery=outbox, handoff=handoff)


def _create_projection(
    store: DiscordProtocolV2Store,
    *,
    envelope: AgentEventEnvelope,
    kind: str,
    channel_id: str | None,
    thread_id: str | None,
    projection_content: str | None,
) -> dict[str, Any]:
    topic = store.get_topic(envelope.topic_id)
    effective_channel_id = channel_id or (str(topic["channel_id"]) if topic else "")
    if not effective_channel_id:
        raise ValueError("channel_id is required when topic is not persisted")
    effective_thread_id = thread_id if thread_id is not None else (topic or {}).get("thread_id")
    mention_id = _target_bot_mention_id(store, envelope.target_agent_id)
    mentions = [mention_id] if mention_id else []
    content = projection_content or _default_projection_content(
        kind=kind,
        envelope=envelope,
        mention_id=mention_id,
    )
    return create_projection_outbox_delivery(
        store,
        agent_event_id=envelope.agent_event_id,
        target_agent_id=envelope.target_agent_id,
        topic_id=envelope.topic_id,
        channel_id=effective_channel_id,
        thread_id=effective_thread_id,
        content=content,
        mentions=mentions,
        payload={
            "content": content,
            "mentions": mentions,
            "agent_event_id": envelope.agent_event_id,
            "event_type": envelope.event_type,
            "source_agent_id": envelope.source_agent_id,
            "target_agent_id": envelope.target_agent_id,
            "idempotency_key": projection_idempotency_key(
                envelope.agent_event_id,
                envelope.target_agent_id,
            ),
        },
    )


def _default_projection_content(
    *,
    kind: str,
    envelope: AgentEventEnvelope,
    mention_id: str | None,
) -> str:
    mention = f"<@{mention_id}>" if mention_id else envelope.target_agent_id
    source = envelope.source_agent_id or "internal"
    summary = envelope.payload.get("summary") or envelope.payload.get("task") or envelope.event_type
    return f"[{kind}] {mention} requested by {source}: {summary}"


def _target_bot_mention_id(store: DiscordProtocolV2Store, target_agent_id: str) -> str | None:
    identity = store.get_identity(target_agent_id)
    if identity is None:
        return None
    value = identity.get("discord_bot_user_id")
    return str(value) if value else None


def _get_handoff(
    store: DiscordProtocolV2Store,
    *,
    handoff_id: str | None,
    agent_event_id: str | None,
) -> dict[str, Any] | None:
    if handoff_id:
        return store.get_handoff(str(handoff_id))
    if agent_event_id:
        return store.get_handoff_by_agent_event(str(agent_event_id))
    raise ValueError("handoff_id or agent_event_id is required")


def _json_load(value: Any, default: dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    try:
        decoded = json.loads(str(value or ""))
    except Exception:
        return default
    return decoded if isinstance(decoded, dict) else default
