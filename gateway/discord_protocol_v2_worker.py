"""Agent invocation worker for Discord Native Multi-Bot Protocol v2.

The worker consumes durable inbound deliveries, invokes a Hermes agent through an
injectable runtime hook, and persists responses into the durable outbox.  It does
not send Discord messages inline; sender/reconciliation slices own outbox I/O.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from gateway.config import Platform
from gateway.discord_identity_registry import DiscordIdentityMetadata, DiscordIdentityRegistry
from gateway.discord_protocol_v2_sessions import (
    DiscordV2TopicAgentSession,
    get_or_create_discord_v2_session,
)
from gateway.discord_protocol_v2_store import (
    DiscordProtocolV2Store,
    response_idempotency_key,
)
from gateway.session import SessionSource

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiscordProtocolV2InvocationContext:
    """Context passed to the Hermes invocation hook for one inbound delivery."""

    delivery: dict[str, Any]
    identity: DiscordIdentityMetadata | dict[str, Any]
    topic: dict[str, Any]
    session: DiscordV2TopicAgentSession
    session_source: SessionSource
    message_text: str
    channel_prompt: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DiscordProtocolV2WorkerResult:
    """Result of one worker tick."""

    delivery_key: str
    outbox_delivery: dict[str, Any] | None
    status: str


Invoker = Callable[[DiscordProtocolV2InvocationContext], Any]


class LeaseLostDuringInvocation(Exception):
    """Raised when an in-flight invocation no longer owns its inbound lease."""


class DiscordProtocolV2Worker:
    """Lease inbound Discord v2 deliveries and enqueue agent responses."""

    def __init__(
        self,
        *,
        store: DiscordProtocolV2Store,
        identity_registry: DiscordIdentityRegistry,
        session_store: Any,
        invoker: Invoker,
        worker_id: str | None = None,
        lease_seconds: int = 60,
    ) -> None:
        self.store = store
        self.identity_registry = identity_registry
        self.session_store = session_store
        self.invoker = invoker
        self.worker_id = worker_id or f"discord-v2-worker:{uuid.uuid4().hex[:12]}"
        self.lease_seconds = int(lease_seconds)
        self._stop_event = asyncio.Event()

    async def run_once(self, *, target_agent_id: str | None = None) -> DiscordProtocolV2WorkerResult | None:
        """Lease and process at most one inbound delivery."""

        delivery = self.store.lease_next_inbound(
            lease_owner=self.worker_id,
            lease_seconds=self.lease_seconds,
            target_agent_id=target_agent_id,
        )
        if delivery is None:
            return None
        return await self.process_delivery(delivery)

    async def run_forever(self, *, idle_sleep_seconds: float = 1.0) -> None:
        """Run bounded ticks until stopped or cancelled."""

        while not self._stop_event.is_set():
            try:
                result = await self.run_once()
                if result is None:
                    await asyncio.sleep(idle_sleep_seconds)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Discord v2 worker tick failed")
                await asyncio.sleep(idle_sleep_seconds)

    def stop(self) -> None:
        self._stop_event.set()

    async def process_delivery(self, delivery: dict[str, Any]) -> DiscordProtocolV2WorkerResult:
        """Invoke Hermes for an already-leased inbound delivery."""

        delivery_key = str(delivery["delivery_key"])
        target_agent_id = str(delivery["target_agent_id"])
        idempotency_key = response_idempotency_key(delivery_key, target_agent_id)

        existing = self.store.get_outbox_delivery_by_key(idempotency_key)
        if existing is not None:
            self._complete_if_leased(delivery_key)
            return DiscordProtocolV2WorkerResult(
                delivery_key=delivery_key,
                outbox_delivery=existing,
                status="already_enqueued",
            )

        if not self._owns_delivery_lease(delivery_key):
            return DiscordProtocolV2WorkerResult(
                delivery_key=delivery_key,
                outbox_delivery=None,
                status="lease_lost",
            )

        try:
            if self.store.refresh_inbound_lease_if_leased_by(
                delivery_key,
                self.worker_id,
                self.lease_seconds,
            ) is None:
                return DiscordProtocolV2WorkerResult(
                    delivery_key=delivery_key,
                    outbox_delivery=None,
                    status="lease_lost",
                )
            context = self._build_invocation_context(delivery)
            raw_response = await self._invoke_with_lease_heartbeat(delivery_key, context)
            response_text = _extract_response_text(raw_response)
            if not self._owns_delivery_lease(delivery_key):
                return DiscordProtocolV2WorkerResult(
                    delivery_key=delivery_key,
                    outbox_delivery=None,
                    status="lease_lost",
                )
            if not response_text:
                self._complete_if_leased(delivery_key)
                return DiscordProtocolV2WorkerResult(
                    delivery_key=delivery_key,
                    outbox_delivery=None,
                    status="completed_no_response",
                )

            outbox = self.store.create_outbox_delivery(
                idempotency_key=idempotency_key,
                target_agent_id=target_agent_id,
                topic_id=str(delivery["topic_id"]),
                channel_id=str(context.topic["channel_id"]),
                thread_id=context.topic.get("thread_id"),
                delivery_kind="response",
                source_inbound_delivery_key=delivery_key,
                source_agent_event_id=delivery.get("agent_event_id"),
                payload={
                    "content": response_text,
                    "target_agent_id": target_agent_id,
                    "source_inbound_delivery_key": delivery_key,
                    "hermes_profile": _identity_field(context.identity, "hermes_profile"),
                },
            )
            self._complete_if_leased(delivery_key)
            return DiscordProtocolV2WorkerResult(
                delivery_key=delivery_key,
                outbox_delivery=outbox,
                status="completed",
            )
        except LeaseLostDuringInvocation:
            return DiscordProtocolV2WorkerResult(
                delivery_key=delivery_key,
                outbox_delivery=None,
                status="lease_lost",
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Discord v2 delivery failed: %s", delivery_key)
            try:
                self._retry_if_leased(delivery_key)
            except Exception:
                logger.debug("Failed to mark Discord v2 delivery retryable", exc_info=True)
            raise

    async def _invoke_with_lease_heartbeat(
        self,
        delivery_key: str,
        context: DiscordProtocolV2InvocationContext,
    ) -> Any:
        """Invoke Hermes while periodically extending the inbound lease."""

        task = asyncio.create_task(_maybe_await(self.invoker(context)))
        heartbeat_seconds = max(0.05, min(30.0, float(self.lease_seconds) / 2.0))
        try:
            while True:
                done, _pending = await asyncio.wait({task}, timeout=heartbeat_seconds)
                if done:
                    return task.result()
                if self.store.refresh_inbound_lease_if_leased_by(
                    delivery_key,
                    self.worker_id,
                    self.lease_seconds,
                ) is None:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                    raise LeaseLostDuringInvocation(delivery_key)
        except asyncio.CancelledError:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            raise

    def _owns_delivery_lease(self, delivery_key: str) -> bool:
        current = self.store.get_inbound_delivery(delivery_key)
        if current is None:
            raise KeyError(delivery_key)
        return current["status"] == "leased" and current.get("lease_owner") == self.worker_id

    def _complete_if_leased(self, delivery_key: str) -> None:
        if self._owns_delivery_lease(delivery_key):
            self.store.complete_inbound_if_leased_by(delivery_key, self.worker_id)

    def _retry_if_leased(self, delivery_key: str) -> None:
        if self._owns_delivery_lease(delivery_key):
            self.store.retry_inbound_if_leased_by(delivery_key, self.worker_id)

    def _build_invocation_context(
        self,
        delivery: dict[str, Any],
    ) -> DiscordProtocolV2InvocationContext:
        target_agent_id = str(delivery["target_agent_id"])
        identity = self.identity_registry.get_identity(target_agent_id, include_disabled=False)
        if identity is None:
            raise KeyError(f"unknown or inactive Discord v2 identity {target_agent_id!r}")

        topic_id = str(delivery["topic_id"])
        topic = self.store.get_topic(topic_id)
        if topic is None:
            raise KeyError(f"unknown Discord v2 topic {topic_id!r}")

        payload = _decode_payload(delivery.get("payload_json"))
        source = build_discord_v2_session_source(
            delivery=delivery,
            topic=topic,
            payload=payload,
            target_agent_id=target_agent_id,
            hermes_profile=identity.hermes_profile,
        )
        session = get_or_create_discord_v2_session(
            protocol_store=self.store,
            session_store=self.session_store,
            topic_id=topic_id,
            agent_id=target_agent_id,
            source=source,
        )
        metadata = {
            "platform": "discord",
            "protocol": "discord_native_multibot_v2",
            "delivery_key": str(delivery["delivery_key"]),
            "source_type": str(delivery["source_type"]),
            "source_id": str(delivery["source_id"]),
            "agent_event_id": delivery.get("agent_event_id"),
            "target_agent_id": target_agent_id,
            "hermes_profile": identity.hermes_profile,
            "topic_id": topic_id,
            "guild_id": topic.get("guild_id"),
            "channel_id": topic.get("channel_id"),
            "thread_id": topic.get("thread_id"),
            "route_reason": delivery.get("route_reason"),
        }
        return DiscordProtocolV2InvocationContext(
            delivery=delivery,
            identity=identity,
            topic=topic,
            session=session,
            session_source=source,
            message_text=str(payload.get("content") or ""),
            channel_prompt=build_discord_v2_channel_prompt(metadata),
            metadata=metadata,
        )


def build_discord_v2_session_source(
    *,
    delivery: dict[str, Any],
    topic: dict[str, Any],
    payload: dict[str, Any],
    target_agent_id: str,
    hermes_profile: str,
) -> SessionSource:
    """Build the Discord SessionSource for a topic × target-agent delivery."""

    thread_id = topic.get("thread_id")
    return SessionSource(
        platform=Platform.DISCORD,
        chat_id=str(topic["channel_id"]),
        chat_name=str(topic.get("title") or topic["channel_id"]),
        chat_type="thread" if thread_id else "channel",
        user_id=str(payload.get("author_id") or "") or None,
        user_name=payload.get("author_name"),
        thread_id=str(thread_id) if thread_id else None,
        guild_id=str(topic.get("guild_id") or "") or None,
        parent_chat_id=str(topic.get("parent_channel_id") or "") or None,
        message_id=delivery.get("discord_message_id") or payload.get("discord_message_id"),
        chat_topic=(
            "Discord Native Multi-Bot Protocol v2 topic "
            f"{delivery['topic_id']} for target agent {target_agent_id} "
            f"(Hermes profile {hermes_profile})."
        ),
    )


def build_discord_v2_channel_prompt(metadata: dict[str, Any]) -> str:
    """Return a compact prompt fragment with v2 routing identity."""

    return (
        "[Discord protocol v2 delivery]\n"
        f"Target agent: {metadata['target_agent_id']}\n"
        f"Hermes profile: {metadata['hermes_profile']}\n"
        f"Topic: {metadata['topic_id']}\n"
        f"Delivery key: {metadata['delivery_key']}\n"
        f"Source type: {metadata['source_type']}"
    )


class GatewayRunnerDiscordV2Invoker:
    """Adapter from worker invocations to ``GatewayRunner._run_agent``."""

    def __init__(self, runner: Any) -> None:
        self.runner = runner

    async def __call__(self, context: DiscordProtocolV2InvocationContext) -> dict[str, Any]:
        session_entry = self.runner.session_store.bind_session_key(
            context.session.session_key,
            context.session.session_id,
            context.session_source,
        )
        history = self.runner.session_store.load_transcript(session_entry.session_id)
        from gateway.session import build_session_context, build_session_context_prompt

        context_prompt = build_session_context_prompt(
            build_session_context(context.session_source, self.runner.config, session_entry),
        )
        context_prompt = f"{context_prompt}\n\n{context.channel_prompt}"
        run_generation = self.runner._begin_session_run_generation(context.session.session_key)
        return await self.runner._run_agent(
            message=context.message_text,
            context_prompt=context_prompt,
            history=history,
            source=context.session_source,
            session_id=session_entry.session_id,
            session_key=context.session.session_key,
            run_generation=run_generation,
            event_message_id=context.session_source.message_id,
            channel_prompt=context.channel_prompt,
            event=None,
            hermes_profile=_identity_field(context.identity, "hermes_profile"),
            hermes_home=_resolve_hermes_profile_home(
                _identity_field(context.identity, "hermes_profile")
            ),
            suppress_inline_delivery=True,
        )


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


def _extract_response_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("content", "final_response", "text", "response"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return ""
    for attr in ("content", "final_response", "text", "response"):
        candidate = getattr(value, attr, None)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return str(value).strip()


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _identity_field(identity: DiscordIdentityMetadata | dict[str, Any], field: str) -> Any:
    if isinstance(identity, dict):
        return identity.get(field)
    return getattr(identity, field)


def _resolve_hermes_profile_home(profile: str | None) -> str | None:
    """Resolve a target Hermes profile home without mutating process globals."""

    raw_profile = str(profile or "").strip()
    if not raw_profile:
        return None
    try:
        from hermes_cli.profiles import normalize_profile_name, resolve_profile_env

        return str(resolve_profile_env(normalize_profile_name(raw_profile)))
    except (FileNotFoundError, ValueError):
        logger.warning("Discord v2 worker profile %r is unavailable; using active profile", raw_profile)
        return None
