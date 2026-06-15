from __future__ import annotations

"""Discord native multibot adapter.

Starts one Discord client runtime per configured identity in
``listen_only``/``shadow``/``active`` mode.  Listen-only modes use the durable
protocol-v2 ingest path for observed messages without inline replies.  Active
mode can tick the durable protocol-v2 outbox through each target identity's own
Discord client.
"""

import asyncio
import hashlib
import inspect
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from gateway.config import DiscordNativeMultibotConfig, Platform, PlatformConfig
from gateway.discord_identity_registry import DiscordIdentityRegistry
from gateway.discord_protocol_v2_ingest import DiscordProtocolV2Ingestor
from gateway.discord_protocol_v2_outbox import (
    DiscordProtocolV2ClientBinding,
    DiscordProtocolV2OutboxResult,
    DiscordProtocolV2OutboxSender,
)
from gateway.discord_protocol_v2_reconcile import (
    DiscordProtocolV2ReconciliationResult,
    RecentHistoryFetcher,
    reconcile_discord_protocol_v2_outbox,
)
from gateway.discord_protocol_v2_store import DiscordProtocolV2Store
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.secret_refs import GatewaySecretResolver, SecretResolver, redact_secret_ref
from plugins.platforms.discord.client_runtime import (
    DiscordClientRuntime,
    DiscordRuntimeEventHandlers,
)

logger = logging.getLogger(__name__)


@dataclass
class NativeRuntimeState:
    agent_id: str
    token_ref_fingerprint: str
    runtime: DiscordClientRuntime | None = None
    ready_event: asyncio.Event = field(default_factory=asyncio.Event)
    bot_task: asyncio.Task | None = None
    status: str = "pending"
    error: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "token_ref_fingerprint": self.token_ref_fingerprint,
            "status": self.status,
            "error": self.error,
        }


def _token_ref_fingerprint(token_secret_ref: str) -> str:
    return hashlib.sha256(token_secret_ref.encode("utf-8")).hexdigest()[:16]


def _message_attr(obj: Any, name: str, default: Any = None) -> Any:
    return getattr(obj, name, default)


def _string_id(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _resolve_history_channel(client: Any, target_id: Any) -> Any | None:
    channel_id = _coerce_discord_id(target_id)
    get_channel = getattr(client, "get_channel", None)
    if callable(get_channel):
        channel = get_channel(channel_id)
        if channel is not None:
            return channel
    fetch_channel = getattr(client, "fetch_channel", None)
    if callable(fetch_channel):
        return await _maybe_await(fetch_channel(channel_id))
    return None


def _coerce_discord_id(value: Any) -> Any:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return value


async def _collect_history_messages(history_value: Any) -> list[Any]:
    history_value = await _maybe_await(history_value)
    if hasattr(history_value, "__aiter__"):
        messages = []
        async for message in history_value:
            messages.append(message)
        return messages
    if isinstance(history_value, str | bytes):
        return []
    try:
        return list(history_value)
    except TypeError:
        return []


def _history_evidence(
    message: Any,
    outbox: dict[str, Any],
    binding: DiscordProtocolV2ClientBinding,
) -> dict[str, Any]:
    author = _message_attr(message, "author")
    author_id = _string_id(_message_attr(author, "id"), default="")
    evidence: dict[str, Any] = {
        "id": _string_id(_message_attr(message, "id")),
        "content": str(_message_attr(message, "content", "") or ""),
        # History was fetched from this outbox target; use the persisted outbox
        # channel/thread ids so thread history does not compare parent-vs-thread
        # channel ids incorrectly during reconciliation.
        "channel_id": str(outbox["channel_id"]),
        "thread_id": outbox.get("thread_id"),
        "author_id": author_id,
    }
    if author_id and author_id == str(binding.author_bot_user_id or ""):
        evidence["author_bot_user_id"] = binding.author_bot_user_id
    return evidence


class DiscordNativeMultibotAdapter(BasePlatformAdapter):
    """Protocol-v2 native Discord multibot adapter, listen-only for now."""

    def __init__(
        self,
        config: PlatformConfig,
        *,
        native_config: DiscordNativeMultibotConfig,
        store: DiscordProtocolV2Store | None = None,
        identity_registry: DiscordIdentityRegistry | None = None,
        secret_resolver: SecretResolver | None = None,
        runtime_factory: Callable[..., DiscordClientRuntime] = DiscordClientRuntime,
        bot_factory: Callable[..., Any] | None = None,
        intents_factory: Any | None = None,
        allowed_mentions_factory: Callable[[], Any] | None = None,
        proxy_kwargs_factory: Callable[[str | None], dict[str, Any]] | None = None,
        ready_timeout_seconds: float = 30.0,
        startup_reconciliation_enabled: bool | None = None,
        recent_history_fetcher: RecentHistoryFetcher | None = None,
    ) -> None:
        super().__init__(config, Platform.DISCORD)
        self.native_config = native_config
        self.gateway_runner = None
        self.store = store or DiscordProtocolV2Store()
        self._owns_store = store is None
        self.identity_registry = identity_registry
        self.secret_resolver = secret_resolver
        self.runtime_factory = runtime_factory
        self.bot_factory = bot_factory
        self.intents_factory = intents_factory
        self.allowed_mentions_factory = allowed_mentions_factory
        self.proxy_kwargs_factory = proxy_kwargs_factory
        self.ready_timeout_seconds = ready_timeout_seconds
        self.startup_reconciliation_enabled = (
            native_config.mode == "active"
            if startup_reconciliation_enabled is None
            else bool(startup_reconciliation_enabled)
        )
        self.recent_history_fetcher = recent_history_fetcher
        self.startup_reconciliation_result: DiscordProtocolV2ReconciliationResult | None = None
        self.runtime_states: dict[str, NativeRuntimeState] = {}
        self.ingestor: DiscordProtocolV2Ingestor | None = None
        self._lock_keys: list[str] = []

    async def connect(self) -> bool:
        """Start one Discord runtime per enabled v2 identity."""

        if not self.native_config.enabled or self.native_config.mode not in {
            "listen_only",
            "shadow",
            "active",
        }:
            self._set_fatal_error(
                "discord_native_multibot_disabled",
                "Discord native multibot adapter requires listen_only/shadow/active mode",
                retryable=False,
            )
            return False

        from plugins.platforms.discord import adapter as discord_adapter

        if not discord_adapter.DISCORD_AVAILABLE:
            logger.error("[%s] discord.py not installed. Run: pip install discord.py", self.name)
            return False

        from gateway.platforms.base import resolve_proxy_url, proxy_kwargs_for_bot

        if self.identity_registry is None:
            self.identity_registry = DiscordIdentityRegistry.load(
                self.native_config,
                self.store,
                self.secret_resolver,
            )

        bot_factory = self.bot_factory or discord_adapter.commands.Bot
        intents_factory = self.intents_factory or discord_adapter.Intents
        allowed_mentions_factory = (
            self.allowed_mentions_factory or discord_adapter._build_allowed_mentions
        )
        proxy_kwargs_factory = self.proxy_kwargs_factory or proxy_kwargs_for_bot
        proxy_url = resolve_proxy_url(platform_env_var="DISCORD_PROXY")

        self.ingestor = DiscordProtocolV2Ingestor(
            store=self.store,
            identity_registry=self.identity_registry,
            default_intake_agent_id=self.native_config.default_intake_agent_id,
            guild_allowlist=self.native_config.guild_allowlist,
            mode=self.native_config.mode,
        )

        active_agent_ids = list(self.identity_registry.active_agent_ids)
        if not active_agent_ids:
            self._set_fatal_error(
                "discord_native_multibot_no_identities",
                "Discord native multibot has no active identities",
                retryable=False,
            )
            return False

        connected = 0
        for agent_id in active_agent_ids:
            state = await self._start_identity_runtime(
                agent_id=agent_id,
                bot_factory=bot_factory,
                intents_factory=intents_factory,
                allowed_mentions_factory=allowed_mentions_factory,
                proxy_kwargs_factory=proxy_kwargs_factory,
                proxy_url=proxy_url,
            )
            if state.status == "connected":
                connected += 1

        self._running = connected > 0
        if not self._running:
            self._set_fatal_error(
                "discord_native_multibot_no_clients",
                "No Discord native multibot identities connected",
                retryable=True,
            )
        else:
            await self._run_startup_reconciliation()
        return self._running

    async def _run_startup_reconciliation(self) -> None:
        """Reconcile crash-sticky protocol-v2 outbox rows during native startup.

        This hook is intentionally conservative: it only runs for configured
        ``active`` native mode, never for listen-only/shadow modes, so startup
        cannot mutate or enqueue outbox deliveries in modes that must not send.
        The history fetcher is injectable for tests; the default production
        fetcher performs read-only recent-history lookups through connected
        native clients and never sends messages.
        """

        if self.native_config.mode != "active" or not self.startup_reconciliation_enabled:
            return

        fetcher = self.recent_history_fetcher or self.fetch_recent_outbox_history
        try:
            result = await reconcile_discord_protocol_v2_outbox(
                store=self.store,
                recent_history_fetcher=fetcher,
                run_id=f"discord-native-startup:{uuid.uuid4().hex}",
            )
        except Exception:  # pragma: no cover - defensive startup guard
            logger.warning(
                "[%s] Discord native startup outbox reconciliation failed",
                self.name,
                exc_info=True,
            )
            return

        self.startup_reconciliation_result = result
        logger.info(
            "[%s] Discord native startup outbox reconciliation: "
            "scanned=%s acked=%s enqueued=%s exhausted=%s",
            self.name,
            result.scanned,
            result.acked,
            result.enqueued,
            result.exhausted,
        )

    async def _start_identity_runtime(
        self,
        *,
        agent_id: str,
        bot_factory: Callable[..., Any],
        intents_factory: Any,
        allowed_mentions_factory: Callable[[], Any],
        proxy_kwargs_factory: Callable[[str | None], dict[str, Any]],
        proxy_url: str | None,
    ) -> NativeRuntimeState:
        assert self.identity_registry is not None
        identity = self.identity_registry.get_identity(agent_id, include_disabled=False)
        if identity is None:
            raise KeyError(f"unknown active identity {agent_id!r}")

        lock_key = _token_ref_fingerprint(identity.token_secret_ref)
        state = NativeRuntimeState(agent_id=agent_id, token_ref_fingerprint=lock_key)
        self.runtime_states[agent_id] = state

        if not self._acquire_runtime_lock(lock_key):
            state.status = "failed"
            state.error = "token_ref_lock_unavailable"
            return state

        runtime = self.runtime_factory(
            agent_id=agent_id,
            bot_user_id=identity.discord_bot_user_id,
            token_resolver=self.identity_registry.resolve_token,
            bot_factory=bot_factory,
            intents_factory=intents_factory,
            allowed_mentions_factory=allowed_mentions_factory,
            proxy_kwargs_factory=proxy_kwargs_factory,
        )
        state.runtime = runtime

        async def on_ready() -> None:
            if state.status in {"failed", "disconnected"}:
                return
            state.status = "connected"
            state.error = None
            state.ready_event.set()
            logger.info("[%s] Discord native identity connected: %s", self.name, agent_id)

        async def on_message(message: Any) -> None:
            await self._handle_listen_only_message(agent_id, identity.discord_bot_user_id, message)

        try:
            runtime.create_client(
                allowed_user_ids=set(),
                allowed_role_ids=set(),
                proxy_url=proxy_url,
            )
            runtime.register_event_handlers(
                DiscordRuntimeEventHandlers(on_ready=on_ready, on_message=on_message)
            )
            token = runtime.resolve_token()
            state.bot_task = runtime.start(token)
            await asyncio.sleep(0)
            if state.bot_task.done():
                exc = state.bot_task.exception()
                if exc is not None:
                    raise exc
            await asyncio.wait_for(
                state.ready_event.wait(), timeout=self.ready_timeout_seconds
            )
        except Exception as exc:  # pragma: no cover - exercised in tests
            state.status = "failed"
            state.error = type(exc).__name__
            await self._cleanup_failed_runtime(state)
            self._release_runtime_lock(lock_key)
            logger.warning(
                "[%s] Discord native identity %s failed to start: %s",
                self.name,
                agent_id,
                type(exc).__name__,
            )
        return state

    async def _cleanup_failed_runtime(self, state: NativeRuntimeState) -> None:
        runtime = state.runtime
        if state.bot_task and not state.bot_task.done():
            state.bot_task.cancel()
            try:
                await state.bot_task
            except asyncio.CancelledError:
                pass
        if runtime is not None:
            try:
                await runtime.close_existing_client(runtime.client)
            except Exception:
                logger.debug(
                    "[%s] Failed to close failed Discord native identity %s",
                    self.name,
                    state.agent_id,
                    exc_info=True,
                )

    def _acquire_runtime_lock(self, lock_key: str) -> bool:
        from gateway.status import acquire_scoped_lock

        acquired, existing = acquire_scoped_lock(
            "discord-native-token-ref",
            lock_key,
            metadata={"platform": self.platform.value},
        )
        if acquired:
            self._lock_keys.append(lock_key)
            return True
        owner_pid = existing.get("pid") if isinstance(existing, dict) else None
        logger.error(
            "[%s] Discord native token ref already in use%s",
            self.name,
            f" (PID {owner_pid})" if owner_pid else "",
        )
        return False

    def _release_runtime_lock(self, lock_key: str) -> None:
        from gateway.status import release_scoped_lock

        if lock_key not in self._lock_keys:
            return
        release_scoped_lock("discord-native-token-ref", lock_key)
        self._lock_keys.remove(lock_key)

    def _release_runtime_locks(self) -> None:
        from gateway.status import release_scoped_lock

        for lock_key in list(self._lock_keys):
            release_scoped_lock("discord-native-token-ref", lock_key)
        self._lock_keys.clear()

    async def disconnect(self) -> None:
        for state in self.runtime_states.values():
            runtime = state.runtime
            client = runtime.client if runtime is not None else None
            if runtime is not None:
                try:
                    await runtime.close_existing_client(client)
                except Exception:
                    logger.debug(
                        "[%s] Failed to close Discord native identity %s",
                        self.name,
                        state.agent_id,
                        exc_info=True,
                    )
            if state.bot_task and not state.bot_task.done():
                state.bot_task.cancel()
                try:
                    await state.bot_task
                except asyncio.CancelledError:
                    pass
            state.status = "disconnected"
        self._release_runtime_locks()
        self._running = False
        if self._owns_store:
            self.store.close()
        logger.info("[%s] Discord native multibot disconnected", self.name)

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SendResult:
        return SendResult(
            success=False,
            error="discord_native_multibot is listen-only in this mode",
            retryable=False,
        )

    def resolve_outbox_client(self, target_agent_id: str) -> DiscordProtocolV2ClientBinding | None:
        """Return the live native client binding for an outbox target agent."""

        if self.identity_registry is None:
            return None
        state = self.runtime_states.get(target_agent_id)
        if state is None or state.status != "connected" or state.runtime is None:
            return None
        client = state.runtime.client
        if client is None:
            return None
        identity = self.identity_registry.get_identity(target_agent_id, include_disabled=False)
        if identity is None:
            return None
        return DiscordProtocolV2ClientBinding(
            client=client,
            source_client_agent_id=target_agent_id,
            author_bot_user_id=identity.discord_bot_user_id,
        )

    async def fetch_recent_outbox_history(self, outbox: dict[str, Any]) -> list[dict[str, Any]]:
        """Read recent Discord history for startup reconciliation evidence.

        This is read-only and is only wired by ``_run_startup_reconciliation`` in
        active mode. Missing clients/channels simply produce no evidence; API
        errors propagate so reconciliation keeps the affected row uncertain
        rather than blindly re-enqueueing it.
        """

        binding = self.resolve_outbox_client(str(outbox["target_agent_id"]))
        if binding is None:
            return []
        channel = await _resolve_history_channel(
            binding.client,
            outbox.get("thread_id") or outbox["channel_id"],
        )
        if channel is None:
            return []
        history = getattr(channel, "history", None)
        if not callable(history):
            return []
        messages = await _collect_history_messages(history(limit=25))
        return [_history_evidence(message, outbox, binding) for message in messages]

    async def run_outbox_once(self) -> DiscordProtocolV2OutboxResult | None:
        """Send at most one durable outbox delivery in active mode.

        This bounded helper is intentionally not a forever-loop; gateway wiring
        can opt in explicitly without changing listen-only behavior.
        """

        if self.native_config.mode != "active":
            return None
        sender = DiscordProtocolV2OutboxSender(
            store=self.store,
            client_resolver=self.resolve_outbox_client,
        )
        return await sender.run_once()

    async def get_chat_info(self, chat_id: str) -> dict[str, Any]:
        return {
            "chat_id": chat_id,
            "platform": self.platform.value,
            "mode": self.native_config.mode,
            "listen_only": True,
        }

    async def _handle_listen_only_message(
        self,
        agent_id: str,
        bot_user_id: str,
        message: Any,
    ) -> None:
        """Persist a durable listen-only observation without invoking Hermes."""

        if self.ingestor is None:
            assert self.identity_registry is not None
            self.ingestor = DiscordProtocolV2Ingestor(
                store=self.store,
                identity_registry=self.identity_registry,
                default_intake_agent_id=self.native_config.default_intake_agent_id,
                guild_allowlist=self.native_config.guild_allowlist,
                mode=self.native_config.mode,
            )
        self.ingestor.ingest_message(
            source_client_agent_id=agent_id,
            message=message,
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "mode": self.native_config.mode,
            "running": self._running,
            "identities": [
                self.runtime_states[agent_id].snapshot()
                for agent_id in sorted(self.runtime_states)
            ],
            "guild_allowlist": list(self.native_config.guild_allowlist),
            "token_secret_refs": [
                redact_secret_ref(identity.token_secret_ref)
                for identity in self.identity_registry.identities.values()
            ]
            if self.identity_registry is not None
            else [],
        }


def is_native_multibot_enabled(config: DiscordNativeMultibotConfig) -> bool:
    return bool(config.enabled and config.mode in {"listen_only", "shadow", "active"})


def build_native_multibot_adapter(
    platform_config: PlatformConfig,
    native_config: DiscordNativeMultibotConfig,
    **kwargs: Any,
) -> DiscordNativeMultibotAdapter:
    if kwargs.get("secret_resolver") is None:
        kwargs["secret_resolver"] = GatewaySecretResolver()
    return DiscordNativeMultibotAdapter(
        platform_config,
        native_config=native_config,
        **kwargs,
    )
