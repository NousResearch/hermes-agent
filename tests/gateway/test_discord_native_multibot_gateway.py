from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from gateway.config import (
    DiscordNativeMultibotConfig,
    DiscordNativeMultibotIdentityConfig,
    PlatformConfig,
)
from gateway.discord_protocol_v2_store import DiscordProtocolV2Store
from gateway.secret_refs import (
    GatewaySecretResolver,
    SecretResolutionError,
    StaticSecretResolver,
)
from plugins.platforms.discord import adapter as discord_adapter
from plugins.platforms.discord.adapter import DiscordAdapter
from plugins.platforms.discord.native_multibot import DiscordNativeMultibotAdapter


class FakeIntents:
    @staticmethod
    def default():
        return SimpleNamespace(
            message_content=False,
            dm_messages=False,
            guild_messages=False,
            members=False,
            voice_states=False,
        )


class FakeBot:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._events = {}
        self.closed = False
        self.started_with = None

    def event(self, fn):
        self._events[fn.__name__] = fn
        return fn

    def is_closed(self):
        return self.closed

    async def close(self):
        self.closed = True

    async def start(self, token):
        self.started_with = token
        if token == "bad-token":
            raise RuntimeError("bad token should not leak")
        await self._events["on_ready"]()


class SlowReadyBot(FakeBot):
    async def start(self, token):
        self.started_with = token
        if token == "slow-token":
            await asyncio.sleep(0.05)
        await self._events["on_ready"]()


@pytest.fixture(autouse=True)
def _fake_discord_runtime(monkeypatch):
    monkeypatch.setattr(discord_adapter, "DISCORD_AVAILABLE", True)
    monkeypatch.setattr(discord_adapter, "Intents", FakeIntents)
    monkeypatch.setattr(discord_adapter.commands, "Bot", lambda **kwargs: FakeBot(**kwargs))
    monkeypatch.setattr(discord_adapter, "_build_allowed_mentions", lambda: "safe-mentions")
    monkeypatch.setattr(
        "gateway.status.acquire_scoped_lock",
        lambda scope, identity, metadata=None: (True, None),
    )
    monkeypatch.setattr("gateway.status.release_scoped_lock", lambda scope, identity: None)


def _native_config(*, token_refs=None, mode="listen_only"):
    token_refs = token_refs or {
        "bohumil": "secret://discord/bohumil-token",
        "reviewer": "secret://discord/reviewer-token",
    }
    return DiscordNativeMultibotConfig(
        enabled=True,
        mode=mode,
        guild_allowlist=["guild-1"],
        default_intake_agent_id="bohumil",
        identities=[
            DiscordNativeMultibotIdentityConfig(
                agent_id=agent_id,
                hermes_profile=f"{agent_id}-profile",
                discord_application_id=f"app-{agent_id}",
                discord_bot_user_id=f"bot-{agent_id}",
                token_secret_ref=ref,
                capabilities=["chat"],
                enabled=True,
            )
            for agent_id, ref in token_refs.items()
        ],
    )


def _message(*, message_id="msg-1", author_id="human-1", bot=False):
    return SimpleNamespace(
        id=message_id,
        guild=SimpleNamespace(id="guild-1"),
        channel=SimpleNamespace(id="channel-1", name="general", parent_id=None),
        author=SimpleNamespace(id=author_id, bot=bot),
        mentions=[],
        webhook_id=None,
        type="default",
        content="hello native multibot",
    )


def _seed_startup_outbox(store: DiscordProtocolV2Store, *, content: str) -> dict:
    store.upsert_topic(
        topic_id="guild-1/100/root",
        guild_id="guild-1",
        channel_id="100",
        title="general",
        state={"mode": "active"},
    )
    outbox = store.create_outbox_delivery(
        idempotency_key=f"startup:{content}",
        target_agent_id="bohumil",
        topic_id="guild-1/100/root",
        channel_id="100",
        delivery_kind="response",
        payload={"content": content},
    )
    leased = store.lease_next_outbox(lease_owner="sender-before-crash", lease_seconds=-1)
    assert leased is not None
    assert leased["outbox_delivery_id"] == outbox["outbox_delivery_id"]
    return store.mark_outbox_sending(outbox["outbox_delivery_id"])


@pytest.mark.asyncio
async def test_native_multibot_listen_only_starts_two_identities_and_persists_observations(tmp_path):
    store = DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3")
    adapter = DiscordNativeMultibotAdapter(
        PlatformConfig(enabled=True, token=None),
        native_config=_native_config(),
        store=store,
        secret_resolver=StaticSecretResolver(
            {
                "secret://discord/bohumil-token": "good-token-a",
                "secret://discord/reviewer-token": "good-token-b",
            }
        ),
        ready_timeout_seconds=0.1,
    )
    handler_calls = []
    adapter.set_message_handler(lambda event: handler_calls.append(event))

    ok = await adapter.connect()

    assert ok is True
    assert set(adapter.runtime_states) == {"bohumil", "reviewer"}
    assert [state.status for state in adapter.runtime_states.values()] == [
        "connected",
        "connected",
    ]
    assert handler_calls == []

    bohumil_bot = adapter.runtime_states["bohumil"].runtime.client
    reviewer_bot = adapter.runtime_states["reviewer"].runtime.client
    assert bohumil_bot is not reviewer_bot
    assert bohumil_bot.started_with == "good-token-a"
    assert reviewer_bot.started_with == "good-token-b"

    await bohumil_bot._events["on_message"](_message(message_id="msg-1"))
    await reviewer_bot._events["on_message"](_message(message_id="msg-2"))

    assert store.count_rows("message_map") == 2
    assert store.count_rows("route_decisions") == 2
    assert store.count_rows("inbound_deliveries") == 2
    assert store.count_rows("outbox_deliveries") == 0
    row = store.get_message_map("msg-1")
    assert row["source_client_agent_id"] == "bohumil"
    assert row["author_kind"] == "human"
    decisions = store.route_decisions_for("discord_message", "msg-1")
    assert decisions[0]["decision"] == "delivered"
    assert decisions[0]["reason"] == "default_intake"

    snapshot = json.dumps(adapter.snapshot(), sort_keys=True)
    assert "good-token-a" not in snapshot
    assert "good-token-b" not in snapshot
    assert "secret://discord/bohumil-token" not in snapshot
    assert "secret://<redacted>" in snapshot

    await adapter.disconnect()


@pytest.mark.asyncio
async def test_native_multibot_active_startup_reconciles_outbox_with_history(tmp_path):
    store = DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3")
    outbox = _seed_startup_outbox(store, content="already reached discord")
    history_calls = []

    async def recent_history(outbox_row):
        history_calls.append(outbox_row["outbox_delivery_id"])
        return [
            {
                "id": "discord-existing-1",
                "content": "already reached discord",
                "channel_id": "100",
                "author_id": "bot-bohumil",
            }
        ]

    adapter = DiscordNativeMultibotAdapter(
        PlatformConfig(enabled=True, token=None),
        native_config=_native_config(
            token_refs={"bohumil": "secret://discord/bohumil-token"},
            mode="active",
        ),
        store=store,
        secret_resolver=StaticSecretResolver(
            {"secret://discord/bohumil-token": "good-token-a"}
        ),
        recent_history_fetcher=recent_history,
        ready_timeout_seconds=0.1,
    )

    ok = await adapter.connect()

    assert ok is True
    assert history_calls == [outbox["outbox_delivery_id"]]
    assert adapter.startup_reconciliation_result is not None
    assert adapter.startup_reconciliation_result.scanned == 1
    assert adapter.startup_reconciliation_result.acked == 1
    stored = store.get_outbox_delivery(outbox["outbox_delivery_id"])
    assert stored is not None
    assert stored["status"] == "acked"
    assert store.count_rows("outbox_parts") == 1
    mapped = store.get_message_map("discord-existing-1")
    assert mapped is not None
    assert mapped["outbox_delivery_id"] == outbox["outbox_delivery_id"]
    assert await adapter.run_outbox_once() is None

    await adapter.disconnect()


@pytest.mark.asyncio
async def test_native_multibot_active_bohumil_only_uses_no_secondary_identity(tmp_path):
    store = DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3")
    adapter = DiscordNativeMultibotAdapter(
        PlatformConfig(enabled=True, token=None),
        native_config=_native_config(
            token_refs={"bohumil": "secret://discord/bohumil-token"},
            mode="active",
        ),
        store=store,
        secret_resolver=StaticSecretResolver(
            {"secret://discord/bohumil-token": "good-token-a"}
        ),
        startup_reconciliation_enabled=False,
        ready_timeout_seconds=0.1,
    )

    ok = await adapter.connect()

    assert ok is True
    assert set(adapter.runtime_states) == {"bohumil"}
    assert adapter.runtime_states["bohumil"].status == "connected"
    assert adapter.resolve_outbox_client("bohumil") is not None
    assert adapter.resolve_outbox_client("reviewer") is None
    snapshot = adapter.snapshot()
    assert [identity["agent_id"] for identity in snapshot["identities"]] == ["bohumil"]
    assert "reviewer" not in json.dumps(snapshot, sort_keys=True)

    await adapter.disconnect()


@pytest.mark.asyncio
async def test_native_multibot_listen_only_skips_startup_reconciliation(tmp_path):
    store = DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3")
    outbox = _seed_startup_outbox(store, content="must not mutate in listen only")
    history_calls = []

    async def recent_history(outbox_row):  # pragma: no cover - guard
        history_calls.append(outbox_row)
        return []

    adapter = DiscordNativeMultibotAdapter(
        PlatformConfig(enabled=True, token=None),
        native_config=_native_config(
            token_refs={"bohumil": "secret://discord/bohumil-token"},
            mode="listen_only",
        ),
        store=store,
        secret_resolver=StaticSecretResolver(
            {"secret://discord/bohumil-token": "good-token-a"}
        ),
        startup_reconciliation_enabled=True,
        recent_history_fetcher=recent_history,
        ready_timeout_seconds=0.1,
    )

    ok = await adapter.connect()

    assert ok is True
    assert history_calls == []
    assert adapter.startup_reconciliation_result is None
    stored = store.get_outbox_delivery(outbox["outbox_delivery_id"])
    assert stored is not None
    assert stored["status"] == "sending"
    assert store.count_rows("outbox_parts") == 0
    assert store.count_rows("message_map", "outbox_delivery_id IS NOT NULL") == 0

    await adapter.disconnect()


@pytest.mark.asyncio
async def test_native_multibot_startup_failure_isolated_per_identity(tmp_path):
    store = DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3")
    adapter = DiscordNativeMultibotAdapter(
        PlatformConfig(enabled=True, token=None),
        native_config=_native_config(),
        store=store,
        secret_resolver=StaticSecretResolver(
            {
                "secret://discord/bohumil-token": "good-token-a",
                "secret://discord/reviewer-token": "bad-token",
            }
        ),
        ready_timeout_seconds=0.1,
    )

    ok = await adapter.connect()

    assert ok is True
    assert adapter.runtime_states["bohumil"].status == "connected"
    assert adapter.runtime_states["reviewer"].status == "failed"
    assert adapter.runtime_states["reviewer"].error == "RuntimeError"
    assert "bad-token" not in json.dumps(adapter.snapshot(), sort_keys=True)

    await adapter.disconnect()


@pytest.mark.asyncio
async def test_native_multibot_timeout_cleans_up_failed_identity(monkeypatch, tmp_path):
    created: list[SlowReadyBot] = []
    released: list[tuple[str, str]] = []
    monkeypatch.setattr(
        discord_adapter.commands,
        "Bot",
        lambda **kwargs: created.append(SlowReadyBot(**kwargs)) or created[-1],
    )
    monkeypatch.setattr(
        "gateway.status.release_scoped_lock",
        lambda scope, identity: released.append((scope, identity)),
    )
    store = DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3")
    adapter = DiscordNativeMultibotAdapter(
        PlatformConfig(enabled=True, token=None),
        native_config=_native_config(),
        store=store,
        secret_resolver=StaticSecretResolver(
            {
                "secret://discord/bohumil-token": "good-token-a",
                "secret://discord/reviewer-token": "slow-token",
            }
        ),
        ready_timeout_seconds=0.01,
    )

    ok = await adapter.connect()
    await asyncio.sleep(0.06)

    assert ok is True
    assert adapter.runtime_states["bohumil"].status == "connected"
    assert adapter.runtime_states["reviewer"].status == "failed"
    assert adapter.runtime_states["reviewer"].error == "TimeoutError"
    slow_bot = created[1]
    assert slow_bot.closed is True
    assert adapter.runtime_states["reviewer"].bot_task.done() is True
    assert len(adapter._lock_keys) == 1
    assert released == [
        (
            "discord-native-token-ref",
            adapter.runtime_states["reviewer"].token_ref_fingerprint,
        )
    ]

    await adapter.disconnect()


def test_discord_factory_default_off_returns_legacy_adapter(monkeypatch):
    import gateway.config as gateway_config

    monkeypatch.setattr(
        gateway_config,
        "load_gateway_config",
        lambda: SimpleNamespace(discord_native_multibot=DiscordNativeMultibotConfig()),
    )

    built = discord_adapter._build_adapter(PlatformConfig(enabled=True, token="legacy-token"))

    assert isinstance(built, DiscordAdapter)


def test_discord_factory_enabled_listen_only_returns_native_adapter(monkeypatch):
    import gateway.config as gateway_config

    monkeypatch.setattr(
        gateway_config,
        "load_gateway_config",
        lambda: SimpleNamespace(discord_native_multibot=_native_config()),
    )

    built = discord_adapter._build_adapter(PlatformConfig(enabled=True, token=None))

    assert isinstance(built, DiscordNativeMultibotAdapter)
    assert isinstance(built.secret_resolver, GatewaySecretResolver)


def test_discord_factory_enabled_v2_fails_closed_instead_of_legacy(monkeypatch):
    import gateway.config as gateway_config
    from plugins.platforms.discord import native_multibot

    monkeypatch.setattr(
        gateway_config,
        "load_gateway_config",
        lambda: SimpleNamespace(discord_native_multibot=_native_config()),
    )

    def boom(config, native_config):  # noqa: ARG001
        raise RuntimeError("native build failed")

    monkeypatch.setattr(native_multibot, "build_native_multibot_adapter", boom)

    with pytest.raises(RuntimeError, match="native build failed"):
        discord_adapter._build_adapter(PlatformConfig(enabled=True, token="legacy-token"))


def test_gateway_secret_resolver_resolves_secret_ref_without_legacy_discord_token(monkeypatch):
    ref = "secret://discord/bohumil-token"
    env_name = GatewaySecretResolver.hashed_env_name(ref)
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "legacy-token-must-not-be-used")
    monkeypatch.setenv(env_name, "native-v2-token")
    resolver = GatewaySecretResolver()

    assert resolver.resolve(ref).reveal() == "native-v2-token"
    monkeypatch.delenv(env_name)
    with pytest.raises(SecretResolutionError):
        resolver.resolve(ref)
