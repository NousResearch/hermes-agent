from __future__ import annotations

import json

import pytest

from gateway.config import (
    DiscordNativeMultibotConfig,
    DiscordNativeMultibotIdentityConfig,
    PlatformConfig,
)
from gateway.discord_identity_registry import DiscordIdentityRegistry
from gateway.discord_protocol_v2_store import DISCORD_SOURCE_TYPE, DiscordProtocolV2Store
from gateway.secret_refs import SensitiveToken
from plugins.platforms.discord.native_multibot import DiscordNativeMultibotAdapter
from tests.gateway.discord_fakes import FakeChannel, FakeGuild, FakeMessage, FakeUser


class ExplodingResolver:
    calls = 0

    def resolve(self, ref: str) -> SensitiveToken:  # pragma: no cover - safety guard
        self.calls += 1
        raise AssertionError(f"listen_only harness must not resolve token refs: {ref}")


def _native_config() -> DiscordNativeMultibotConfig:
    return DiscordNativeMultibotConfig(
        enabled=True,
        mode="listen_only",
        guild_allowlist=["guild-listen"],
        default_intake_agent_id="bohumil",
        identities=[
            DiscordNativeMultibotIdentityConfig(
                agent_id="bohumil",
                hermes_profile="bohumil-profile",
                discord_application_id="app-bohumil",
                discord_bot_user_id="bot-bohumil",
                token_secret_ref="secret://discord/bohumil-token",
                capabilities=["intake", "chat"],
                enabled=True,
            ),
            DiscordNativeMultibotIdentityConfig(
                agent_id="reviewer",
                hermes_profile="reviewer-profile",
                discord_application_id="app-reviewer",
                discord_bot_user_id="bot-reviewer",
                token_secret_ref="secret://discord/reviewer-token",
                capabilities=["chat"],
                enabled=True,
            ),
        ],
    )


def _adapter(
    store: DiscordProtocolV2Store,
    resolver: ExplodingResolver,
) -> DiscordNativeMultibotAdapter:
    cfg = _native_config()
    registry = DiscordIdentityRegistry.load(cfg, store, secret_resolver=resolver)
    return DiscordNativeMultibotAdapter(
        PlatformConfig(enabled=True, token=None),
        native_config=cfg,
        store=store,
        identity_registry=registry,
        secret_resolver=resolver,
    )


def _guild_channel() -> tuple[FakeGuild, FakeChannel]:
    guild = FakeGuild(id="guild-listen", name="staging")
    channel = FakeChannel(id="channel-listen", name="listen-only", guild=guild)
    guild.channels.append(channel)
    return guild, channel


def _decision(store: DiscordProtocolV2Store, source_id: str) -> dict:
    decisions = store.route_decisions_for(DISCORD_SOURCE_TYPE, source_id)
    assert len(decisions) == 1
    return decisions[0]


@pytest.mark.asyncio
async def test_listen_only_fake_runtime_ingests_without_worker_outbox_send_or_tokens(tmp_path):
    resolver = ExplodingResolver()
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        adapter = _adapter(store, resolver)
        guild, channel = _guild_channel()
        human = FakeUser(id="human-listen", name="human", bot=False)
        reviewer_bot = FakeUser(id="bot-reviewer", name="reviewer", bot=True)

        await adapter._handle_listen_only_message(
            "bohumil",
            "bot-bohumil",
            FakeMessage(
                "please review this",
                id="listen-human-explicit",
                author=human,
                channel=channel,
                guild=guild,
                mentions=[reviewer_bot],
            ),
        )

        assert resolver.calls == 0
        assert channel.sent_messages == []
        assert store.count_rows("topics") == 1
        assert store.count_rows("message_map") == 1
        assert store.count_rows("route_decisions") == 1
        assert store.count_rows("inbound_deliveries") == 1
        assert store.count_rows("outbox_deliveries") == 0
        assert store.count_rows("outbox_parts") == 0

        topic = store.get_topic("guild-listen/channel-listen/root")
        assert topic is not None
        assert topic["guild_id"] == "guild-listen"
        assert topic["channel_id"] == "channel-listen"
        assert json.loads(topic["state_json"])["mode"] == "listen_only"

        mapped = store.get_message_map("listen-human-explicit")
        assert mapped is not None
        assert mapped["direction"] == "inbound"
        assert mapped["author_kind"] == "human"
        assert mapped["source_client_agent_id"] == "bohumil"
        assert json.loads(mapped["mentions_json"]) == ["bot-reviewer"]

        decision = _decision(store, "listen-human-explicit")
        assert decision["decision"] == "delivered"
        assert decision["reason"] == "explicit_mention"
        assert json.loads(decision["target_agent_ids_json"]) == ["reviewer"]


@pytest.mark.asyncio
async def test_listen_only_bot_loop_diagnostics_do_not_create_worker_or_outbox_loop(tmp_path):
    resolver = ExplodingResolver()
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        adapter = _adapter(store, resolver)
        guild, channel = _guild_channel()
        registered_bot = FakeUser(id="bot-bohumil", name="bohumil", bot=True)
        reviewer_bot = FakeUser(id="bot-reviewer", name="reviewer", bot=True)
        await adapter._handle_listen_only_message(
            "bohumil",
            "bot-bohumil",
            FakeMessage(
                "projection echo mentioning reviewer",
                id="listen-bot-loop",
                author=registered_bot,
                channel=channel,
                guild=guild,
                mentions=[reviewer_bot],
            ),
        )

        assert resolver.calls == 0
        assert channel.sent_messages == []
        assert await adapter.run_outbox_once() is None
        assert store.count_rows("topics") == 1
        assert store.count_rows("message_map") == 1
        assert store.count_rows("route_decisions") == 1
        assert store.count_rows("inbound_deliveries") == 0
        assert store.count_rows("outbox_deliveries") == 0
        assert store.count_rows("outbox_parts") == 0

        mapped = store.get_message_map("listen-bot-loop")
        assert mapped is not None
        assert mapped["author_kind"] == "registered_bot"
        assert mapped["author_bot_user_id"] == "bot-bohumil"
        assert mapped["source_client_agent_id"] == "bohumil"

        decision = _decision(store, "listen-bot-loop")
        assert decision["decision"] == "zero_delivery"
        assert decision["reason"] == "non_human_author:registered_bot"
        assert json.loads(decision["target_agent_ids_json"]) == []
        payload = json.loads(decision["payload_json"])
        assert payload["suppressed_discord_origin"] is True
