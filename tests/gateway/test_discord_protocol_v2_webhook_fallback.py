from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gateway.config import (
    DiscordNativeMultibotConfig,
    DiscordNativeMultibotIdentityConfig,
    GatewayConfig,
    Platform,
)
from gateway.discord_identity_registry import DiscordIdentityRegistry
from gateway.discord_protocol_v2_ingest import DiscordProtocolV2Ingestor
from gateway.discord_protocol_v2_store import DISCORD_SOURCE_TYPE, DiscordProtocolV2Store


def _identity(agent_id: str) -> DiscordNativeMultibotIdentityConfig:
    return DiscordNativeMultibotIdentityConfig(
        agent_id=agent_id,
        hermes_profile=agent_id,
        discord_application_id=f"app-{agent_id}",
        discord_bot_user_id=f"bot-{agent_id}",
        token_secret_ref=f"secret://discord/{agent_id}-token",
        capabilities=["chat"],
        enabled=True,
    )


def _native_config() -> DiscordNativeMultibotConfig:
    return DiscordNativeMultibotConfig(
        enabled=True,
        mode="active",
        guild_allowlist=["guild-1"],
        default_intake_agent_id="agent-a",
        identities=[_identity("agent-a"), _identity("agent-b")],
    )


def _registry(store: DiscordProtocolV2Store) -> DiscordIdentityRegistry:
    return DiscordIdentityRegistry.load(_native_config(), store, secret_resolver=None)


def _ingestor(store: DiscordProtocolV2Store) -> DiscordProtocolV2Ingestor:
    cfg = _native_config()
    return DiscordProtocolV2Ingestor(
        store=store,
        identity_registry=_registry(store),
        default_intake_agent_id=cfg.default_intake_agent_id,
        guild_allowlist=cfg.guild_allowlist,
    )


def _mention(user_id: str, *, bot: bool = True) -> SimpleNamespace:
    return SimpleNamespace(id=user_id, bot=bot)


def _message(
    *,
    message_id: str,
    author_id: str = "human-1",
    author_bot: bool = False,
    webhook_id: str | None = None,
    mentions: list[SimpleNamespace] | None = None,
    content: str = "diagnostic fallback projection",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=message_id,
        guild=SimpleNamespace(id="guild-1"),
        channel=SimpleNamespace(id="channel-1", name="ops", parent_id=None),
        author=SimpleNamespace(id=author_id, bot=author_bot),
        mentions=list(mentions or []),
        webhook_id=webhook_id,
        type="default",
        content=content,
        attachments=[],
    )


def _only_decision(store: DiscordProtocolV2Store, message_id: str) -> dict[str, object]:
    decisions = store.route_decisions_for(DISCORD_SOURCE_TYPE, message_id)
    assert len(decisions) == 1
    return decisions[0]


def _targets(decision: dict[str, object]) -> list[str]:
    return json.loads(str(decision["target_agent_ids_json"]))


def test_active_v2_without_webhook_config_does_not_enable_webhook_platform() -> None:
    config = GatewayConfig.from_dict({"discord_native_multibot": _native_config().to_dict()})

    assert config.discord_native_multibot.enabled is True
    assert config.discord_native_multibot.mode == "active"
    assert Platform.WEBHOOK not in config.platforms
    assert Platform.WEBHOOK not in config.get_connected_platforms()


@pytest.mark.parametrize(
    "fallback_block",
    [
        {"webhook_fallback": True},
        {"diagnostic_webhook_fallback": {"enabled": True}},
        {"enable_webhook_fallback": "yes"},
        {"auto_enable_webhook_fallback": 1},
    ],
)
def test_v2_config_cannot_auto_enable_webhook_fallback(fallback_block: dict[str, object]) -> None:
    data = _native_config().to_dict()
    data.update(fallback_block)

    with pytest.raises(ValueError, match="webhook fallback"):
        DiscordNativeMultibotConfig.from_dict(data)


def test_webhook_authored_mention_is_diagnostic_only_zero_inbound_deliveries(tmp_path) -> None:
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        result = _ingestor(store).ingest_message(
            source_client_agent_id="agent-a",
            message=_message(
                message_id="webhook-mention-agent-b",
                author_id="webhook-author",
                webhook_id="discord-webhook-1",
                mentions=[_mention("bot-agent-b")],
                content="<@bot-agent-b> projection only",
            ),
        )

        assert result.normalized is not None
        assert result.normalized.author_kind == "webhook"
        assert result.deliveries == []
        assert store.count_rows("inbound_deliveries") == 0

        row = store.get_message_map("webhook-mention-agent-b")
        assert row is not None
        assert row["author_kind"] == "webhook"
        assert row["direction"] == "inbound"
        assert json.loads(row["mentions_json"]) == ["bot-agent-b"]

        decision = _only_decision(store, "webhook-mention-agent-b")
        assert decision["decision"] == "zero_delivery"
        assert decision["reason"] == "non_human_author:webhook"
        assert _targets(decision) == []
        payload = json.loads(str(decision["payload_json"]))
        assert payload["suppressed_discord_origin"] is True
        assert payload["route_reason"] == "non_human_author:webhook"


def test_diagnostic_projection_replay_cannot_be_authoritative_agent_trigger(tmp_path) -> None:
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        _registry(store)
        store.upsert_topic(
            topic_id="guild-1/channel-1/root",
            guild_id="guild-1",
            channel_id="channel-1",
            title="ops",
        )
        event, _delivery = store.create_internal_handoff(
            event_type="handoff.requested",
            source_agent_id="agent-a",
            target_agent_id="agent-b",
            topic_id="guild-1/channel-1/root",
            payload={"reason": "operator diagnostic projection"},
            agent_event_id="handoff-agent-a-agent-b",
        )
        store.record_projection_message(
            agent_event_id=event["agent_event_id"],
            discord_message_id="webhook-projection-replay",
            guild_id="guild-1",
            channel_id="channel-1",
            topic_id="guild-1/channel-1/root",
            target_agent_id="agent-b",
            author_id="webhook-author",
            author_kind="webhook",
            source_client_agent_id="agent-a",
            mentions=["bot-agent-b"],
            payload={"projection": "diagnostic_webhook_fallback"},
        )

        result = _ingestor(store).ingest_message(
            source_client_agent_id="agent-a",
            message=_message(
                message_id="webhook-projection-replay",
                author_id="webhook-author",
                webhook_id="discord-webhook-1",
                mentions=[_mention("bot-agent-b")],
            ),
        )

        assert result.deliveries == []
        assert store.count_rows(
            "inbound_deliveries",
            "source_type = ?",
            (DISCORD_SOURCE_TYPE,),
        ) == 0
        projection = store.get_message_map("webhook-projection-replay")
        assert projection is not None
        assert projection["direction"] == "projection"
        assert projection["author_kind"] == "webhook"
        assert projection["agent_event_id"] == event["agent_event_id"]
        decision = _only_decision(store, "webhook-projection-replay")
        assert decision["decision"] == "zero_delivery"
        assert decision["reason"] == "non_human_author:webhook"
        assert _targets(decision) == []
