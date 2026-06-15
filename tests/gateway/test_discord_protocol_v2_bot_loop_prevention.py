from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gateway.config import DiscordNativeMultibotConfig, DiscordNativeMultibotIdentityConfig
from gateway.discord_identity_registry import DiscordIdentityRegistry
from gateway.discord_protocol_v2_ingest import DiscordProtocolV2Ingestor
from gateway.discord_protocol_v2_store import (
    DISCORD_SOURCE_TYPE,
    INTERNAL_SOURCE_TYPE,
    DiscordProtocolV2Store,
)


def _native_config():
    return DiscordNativeMultibotConfig(
        enabled=True,
        mode="listen_only",
        guild_allowlist=["guild-1"],
        default_intake_agent_id="agent-a",
        identities=[
            DiscordNativeMultibotIdentityConfig(
                agent_id="agent-a",
                hermes_profile="agent-a",
                discord_application_id="app-agent-a",
                discord_bot_user_id="bot-agent-a",
                token_secret_ref="secret://discord/agent-a-token",
                capabilities=["intake", "chat"],
                enabled=True,
            ),
            DiscordNativeMultibotIdentityConfig(
                agent_id="agent-b",
                hermes_profile="agent-b",
                discord_application_id="app-agent-b",
                discord_bot_user_id="bot-agent-b",
                token_secret_ref="secret://discord/agent-b-token",
                capabilities=["chat"],
                enabled=True,
            ),
        ],
    )


def _registry(store):
    return DiscordIdentityRegistry.load(_native_config(), store, secret_resolver=None)


def _ingestor(store):
    cfg = _native_config()
    return DiscordProtocolV2Ingestor(
        store=store,
        identity_registry=_registry(store),
        default_intake_agent_id=cfg.default_intake_agent_id,
        guild_allowlist=cfg.guild_allowlist,
    )


def _mention(user_id, *, bot=True):
    return SimpleNamespace(id=user_id, bot=bot)


def _message(
    *,
    message_id="msg-1",
    author_id="human-1",
    author_bot=False,
    mentions=None,
    webhook_id=None,
    message_type="default",
):
    return SimpleNamespace(
        id=message_id,
        guild=SimpleNamespace(id="guild-1"),
        channel=SimpleNamespace(id="channel-1", name="general", parent_id=None),
        author=SimpleNamespace(id=author_id, bot=author_bot),
        mentions=list(mentions or []),
        webhook_id=webhook_id,
        type=message_type,
        content="projected mention",
        attachments=[],
    )


def _decision(store, source_type, source_id, *, decision=None):
    decisions = store.route_decisions_for(source_type, source_id)
    if decision is not None:
        decisions = [row for row in decisions if row["decision"] == decision]
    assert len(decisions) == 1
    return decisions[0]


def _targets(decision):
    return json.loads(decision["target_agent_ids_json"])


def test_registered_agent_projection_reingest_does_not_create_discord_delivery(tmp_path):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        _registry(store)
        store.upsert_message_map(
            discord_message_id="projection-agent-a-to-b",
            guild_id="guild-1",
            channel_id="channel-1",
            direction="projection",
            agent_id="agent-a",
            author_id="bot-agent-a",
            author_kind="registered_bot",
            author_bot_user_id="bot-agent-a",
            source_client_agent_id="agent-a",
            mentions=["bot-agent-b"],
            payload={"projection": True},
        )

        result = _ingestor(store).ingest_message(
            source_client_agent_id="agent-a",
            message=_message(
                message_id="projection-agent-a-to-b",
                author_id="bot-agent-a",
                author_bot=True,
                mentions=[_mention("bot-agent-b")],
            ),
        )

        assert result.deliveries == []
        assert store.count_rows("inbound_deliveries") == 0
        row = store.get_message_map("projection-agent-a-to-b")
        assert row["author_kind"] == "registered_bot"
        assert row["direction"] == "projection"
        assert row["source_client_agent_id"] == "agent-a"
        assert row["author_bot_user_id"] == "bot-agent-a"
        assert json.loads(row["mentions_json"]) == ["bot-agent-b"]
        decision = _decision(store, DISCORD_SOURCE_TYPE, "projection-agent-a-to-b")
        assert decision["decision"] == "zero_delivery"
        assert decision["reason"] == "non_human_author:registered_bot"
        assert _targets(decision) == []


@pytest.mark.parametrize(
    ("message_id", "message_kwargs", "expected_kind"),
    [
        ("external-bot-loop", {"author_id": "external-bot", "author_bot": True}, "external_bot"),
        ("webhook-loop", {"author_id": "webhook-user", "webhook_id": "hook-1"}, "webhook"),
        ("system-loop", {"author_id": "system", "message_type": "thread_created"}, "system"),
    ],
)
def test_external_webhook_and_system_mentions_do_not_create_discord_deliveries(
    tmp_path, message_id, message_kwargs, expected_kind
):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        result = _ingestor(store).ingest_message(
            source_client_agent_id="agent-a",
            message=_message(
                message_id=message_id,
                mentions=[_mention("bot-agent-b")],
                **message_kwargs,
            ),
        )

        assert result.deliveries == []
        assert store.count_rows("inbound_deliveries") == 0
        row = store.get_message_map(message_id)
        assert row["author_kind"] == expected_kind
        decision = _decision(store, DISCORD_SOURCE_TYPE, message_id)
        assert decision["decision"] == "zero_delivery"
        assert decision["reason"] == f"non_human_author:{expected_kind}"
        assert _targets(decision) == []


def test_internal_handoff_creates_internal_delivery_and_projection_replay_does_not_duplicate(tmp_path):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        _registry(store)
        store.upsert_topic(
            topic_id="guild-1/channel-1/root",
            guild_id="guild-1",
            channel_id="channel-1",
            title="general",
        )
        event, internal_delivery = store.create_internal_handoff(
            event_type="handoff.requested",
            source_agent_id="agent-a",
            target_agent_id="agent-b",
            topic_id="guild-1/channel-1/root",
            payload={"reason": "please review"},
            agent_event_id="handoff-agent-a-agent-b",
        )
        assert internal_delivery["source_type"] == INTERNAL_SOURCE_TYPE
        assert internal_delivery["agent_event_id"] == event["agent_event_id"]
        assert internal_delivery["discord_message_id"] is None
        assert internal_delivery["target_agent_id"] == "agent-b"
        store.record_projection_message(
            agent_event_id=event["agent_event_id"],
            discord_message_id="handoff-projection-message",
            guild_id="guild-1",
            channel_id="channel-1",
            topic_id="guild-1/channel-1/root",
            target_agent_id="agent-b",
            author_id="bot-agent-a",
            author_kind="registered_bot",
            source_client_agent_id="agent-a",
            mentions=["bot-agent-b"],
            payload={"projection": "handoff.requested"},
        )
        projection_before = store.get_message_map("handoff-projection-message")
        assert projection_before["direction"] == "projection"
        assert projection_before["agent_event_id"] == event["agent_event_id"]

        result = _ingestor(store).ingest_message(
            source_client_agent_id="agent-a",
            message=_message(
                message_id="handoff-projection-message",
                author_id="bot-agent-a",
                author_bot=True,
                mentions=[_mention("bot-agent-b")],
            ),
        )

        assert result.deliveries == []
        projection_after = store.get_message_map("handoff-projection-message")
        assert projection_after["direction"] == "projection"
        assert projection_after["agent_event_id"] == event["agent_event_id"]
        assert projection_after["author_bot_user_id"] == "bot-agent-a"
        all_deliveries = store.list_inbound_deliveries()
        assert len(all_deliveries) == 1
        assert all_deliveries[0]["source_type"] == INTERNAL_SOURCE_TYPE
        assert all_deliveries[0]["target_agent_id"] == "agent-b"
        discord_decision = _decision(store, DISCORD_SOURCE_TYPE, "handoff-projection-message")
        assert discord_decision["decision"] == "zero_delivery"
        assert discord_decision["reason"] == "non_human_author:registered_bot"
        internal_decision = _decision(
            store, INTERNAL_SOURCE_TYPE, event["agent_event_id"], decision="delivered"
        )
        assert internal_decision["decision"] == "delivered"
        assert _targets(internal_decision) == ["agent-b"]
