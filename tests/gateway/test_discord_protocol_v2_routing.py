from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gateway.config import DiscordNativeMultibotConfig, DiscordNativeMultibotIdentityConfig
from gateway.discord_identity_registry import DiscordIdentityRegistry
from gateway.discord_protocol_v2_ingest import DiscordProtocolV2Ingestor
from gateway.discord_protocol_v2_routing import DiscordProtocolV2Router
from gateway.discord_protocol_v2_store import DISCORD_SOURCE_TYPE, DiscordProtocolV2Store


def _native_config(*, default_intake_agent_id="bohumil"):
    return DiscordNativeMultibotConfig(
        enabled=True,
        mode="listen_only",
        guild_allowlist=["guild-1"],
        default_intake_agent_id=default_intake_agent_id,
        identities=[
            DiscordNativeMultibotIdentityConfig(
                agent_id="bohumil",
                hermes_profile="bohumil",
                discord_application_id="app-bohumil",
                discord_bot_user_id="bot-bohumil",
                token_secret_ref="secret://discord/bohumil-token",
                capabilities=["intake", "chat"],
                enabled=True,
            ),
            DiscordNativeMultibotIdentityConfig(
                agent_id="agent-a",
                hermes_profile="agent-a",
                discord_application_id="app-agent-a",
                discord_bot_user_id="bot-agent-a",
                token_secret_ref="secret://discord/agent-a-token",
                capabilities=["chat"],
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


def _registry(store, *, default_intake_agent_id="bohumil"):
    return DiscordIdentityRegistry.load(
        _native_config(default_intake_agent_id=default_intake_agent_id),
        store,
        secret_resolver=None,
    )


def _ingestor(store, *, default_intake_agent_id="bohumil"):
    cfg = _native_config(default_intake_agent_id=default_intake_agent_id)
    return DiscordProtocolV2Ingestor(
        store=store,
        identity_registry=_registry(store, default_intake_agent_id=default_intake_agent_id),
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
    reference_message_id=None,
):
    return SimpleNamespace(
        id=message_id,
        guild=SimpleNamespace(id="guild-1"),
        channel=SimpleNamespace(id="channel-1", name="general", parent_id=None),
        author=SimpleNamespace(id=author_id, bot=author_bot),
        mentions=list(mentions or []),
        webhook_id=webhook_id,
        type=message_type,
        content="hello routing",
        reference=SimpleNamespace(message_id=reference_message_id)
        if reference_message_id
        else None,
        attachments=[],
    )


def _decision(store, message_id):
    decisions = store.route_decisions_for(DISCORD_SOURCE_TYPE, message_id)
    assert len(decisions) == 1
    return decisions[0]


def _decision_targets(decision):
    return json.loads(decision["target_agent_ids_json"])


def test_human_multi_mention_routes_only_to_mentioned_agents(tmp_path):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        result = _ingestor(store).ingest_message(
            source_client_agent_id="bohumil",
            message=_message(
                message_id="msg-multi",
                mentions=[_mention("bot-agent-a"), _mention("bot-agent-b")],
            ),
        )

        assert [row["target_agent_id"] for row in result.deliveries] == [
            "agent-a",
            "agent-b",
        ]
        assert store.count_rows("inbound_deliveries") == 2
        assert store.list_inbound_deliveries(target_agent_id="bohumil") == []
        decision = _decision(store, "msg-multi")
        assert decision["decision"] == "delivered"
        assert decision["reason"] == "explicit_mention"
        assert _decision_targets(decision) == ["agent-a", "agent-b"]


def test_human_unknown_bot_mention_policy_fails_without_delivery(tmp_path):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        result = _ingestor(store).ingest_message(
            source_client_agent_id="bohumil",
            message=_message(
                message_id="msg-unknown-bot",
                mentions=[_mention("bot-unknown", bot=True)],
            ),
        )

        assert result.deliveries == []
        assert store.count_rows("inbound_deliveries") == 0
        decision = _decision(store, "msg-unknown-bot")
        assert decision["decision"] == "policy_failed"
        assert decision["reason"] == "unknown_bot_mention"
        assert _decision_targets(decision) == []
        payload = json.loads(decision["payload_json"])
        assert payload["unknown_bot_mentions"] == ["bot-unknown"]


def test_human_reply_to_agent_message_routes_to_replied_agent_without_mention(tmp_path):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        _registry(store)
        store.upsert_message_map(
            discord_message_id="agent-b-projection",
            guild_id="guild-1",
            channel_id="channel-1",
            direction="projection",
            agent_id="agent-b",
            author_id="bot-agent-b",
            author_kind="registered_bot",
            author_bot_user_id="bot-agent-b",
            source_client_agent_id="agent-b",
            mentions=[],
            payload={"projection": True},
        )

        result = _ingestor(store).ingest_message(
            source_client_agent_id="bohumil",
            message=_message(
                message_id="msg-reply",
                mentions=[],
                reference_message_id="agent-b-projection",
            ),
        )

        assert [row["target_agent_id"] for row in result.deliveries] == ["agent-b"]
        decision = _decision(store, "msg-reply")
        assert decision["decision"] == "delivered"
        assert decision["reason"] == "reply_to_agent"


def test_human_reply_to_human_inbound_message_uses_default_not_observer(tmp_path):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        ingestor = _ingestor(store)
        ingestor.ingest_message(
            source_client_agent_id="bohumil",
            message=_message(
                message_id="human-original",
                mentions=[_mention("bot-agent-a")],
            ),
        )

        result = ingestor.ingest_message(
            source_client_agent_id="bohumil",
            message=_message(
                message_id="human-reply-to-human",
                mentions=[],
                reference_message_id="human-original",
            ),
        )

        assert [row["target_agent_id"] for row in result.deliveries] == ["bohumil"]
        decision = _decision(store, "human-reply-to-human")
        assert decision["decision"] == "delivered"
        assert decision["reason"] == "default_intake"


def test_human_unmentioned_message_routes_to_default_intake_only(tmp_path):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        result = _ingestor(store).ingest_message(
            source_client_agent_id="agent-a",
            message=_message(message_id="msg-default", mentions=[]),
        )

        assert [row["target_agent_id"] for row in result.deliveries] == ["bohumil"]
        assert store.count_rows("inbound_deliveries") == 1
        decision = _decision(store, "msg-default")
        assert decision["decision"] == "delivered"
        assert decision["reason"] == "default_intake"
        assert _decision_targets(decision) == ["bohumil"]


def test_default_intake_policy_can_be_disallowed(tmp_path):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        registry = _registry(store)
        ingestor = _ingestor(store)
        ingestor.router = DiscordProtocolV2Router(
            store=store,
            identity_registry=registry,
            default_intake_agent_id="bohumil",
            allow_default_intake=False,
        )

        result = ingestor.ingest_message(
            source_client_agent_id="agent-a",
            message=_message(message_id="msg-no-default", mentions=[]),
        )

        assert result.deliveries == []
        assert store.count_rows("inbound_deliveries") == 0
        decision = _decision(store, "msg-no-default")
        assert decision["decision"] == "policy_failed"
        assert decision["reason"] == "default_intake_disallowed"


@pytest.mark.parametrize(
    ("message_id", "message_kwargs", "expected_kind"),
    [
        (
            "msg-registered",
            {"author_id": "bot-agent-a", "author_bot": True},
            "registered_bot",
        ),
        ("msg-external", {"author_id": "external-bot", "author_bot": True}, "external_bot"),
        ("msg-webhook", {"author_id": "webhook-user", "webhook_id": "hook-1"}, "webhook"),
        ("msg-system", {"author_id": "system", "message_type": "thread_created"}, "system"),
    ],
)
def test_non_human_mentions_registered_agent_create_only_diagnostic(
    tmp_path, message_id, message_kwargs, expected_kind
):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        result = _ingestor(store).ingest_message(
            source_client_agent_id="bohumil",
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
        decision = _decision(store, message_id)
        assert decision["decision"] == "zero_delivery"
        assert decision["reason"] == f"non_human_author:{expected_kind}"
        assert _decision_targets(decision) == []
