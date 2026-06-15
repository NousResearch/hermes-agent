from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gateway.config import DiscordNativeMultibotConfig, DiscordNativeMultibotIdentityConfig
from gateway.discord_identity_registry import DiscordIdentityRegistry
from gateway.discord_protocol_v2_ingest import (
    DiscordProtocolV2Ingestor,
    classify_author_kind,
    normalize_topic_id,
)
from gateway.discord_protocol_v2_store import DISCORD_SOURCE_TYPE, DiscordProtocolV2Store


def _native_config():
    return DiscordNativeMultibotConfig(
        enabled=True,
        mode="listen_only",
        guild_allowlist=["guild-1"],
        default_intake_agent_id="bohumil",
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
                agent_id="reviewer",
                hermes_profile="reviewer",
                discord_application_id="app-reviewer",
                discord_bot_user_id="bot-reviewer",
                token_secret_ref="secret://discord/reviewer-token",
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


def _message(
    *,
    message_id="msg-1",
    guild_id="guild-1",
    channel_id="channel-1",
    channel_name="general",
    parent_channel_id=None,
    author_id="human-1",
    author_bot=False,
    webhook_id=None,
    message_type="default",
    content="ahoj <@bot-bohumil>",
    mentions=None,
):
    return SimpleNamespace(
        id=message_id,
        guild=SimpleNamespace(id=guild_id) if guild_id is not None else None,
        channel=SimpleNamespace(
            id=channel_id,
            name=channel_name,
            parent_id=parent_channel_id,
        ),
        author=SimpleNamespace(id=author_id, bot=author_bot),
        mentions=[SimpleNamespace(id=item) for item in (mentions or ["bot-bohumil"])],
        webhook_id=webhook_id,
        type=message_type,
        content=content,
        attachments=[],
    )


def test_normalize_topic_id_contract():
    assert normalize_topic_id(guild_id="g", channel_id="c", thread_id=None) == "g/c/root"
    assert normalize_topic_id(guild_id="g", channel_id="c", thread_id="t") == "g/c/t"


def test_thread_after_restart_maps_to_same_topic_and_preserves_parent(tmp_path):
    db_path = tmp_path / "discord-v2.sqlite3"
    first_store = DiscordProtocolV2Store(db_path)
    first_ingestor = _ingestor(first_store)
    thread_message = _message(
        message_id="thread-msg-1",
        channel_id="thread-99",
        channel_name="decision-thread",
        parent_channel_id="parent-10",
        mentions=[],
    )

    first = first_ingestor.ingest_message(
        source_client_agent_id="bohumil",
        message=thread_message,
    )
    first_store.close()

    reopened = DiscordProtocolV2Store(db_path)
    try:
        second = _ingestor(reopened).ingest_message(
            source_client_agent_id="reviewer",
            message=_message(
                message_id="thread-msg-2",
                channel_id="thread-99",
                channel_name="decision-thread",
                parent_channel_id="parent-10",
                mentions=[],
            ),
        )

        assert first.normalized.topic_id == "guild-1/parent-10/thread-99"
        assert second.normalized.topic_id == first.normalized.topic_id
        topic = reopened.get_topic(first.normalized.topic_id)
        assert topic["channel_id"] == "parent-10"
        assert topic["thread_id"] == "thread-99"
        assert topic["parent_channel_id"] == "parent-10"
        message_row = reopened.get_message_map("thread-msg-2")
        assert message_row["channel_id"] == "parent-10"
        assert message_row["thread_id"] == "thread-99"
        assert message_row["parent_channel_id"] == "parent-10"
    finally:
        reopened.close()


def test_human_ingest_is_idempotent_and_routes_explicit_mention(tmp_path):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        ingestor = _ingestor(store)
        message = _message(message_id="human-msg-1", mentions=["bot-bohumil"])

        first = ingestor.ingest_message(source_client_agent_id="bohumil", message=message)
        second = ingestor.ingest_message(source_client_agent_id="bohumil", message=message)

        assert first.normalized.topic_id == "guild-1/channel-1/root"
        assert len(first.deliveries) == 1
        assert first.deliveries == second.deliveries
        assert first.deliveries[0]["target_agent_id"] == "bohumil"
        assert first.deliveries[0]["route_reason"] == "explicit_mention"
        assert store.count_rows("topics") == 1
        assert store.count_rows("message_map") == 1
        assert store.count_rows("inbound_deliveries") == 1
        assert store.count_rows("route_decisions") == 1
        message_row = store.get_message_map("human-msg-1")
        assert message_row["author_kind"] == "human"
        assert message_row["source_client_agent_id"] == "bohumil"
        assert json.loads(message_row["mentions_json"]) == ["bot-bohumil"]
        payload = json.loads(message_row["payload_json"])
        assert payload["content"] == "ahoj <@bot-bohumil>"
        decisions = store.route_decisions_for(DISCORD_SOURCE_TYPE, "human-msg-1")
        assert decisions[0]["decision"] == "delivered"
        assert decisions[0]["reason"] == "explicit_mention"
        assert json.loads(decisions[0]["target_agent_ids_json"]) == ["bohumil"]


@pytest.mark.parametrize(
    ("label", "kwargs", "expected_kind"),
    [
        ("registered", {"author_id": "bot-reviewer", "author_bot": True}, "registered_bot"),
        ("external", {"author_id": "external-bot", "author_bot": True}, "external_bot"),
        ("webhook", {"author_id": "webhook-user", "webhook_id": "hook-1"}, "webhook"),
        ("system", {"author_id": "system", "message_type": "thread_created"}, "system"),
    ],
)
def test_non_human_ingest_records_diagnostics_but_zero_deliveries(
    tmp_path, label, kwargs, expected_kind
):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        ingestor = _ingestor(store)
        message = _message(message_id=f"msg-{label}", **kwargs)

        result = ingestor.ingest_message(source_client_agent_id="bohumil", message=message)

        assert result.normalized.author_kind == expected_kind
        assert result.deliveries == []
        assert store.count_rows("inbound_deliveries") == 0
        assert store.count_rows("message_map") == 1
        row = store.get_message_map(f"msg-{label}")
        assert row["author_kind"] == expected_kind
        if expected_kind in {"registered_bot", "external_bot"}:
            assert row["author_bot_user_id"] == kwargs["author_id"]
        decisions = store.route_decisions_for(DISCORD_SOURCE_TYPE, f"msg-{label}")
        assert len(decisions) == 1
        assert decisions[0]["decision"] == "zero_delivery"
        assert decisions[0]["reason"] == f"non_human_author:{expected_kind}"


def test_disallowed_guild_is_ignored_without_rows(tmp_path):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        ingestor = _ingestor(store)
        result = ingestor.ingest_message(
            source_client_agent_id="bohumil",
            message=_message(message_id="bad-guild", guild_id="guild-2"),
        )

        assert result.ignored is True
        assert result.reason == "guild_not_allowed"
        assert store.count_rows("topics") == 0
        assert store.count_rows("message_map") == 0
        assert store.count_rows("inbound_deliveries") == 0
        assert store.count_rows("route_decisions") == 0


def test_classify_author_kind_treats_registered_bot_as_registered_even_if_bot_flag_false(tmp_path):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        registry = _registry(store)
        message = _message(author_id="bot-bohumil", author_bot=False)

        assert (
            classify_author_kind(
                message=message,
                author_id="bot-bohumil",
                identity_registry=registry,
            )
            == "registered_bot"
        )
