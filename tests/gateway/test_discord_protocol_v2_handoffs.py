from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gateway.config import DiscordNativeMultibotConfig, DiscordNativeMultibotIdentityConfig
from gateway.discord_identity_registry import DiscordIdentityRegistry
from gateway.discord_protocol_v2_handoffs import (
    accept_handoff,
    request_consult,
    request_handoff,
    request_review,
)
from gateway.discord_protocol_v2_ingest import DiscordProtocolV2Ingestor
from gateway.discord_protocol_v2_outbox import (
    DiscordProtocolV2ClientBinding,
    DiscordProtocolV2OutboxSender,
)
from gateway.discord_protocol_v2_store import (
    DISCORD_SOURCE_TYPE,
    INTERNAL_SOURCE_TYPE,
    DiscordProtocolV2Store,
    inbound_delivery_key,
    projection_idempotency_key,
)


def _native_config() -> DiscordNativeMultibotConfig:
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
                capabilities=["chat"],
                enabled=True,
            ),
            DiscordNativeMultibotIdentityConfig(
                agent_id="agent-b",
                hermes_profile="agent-b",
                discord_application_id="app-agent-b",
                discord_bot_user_id="bot-agent-b",
                token_secret_ref="secret://discord/agent-b-token",
                capabilities=["chat", "review"],
                enabled=True,
            ),
        ],
    )


def _registry(store: DiscordProtocolV2Store) -> DiscordIdentityRegistry:
    return DiscordIdentityRegistry.load(_native_config(), store, secret_resolver=None)


def _seed(store: DiscordProtocolV2Store) -> None:
    _registry(store)
    store.upsert_topic(
        topic_id="guild-1/channel-1/root",
        guild_id="guild-1",
        channel_id="channel-1",
        title="general",
        state={"mode": "listen_only"},
    )


def _mention(user_id: str, *, bot: bool = True) -> SimpleNamespace:
    return SimpleNamespace(id=user_id, bot=bot)


def _message(*, message_id: str, author_id: str, mentions=None, author_bot: bool = True):
    return SimpleNamespace(
        id=message_id,
        guild=SimpleNamespace(id="guild-1"),
        channel=SimpleNamespace(id="channel-1", name="general", parent_id=None),
        author=SimpleNamespace(id=author_id, bot=author_bot),
        mentions=list(mentions or []),
        webhook_id=None,
        type="default",
        content="projection mentions a target bot",
        reference=None,
        attachments=[],
    )


class _FakeChannel:
    def __init__(self, client_name: str) -> None:
        self.client_name = client_name
        self.sent: list[str] = []

    async def send(self, *, content: str, reference=None):
        assert reference is None
        self.sent.append(content)
        return SimpleNamespace(id=f"{self.client_name}-message-{len(self.sent)}")


class _FakeDiscordClient:
    def __init__(self, name: str, channel_id: str = "channel-1") -> None:
        self.name = name
        self.channel = _FakeChannel(name)
        self.channel_id = channel_id

    def get_channel(self, channel_id: int):
        if str(channel_id) == self.channel_id:
            return self.channel
        return None

    async def fetch_channel(self, channel_id: int):
        return self.get_channel(channel_id)


def _resolver(clients: dict[str, _FakeDiscordClient]):
    def resolve(agent_id: str):
        client = clients.get(agent_id)
        if client is None:
            return None
        return DiscordProtocolV2ClientBinding(
            client=client,
            source_client_agent_id=agent_id,
            author_bot_user_id=f"bot-{agent_id}",
        )

    return resolve


def test_internal_handoff_creates_target_delivery_and_projection_outbox(tmp_path):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        _seed(store)

        result = request_handoff(
            store,
            agent_event_id="evt_handoff_requested_0001",
            source_agent_id="agent-a",
            target_agent_id="agent-b",
            topic_id="guild-1/channel-1/root",
            payload={
                "task": "review this patch",
                "api_token": "sk-secret-value-that-must-not-persist",
            },
        )

        assert result.event["agent_event_id"] == "evt_handoff_requested_0001"
        assert result.event["event_type"] == "handoff.requested"
        assert result.delivery is not None
        assert result.delivery["source_type"] == INTERNAL_SOURCE_TYPE
        assert result.delivery["source_id"] == "evt_handoff_requested_0001"
        assert result.delivery["agent_event_id"] == "evt_handoff_requested_0001"
        assert result.delivery["discord_message_id"] is None
        assert result.delivery["target_agent_id"] == "agent-b"
        assert result.delivery["delivery_key"] == inbound_delivery_key(
            INTERNAL_SOURCE_TYPE,
            "evt_handoff_requested_0001",
            "agent-b",
        )
        assert result.handoff is not None
        assert result.handoff["status"] == "requested"
        assert result.outbox_delivery is not None
        assert result.outbox_delivery["delivery_kind"] == "projection"
        assert result.outbox_delivery["idempotency_key"] == projection_idempotency_key(
            "evt_handoff_requested_0001",
            "agent-b",
        )
        assert result.outbox_delivery["source_agent_event_id"] == "evt_handoff_requested_0001"
        assert store.count_rows("inbound_deliveries") == 1
        assert store.count_rows("outbox_deliveries") == 1
        assert json.loads(result.event["payload_json"])["api_token"] == "<redacted>"


def test_replay_same_internal_review_and_consult_create_no_duplicate_deliveries(tmp_path):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        _seed(store)

        first = request_review(
            store,
            agent_event_id="evt_review_requested_0001",
            source_agent_id="agent-a",
            target_agent_id="agent-b",
            topic_id="guild-1/channel-1/root",
            payload={"artifact": "draft"},
        )
        second = request_review(
            store,
            agent_event_id="evt_review_requested_0001",
            source_agent_id="agent-a",
            target_agent_id="agent-b",
            topic_id="guild-1/channel-1/root",
            payload={"artifact": "draft"},
        )
        consult = request_consult(
            store,
            agent_event_id="evt_consult_requested_0001",
            source_agent_id="agent-a",
            target_agent_id="agent-b",
            topic_id="guild-1/channel-1/root",
            payload={"question": "look?"},
        )

        assert second.event == first.event
        assert second.delivery == first.delivery
        assert second.outbox_delivery == first.outbox_delivery
        assert consult.event["event_type"] == "consult.requested"
        assert store.count_rows("agent_events") == 2
        assert store.count_rows("inbound_deliveries") == 2
        assert store.count_rows("outbox_deliveries") == 2
        assert store.count_rows(
            "inbound_deliveries",
            "source_type = ? AND source_id = ? AND target_agent_id = ?",
            (INTERNAL_SOURCE_TYPE, "evt_review_requested_0001", "agent-b"),
        ) == 1


def test_discord_replay_of_projection_preserves_correlation_and_creates_no_delivery(tmp_path):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        _seed(store)
        result = request_handoff(
            store,
            agent_event_id="evt_projection_replay_0001",
            source_agent_id="agent-a",
            target_agent_id="agent-b",
            topic_id="guild-1/channel-1/root",
            payload={"task": "projection replay"},
        )
        assert result.outbox_delivery is not None
        store.upsert_message_map(
            discord_message_id="discord-projection-1",
            guild_id="guild-1",
            channel_id="channel-1",
            direction="projection",
            agent_id="agent-b",
            outbox_delivery_id=result.outbox_delivery["outbox_delivery_id"],
            agent_event_id="evt_projection_replay_0001",
            author_id="bot-agent-b",
            author_kind="registered_bot",
            author_bot_user_id="bot-agent-b",
            source_client_agent_id="agent-b",
            mentions=["bot-agent-b"],
            payload={"part_index": 0},
        )

        ingestor = DiscordProtocolV2Ingestor(
            store=store,
            identity_registry=_registry(store),
            default_intake_agent_id="agent-a",
            guild_allowlist=["guild-1"],
        )
        before = store.count_rows("inbound_deliveries")
        ingest_result = ingestor.ingest_message(
            source_client_agent_id="agent-a",
            message=_message(
                message_id="discord-projection-1",
                author_id="bot-agent-b",
                mentions=[_mention("bot-agent-b")],
            ),
        )

        assert ingest_result.deliveries == []
        assert store.count_rows("inbound_deliveries") == before
        mapped = store.get_message_map("discord-projection-1")
        assert mapped is not None
        assert mapped["direction"] == "projection"
        assert mapped["agent_event_id"] == "evt_projection_replay_0001"
        assert mapped["outbox_delivery_id"] == result.outbox_delivery["outbox_delivery_id"]
        assert mapped["source_client_agent_id"] == "agent-b"
        decisions = store.route_decisions_for(DISCORD_SOURCE_TYPE, "discord-projection-1")
        assert len(decisions) == 1
        assert decisions[0]["decision"] == "zero_delivery"
        assert decisions[0]["reason"] == "non_human_author:registered_bot"


@pytest.mark.asyncio
async def test_internal_event_projection_send_reingest_and_replay_are_idempotent(tmp_path):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        _seed(store)
        request = request_review(
            store,
            agent_event_id="evt_integrated_projection_replay_0001",
            source_agent_id="agent-a",
            target_agent_id="agent-b",
            topic_id="guild-1/channel-1/root",
            payload={"artifact": "patch"},
            projection_content="<@bot-agent-b> please review patch",
        )
        assert request.delivery is not None
        assert request.outbox_delivery is not None
        assert request.delivery["source_type"] == INTERNAL_SOURCE_TYPE

        agent_a_client = _FakeDiscordClient("agent-a")
        agent_b_client = _FakeDiscordClient("agent-b")

        async def fake_send_part(client, *, content: str, **_kwargs):
            return await client.channel.send(content=content, reference=None)

        sender = DiscordProtocolV2OutboxSender(
            store=store,
            client_resolver=_resolver({"agent-a": agent_a_client, "agent-b": agent_b_client}),
            send_part=fake_send_part,
            worker_id="projection-sender",
        )

        sent = await sender.run_once()
        replayed_request = request_review(
            store,
            agent_event_id="evt_integrated_projection_replay_0001",
            source_agent_id="agent-a",
            target_agent_id="agent-b",
            topic_id="guild-1/channel-1/root",
            payload={"artifact": "patch"},
            projection_content="duplicate text must not replace stable projection",
        )
        duplicate_send = await sender.run_once()

        assert sent is not None
        assert sent.status == "sent"
        assert duplicate_send is None
        assert replayed_request.event == request.event
        assert replayed_request.delivery == request.delivery
        assert replayed_request.outbox_delivery is not None
        assert replayed_request.outbox_delivery["outbox_delivery_id"] == request.outbox_delivery[
            "outbox_delivery_id"
        ]
        assert replayed_request.outbox_delivery["status"] == "sent"
        assert agent_a_client.channel.sent == []
        assert agent_b_client.channel.sent == ["<@bot-agent-b> please review patch"]
        assert store.count_rows("agent_events") == 1
        assert store.count_rows("inbound_deliveries") == 1
        assert store.count_rows("outbox_deliveries") == 1

        mapped = store.get_message_map("agent-b-message-1")
        assert mapped is not None
        assert mapped["direction"] == "projection"
        assert mapped["agent_event_id"] == request.event["agent_event_id"]
        assert mapped["outbox_delivery_id"] == request.outbox_delivery["outbox_delivery_id"]
        assert mapped["source_client_agent_id"] == "agent-b"
        assert mapped["author_bot_user_id"] == "bot-agent-b"

        ingestor = DiscordProtocolV2Ingestor(
            store=store,
            identity_registry=_registry(store),
            default_intake_agent_id="agent-a",
            guild_allowlist=["guild-1"],
        )
        before_reingest = store.count_rows("inbound_deliveries")
        reingest = ingestor.ingest_message(
            source_client_agent_id="agent-b",
            message=_message(
                message_id="agent-b-message-1",
                author_id="bot-agent-b",
                mentions=[_mention("bot-agent-b")],
            ),
        )

        assert reingest.deliveries == []
        assert store.count_rows("inbound_deliveries") == before_reingest
        assert store.count_rows("agent_events") == 1
        assert store.count_rows("outbox_deliveries") == 1
        projected_after = store.get_message_map("agent-b-message-1")
        assert projected_after is not None
        assert projected_after["direction"] == "projection"
        assert projected_after["agent_event_id"] == request.event["agent_event_id"]
        decisions = store.route_decisions_for(DISCORD_SOURCE_TYPE, "agent-b-message-1")
        assert len(decisions) == 1
        assert decisions[0]["decision"] == "zero_delivery"
        assert decisions[0]["reason"] == "non_human_author:registered_bot"


def test_handoff_state_survives_restart_and_transition_is_not_duplicated(tmp_path):
    db_path = tmp_path / "discord-v2.sqlite3"
    store = DiscordProtocolV2Store(db_path)
    _seed(store)
    result = request_handoff(
        store,
        agent_event_id="evt_handoff_state_0001",
        source_agent_id="agent-a",
        target_agent_id="agent-b",
        topic_id="guild-1/channel-1/root",
        payload={"task": "durable state"},
    )
    assert result.handoff is not None
    store.close()

    restarted = DiscordProtocolV2Store(db_path)
    try:
        first = accept_handoff(
            restarted,
            agent_event_id="evt_handoff_state_0001",
            actor_agent_id="agent-b",
            payload={"note": "accepted"},
        )
        second = accept_handoff(
            restarted,
            agent_event_id="evt_handoff_state_0001",
            actor_agent_id="agent-b",
            payload={"note": "accepted"},
        )
        assert first.event["agent_event_id"] == second.event["agent_event_id"]
        assert second.handoff is not None
        assert second.handoff["status"] == "accepted"
        assert restarted.count_rows("handoffs", "agent_event_id = ?", ("evt_handoff_state_0001",)) == 1
        assert restarted.count_rows("agent_events") == 2
    finally:
        restarted.close()

    reopened = DiscordProtocolV2Store(db_path)
    try:
        handoff = reopened.get_handoff_by_agent_event("evt_handoff_state_0001")
        assert handoff is not None
        assert handoff["status"] == "accepted"
        assert reopened.count_rows("agent_events") == 2
        assert reopened.count_rows("inbound_deliveries") == 1
        assert reopened.count_rows("outbox_deliveries") == 1
    finally:
        reopened.close()


def test_replaying_requested_handoff_after_accept_does_not_reset_durable_status(tmp_path):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        _seed(store)
        first = request_handoff(
            store,
            agent_event_id="evt_handoff_replay_after_accept_0001",
            source_agent_id="agent-a",
            target_agent_id="agent-b",
            topic_id="guild-1/channel-1/root",
            payload={"task": "keep accepted"},
        )
        assert first.handoff is not None

        accepted = accept_handoff(
            store,
            agent_event_id="evt_handoff_replay_after_accept_0001",
            actor_agent_id="agent-b",
            payload={"note": "accepted"},
        )
        assert accepted.handoff is not None
        assert accepted.handoff["status"] == "accepted"
        inbound_before = store.count_rows("inbound_deliveries")
        outbox_before = store.count_rows("outbox_deliveries")

        replay = request_handoff(
            store,
            agent_event_id="evt_handoff_replay_after_accept_0001",
            source_agent_id="agent-a",
            target_agent_id="agent-b",
            topic_id="guild-1/channel-1/root",
            payload={"task": "keep accepted"},
        )

        assert replay.handoff is not None
        assert replay.handoff["status"] == "accepted"
        stored = store.get_handoff_by_agent_event("evt_handoff_replay_after_accept_0001")
        assert stored is not None
        assert stored["status"] == "accepted"
        assert store.count_rows("inbound_deliveries") == inbound_before
        assert store.count_rows("outbox_deliveries") == outbox_before


def test_agent_event_id_conflict_fails_closed_without_second_target_artifacts(tmp_path):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        _seed(store)
        request_handoff(
            store,
            agent_event_id="evt_handoff_conflict_0000001",
            source_agent_id="agent-a",
            target_agent_id="agent-b",
            topic_id="guild-1/channel-1/root",
            payload={"task": "original"},
        )

        with pytest.raises(ValueError, match="agent_event_id conflict"):
            request_handoff(
                store,
                agent_event_id="evt_handoff_conflict_0000001",
                source_agent_id="agent-a",
                target_agent_id="agent-a",
                topic_id="guild-1/channel-1/root",
                payload={"task": "original"},
            )

        assert store.count_rows("agent_events") == 1
        assert store.count_rows("handoffs") == 1
        assert store.count_rows("inbound_deliveries") == 1
        assert store.count_rows("outbox_deliveries") == 1
        assert store.count_rows("inbound_deliveries", "target_agent_id = ?", ("agent-a",)) == 0
        assert store.count_rows("outbox_deliveries", "target_agent_id = ?", ("agent-a",)) == 0


def test_projection_content_is_redacted_before_outbox_persistence(tmp_path):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        _seed(store)
        result = request_handoff(
            store,
            agent_event_id="evt_handoff_projection_redact_0001",
            source_agent_id="agent-a",
            target_agent_id="agent-b",
            topic_id="guild-1/channel-1/root",
            payload={"summary": "safe"},
            projection_content="token=abc.def.ghijklmnopqrstuvwxyz0123456789",
        )

        assert result.outbox_delivery is not None
        payload = json.loads(result.outbox_delivery["payload_json"])
        assert payload["content"] == "token=<redacted>"
        assert "abcdefghijklmnopqrstuvwxyz" not in result.outbox_delivery["payload_json"]
