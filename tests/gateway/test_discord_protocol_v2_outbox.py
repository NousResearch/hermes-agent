from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gateway.discord_protocol_v2_outbox import (
    DiscordProtocolV2ClientBinding,
    DiscordProtocolV2OutboxSender,
    PossibleSendCommittedError,
    create_projection_outbox_delivery,
    create_response_outbox_delivery,
)
from gateway.discord_protocol_v2_store import (
    DiscordProtocolV2Store,
    projection_idempotency_key,
    response_idempotency_key,
)


class FakeChannel:
    def __init__(self, client_name: str) -> None:
        self.client_name = client_name
        self.sent: list[str] = []

    async def send(self, *, content: str, reference=None):
        assert reference is None
        self.sent.append(content)
        return SimpleNamespace(id=f"{self.client_name}-message-{len(self.sent)}")


class FakeDiscordClient:
    def __init__(self, name: str, channel_id: str = "100") -> None:
        self.name = name
        self.channel = FakeChannel(name)
        self.channel_id = int(channel_id)

    def get_channel(self, channel_id: int):
        if channel_id == self.channel_id:
            return self.channel
        return None

    async def fetch_channel(self, channel_id: int):
        return self.get_channel(channel_id)


def _parts(store: DiscordProtocolV2Store) -> list[dict]:
    return [
        dict(row)
        for row in store.conn.execute(
            "SELECT * FROM outbox_parts ORDER BY outbox_delivery_id, part_index"
        ).fetchall()
    ]


def _seed_store(store: DiscordProtocolV2Store) -> dict:
    store.upsert_identity(
        agent_id="bohumil",
        hermes_profile="bohumil-profile",
        discord_application_id="app-bohumil",
        discord_bot_user_id="bot-bohumil",
        token_secret_ref="secret://discord/bohumil-token",
        capabilities=["chat"],
        scopes={"guild_ids": ["guild-1"]},
        enabled=True,
    )
    store.upsert_identity(
        agent_id="reviewer",
        hermes_profile="reviewer-profile",
        discord_application_id="app-reviewer",
        discord_bot_user_id="bot-reviewer",
        token_secret_ref="secret://discord/reviewer-token",
        capabilities=["chat"],
        scopes={"guild_ids": ["guild-1"]},
        enabled=True,
    )
    store.upsert_topic(
        topic_id="topic-1",
        guild_id="guild-1",
        channel_id="100",
        title="general",
        state={"mode": "active"},
    )
    return store.create_discord_inbound_deliveries(
        discord_message_id="inbound-discord-1",
        guild_id="guild-1",
        channel_id="100",
        topic_id="topic-1",
        author_id="human-1",
        author_kind="human",
        target_agent_ids=["bohumil"],
        route_reason="mention",
        mentions=["bohumil"],
        payload={"content": "hello", "author_id": "human-1"},
    )[0]


def _resolver(clients: dict[str, FakeDiscordClient]):
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


@pytest.mark.asyncio
async def test_same_logical_response_is_sent_once_and_mapped(tmp_path):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        inbound = _seed_store(store)
        first = create_response_outbox_delivery(
            store,
            inbound_delivery_key=inbound["delivery_key"],
            target_agent_id="bohumil",
            topic_id="topic-1",
            channel_id="100",
            content="ahoj z outboxu",
            mentions=["human-1"],
        )
        replay = create_response_outbox_delivery(
            store,
            inbound_delivery_key=inbound["delivery_key"],
            target_agent_id="bohumil",
            topic_id="topic-1",
            channel_id="100",
            content="different text must not replace the stable row",
            mentions=["human-1"],
        )
        client = FakeDiscordClient("bohumil")
        sender = DiscordProtocolV2OutboxSender(
            store=store,
            client_resolver=_resolver({"bohumil": client}),
            worker_id="sender-a",
        )

        result = await sender.run_once()
        duplicate = await sender.deliver_outbox(store.get_outbox_delivery(first["outbox_delivery_id"]))
        idle = await sender.run_once()

        assert first == replay
        assert result.status == "sent"
        assert duplicate.status == "already_sent"
        assert idle is None
        assert store.count_rows("outbox_deliveries") == 1
        assert client.channel.sent == ["ahoj z outboxu"]
        assert first["idempotency_key"] == response_idempotency_key(
            inbound["delivery_key"], "bohumil"
        )
        assert first["source_inbound_delivery_key"] == inbound["delivery_key"]

        parts = _parts(store)
        assert [(part["part_index"], part["discord_message_id"]) for part in parts] == [
            (0, "bohumil-message-1")
        ]
        mapped = store.get_message_map("bohumil-message-1")
        assert mapped["direction"] == "outbound"
        assert mapped["source_client_agent_id"] == "bohumil"
        assert mapped["author_kind"] == "registered_bot"
        assert mapped["author_bot_user_id"] == "bot-bohumil"
        assert json.loads(mapped["mentions_json"]) == ["human-1"]


@pytest.mark.asyncio
async def test_same_projection_is_sent_once_and_uses_projection_idempotency(tmp_path):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        _seed_store(store)
        event = store.create_agent_event(
            agent_event_id="event-1",
            event_type="handoff.requested",
            source_agent_id="bohumil",
            target_agent_id="reviewer",
            topic_id="topic-1",
            payload={"task": "review"},
        )
        first = create_projection_outbox_delivery(
            store,
            agent_event_id=event["agent_event_id"],
            target_agent_id="reviewer",
            topic_id="topic-1",
            channel_id="100",
            content="Reviewer, please inspect this handoff.",
            mentions=["reviewer"],
        )
        replay = create_projection_outbox_delivery(
            store,
            agent_event_id=event["agent_event_id"],
            target_agent_id="reviewer",
            topic_id="topic-1",
            channel_id="100",
            content="duplicate projection",
            mentions=["reviewer"],
        )
        client = FakeDiscordClient("reviewer")
        sender = DiscordProtocolV2OutboxSender(
            store=store,
            client_resolver=_resolver({"reviewer": client}),
            worker_id="sender-projection",
        )

        result = await sender.run_once()
        idle = await sender.run_once()

        assert first == replay
        assert result.status == "sent"
        assert idle is None
        assert store.count_rows("outbox_deliveries") == 1
        assert client.channel.sent == ["Reviewer, please inspect this handoff."]
        assert first["idempotency_key"] == projection_idempotency_key(
            event["agent_event_id"], "reviewer"
        )
        assert first["source_agent_event_id"] == event["agent_event_id"]
        mapped = store.get_message_map("reviewer-message-1")
        assert mapped["direction"] == "projection"
        assert mapped["agent_event_id"] == event["agent_event_id"]
        assert mapped["source_client_agent_id"] == "reviewer"
        assert mapped["author_bot_user_id"] == "bot-reviewer"
        assert json.loads(mapped["mentions_json"]) == ["reviewer"]


@pytest.mark.asyncio
async def test_multi_part_response_persists_every_stable_part_index(tmp_path):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        inbound = _seed_store(store)
        content = "\n".join(["x" * 120 for _ in range(45)])
        outbox = create_response_outbox_delivery(
            store,
            inbound_delivery_key=inbound["delivery_key"],
            target_agent_id="bohumil",
            topic_id="topic-1",
            channel_id="100",
            content=content,
        )
        client = FakeDiscordClient("bohumil")
        sender = DiscordProtocolV2OutboxSender(
            store=store,
            client_resolver=_resolver({"bohumil": client}),
            worker_id="sender-multipart",
        )

        result = await sender.run_once()

        assert result.status == "sent"
        assert result.sent_parts > 1
        assert store.get_outbox_delivery(outbox["outbox_delivery_id"])["status"] == "sent"
        assert all(len(chunk) <= 2000 for chunk in client.channel.sent)
        parts = _parts(store)
        assert [part["part_index"] for part in parts] == list(range(len(parts)))
        assert [part["status"] for part in parts] == ["sent"] * len(parts)
        assert [part["discord_message_id"] for part in parts] == [
            f"bohumil-message-{index + 1}" for index in range(len(parts))
        ]
        assert store.count_rows("message_map") == len(parts) + 1  # includes inbound map


@pytest.mark.asyncio
async def test_missing_client_fails_closed_and_expired_lease_is_retryable(tmp_path):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        inbound = _seed_store(store)
        outbox = create_response_outbox_delivery(
            store,
            inbound_delivery_key=inbound["delivery_key"],
            target_agent_id="bohumil",
            topic_id="topic-1",
            channel_id="100",
            content="retry me later",
        )
        missing_sender = DiscordProtocolV2OutboxSender(
            store=store,
            client_resolver=_resolver({}),
            worker_id="missing-sender",
            lease_seconds=30,
        )

        missing = await missing_sender.run_once()

        assert missing.status == "missing_client"
        assert store.get_outbox_delivery(outbox["outbox_delivery_id"])["status"] == "leased"
        assert store.count_rows("outbox_parts") == 0
        store.conn.execute(
            "UPDATE outbox_deliveries SET lease_until = '2000-01-01T00:00:00+00:00' WHERE outbox_delivery_id = ?",
            (outbox["outbox_delivery_id"],),
        )
        store.conn.commit()

        client = FakeDiscordClient("bohumil")
        retry_sender = DiscordProtocolV2OutboxSender(
            store=store,
            client_resolver=_resolver({"bohumil": client}),
            worker_id="retry-sender",
        )
        retried = await retry_sender.run_once()

        assert retried.status == "sent"
        assert client.channel.sent == ["retry me later"]
        assert store.get_outbox_delivery(outbox["outbox_delivery_id"])["status"] == "sent"


@pytest.mark.asyncio
async def test_wrong_client_binding_fails_closed_without_wrong_agent_artifacts(tmp_path):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        inbound = _seed_store(store)
        outbox = create_response_outbox_delivery(
            store,
            inbound_delivery_key=inbound["delivery_key"],
            target_agent_id="bohumil",
            topic_id="topic-1",
            channel_id="100",
            content="must use bohumil client",
        )
        reviewer_client = FakeDiscordClient("reviewer")

        def wrong_resolver(_agent_id: str):
            return DiscordProtocolV2ClientBinding(
                client=reviewer_client,
                source_client_agent_id="reviewer",
                author_bot_user_id="bot-reviewer",
            )

        mismatch_sender = DiscordProtocolV2OutboxSender(
            store=store,
            client_resolver=wrong_resolver,
            worker_id="wrong-client-sender",
            lease_seconds=30,
        )

        mismatch = await mismatch_sender.run_once()

        assert mismatch is not None
        assert mismatch.status == "missing_client"
        assert mismatch.error is not None
        assert "does not match target agent" in mismatch.error
        assert reviewer_client.channel.sent == []
        after_mismatch = store.get_outbox_delivery(outbox["outbox_delivery_id"])
        assert after_mismatch is not None
        assert after_mismatch["status"] == "leased"
        assert store.count_rows("outbox_parts") == 0
        wrong_client_maps = store.conn.execute(
            """
            SELECT COUNT(*) FROM message_map
            WHERE outbox_delivery_id = ? AND source_client_agent_id = 'reviewer'
            """,
            (outbox["outbox_delivery_id"],),
        ).fetchone()[0]
        assert wrong_client_maps == 0
        store.conn.execute(
            "UPDATE outbox_deliveries SET lease_until = '2000-01-01T00:00:00+00:00' WHERE outbox_delivery_id = ?",
            (outbox["outbox_delivery_id"],),
        )
        store.conn.commit()

        bohumil_client = FakeDiscordClient("bohumil")
        retry_sender = DiscordProtocolV2OutboxSender(
            store=store,
            client_resolver=_resolver({"bohumil": bohumil_client}),
            worker_id="right-client-sender",
        )
        retried = await retry_sender.run_once()

        assert retried is not None
        assert retried.status == "sent"
        assert bohumil_client.channel.sent == ["must use bohumil client"]
        assert reviewer_client.channel.sent == []
        after_retry = store.get_outbox_delivery(outbox["outbox_delivery_id"])
        assert after_retry is not None
        assert after_retry["status"] == "sent"
        mapped = store.get_message_map("bohumil-message-1")
        assert mapped is not None
        assert mapped["source_client_agent_id"] == "bohumil"


@pytest.mark.asyncio
async def test_expired_same_owner_outbox_lease_cannot_send_or_persist_artifacts(tmp_path):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        inbound = _seed_store(store)
        outbox = create_response_outbox_delivery(
            store,
            inbound_delivery_key=inbound["delivery_key"],
            target_agent_id="bohumil",
            topic_id="topic-1",
            channel_id="100",
            content="expired same owner must not send",
        )
        expired_lease = store.lease_next_outbox(lease_owner="sender-a", lease_seconds=-1)
        assert expired_lease is not None
        client = FakeDiscordClient("bohumil")
        sender = DiscordProtocolV2OutboxSender(
            store=store,
            client_resolver=_resolver({"bohumil": client}),
            worker_id="sender-a",
        )

        result = await sender.deliver_outbox(expired_lease)

        current = store.get_outbox_delivery(outbox["outbox_delivery_id"])
        assert current is not None
        assert result.status == "lease_lost"
        assert current["status"] == "leased"
        assert current["lease_owner"] == "sender-a"
        assert client.channel.sent == []
        assert store.count_rows("outbox_parts") == 0
        assert store.count_rows("message_map", "outbox_delivery_id IS NOT NULL") == 0


@pytest.mark.asyncio
async def test_stale_outbox_lease_owner_cannot_send_or_persist_artifacts(tmp_path):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        inbound = _seed_store(store)
        outbox = create_response_outbox_delivery(
            store,
            inbound_delivery_key=inbound["delivery_key"],
            target_agent_id="bohumil",
            topic_id="topic-1",
            channel_id="100",
            content="stale sender must not send",
        )
        stale_lease = store.lease_next_outbox(lease_owner="sender-a", lease_seconds=-1)
        fresh_lease = store.lease_next_outbox(lease_owner="sender-b", lease_seconds=30)
        client = FakeDiscordClient("bohumil")
        sender = DiscordProtocolV2OutboxSender(
            store=store,
            client_resolver=_resolver({"bohumil": client}),
            worker_id="sender-a",
        )

        result = await sender.deliver_outbox(stale_lease)

        current = store.get_outbox_delivery(outbox["outbox_delivery_id"])
        assert result.status == "lease_lost"
        assert fresh_lease["lease_owner"] == "sender-b"
        assert current["status"] == "leased"
        assert current["lease_owner"] == "sender-b"
        assert client.channel.sent == []
        assert store.count_rows("outbox_parts") == 0
        assert store.count_rows("message_map", "outbox_delivery_id IS NOT NULL") == 0


@pytest.mark.asyncio
async def test_outbox_sender_does_not_persist_parts_or_map_after_losing_lease_mid_send(tmp_path):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        inbound = _seed_store(store)
        outbox = create_response_outbox_delivery(
            store,
            inbound_delivery_key=inbound["delivery_key"],
            target_agent_id="bohumil",
            topic_id="topic-1",
            channel_id="100",
            content="lease lost after side effect",
        )

        async def steal_lease_after_send(*args, **kwargs):
            store.conn.execute(
                """
                UPDATE outbox_deliveries
                SET lease_owner = 'sender-b', state_version = state_version + 1
                WHERE outbox_delivery_id = ?
                """,
                (outbox["outbox_delivery_id"],),
            )
            store.conn.commit()
            return SimpleNamespace(id="discord-sent-but-not-owned")

        sender = DiscordProtocolV2OutboxSender(
            store=store,
            client_resolver=_resolver({"bohumil": FakeDiscordClient("bohumil")}),
            send_part=steal_lease_after_send,
            worker_id="sender-a",
        )

        result = await sender.run_once()

        current = store.get_outbox_delivery(outbox["outbox_delivery_id"])
        assert result.status == "lease_lost"
        assert current["status"] == "sending"
        assert current["lease_owner"] == "sender-b"
        assert store.count_rows("outbox_parts") == 1
        assert _parts(store)[0]["discord_message_id"] is None
        assert store.count_rows("message_map", "outbox_delivery_id IS NOT NULL") == 0


@pytest.mark.asyncio
async def test_uncertain_send_exception_marks_uncertain_without_failed_state(tmp_path):
    with DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3") as store:
        inbound = _seed_store(store)
        outbox = create_response_outbox_delivery(
            store,
            inbound_delivery_key=inbound["delivery_key"],
            target_agent_id="bohumil",
            topic_id="topic-1",
            channel_id="100",
            content="maybe committed",
        )

        async def uncertain_send(*args, **kwargs):
            raise PossibleSendCommittedError("network result unknown")

        sender = DiscordProtocolV2OutboxSender(
            store=store,
            client_resolver=_resolver({"bohumil": FakeDiscordClient("bohumil")}),
            send_part=uncertain_send,
            worker_id="sender-uncertain",
        )

        result = await sender.run_once()

        assert result.status == "uncertain"
        assert result.error == "PossibleSendCommittedError"
        assert store.get_outbox_delivery(outbox["outbox_delivery_id"])["status"] == "uncertain"
        assert {row["status"] for row in _parts(store)} == {"pending"}
