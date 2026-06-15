from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from gateway.config import Platform
from gateway.discord_identity_registry import DiscordIdentityMetadata, DiscordIdentityRegistry
from gateway.discord_protocol_v2_outbox import (
    DiscordProtocolV2ClientBinding,
    DiscordProtocolV2OutboxSender,
    record_outbox_message_map,
    split_discord_message,
)
from gateway.discord_protocol_v2_reconcile import reconcile_discord_protocol_v2_outbox
from gateway.discord_protocol_v2_store import DiscordProtocolV2Store
from gateway.discord_protocol_v2_worker import DiscordProtocolV2Worker
from gateway.session import SessionSource


@dataclass
class _FakeEntry:
    session_key: str
    session_id: str
    origin: SessionSource


class _FakeSessionStore:
    def __init__(self) -> None:
        self.entries: dict[str, _FakeEntry] = {}

    def get_or_create_session_for_key(
        self,
        session_key: str,
        source: SessionSource,
        force_new: bool = False,
    ) -> _FakeEntry:
        if force_new or session_key not in self.entries:
            self.entries[session_key] = _FakeEntry(
                session_key=session_key,
                session_id=f"session-{len(self.entries) + 1}",
                origin=source,
            )
        return self.entries[session_key]

    def bind_session_key(
        self,
        session_key: str,
        session_id: str,
        source: SessionSource,
    ) -> _FakeEntry:
        self.entries[session_key] = _FakeEntry(
            session_key=session_key,
            session_id=session_id,
            origin=source,
        )
        return self.entries[session_key]

    def load_transcript(self, session_id: str) -> list:
        return []


def _registry() -> DiscordIdentityRegistry:
    identity = DiscordIdentityMetadata(
        agent_id="bohumil",
        hermes_profile="bohumil-profile",
        discord_application_id="app-bohumil",
        discord_bot_user_id="bot-bohumil",
        token_secret_ref="secret://discord/bohumil-token",
        capabilities=("chat",),
        allowed_scopes={"guild_ids": ["guild-1"]},
        enabled=True,
    )
    return DiscordIdentityRegistry(
        enabled=True,
        mode="active",
        identities={"bohumil": identity},
        active_agent_ids={"bohumil"},
        secret_resolver=None,
    )


def _seed_restart_store(store: DiscordProtocolV2Store) -> dict:
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
    store.upsert_topic(
        topic_id="guild-1/100/root",
        guild_id="guild-1",
        channel_id="100",
        title="general",
        state={"mode": "active"},
    )
    delivery = store.create_discord_inbound_deliveries(
        discord_message_id="discord-inbound-1",
        guild_id="guild-1",
        channel_id="100",
        topic_id="guild-1/100/root",
        author_id="human-1",
        author_kind="human",
        target_agent_ids=["bohumil"],
        route_reason="mention",
        mentions=["bohumil"],
        payload={"content": "hello bohumil", "author_id": "human-1"},
    )[0]
    event = store.create_agent_event(
        agent_event_id="event-handoff-1",
        event_type="handoff.requested",
        source_agent_id="bohumil",
        target_agent_id="bohumil",
        topic_id="guild-1/100/root",
        payload={"task": "review"},
    )
    store.upsert_approval(
        approval_id="approval-1",
        source_inbound_delivery_key=delivery["delivery_key"],
        target_agent_id="bohumil",
        topic_id="guild-1/100/root",
        payload={"prompt": "approve send"},
    )
    store.upsert_handoff(
        handoff_id="handoff-1",
        agent_event_id=event["agent_event_id"],
        source_agent_id="bohumil",
        target_agent_id="bohumil",
        topic_id="guild-1/100/root",
        payload={"task": "review"},
    )
    store.upsert_topic_agent_session(
        topic_id="guild-1/100/root",
        agent_id="bohumil",
        hermes_session_id="session-persisted-1",
        session_key="discord:v2:topic:guild-1/100/root:agent:bohumil",
    )
    return delivery


def _worker(store: DiscordProtocolV2Store, invoker) -> DiscordProtocolV2Worker:
    return DiscordProtocolV2Worker(
        store=store,
        identity_registry=_registry(),
        session_store=_FakeSessionStore(),
        invoker=invoker,
        worker_id="restart-worker",
        lease_seconds=30,
    )


@pytest.mark.asyncio
async def test_restart_reconciles_crashed_inbound_and_outbox_without_duplicate_send(tmp_path):
    db_path = tmp_path / "discord-v2.sqlite3"

    store = DiscordProtocolV2Store(db_path)
    delivery = _seed_restart_store(store)
    crashed_inbound = store.lease_next_inbound(
        lease_owner="crashed-worker",
        lease_seconds=-1,
    )
    assert crashed_inbound is not None
    assert crashed_inbound["delivery_key"] == delivery["delivery_key"]
    store.close()

    restarted = DiscordProtocolV2Store(db_path)
    calls = 0

    def fake_agent(context):
        nonlocal calls
        calls += 1
        assert context.delivery["attempts"] == 2
        assert context.session.session_id == "session-persisted-1"
        return "restart-safe response"

    worker_result = await _worker(restarted, fake_agent).run_once()
    assert worker_result is not None
    assert worker_result.status == "completed"
    assert calls == 1
    assert restarted.get_inbound_delivery(delivery["delivery_key"])["status"] == "completed"
    assert restarted.get_inbound_delivery(delivery["delivery_key"])["attempts"] == 2
    assert restarted.count_rows("outbox_deliveries") == 1

    outbox = worker_result.outbox_delivery
    assert outbox is not None
    restarted.mark_outbox_sending(outbox["outbox_delivery_id"])
    restarted.close()

    after_crash = DiscordProtocolV2Store(db_path)
    assert after_crash.count_rows("approvals", "status = 'pending'") == 1
    assert after_crash.count_rows("handoffs", "status = 'pending'") == 1
    persisted_session = after_crash.get_topic_agent_session(
        topic_id="guild-1/100/root",
        agent_id="bohumil",
    )
    assert persisted_session is not None
    assert persisted_session["hermes_session_id"] == "session-persisted-1"
    persisted_topic = after_crash.get_topic("guild-1/100/root")
    assert persisted_topic is not None
    assert persisted_topic["channel_id"] == "100"

    history_calls = 0

    async def recent_history(outbox_row):
        nonlocal history_calls
        history_calls += 1
        assert outbox_row["status"] == "sending"
        return [
            {
                "id": "discord-outbound-1",
                "content": "restart-safe response",
                "channel_id": "100",
                "author_id": "bot-bohumil",
            }
        ]

    reconciliation = await reconcile_discord_protocol_v2_outbox(
        store=after_crash,
        recent_history_fetcher=recent_history,
        run_id="restart-test-reconcile",
    )

    assert reconciliation.scanned == 1
    assert reconciliation.acked == 1
    assert reconciliation.enqueued == 0
    assert history_calls == 1
    stored_outbox = after_crash.get_outbox_delivery(outbox["outbox_delivery_id"])
    assert stored_outbox["status"] == "acked"
    assert after_crash.count_rows("outbox_parts") == 1
    mapped = after_crash.get_message_map("discord-outbound-1")
    assert mapped is not None
    assert mapped["outbox_delivery_id"] == outbox["outbox_delivery_id"]
    audit_row = after_crash.conn.execute(
        "SELECT payload_json FROM reconciliation_runs WHERE outbox_delivery_id = ?",
        (outbox["outbox_delivery_id"],),
    ).fetchone()
    assert audit_row is not None
    audit_payload = json.loads(audit_row[0])
    assert audit_payload["run_id"] == "restart-test-reconcile"

    sends: list[str] = []

    async def send_part(_client, *, content: str, **_kwargs):  # pragma: no cover - guard
        sends.append(content)
        return {"id": f"duplicate-{len(sends)}"}

    sender = DiscordProtocolV2OutboxSender(
        store=after_crash,
        client_resolver=lambda agent_id: DiscordProtocolV2ClientBinding(
            client=object(),
            source_client_agent_id=agent_id,
            author_bot_user_id="bot-bohumil",
        ),
        send_part=send_part,
        worker_id="post-reconcile-sender",
    )
    assert await sender.run_once() is None
    assert sends == []
    assert await _worker(after_crash, fake_agent).run_once() is None
    assert calls == 1
    assert after_crash.count_rows("outbox_deliveries") == 1
    assert after_crash.count_rows("message_map", "outbox_delivery_id IS NOT NULL") == 1


@pytest.mark.asyncio
async def test_reconciliation_resets_uncommitted_uncertain_outbox_for_one_retry(tmp_path):
    db_path = tmp_path / "discord-v2.sqlite3"
    with DiscordProtocolV2Store(db_path) as store:
        delivery = _seed_restart_store(store)
        outbox = store.create_outbox_delivery(
            idempotency_key="response:uncertain-retry",
            target_agent_id="bohumil",
            topic_id="guild-1/100/root",
            channel_id="100",
            delivery_kind="response",
            source_inbound_delivery_key=delivery["delivery_key"],
            payload={"content": "send after reconcile"},
        )
        leased = store.lease_next_outbox(lease_owner="sender-before-crash", lease_seconds=-1)
        assert leased["outbox_delivery_id"] == outbox["outbox_delivery_id"]
        store.mark_outbox_uncertain(outbox["outbox_delivery_id"])

    reopened = DiscordProtocolV2Store(db_path)
    reconciliation = await reconcile_discord_protocol_v2_outbox(
        store=reopened,
        recent_history_fetcher=lambda _outbox: [],
        max_attempts=3,
        run_id="retry-test-reconcile",
    )
    assert reconciliation.scanned == 1
    assert reconciliation.enqueued == 1
    pending_outbox = reopened.get_outbox_delivery(outbox["outbox_delivery_id"])
    assert pending_outbox is not None
    assert pending_outbox["status"] == "pending"

    sends: list[str] = []

    async def send_part(_client, *, content: str, **_kwargs):
        sends.append(content)
        return {"id": "discord-retry-1"}

    sender = DiscordProtocolV2OutboxSender(
        store=reopened,
        client_resolver=lambda agent_id: DiscordProtocolV2ClientBinding(
            client=object(),
            source_client_agent_id=agent_id,
            author_bot_user_id="bot-bohumil",
        ),
        send_part=send_part,
        worker_id="retry-sender",
    )
    result = await sender.run_once()

    assert result is not None
    assert result.status == "sent"
    assert sends == ["send after reconcile"]
    assert await sender.run_once() is None
    sent_outbox = reopened.get_outbox_delivery(outbox["outbox_delivery_id"])
    assert sent_outbox is not None
    assert sent_outbox["status"] == "sent"
    assert reopened.count_rows("message_map", "outbox_delivery_id IS NOT NULL") == 1


@pytest.mark.asyncio
async def test_reconciliation_persists_partial_history_before_retrying_missing_parts(tmp_path):
    db_path = tmp_path / "discord-v2.sqlite3"
    content = ("a" * 2000) + "b"
    first_chunk, second_chunk = split_discord_message(content)
    with DiscordProtocolV2Store(db_path) as store:
        _seed_restart_store(store)
        outbox = store.create_outbox_delivery(
            idempotency_key="response:partial-history-evidence",
            target_agent_id="bohumil",
            topic_id="guild-1/100/root",
            channel_id="100",
            delivery_kind="response",
            payload={"content": content},
        )
        leased = store.lease_next_outbox(
            lease_owner="sender-before-crash",
            lease_seconds=-1,
        )
        assert leased is not None
        assert leased["outbox_delivery_id"] == outbox["outbox_delivery_id"]
        store.mark_outbox_sending(outbox["outbox_delivery_id"])

    reopened = DiscordProtocolV2Store(db_path)
    history_calls = 0

    async def recent_history(outbox_row):
        nonlocal history_calls
        history_calls += 1
        assert outbox_row["status"] == "sending"
        return [
            {
                "id": "discord-history-part-0",
                "content": first_chunk,
                "channel_id": "100",
                "author_id": "bot-bohumil",
            }
        ]

    reconciliation = await reconcile_discord_protocol_v2_outbox(
        store=reopened,
        recent_history_fetcher=recent_history,
        max_attempts=3,
        run_id="partial-history-reconcile",
    )

    assert reconciliation.scanned == 1
    assert reconciliation.acked == 0
    assert reconciliation.enqueued == 1
    assert history_calls == 1
    pending_outbox = reopened.get_outbox_delivery(outbox["outbox_delivery_id"])
    assert pending_outbox is not None
    assert pending_outbox["status"] == "pending"
    part_rows = reopened.conn.execute(
        "SELECT * FROM outbox_parts ORDER BY part_index"
    ).fetchall()
    assert [(row["part_index"], row["discord_message_id"]) for row in part_rows] == [
        (0, "discord-history-part-0")
    ]
    history_map = reopened.get_message_map("discord-history-part-0")
    assert history_map is not None
    assert history_map["outbox_delivery_id"] == outbox["outbox_delivery_id"]

    sends: list[tuple[int, str]] = []

    async def send_part(_client, *, content: str, part_index: int, **_kwargs):
        sends.append((part_index, content))
        return {"id": f"discord-retry-part-{part_index}"}

    sender = DiscordProtocolV2OutboxSender(
        store=reopened,
        client_resolver=lambda agent_id: DiscordProtocolV2ClientBinding(
            client=object(),
            source_client_agent_id=agent_id,
            author_bot_user_id="bot-bohumil",
        ),
        send_part=send_part,
        worker_id="partial-history-sender",
    )
    result = await sender.run_once()

    assert result is not None
    assert result.status == "sent"
    assert result.sent_parts == 1
    assert result.skipped_parts == 1
    assert sends == [(1, second_chunk)]
    sent_outbox = reopened.get_outbox_delivery(outbox["outbox_delivery_id"])
    assert sent_outbox is not None
    assert sent_outbox["status"] == "sent"
    assert reopened.count_rows("outbox_parts") == 2
    assert reopened.get_message_map("discord-history-part-0") is not None
    assert reopened.get_message_map("discord-retry-part-1") is not None


@pytest.mark.asyncio
async def test_reconciliation_acks_from_existing_message_map_and_outbox_parts(tmp_path):
    db_path = tmp_path / "discord-v2.sqlite3"
    with DiscordProtocolV2Store(db_path) as store:
        _seed_restart_store(store)
        content = ("a" * 2000) + "b"
        outbox = store.create_outbox_delivery(
            idempotency_key="response:existing-local-evidence",
            target_agent_id="bohumil",
            topic_id="guild-1/100/root",
            channel_id="100",
            delivery_kind="response",
            payload={"content": content},
        )
        leased = store.lease_next_outbox(
            lease_owner="sender-before-crash",
            lease_seconds=-1,
        )
        assert leased is not None
        assert leased["outbox_delivery_id"] == outbox["outbox_delivery_id"]
        sending = store.mark_outbox_sending(outbox["outbox_delivery_id"])
        store.add_outbox_part(
            outbox_delivery_id=outbox["outbox_delivery_id"],
            part_index=0,
            status="sent",
            discord_message_id="discord-part-0",
        )
        record_outbox_message_map(
            store=store,
            outbox=sending,
            payload={"content": content},
            part_index=1,
            content="b",
            discord_message_id="discord-part-1",
            source_client_agent_id="bohumil",
            author_bot_user_id="bot-bohumil",
        )

    reopened = DiscordProtocolV2Store(db_path)
    history_calls = 0

    async def recent_history(_outbox):  # pragma: no cover - guard
        nonlocal history_calls
        history_calls += 1
        return []

    reconciliation = await reconcile_discord_protocol_v2_outbox(
        store=reopened,
        recent_history_fetcher=recent_history,
        run_id="existing-local-evidence-reconcile",
    )

    assert reconciliation.scanned == 1
    assert reconciliation.acked == 1
    assert history_calls == 0
    stored_outbox = reopened.get_outbox_delivery(outbox["outbox_delivery_id"])
    assert stored_outbox is not None
    assert stored_outbox["status"] == "acked"
    assert reopened.count_rows("outbox_parts") == 2
    assert reopened.get_message_map("discord-part-0") is not None
    assert reopened.get_message_map("discord-part-1") is not None
