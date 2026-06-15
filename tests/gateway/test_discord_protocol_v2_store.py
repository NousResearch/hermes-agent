"""Slice 0B durable contract tests for Discord protocol v2 store."""

from __future__ import annotations

import json
import sqlite3

import pytest

from gateway.discord_protocol_v2_store import (
    DISCORD_SOURCE_TYPE,
    INTERNAL_SOURCE_TYPE,
    DiscordProtocolV2Store,
    inbound_delivery_key,
    projection_idempotency_key,
    response_idempotency_key,
)
from gateway.secret_refs import redact_sensitive_data


REQUIRED_COLUMNS = {
    "identity_registry": {
        "agent_id",
        "hermes_profile",
        "discord_application_id",
        "discord_bot_user_id",
        "token_secret_ref",
        "capabilities_json",
        "scopes_json",
        "enabled",
        "version",
    },
    "topics": {
        "topic_id",
        "guild_id",
        "channel_id",
        "thread_id",
        "parent_channel_id",
        "title",
        "state_json",
        "version",
    },
    "topic_agent_sessions": {
        "topic_id",
        "agent_id",
        "hermes_session_id",
        "session_key",
        "state",
        "version",
    },
    "message_map": {
        "discord_message_id",
        "guild_id",
        "channel_id",
        "thread_id",
        "parent_channel_id",
        "direction",
        "agent_id",
        "delivery_key",
        "outbox_delivery_id",
        "agent_event_id",
        "author_id",
        "author_kind",
        "author_bot_user_id",
        "source_client_agent_id",
        "mentions_json",
        "created_at",
        "updated_at",
        "payload_json",
    },
    "agent_events": {
        "agent_event_id",
        "event_type",
        "source_agent_id",
        "target_agent_id",
        "topic_id",
        "payload_json",
        "status",
        "created_at",
        "version",
    },
    "inbound_deliveries": {
        "delivery_key",
        "source_type",
        "source_id",
        "discord_message_id",
        "agent_event_id",
        "target_agent_id",
        "topic_id",
        "route_reason",
        "author_kind",
        "payload_json",
        "status",
        "lease_owner",
        "lease_until",
        "attempts",
        "created_at",
        "updated_at",
        "state_version",
    },
    "outbox_deliveries": {
        "outbox_delivery_id",
        "idempotency_key",
        "target_agent_id",
        "topic_id",
        "channel_id",
        "thread_id",
        "source_inbound_delivery_key",
        "source_agent_event_id",
        "delivery_kind",
        "payload_json",
        "status",
        "lease_owner",
        "lease_until",
        "attempts",
        "created_at",
        "updated_at",
        "state_version",
    },
    "outbox_parts": {
        "outbox_delivery_id",
        "part_index",
        "status",
        "discord_message_id",
    },
    "route_decisions": {
        "decision_id",
        "source_type",
        "source_id",
        "topic_id",
        "author_kind",
        "decision",
        "target_agent_ids_json",
        "reason",
        "created_at",
        "payload_json",
    },
    "approvals": {
        "approval_id",
        "source_inbound_delivery_key",
        "source_agent_event_id",
        "target_agent_id",
        "topic_id",
        "status",
        "payload_json",
        "created_at",
        "updated_at",
        "version",
    },
    "handoffs": {
        "handoff_id",
        "agent_event_id",
        "source_agent_id",
        "target_agent_id",
        "topic_id",
        "status",
        "payload_json",
        "created_at",
        "updated_at",
        "version",
    },
    "reconciliation_runs": {
        "reconciliation_run_id",
        "source_agent_event_id",
        "outbox_delivery_id",
        "status",
        "payload_json",
        "created_at",
        "updated_at",
        "version",
    },
}


def _store(tmp_path) -> DiscordProtocolV2Store:
    return DiscordProtocolV2Store(tmp_path / "discord-v2.sqlite3")


def _seed_identity_and_topic(store: DiscordProtocolV2Store) -> None:
    store.upsert_identity(
        agent_id="bohumil",
        hermes_profile="default",
        discord_application_id="111111111111111111",
        discord_bot_user_id="222222222222222222",
        token_secret_ref="secret://hermes/discord/bohumil-token",
        capabilities=["intake", "reply"],
        scopes={"guild_ids": ["333333333333333333"]},
        enabled=True,
    )
    store.upsert_identity(
        agent_id="karel",
        hermes_profile="default",
        discord_application_id="444444444444444444",
        discord_bot_user_id="555555555555555555",
        token_secret_ref="secret://hermes/discord/karel-token",
        capabilities=["consult"],
        scopes={"guild_ids": ["333333333333333333"]},
        enabled=True,
    )
    store.upsert_topic(
        topic_id="topic-1",
        guild_id="333333333333333333",
        channel_id="777777777777777777",
        title="incident triage",
        state={"phase": "open"},
    )
    store.upsert_topic_agent_session(
        topic_id="topic-1",
        agent_id="bohumil",
        hermes_session_id="session-bohumil",
        session_key="discord:333:777:bohumil",
    )


def test_schema_contract_contains_slice_0b_tables_and_columns(tmp_path):
    with _store(tmp_path) as store:
        for table, columns in REQUIRED_COLUMNS.items():
            assert columns <= store.table_columns(table)

        indexes = {
            row["name"]
            for row in store.conn.execute("PRAGMA index_list(inbound_deliveries)").fetchall()
        }
        assert any("source" in name or "sqlite_autoindex" in name for name in indexes)


def test_restart_gate_persists_pending_records_and_stable_ids(tmp_path):
    db_path = tmp_path / "restart.sqlite3"
    store = DiscordProtocolV2Store(db_path)
    _seed_identity_and_topic(store)
    discord_delivery = store.create_discord_inbound_deliveries(
        discord_message_id="dm-1",
        guild_id="333333333333333333",
        channel_id="777777777777777777",
        topic_id="topic-1",
        author_id="human-1",
        author_kind="human",
        target_agent_ids=["bohumil"],
        route_reason="mention",
        payload={"content": "ahoj @bohumil"},
    )[0]
    event, internal_delivery = store.create_internal_handoff(
        event_type="handoff.requested",
        agent_event_id="event-handoff-1",
        source_agent_id="bohumil",
        target_agent_id="karel",
        topic_id="topic-1",
        payload={"task": "review"},
    )
    outbox = store.create_outbox_delivery(
        idempotency_key=response_idempotency_key(
            discord_delivery["delivery_key"], "bohumil"
        ),
        target_agent_id="bohumil",
        topic_id="topic-1",
        channel_id="777777777777777777",
        delivery_kind="response",
        source_inbound_delivery_key=discord_delivery["delivery_key"],
        payload={"content": "odpověď"},
    )
    store.add_outbox_part(
        outbox_delivery_id=outbox["outbox_delivery_id"],
        part_index=0,
        status="pending",
    )
    store.record_projection_message(
        agent_event_id=event["agent_event_id"],
        discord_message_id="projection-1",
        guild_id="333333333333333333",
        channel_id="777777777777777777",
        topic_id="topic-1",
        target_agent_id="karel",
        author_id="bot-karel",
        payload={"summary": "projected"},
    )
    store.upsert_approval(
        approval_id="approval-1",
        source_inbound_delivery_key=discord_delivery["delivery_key"],
        target_agent_id="bohumil",
        topic_id="topic-1",
        payload={"action": "send"},
    )
    store.create_reconciliation_run(
        reconciliation_run_id="reconcile-1",
        source_agent_event_id=event["agent_event_id"],
        outbox_delivery_id=outbox["outbox_delivery_id"],
    )
    store.close()

    reopened = DiscordProtocolV2Store(db_path)
    try:
        assert reopened.get_identity("bohumil")["token_secret_ref"].startswith("secret://")
        assert reopened.get_topic("topic-1")["title"] == "incident triage"
        assert reopened.get_inbound_delivery(discord_delivery["delivery_key"])["status"] == "pending"
        assert reopened.get_inbound_delivery(internal_delivery["delivery_key"])["source_type"] == INTERNAL_SOURCE_TYPE
        assert reopened.get_agent_event(event["agent_event_id"])["event_type"] == "handoff.requested"
        assert reopened.get_outbox_delivery(outbox["outbox_delivery_id"])["status"] == "pending"
        assert reopened.get_message_map("projection-1")["direction"] == "projection"
        assert reopened.count_rows("outbox_parts") == 1
        assert reopened.count_rows("approvals") == 1
        assert reopened.count_rows("handoffs") == 1
        assert reopened.count_rows("reconciliation_runs") == 1
    finally:
        reopened.close()


def test_replaying_same_discord_message_same_target_returns_one_delivery(tmp_path):
    with _store(tmp_path) as store:
        _seed_identity_and_topic(store)
        kwargs = dict(
            discord_message_id="dm-idempotent",
            guild_id="333333333333333333",
            channel_id="777777777777777777",
            topic_id="topic-1",
            author_id="human-1",
            author_kind="human",
            target_agent_ids=["bohumil"],
            route_reason="mention",
            payload={"content": "ping"},
        )

        first = store.create_discord_inbound_deliveries(**kwargs)
        second = store.create_discord_inbound_deliveries(**kwargs)

        assert len(first) == 1
        assert second == first
        assert first[0]["delivery_key"] == inbound_delivery_key(
            DISCORD_SOURCE_TYPE, "dm-idempotent", "bohumil"
        )
        assert (
            store.count_rows(
                "inbound_deliveries",
                "source_type = ? AND source_id = ? AND target_agent_id = ?",
                (DISCORD_SOURCE_TYPE, "dm-idempotent", "bohumil"),
            )
            == 1
        )
        decisions = store.route_decisions_for(DISCORD_SOURCE_TYPE, "dm-idempotent")
        assert len(decisions) == 1
        assert decisions[0]["decision"] == "delivered"
        assert json.loads(decisions[0]["target_agent_ids_json"]) == ["bohumil"]


@pytest.mark.parametrize("author_kind", ["registered_bot", "external_bot", "webhook", "system"])
def test_non_human_discord_authors_create_zero_deliveries_and_diagnostic_decision(
    tmp_path, author_kind
):
    with _store(tmp_path) as store:
        _seed_identity_and_topic(store)

        deliveries = store.create_discord_inbound_deliveries(
            discord_message_id=f"dm-{author_kind}",
            guild_id="333333333333333333",
            channel_id="777777777777777777",
            topic_id="topic-1",
            author_id=f"author-{author_kind}",
            author_kind=author_kind,
            target_agent_ids=["bohumil"],
            route_reason="mention",
            payload={"content": "ignored mention"},
        )

        assert deliveries == []
        assert (
            store.count_rows(
                "inbound_deliveries",
                "source_type = ? AND source_id = ?",
                (DISCORD_SOURCE_TYPE, f"dm-{author_kind}"),
            )
            == 0
        )
        message = store.get_message_map(f"dm-{author_kind}")
        assert message["direction"] == "inbound"
        assert message["author_kind"] == author_kind
        decisions = store.route_decisions_for(DISCORD_SOURCE_TYPE, f"dm-{author_kind}")
        assert len(decisions) == 1
        assert decisions[0]["decision"] == "zero_delivery"
        assert decisions[0]["reason"] == f"non_human_author:{author_kind}"


def test_direct_discord_delivery_insert_rejects_non_human_author_kind(tmp_path):
    with _store(tmp_path) as store:
        with pytest.raises(ValueError, match="author_kind=human"):
            store._create_inbound_delivery(  # deliberate contract-level guard check
                source_type=DISCORD_SOURCE_TYPE,
                source_id="dm-bot",
                discord_message_id="dm-bot",
                agent_event_id=None,
                target_agent_id="bohumil",
                topic_id="topic-1",
                route_reason="mention",
                author_kind="registered_bot",
                payload={},
            )


def test_db_check_rejects_direct_discord_delivery_insert_with_non_human_author(tmp_path):
    with _store(tmp_path) as store:
        with pytest.raises(sqlite3.IntegrityError):
            store.conn.execute(
                """
                INSERT INTO inbound_deliveries (
                    delivery_key, source_type, source_id, discord_message_id,
                    target_agent_id, topic_id, route_reason, author_kind,
                    payload_json, status, created_at, updated_at
                ) VALUES (
                    'discord_message:dm-bot-sql:bohumil', 'discord_message', 'dm-bot-sql',
                    'dm-bot-sql', 'bohumil', 'topic-1', 'mention', 'registered_bot',
                    '{}', 'pending', 'now', 'now'
                )
                """
            )


@pytest.mark.parametrize(
    "event_type",
    ["handoff.requested", "consult.requested", "review.requested"],
)
def test_internal_requested_event_helpers_store_exact_event_type(tmp_path, event_type):
    with _store(tmp_path) as store:
        _seed_identity_and_topic(store)

        event, delivery = store.create_internal_handoff(
            event_type=event_type,
            agent_event_id=f"{event_type}:event-exact",
            source_agent_id="bohumil",
            target_agent_id="karel",
            topic_id="topic-1",
            payload={"kind": event_type},
        )

        assert event["event_type"] == event_type
        assert store.get_agent_event(event["agent_event_id"])["event_type"] == event_type
        assert delivery["source_type"] == INTERNAL_SOURCE_TYPE
        assert delivery["route_reason"] == event_type


def test_replaying_same_internal_handoff_event_same_target_returns_one_delivery(tmp_path):
    with _store(tmp_path) as store:
        _seed_identity_and_topic(store)

        first_event, first_delivery = store.create_internal_handoff(
            event_type="handoff.requested",
            agent_event_id="handoff-event-idempotent",
            source_agent_id="bohumil",
            target_agent_id="karel",
            topic_id="topic-1",
            payload={"task": "look"},
        )
        second_event, second_delivery = store.create_internal_handoff(
            event_type="handoff.requested",
            agent_event_id="handoff-event-idempotent",
            source_agent_id="bohumil",
            target_agent_id="karel",
            topic_id="topic-1",
            payload={"task": "look"},
        )

        assert second_event == first_event
        assert first_event["event_type"] == "handoff.requested"
        assert second_delivery == first_delivery
        assert first_delivery["delivery_key"] == inbound_delivery_key(
            INTERNAL_SOURCE_TYPE, first_event["agent_event_id"], "karel"
        )
        assert (
            store.count_rows(
                "inbound_deliveries",
                "source_type = ? AND source_id = ? AND target_agent_id = ?",
                (INTERNAL_SOURCE_TYPE, first_event["agent_event_id"], "karel"),
            )
            == 1
        )
        assert (
            store.count_rows("agent_events", "agent_event_id = ?", (first_event["agent_event_id"],))
            == 1
        )
        assert (
            store.count_rows("handoffs", "agent_event_id = ?", (first_event["agent_event_id"],))
            == 1
        )


def test_projection_reingest_updates_message_map_and_route_without_extra_delivery(tmp_path):
    with _store(tmp_path) as store:
        _seed_identity_and_topic(store)
        event, delivery = store.create_internal_handoff(
            event_type="review.requested",
            agent_event_id="review-event-1",
            source_agent_id="bohumil",
            target_agent_id="karel",
            topic_id="topic-1",
            payload={"artifact": "draft"},
        )
        assert event["event_type"] == "review.requested"
        assert store.count_rows("inbound_deliveries") == 1

        first = store.record_projection_message(
            agent_event_id=event["agent_event_id"],
            discord_message_id="projected-discord-message",
            guild_id="333333333333333333",
            channel_id="777777777777777777",
            topic_id="topic-1",
            target_agent_id="karel",
            author_id="bot-karel",
            payload={"version": 1},
        )
        second = store.record_projection_message(
            agent_event_id=event["agent_event_id"],
            discord_message_id="projected-discord-message",
            guild_id="333333333333333333",
            channel_id="777777777777777777",
            topic_id="topic-1",
            target_agent_id="karel",
            author_id="bot-karel",
            payload={"version": 2},
        )

        assert first["discord_message_id"] == second["discord_message_id"]
        assert store.count_rows("message_map", "discord_message_id = ?", ("projected-discord-message",)) == 1
        assert json.loads(store.get_message_map("projected-discord-message")["payload_json"]) == {"version": 2}
        assert store.list_inbound_deliveries() == [delivery]
        decisions = store.route_decisions_for(INTERNAL_SOURCE_TYPE, event["agent_event_id"])
        assert {decision["decision"] for decision in decisions} == {"delivered", "projection"}
        projection = [d for d in decisions if d["decision"] == "projection"][0]
        assert projection["reason"] == "discord_projection_no_delivery"


def test_inbound_lease_and_crash_safe_state_transitions(tmp_path):
    with _store(tmp_path) as store:
        _seed_identity_and_topic(store)
        delivery = store.create_discord_inbound_deliveries(
            discord_message_id="dm-lease",
            guild_id="333333333333333333",
            channel_id="777777777777777777",
            topic_id="topic-1",
            author_id="human-1",
            author_kind="human",
            target_agent_ids=["bohumil"],
            route_reason="mention",
        )[0]

        with pytest.raises(ValueError, match="invalid inbound transition"):
            store.complete_inbound(delivery["delivery_key"])

        leased = store.lease_next_inbound(lease_owner="worker-a", lease_seconds=30)
        assert leased["delivery_key"] == delivery["delivery_key"]
        assert leased["status"] == "leased"
        assert leased["lease_owner"] == "worker-a"
        assert leased["attempts"] == 1

        retryable = store.retry_inbound(delivery["delivery_key"])
        assert retryable["status"] == "retryable"
        leased_again = store.lease_next_inbound(lease_owner="worker-b", lease_seconds=30)
        assert leased_again["attempts"] == 2
        completed = store.complete_inbound(delivery["delivery_key"])
        assert completed["status"] == "completed"
        assert completed["lease_owner"] is None


def test_inbound_lease_stale_state_version_does_not_return_other_owner_lease(tmp_path, monkeypatch):
    with _store(tmp_path) as store:
        _seed_identity_and_topic(store)
        delivery = store.create_discord_inbound_deliveries(
            discord_message_id="dm-inbound-race",
            guild_id="333333333333333333",
            channel_id="777777777777777777",
            topic_id="topic-1",
            author_id="human-1",
            author_kind="human",
            target_agent_ids=["bohumil"],
            route_reason="mention",
        )[0]
        original_cas = store._cas_lease_inbound
        raced = False

        def lose_first_cas(current, lease_owner, lease_until, now):
            nonlocal raced
            if not raced:
                raced = True
                store.conn.execute(
                    """
                    UPDATE inbound_deliveries
                    SET status = 'leased', lease_owner = 'worker-other',
                        lease_until = '9999-01-01T00:00:00+00:00',
                        state_version = state_version + 1
                    WHERE delivery_key = ?
                    """,
                    (current["delivery_key"],),
                )
            return original_cas(current, lease_owner, lease_until, now)

        monkeypatch.setattr(store, "_cas_lease_inbound", lose_first_cas)

        assert store.lease_next_inbound(lease_owner="worker-us", lease_seconds=30) is None
        stored = store.get_inbound_delivery(delivery["delivery_key"])
        assert stored["lease_owner"] == "worker-other"
        assert stored["attempts"] == 0


def test_outbox_idempotency_helpers_lease_and_transitions(tmp_path):
    with _store(tmp_path) as store:
        _seed_identity_and_topic(store)
        delivery = store.create_discord_inbound_deliveries(
            discord_message_id="dm-outbox",
            guild_id="333333333333333333",
            channel_id="777777777777777777",
            topic_id="topic-1",
            author_id="human-1",
            author_kind="human",
            target_agent_ids=["bohumil"],
            route_reason="mention",
        )[0]
        key = response_idempotency_key(delivery["delivery_key"], "bohumil")
        first = store.create_outbox_delivery(
            idempotency_key=key,
            target_agent_id="bohumil",
            topic_id="topic-1",
            channel_id="777777777777777777",
            delivery_kind="response",
            source_inbound_delivery_key=delivery["delivery_key"],
            payload={"content": "hello"},
        )
        second = store.create_outbox_delivery(
            idempotency_key=key,
            target_agent_id="bohumil",
            topic_id="topic-1",
            channel_id="777777777777777777",
            delivery_kind="response",
            source_inbound_delivery_key=delivery["delivery_key"],
            payload={"content": "hello"},
        )
        assert second == first
        assert store.count_rows("outbox_deliveries") == 1
        assert projection_idempotency_key("event-1", "karel") == "projection:event-1:karel"

        with pytest.raises(ValueError, match="invalid outbox transition"):
            store.mark_outbox_acked(first["outbox_delivery_id"])

        leased = store.lease_next_outbox(lease_owner="sender-a", lease_seconds=30)
        assert leased["status"] == "leased"
        assert leased["attempts"] == 1
        sending = store.mark_outbox_sending(first["outbox_delivery_id"])
        assert sending["status"] == "sending"
        sent = store.mark_outbox_sent(first["outbox_delivery_id"])
        assert sent["status"] == "sent"
        acked = store.mark_outbox_acked(first["outbox_delivery_id"])
        assert acked["status"] == "acked"
        reconciled = store.mark_outbox_reconciled(first["outbox_delivery_id"])
        assert reconciled["status"] == "reconciled"


def test_outbox_lease_stale_state_version_does_not_return_other_owner_lease(tmp_path, monkeypatch):
    with _store(tmp_path) as store:
        _seed_identity_and_topic(store)
        outbox = store.create_outbox_delivery(
            idempotency_key="response:race",
            target_agent_id="bohumil",
            topic_id="topic-1",
            channel_id="777777777777777777",
            delivery_kind="response",
            payload={"content": "hello"},
        )
        original_cas = store._cas_lease_outbox
        raced = False

        def lose_first_cas(current, lease_owner, lease_until, now):
            nonlocal raced
            if not raced:
                raced = True
                store.conn.execute(
                    """
                    UPDATE outbox_deliveries
                    SET status = 'leased', lease_owner = 'sender-other',
                        lease_until = '9999-01-01T00:00:00+00:00',
                        state_version = state_version + 1
                    WHERE outbox_delivery_id = ?
                    """,
                    (current["outbox_delivery_id"],),
                )
            return original_cas(current, lease_owner, lease_until, now)

        monkeypatch.setattr(store, "_cas_lease_outbox", lose_first_cas)

        assert store.lease_next_outbox(lease_owner="sender-us", lease_seconds=30) is None
        stored = store.get_outbox_delivery(outbox["outbox_delivery_id"])
        assert stored["lease_owner"] == "sender-other"
        assert stored["attempts"] == 0


def test_schema_check_constraints_reject_invalid_enums(tmp_path):
    with _store(tmp_path) as store:
        with pytest.raises(sqlite3.IntegrityError):
            store.conn.execute(
                """
                INSERT INTO message_map (
                    discord_message_id, guild_id, channel_id, direction,
                    author_id, author_kind, mentions_json, created_at, updated_at, payload_json
                ) VALUES ('bad-direction', 'g', 'c', 'invalid', 'a', 'human', '[]', 'now', 'now', '{}')
                """
            )
        with pytest.raises(sqlite3.IntegrityError):
            store.conn.execute(
                """
                INSERT INTO inbound_deliveries (
                    delivery_key, source_type, source_id, discord_message_id,
                    target_agent_id, topic_id, route_reason, author_kind,
                    payload_json, status, created_at, updated_at
                ) VALUES ('bad', 'discord_message', 'dm', 'dm', 'a', 't', 'r', 'human', '{}', 'bogus', 'now', 'now')
                """
            )


def test_create_agent_event_conflict_is_immutable_and_fails_closed(tmp_path):
    with _store(tmp_path) as store:
        _seed_identity_and_topic(store)
        first = store.create_agent_event(
            agent_event_id="legacy-event-conflict",
            event_type="handoff.requested",
            source_agent_id="bohumil",
            target_agent_id="karel",
            topic_id="topic-1",
            payload={"task": "same"},
            status="requested",
        )
        replay = store.create_agent_event(
            agent_event_id="legacy-event-conflict",
            event_type="handoff.requested",
            source_agent_id="bohumil",
            target_agent_id="karel",
            topic_id="topic-1",
            payload={"task": "same"},
            status="pending",
        )

        assert replay == first
        with pytest.raises(ValueError, match="agent_event_id conflict"):
            store.create_agent_event(
                agent_event_id="legacy-event-conflict",
                event_type="handoff.requested",
                source_agent_id="bohumil",
                target_agent_id="bohumil",
                topic_id="topic-1",
                payload={"task": "same"},
            )

        assert store.count_rows("agent_events") == 1
        assert store.get_agent_event("legacy-event-conflict")["target_agent_id"] == "karel"


def test_store_compat_internal_handoff_creates_projection_and_redacts_payloads(tmp_path):
    with _store(tmp_path) as store:
        _seed_identity_and_topic(store)
        event, delivery = store.create_internal_handoff(
            event_type="handoff.requested",
            agent_event_id="compat-handoff-redact",
            source_agent_id="bohumil",
            target_agent_id="karel",
            topic_id="topic-1",
            payload={
                "task": "review",
                "token": "abc.def.ghijklmnopqrstuvwxyz0123456789",
                "api_key": "sk-secret-api-key",
            },
        )

        event_id = event["agent_event_id"]
        assert event_id.startswith("evt_")
        assert event_id != "compat-handoff-redact"
        assert store.get_agent_event("compat-handoff-redact") is None
        assert event["status"] == "requested"
        assert delivery["agent_event_id"] == event_id
        assert (
            store.count_rows("outbox_deliveries", "source_agent_event_id = ?", (event_id,))
            == 1
        )
        event_payload = json.loads(event["payload_json"])
        delivery_payload = json.loads(delivery["payload_json"])
        assert event_payload["token"] == "<redacted>"
        assert event_payload["api_key"] == "<redacted>"
        assert delivery_payload["token"] == "<redacted>"
        assert delivery_payload["api_key"] == "<redacted>"
        handoff = store.get_handoff_by_agent_event(event_id)
        assert handoff is not None
        assert handoff["status"] == "requested"
        handoff_payload = json.loads(handoff["payload_json"])
        assert handoff_payload["token"] == "<redacted>"
        assert handoff_payload["api_key"] == "<redacted>"
        outbox = store.get_outbox_delivery_by_key(projection_idempotency_key(event_id, "karel"))
        assert outbox is not None
        assert "ghijklmnopqrstuvwxyz" not in outbox["payload_json"]
        assert "sk-secret-api-key" not in outbox["payload_json"]


def test_store_redacts_api_key_in_payload_and_projection_json(tmp_path):
    assert redact_sensitive_data({"api_key": "plaintext"})["api_key"] == "<redacted>"
    assert redact_sensitive_data({"service_key": "plaintext"})["service_key"] == "<redacted>"

    with _store(tmp_path) as store:
        _seed_identity_and_topic(store)
        delivery = store.create_outbox_delivery(
            idempotency_key="projection-api-key-redaction",
            target_agent_id="karel",
            topic_id="topic-1",
            channel_id="777777777777777777",
            delivery_kind="projection",
            payload={"content": "safe", "api_key": "plaintext", "nested": {"service_key": "nested-secret"}},
        )

        payload = json.loads(delivery["payload_json"])
        assert payload["api_key"] == "<redacted>"
        assert payload["nested"]["service_key"] == "<redacted>"
        assert "plaintext" not in delivery["payload_json"]
        assert "nested-secret" not in delivery["payload_json"]


def test_record_projection_message_preserves_agent_correlation_and_redacts(tmp_path):
    with _store(tmp_path) as store:
        _seed_identity_and_topic(store)
        event, _delivery = store.create_internal_handoff(
            event_type="review.requested",
            agent_event_id="projection-correlation-event",
            source_agent_id="bohumil",
            target_agent_id="karel",
            topic_id="topic-1",
            payload={"artifact": "draft"},
        )
        outbox = store.get_outbox_delivery_by_key(
            projection_idempotency_key(event["agent_event_id"], "karel")
        )
        message = store.record_projection_message(
            agent_event_id=event["agent_event_id"],
            discord_message_id="projection-correlation-message",
            guild_id="333333333333333333",
            channel_id="777777777777777777",
            topic_id="topic-1",
            target_agent_id="karel",
            author_id="bot-karel",
            outbox_delivery_id=outbox["outbox_delivery_id"],
            source_client_agent_id="karel",
            payload={"token": "abc.def.ghijklmnopqrstuvwxyz0123456789"},
        )

        assert message["agent_id"] == "karel"
        assert message["source_client_agent_id"] == "karel"
        assert message["outbox_delivery_id"] == outbox["outbox_delivery_id"]
        assert message["agent_event_id"] == event["agent_event_id"]
        assert json.loads(message["payload_json"])["token"] == "<redacted>"


def test_old_inbound_schema_migrates_internal_event_source_fields(tmp_path):
    db_path = tmp_path / "old-schema.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE inbound_deliveries (
            delivery_key TEXT PRIMARY KEY,
            source_type TEXT NOT NULL CHECK (source_type IN ('discord_message')),
            discord_message_id TEXT NOT NULL,
            target_agent_id TEXT NOT NULL,
            topic_id TEXT NOT NULL,
            route_reason TEXT NOT NULL,
            author_kind TEXT NOT NULL CHECK (author_kind IN ('human', 'registered_bot', 'external_bot', 'webhook', 'system')),
            payload_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'leased', 'completed', 'failed', 'retryable')),
            lease_owner TEXT,
            lease_until TEXT,
            attempts INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            state_version INTEGER NOT NULL DEFAULT 1,
            UNIQUE (source_type, discord_message_id, target_agent_id),
            CHECK (source_type != 'discord_message' OR author_kind = 'human')
        );
        INSERT INTO inbound_deliveries (
            delivery_key, source_type, discord_message_id, target_agent_id, topic_id,
            route_reason, author_kind, payload_json, status, created_at, updated_at
        ) VALUES (
            'discord_message:old-dm:bohumil', 'discord_message', 'old-dm', 'bohumil',
            'topic-1', 'mention', 'human', '{}', 'pending', 'now', 'now'
        );
        CREATE TABLE route_decisions (
            decision_id TEXT PRIMARY KEY,
            source_type TEXT NOT NULL CHECK (source_type IN ('discord_message')),
            source_id TEXT NOT NULL,
            topic_id TEXT NOT NULL,
            author_kind TEXT NOT NULL CHECK (author_kind IN ('human', 'registered_bot', 'external_bot', 'webhook', 'system')),
            decision TEXT NOT NULL,
            target_agent_ids_json TEXT NOT NULL DEFAULT '[]',
            reason TEXT NOT NULL,
            created_at TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}'
        );
        """
    )
    conn.close()

    with DiscordProtocolV2Store(db_path) as store:
        _seed_identity_and_topic(store)
        old = store.get_inbound_delivery("discord_message:old-dm:bohumil")
        assert old["source_id"] == "old-dm"
        assert old["discord_message_id"] == "old-dm"
        event, delivery = store.create_internal_handoff(
            event_type="review.requested",
            agent_event_id="migrated-internal-event",
            source_agent_id="bohumil",
            target_agent_id="karel",
            topic_id="topic-1",
            payload={"task": "migrated"},
        )
        assert delivery["source_type"] == INTERNAL_SOURCE_TYPE
        assert delivery["agent_event_id"] == event["agent_event_id"]
        assert store.route_decisions_for(INTERNAL_SOURCE_TYPE, event["agent_event_id"])
