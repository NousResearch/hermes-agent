"""Durable Discord protocol v2 approval tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import gateway.run as gateway_run
from gateway.config import DiscordPrimaryUIConfig, GatewayConfig, Platform
from gateway.discord_approvals import DiscordApprovalStore
from gateway.discord_protocol_v2_approvals import (
    agent_has_capability_or_scope,
    create_component_custom_id,
    create_pending_approval,
    decide_approval,
    new_approval_id,
    parse_component_custom_id,
)
from gateway.discord_protocol_v2_store import DiscordProtocolV2Store
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


def _store(path: Path) -> DiscordProtocolV2Store:
    store = DiscordProtocolV2Store(path)
    store.upsert_identity(
        agent_id="bohumil",
        hermes_profile="default",
        discord_application_id="111111111111111111",
        discord_bot_user_id="222222222222222222",
        token_secret_ref="secret://hermes/discord/bohumil-token",
        capabilities=["approve_tools", "reply"],
        scopes={"guild_ids": ["guild-1"], "topics": ["topic-1"]},
        enabled=True,
    )
    store.upsert_topic(
        topic_id="topic-1",
        guild_id="guild-1",
        channel_id="chan-1",
        title="approval topic",
    )
    return store


def _runner(tmp_path):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = GatewayConfig(
        sessions_dir=tmp_path,
        primary_ui="discord",
        discord_primary_ui=DiscordPrimaryUIConfig(
            enabled=True,
            guild_id="guild-1",
            owner_user_ids=["100"],
            approvals_channel_id="chan-1",
        ),
    )
    runner.adapters = {}
    runner._pending_approvals = {}
    return runner


def _event(text: str, *, chat_id="chan-1", parent_chat_id=None):
    source = SessionSource(
        platform=Platform.DISCORD,
        chat_id=chat_id,
        chat_type="thread",
        user_id="100",
        guild_id="guild-1",
        thread_id="thread-1",
        parent_chat_id=parent_chat_id,
    )
    raw_user = SimpleNamespace(id=100, roles=[])
    return MessageEvent(text=text, source=source, raw_message=SimpleNamespace(user=raw_user))


def test_create_store_pending_approval_with_stable_opaque_fields(tmp_path):
    with _store(tmp_path / "discord-v2.sqlite3") as store:
        approval_id = new_approval_id()
        row = create_pending_approval(
            store,
            approval_id=approval_id,
            topic_id="topic-1",
            agent_id="bohumil",
            requesting_event_id="event-123",
            payload={"summary": "safe", "token_secret": "plain-secret"},
        )

        assert row["approval_id"] == approval_id
        assert row["topic_id"] == "topic-1"
        assert row["agent_id"] == "bohumil"
        assert row["target_agent_id"] == "bohumil"
        assert row["requesting_event_id"] == "event-123"
        assert row["status"] == "pending"
        assert json.loads(row["payload_json"])["token_secret"] == "<redacted>"


def test_payload_and_audit_redact_secret_values_under_innocent_keys(tmp_path):
    with _store(tmp_path / "discord-v2-redact.sqlite3") as store:
        bearer = "bearer-token-abcdefghijklmnopqrstuvwxyz"
        openai_key = "sk-abcdefghijklmnopqrstuvwxyz123456"
        approval = create_pending_approval(
            store,
            topic_id="topic-1",
            agent_id="bohumil",
            requesting_event_id="event-redact",
            payload={
                "command": f"curl -H 'Authorization: Bearer {bearer}' https://example.test",
                "metadata": {"note": f"using {openai_key}"},
            },
        )
        decide_approval(
            store,
            approval_id=approval["approval_id"],
            decision="approve",
            actor_user_id="100",
            audit_payload={"comment": f"Authorization: Bearer {bearer}"},
        )

        stored = store.get_approval(approval["approval_id"])
        assert stored is not None
        raw_payload = stored["payload_json"]
        raw_audit = store.list_approval_audit_events(approval["approval_id"])[0]["payload_json"]

        assert bearer not in raw_payload
        assert bearer not in raw_audit
        assert openai_key not in raw_payload


def test_component_custom_ids_contain_only_decision_and_opaque_id():
    approval_id = "apv_0123456789abcdef"
    custom_id = create_component_custom_id("approve", approval_id)

    assert custom_id == f"hermes_v2_approval:approve:{approval_id}"
    assert parse_component_custom_id(custom_id) == ("approve", approval_id)
    assert "topic-1" not in custom_id
    assert "bohumil" not in custom_id
    assert "event-123" not in custom_id
    assert "secret" not in custom_id.lower()
    assert len(custom_id) <= 100


def test_pending_approval_survives_store_reopen(tmp_path):
    db_path = tmp_path / "discord-v2.sqlite3"
    store = _store(db_path)
    approval_id = new_approval_id()
    create_pending_approval(
        store,
        approval_id=approval_id,
        topic_id="topic-1",
        agent_id="bohumil",
        requesting_event_id="event-restart",
    )
    store.close()

    reopened = DiscordProtocolV2Store(db_path)
    try:
        row = reopened.get_approval(approval_id)
        assert row is not None
        assert row["status"] == "pending"
        assert row["topic_id"] == "topic-1"
        assert row["agent_id"] == "bohumil"
        assert row["requesting_event_id"] == "event-restart"
    finally:
        reopened.close()


@pytest.mark.parametrize("decision,final_status", [("approve", "approved"), ("deny", "denied")])
def test_duplicate_decision_is_idempotent_and_audit_is_not_duplicated(tmp_path, decision, final_status):
    with _store(tmp_path / f"discord-v2-{decision}.sqlite3") as store:
        approval = create_pending_approval(
            store,
            topic_id="topic-1",
            agent_id="bohumil",
            requesting_event_id=f"event-{decision}",
        )

        first = decide_approval(store, approval_id=approval["approval_id"], decision=decision, actor_user_id="100")
        second = decide_approval(store, approval_id=approval["approval_id"], decision=decision, actor_user_id="100")

        assert first.ok is True
        assert first.status == final_status
        assert second.ok is True
        assert second.duplicate is True
        assert second.status == final_status
        stored = store.get_approval(approval["approval_id"])
        assert stored is not None
        assert stored["status"] == final_status
        events = store.list_approval_audit_events(approval["approval_id"])
        assert len(events) == 1
        assert events[0]["status"] == final_status


def test_recreating_pending_approval_does_not_reopen_terminal_status_or_duplicate_audit(tmp_path):
    with _store(tmp_path / "discord-v2-replay.sqlite3") as store:
        approval_id = new_approval_id()
        create_pending_approval(
            store,
            approval_id=approval_id,
            topic_id="topic-1",
            agent_id="bohumil",
            requesting_event_id="event-original",
            payload={"summary": "first"},
        )
        first = decide_approval(
            store,
            approval_id=approval_id,
            decision="approve",
            actor_user_id="100",
        )

        replay = create_pending_approval(
            store,
            approval_id=approval_id,
            topic_id="topic-1",
            agent_id="bohumil",
            requesting_event_id="event-replay",
            payload={"summary": "replay"},
        )

        assert first.ok is True
        assert replay["status"] == "approved"
        assert replay["requesting_event_id"] == "event-original"
        assert json.loads(replay["payload_json"])["summary"] == "first"
        events = store.list_approval_audit_events(approval_id)
        assert len(events) == 1
        assert events[0]["status"] == "approved"


def test_identity_capability_scope_policy_uses_persisted_registry(tmp_path):
    with _store(tmp_path / "discord-v2.sqlite3") as store:
        assert agent_has_capability_or_scope(
            store,
            agent_id="bohumil",
            capability="approve_tools",
            scope_key="guild_ids",
            scope_value="guild-1",
        )
        assert not agent_has_capability_or_scope(
            store,
            agent_id="bohumil",
            capability="admin_everything",
            scope_key="guild_ids",
            scope_value="guild-1",
        )


def test_decide_approval_unknown_valid_v2_id_does_not_echo_raw_id(tmp_path):
    approval_id = "apv_secret-like-token-0123456789abcdef"
    with _store(tmp_path / "discord-v2-unknown-direct.sqlite3") as store:
        result = decide_approval(store, approval_id=approval_id, decision="approve", actor_user_id="100")

    assert result.ok is False
    assert result.status == "unknown"
    assert "unknown approval action" in result.message.lower()
    assert approval_id not in result.message


@pytest.mark.asyncio
async def test_gateway_runner_dispatches_v2_component_to_durable_sqlite(tmp_path):
    runner = _runner(tmp_path)
    store = _store(tmp_path / "discord-v2-runtime.sqlite3")
    runner.discord_protocol_v2_store = store
    try:
        approval = create_pending_approval(
            store,
            topic_id="topic-1",
            agent_id="bohumil",
            requesting_event_id="event-runtime",
            payload={
                "policy": {
                    "required_capability": "approve_tools",
                    "scope_key": "guild_ids",
                    "scope_value": "guild-1",
                }
            },
        )
        custom_id = create_component_custom_id("approve", approval["approval_id"])

        response = await gateway_run.GatewayRunner._handle_approve_command(
            runner,
            _event(f"/approve {custom_id}"),
        )

        assert response is not None
        assert "approved" in response.lower()
        stored = store.get_approval(approval["approval_id"])
        assert stored is not None
        assert stored["status"] == "approved"
        audit_events = store.list_approval_audit_events(approval["approval_id"])
        assert len(audit_events) == 1
        assert audit_events[0]["event_type"] == "discord_v2_approval_approved"
    finally:
        store.close()


@pytest.mark.asyncio
async def test_gateway_runner_missing_v2_store_redacts_unknown_raw_id_response_and_audit(tmp_path):
    runner = _runner(tmp_path)
    approval_id = "apv_secret-like-token-0123456789abcdef"

    response = await gateway_run.GatewayRunner._handle_approve_command(
        runner,
        _event(f"/approve {approval_id}"),
    )

    assert response is not None
    assert "unknown approval action" in response.lower()
    assert approval_id not in response

    audit_text = (tmp_path / "discord_audit.jsonl").read_text(encoding="utf-8")
    assert approval_id not in audit_text
    audit_record = json.loads(audit_text.splitlines()[-1])
    assert audit_record["details"] == {
        "action_id": "<redacted_structured_approval_id>",
        "status": "unknown",
        "v2": True,
    }


@pytest.mark.asyncio
async def test_gateway_runner_unknown_raw_v2_id_redacts_response_and_audit(tmp_path):
    runner = _runner(tmp_path)
    store = _store(tmp_path / "discord-v2-unknown-raw.sqlite3")
    runner.discord_protocol_v2_store = store
    approval_id = "apv_secret-like-token-0123456789abcdef"
    try:
        response = await gateway_run.GatewayRunner._handle_approve_command(
            runner,
            _event(f"/approve {approval_id}"),
        )

        assert response is not None
        assert "unknown approval action" in response.lower()
        assert approval_id not in response

        audit_text = (tmp_path / "discord_audit.jsonl").read_text(encoding="utf-8")
        assert approval_id not in audit_text
        audit_record = json.loads(audit_text.splitlines()[-1])
        assert audit_record["details"] == {
            "action_id": "<redacted_structured_approval_id>",
            "ok": False,
            "status": "unknown",
            "v2": True,
        }
    finally:
        store.close()


@pytest.mark.asyncio
async def test_gateway_runner_unknown_v2_component_redacts_response_and_audit(tmp_path):
    runner = _runner(tmp_path)
    store = _store(tmp_path / "discord-v2-unknown-component.sqlite3")
    runner.discord_protocol_v2_store = store
    approval_id = "apv_secret-like-token-0123456789abcdef"
    custom_id = f"hermes_v2_approval:approve:{approval_id}"
    try:
        response = await gateway_run.GatewayRunner._handle_approve_command(
            runner,
            _event(f"/approve {custom_id}"),
        )

        assert response is not None
        assert "unknown approval action" in response.lower()
        assert approval_id not in response
        assert custom_id not in response

        audit_text = (tmp_path / "discord_audit.jsonl").read_text(encoding="utf-8")
        assert approval_id not in audit_text
        assert custom_id not in audit_text
        audit_record = json.loads(audit_text.splitlines()[-1])
        assert audit_record["details"] == {
            "action_id": "<redacted_structured_approval_id>",
            "ok": False,
            "status": "unknown",
            "v2": True,
        }
    finally:
        store.close()


@pytest.mark.asyncio
async def test_gateway_runner_v2_policy_requires_agent_capability_scope_even_for_owner(tmp_path):
    runner = _runner(tmp_path)
    store = _store(tmp_path / "discord-v2-policy.sqlite3")
    runner.discord_protocol_v2_store = store
    try:
        store.upsert_identity(
            agent_id="limited",
            hermes_profile="default",
            discord_application_id="333333333333333333",
            discord_bot_user_id="444444444444444444",
            token_secret_ref="secret://hermes/discord/limited-token",
            capabilities=["reply"],
            scopes={"guild_ids": ["guild-1"]},
            enabled=True,
        )
        approval = create_pending_approval(
            store,
            topic_id="topic-1",
            agent_id="limited",
            requesting_event_id="event-policy",
            payload={
                "policy": {
                    "required_capability": "approve_tools",
                    "scope_key": "guild_ids",
                    "scope_value": "guild-1",
                }
            },
        )

        denied = await gateway_run.GatewayRunner._handle_approve_command(
            runner,
            _event(f"/approve {approval['approval_id']}"),
        )
        assert denied is not None
        assert "not authorized" in denied.lower()
        denied_stored = store.get_approval(approval["approval_id"])
        assert denied_stored is not None
        assert denied_stored["status"] == "pending"
        assert store.list_approval_audit_events(approval["approval_id"]) == []

        store.upsert_identity(
            agent_id="limited",
            hermes_profile="default",
            discord_application_id="333333333333333333",
            discord_bot_user_id="444444444444444444",
            token_secret_ref="secret://hermes/discord/limited-token",
            capabilities=["approve_tools", "reply"],
            scopes={"guild_ids": ["guild-1"]},
            enabled=True,
        )
        approved = await gateway_run.GatewayRunner._handle_approve_command(
            runner,
            _event(f"/approve {approval['approval_id']}"),
        )

        assert approved is not None
        assert "approved" in approved.lower()
        approved_stored = store.get_approval(approval["approval_id"])
        assert approved_stored is not None
        assert approved_stored["status"] == "approved"
        assert len(store.list_approval_audit_events(approval["approval_id"])) == 1
    finally:
        store.close()


@pytest.mark.asyncio
async def test_raw_malformed_act_guard_does_not_fall_through_to_legacy(tmp_path):
    runner = _runner(tmp_path)
    runner.discord_approval_store = DiscordApprovalStore(tmp_path / "approvals.json")

    with patch("tools.approval.has_blocking_approval", return_value=True) as has_blocking:
        response = await gateway_run.GatewayRunner._handle_approve_command(
            runner,
            _event("/approve act_!!!"),
        )

    assert response is not None
    assert "unknown approval action" in response.lower()
    has_blocking.assert_not_called()
    assert runner.discord_approval_store.list_audit_events()[0].payload_json == {
        "action_id": "act_!!!",
        "status": "unknown",
    }


@pytest.mark.asyncio
async def test_raw_malformed_v2_guard_does_not_fall_through_to_legacy(tmp_path):
    runner = _runner(tmp_path)
    runner.discord_approval_store = DiscordApprovalStore(tmp_path / "approvals.json")

    with patch("tools.approval.has_blocking_approval", return_value=True) as has_blocking:
        response = await gateway_run.GatewayRunner._handle_approve_command(
            runner,
            _event("/approve apv_!!!"),
        )

    assert response is not None
    assert "unknown approval action" in response.lower()
    has_blocking.assert_not_called()
    assert runner.discord_approval_store.list_audit_events() == []


@pytest.mark.asyncio
async def test_malformed_v2_component_redacts_raw_token_from_response_and_audit(tmp_path):
    runner = _runner(tmp_path)
    secret_like = "secret-like-token-0123456789abcdef"
    malformed_id = f"hermes_v2_approval:approve:{secret_like}"

    with patch("tools.approval.has_blocking_approval", return_value=True) as has_blocking:
        response = await gateway_run.GatewayRunner._handle_approve_command(
            runner,
            _event(f"/approve {malformed_id}"),
        )

    assert response is not None
    assert "unknown approval action" in response.lower()
    assert secret_like not in response
    assert malformed_id not in response
    has_blocking.assert_not_called()

    audit_text = (tmp_path / "discord_audit.jsonl").read_text(encoding="utf-8")
    assert secret_like not in audit_text
    assert malformed_id not in audit_text
    audit_record = json.loads(audit_text.splitlines()[-1])
    assert audit_record["details"] == {
        "action_id": "<redacted_structured_approval_id>",
        "status": "unknown",
        "v2": True,
    }


@pytest.mark.asyncio
async def test_malformed_v2_surface_deny_redacts_raw_token_from_audit(tmp_path):
    runner = _runner(tmp_path)
    secret_like = "secret-like-token-0123456789abcdef"
    malformed_id = f"hermes_v2_approval:approve:{secret_like}"

    response = await gateway_run.GatewayRunner._handle_approve_command(
        runner,
        _event(f"/approve {malformed_id}", chat_id="other-chan"),
    )

    assert response is not None
    assert secret_like not in response
    assert malformed_id not in response
    audit_text = (tmp_path / "discord_audit.jsonl").read_text(encoding="utf-8")
    assert secret_like not in audit_text
    assert malformed_id not in audit_text
    audit_record = json.loads(audit_text.splitlines()[-1])
    assert audit_record["details"]["action_id"] == "<redacted_structured_approval_id>"
    assert audit_record["details"]["reason"] == "structured_approvals_not_enabled_for_source"
