from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest
import yaml

from communication_core.adapters import (
    AdapterCapabilities,
    FakeCommunicationAdapter,
    NormalizedGroup,
    NormalizedProfile,
    NormalizedReceipt,
    TelegramCommunicationAdapter,
)
from communication_core.errors import CapabilityUnsupportedError
from communication_core.repository import CommunicationRepository
from communication_core.service import CommunicationService
from hermes_cli import communication as communication_cli
from hermes_cli.config import DEFAULT_CONFIG


def _account(repository: CommunicationRepository, provider: str, namespace: str):
    return repository.add_account(
        provider=provider,
        account_namespace=namespace,
        label=namespace,
        owner_profile="test",
    )


def test_cli_has_no_sql_or_connection_calls_and_person_detail_is_repository_owned(tmp_path):
    tree = ast.parse(inspect.getsource(communication_cli))
    forbidden = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr in {"execute", "executemany", "executescript", "read_connection", "transaction"}
    }
    assert forbidden == set()

    repository = CommunicationRepository(tmp_path / "communication.db")
    repository.initialize()
    account = _account(repository, "fake", "one")
    person = repository.create_person("Person")
    _, endpoint = repository.upsert_identity(
        connected_account_id=account["id"],
        external_id="external",
        display_name="Person",
        person_id=person["id"],
    )
    detail = repository.person_detail(person["id"])
    assert detail["person"]["id"] == person["id"]
    assert detail["identities"][0]["endpoint_id"] == endpoint["id"]
    assert detail["identities"][0]["endpoint_account_id"] == account["id"]


def test_adapter_profile_group_and_receipt_reads_are_explicit_and_fail_closed():
    account = {"id": "acct", "provider": "fake", "enabled": 1}
    fake = FakeCommunicationAdapter(
        profiles=[NormalizedProfile("person", display_name="Person")],
        groups=[NormalizedGroup("group", title="Friends", member_external_ids=("person",))],
        receipts=[NormalizedReceipt("receipt", "message", "read")],
    )
    assert tuple(fake.sync_profiles(account))[0].external_id == "person"
    assert tuple(fake.sync_groups(account))[0].member_external_ids == ("person",)
    assert tuple(fake.sync_receipts(account))[0].status == "read"

    telegram = TelegramCommunicationAdapter(
        capabilities=AdapterCapabilities(
            contacts_read=True,
            profiles_read=True,
            conversations_read=True,
            messages_read=True,
            events_read=True,
        )
    )
    telegram_account = {"id": "tg", "provider": "telegram", "enabled": 1}
    with pytest.raises(CapabilityUnsupportedError, match="groups.read"):
        tuple(telegram.sync_groups(telegram_account))
    with pytest.raises(CapabilityUnsupportedError, match="receipts.read"):
        tuple(telegram.sync_receipts(telegram_account))


def test_timeline_filters_every_record_type_and_redacts_restricted_pii(tmp_path):
    repository = CommunicationRepository(tmp_path / "communication.db")
    repository.initialize()
    first_account = _account(repository, "fake", "one")
    second_account = _account(repository, "fake", "two")
    person = repository.create_person("Secret Person", pii_policy="restricted")
    first_identity, first_endpoint = repository.upsert_identity(
        connected_account_id=first_account["id"], external_id="secret-first",
        display_name="Secret Person", person_id=person["id"],
    )
    second_identity, second_endpoint = repository.upsert_identity(
        connected_account_id=second_account["id"], external_id="secret-second",
        display_name="Secret Person", person_id=person["id"],
    )
    first_conversation = repository.upsert_conversation(
        connected_account_id=first_account["id"], endpoint_id=first_endpoint["id"],
        external_id="secret-thread-one", kind="direct", title="Secret one",
        provenance={"source": "fixture"}, observed_at="2026-07-23T10:00:00Z",
    )
    second_conversation = repository.upsert_conversation(
        connected_account_id=second_account["id"], endpoint_id=second_endpoint["id"],
        external_id="secret-thread-two", kind="direct", title="Secret two",
        provenance={"source": "fixture"}, observed_at="2026-07-24T10:00:00Z",
    )
    repository.upsert_message(
        connected_account_id=first_account["id"], endpoint_id=first_endpoint["id"],
        conversation_id=first_conversation["id"], external_id="message-one",
        direction="incoming", body="private codeword", sent_at="2026-07-23T10:00:00Z",
        sender_identity_id=first_identity["id"], provenance={"secret": "one"},
        observed_at="2026-07-23T10:00:00Z",
    )
    repository.upsert_message(
        connected_account_id=second_account["id"], endpoint_id=second_endpoint["id"],
        conversation_id=second_conversation["id"], external_id="message-two",
        direction="incoming", body="other secret", sent_at="2026-07-24T10:00:00Z",
        sender_identity_id=second_identity["id"], provenance={"secret": "two"},
        observed_at="2026-07-24T10:00:00Z",
    )
    repository.add_contact_event(
        person_id=person["id"], connected_account_id=first_account["id"],
        endpoint_id=first_endpoint["id"], event_type="private.meeting",
        external_id="secret-event-one", happened_at="2026-07-23T11:00:00Z",
        data={"place": "private"}, provenance={"secret": "event"},
    )
    repository.add_contact_event(
        person_id=person["id"], connected_account_id=second_account["id"],
        endpoint_id=second_endpoint["id"], event_type="private.meeting",
        external_id="secret-event-two", happened_at="2026-07-24T11:00:00Z",
        data={"place": "other"}, provenance={"secret": "event"},
    )
    repository.record_transition(
        person_id=person["id"], from_endpoint_id=first_endpoint["id"],
        to_endpoint_id=second_endpoint["id"], initiator="person",
        evidence_type="person_request", evidence_ref="fixture:request",
        happened_at="2026-07-23T12:00:00Z",
    )

    timeline = repository.timeline(
        person["id"], endpoint_id=first_endpoint["id"],
        start_at="2026-07-23T00:00:00Z", end_at="2026-07-24T00:00:00Z",
    )
    assert {item["endpoint_id"] for item in timeline["messages"]} == {first_endpoint["id"]}
    assert {item["endpoint_id"] for item in timeline["events"]} == {first_endpoint["id"]}
    assert all(item["endpoint_id"] == first_endpoint["id"] for item in timeline["episodes"])
    assert len(timeline["transitions"]) == 1
    assert timeline["messages"][0]["body"] == "[redacted]"
    assert timeline["events"][0]["external_id"] == "[redacted]"
    assert json.loads(timeline["events"][0]["data_json"]) == {}

    search = repository.search_all("secret")
    assert search["people"][0]["display_name"] == "[redacted]"
    assert all(item["external_id"] == "[redacted]" for item in search["identities"])
    assert all(item["external_id"] == "[redacted]" for item in search["conversations"])
    assert search["messages"] == []
    assert all(item["external_id"] == "[redacted]" for item in search["events"])


def test_merge_unmerge_preserves_messages_events_journey_routes_and_preferences(tmp_path):
    repository = CommunicationRepository(tmp_path / "communication.db")
    repository.initialize()
    source_account = _account(repository, "fake", "source")
    target_account = _account(repository, "fake", "target")
    winner = repository.create_person("Winner")
    duplicate = repository.create_person("Duplicate")
    duplicate_identity, duplicate_source = repository.upsert_identity(
        connected_account_id=source_account["id"], external_id="duplicate-source",
        display_name="Duplicate", person_id=duplicate["id"],
    )
    _, duplicate_target = repository.upsert_identity(
        connected_account_id=target_account["id"], external_id="duplicate-target",
        display_name="Duplicate", person_id=duplicate["id"],
    )
    conversation = repository.upsert_conversation(
        connected_account_id=source_account["id"], endpoint_id=duplicate_source["id"],
        external_id="thread", kind="direct", title=None,
        provenance={"source": "fixture"}, observed_at="2026-07-23T10:00:00Z",
    )
    message, _ = repository.upsert_message(
        connected_account_id=source_account["id"], endpoint_id=duplicate_source["id"],
        conversation_id=conversation["id"], external_id="message", direction="incoming",
        body="hello", sent_at="2026-07-23T10:00:00Z",
        sender_identity_id=duplicate_identity["id"], provenance={"source": "fixture"},
        observed_at="2026-07-23T10:00:00Z",
    )
    event = repository.add_contact_event(
        person_id=duplicate["id"], connected_account_id=source_account["id"],
        endpoint_id=duplicate_source["id"], event_type="meeting", external_id="event",
        happened_at="2026-07-23T11:00:00Z", provenance={"source": "fixture"},
    )
    repository.allow_account_link(
        source_account["id"], target_account["id"], allowed=True,
        actor="test", reason="fixture",
    )
    service = CommunicationService(repository)
    route = service.apply_route(
        person_id=duplicate["id"], source_endpoint_id=duplicate_source["id"],
        target_endpoint_id=duplicate_target["id"], actor="test", reason="fixture",
    )
    transition = repository.record_transition(
        person_id=duplicate["id"], from_endpoint_id=duplicate_source["id"],
        to_endpoint_id=duplicate_target["id"], initiator="person",
        evidence_type="person_request", evidence_ref="fixture:request",
        happened_at="2026-07-23T12:00:00Z",
    )

    merge = repository.merge_people(
        winner["id"], duplicate["id"], actor="test", evidence={"manual": "verified"}
    )
    merged_timeline = repository.timeline(winner["id"])
    assert message["id"] in {item["id"] for item in merged_timeline["messages"]}
    assert event["id"] in {item["id"] for item in merged_timeline["events"]}
    assert transition["id"] in {item["id"] for item in merged_timeline["transitions"]}
    assert repository.get_route(winner["id"], duplicate_source["id"])["id"] == route["id"]

    repository.unmerge_people(merge["audit_id"], actor="test")
    restored = repository.timeline(duplicate["id"])
    assert message["id"] in {item["id"] for item in restored["messages"]}
    assert event["id"] in {item["id"] for item in restored["events"]}
    assert transition["id"] in {item["id"] for item in restored["transitions"]}
    assert repository.get_route(duplicate["id"], duplicate_source["id"])["id"] == route["id"]
    assert repository.get_route(winner["id"], duplicate_source["id"]) is None


def test_group_and_segment_previews_freeze_recipient_sets_for_drafts(tmp_path):
    repository = CommunicationRepository(tmp_path / "communication.db")
    repository.initialize()
    source_account = _account(repository, "fake", "source")
    target_account = _account(repository, "fake", "target")
    first = repository.create_person("First", timezone="UTC")
    second = repository.create_person("Second", timezone="UTC")
    _, source_endpoint = repository.upsert_identity(
        connected_account_id=source_account["id"], external_id="first-source",
        display_name="First", person_id=first["id"],
    )
    _, target_endpoint = repository.upsert_identity(
        connected_account_id=target_account["id"], external_id="first-target",
        display_name="First", person_id=first["id"],
    )
    group = repository.create_group("Recipients")
    repository.add_group_member(group["id"], first["id"])
    frozen_group = repository.group_preview(group["id"])
    segment = repository.create_segment("UTC people", {"timezone": "UTC"})
    frozen_segment = repository.segment_preview(segment["id"])

    route_version = "fixture-route-version"
    draft = repository.create_draft(
        person_id=first["id"], source_account_id=source_account["id"],
        source_endpoint_id=source_endpoint["id"], target_account_id=target_account["id"],
        endpoint_id=target_endpoint["id"], route_version=route_version,
        recipients=frozen_group["members"], payload="draft only",
    )
    segment_draft = repository.create_draft(
        person_id=first["id"], source_account_id=source_account["id"],
        source_endpoint_id=source_endpoint["id"], target_account_id=target_account["id"],
        endpoint_id=target_endpoint["id"], route_version=route_version,
        recipients=frozen_segment["members"], payload="segment draft only",
    )
    repository.add_group_member(group["id"], second["id"])
    with repository.transaction() as connection:
        connection.execute(
            "UPDATE persons SET timezone = 'Europe/Moscow' WHERE id = ?", (second["id"],)
        )
    changed_group = repository.group_preview(group["id"])
    changed_segment = repository.segment_preview(segment["id"])
    assert changed_group["preview_hash"] != frozen_group["preview_hash"]
    assert changed_segment["preview_hash"] != frozen_segment["preview_hash"]
    assert json.loads(repository.get_draft(draft["id"])["recipient_preview_json"]) == frozen_group["members"]
    assert {item["person_id"] for item in frozen_segment["members"]} == {first["id"], second["id"]}
    assert json.loads(repository.get_draft(segment_draft["id"])["recipient_preview_json"]) == frozen_segment["members"]


def test_legacy_skill_shims_route_to_core_without_direct_sql_or_senders():
    root = Path(__file__).resolve().parents[2]
    facebook = (root / "skills/social-media/facebook/SKILL.md").read_text(encoding="utf-8")
    dialogue = (root / "skills/dialogue_campaigns/SKILL.md").read_text(encoding="utf-8")
    for content in (facebook, dialogue):
        assert "$manage-communications" in content
        assert "hermes communication" in content
        lowered = content.lower()
        assert "docker exec" not in lowered
        assert "sqlite3" not in lowered
        assert "--approval-token" not in lowered
        assert "facebook_messages_send.py" not in lowered
    assert "retired" in dialogue.lower()
    assert "write_actions_enabled=0" in facebook


def test_communication_config_defaults_keep_all_execution_workers_off():
    defaults = DEFAULT_CONFIG["communication"]
    assert defaults["outbox_workers_enabled"] is False
    assert defaults["test_sink_enabled"] is False
    example_path = Path(__file__).resolve().parents[2] / "cli-config.yaml.example"
    example = yaml.safe_load(example_path.read_text(encoding="utf-8"))
    assert example["communication"]["outbox_workers_enabled"] is False
    assert example["communication"]["test_sink_enabled"] is False
