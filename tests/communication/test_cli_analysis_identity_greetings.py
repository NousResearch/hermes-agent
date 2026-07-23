from __future__ import annotations

import argparse
import json

from communication_core.repository import CommunicationRepository
from hermes_cli.communication import communication_command
from hermes_cli.subcommands.communication import build_communication_parser


def _parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_communication_parser(subparsers, cmd_communication=communication_command)
    return parser


def test_cli_catalog_and_redacted_account_output(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    parser = _parser()
    assert parser.parse_args(["communication", "init"]).func(
        parser.parse_args(["communication", "init"])
    ) == 0
    add = parser.parse_args(
        [
            "communication", "accounts", "add",
            "--provider", "fake",
            "--namespace", "test",
            "--label", "test",
            "--owner-profile", "test",
            "--credential-ref", "keyring://communication/test",
            "--browser-profile-ref", "profile://test",
        ]
    )
    assert add.func(add) == 0
    output = capsys.readouterr().out
    assert "keyring://" not in output
    assert "profile://" not in output

    catalog = [
        ["accounts", "list"], ["sync", "run", "acct"],
        ["people", "merge", "a", "b", "--evidence", "manual"],
        ["routes", "dry-run", "p", "s", "t"],
        ["groups", "preview", "g"], ["timeline", "show", "p"],
        ["brief", "daily"], ["analyze", "conversation", "c"],
        ["drafts", "create", "p", "e", "--text", "draft"],
        ["approvals", "approve", "d"], ["greetings", "plan"],
        ["migration", "facebook-import", "acct", "legacy.db"],
        ["migration", "facebook-rollback", "migration-run"],
    ]
    for suffix in catalog:
        assert parser.parse_args(["communication", *suffix]).func is communication_command


def test_manual_merge_unmerge_restores_identity_ownership(tmp_path):
    repository = CommunicationRepository(tmp_path / "communication.db")
    repository.initialize()
    account = repository.add_account(
        provider="fake", account_namespace="one", label="one", owner_profile="test"
    )
    winner = repository.create_person("Same Name")
    duplicate = repository.create_person("Same Name")
    identity, _ = repository.upsert_identity(
        connected_account_id=account["id"],
        external_id="duplicate",
        display_name="Same Name",
        person_id=duplicate["id"],
    )
    candidates = repository.find_duplicate_candidates()
    assert candidates[0]["confidence"] == 0.35
    assert candidates[0]["auto_merge"] is False
    merge = repository.merge_people(
        winner["id"], duplicate["id"], actor="test", evidence={"manual": "verified"}
    )
    assert repository.get_identity_by_external(account["id"], "duplicate")["person_id"] == winner["id"]

    repository.unmerge_people(merge["audit_id"], actor="test")
    assert repository.get_identity_by_external(account["id"], "duplicate")["person_id"] == duplicate["id"]


def test_explainable_analysis_and_timezone_greeting_dedup(tmp_path):
    repository = CommunicationRepository(tmp_path / "communication.db")
    repository.initialize()
    account = repository.add_account(
        provider="fake", account_namespace="one", label="one", owner_profile="test"
    )
    person = repository.create_person("Person", timezone="Europe/Moscow")
    identity, endpoint = repository.upsert_identity(
        connected_account_id=account["id"], external_id="person", display_name="Person",
        person_id=person["id"],
    )
    conversation = repository.upsert_conversation(
        connected_account_id=account["id"], endpoint_id=endpoint["id"], external_id="thread",
        kind="direct", title=None, provenance={"source": "fixture"}, observed_at="2026-07-23T09:00:00Z",
    )
    repository.upsert_message(
        connected_account_id=account["id"], endpoint_id=endpoint["id"], conversation_id=conversation["id"],
        external_id="m1", direction="incoming", body="Can you help?", sent_at="2026-07-23T09:00:00Z",
        sender_identity_id=identity["id"], provenance={"source": "fixture"}, observed_at="2026-07-23T09:00:00Z",
    )
    analysis = repository.analyze_conversation(conversation["id"])
    assert analysis["commitments"][0]["kind"] == "unanswered_question"
    assert "not a psychological classification" in analysis["tone_signal"]["method"]

    event = repository.add_contact_event(
        person_id=person["id"], connected_account_id=account["id"], endpoint_id=endpoint["id"],
        event_type="birthday", external_id="birthday", happened_at="1990-07-23T00:00:00Z",
        timezone="Europe/Moscow", provenance={"source": "fixture"},
    )
    first = repository.plan_greetings("2026-07-23")
    second = repository.plan_greetings("2026-07-23")
    assert first["items"][0]["event_id"] == event["id"]
    assert second["items"][0]["id"] == first["items"][0]["id"]
    assert len(repository.list_greetings("2026-07-23")) == 1

    exclusion = repository.create_group("No greetings", exclusion=True)
    other = repository.create_person("Other", timezone="UTC")
    _, other_endpoint = repository.upsert_identity(
        connected_account_id=account["id"], external_id="other", display_name="Other", person_id=other["id"]
    )
    repository.add_group_member(exclusion["id"], other["id"])
    repository.add_contact_event(
        person_id=other["id"], connected_account_id=account["id"], endpoint_id=other_endpoint["id"],
        event_type="birthday", external_id="other-birthday", happened_at="1991-07-23T00:00:00Z",
        provenance={"source": "fixture"},
    )
    planned = repository.plan_greetings("2026-07-23")
    excluded = next(item for item in planned["items"] if item["person_id"] == other["id"])
    assert excluded["status"] == "excluded"
