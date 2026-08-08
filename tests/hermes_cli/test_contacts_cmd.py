from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta

import pytest

from hermes_cli.contacts_cmd import (
    ContactRegistryError,
    build_parser,
    find_contact,
    init_registry,
    load_registry,
    resolve_contact,
    validate_registry,
)


def _registry(*contacts):
    return validate_registry(
        {
            "schema_version": 1,
            "policy": {
                "default_send": "deny",
            },
            "contacts": list(contacts),
        }
    )


def _contact(*, status="verified", preferred_for=None, destination="123"):
    return {
        "id": "alice-example",
        "display_name": "Alice Example",
        "aliases": ["Alice"],
        "routes": [
            {
                "key": "discord-dm",
                "platform": "discord",
                "destination_type": "dm",
                "destination": destination,
                "preferred_for": preferred_for or ["internal"],
                "status": status,
                "last_verified": "2026-01-02",
                "constraints": ["Check messaging authority before sending."],
            }
        ],
    }


def _directory(destination="123"):
    return {
        "updated_at": datetime.now().astimezone().isoformat(),
        "platforms": {"discord": [{"id": destination, "name": "alice"}]},
    }


def test_init_creates_empty_profile_registry_with_owner_only_mode(tmp_path):
    target = tmp_path / "contacts.yaml"

    assert init_registry(target) == target
    data = load_registry(target)

    assert data["schema_version"] == 1
    assert data["policy"]["default_send"] == "deny"
    assert data["contacts"] == []
    if os.name != "nt":
        assert target.stat().st_mode & 0o777 == 0o600


def test_init_refuses_to_replace_existing_registry_without_force(tmp_path):
    target = tmp_path / "contacts.yaml"
    init_registry(target)

    with pytest.raises(ContactRegistryError, match="already exists"):
        init_registry(target)


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits are not available")
def test_load_fails_closed_when_registry_is_group_or_world_readable(tmp_path):
    target = tmp_path / "contacts.yaml"
    init_registry(target)
    target.chmod(0o644)

    with pytest.raises(ContactRegistryError, match="permissions are too broad"):
        load_registry(target)


def test_registry_rejects_ambiguous_names_across_contacts():
    first = _contact()
    second = {
        "id": "another-person",
        "display_name": "Another Person",
        "aliases": ["Alice"],
        "routes": [],
    }

    with pytest.raises(ContactRegistryError, match="ambiguous contact name"):
        _registry(first, second)


def test_registry_rejects_duplicate_route_keys():
    contact = _contact()
    contact["routes"].append(dict(contact["routes"][0]))

    with pytest.raises(ContactRegistryError, match="duplicate route key"):
        _registry(contact)


@pytest.mark.parametrize("value", [0, "false", None])
def test_registry_rejects_non_boolean_sendable(value):
    contact = _contact()
    contact["routes"][0]["sendable"] = value

    with pytest.raises(ContactRegistryError, match="sendable must be a boolean"):
        _registry(contact)


def test_find_contact_matches_id_display_name_or_alias_exactly():
    data = _registry(_contact())

    assert find_contact(data["contacts"], "alice")[0] == "ok"
    assert find_contact(data["contacts"], "Alice Example")[0] == "ok"
    assert find_contact(data["contacts"], "alice-example")[0] == "ok"
    assert find_contact(data["contacts"], "ali")[0] == "unknown_contact"


@pytest.mark.parametrize("query", ["Alice?", "a-l-i-c-e", "AliceExample"])
def test_find_contact_preserves_punctuation_and_word_boundaries(query):
    data = _registry(_contact())

    assert find_contact(data["contacts"], query)[0] == "unknown_contact"


def test_find_contact_supports_unicode_names():
    contact = _contact()
    contact.update({"id": "zoe", "display_name": "Zoë", "aliases": ["佐伊"]})
    data = _registry(contact)

    assert find_contact(data["contacts"], "ZOË")[0] == "ok"
    assert find_contact(data["contacts"], "佐伊")[0] == "ok"


def test_valid_resolution_is_non_sending_and_redacts_destination_by_default():
    code, result = resolve_contact(
        _registry(_contact()),
        "Alice",
        purpose="internal",
        directory=_directory(),
    )

    assert code == 0
    assert result["status"] == "ok"
    assert result["live_check"] == "fresh_directory_match"
    assert result["send_performed"] is False
    assert result["authorization_check"] == "required"
    assert "destination" not in result


def test_resolution_can_explicitly_show_destination():
    code, result = resolve_contact(
        _registry(_contact()),
        "Alice",
        route_key="discord-dm",
        directory=_directory(),
        show_destination=True,
    )

    assert code == 0
    assert result["destination"] == "123"


def test_unknown_contact_fails_closed():
    code, result = resolve_contact(
        _registry(_contact()),
        "Bob",
        purpose="internal",
        directory=_directory(),
    )

    assert code == 2
    assert result == {
        "status": "unknown_contact",
        "query": "Bob",
        "send_performed": False,
    }


def test_contact_without_matching_preferred_route_fails_closed():
    code, result = resolve_contact(
        _registry(_contact(preferred_for=["internal"])),
        "Alice",
        purpose="external",
        directory=_directory(),
    )

    assert code == 3
    assert result["status"] == "no_preferred_route"
    assert result["send_performed"] is False


def test_stale_route_fails_before_directory_match():
    code, result = resolve_contact(
        _registry(_contact(status="stale")),
        "Alice",
        purpose="internal",
        directory=_directory(),
    )

    assert code == 4
    assert result["status"] == "stale_destination"


def test_live_directory_mismatch_fails_closed():
    code, result = resolve_contact(
        _registry(_contact()),
        "Alice",
        purpose="internal",
        directory=_directory("different"),
    )

    assert code == 4
    assert result["status"] == "destination_not_in_live_directory"
    assert result["live_check"] == "failed"


def test_stale_channel_directory_fails_closed_even_when_destination_matches():
    directory = _directory()
    directory["updated_at"] = (
        datetime.now().astimezone() - timedelta(minutes=11)
    ).isoformat()

    code, result = resolve_contact(
        _registry(_contact()),
        "Alice",
        purpose="internal",
        directory=directory,
    )

    assert code == 4
    assert result["status"] == "stale_channel_directory"
    assert result["live_check"] == "directory_missing_or_stale"


def test_plugin_platform_can_use_fresh_directory_membership():
    contact = _contact()
    contact["routes"][0]["platform"] = "matrix"
    directory = {
        "updated_at": datetime.now().astimezone().isoformat(),
        "platforms": {"matrix": [{"id": "123", "name": "alice"}]},
    }

    code, result = resolve_contact(
        _registry(contact),
        "Alice",
        purpose="internal",
        directory=directory,
    )

    assert code == 0
    assert result["status"] == "ok"
    assert result["live_check"] == "fresh_directory_match"


def test_email_resolution_fails_until_external_live_check_and_never_sends():
    contact = _contact()
    route = contact["routes"][0]
    route.update(
        {
            "key": "email",
            "platform": "email",
            "destination_type": "address",
            "destination": "alice@example.invalid",
        }
    )

    code, result = resolve_contact(
        _registry(contact),
        "Alice",
        purpose="internal",
        directory={},
    )

    assert code == 4
    assert result["status"] == "live_check_unavailable"
    assert result["live_check"] == "unsupported_for_platform"
    assert result["authorization_check"] == "required"
    assert result["send_performed"] is False


def test_cli_parser_initializes_and_validates_selected_file(tmp_path, capsys):
    target = tmp_path / "contacts.yaml"
    parser = argparse.ArgumentParser()
    build_parser(parser.add_subparsers(dest="command"))

    initialized_args = parser.parse_args(
        ["contacts", "--file", str(target), "init"]
    )
    assert initialized_args.func(initialized_args) == 0
    initialized = json.loads(capsys.readouterr().out)

    validated_args = parser.parse_args(
        ["contacts", "--file", str(target), "validate"]
    )
    assert validated_args.func(validated_args) == 0
    validated = json.loads(capsys.readouterr().out)

    assert initialized["path"] == str(target)
    assert validated["contact_count"] == 0
    if os.name != "nt":
        assert target.stat().st_mode & 0o777 == 0o600
