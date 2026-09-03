from __future__ import annotations

from copy import deepcopy

import pytest

from gateway.discord_command_projection import (
    DiscordProjectionError,
    DiscordProjectionMismatch,
    build_relay_discord_projection,
    project_discord_commands,
    verify_discord_projection_readback,
)
from gateway.relay.command_manifest import build_relay_command_manifest


def test_relay_manifest_is_the_shared_projection_wire_output() -> None:
    projection = build_relay_discord_projection()

    assert build_relay_command_manifest() == projection.wire_commands()
    assert [row["name"] for row in projection.wire_commands()] == [
        "new",
        "reset",
        "model",
        "reasoning",
        "personality",
        "retry",
        "undo",
        "status",
        "sethome",
        "stop",
        "steer",
        "compress",
        "title",
        "resume",
        "usage",
        "help",
        "insights",
        "reload-mcp",
        "reload-skills",
        "voice",
        "update",
        "restart",
        "approve",
        "deny",
        "thread",
        "queue",
        "bg",
        "btw",
    ]


def test_alias_projection_preserves_one_semantic_identity() -> None:
    projection = build_relay_discord_projection()
    by_name = {entry.entered_name: entry for entry in projection.entries}

    assert by_name["new"].command_id == by_name["reset"].command_id
    assert by_name["new"].canonical_name == "new"
    assert by_name["reset"].canonical_name == "new"


def test_projection_revision_is_independent_of_command_order() -> None:
    rows = build_relay_command_manifest()

    forward = project_discord_commands(rows)
    reverse = project_discord_commands(reversed(rows))

    assert forward.revision == reverse.revision
    assert forward.canonical_by_key() == reverse.canonical_by_key()


def test_readback_accepts_remote_defaults_and_order_changes() -> None:
    desired = build_relay_discord_projection()
    remote = list(reversed(deepcopy(desired.wire_commands())))
    for row in remote:
        row["type"] = 1
        row["dm_permission"] = True
        row["nsfw"] = False
        row["default_member_permissions"] = None

    observed = verify_discord_projection_readback(desired, remote)

    assert observed.revision == desired.revision


def test_readback_fails_closed_on_changed_remote_shape() -> None:
    desired = build_relay_discord_projection()
    remote = deepcopy(desired.wire_commands())
    remote[0]["description"] = "drifted"

    with pytest.raises(DiscordProjectionMismatch) as exc_info:
        verify_discord_projection_readback(desired, remote)

    assert exc_info.value.changed == ((1, "new"),)


def test_readback_fails_closed_on_missing_and_unexpected_commands() -> None:
    desired = project_discord_commands(
        [{"name": "new", "description": "Start a new conversation"}]
    )
    remote = [{"name": "foreign", "description": "Not Hermes"}]

    with pytest.raises(DiscordProjectionMismatch) as exc_info:
        verify_discord_projection_readback(desired, remote)

    assert exc_info.value.missing == ((1, "new"),)
    assert exc_info.value.unexpected == ((1, "foreign"),)


@pytest.mark.parametrize("name", ["UPPER", "has space", "", "x" * 33])
def test_invalid_discord_names_fail_projection(name: str) -> None:
    with pytest.raises(DiscordProjectionError, match="invalid Discord command name"):
        project_discord_commands([{"name": name, "description": "invalid"}])


def test_duplicate_wire_identity_fails_projection() -> None:
    with pytest.raises(DiscordProjectionError, match="duplicate Discord command"):
        project_discord_commands(
            [
                {"name": "status", "description": "one"},
                {"name": "status", "description": "two"},
            ]
        )


def test_projection_consumes_versioned_catalog_identity_when_available(
    monkeypatch,
) -> None:
    catalog = object()
    spec = type("Spec", (), {"command_id": "command.new", "name": "new"})()

    def resolver(actual_catalog, token):
        assert actual_catalog is catalog
        return spec if str(token).lstrip("/") in {"new", "reset"} else None

    monkeypatch.setattr(
        "gateway.discord_projection.identity.versioned_catalog_resolver",
        lambda: (catalog, resolver),
    )

    projection = project_discord_commands(
        [
            {"name": "new", "description": "Start a new conversation"},
            {"name": "reset", "description": "Reset your Hermes session"},
        ]
    )

    assert {entry.command_id for entry in projection.entries} == {"command.new"}
    assert {entry.canonical_name for entry in projection.entries} == {"new"}
