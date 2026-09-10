from __future__ import annotations

import json

import pytest

from hermes_cli.command_catalog import (
    SCHEMA_VERSION,
    build_command_catalog,
    resolve_catalog_command,
)
from hermes_cli.commands import COMMAND_REGISTRY, command_desktop_meta


def test_catalog_is_deterministic_and_aliases_preserve_identity():
    first = build_command_catalog()
    second = build_command_catalog()

    assert first.schema_version == SCHEMA_VERSION
    assert first.revision == second.revision
    assert first.to_dict() == second.to_dict()
    assert len(first.revision) == 64

    new = resolve_catalog_command(first, "/new")
    reset = resolve_catalog_command(first, "/reset")
    assert new is not None
    assert reset is not None
    assert new is reset
    assert new.command_id == "command.new"


def test_catalog_is_json_serializable_and_projects_current_registry():
    catalog = build_command_catalog()
    payload = catalog.to_dict()

    json.dumps(payload, sort_keys=True)
    assert len(catalog.commands) == len(COMMAND_REGISTRY)

    by_name = {spec.name: spec for spec in catalog.commands}
    for command in COMMAND_REGISTRY:
        spec = by_name[command.name]
        assert spec.aliases == command.aliases
        assert spec.description_fallback == command.description
        assert spec.category == command.category
        assert spec.args_hint == command.args_hint
        assert spec.subcommands == command.subcommands
        assert spec.legacy["busy_policy"] == command.busy_policy
        assert spec.legacy["execute"] == command.execute
        assert {
            "argument_mode": spec.legacy["argument_mode"],
            "desktop": spec.legacy["desktop"],
        } == command_desktop_meta(command)


def test_dynamic_contributions_require_explicit_identity_and_change_revision():
    baseline = build_command_catalog()
    catalog = build_command_catalog(
        contributions=(
            (
                "plugin.example",
                {
                    "command_id": "plugin.example.deploy",
                    "name": "deploy",
                    "aliases": ["ship"],
                    "description": "Deploy the current project",
                    "category": "Plugins",
                },
            ),
        )
    )

    deployed = resolve_catalog_command(catalog, "ship")
    assert deployed is not None
    assert deployed.command_id == "plugin.example.deploy"
    assert deployed.origin == "plugin.example"
    assert catalog.revision != baseline.revision

    with pytest.raises(ValueError, match="missing command_id"):
        build_command_catalog(
            contributions=(("plugin.example", {"name": "anonymous"}),)
        )


def test_duplicate_alias_and_duplicate_id_fail_closed():
    with pytest.raises(ValueError, match="collides"):
        build_command_catalog(
            contributions=(
                (
                    "plugin.example",
                    {
                        "command_id": "plugin.example.not-new",
                        "name": "not-new",
                        "aliases": ["new"],
                    },
                ),
            )
        )

    with pytest.raises(ValueError, match="duplicate command_id"):
        build_command_catalog(
            contributions=(
                (
                    "plugin.one",
                    {"command_id": "extension.same", "name": "one"},
                ),
                (
                    "plugin.two",
                    {"command_id": "extension.same", "name": "two"},
                ),
            )
        )
