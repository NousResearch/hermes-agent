from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

from hermes_cli.commands import (
    COMMAND_REGISTRY,
    VALID_BUSY_POLICIES,
    CommandDef,
    command_desktop_meta,
    resolve_command,
)

ROOT = Path(__file__).resolve().parents[2]
INVENTORY_PATH = ROOT / "docs/architecture/command-plane/pr0-inventory.json"
EXPECTED_FILES = [
    "docs/architecture/command-plane/pr0-inventory.json",
    "docs/architecture/command-plane/pr0-characterization.md",
    "tests/conformance/test_command_plane_characterization.py",
]
EXPECTED_INTERLOCKS = {
    96990,
    96955,
    96462,
    50054,
    95388,
    96361,
    66163,
    96243,
    86508,
}


def _inventory() -> dict:
    return json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))


def test_inventory_shape_paths_and_file_list() -> None:
    inventory = _inventory()

    assert inventory["schema_version"] == 1
    assert inventory["issue"] == 96692
    assert len(inventory["source_pin"]) == 40
    assert inventory["planned_files"] == EXPECTED_FILES

    authorities = inventory["authorities"]
    ids = [authority["id"] for authority in authorities]
    paths = [authority["path"] for authority in authorities]

    assert len(ids) == len(set(ids))
    assert len(paths) == len(set(paths))

    for relative in paths:
        path = Path(relative)
        assert not path.is_absolute()
        assert "\\" not in relative
        assert ".." not in path.parts
        assert (ROOT / path).exists(), f"missing characterized authority: {relative}"

    interlocks = {item["pr"] for item in inventory["semantic_interlocks"]}
    assert interlocks == EXPECTED_INTERLOCKS
    assert {item["collision"] for item in inventory["semantic_interlocks"]} == {"none"}


def test_core_registry_tokens_resolve_to_one_object() -> None:
    token_owners: dict[str, str] = {}

    for command in COMMAND_REGISTRY:
        for entered_name in (command.name, *command.aliases):
            token = entered_name.casefold()
            previous = token_owners.setdefault(token, command.name)
            assert previous == command.name, (
                f"command token {entered_name!r} is owned by both "
                f"{previous!r} and {command.name!r}"
            )

            assert resolve_command(entered_name) is command
            assert resolve_command(f"/{entered_name}") is command
            assert resolve_command(entered_name.upper()) is command


def test_desktop_metadata_is_a_projection_of_commanddef() -> None:
    for command in COMMAND_REGISTRY:
        metadata = command_desktop_meta(command)

        assert set(metadata) == {"argument_mode", "desktop"}
        assert metadata["argument_mode"] in {None, "options", "text", "mixed"}
        assert metadata["desktop"] == command.desktop

        for alias in command.aliases:
            resolved = resolve_command(alias)
            assert resolved is command
            assert command_desktop_meta(resolved) == metadata


def test_inventory_matches_current_commanddef_boundary() -> None:
    inventory = _inventory()
    current_fields = {field.name for field in fields(CommandDef)}

    assert set(inventory["current_commanddef_fields"]) == current_fields
    assert current_fields.isdisjoint(inventory["missing_command_plane_fields"])
    assert {command.busy_policy for command in COMMAND_REGISTRY} <= VALID_BUSY_POLICIES
