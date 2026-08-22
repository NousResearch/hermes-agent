import json
from types import SimpleNamespace

import hermes_cli.commands as command_module
from hermes_cli.discord_slash_discovery import (
    discover_discord_slash_metadata,
    serialize_discord_slash_metadata,
)


def test_serialized_discovery_matches_direct_metadata_without_mutating_handlers(
    monkeypatch,
):
    registry = [
        SimpleNamespace(
            name="status",
            description="Show status",
            args_hint="",
        ),
        SimpleNamespace(
            name="hidden",
            description="Hidden command",
            args_hint="",
        ),
    ]
    handler = object()
    parent_plugin_handlers = {"wave": handler}
    monkeypatch.setattr(command_module, "COMMAND_REGISTRY", registry)
    monkeypatch.setattr(command_module, "_resolve_config_gates", lambda: {"status"})
    monkeypatch.setattr(
        command_module,
        "_is_gateway_available",
        lambda command, overrides: command.name in overrides,
    )
    monkeypatch.setattr(
        command_module,
        "_iter_plugin_command_entries",
        lambda: [("wave", "Wave hello", "[name]")],
    )
    monkeypatch.setattr(
        command_module,
        "discord_skill_commands_by_category",
        lambda reserved_names: (
            {"testing": [("proof", "Run proof", "/proof")]},
            [("plain", "Plain skill", "/plain")],
            2,
        ),
    )

    direct = discover_discord_slash_metadata()
    serialized = json.loads(serialize_discord_slash_metadata())

    assert serialized == direct
    assert direct == {
        "schema_version": 1,
        "commands": [["status", "Show status", ""]],
        "plugins": [["wave", "Wave hello", "[name]"]],
        "skills": [
            ["plain", "Plain skill", "/plain"],
            ["proof", "Run proof", "/proof"],
        ],
        "skill_hidden": 2,
    }
    assert parent_plugin_handlers == {"wave": handler}
