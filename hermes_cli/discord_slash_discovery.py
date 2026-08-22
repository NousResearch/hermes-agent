"""Isolated discovery for Discord slash-command metadata.

This module is both the single discovery implementation and the child-process
entrypoint used by the Discord adapter. Keep its output JSON-only: callback
objects stay in the gateway process and are built from this metadata there.
"""

from __future__ import annotations

import json
from typing import Any


def discover_discord_slash_metadata() -> dict[str, Any]:
    """Return JSON-safe gateway, plugin, and skill command metadata."""
    from hermes_cli.commands import (
        COMMAND_REGISTRY,
        _is_gateway_available,
        _iter_plugin_command_entries,
        _resolve_config_gates,
        discord_skill_commands_by_category,
    )

    config_overrides = _resolve_config_gates()
    commands = [
        [command.name, command.description, command.args_hint]
        for command in COMMAND_REGISTRY
        if _is_gateway_available(command, config_overrides)
    ]
    plugins = [list(entry) for entry in _iter_plugin_command_entries()]
    categories, uncategorized, hidden = discord_skill_commands_by_category(
        reserved_names=set()
    )
    skills = list(uncategorized)
    for category_skills in categories.values():
        skills.extend(category_skills)

    return {
        "schema_version": 1,
        "commands": commands,
        "plugins": plugins,
        "skills": [list(entry) for entry in skills],
        "skill_hidden": hidden,
    }


def serialize_discord_slash_metadata() -> str:
    """Serialize discovery output for the adapter child-process protocol."""
    return json.dumps(
        discover_discord_slash_metadata(),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def main() -> None:
    print(serialize_discord_slash_metadata())


if __name__ == "__main__":
    main()
