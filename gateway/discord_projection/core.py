"""Canonical Discord wire normalization and typed mismatch errors."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

DISCORD_COMMAND_NAME_RE = re.compile(r"^[a-z0-9_-]{1,32}$")
STRING_OPTION = 3
CommandKey = tuple[int, str]


class DiscordProjectionError(ValueError):
    """Base failure for invalid or contradictory Discord projections."""


class DiscordProjectionMismatch(DiscordProjectionError):
    """Remote read-back did not settle to the desired projection."""

    def __init__(
        self,
        *,
        missing: Sequence[CommandKey] = (),
        unexpected: Sequence[CommandKey] = (),
        changed: Sequence[CommandKey] = (),
    ) -> None:
        self.missing = tuple(missing)
        self.unexpected = tuple(unexpected)
        self.changed = tuple(changed)
        parts: list[str] = []
        if self.missing:
            parts.append(f"missing={list(self.missing)!r}")
        if self.unexpected:
            parts.append(f"unexpected={list(self.unexpected)!r}")
        if self.changed:
            parts.append(f"changed={list(self.changed)!r}")
        detail = ", ".join(parts) if parts else "unknown mismatch"
        super().__init__(f"Discord command read-back mismatch: {detail}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def deep_copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def canonicalize_discord_option(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce one Discord option to semantic fields Hermes manages."""
    return {
        "type": int(payload.get("type", 0) or 0),
        "name": str(payload.get("name", "") or ""),
        "description": str(payload.get("description", "") or ""),
        "required": bool(payload.get("required", False)),
        "autocomplete": bool(payload.get("autocomplete", False)),
        "choices": [
            {
                "name": str(choice.get("name", "") or ""),
                "value": choice.get("value"),
            }
            for choice in payload.get("choices", []) or []
            if isinstance(choice, Mapping)
        ],
        "channel_types": list(payload.get("channel_types", []) or []),
        "min_value": payload.get("min_value"),
        "max_value": payload.get("max_value"),
        "min_length": payload.get("min_length"),
        "max_length": payload.get("max_length"),
        "options": [
            canonicalize_discord_option(item)
            for item in payload.get("options", []) or []
            if isinstance(item, Mapping)
        ],
    }


def canonicalize_discord_command(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce one command payload to semantic fields Hermes manages."""
    contexts = payload.get("contexts")
    integration_types = payload.get("integration_types")
    permissions = payload.get("default_member_permissions")
    return {
        "type": int(payload.get("type", 1) or 1),
        "name": str(payload.get("name", "") or ""),
        "description": str(payload.get("description", "") or ""),
        "default_member_permissions": (
            None if permissions is None else str(permissions)
        ),
        "dm_permission": bool(payload.get("dm_permission", True)),
        "nsfw": bool(payload.get("nsfw", False)),
        "contexts": sorted(int(value) for value in contexts) if contexts else None,
        "integration_types": (
            sorted(int(value) for value in integration_types)
            if integration_types
            else None
        ),
        "options": [
            canonicalize_discord_option(item)
            for item in payload.get("options", []) or []
            if isinstance(item, Mapping)
        ],
    }
