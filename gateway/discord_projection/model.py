"""Immutable Discord command projection, fingerprint, and read-back proof."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .core import (
    DISCORD_COMMAND_NAME_RE,
    CommandKey,
    DiscordProjectionError,
    DiscordProjectionMismatch,
    canonical_json,
    canonicalize_discord_command,
    deep_copy_json,
)
from .identity import semantic_identity


@dataclass(frozen=True, slots=True)
class DiscordProjectedCommand:
    """One immutable command projection with wire and semantic identities."""

    command_id: str
    canonical_name: str
    entered_name: str
    command_type: int
    wire_json: str
    canonical_json: str

    @property
    def key(self) -> CommandKey:
        return self.command_type, self.entered_name.casefold()

    def wire_payload(self) -> dict[str, Any]:
        return json.loads(self.wire_json)

    def canonical_payload(self) -> dict[str, Any]:
        return json.loads(self.canonical_json)


@dataclass(frozen=True, slots=True)
class DiscordCommandProjection:
    """Immutable ordered projection plus deterministic semantic revision."""

    entries: tuple[DiscordProjectedCommand, ...]
    revision: str

    def wire_commands(self) -> list[dict[str, Any]]:
        return [entry.wire_payload() for entry in self.entries]

    def canonical_by_key(self) -> dict[CommandKey, dict[str, Any]]:
        return {entry.key: entry.canonical_payload() for entry in self.entries}

    def semantic_by_key(self) -> dict[CommandKey, tuple[str, str]]:
        return {
            entry.key: (entry.command_id, entry.canonical_name)
            for entry in self.entries
        }


def project_discord_commands(
    payloads: Iterable[Mapping[str, Any]],
) -> DiscordCommandProjection:
    """Build a fail-closed, deterministic projection from Discord wire rows."""
    entries: list[DiscordProjectedCommand] = []
    seen_keys: set[CommandKey] = set()

    for raw in payloads:
        if not isinstance(raw, Mapping):
            raise DiscordProjectionError(
                f"Discord command payload must be a mapping, got {type(raw).__name__}"
            )
        wire = deep_copy_json(dict(raw))
        canonical = canonicalize_discord_command(wire)
        name = canonical["name"]
        command_type = canonical["type"]
        if not DISCORD_COMMAND_NAME_RE.fullmatch(name):
            raise DiscordProjectionError(f"invalid Discord command name: {name!r}")
        key = command_type, name.casefold()
        if key in seen_keys:
            raise DiscordProjectionError(
                f"duplicate Discord command projection: type={command_type} name={name!r}"
            )
        seen_keys.add(key)
        command_id, canonical_name = semantic_identity(name, command_type)
        entries.append(
            DiscordProjectedCommand(
                command_id=command_id,
                canonical_name=canonical_name,
                entered_name=name,
                command_type=command_type,
                wire_json=canonical_json(wire),
                canonical_json=canonical_json(canonical),
            )
        )

    revision_rows = [
        {
            "command_id": entry.command_id,
            "canonical_name": entry.canonical_name,
            "entered_name": entry.entered_name,
            "type": entry.command_type,
            "payload": entry.canonical_payload(),
        }
        for entry in sorted(entries, key=lambda item: item.key)
    ]
    revision = hashlib.sha256(
        canonical_json(revision_rows).encode("utf-8")
    ).hexdigest()
    return DiscordCommandProjection(entries=tuple(entries), revision=revision)


def verify_discord_projection_readback(
    expected: DiscordCommandProjection,
    remote_payloads: Iterable[Mapping[str, Any]],
) -> DiscordCommandProjection:
    """Verify exact remote settlement and return the observed projection."""
    observed = project_discord_commands(remote_payloads)
    desired_by_key = expected.canonical_by_key()
    observed_by_key = observed.canonical_by_key()

    desired_keys = set(desired_by_key)
    observed_keys = set(observed_by_key)
    missing = sorted(desired_keys - observed_keys)
    unexpected = sorted(observed_keys - desired_keys)
    changed = sorted(
        key
        for key in desired_keys & observed_keys
        if desired_by_key[key] != observed_by_key[key]
        or expected.semantic_by_key()[key] != observed.semantic_by_key()[key]
    )
    if missing or unexpected or changed:
        raise DiscordProjectionMismatch(
            missing=missing,
            unexpected=unexpected,
            changed=changed,
        )
    if observed.revision != expected.revision:
        raise DiscordProjectionMismatch(changed=sorted(desired_keys))
    return observed
