"""Versioned semantic projection of the canonical Hermes slash-command registry.

PR 1 of the unified command-plane migration.  The existing
``hermes_cli.commands.COMMAND_REGISTRY`` remains the semantic owner; this
module gives that owner a client-safe, immutable schema with canonical IDs,
alias-preserving resolution, deterministic fingerprints, and a fail-closed
collision boundary for later contributors.

No surface dispatches through this module yet.  Adoption belongs to later
migration slices after the compatibility boundary is proved.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from hermes_cli.commands import COMMAND_REGISTRY, CommandDef, command_desktop_meta

SCHEMA_VERSION = 2


@dataclass(frozen=True)
class CommandSpec:
    """JSON-safe semantic projection of one command definition."""

    schema_version: int
    command_id: str
    name: str
    aliases: tuple[str, ...]
    description_fallback: str
    category: str
    args_hint: str
    subcommands: tuple[str, ...]
    origin: str
    legacy: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "command_id": self.command_id,
            "name": self.name,
            "aliases": list(self.aliases),
            "description_fallback": self.description_fallback,
            "category": self.category,
            "argument_schema": {
                "kind": "legacy",
                "hint": self.args_hint,
                "subcommands": list(self.subcommands),
            },
            "origin": self.origin,
            "legacy": dict(self.legacy),
        }


@dataclass(frozen=True)
class CommandCatalog:
    """Immutable command snapshot addressed by deterministic revision."""

    schema_version: int
    revision: str
    commands: tuple[CommandSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "revision": self.revision,
            "commands": [command.to_dict() for command in self.commands],
        }


def _command_id(name: str) -> str:
    """Seed the stable core ID namespace from the current canonical owner.

    The ID is intentionally distinct from entered spelling: aliases always
    resolve to this one identity.  A future canonical-name migration must
    preserve the seeded ID explicitly rather than minting a new semantic
    command; PR 1 pins the initial namespace and later slices consume it.
    """

    return "command." + name.replace("_", "-")


def _legacy_spec(command: CommandDef) -> CommandSpec:
    desktop_meta = command_desktop_meta(command)
    return CommandSpec(
        schema_version=SCHEMA_VERSION,
        command_id=_command_id(command.name),
        name=command.name,
        aliases=tuple(command.aliases),
        description_fallback=command.description,
        category=command.category,
        args_hint=command.args_hint,
        subcommands=tuple(command.subcommands),
        origin="core",
        legacy={
            "cli_only": command.cli_only,
            "gateway_only": command.gateway_only,
            "gateway_config_gate": command.gateway_config_gate,
            "busy_policy": command.busy_policy,
            "busy_handler": command.busy_handler,
            "execute": command.execute,
            "argument_mode": desktop_meta["argument_mode"],
            "desktop": desktop_meta["desktop"],
        },
    )


def _normalize_external_spec(value: Mapping[str, Any], origin: str) -> CommandSpec:
    """Normalize a later contributor without granting it mutation authority."""

    name = str(value.get("name") or "").lstrip("/").strip()
    if not name:
        raise ValueError(f"{origin} command contribution missing name")

    command_id = str(value.get("command_id") or "").strip()
    if not command_id:
        raise ValueError(f"{origin} command contribution missing command_id")

    aliases = tuple(str(item).lstrip("/") for item in value.get("aliases", ()) or ())
    subcommands = tuple(str(item) for item in value.get("subcommands", ()) or ())
    legacy = value.get("legacy") or {}
    if not isinstance(legacy, Mapping):
        raise ValueError(f"{command_id}: legacy metadata must be a mapping")

    return CommandSpec(
        schema_version=SCHEMA_VERSION,
        command_id=command_id,
        name=name,
        aliases=aliases,
        description_fallback=str(
            value.get("description_fallback") or value.get("description") or ""
        ),
        category=str(value.get("category") or "Extensions"),
        args_hint=str(value.get("args_hint") or ""),
        subcommands=subcommands,
        origin=origin,
        legacy=dict(legacy),
    )


def validate_command_specs(specs: Sequence[CommandSpec]) -> None:
    """Fail closed on duplicate semantic IDs or entered command tokens."""

    ids: dict[str, str] = {}
    tokens: dict[str, str] = {}

    for spec in specs:
        previous = ids.setdefault(spec.command_id, spec.origin)
        if previous != spec.origin:
            raise ValueError(
                f"duplicate command_id {spec.command_id!r}: {previous} vs {spec.origin}"
            )

        for token in (spec.name, *spec.aliases):
            key = token.casefold()
            owner = tokens.setdefault(key, spec.command_id)
            if owner != spec.command_id:
                raise ValueError(
                    f"command token {token!r} collides: {owner} vs {spec.command_id}"
                )


def _fingerprint(specs: Sequence[CommandSpec]) -> str:
    payload = json.dumps(
        [spec.to_dict() for spec in specs],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_command_catalog(
    *,
    contributions: Iterable[tuple[str, Mapping[str, Any]]] = (),
) -> CommandCatalog:
    """Project the canonical registry plus explicit contributions.

    PR 1 exposes contributor validation but does not perform runtime discovery;
    context-scoped assembly is the next migration slice.
    """

    specs = [_legacy_spec(command) for command in COMMAND_REGISTRY]
    specs.extend(_normalize_external_spec(value, origin) for origin, value in contributions)
    specs.sort(key=lambda item: (item.category.casefold(), item.name.casefold(), item.command_id))
    validate_command_specs(specs)
    return CommandCatalog(
        schema_version=SCHEMA_VERSION,
        revision=_fingerprint(specs),
        commands=tuple(specs),
    )


def resolve_catalog_command(catalog: CommandCatalog, token: str) -> CommandSpec | None:
    normalized = str(token or "").lstrip("/").casefold()
    if not normalized:
        return None
    for spec in catalog.commands:
        if spec.name.casefold() == normalized:
            return spec
        if any(alias.casefold() == normalized for alias in spec.aliases):
            return spec
    return None
