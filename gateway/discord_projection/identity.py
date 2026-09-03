"""Canonical slash-command identity resolution for Discord projections."""

from __future__ import annotations

from functools import lru_cache

from hermes_cli.commands import resolve_command


@lru_cache(maxsize=1)
def versioned_catalog_resolver():
    """Return PR-1's catalog resolver when that prerequisite is present."""
    try:
        from hermes_cli.command_catalog import (
            build_command_catalog,
            resolve_catalog_command,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "hermes_cli.command_catalog":
            raise
        return None
    catalog = build_command_catalog()
    return catalog, resolve_catalog_command


def semantic_identity(name: str, command_type: int) -> tuple[str, str]:
    """Resolve an entered Discord name to stable identity and canonical name."""
    versioned = versioned_catalog_resolver()
    if versioned is not None:
        catalog, resolver = versioned
        spec = resolver(catalog, name)
        if spec is not None:
            return spec.command_id, spec.name

    command = resolve_command(name)
    if command is None:
        return f"discord.{command_type}.{name}", name
    value = getattr(command, "command_id", None)
    command_id = (
        value.strip()
        if isinstance(value, str) and value.strip()
        else command.name
    )
    return command_id, command.name
