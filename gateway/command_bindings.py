"""Canonical gateway slash-command handler bindings.

This module is deliberately smaller than the legacy gateway command owner.  It
owns the *binding* from canonical command identity to the gateway callable name;
it does not own command spelling, aliases, help text, or busy policy.  Those
remain in :mod:`hermes_cli.commands` until the versioned command contract lands.

The compatibility fallback from ``command_id`` to ``CommandDef.name`` is
intentional: current main does not yet expose the PR-1 stable-id field.  Once
that field exists, the same binding objects automatically key themselves by the
stable identity without another gateway table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from hermes_cli.commands import COMMAND_REGISTRY, CommandDef, resolve_command


# Handler names that cannot be derived mechanically from the canonical command
# spelling.  Keep this map tiny: every ordinary command binds by convention.
_HANDLER_NAME_OVERRIDES: dict[str, str] = {
    "new": "_handle_reset_command",
    "sethome": "_handle_set_home_command",
}


def _handler_name(command: CommandDef) -> str:
    return _HANDLER_NAME_OVERRIDES.get(
        command.name,
        f"_handle_{command.name.replace('-', '_')}_command",
    )


def _command_id(command: CommandDef) -> str:
    """Return the canonical semantic identity available on this source tree."""
    value = getattr(command, "command_id", None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return command.name


@dataclass(frozen=True, slots=True)
class GatewayCommandBinding:
    """One gateway execution binding for one canonical command identity."""

    command_id: str
    canonical_name: str
    handler_name: str
    busy_handler_name: str | None
    shared_execute: str | None

    @classmethod
    def from_command(cls, command: CommandDef) -> "GatewayCommandBinding":
        busy_handler = command.busy_handler
        return cls(
            command_id=_command_id(command),
            canonical_name=command.name,
            handler_name=_handler_name(command),
            busy_handler_name=(
                f"_busy_{busy_handler.replace('-', '_')}_command"
                if busy_handler
                else None
            ),
            shared_execute=command.execute,
        )


def _gateway_candidate(command: CommandDef) -> bool:
    """Return whether a core command can be projected to a gateway context.

    ``gateway_config_gate`` is a conditional gateway projection even when the
    command is otherwise marked CLI-only, so it remains part of the candidate
    binding set.  Runtime availability still belongs to policy, not this table.
    """
    return not command.cli_only or command.gateway_only or bool(command.gateway_config_gate)


def build_gateway_bindings(
    commands: Iterable[CommandDef] = COMMAND_REGISTRY,
) -> tuple[GatewayCommandBinding, ...]:
    """Build the immutable canonical gateway binding projection.

    Duplicate semantic identities fail closed.  Alias tokens never create new
    bindings because aliases resolve through ``resolve_command`` to the same
    canonical ``CommandDef`` first.
    """
    bindings: list[GatewayCommandBinding] = []
    seen: set[str] = set()
    for command in commands:
        if not _gateway_candidate(command):
            continue
        binding = GatewayCommandBinding.from_command(command)
        key = binding.command_id.casefold()
        if key in seen:
            raise ValueError(f"duplicate gateway command identity: {binding.command_id}")
        seen.add(key)
        bindings.append(binding)
    return tuple(bindings)


GATEWAY_COMMAND_BINDINGS: tuple[GatewayCommandBinding, ...] = build_gateway_bindings()
_BINDINGS_BY_ID = {binding.command_id.casefold(): binding for binding in GATEWAY_COMMAND_BINDINGS}
_BINDINGS_BY_NAME = {
    binding.canonical_name.casefold(): binding for binding in GATEWAY_COMMAND_BINDINGS
}


def resolve_gateway_command_binding(token: str) -> GatewayCommandBinding | None:
    """Resolve a typed name/alias to exactly one canonical gateway binding."""
    command = resolve_command(token)
    if command is None or not _gateway_candidate(command):
        return None
    command_id = _command_id(command).casefold()
    binding = _BINDINGS_BY_ID.get(command_id)
    if binding is not None:
        return binding
    # Defensive compatibility for a partially upgraded process in which the
    # registry object gained an id after this module's immutable snapshot was
    # created.  Never guess across canonical names: resolve only the same owner.
    return _BINDINGS_BY_NAME.get(command.name.casefold())


def bind_gateway_handler(runner: object, token: str):
    """Return the runner callable for a canonical gateway command, or ``None``.

    Shared ``execute`` commands are intentionally not synthesized here.  Their
    execution owner is the shared executor/dispatcher; this function only binds
    gateway-owned handlers.  Missing attributes fail closed rather than falling
    through to prompt text.
    """
    binding = resolve_gateway_command_binding(token)
    if binding is None:
        return None
    handler = getattr(runner, binding.handler_name, None)
    return handler if callable(handler) else None
