"""Canonical classic-CLI slash-command bindings and invocation normalization.

This module is the bounded PR-3 seam between the central command definition
owner and the legacy :meth:`cli.HermesCLI.process_command` dispatcher.  It does
not own command spelling, aliases, help text, argument semantics, policy, or
result rendering.  Those remain registry/dispatcher concerns.

Current main does not yet expose the versioned ``command_id`` field declared by
#96692.  Bindings therefore use it when present and otherwise fall back to the
canonical command name.  That compatibility rule lets the classic CLI adopt the
stable identity automatically when the schema slice lands, without creating a
second registry in the meantime.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from hermes_cli.commands import COMMAND_REGISTRY, CommandDef, resolve_command


_COMMAND_LINE_RE = re.compile(r"^/(?P<entered>[^\s/]+)(?P<tail>\s.*)?$", re.DOTALL)


def _command_id(command: CommandDef) -> str:
    """Return the stable identity available on this source tree."""
    value = getattr(command, "command_id", None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return command.name


def _classic_cli_candidate(command: CommandDef) -> bool:
    """Return whether a command can be invoked from the classic CLI."""
    return not command.gateway_only


@dataclass(frozen=True, slots=True)
class ClassicCLICommandBinding:
    """One classic-CLI execution binding for one canonical command identity."""

    command_id: str
    canonical_name: str
    dispatch_method: str
    shared_execute: str | None

    @classmethod
    def from_command(cls, command: CommandDef) -> "ClassicCLICommandBinding":
        return cls(
            command_id=_command_id(command),
            canonical_name=command.name,
            dispatch_method="process_command",
            shared_execute=getattr(command, "execute", None),
        )


@dataclass(frozen=True, slots=True)
class ClassicCLIInvocation:
    """Normalized classic-CLI command attempt bound to canonical identity."""

    binding: ClassicCLICommandBinding
    entered_name: str
    raw_arguments: str
    canonical_input: str


def build_classic_cli_bindings(
    commands: Iterable[CommandDef] = COMMAND_REGISTRY,
) -> tuple[ClassicCLICommandBinding, ...]:
    """Build the immutable classic-CLI binding projection.

    Duplicate semantic identities or canonical names fail closed.  Aliases do
    not create additional bindings; they resolve to the canonical ``CommandDef``
    before invocation settlement.
    """
    bindings: list[ClassicCLICommandBinding] = []
    seen_ids: set[str] = set()
    seen_names: set[str] = set()

    for command in commands:
        if not _classic_cli_candidate(command):
            continue
        binding = ClassicCLICommandBinding.from_command(command)
        command_id = binding.command_id.strip()
        canonical_name = binding.canonical_name.strip()
        if not command_id:
            raise ValueError("classic CLI command identity must not be blank")
        if not canonical_name:
            raise ValueError("classic CLI canonical command name must not be blank")

        id_key = command_id.casefold()
        name_key = canonical_name.casefold()
        if id_key in seen_ids:
            raise ValueError(f"duplicate classic CLI command identity: {command_id}")
        if name_key in seen_names:
            raise ValueError(
                f"duplicate classic CLI canonical command name: {canonical_name}"
            )
        seen_ids.add(id_key)
        seen_names.add(name_key)
        bindings.append(binding)

    return tuple(bindings)


CLASSIC_CLI_COMMAND_BINDINGS: tuple[ClassicCLICommandBinding, ...] = (
    build_classic_cli_bindings()
)
_BINDINGS_BY_ID = {
    binding.command_id.casefold(): binding for binding in CLASSIC_CLI_COMMAND_BINDINGS
}
_BINDINGS_BY_NAME = {
    binding.canonical_name.casefold(): binding
    for binding in CLASSIC_CLI_COMMAND_BINDINGS
}


def _typed_token(value: str) -> str:
    normalized = str(value or "").strip()
    if normalized.startswith("/"):
        normalized = normalized[1:]
    return normalized.split(None, 1)[0].casefold() if normalized else ""


def resolve_classic_cli_command_binding(
    token: str,
) -> ClassicCLICommandBinding | None:
    """Resolve a typed name or alias to exactly one classic-CLI binding."""
    typed = _typed_token(token)
    if not typed:
        return None
    command = resolve_command(typed)
    if command is None or not _classic_cli_candidate(command):
        return None

    binding = _BINDINGS_BY_ID.get(_command_id(command).casefold())
    if binding is not None:
        return binding
    # Mixed-version compatibility: a process may gain stable ids after this
    # immutable module snapshot was built.  Resolve only the same canonical
    # owner; never guess across aliases or catalog order.
    return _BINDINGS_BY_NAME.get(command.name.casefold())


def normalize_classic_cli_invocation(
    command_line: str,
) -> ClassicCLIInvocation | None:
    """Normalize one slash-shaped CLI input without changing its arguments.

    Unknown commands, gateway-only commands, and ordinary prompt text return
    ``None``.  The entered alias is retained for provenance while execution is
    routed with the canonical name.  Argument spelling and case are preserved;
    only the separator between the command token and arguments is normalized to
    the exact text accepted by the legacy dispatcher.
    """
    if not isinstance(command_line, str):
        return None
    text = command_line.strip()
    match = _COMMAND_LINE_RE.fullmatch(text)
    if match is None:
        return None

    entered_name = match.group("entered")
    binding = resolve_classic_cli_command_binding(entered_name)
    if binding is None:
        return None

    tail = match.group("tail") or ""
    raw_arguments = tail.lstrip()
    canonical_input = f"/{binding.canonical_name}"
    if raw_arguments:
        canonical_input = f"{canonical_input} {raw_arguments}"

    return ClassicCLIInvocation(
        binding=binding,
        entered_name=entered_name,
        raw_arguments=raw_arguments,
        canonical_input=canonical_input,
    )


def bind_classic_cli_dispatch(
    cli: object,
    token: str,
) -> Callable[[str], bool] | None:
    """Return the legacy dispatcher callable for a canonical CLI command."""
    binding = resolve_classic_cli_command_binding(token)
    if binding is None:
        return None
    dispatcher = getattr(cli, binding.dispatch_method, None)
    return dispatcher if callable(dispatcher) else None


def dispatch_classic_cli_command(cli: object, command_line: str) -> bool | None:
    """Dispatch through the canonical binding while preserving legacy output.

    The legacy ``process_command`` method remains the current execution owner;
    this seam canonicalizes identity and alias routing only.  ``None`` means the
    attempt was not admitted or no callable owner exists.  A future shared
    dispatcher can replace the callable without changing the binding contract.
    """
    invocation = normalize_classic_cli_invocation(command_line)
    if invocation is None:
        return None
    dispatcher = getattr(cli, invocation.binding.dispatch_method, None)
    if not callable(dispatcher):
        return None
    return dispatcher(invocation.canonical_input)
