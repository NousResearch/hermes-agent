"""Invariant for registry-owned slash execution (CommandDef.execute).

Every ``CommandDef`` with ``execute`` set must name a key that exists in
:data:`hermes_cli.slash_exec.EXECUTORS`; otherwise the command silently falls
through to "unknown command" on every surface.
"""

from hermes_cli.commands import COMMAND_REGISTRY
from hermes_cli.slash_exec import resolve_executor


def test_every_execute_key_resolves_to_an_executor():
    migrated = [cmd for cmd in COMMAND_REGISTRY if cmd.execute]
    assert migrated
    unresolved = [cmd.name for cmd in migrated if resolve_executor(cmd) is None]
    assert not unresolved, f"CommandDef.execute names no EXECUTORS entry: {unresolved}"
