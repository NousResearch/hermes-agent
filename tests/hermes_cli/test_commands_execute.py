"""Invariant tests for registry-owned slash execution (CommandDef.execute).

Every ``CommandDef`` with ``execute`` set must:
  * name a key that exists in :data:`hermes_cli.slash_exec.EXECUTORS`, and
  * produce IDENTICAL core text across surfaces for a fixed context — the
    executor may only vary on ``args``/``options``, never on ``surface``.
"""

import pytest

from hermes_cli.commands import COMMAND_REGISTRY, resolve_command
from hermes_cli.slash_exec import (
    EXECUTORS,
    CommandContext,
    CommandReply,
    execute_command,
    resolve_executor,
    run_execute,
)

MIGRATED = [cmd for cmd in COMMAND_REGISTRY if cmd.execute]

SURFACES = ("cli", "gateway", "tui")


def test_some_commands_are_migrated():
    names = {cmd.name for cmd in MIGRATED}
    # The thin-slice set — extend as more commands migrate.
    assert {"version", "egress", "profile", "bundles", "help", "commands"} <= names


def test_every_execute_key_resolves():
    for cmd in MIGRATED:
        assert cmd.execute in EXECUTORS, (
            f"/{cmd.name} names execute={cmd.execute!r} but no such key in "
            f"hermes_cli.slash_exec.EXECUTORS"
        )
        assert resolve_executor(cmd) is EXECUTORS[cmd.execute]


def test_unmigrated_commands_have_no_executor():
    for cmd in COMMAND_REGISTRY:
        if not cmd.execute:
            assert resolve_executor(cmd) is None
            assert run_execute(cmd, CommandContext()) is None


@pytest.mark.parametrize("cmd", MIGRATED, ids=lambda c: c.name)
def test_core_text_is_surface_invariant(cmd):
    """Fixed context ⇒ identical CommandReply text on every surface."""
    replies = []
    for surface in SURFACES:
        ctx = CommandContext(surface=surface, args="", options={"page_size": 20})
        reply = run_execute(cmd, ctx)
        assert isinstance(reply, CommandReply)
        assert isinstance(reply.text, str) and reply.text
        replies.append(reply.text)
    assert replies[0] == replies[1] == replies[2], (
        f"/{cmd.name} core text varies by surface — executors must not "
        f"branch on ctx.surface"
    )


def test_execute_command_helper_resolves_aliases():
    # /v is an alias of /version — the helper resolves through the registry.
    a = execute_command("version", CommandContext(surface="cli"))
    b = execute_command("v", CommandContext(surface="gateway"))
    assert a.text == b.text


def test_profile_options_override_process_values():
    reply = run_execute(
        resolve_command("profile"),
        CommandContext(options={"profile_name": "work", "home_display": "~/.hermes-work"}),
    )
    assert reply.data == {"profile": "work", "home": "~/.hermes-work"}
    assert reply.text == "Profile: work\nHome: ~/.hermes-work"


def test_commands_page_size_is_an_option_not_a_surface_branch():
    """Telegram's smaller page is parameterized — same option, same text."""
    cmd = resolve_command("commands")
    tg = run_execute(cmd, CommandContext(surface="gateway", options={"page_size": 15}))
    cli = run_execute(cmd, CommandContext(surface="cli", options={"page_size": 15}))
    assert tg.text == cli.text


