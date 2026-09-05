"""Characterization for the classic-CLI command binding seam."""

from types import SimpleNamespace

import pytest

from hermes_cli.classic_cli_command_bindings import (
    CLASSIC_CLI_COMMAND_BINDINGS,
    bind_classic_cli_dispatch,
    build_classic_cli_bindings,
    dispatch_classic_cli_command,
    normalize_classic_cli_invocation,
    resolve_classic_cli_command_binding,
)
from hermes_cli.commands import COMMAND_REGISTRY, CommandDef, resolve_command


def test_projection_is_immutable_unique_and_excludes_gateway_only_commands():
    assert isinstance(CLASSIC_CLI_COMMAND_BINDINGS, tuple)
    assert CLASSIC_CLI_COMMAND_BINDINGS

    ids = [binding.command_id.casefold() for binding in CLASSIC_CLI_COMMAND_BINDINGS]
    names = [
        binding.canonical_name.casefold()
        for binding in CLASSIC_CLI_COMMAND_BINDINGS
    ]
    assert len(ids) == len(set(ids))
    assert len(names) == len(set(names))

    projected = set(names)
    expected = {
        command.name.casefold()
        for command in COMMAND_REGISTRY
        if not command.gateway_only
    }
    assert projected == expected


def test_alias_and_canonical_name_resolve_to_the_same_binding():
    canonical = resolve_classic_cli_command_binding("branch")
    alias = resolve_classic_cli_command_binding("/fork")

    assert canonical is not None
    assert alias is canonical
    assert resolve_command("fork") is resolve_command("branch")


def test_invocation_preserves_entered_alias_and_argument_text():
    invocation = normalize_classic_cli_invocation("  /FoRk Mixed CASE --Flag  ")

    assert invocation is not None
    assert invocation.binding.canonical_name == "branch"
    assert invocation.entered_name == "FoRk"
    assert invocation.raw_arguments == "Mixed CASE --Flag"
    assert invocation.canonical_input == "/branch Mixed CASE --Flag"


def test_unknown_gateway_only_and_non_command_inputs_fail_closed():
    assert normalize_classic_cli_invocation("plain prompt text") is None
    assert normalize_classic_cli_invocation("/definitely-not-a-command") is None
    assert normalize_classic_cli_invocation("/approve session") is None
    assert resolve_classic_cli_command_binding("approve") is None


def test_duplicate_fallback_identity_fails_closed():
    first = CommandDef("duplicate", "first", "Test")
    second = CommandDef("duplicate", "second", "Test")

    with pytest.raises(ValueError, match="duplicate classic CLI command identity"):
        build_classic_cli_bindings((first, second))


def test_future_stable_identity_is_adopted_without_another_registry():
    command = SimpleNamespace(
        name="future-command",
        command_id="session.future",
        gateway_only=False,
        execute=None,
    )

    bindings = build_classic_cli_bindings((command,))  # type: ignore[arg-type]

    assert bindings[0].command_id == "session.future"
    assert bindings[0].canonical_name == "future-command"


def test_blank_future_identity_falls_back_to_canonical_name():
    command = SimpleNamespace(
        name="fallback-command",
        command_id="   ",
        gateway_only=False,
        execute=None,
    )

    bindings = build_classic_cli_bindings((command,))  # type: ignore[arg-type]

    assert bindings[0].command_id == "fallback-command"


def test_dispatch_uses_canonical_input_and_preserves_legacy_result():
    class StubCLI:
        seen: str | None = None

        def process_command(self, command: str) -> bool:
            self.seen = command
            return False

    cli = StubCLI()

    assert dispatch_classic_cli_command(cli, "/fork Mixed CASE") is False
    assert cli.seen == "/branch Mixed CASE"


def test_binding_returns_the_existing_process_command_owner():
    class StubCLI:
        def process_command(self, command: str) -> bool:
            return command == "/version"

    cli = StubCLI()
    dispatcher = bind_classic_cli_dispatch(cli, "version")

    assert dispatcher is not None
    assert dispatcher("/version") is True


def test_missing_dispatch_owner_fails_closed():
    assert bind_classic_cli_dispatch(object(), "version") is None
    assert dispatch_classic_cli_command(object(), "/version") is None
