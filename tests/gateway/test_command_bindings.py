from __future__ import annotations

from dataclasses import replace

import pytest

from gateway.command_bindings import (
    GATEWAY_COMMAND_BINDINGS,
    bind_gateway_handler,
    build_gateway_bindings,
    resolve_gateway_command_binding,
)
from gateway.slash_commands import GatewaySlashCommandsMixin
from hermes_cli.commands import COMMAND_REGISTRY, resolve_command


def test_aliases_resolve_to_the_same_gateway_binding() -> None:
    for command in COMMAND_REGISTRY:
        canonical = resolve_gateway_command_binding(command.name)
        if canonical is None:
            continue
        for alias in command.aliases:
            assert resolve_gateway_command_binding(alias) is canonical
            assert resolve_gateway_command_binding(f"/{alias}") is canonical
            assert resolve_gateway_command_binding(alias.upper()) is canonical


def test_cli_only_commands_do_not_gain_gateway_bindings() -> None:
    assert resolve_gateway_command_binding("clear") is None
    assert resolve_gateway_command_binding("/history") is None


def test_irregular_gateway_handler_names_are_explicit() -> None:
    new = resolve_gateway_command_binding("new")
    reset = resolve_gateway_command_binding("reset")
    sethome = resolve_gateway_command_binding("sethome")

    assert new is not None
    assert reset is new
    assert new.handler_name == "_handle_reset_command"
    assert sethome is not None
    assert sethome.handler_name == "_handle_set_home_command"


def test_ordinary_handler_names_are_derived_from_canonical_name() -> None:
    usage = resolve_gateway_command_binding("usage")
    reload_mcp = resolve_gateway_command_binding("reload-mcp")

    assert usage is not None
    assert usage.handler_name == "_handle_usage_command"
    assert reload_mcp is not None
    assert reload_mcp.handler_name == "_handle_reload_mcp_command"


def test_binding_projection_has_one_casefolded_semantic_identity() -> None:
    identities = [binding.command_id.casefold() for binding in GATEWAY_COMMAND_BINDINGS]
    assert len(identities) == len(set(identities))


def test_duplicate_semantic_identity_fails_closed() -> None:
    source = resolve_command("usage")
    assert source is not None
    duplicate = replace(source, name="USAGE")

    with pytest.raises(ValueError, match="duplicate gateway command identity"):
        build_gateway_bindings((source, duplicate))


def test_gateway_binding_resolves_real_mixin_handlers_without_prompt_fallback() -> None:
    runner = object.__new__(GatewaySlashCommandsMixin)

    for token in ("new", "reset", "usage", "status", "reload-mcp", "sethome"):
        binding = resolve_gateway_command_binding(token)
        assert binding is not None
        handler = bind_gateway_handler(runner, token)
        assert callable(handler), (token, binding.handler_name)

    assert bind_gateway_handler(runner, "definitely-not-a-command") is None


def test_shared_executor_metadata_stays_attached_to_the_canonical_binding() -> None:
    shared = [command for command in COMMAND_REGISTRY if command.execute]
    assert shared, "expected current main to expose at least one shared command executor"

    for command in shared:
        binding = resolve_gateway_command_binding(command.name)
        if binding is not None:
            assert binding.shared_execute == command.execute
