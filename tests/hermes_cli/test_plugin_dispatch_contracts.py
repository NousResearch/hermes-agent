"""Ownership and compatibility contracts for plugin hook dispatch metadata."""

from __future__ import annotations

import ast
from pathlib import Path


def test_runtime_hook_contracts_are_coherent() -> None:
    from plugin_runtime.dispatch import SHELL_UNSUPPORTED_HOOKS, VALID_HOOKS

    assert {"pre_tool_call", "post_tool_call", "pre_llm_call"} <= VALID_HOOKS
    assert SHELL_UNSUPPORTED_HOOKS <= VALID_HOOKS
    assert "transform_api_error_classification" in SHELL_UNSUPPORTED_HOOKS


def test_runtime_middleware_contracts_are_canonical() -> None:
    import hermes_cli.middleware as middleware_api
    import hermes_cli.plugins as plugin_api
    import plugin_runtime.dispatch as runtime_dispatch

    assert runtime_dispatch.VALID_MIDDLEWARE == {
        "tool_request", "tool_execution", "llm_request", "llm_execution",
    }
    assert middleware_api.VALID_MIDDLEWARE is runtime_dispatch.VALID_MIDDLEWARE
    assert plugin_api.VALID_MIDDLEWARE is runtime_dispatch.VALID_MIDDLEWARE
    assert middleware_api.TOOL_REQUEST_MIDDLEWARE == runtime_dispatch.TOOL_REQUEST_MIDDLEWARE
    assert middleware_api.TOOL_EXECUTION_MIDDLEWARE == runtime_dispatch.TOOL_EXECUTION_MIDDLEWARE
    assert middleware_api.LLM_REQUEST_MIDDLEWARE == runtime_dispatch.LLM_REQUEST_MIDDLEWARE
    assert middleware_api.LLM_EXECUTION_MIDDLEWARE == runtime_dispatch.LLM_EXECUTION_MIDDLEWARE


def test_legacy_plugin_api_reexports_runtime_hook_contracts() -> None:
    import hermes_cli.plugins as plugin_api
    import plugin_runtime.dispatch as runtime_dispatch

    assert plugin_api.VALID_HOOKS is runtime_dispatch.VALID_HOOKS
    assert (
        plugin_api.SHELL_UNSUPPORTED_HOOKS
        is runtime_dispatch.SHELL_UNSUPPORTED_HOOKS
    )


def test_legacy_plugin_api_reexports_runtime_command_result_resolver() -> None:
    import hermes_cli.plugins as plugin_api
    import plugin_runtime.dispatch as runtime_dispatch

    assert (
        plugin_api.resolve_plugin_command_result
        is runtime_dispatch.resolve_plugin_command_result
    )


def test_cli_plugin_module_does_not_own_hook_contracts() -> None:
    import hermes_cli.plugins as plugin_api

    source_path = Path(plugin_api.__file__).resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    assigned_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assigned_names.add(node.target.id)
        elif isinstance(node, ast.Assign):
            assigned_names.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )

    assert {"VALID_HOOKS", "SHELL_UNSUPPORTED_HOOKS"}.isdisjoint(assigned_names)
