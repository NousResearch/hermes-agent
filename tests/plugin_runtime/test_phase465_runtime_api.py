"""Ownership contracts for Phase 4.6.5 public runtime helper extraction."""

from __future__ import annotations

import ast
from pathlib import Path


RUNTIME_HELPERS = {
    "invoke_hook",
    "ainvoke_hook",
    "render_system_prompt_sections",
    "invoke_middleware",
    "has_middleware",
    "has_hook",
    "iter_hook_callbacks",
    "get_plugin_context_engine",
    "get_plugin_command_handler",
    "get_plugin_commands",
    "get_plugin_auxiliary_tasks",
    "get_plugin_toolsets",
    "get_plugin_subscriptions",
    "unload_plugins",
}

HOST_POLICY_HELPERS = {
    "fire_pre_command_hook",
    "get_pre_tool_call_directive",
    "get_pre_tool_call_block_message",
    "resolve_pre_tool_block",
    "get_pre_verify_continue_message",
    "get_plugin_error_classification",
}


def _top_level_functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_runtime_api_is_canonical_helper_owner() -> None:
    import hermes_cli.plugins as plugin_api
    import plugin_runtime.api as runtime_api

    runtime_functions = _top_level_functions(Path(runtime_api.__file__))
    cli_functions = _top_level_functions(Path(plugin_api.__file__))

    assert RUNTIME_HELPERS <= runtime_functions
    assert RUNTIME_HELPERS.isdisjoint(cli_functions)


def test_cli_runtime_helper_exports_are_canonical_identities() -> None:
    import hermes_cli.plugins as plugin_api
    import plugin_runtime.api as runtime_api

    for name in RUNTIME_HELPERS:
        assert getattr(plugin_api, name) is getattr(runtime_api, name)


def test_runtime_api_has_no_cli_plugin_monolith_backedge() -> None:
    import plugin_runtime.api as runtime_api

    tree = ast.parse(Path(runtime_api.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module != "hermes_cli.plugins"
        elif isinstance(node, ast.Import):
            assert all(alias.name != "hermes_cli.plugins" for alias in node.names)


def test_host_policy_stays_out_of_runtime_api() -> None:
    import hermes_cli.plugins as plugin_api
    import plugin_runtime.api as runtime_api

    runtime_functions = _top_level_functions(Path(runtime_api.__file__))

    assert HOST_POLICY_HELPERS.isdisjoint(runtime_functions)
    assert all(hasattr(plugin_api, name) for name in HOST_POLICY_HELPERS)


def test_runtime_api_uses_runtime_lifecycle_directly() -> None:
    import plugin_runtime.api as runtime_api

    source = Path(runtime_api.__file__).read_text(encoding="utf-8")
    assert "from plugin_runtime.lifecycle import (" in source
    assert "delivery_manager" in source
    assert "ensure_plugins_discovered" in source
    assert "get_plugin_manager" in source
    assert "join_background_discovery" in source
