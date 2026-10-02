"""Ownership contracts for Phase 4.6.6 host-policy extraction."""

from __future__ import annotations

import ast
from pathlib import Path


POLICY_NAMES = {
    "fire_pre_command_hook",
    "_PreToolCallDirective",
    "set_thread_tool_whitelist",
    "clear_thread_tool_whitelist",
    "_get_pre_tool_call_directive_details",
    "get_pre_tool_call_directive",
    "get_pre_tool_call_block_message",
    "resolve_pre_tool_block",
    "_resolve_block_from_details",
    "_dispatch_pre_tool_call_hooks",
    "get_pre_verify_continue_message",
    "get_plugin_error_classification",
}


def _definitions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def test_host_policy_is_canonical_owner() -> None:
    import hermes_cli.plugin_policy as policy
    import hermes_cli.plugins as plugin_api

    assert POLICY_NAMES <= _definitions(Path(policy.__file__))
    assert POLICY_NAMES.isdisjoint(_definitions(Path(plugin_api.__file__)))


def test_cli_policy_exports_are_canonical_identities() -> None:
    import hermes_cli.plugin_policy as policy
    import hermes_cli.plugins as plugin_api

    for name in POLICY_NAMES:
        assert getattr(plugin_api, name) is getattr(policy, name)


def test_host_policy_has_no_plugin_monolith_backedge() -> None:
    import hermes_cli.plugin_policy as policy

    tree = ast.parse(Path(policy.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module != "hermes_cli.plugins"
        elif isinstance(node, ast.Import):
            assert all(alias.name != "hermes_cli.plugins" for alias in node.names)


def test_plugin_runtime_does_not_depend_on_host_policy() -> None:
    root = Path(__file__).resolve().parents[2] / "plugin_runtime"
    for path in root.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "hermes_cli.plugin_policy" not in source, path


def test_first_party_policy_consumers_use_canonical_owner() -> None:
    root = Path(__file__).resolve().parents[2]
    expected = {
        "agent/agent_runtime_helpers.py": "_dispatch_pre_tool_call_hooks",
        "agent/background_review.py": "set_thread_tool_whitelist",
        "agent/error_classifier.py": "get_plugin_error_classification",
        "agent/side_question.py": "set_thread_tool_whitelist",
        "agent/tool_executor.py": "_dispatch_pre_tool_call_hooks",
        "agent/turn_stop_gates.py": "get_pre_verify_continue_message",
        "cli.py": "fire_pre_command_hook",
        "gateway/run_inbound.py": "fire_pre_command_hook",
        "model_tools.py": "_dispatch_pre_tool_call_hooks",
    }
    for relative, symbol in expected.items():
        source = (root / relative).read_text(encoding="utf-8")
        assert "from hermes_cli.plugin_policy import" in source
        assert symbol in source
        assert f"from hermes_cli.plugins import {symbol}" not in source
