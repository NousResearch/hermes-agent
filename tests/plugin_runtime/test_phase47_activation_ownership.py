"""Ownership checks for Phase 4.7 step 1 activation extraction."""

from __future__ import annotations

import ast
from pathlib import Path


MOVED_FUNCTIONS = {
    "plugin_activation_summary",
    "activation_summaries",
    "find_activation",
    "load_and_go_live",
    "_go_live",
}


def _defined_functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_runtime_activation_is_canonical_owner() -> None:
    import hermes_cli.plugins_activation as edge
    import plugin_runtime.activation as activation

    runtime_defs = _defined_functions(Path(activation.__file__))
    edge_defs = _defined_functions(Path(edge.__file__))

    assert MOVED_FUNCTIONS <= runtime_defs
    assert MOVED_FUNCTIONS.isdisjoint(edge_defs)


def test_manager_consumes_runtime_activation_directly() -> None:
    import plugin_runtime.manager as manager

    source = Path(manager.__file__).read_text(encoding="utf-8")
    assert "from plugin_runtime.activation import activation_summaries" in source
    assert "from hermes_cli.plugins_activation import activation_summaries" not in source


def test_runtime_activation_does_not_import_cli_plugin_monolith() -> None:
    import plugin_runtime.activation as activation

    tree = ast.parse(Path(activation.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module != "hermes_cli.plugins"
        elif isinstance(node, ast.Import):
            assert all(alias.name != "hermes_cli.plugins" for alias in node.names)
