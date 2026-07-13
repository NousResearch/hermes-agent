"""Regression tests for module bindings used by the conversation loop."""

import ast
from pathlib import Path


def test_run_conversation_uses_module_time_binding() -> None:
    """A nested ``import time`` makes every earlier reference unbound."""
    source = Path("agent/conversation_loop.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_conversation"
    )

    local_imports = {
        alias.asname or alias.name.split(".", 1)[0]
        for node in ast.walk(function)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "time" not in local_imports
