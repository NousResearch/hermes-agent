"""Phase 4.9 compatibility-facade cleanup gates."""

from __future__ import annotations

import ast
import logging
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FACADE = ROOT / "hermes_cli" / "plugins.py"
DEBUG = ROOT / "plugin_runtime" / "debug.py"
LIFECYCLE = ROOT / "plugin_runtime" / "lifecycle.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_cli_plugin_facade_defines_only_lazy_compat_behavior() -> None:
    tree = _tree(FACADE)

    functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    assigned = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    top_level_calls = [
        node
        for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    ]

    assert functions == {"__getattr__"}
    assert classes == set()
    assert assigned == {"_PLUGIN_COMPAT_LAZY"}
    assert top_level_calls == []


def test_cli_plugin_facade_has_no_runtime_debug_or_manager_state() -> None:
    tree = _tree(FACADE)
    assigned = {
        target.id
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        )
        if isinstance(target, ast.Name)
    }
    functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert {
        "_PLUGINS_DEBUG",
        "_DEBUG_HANDLER_INSTALLED",
        "_plugin_manager",
        "_plugin_managers_by_home",
        "_background_discovery_thread",
    }.isdisjoint(assigned)
    assert "_install_plugin_debug_handler" not in functions


def test_runtime_debug_owns_handler_installation_and_lifecycle_triggers_it(monkeypatch) -> None:
    import plugin_runtime.debug as debug

    assert debug.install_plugin_debug_handler.__module__ == "plugin_runtime.debug"

    lifecycle_tree = _tree(LIFECYCLE)
    assert "from plugin_runtime.debug import install_plugin_debug_handler" in LIFECYCLE.read_text(encoding="utf-8")
    manager_fn = next(
        node for node in lifecycle_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_plugin_manager"
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "install_plugin_debug_handler"
        for node in ast.walk(manager_fn)
    )

    logger = logging.getLogger("hermes_cli.plugins")
    original_handlers = list(logger.handlers)
    original_level = logger.level
    original_propagate = logger.propagate

    monkeypatch.setenv("HERMES_PLUGINS_DEBUG", "1")
    monkeypatch.setattr(debug, "_PLUGINS_DEBUG", False)
    monkeypatch.setattr(debug, "_DEBUG_HANDLER_INSTALLED", False)

    try:
        debug.install_plugin_debug_handler(force=True)
        added = [handler for handler in logger.handlers if handler not in original_handlers]
        assert len(added) == 1
        assert added[0].level == logging.DEBUG
        assert debug.plugin_debug_enabled() is True

        debug.install_plugin_debug_handler()
        assert [handler for handler in logger.handlers if handler not in original_handlers] == added
    finally:
        logger.handlers[:] = original_handlers
        logger.setLevel(original_level)
        logger.propagate = original_propagate
