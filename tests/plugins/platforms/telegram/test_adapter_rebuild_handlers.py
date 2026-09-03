"""A transient-init rebuild must restore plugin handlers, not just core ones.

When ``Application.initialize()`` hits a transient TLS reset, the adapter
discards the half-built application and rebuilds it from the same token/config.
The rebuild branch re-registered only the core handlers, so every PTB handler a
plugin contributed through ``ctx.register_telegram_handler`` was lost for the
life of the process: the bot came up healthy and silently stopped answering
plugin commands.

Normal startup wires both, in a fixed order — plugin handlers first, because PTB
dispatches the first matching handler per group and core handlers would
otherwise shadow every pattern-scoped plugin handler.

These tests read the adapter's AST rather than driving a real ``Application``:
the rebuild lives inside a retry loop around live network initialisation, and
the property under test is structural — which registration calls that branch
makes, and in what order.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ADAPTER = (Path(__file__).resolve().parents[4]
           / "plugins" / "platforms" / "telegram" / "adapter.py")

WIRE_PLUGINS = "_wire_plugin_handlers"
REGISTER_CORE = "_register_handlers"


def _rebuild_branch() -> ast.If:
    tree = ast.parse(ADAPTER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test_src = ast.dump(node.test)
            if "rebuild_app" in test_src and "_attempt" in test_src:
                return node
    pytest.fail("rebuild branch not found; adapter structure changed")


def _self_calls(node: ast.AST) -> list[str]:
    return [c.func.attr for c in ast.walk(node)
            if isinstance(c, ast.Call)
            and isinstance(c.func, ast.Attribute)
            and isinstance(c.func.value, ast.Name)
            and c.func.value.id == "self"]


def test_the_rebuild_branch_rewires_plugin_handlers():
    assert WIRE_PLUGINS in _self_calls(_rebuild_branch())


def test_the_rebuild_branch_still_registers_core_handlers():
    assert REGISTER_CORE in _self_calls(_rebuild_branch())


def test_plugin_handlers_are_wired_before_core_handlers():
    calls = _self_calls(_rebuild_branch())
    assert calls.index(WIRE_PLUGINS) < calls.index(REGISTER_CORE)


def test_rebuild_registers_everything_normal_startup_does():
    """Stated as a relation between the two sites, not a frozen list.

    Adding a third registration call to startup fails here until the rebuild
    path is updated too — which is exactly how this bug arose.
    """
    source = ADAPTER.read_text(encoding="utf-8")
    startup = {name for name in (WIRE_PLUGINS, REGISTER_CORE)
               if f"self.{name}(self._app)" in source}
    assert set(_self_calls(_rebuild_branch())) & startup == startup
