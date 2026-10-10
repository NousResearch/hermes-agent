"""Source contract: heavy synchronous construction never runs directly on the gateway event loop.

``AIAgent.__init__`` loads the context engine under a process-global load lock; while concurrent
worker turns hold it, construction takes tens of seconds. Session hygiene and ``/compress`` built
their throwaway ``AIAgent`` inline in a coroutine, so the loop stalled past Discord's ~41 s
heartbeat ACK window and the socket was closed. The unavailable-skill scan
(``_check_unavailable_skill``: rglob + read_text over every SKILL.md) is the same class.

This locks the class, not the sites: no ``async def`` in the gateway package may call
``AIAgent(...)`` or ``_check_unavailable_skill(...)`` directly. A call inside a nested
``def``/``lambda`` (the ``asyncio.to_thread`` payload) is fine; the walk stops at nested scopes.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
FILES = sorted((REPO / "gateway").glob("*.py"))
# Constructors / helpers measured blocking the loop. Extend when a new loop-block site is
# root-caused to a direct call.
BLOCKING_CALLEES = {"AIAgent", "_check_unavailable_skill"}


def _callee_name(node: ast.Call) -> str | None:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def _direct_calls_in_async_bodies(tree: ast.AST):
    """Yield (async_fn_name, lineno, callee) for direct calls in async bodies.

    Nested ``def``/``async def``/``lambda`` scopes are not descended: a call
    inside them is executed by whoever invokes that closure (to_thread), not by
    the coroutine itself.
    """
    class Walker(ast.NodeVisitor):
        def __init__(self):
            self.hits = []
            self._async_stack = []

        def visit_AsyncFunctionDef(self, node):
            self._async_stack.append(node.name)
            for stmt in node.body:
                self._scan(stmt)
            self._async_stack.pop()

        def _scan(self, node):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                if isinstance(node, ast.AsyncFunctionDef):
                    self.visit_AsyncFunctionDef(node)
                return
            if isinstance(node, ast.Call):
                name = _callee_name(node)
                if name in BLOCKING_CALLEES:
                    self.hits.append((self._async_stack[-1], node.lineno, name))
            for child in ast.iter_child_nodes(node):
                self._scan(child)

    w = Walker()
    w.visit(tree)
    return w.hits


@pytest.mark.parametrize("path", FILES, ids=[p.name for p in FILES])
def test_no_blocking_construction_directly_in_async_def(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits = _direct_calls_in_async_bodies(tree)
    assert not hits, (
        f"{path.name}: heavy sync call(s) directly on the event loop — wrap in "
        f"asyncio.to_thread (see module docstring): "
        + ", ".join(f"{fn}:{ln} {callee}(...)" for fn, ln, callee in hits)
    )


def test_walker_catches_a_direct_call_and_ignores_to_thread_payload():
    """Negative control: the contract must actually fire."""
    bad = ast.parse(
        "async def h():\n"
        "    a = AIAgent(model='x')\n"
        "    b = await asyncio.to_thread(lambda: AIAgent(model='y'))\n"
        "    def inner():\n"
        "        return _check_unavailable_skill('z')\n"
        "    return a, b, inner\n"
    )
    hits = _direct_calls_in_async_bodies(bad)
    assert hits == [("h", 2, "AIAgent")]