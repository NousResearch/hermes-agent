"""AST helpers shared by the HX checkers (``py_rules`` and its ``py_rules_*`` siblings).

A leaf module: it imports nothing from the engine, so every checker module can use it without a
cycle.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass, field

_FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)


@dataclass
class Ctx:
    known_env: set[str] = field(default_factory=set)


def _dotted(node: ast.AST) -> str:
    """``os.environ.get`` for an Attribute chain, ``""`` for anything else."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    if isinstance(node, ast.Call):
        inner = _dotted(node.func)
        return ".".join([f"{inner}()", *reversed(parts)]) if inner else ""
    return ""


def _call_name(call: ast.Call) -> str:
    return _dotted(call.func)


def _deferred_parts(node: ast.AST) -> list[ast.AST] | None:
    """For a node whose evaluation defers part of itself, the parts that run NOW; else None.

    A lambda/def runs only its defaults and decorators when evaluated; a generator expression
    runs only its first iterable (the element, conditions and later loops run on consumption).
    Comprehensions and class bodies run immediately, so they are not deferred.
    """
    if isinstance(node, ast.Lambda):
        return [d for d in (*node.args.defaults, *node.args.kw_defaults) if d is not None]
    if isinstance(node, _FUNCS):
        defaults = [d for d in (*node.args.defaults, *node.args.kw_defaults) if d is not None]
        return [*node.decorator_list, *defaults]
    if isinstance(node, ast.GeneratorExp):
        return [node.generators[0].iter]
    return None


def _eager(node: ast.AST) -> Iterator[ast.AST]:
    """Every node evaluated when ``node`` is evaluated (or a statement executes), root included."""
    stack = [node]
    while stack:
        current = stack.pop()
        parts = _deferred_parts(current)
        if parts is not None:
            stack.extend(parts)
            continue
        yield current
        stack.extend(ast.iter_child_nodes(current))
