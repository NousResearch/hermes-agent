"""HX018: a ``utils/`` module reaching first-party product code (``py_rules.CHECKERS``).

``utils`` is importable from every package only while it pulls none of the product in. A reach
is an import statement at any depth, ``importlib.import_module``/``__import__`` of a literal name,
or a ``sys.modules`` lookup of one: the last reaches a module with no import statement at all.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator

from scripts.code_health.py_ast import Ctx, _call_name, _dotted

# First-party modules ``utils`` may build on. ``hermes_constants`` reaches ``agent`` and ``hermes_cli``
# inside a few functions; ``utils`` calls only its home helpers.
ALLOWED = frozenset({"utils", "hermes_constants", "hermes_platform", "hermes_yaml"})
_IMPORT_CALLS = {"importlib.import_module", "__import__"}


def _literal(node: ast.AST | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _reached(node: ast.AST) -> Iterator[str]:
    """Absolute module names ``node`` reaches by name; relative imports stay inside ``utils``."""
    if isinstance(node, ast.Import):
        yield from (alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        yield node.module
    elif isinstance(node, ast.Call) and node.args:
        name = _call_name(node)
        if name in _IMPORT_CALLS or name == "sys.modules.get":
            yield from filter(None, [_literal(node.args[0])])
    elif isinstance(node, ast.Subscript) and _dotted(node.value) == "sys.modules":
        yield from filter(None, [_literal(node.slice)])
    elif isinstance(node, ast.Compare) and isinstance(node.ops[0], (ast.In, ast.NotIn)):
        if _dotted(node.comparators[0]) == "sys.modules":
            yield from filter(None, [_literal(node.left)])


def product_reach(tree: ast.Module, ctx: Ctx) -> Iterable[int]:
    for node in ast.walk(tree):
        for name in _reached(node):
            top = name.split(".", 1)[0]
            if top in ctx.first_party and top not in ALLOWED:
                yield node.lineno
