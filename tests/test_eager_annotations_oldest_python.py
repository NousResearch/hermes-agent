"""Annotations evaluated at definition time must work on the oldest supported Python (3.11).

Before 3.13 ``threading.Lock`` / ``threading.RLock`` are factory functions, not types, so an
unquoted ``threading.Lock | None`` annotation raises ``TypeError`` the moment its module is
imported. CI runs 3.14, where the same line imports fine: ``tools/mcp_oauth.py`` shipped such an
annotation, and on 3.11/3.12 every MCP failure then surfaced as that ``TypeError`` from the
auth-error classifier instead of reaching the session-expired / outcome-uncertain recovery.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SKIP = {".venv", "venv", "node_modules", ".worktrees", ".build", "website", "__pycache__"}
_FACTORIES = {"Lock", "RLock"}


def _defers_annotations(tree: ast.Module) -> bool:
    return any(isinstance(n, ast.ImportFrom) and n.module == "__future__"
               and any(a.name == "annotations" for a in n.names) for n in tree.body)


def _eager_annotations(tree: ast.Module):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.args
            params = [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg]
            yield from (p.annotation for p in params if p is not None and p.annotation is not None)
            if node.returns is not None:
                yield node.returns
        elif isinstance(node, ast.AnnAssign):
            yield node.annotation


def _names_lock_factory(annotation: ast.expr) -> bool:
    return any((isinstance(n, ast.Attribute) and n.attr in _FACTORIES
                and isinstance(n.value, ast.Name) and n.value.id == "threading")
               for n in ast.walk(annotation))


def test_no_union_with_a_lock_factory_in_an_eagerly_evaluated_annotation():
    offenders = []
    for path in _REPO.rglob("*.py"):
        if _SKIP & set(path.relative_to(_REPO).parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if _defers_annotations(tree):
            continue
        offenders += [f"{path.relative_to(_REPO)}:{ann.lineno}: {ast.unparse(ann)}"
                      for ann in _eager_annotations(tree)
                      if isinstance(ann, ast.BinOp) and _names_lock_factory(ann)]
    assert not offenders, "TypeError at import on Python < 3.13 (quote it):\n" + "\n".join(offenders)
