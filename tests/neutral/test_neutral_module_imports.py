"""The ``session_rules`` modules import no application package.

Each module may import the standard library, ``psutil``, the persistence contracts and the other ``session_rules`` modules, and nothing else.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

NEUTRAL_MODULES = tuple(sorted(str(p.relative_to(ROOT)) for p in (ROOT / "session_rules").glob("*.py")))

# The root modules a neutral module may import: the contracts and the rule modules themselves.
NEUTRAL_ROOTS = {"persistence", "session_rules"}


def _imports(tree: ast.Module):
    """``(module, names, at_module_level)`` for every import statement, ``import x`` giving ``(x, ())``."""
    top = {id(node) for node in tree.body}
    top |= {id(n) for node in tree.body if isinstance(node, ast.Try) for n in node.body + [h for handler in node.handlers for h in handler.body]}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, (), id(node) in top
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            yield node.module, tuple(alias.name for alias in node.names), id(node) in top


def _neutral(module: str, roots: set[str] = NEUTRAL_ROOTS) -> bool:
    root = module.partition(".")[0]
    return root in sys.stdlib_module_names or root in roots or root == "psutil"


@pytest.mark.parametrize("name", NEUTRAL_MODULES)
def test_a_neutral_module_imports_only_what_it_is_allowed(name):
    tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
    assert {entry for entry in _imports(tree) if not _neutral(entry[0])} == set()
