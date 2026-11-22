"""HX018 / HX019: which regex module a file may import (registered in ``py_rules.CHECKERS``).

HX018: a stdlib ``re`` import in product code. Product code runs regexes on RE2 through
``import hermes_re as re``; the backtracking engine stays only in tests and in the files listed in
``stdlib_re_exempt.RE_EXEMPT`` (the rule's ``exclude``). Existing imports are grandfathered by the
ratchet until their file is migrated; a new one fails.

HX019: a ``hermes_re`` import in a file that must keep stdlib ``re`` (the rule's ``include``):
such a file can run where google-re2 is not importable, and there ``hermes_re`` cannot match.

Both rules judge the import statement, wherever it sits (module top, a function, an
``if TYPE_CHECKING:`` block), plus the two dynamic spellings with a literal module name.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator

from scripts.code_health.py_ast import Ctx, _call_name

# ``re`` and the modules of its implementation that code used to reach into.
_STDLIB_RE = ("re", "sre_parse", "sre_compile", "sre_constants")
_FACADE = ("hermes_re",)
# canonical_tree spells an imported ``import_module`` out as ``importlib.import_module``.
_DYNAMIC_IMPORTS = ("__import__", "importlib.import_module")


def _names(module: str | None, roots: tuple[str, ...]) -> bool:
    """``module`` is one of ``roots`` or a submodule of one (``re.x``), never a look-alike (``regex``)."""
    return module is not None and any(module == root or module.startswith(root + ".") for root in roots)


def _module_argument(call: ast.Call) -> ast.expr | None:
    """The module name of ``__import__`` / ``import_module``: first positional argument or ``name=``."""
    if call.args:
        return call.args[0]
    return next((k.value for k in call.keywords if k.arg == "name"), None)


def _imports(tree: ast.Module, roots: tuple[str, ...]) -> Iterator[int]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(_names(alias.name, roots) for alias in node.names):
                yield node.lineno
        elif isinstance(node, ast.ImportFrom):
            # level > 0 is a sibling module that happens to share the name (``from . import re``).
            if not node.level and _names(node.module, roots):
                yield node.lineno
        elif isinstance(node, ast.Call) and _call_name(node) in _DYNAMIC_IMPORTS:
            target = _module_argument(node)
            if isinstance(target, ast.Constant) and isinstance(target.value, str) and _names(target.value, roots):
                yield node.lineno


def stdlib_re_import(tree: ast.Module, ctx: Ctx) -> Iterable[int]:
    """``import re``, ``import re as X``, ``from re import ...``, ``import re.<x>``, and the same
    for ``sre_parse`` / ``sre_compile`` / ``sre_constants``."""
    return _imports(tree, _STDLIB_RE)


def facade_import(tree: ast.Module, ctx: Ctx) -> Iterable[int]:
    """Any import of ``hermes_re`` or one of its submodules."""
    return _imports(tree, _FACADE)
