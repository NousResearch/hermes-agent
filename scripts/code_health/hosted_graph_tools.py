"""Built-in tool discovery, for the hosted import graph.

``model_tools`` calls ``tools.registry.discover_builtin_tools()`` at import time, which imports
every built-in tool module that registers a tool at module level. Those imports are not
statements the graph walker can see, so the selection is recomputed here over the measured tree
(a revision's blobs, not the files on disk), with the same candidates and the same AST predicate
as ``tools.registry._tool_module_candidates`` / ``_module_registers_tools``. Those take paths on
disk and the registry module is not importable from the lint without loading the tool stack's
dependencies, so the predicate is mirrored; a test holds the two to the same answer on the
real tree.
"""

from __future__ import annotations

import ast
import warnings

DISCOVERING_MODULE = "model_tools.py"
# discover_builtin_tools() never imports these itself (the registry, its package, MCP).
_SKIPPED = {"__init__.py", "registry.py", "mcp_tool.py"}


def _is_register_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
        return False
    func = node.value.func
    return (isinstance(func, ast.Attribute) and func.attr == "register"
            and isinstance(func.value, ast.Name) and func.value.id == "registry")


def registers_tools(source: str) -> bool:
    """The module body (or a module-level ``for``) calls ``registry.register(...)``."""
    if "registry" not in source or "register" not in source:
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(source)
    except SyntaxError:
        return False
    return any(_is_register_call(stmt)
               or (isinstance(stmt, ast.For) and any(_is_register_call(s) for s in stmt.body))
               for stmt in tree.body)


def _candidate(path: str, files: set[str]) -> bool:
    """``tools/<name>.py`` or ``tools/<pkg>/tool.py`` whose package has an ``__init__``."""
    parts = path.split("/")
    if len(parts) == 2:
        return parts[1].endswith(".py") and parts[1] not in _SKIPPED
    return len(parts) == 3 and parts[2] == "tool.py" and f"tools/{parts[1]}/__init__.py" in files


def discovered_tool_modules(files: set[str], texts: dict[str, str]) -> list[str]:
    """Paths of the tool modules discovery imports, given the tree's ``.py`` *files* and the
    *texts* of (at least) its ``tools/`` modules."""
    return sorted(p for p in files
                  if p.startswith("tools/") and _candidate(p, files) and registers_tools(texts.get(p, "")))
