"""HX015: a raw ``sqlite3.connect`` on the session store (registered in ``py_rules.CHECKERS``).

``hermes_cli.sqlite_safe_read.has_live_connection()`` / ``offline_file_access()`` decide when raw
byte I/O on a live ``state.db`` is safe, and they only see connections opened through
``connect_tracked`` (closing any raw fd on the file cancels the process's POSIX locks on it).

The target is judged textually, so the rule is a heuristic: a ``sqlite3.connect`` call is flagged
when its database argument mentions ``"state.db"`` in a string, a name or attribute containing
``state_db``, or a ``read_only_db_uri(...)`` call, or a local bound (in the same or an enclosing
scope) from such an expression. Limits: a state.db path that arrives through a parameter named
anything else (``path``, ``db``), a dict/attribute lookup, or another function's return value is
missed, and so is a URI built from a plain path (``f"file:{path}?mode=ro"``, ``path.as_uri()``) or
a connect reached through dynamic dispatch (``getattr(sqlite3, "connect")``, a ``connect`` passed
in as a callable). A ``read_only_db_uri`` over some other database (kanban) is flagged and takes a
waiver. The commonest spellings of the store are among the misses: ``DEFAULT_DB_PATH``,
``_default_db_path()``, ``session_db.db_path``. Those limits are why this rule is advisory; the
matcher stays textual on purpose.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator

from scripts.code_health.py_ast import _FUNCS, Ctx, _call_name

_CONNECT = "sqlite3.connect"
_STATE_FILE = "state.db"
_STATE_NAME = "state_db"
_STATE_URI_HELPERS = {"read_only_db_uri"}
_SCOPES = (*_FUNCS, ast.Lambda, ast.ClassDef)


def _own(scope: ast.AST) -> Iterator[ast.AST]:
    """Nodes of ``scope`` outside any nested def/lambda/class (those are scopes of their own)."""
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        node = stack.pop()
        yield node
        if not isinstance(node, _SCOPES):
            stack.extend(ast.iter_child_nodes(node))


def _names_state(expr: ast.AST, tainted: set[str]) -> bool:
    for node in ast.walk(expr):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and _STATE_FILE in node.value:
            return True
        if isinstance(node, ast.Name) and (_STATE_NAME in node.id.lower() or node.id in tainted):
            return True
        if isinstance(node, ast.Attribute) and _STATE_NAME in node.attr.lower():
            return True
        if isinstance(node, ast.Call) and _call_name(node).rsplit(".", 1)[-1] in _STATE_URI_HELPERS:
            return True
    return False


def _target(call: ast.Call) -> ast.expr | None:
    if call.args:
        return call.args[0]
    return next((kw.value for kw in call.keywords if kw.arg == "database"), None)


def _tainted(own: list[ast.AST], inherited: set[str]) -> set[str]:
    """Locals bound from a state.db expression, to a fixed point (``p = home / "state.db"``)."""
    pairs = [(t.id, node.value) for node in own if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr))
             and node.value is not None
             for t in (node.targets if isinstance(node, ast.Assign) else [node.target])
             if isinstance(t, ast.Name)]
    tainted = set(inherited)
    while True:
        grown = {name for name, value in pairs if name not in tainted and _names_state(value, tainted)}
        if not grown:
            return tainted
        tainted |= grown


def _visit(scope: ast.AST, inherited: set[str]) -> Iterator[int]:
    own = list(_own(scope))
    tainted = _tainted(own, inherited)
    for node in own:
        if isinstance(node, ast.Call) and _call_name(node) == _CONNECT:
            target = _target(node)
            if target is not None and _names_state(target, tainted):
                yield node.lineno
        if isinstance(node, _SCOPES):
            yield from _visit(node, tainted)


def untracked_state_connect(tree: ast.Module, ctx: Ctx) -> Iterable[int]:
    return _visit(tree, set())
