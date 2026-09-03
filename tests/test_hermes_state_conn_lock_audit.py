"""Lock audit for every call on the shared writer connection (#99349).

``SessionDB._conn`` is opened with ``check_same_thread=False`` and shared
across threads (``AsyncSessionDB`` offloads every method via
``asyncio.to_thread``), so every *call* on it must hold ``self._lock``.
A lock-free ``self._conn.execute(...)`` — even a pure SELECT — can run
concurrently with ``close()`` deallocating the connection's pysqlite
statement cache, which segfaults the interpreter (observed in the field:
``_PyDict_GetItem_KnownHash`` via ``bounded_lru_cache_wrapper`` on one
thread while ``pysqlite_connection_close`` tears the cache down on
another). "Read-only" is not an exemption: the race is on the connection
object, not the database file.

Reads that must not contend on the writer lock go through
``SessionDB._read_ctx()`` instead — it borrows a pooled read connection
(exclusively checked out for the block) and its non-WAL fallback is the
writer connection *under* ``self._lock``.

This is an AST audit in the spirit of
``tests/gateway/test_async_session_db.py``: it fails on the next
``self._conn.<method>(...)`` call site added outside ``with self._lock:``.
"""

import ast
from pathlib import Path

# Functions allowed to touch self._conn without the lock: construction-time
# code that runs before the instance is ever shared with another thread.
_ALLOWED_UNLOCKED_FNS = frozenset({
    "__init__",
    "_connect_and_init",
    "_connect_and_init_with_lock_patience",
})


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _nearest_enclosing_fn(tree: ast.AST) -> dict:
    """Map id(node) -> name of the nearest enclosing function ("<module>"
    at module level). Nested defs override their parents."""
    enclosing: dict = {}

    def visit(node: ast.AST, current: str) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            current = node.name
        enclosing[id(node)] = current
        for child in ast.iter_child_nodes(node):
            visit(child, current)

    visit(tree, "<module>")
    return enclosing


def _unlocked_conn_calls(tree: ast.AST):
    """Return (lineno, enclosing_fn, method) for each self._conn.<m>(...)
    call that is not lexically inside a ``with self._lock:`` block."""
    locked_ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                ctx = item.context_expr
                if (
                    isinstance(ctx, ast.Attribute)
                    and ctx.attr == "_lock"
                    and isinstance(ctx.value, ast.Name)
                    and ctx.value.id == "self"
                ):
                    locked_ids.update(id(child) for child in ast.walk(node))

    enclosing = _nearest_enclosing_fn(tree)

    offending = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "_conn"
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "self"
        ):
            continue
        if id(node) in locked_ids:
            continue
        fn = enclosing.get(id(node), "<module>")
        if fn in _ALLOWED_UNLOCKED_FNS:
            continue
        offending.append((node.lineno, fn, func.attr))
    return offending


def test_every_conn_call_outside_construction_holds_the_lock():
    src = (_repo_root() / "hermes_state.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    offending = _unlocked_conn_calls(tree)
    assert not offending, (
        "self._conn.<method>() called without `with self._lock:` — this "
        "races SessionDB.close() inside pysqlite's statement cache and "
        "segfaults the process (#99349). Use `with self._read_ctx() as "
        "conn:` for reads, or take self._lock. Sites: "
        + ", ".join(
            f"line {lineno} in {fn}(): self._conn.{meth}(...)"
            for lineno, fn, meth in offending
        )
    )
