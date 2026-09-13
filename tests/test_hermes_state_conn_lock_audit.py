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

``SessionDB`` is declared in ``hermes_state.py`` as ``class
SessionDB(<12 mixins>)`` — its methods live across ``hermes_state.py``
plus every one of those mixin files, so this audit scans all 13.
``test_state_files_match_sessiondb_bases`` re-derives that file list from
``hermes_state.py`` itself and fails loudly if a future split
adds/removes a mixin without a matching update here — this audit
previously scanned only ``hermes_state.py``, invisible to ~31 of the
~42 ``self._conn`` call sites once ``SessionDB``'s body was split into
12 mixins.
"""

import ast
from pathlib import Path

# Functions allowed to touch self._conn without holding self._lock, for one of two
# proven-safe reasons — keep this list SHRINKING, never add a name without tracing
# every one of its callers the way the ones below were traced:
#  (a) construction-time: runs before the instance is ever shared with another thread;
#  (b) callee-locked: the function has no caller that isn't itself inside self._lock.
_ALLOWED_UNLOCKED_FNS = frozenset({
    # (a) construction-time
    "__init__",
    "_connect_and_init",
    "_connect_and_init_with_lock_patience",
    "_init_schema",  # sole caller: _connect_and_init, above
    # (b) callee-locked (every caller traced and confirmed to hold self._lock)
    "_fts_table_exists",  # sole caller _present_fts_tables's 3 call sites all `with self._lock:`
    "_recover_stale_fts_locked",  # reached only via _init_schema (construction-time) or
                                   # retry_deferred_fts_recovery, which holds self._lock
    "_try_checkpoint",  # sole caller vacuum(), both call sites inside `with self._lock:`
})


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sessiondb_state_files() -> list[str]:
    """hermes_state.py plus every file SessionDB's actual class bases are
    imported from — independent ground truth (see
    test_state_files_match_sessiondb_bases), not a hand-maintained list."""
    root = _repo_root()
    tree = ast.parse((root / "hermes_state.py").read_text(encoding="utf-8"))
    base_to_file: dict[str, str] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("hermes_state_")
        ):
            for alias in node.names:
                base_to_file[alias.asname or alias.name] = f"{node.module}.py"

    session_db = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "SessionDB"
    )
    files = ["hermes_state.py"]
    for base in session_db.bases:
        if isinstance(base, ast.Name) and base.id in base_to_file:
            fname = base_to_file[base.id]
            if fname not in files:
                files.append(fname)
    return files


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
    root = _repo_root()
    offending: list[str] = []
    for filename in _sessiondb_state_files():
        src = (root / filename).read_text(encoding="utf-8")
        tree = ast.parse(src)
        for lineno, fn, meth in _unlocked_conn_calls(tree):
            offending.append(f"{filename}:{lineno} in {fn}(): self._conn.{meth}(...)")
    assert not offending, (
        "self._conn.<method>() called without `with self._lock:` — this "
        "races SessionDB.close() inside pysqlite's statement cache and "
        "segfaults the process (#99349). Use `with self._read_ctx() as "
        "conn:` for reads, or take self._lock. Sites: " + ", ".join(offending)
    )


def test_gate_detects_an_unlocked_call_in_a_mixin_file(tmp_path):
    """Sabotage self-check for the multi-file scope itself: a lock-free
    self._conn call planted in a MIXIN file (not hermes_state.py) must
    still be caught. Guards against the scan narrowing back to
    hermes_state.py alone — exactly how this audit's real blind spot
    happened (it scanned only hermes_state.py while ~31 of SessionDB's
    ~42 self._conn call sites lived in mixin files)."""
    sabotage = (
        "class FakeMixin:\n"
        "    def guilty_reader(self):\n"
        "        return self._conn.execute(\"SELECT 1\").fetchone()\n"
    )
    p = tmp_path / "fake_mixin.py"
    p.write_text(sabotage, encoding="utf-8")
    tree = ast.parse(p.read_text(encoding="utf-8"))
    offending = _unlocked_conn_calls(tree)
    assert any(
        fn == "guilty_reader" and meth == "execute" for _, fn, meth in offending
    ), offending


def test_state_files_include_every_mixin():
    """_sessiondb_state_files() must resolve to hermes_state.py PLUS every
    one of SessionDB's mixin bases, not silently collapse back to just
    hermes_state.py if the AST-based base-class/import discovery ever
    breaks (e.g. a mixin imported under an alias, or SessionDB's bases
    written in a shape the walk doesn't expect). A silent collapse here
    is exactly how this audit's real blind spot happened, just one layer
    up: it hardcoded a single file instead of deriving the list at all."""
    files = _sessiondb_state_files()
    assert files[0] == "hermes_state.py"
    assert len(files) >= 12, (
        "expected hermes_state.py plus at least 11 mixin files, got only "
        f"{files} -- base-class/import discovery in _sessiondb_state_files() "
        "may be broken, silently narrowing this audit back toward "
        "hermes_state.py alone"
    )
    assert len(files) == len(set(files)), f"duplicate entries: {files}"
