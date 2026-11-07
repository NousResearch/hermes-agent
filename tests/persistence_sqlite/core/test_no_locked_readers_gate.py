"""Pattern-C gate: pure-read store functions must not take the writer lock.

The gateway shares ONE ``StoreHolder`` per state.db across every agent.
``holder._lock`` guards the single writer connection — any read-only query
executed under it convoys every concurrent turn's persistence behind that
reader (Pattern C of the 2026-08 perf triage; #90734 shipped the
unlocked-reader subset, this gate covers the locked-reader subset).

``_read_ctx()`` exists precisely for reads: WAL reader from a bounded
pool, no lock, with a byte-identical fallback to the locked writer when
WAL is off. Reads have no reason to hold the writer lock.

The gate parses the SQLite backend's store modules with ``ast`` and flags
any function that (a) opens ``with self._lock:`` or ``with holder._lock:``
and (b) runs ONLY read statements (SELECT/PRAGMA-read) on that receiver's
``_conn`` inside it — i.e. a pure reader convoying on the writer lock.
Functions that write under the lock are the lock's legitimate users and
pass. New violations fail with the function name and the fix (route through
``_read_ctx()``).

The store's code lives in ``StoreHolder`` (``persistence/sqlite/_holder.py``,
receiver ``self``) and in module functions that take the holder as their
first argument (``holder``) spread over several modules, so
``_ALL_STATE_SOURCES`` lists each of them; a gate that opened one file would
never see a locked reader declared in another.

Deliberately NOT flagged:
- functions that INSERT/UPDATE/DELETE/REPLACE under the lock (writers);
- read-modify-write functions (the read is ordered against its own write);
- ``_read_ctx``'s own writer-fallback (``yield self._conn`` — no execute);
- SELECTs on ``conn``/other objects (already pooled readers).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BACKEND = _REPO_ROOT / "persistence" / "sqlite"
_STATE_PY = _BACKEND / "_holder.py"

# The holder's own class body lives in _holder.py; the rest of the store's
# code comes from these modules (see module docstring). Each entry is
# (source file, class name to scan in that file, or None for every function
# in it).
_ALL_STATE_SOURCES: list[tuple[Path, str | None]] = [
    (_STATE_PY, "StoreHolder"),
    (_BACKEND / "conversation" / "_store" / "core.py", None),
    (_BACKEND / "conversation" / "_store" / "search.py", None),
    (_BACKEND / "conversation" / "_store" / "portability.py", None),
    (_BACKEND / "_schema.py", None),
    (_BACKEND / "admin" / "_maintenance.py", None),
    (_BACKEND / "housekeeping" / "_portability.py", None),
]

# The names the store's code calls its holder by.
_RECEIVERS = frozenset({"self", "holder"})

_WRITE_RE = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|REPLACE|CREATE|DROP|ALTER|VACUUM|BEGIN|COMMIT|ANALYZE)\b",
    re.IGNORECASE,
)
# PRAGMA is read-only EXCEPT the checkpoint/optimize family, which mutates
# the database file and legitimately belongs on the writer connection.
_PRAGMA_WRITE_RE = re.compile(
    r"^\s*PRAGMA\s+(wal_checkpoint|optimize|incremental_vacuum|integrity_check)",
    re.IGNORECASE,
)
_READ_RE = re.compile(r"^\s*(SELECT|PRAGMA)\b", re.IGNORECASE)

# Functions allowed to keep a pure-read body under the writer lock, each with
# the reason. Keep this list SHRINKING — never add to it without the same
# scrutiny a new blocking call would get.
_ALLOWED_LOCKED_READERS: dict[str, str] = {
    # get_meta stays on the writer lock BY DESIGN (see its inline comment):
    # fts_rebuild_step reads rebuild progress before entering a write
    # transaction, and a pooled WAL reader sees only committed data — the
    # writer's own just-staged meta updates would be invisible to it.
    "get_meta": "read-your-writes: rebuild progress read before write txn",
}


def _first_sql_text(call: ast.Call) -> str | None:
    """Best-effort SQL text from an execute()'s first argument."""
    if not call.args:
        return None
    arg = call.args[0]
    text = None
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        text = arg.value
    elif isinstance(arg, ast.JoinedStr):
        parts = [
            v.value for v in arg.values
            if isinstance(v, ast.Constant) and isinstance(v.value, str)
        ]
        text = "".join(parts)
    if not text or not text.strip():
        return None
    return text.strip()


def _is_receiver_conn(node: ast.AST) -> bool:
    """``self._conn`` or ``holder._conn``."""
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "_conn"
        and isinstance(node.value, ast.Name)
        and node.value.id in _RECEIVERS
    )


def _is_self_conn_execute(call: ast.Call, aliases: set[str]) -> bool:
    """Match ``<receiver>._conn.execute*`` and ``<alias>.execute*`` where the
    alias was bound from ``<receiver>._conn`` (``conn = holder._conn``)."""
    f = call.func
    if not (
        isinstance(f, ast.Attribute)
        and f.attr in ("execute", "executemany", "executescript")
    ):
        return False
    target = f.value
    if _is_receiver_conn(target):
        return True
    return isinstance(target, ast.Name) and target.id in aliases


def _collect_conn_aliases(method: ast.AST) -> set[str]:
    """Names bound from ``<receiver>._conn`` anywhere in the function body."""
    aliases: set[str] = set()
    for node in ast.walk(method):
        if isinstance(node, ast.Assign) and _is_receiver_conn(node.value):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    aliases.add(t.id)
    return aliases


def _is_self_lock_with(item: ast.withitem) -> bool:
    ctx = item.context_expr
    return (
        isinstance(ctx, ast.Attribute)
        and ctx.attr == "_lock"
        and isinstance(ctx.value, ast.Name)
        and ctx.value.id in _RECEIVERS
    )


def _may_write(name: str) -> bool:
    return (
        "write" in name
        or "commit" in name
        or name.startswith(("set_", "record_", "insert_", "update_", "delete_", "clear_"))
    )


def _functions(tree: ast.Module, class_name: str | None, target: Path) -> list[ast.AST]:
    """The methods of *class_name*, or every function of the module (module level and in its classes)."""
    if class_name is not None:
        cls = None
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                cls = node
                break
        assert cls is not None, f"{class_name} class not found in {target}"
        return [m for m in cls.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
    found = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found.append(node)
        elif isinstance(node, ast.ClassDef):
            found += [m for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
    return found


def _scan_locked_readers(
    state_py: Path | None = None, class_name: str | None = None
) -> list[str]:
    target = state_py if state_py is not None else _STATE_PY
    tree = ast.parse(target.read_text(encoding="utf-8"))
    violations: list[str] = []

    for method in _functions(tree, class_name, target):
        aliases = _collect_conn_aliases(method)
        for node in ast.walk(method):
            if not isinstance(node, ast.With):
                continue
            if not any(_is_self_lock_with(i) for i in node.items):
                continue
            reads, writes, unknown = 0, 0, 0
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call) and _is_self_conn_execute(inner, aliases):
                    word_full = _first_sql_text(inner)
                    word = word_full.split(None, 1)[0].upper() if word_full else None
                    if word is None:
                        # SQL held in a variable or built f-string: the
                        # scanner cannot prove it reads. A lock block whose
                        # ONLY statements are unprovable is still flagged
                        # below — writers name their verbs in literals
                        # throughout these files, so opacity correlates with
                        # composed SELECTs, and silently skipping these is
                        # how 5 readers hid from the first version of this
                        # gate.
                        unknown += 1
                    elif _PRAGMA_WRITE_RE.match(word_full or "") or _WRITE_RE.match(word):
                        writes += 1
                    elif _READ_RE.match(word):
                        reads += 1
                    else:
                        unknown += 1
                # Calls under the lock may write internally (e.g.
                # holder._execute_write, cursor ops) — treat any
                # <receiver>.<something>() or store function <something>(holder)
                # with a writing name as potentially writing.
                elif isinstance(inner, ast.Call):
                    f = inner.func
                    if (
                        isinstance(f, ast.Attribute)
                        and isinstance(f.value, ast.Name)
                        and f.value.id in _RECEIVERS
                        and _may_write(f.attr)
                    ) or (isinstance(f, ast.Name) and _may_write(f.id)):
                        writes += 1
            if writes == 0 and (reads > 0 or unknown > 0):
                if method.name not in _ALLOWED_LOCKED_READERS:
                    kind = "pure-read" if unknown == 0 else "no-proven-write"
                    violations.append(
                        f"{method.name} (line {node.lineno}): {kind} "
                        f"body under `with {_lock_receiver(node)}._lock:` — route through "
                        f"_read_ctx() instead (or add a justified "
                        f"allowlist entry)"
                    )
    return violations


def _lock_receiver(node: ast.With) -> str:
    return next(i.context_expr.value.id for i in node.items if _is_self_lock_with(i))


def _scan_all_state_sources() -> list[str]:
    """Run ``_scan_locked_readers`` over every file that holds the store's
    code — the ``StoreHolder`` class body in ``_holder.py`` plus each module of
    holder functions (see module docstring). Violations are prefixed with
    their source filename since functions can share names across modules.
    """
    violations: list[str] = []
    for path, class_name in _ALL_STATE_SOURCES:
        for v in _scan_locked_readers(path, class_name):
            violations.append(f"{path.relative_to(_BACKEND)}: {v}")
    return violations


class TestNoPureReadersUnderWriterLock:
    def test_no_locked_pure_readers(self):
        violations = _scan_all_state_sources()
        assert violations == [], (
            "Pure-read store functions holding the writer lock "
            "(Pattern C — every concurrent turn's persistence convoys "
            "behind these reads):\n  " + "\n  ".join(violations)
        )

    def test_gate_detects_a_locked_reader(self, tmp_path):
        """Sabotage self-check: the scanner must flag a synthetic violation."""
        sabotage = (
            "class StoreHolder:\n"
            "    def innocent_writer(self):\n"
            "        with self._lock:\n"
            "            self._conn.execute(\"UPDATE t SET x = 1\")\n"
            "    def guilty_reader(self):\n"
            "        with self._lock:\n"
            "            return self._conn.execute(\"SELECT 1\").fetchone()\n"
            "    def guilty_alias_reader(self):\n"
            "        with self._lock:\n"
            "            conn = self._conn\n"
            "            return conn.execute(\"SELECT 2\").fetchone()\n"
            "    def guilty_variable_sql(self, query):\n"
            "        with self._lock:\n"
            "            return self._conn.execute(query).fetchall()\n"
            "    def innocent_variable_writer(self, query):\n"
            "        with self._lock:\n"
            "            self._conn.execute(query)\n"
            "            self._conn.execute(\"UPDATE t SET x = 2\")\n"
        )
        p = tmp_path / "fake_state.py"
        p.write_text(sabotage, encoding="utf-8")
        violations = _scan_locked_readers(p)
        flagged = {v.split(" ")[0] for v in violations}
        assert flagged == {
            "guilty_reader", "guilty_alias_reader", "guilty_variable_sql"
        }, violations

    def test_scan_all_state_sources_visits_every_mixin_file(self, tmp_path):
        """Sabotage self-check for the multi-file scope itself: a locked
        reader planted in a module of holder functions (not _holder.py) must
        still be caught. Guards against the gate's scope silently narrowing
        back to one file — exactly how the real 2026-08 gap (9 locked readers
        across three files, invisible to the single-file scanner) happened.
        """
        mixin_sabotage = (
            "def guilty_mixin_reader(holder):\n"
            "    with holder._lock:\n"
            "        return holder._conn.execute(\"SELECT 1\").fetchone()\n"
        )
        p = tmp_path / "fake_mixin.py"
        p.write_text(mixin_sabotage, encoding="utf-8")

        violations = [
            f"{p.name}: {v}" for v in _scan_locked_readers(p)
        ]
        assert any("guilty_mixin_reader" in v for v in violations), violations
