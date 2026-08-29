"""Pattern-C gate: pure-read SessionDB methods must not take the writer lock.

The gateway shares ONE SessionDB across every agent. ``self._lock`` guards
the single writer connection — any read-only query executed under it
convoys every concurrent turn's persistence behind that reader (Pattern C
of the 2026-08 perf triage; #90734 shipped the unlocked-reader subset,
this gate covers the locked-reader subset).

``_read_ctx()`` exists precisely for reads: WAL reader from a bounded
pool, no lock, with a byte-identical fallback to the locked writer when
WAL is off. Reads have no reason to hold the writer lock.

The gate parses ``hermes_state.py`` with ``ast`` and flags any method
that (a) opens ``with self._lock:`` and (b) runs ONLY read statements
(SELECT/PRAGMA-read) on ``self._conn`` inside it — i.e. a pure reader
convoying on the writer lock. Methods that write under the lock are the
lock's legitimate users and pass. New violations fail with the method
name and the fix (route through ``_read_ctx()``).

Deliberately NOT flagged:
- methods that INSERT/UPDATE/DELETE/REPLACE under the lock (writers);
- read-modify-write methods (the read is ordered against its own write);
- ``_read_ctx``'s own writer-fallback (``yield self._conn`` — no execute);
- SELECTs on ``conn``/other objects (already pooled readers).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_STATE_PY = Path(__file__).resolve().parents[2] / "hermes_state.py"

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

# Methods allowed to keep a pure-read body under the writer lock, each with
# the reason. Keep this list SHRINKING — never add to it without the same
# scrutiny a new blocking call would get.
_ALLOWED_LOCKED_READERS: dict[str, str] = {
    # _enter_fts_fail_open reads schema state under the lock as part of the
    # fail-open WRITE transition (the writes live in sibling statements the
    # simple statement scanner attributes to other calls).
    "_enter_fts_fail_open": "fail-open transition; lock orders vs FTS writes",
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


def _is_self_conn_execute(call: ast.Call) -> bool:
    f = call.func
    return (
        isinstance(f, ast.Attribute)
        and f.attr in ("execute", "executemany", "executescript")
        and isinstance(f.value, ast.Attribute)
        and f.value.attr == "_conn"
        and isinstance(f.value.value, ast.Name)
        and f.value.value.id == "self"
    )


def _is_self_lock_with(item: ast.withitem) -> bool:
    ctx = item.context_expr
    return (
        isinstance(ctx, ast.Attribute)
        and ctx.attr == "_lock"
        and isinstance(ctx.value, ast.Name)
        and ctx.value.id == "self"
    )


def _scan_locked_readers(state_py: "Path | None" = None) -> list[str]:
    target = state_py if state_py is not None else _STATE_PY
    tree = ast.parse(target.read_text(encoding="utf-8"))
    violations: list[str] = []

    session_db = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "SessionDB":
            session_db = node
            break
    assert session_db is not None, "SessionDB class not found"

    for method in session_db.body:
        if not isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(method):
            if not isinstance(node, ast.With):
                continue
            if not any(_is_self_lock_with(i) for i in node.items):
                continue
            reads, writes, unknown = 0, 0, 0
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call) and _is_self_conn_execute(inner):
                    word_full = _first_sql_text(inner)
                    word = word_full.split(None, 1)[0].upper() if word_full else None
                    if word is None:
                        unknown += 1
                    elif _PRAGMA_WRITE_RE.match(word_full or ""):
                        writes += 1
                    elif _WRITE_RE.match(word):
                        writes += 1
                    elif _READ_RE.match(word):
                        reads += 1
                    else:
                        unknown += 1
                # Method calls under the lock may write internally
                # (e.g. self._execute_write, cursor ops) — treat any
                # self.<something>() as potentially writing.
                elif isinstance(inner, ast.Call):
                    f = inner.func
                    if (
                        isinstance(f, ast.Attribute)
                        and isinstance(f.value, ast.Name)
                        and f.value.id == "self"
                        and (
                            "write" in f.attr
                            or "commit" in f.attr
                            or f.attr.startswith(("set_", "record_", "insert_",
                                                  "update_", "delete_", "clear_"))
                        )
                    ):
                        writes += 1
            if reads > 0 and writes == 0 and unknown == 0:
                if method.name not in _ALLOWED_LOCKED_READERS:
                    violations.append(
                        f"{method.name} (line {node.lineno}): pure-read "
                        f"body under `with self._lock:` — route through "
                        f"_read_ctx() instead"
                    )
    return violations


class TestNoPureReadersUnderWriterLock:
    def test_no_locked_pure_readers(self):
        violations = _scan_locked_readers()
        assert violations == [], (
            "Pure-read SessionDB methods holding the writer lock "
            "(Pattern C — every concurrent turn's persistence convoys "
            "behind these reads):\n  " + "\n  ".join(violations)
        )

    def test_gate_detects_a_locked_reader(self, tmp_path):
        """Sabotage self-check: the scanner must flag a synthetic violation."""
        sabotage = (
            "class SessionDB:\n"
            "    def innocent_writer(self):\n"
            "        with self._lock:\n"
            "            self._conn.execute(\"UPDATE t SET x = 1\")\n"
            "    def guilty_reader(self):\n"
            "        with self._lock:\n"
            "            return self._conn.execute(\"SELECT 1\").fetchone()\n"
        )
        p = tmp_path / "fake_state.py"
        p.write_text(sabotage, encoding="utf-8")
        violations = _scan_locked_readers(p)
        assert len(violations) == 1
        assert "guilty_reader" in violations[0]
        assert "innocent_writer" not in violations[0]
