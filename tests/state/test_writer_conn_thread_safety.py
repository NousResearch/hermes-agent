"""Cross-thread races on the shared writer connection (2026-08-20 incident).

``SessionDB`` shares ONE writer connection (``check_same_thread=False``)
guarded by ``self._lock``.  A handful of read-only methods executed
statements on that same connection object WITHOUT taking the lock
(``get_compression_lock_holder``, ``list_pending_handoffs``,
``get_handoff_state``, and the no-op fast path of
``clear_session_activity_labels``).  When one of those SELECTs raced a
turn-boundary ``append_messages_batch`` on the same connection, CPython's
sqlite3 layer raised a bare ``SystemError`` ("<Connection> returned NULL
without setting an exception") — which is NOT a ``sqlite3.Error``, so it
escaped ``_execute_write``'s entire retry net and destroyed the user's
turn as ``session_persistence_failed``.

Two independent layers are asserted here:

1. The unlocked readers now route through ``_read_ctx()`` (per-thread
   read-only connections under WAL), so hammering them concurrently with
   turn-shaped batch writes must produce ZERO errors on either side.
2. Exactly-once containment: a bare ``SystemError`` inside the transaction
   callback is never replayed, while post-commit maintenance failures are
   logged without invalidating the already-durable write.
"""

import sqlite3
import threading
import time

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(SessionDB, "_WRITE_PATIENCE_S", 2.0)
    monkeypatch.setattr(SessionDB, "_WRITE_RETRY_MIN_S", 0.001)
    monkeypatch.setattr(SessionDB, "_WRITE_RETRY_MAX_S", 0.005)
    d = SessionDB(db_path=tmp_path / "state.db")
    yield d
    d.close()


class TestConcurrentReadersDoNotRaceTheWriter:
    """Layer 1: the formerly-unlocked readers, hammered against batch writes.

    Before the fix this reproduced the production ``SystemError`` within a
    few seconds on every run (each reader independently); after routing the
    readers through ``_read_ctx()`` the writer and readers touch different
    connection objects and the race is structurally gone.
    """

    READER_METHODS = (
        "get_compression_lock_holder",
        "list_pending_handoffs",
        "get_handoff_state",
        "clear_session_activity_labels",
    )

    def _run_race(self, db, reader_fn, duration_s=3.0):
        sid = db.create_session("race-sess", "test")
        errors = []
        stop = threading.Event()

        def writer():
            n = 0
            while not stop.is_set():
                try:
                    db.append_messages_batch(sid, [
                        {"role": "user", "content": "u%d" % n},
                        {"role": "assistant", "content": "a%d" % n},
                        {"role": "tool", "content": "t%d" % n,
                         "tool_name": "x", "tool_call_id": "c%d" % n},
                    ])
                    n += 1
                except Exception as exc:  # noqa: BLE001 — the assertion IS the catch
                    errors.append(("writer", type(exc).__name__, str(exc)))
                    stop.set()
                    return

        def reader():
            while not stop.is_set():
                try:
                    reader_fn(db, sid)
                except Exception as exc:  # noqa: BLE001
                    errors.append(("reader", type(exc).__name__, str(exc)))
                    stop.set()
                    return

        threads = [threading.Thread(target=writer)] + [
            threading.Thread(target=reader) for _ in range(3)
        ]
        for t in threads:
            t.start()
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline and not stop.is_set():
            time.sleep(0.05)
        stop.set()
        for t in threads:
            t.join(timeout=10)
        return errors

    def test_compression_lock_holder_reads_race_free(self, db):
        errors = self._run_race(
            db, lambda d, sid: d.get_compression_lock_holder(sid)
        )
        assert errors == []

    def test_pending_handoff_reads_race_free(self, db):
        errors = self._run_race(
            db, lambda d, sid: (d.list_pending_handoffs(),
                                d.get_handoff_state(sid))
        )
        assert errors == []

    def test_activity_label_noop_fast_path_race_free(self, db):
        errors = self._run_race(
            db, lambda d, sid: d.clear_session_activity_labels(sid)
        )
        assert errors == []

    def test_unlocked_writer_conn_use_is_enumerated(self):
        """Sibling-sweep tripwire: no method may touch ``self._conn``
        outside ``with self._lock`` (or the __init__/close lifecycle, which
        runs before/after any concurrent access is possible).

        This is the AST sweep that found the four incident sites, frozen as
        a test so a future convenience read can't quietly reintroduce the
        class.
        """
        import ast
        import inspect

        import hermes_state as hs

        src = inspect.getsource(hs)
        tree = ast.parse(src)

        ALLOWED_FUNCS = {
            # Lifecycle: run before the instance is shared / after readers
            # are drained. Not reachable concurrently with writers.
            "__init__", "_open_writer", "_connect_and_init",
            "_connect_and_init_with_lock_patience", "close",
        }

        def is_lock_with(node):
            if isinstance(node, ast.With):
                for item in node.items:
                    ctx = item.context_expr
                    if (isinstance(ctx, ast.Attribute)
                            and ctx.attr == "_lock"
                            and isinstance(ctx.value, ast.Name)
                            and ctx.value.id == "self"):
                        return True
            return False

        violations = []

        class Sweep(ast.NodeVisitor):
            def __init__(self):
                self.lock_depth = 0
                self.func_stack = []

            @staticmethod
            def _is_conn_attr(node):
                return (isinstance(node, ast.Attribute)
                        and isinstance(node.value, ast.Name)
                        and node.value.id == "self"
                        and node.attr == "_conn")

            def _flag(self, node):
                fn = self.func_stack[-1] if self.func_stack else "<module>"
                if fn not in ALLOWED_FUNCS:
                    violations.append((node.lineno, fn))

            def generic_visit(self, node):
                locked = is_lock_with(node)
                is_func = isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef)
                )
                if locked:
                    self.lock_depth += 1
                if is_func:
                    self.func_stack.append(node.name)
                if isinstance(node, ast.Call) and self.lock_depth == 0:
                    # A method call ON the connection (self._conn.execute(...))
                    func = node.func
                    if (isinstance(func, ast.Attribute)
                            and self._is_conn_attr(func.value)):
                        self._flag(func.value)
                    # ...or the connection handed to a helper that will
                    # execute on it (e.g. _collect_delegate_child_ids(self._conn, ...)).
                    for arg in list(node.args) + [k.value for k in node.keywords]:
                        if self._is_conn_attr(arg):
                            self._flag(arg)
                super().generic_visit(node)
                if locked:
                    self.lock_depth -= 1
                if is_func:
                    self.func_stack.pop()

        Sweep().visit(tree)
        assert violations == [], (
            "self._conn used outside 'with self._lock' in: %r — route reads "
            "through _read_ctx() (or take the lock); an unlocked statement "
            "on the shared writer connection races concurrent writers and "
            "raises SystemError, destroying the turn as "
            "session_persistence_failed" % (violations,)
        )


class TestSystemErrorTransactionBoundary:
    """A bare SystemError must never replay an ambiguous write."""

    def test_matching_error_inside_callback_is_not_replayed(self, db):
        calls = {"n": 0}

        def broken(_conn):
            calls["n"] += 1
            raise SystemError(
                "<TrackedConnection object at 0x0> returned NULL "
                "without setting an exception"
            )

        with pytest.raises(SystemError, match="returned NULL"):
            db._execute_write(broken)
        assert calls["n"] == 1

    def test_post_commit_maintenance_error_does_not_replay_message(
        self, db, monkeypatch
    ):
        db.create_session("s1", "cli")
        before = db._write_count
        maintenance_calls = {"n": 0}

        def fail_once(*, max_pages):
            maintenance_calls["n"] += 1
            if maintenance_calls["n"] == 1:
                raise SystemError(
                    "<TrackedConnection object at 0x0> returned NULL "
                    "without setting an exception"
                )
            return 0

        monkeypatch.setattr(db, "_FTS_MERGE_EVERY_N_WRITES", 1)
        monkeypatch.setattr(db, "_merge_fts_incrementally", fail_once)

        message_id = db.append_message("s1", "user", "exactly-once")
        matching = [
            row for row in db.get_messages("s1")
            if row["content"] == "exactly-once"
        ]

        assert isinstance(message_id, int)
        assert len(matching) == 1
        assert db.get_session("s1")["message_count"] == 1
        assert db._write_count == before + 1
        assert maintenance_calls["n"] == 1
