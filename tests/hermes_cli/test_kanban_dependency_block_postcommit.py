"""Dependency-block lifecycle-hook post-commit timing (R-1).

``_fire_kanban_lifecycle_hook`` documents itself as being called by the
claim/complete/block transitions AFTER their write txn has committed, and
``VALID_HOOKS`` repeats that promise for ``kanban_task_blocked``: the observer
always sees durable board state and can never hold the SQLite write lock.

The ``kind="dependency"`` branch of ``block_task`` was the one lifecycle
dispatch site that fired from *inside* ``with write_txn(conn):``. These tests
pin the documented behaviour for that branch, and pin the surrounding
behaviour (failure boundaries, sibling routing, kwargs, business effects) that
must not change with it.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.plugins import OBSERVER_SCHEMA_VERSION, get_plugin_manager


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


class _Hooks:
    """Register callbacks on the live plugin registry and record their calls."""

    def __init__(self, mgr) -> None:
        self._mgr = mgr
        self.calls: list[tuple[str, dict]] = []

    def add(self, hook: str, fn) -> None:
        self._mgr._hooks.setdefault(hook, []).append(fn)

    def record(self, hook: str, also=None) -> None:
        def _cb(**kw):
            self.calls.append((hook, kw))
            if also is not None:
                also(**kw)

        self.add(hook, _cb)

    def blocked(self) -> list[dict]:
        return [kw for name, kw in self.calls if name == "kanban_task_blocked"]


@pytest.fixture
def hooks():
    """Yield a recorder bound to the plugin manager's hook registry.

    Patches ``_hooks`` directly (the registry ``invoke_hook`` reads) and
    restores it afterwards, matching ``test_kanban_lifecycle_hooks.py``.
    """
    mgr = get_plugin_manager()
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    try:
        yield _Hooks(mgr)
    finally:
        mgr._hooks = saved


def _running_task(conn, title: str = "t") -> str:
    """Create a task and drive it to ``running`` so ``block_task`` can act."""
    tid = kb.create_task(conn, title=title, assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None
    return tid


# ---------------------------------------------------------------------------
# R1N-T01 — the callback must not observe an open originating transaction
# ---------------------------------------------------------------------------


def test_dependency_callback_sees_origin_txn_closed(kanban_home, hooks):
    conn = kb.connect()
    try:
        tid = _running_task(conn)
        seen: list[bool] = []
        hooks.record(
            "kanban_task_blocked",
            also=lambda **kw: seen.append(conn.in_transaction),
        )
        result = kb.block_task(
            conn, tid, reason="waiting on parent", kind="dependency",
        )
    finally:
        conn.close()

    assert result is True
    assert seen == [False], (
        "kanban_task_blocked fired while the originating write transaction "
        f"was still open (in_transaction={seen})"
    )


# ---------------------------------------------------------------------------
# R1N-T02 — an independent connection must already see the durable result
# ---------------------------------------------------------------------------


def test_dependency_callback_sees_durable_state_from_second_connection(
    kanban_home, hooks,
):
    """A separate connection reads the committed transition during the hook.

    ``invoke_hook`` swallows callback exceptions, so the callback only
    *records* what the second connection saw; the assertions run in the test
    body. A callback that blew up therefore shows up as an empty recording,
    never as a silent pass.
    """
    conn = kb.connect()
    db_path = kb.kanban_db_path()
    observed: list[dict] = []

    try:
        tid = _running_task(conn)

        def _peek_from_second_connection(**kw):
            # Deliberately a raw sqlite3 connection to the same file rather
            # than kb.connect(): no shared handle, no shared transaction
            # state, nothing cached from the writer.
            other = sqlite3.connect(str(db_path), timeout=5.0)
            try:
                other.row_factory = sqlite3.Row
                row = other.execute(
                    "SELECT status, block_kind FROM tasks WHERE id = ?", (tid,),
                ).fetchone()
                waits = other.execute(
                    "SELECT COUNT(*) FROM task_events "
                    "WHERE task_id = ? AND kind = 'dependency_wait'",
                    (tid,),
                ).fetchone()[0]
                observed.append(
                    {
                        "distinct_connection": other is not conn,
                        "status": row["status"] if row else None,
                        "block_kind": row["block_kind"] if row else None,
                        "dependency_wait_rows": waits,
                    }
                )
            finally:
                other.close()

        hooks.record("kanban_task_blocked", also=_peek_from_second_connection)
        result = kb.block_task(
            conn, tid, reason="waiting on parent", kind="dependency",
        )
    finally:
        conn.close()

    assert result is True
    assert observed == [
        {
            "distinct_connection": True,
            "status": "todo",
            "block_kind": "dependency",
            "dependency_wait_rows": 1,
        }
    ], (
        "a second connection did not observe the committed dependency block "
        f"while kanban_task_blocked was running (observed={observed})"
    )


# ---------------------------------------------------------------------------
# R1N-T03 — the SQLite write lock must already be released during the callback
# ---------------------------------------------------------------------------


#: Busy timeout for the competing writer in R1N-T03. Tight on purpose: if the
#: originating transaction were still open, SQLite gives up after this and
#: raises "database is locked" instead of the test hanging. It is a failure
#: bound, never the thing that makes the test pass — an unheld lock is
#: acquired immediately.
_LOCK_ACQUIRE_TIMEOUT_S = 0.25


def test_dependency_callback_leaves_write_lock_available(kanban_home, hooks):
    """A competing writer can take BEGIN IMMEDIATE while the hook runs.

    Asserts lock acquisition objectively (``in_transaction`` flips true after
    BEGIN IMMEDIATE, false again after ROLLBACK) rather than timing anything.
    """
    conn = kb.connect()
    db_path = kb.kanban_db_path()
    attempts: list[dict] = []

    try:
        tid = _running_task(conn)

        def _try_acquire_write_lock(**kw):
            # isolation_level=None: no implicit transaction management from the
            # sqlite3 module, so BEGIN IMMEDIATE here is the real thing and
            # in_transaction reports SQLite's own state.
            other = sqlite3.connect(
                str(db_path),
                timeout=_LOCK_ACQUIRE_TIMEOUT_S,
                isolation_level=None,
            )
            try:
                try:
                    other.execute("BEGIN IMMEDIATE")
                except sqlite3.OperationalError as exc:
                    attempts.append(
                        {
                            "acquired": False,
                            "released": None,
                            "error": str(exc),
                        }
                    )
                    return
                acquired = other.in_transaction
                other.execute("ROLLBACK")
                attempts.append(
                    {
                        "acquired": acquired,
                        "released": not other.in_transaction,
                        "error": None,
                    }
                )
            finally:
                other.close()

        hooks.record("kanban_task_blocked", also=_try_acquire_write_lock)
        result = kb.block_task(
            conn, tid, reason="waiting on parent", kind="dependency",
        )
    finally:
        conn.close()

    assert result is True
    assert attempts == [
        {"acquired": True, "released": True, "error": None}
    ], (
        "a competing writer could not take BEGIN IMMEDIATE within "
        f"{_LOCK_ACQUIRE_TIMEOUT_S}s while kanban_task_blocked was running, so "
        f"the originating transaction still held the write lock ({attempts})"
    )


# ---------------------------------------------------------------------------
# R1N-T04 — a body exception rolls everything back and dispatches nothing
# ---------------------------------------------------------------------------


class _InjectedBodyFailure(RuntimeError):
    """Raised from inside the dependency transaction body by R1N-T04."""


def test_dependency_body_exception_rolls_back_and_fires_no_callback(
    kanban_home, hooks, monkeypatch,
):
    """An exception inside the txn body: nothing durable, nothing dispatched.

    The injection point is ``_append_event`` returning for the branch's own
    ``dependency_wait`` row — the last statement of the dependency body. By
    then the UPDATE, the run closure and the event insert have all executed,
    so a genuine rollback has real work to undo; and it is still strictly
    inside ``with write_txn(conn):``, so COMMIT (R1N-T05) and the post-commit
    file-length invariant (R1N-T06) are never reached and are not what this
    test exercises.
    """
    conn = kb.connect()
    db_path = kb.kanban_db_path()

    try:
        tid = _running_task(conn)

        # Snapshot the pre-block run pointer so rollback can be checked
        # against a concrete value rather than "something non-null".
        run_id_before = conn.execute(
            "SELECT current_run_id FROM tasks WHERE id = ?", (tid,),
        ).fetchone()["current_run_id"]
        assert run_id_before is not None

        hooks.record("kanban_task_blocked")

        real_append_event = kb._append_event
        appended: list[str] = []

        def _append_then_fail(conn_, task_id_, kind_, *args, **kwargs):
            real_append_event(conn_, task_id_, kind_, *args, **kwargs)
            appended.append(kind_)
            if kind_ == "dependency_wait":
                raise _InjectedBodyFailure("injected inside dependency txn body")

        monkeypatch.setattr(kb, "_append_event", _append_then_fail)

        with pytest.raises(_InjectedBodyFailure):
            kb.block_task(
                conn, tid, reason="waiting on parent", kind="dependency",
            )

        # The injection actually landed where intended.
        assert appended == ["dependency_wait"]
        # write_txn rolled back rather than leaving the connection poisoned.
        assert conn.in_transaction is False
    finally:
        conn.close()

    assert hooks.blocked() == [], (
        "kanban_task_blocked was dispatched even though the transaction body "
        f"raised and rolled back ({hooks.blocked()})"
    )

    # Durable state, read fresh: no transition, no event, no run closure.
    durable = sqlite3.connect(str(db_path), timeout=5.0)
    try:
        durable.row_factory = sqlite3.Row
        task_row = durable.execute(
            "SELECT status, block_kind, current_run_id FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
        wait_rows = durable.execute(
            "SELECT COUNT(*) FROM task_events "
            "WHERE task_id = ? AND kind = 'dependency_wait'",
            (tid,),
        ).fetchone()[0]
        run_row = durable.execute(
            "SELECT status, outcome, ended_at FROM task_runs WHERE id = ?",
            (run_id_before,),
        ).fetchone()
    finally:
        durable.close()

    assert task_row["status"] == "running"
    assert task_row["block_kind"] is None
    assert task_row["current_run_id"] == run_id_before
    assert wait_rows == 0
    assert run_row["status"] == "running"
    assert run_row["outcome"] is None
    assert run_row["ended_at"] is None


# ---------------------------------------------------------------------------
# R1N-T05 — a terminal COMMIT failure dispatches nothing and leaves no txn
# ---------------------------------------------------------------------------


class _InjectedCommitFailure(RuntimeError):
    """Raised in place of a terminal COMMIT failure by R1N-T05."""


def test_dependency_commit_failure_fires_no_callback(
    kanban_home, hooks, monkeypatch,
):
    """COMMIT fails terminally: nothing durable, nothing dispatched.

    The seam is ``_execute_boundary_with_retry`` — the one function
    ``write_txn`` routes both of its boundary statements through. Only
    ``COMMIT`` is failed; ``BEGIN IMMEDIATE`` is delegated to the real
    implementation, so the transaction genuinely opens and the mutation
    genuinely runs before the boundary blows up. The post-commit file-length
    invariant (R1N-T06) is a different function and is asserted here to have
    never been reached.
    """
    conn = kb.connect()
    db_path = kb.kanban_db_path()

    try:
        tid = _running_task(conn)

        run_id_before = conn.execute(
            "SELECT current_run_id FROM tasks WHERE id = ?", (tid,),
        ).fetchone()["current_run_id"]
        assert run_id_before is not None

        hooks.record("kanban_task_blocked")

        real_boundary = kb._execute_boundary_with_retry
        boundary_calls: list[str] = []
        invariant_calls: list[object] = []

        def _fail_commit_only(conn_, sql):
            boundary_calls.append(sql)
            if sql.strip().upper().startswith("COMMIT"):
                raise _InjectedCommitFailure("injected terminal COMMIT failure")
            return real_boundary(conn_, sql)

        real_invariant = kb._check_file_length_invariant

        def _count_invariant(conn_):
            invariant_calls.append(conn_)
            return real_invariant(conn_)

        monkeypatch.setattr(kb, "_execute_boundary_with_retry", _fail_commit_only)
        monkeypatch.setattr(kb, "_check_file_length_invariant", _count_invariant)

        with pytest.raises(_InjectedCommitFailure):
            kb.block_task(
                conn, tid, reason="waiting on parent", kind="dependency",
            )

        # The injection hit COMMIT, and only COMMIT: BEGIN IMMEDIATE ran for
        # real, and the post-commit invariant was never reached (that is
        # R1N-T06's boundary, not this one).
        assert [c.strip().upper() for c in boundary_calls] == [
            "BEGIN IMMEDIATE",
            "COMMIT",
        ]
        assert invariant_calls == []

        # write_txn rolled back, so the connection is closed and reusable.
        assert conn.in_transaction is False
        conn.execute("BEGIN IMMEDIATE")
        reacquired = conn.in_transaction
        conn.execute("ROLLBACK")
        assert reacquired is True
    finally:
        conn.close()

    assert hooks.blocked() == [], (
        "kanban_task_blocked was dispatched even though COMMIT failed "
        f"terminally ({hooks.blocked()})"
    )

    durable = sqlite3.connect(str(db_path), timeout=5.0)
    try:
        durable.row_factory = sqlite3.Row
        task_row = durable.execute(
            "SELECT status, block_kind, current_run_id FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
        wait_rows = durable.execute(
            "SELECT COUNT(*) FROM task_events "
            "WHERE task_id = ? AND kind = 'dependency_wait'",
            (tid,),
        ).fetchone()[0]
        run_row = durable.execute(
            "SELECT status, outcome, ended_at FROM task_runs WHERE id = ?",
            (run_id_before,),
        ).fetchone()
    finally:
        durable.close()

    assert task_row["status"] == "running"
    assert task_row["block_kind"] is None
    assert task_row["current_run_id"] == run_id_before
    assert wait_rows == 0
    assert run_row["status"] == "running"
    assert run_row["outcome"] is None
    assert run_row["ended_at"] is None


# ---------------------------------------------------------------------------
# R1N-T06 — a post-commit invariant failure still dispatches nothing
# ---------------------------------------------------------------------------


class _InjectedInvariantFailure(RuntimeError):
    """Raised from the post-commit file-length invariant by R1N-T06."""


def test_dependency_post_commit_invariant_failure_fires_no_callback(
    kanban_home, hooks, monkeypatch,
):
    """COMMIT succeeded, the post-commit invariant raised: zero callbacks.

    Dispatch is gated on the *full* ``write_txn`` wrapper exiting normally,
    not on COMMIT alone. ``_check_file_length_invariant`` runs after COMMIT
    and outside write_txn's rollback arm, so this leaves a genuinely committed
    row with no callback. Packet §6 row 4 states that window is pre-existing,
    deliberate, and must not be closed — so this test asserts the durable row
    *is* there and still demands zero callbacks.
    """
    conn = kb.connect()
    db_path = kb.kanban_db_path()

    try:
        tid = _running_task(conn)

        run_id_before = conn.execute(
            "SELECT current_run_id FROM tasks WHERE id = ?", (tid,),
        ).fetchone()["current_run_id"]
        assert run_id_before is not None

        hooks.record("kanban_task_blocked")

        real_boundary = kb._execute_boundary_with_retry
        boundary_calls: list[str] = []

        def _record_boundary(conn_, sql):
            # Pass-through: both boundaries really run, COMMIT included.
            result = real_boundary(conn_, sql)
            boundary_calls.append(sql)
            return result

        invariant_calls: list[object] = []

        def _fail_invariant(conn_):
            invariant_calls.append(conn_)
            raise _InjectedInvariantFailure(
                "injected post-commit invariant failure"
            )

        monkeypatch.setattr(kb, "_execute_boundary_with_retry", _record_boundary)
        monkeypatch.setattr(kb, "_check_file_length_invariant", _fail_invariant)

        with pytest.raises(_InjectedInvariantFailure):
            kb.block_task(
                conn, tid, reason="waiting on parent", kind="dependency",
            )

        # COMMIT was reached and completed — a COMMIT failure is R1N-T05's
        # mode, not this one. The invariant is the only thing that failed.
        assert [c.strip().upper() for c in boundary_calls] == [
            "BEGIN IMMEDIATE",
            "COMMIT",
        ]
        assert len(invariant_calls) == 1

        # COMMIT closed the transaction; no rollback arm runs after it.
        assert conn.in_transaction is False
    finally:
        conn.close()

    assert hooks.blocked() == [], (
        "kanban_task_blocked was dispatched even though the post-commit "
        f"invariant raised before write_txn exited normally ({hooks.blocked()})"
    )

    # Tolerated and asserted: the mutation IS durable — COMMIT already ran.
    durable = sqlite3.connect(str(db_path), timeout=5.0)
    try:
        durable.row_factory = sqlite3.Row
        task_row = durable.execute(
            "SELECT status, block_kind, current_run_id FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
        wait_rows = durable.execute(
            "SELECT COUNT(*) FROM task_events "
            "WHERE task_id = ? AND kind = 'dependency_wait'",
            (tid,),
        ).fetchone()[0]
        run_row = durable.execute(
            "SELECT status, outcome, ended_at FROM task_runs WHERE id = ?",
            (run_id_before,),
        ).fetchone()
    finally:
        durable.close()

    assert task_row["status"] == "todo"
    assert task_row["block_kind"] == "dependency"
    assert task_row["current_run_id"] is None
    assert wait_rows == 1
    assert run_row["status"] == "blocked"
    assert run_row["outcome"] == "blocked"
    assert run_row["ended_at"] is not None


# ---------------------------------------------------------------------------
# R1N-T07 — a raising callback is isolated: fail-open, no rollback, no spread
# ---------------------------------------------------------------------------


class _InjectedCallbackFailure(RuntimeError):
    """Raised by the misbehaving observer registered in R1N-T07."""


def test_dependency_raising_callback_is_isolated(kanban_home, hooks):
    """A misbehaving observer cannot break or undo the board transition.

    The raising callback is registered BEFORE the recording one, so the
    recorder firing is positive evidence that one bad observer does not
    suppress the others. Deliberately asserts nothing about timing —
    R1N-T01 owns that.
    """
    conn = kb.connect()
    db_path = kb.kanban_db_path()
    raised: list[str] = []

    try:
        tid = _running_task(conn)

        def _boom(**kw):
            raised.append(kw.get("task_id"))
            raise _InjectedCallbackFailure("observer exploded")

        hooks.add("kanban_task_blocked", _boom)
        hooks.record("kanban_task_blocked")

        # No pytest.raises: the callback's exception must not escape
        # _fire_kanban_lifecycle_hook, so this call simply returns.
        result = kb.block_task(
            conn, tid, reason="waiting on parent", kind="dependency",
        )
        assert conn.in_transaction is False
    finally:
        conn.close()

    assert result is True
    # The raising observer ran exactly once, and did not stop the next one.
    assert raised == [tid]
    assert len(hooks.blocked()) == 1

    # The committed mutation is untouched: not rolled back, not retried.
    durable = sqlite3.connect(str(db_path), timeout=5.0)
    try:
        durable.row_factory = sqlite3.Row
        task_row = durable.execute(
            "SELECT status, block_kind, current_run_id FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
        wait_rows = durable.execute(
            "SELECT COUNT(*) FROM task_events "
            "WHERE task_id = ? AND kind = 'dependency_wait'",
            (tid,),
        ).fetchone()[0]
        run_row = durable.execute(
            "SELECT status, outcome, ended_at FROM task_runs "
            "WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()
    finally:
        durable.close()

    assert task_row["status"] == "todo"
    assert task_row["block_kind"] == "dependency"
    assert task_row["current_run_id"] is None
    assert wait_rows == 1
    assert run_row["status"] == "blocked"
    assert run_row["outcome"] == "blocked"
    assert run_row["ended_at"] is not None


# ---------------------------------------------------------------------------
# R1N-T08 (task-not-found) — shared-preamble False return, no dispatch
# ---------------------------------------------------------------------------


_MISSING_TASK_ID = "no-such-task-id"


def test_block_unknown_task_returns_false_and_fires_nothing(kanban_home, hooks):
    """An unknown task id returns False and dispatches no callback.

    This path is NOT dependency-branch-local. ``block_task`` decides it in its
    shared preamble — the initial task SELECT returns no row and it returns
    False — *before* the ``kind == "dependency"`` test is ever evaluated, so it
    is common to every block kind and lies outside the R-1 edit range. R-1
    leaves it untouched; this test exists to prove that.

    Both a dependency and a non-dependency kind are exercised for exactly that
    reason: identical behaviour on both is what makes it shared rather than
    dependency-local.
    """
    conn = kb.connect()
    db_path = kb.kanban_db_path()

    try:
        # A real, claimed task that the failed calls must not disturb.
        control_tid = _running_task(conn, title="control")
        control_before = dict(
            conn.execute(
                "SELECT status, block_kind, block_recurrences, current_run_id "
                "FROM tasks WHERE id = ?",
                (control_tid,),
            ).fetchone()
        )
        events_before = conn.execute(
            "SELECT COUNT(*) FROM task_events"
        ).fetchone()[0]
        runs_before = conn.execute(
            "SELECT COUNT(*) FROM task_runs"
        ).fetchone()[0]

        hooks.record("kanban_task_blocked")

        results = {
            "dependency": kb.block_task(
                conn, _MISSING_TASK_ID, reason="waiting on parent",
                kind="dependency",
            ),
            "needs_input": kb.block_task(
                conn, _MISSING_TASK_ID, reason="needs a human",
                kind="needs_input",
            ),
        }

        assert conn.in_transaction is False

        control_after = dict(
            conn.execute(
                "SELECT status, block_kind, block_recurrences, current_run_id "
                "FROM tasks WHERE id = ?",
                (control_tid,),
            ).fetchone()
        )
        events_after = conn.execute(
            "SELECT COUNT(*) FROM task_events"
        ).fetchone()[0]
        runs_after = conn.execute("SELECT COUNT(*) FROM task_runs").fetchone()[0]
    finally:
        conn.close()

    assert results == {"dependency": False, "needs_input": False}
    assert hooks.blocked() == [], (
        "kanban_task_blocked was dispatched for a task id that does not exist "
        f"({hooks.blocked()})"
    )

    # Nothing was created for the missing id, and nothing else moved.
    assert control_after == control_before
    assert events_after == events_before
    assert runs_after == runs_before

    durable = sqlite3.connect(str(db_path), timeout=5.0)
    try:
        assert durable.execute(
            "SELECT COUNT(*) FROM tasks WHERE id = ?", (_MISSING_TASK_ID,),
        ).fetchone()[0] == 0
        assert durable.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ?",
            (_MISSING_TASK_ID,),
        ).fetchone()[0] == 0
        assert durable.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id = ?",
            (_MISSING_TASK_ID,),
        ).fetchone()[0] == 0
    finally:
        durable.close()


# ---------------------------------------------------------------------------
# R1N-T08 (CAS / state miss) — dependency-branch-local False return
# ---------------------------------------------------------------------------


def _task_snapshot(conn, task_id: str) -> dict:
    """Everything block_task could have touched for one task."""
    task = conn.execute(
        "SELECT status, block_kind, block_recurrences, current_run_id, "
        "claim_lock, claim_expires, worker_pid FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    events = conn.execute(
        "SELECT id, kind, run_id FROM task_events WHERE task_id = ? ORDER BY id",
        (task_id,),
    ).fetchall()
    runs = conn.execute(
        "SELECT id, status, outcome, ended_at FROM task_runs "
        "WHERE task_id = ? ORDER BY id",
        (task_id,),
    ).fetchall()
    return {
        "task": dict(task) if task is not None else None,
        "events": [tuple(row) for row in events],
        "runs": [tuple(row) for row in runs],
    }


def _setup_run_id_mismatch(conn):
    """Claimed and running, but the caller's expected_run_id is stale."""
    tid = _running_task(conn, title="cas-mismatch")
    current = conn.execute(
        "SELECT current_run_id FROM tasks WHERE id = ?", (tid,),
    ).fetchone()["current_run_id"]
    assert current is not None
    return tid, {"expected_run_id": int(current) + 1000}


def _setup_status_not_blockable(conn):
    """Exists, but sits in a status the UPDATE's WHERE clause excludes."""
    tid = kb.create_task(conn, title="not-blockable", triage=True)
    return tid, {}


@pytest.mark.parametrize(
    "setup",
    [_setup_run_id_mismatch, _setup_status_not_blockable],
    ids=["expected_run_id_mismatch", "status_outside_running_ready"],
)
def test_dependency_cas_miss_returns_false_and_fires_nothing(
    kanban_home, hooks, setup,
):
    """A dependency block whose UPDATE matches no row: False, no dispatch.

    Unlike the missing-id case (which ``block_task`` decides in its shared
    preamble before the ``kind`` test), this is the dependency-branch-local
    ``cur.rowcount != 1`` guard. R-1 preserves it unchanged, including its
    return from inside the transaction — it dispatches no callback, so there
    is nothing for a post-commit boundary to protect.
    """
    conn = kb.connect()
    db_path = kb.kanban_db_path()

    try:
        tid, extra_kwargs = setup(conn)
        before = _task_snapshot(conn, tid)

        # The precondition each case actually relies on.
        if "expected_run_id" in extra_kwargs:
            assert before["task"]["status"] == "running"
            assert extra_kwargs["expected_run_id"] != before["task"]["current_run_id"]
        else:
            assert before["task"]["status"] not in ("running", "ready")

        hooks.record("kanban_task_blocked")

        result = kb.block_task(
            conn, tid, reason="waiting on parent", kind="dependency",
            **extra_kwargs,
        )

        assert conn.in_transaction is False
    finally:
        conn.close()

    assert result is False
    assert hooks.blocked() == [], (
        "kanban_task_blocked was dispatched even though the dependency UPDATE "
        f"matched no row ({hooks.blocked()})"
    )

    durable = sqlite3.connect(str(db_path), timeout=5.0)
    try:
        durable.row_factory = sqlite3.Row
        after = _task_snapshot(durable, tid)
    finally:
        durable.close()

    assert after == before


# ---------------------------------------------------------------------------
# R1N-T09 — the ordinary (truly-blocked) sibling branch is unchanged
# ---------------------------------------------------------------------------


def test_ordinary_block_parity(kanban_home, hooks):
    """``kind="needs_input"`` still routes to ``blocked`` and dispatches once.

    The sibling branch already dispatched after the ``with`` block, so this is
    a preservation test: R-1 re-indented this code and must not have altered
    it. Callback kwargs are only spot-checked here (``task_id``); R1N-T11 owns
    exact kwarg parity.
    """
    # A single block must land in ``blocked``, not ``triage`` — otherwise this
    # test would be exercising the loop breaker (R1N-T10) instead.
    assert kb.BLOCK_RECURRENCE_LIMIT > 1

    conn = kb.connect()
    db_path = kb.kanban_db_path()
    seen_in_transaction: list[bool] = []

    try:
        tid = _running_task(conn, title="ordinary")
        run_id_before = conn.execute(
            "SELECT current_run_id FROM tasks WHERE id = ?", (tid,),
        ).fetchone()["current_run_id"]
        assert run_id_before is not None

        hooks.record(
            "kanban_task_blocked",
            also=lambda **kw: seen_in_transaction.append(conn.in_transaction),
        )

        result = kb.block_task(
            conn, tid, reason="needs a human", kind="needs_input",
        )
    finally:
        conn.close()

    assert result is True
    # Exactly one dispatch, and it happened after the transaction closed.
    fired = hooks.blocked()
    assert len(fired) == 1
    assert fired[0]["task_id"] == tid
    assert seen_in_transaction == [False]

    durable = sqlite3.connect(str(db_path), timeout=5.0)
    try:
        durable.row_factory = sqlite3.Row
        task_row = durable.execute(
            "SELECT status, block_kind, block_recurrences, current_run_id, "
            "claim_lock, claim_expires, worker_pid FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
        block_events = durable.execute(
            "SELECT payload, run_id FROM task_events "
            "WHERE task_id = ? AND kind = 'blocked' ORDER BY id",
            (tid,),
        ).fetchall()
        other_kinds = durable.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? "
            "AND kind IN ('dependency_wait', 'block_loop_detected')",
            (tid,),
        ).fetchone()[0]
        run_row = durable.execute(
            "SELECT status, outcome, summary, ended_at FROM task_runs "
            "WHERE id = ?",
            (run_id_before,),
        ).fetchone()
    finally:
        durable.close()

    # Ordinary routing: the human ``blocked`` bucket, never todo or triage.
    assert task_row["status"] == "blocked"
    assert task_row["block_kind"] == "needs_input"
    assert task_row["block_recurrences"] == 1
    assert task_row["current_run_id"] is None
    assert task_row["claim_lock"] is None
    assert task_row["claim_expires"] is None
    assert task_row["worker_pid"] is None

    # Exactly one ``blocked`` event, and no dependency/loop event.
    assert len(block_events) == 1
    assert json.loads(block_events[0]["payload"]) == {
        "reason": "needs a human",
        "kind": "needs_input",
        "recurrences": 1,
        "source_status": "ready",
    }
    assert block_events[0]["run_id"] == run_id_before
    assert other_kinds == 0

    # The run was closed by the block, as before.
    assert run_row["status"] == "blocked"
    assert run_row["outcome"] == "blocked"
    assert run_row["summary"] == "needs a human"
    assert run_row["ended_at"] is not None


# ---------------------------------------------------------------------------
# R1N-T10 — the loop-detected (triage) sibling branch is unchanged
# ---------------------------------------------------------------------------


_LOOP_KIND = "capability"


def test_loop_detected_block_parity(kanban_home, hooks):
    """Re-blocking for the same cause routes to ``triage`` and dispatches once.

    The recurrence state is built with real production calls — block →
    unblock → claim, repeated — rather than by writing ``block_recurrences``
    directly, so the loop breaker is driven the way the cron/worker loop it
    exists to break would drive it. Callback kwargs are only spot-checked
    (``task_id``); R1N-T11 owns exact kwarg parity.
    """
    limit = kb.BLOCK_RECURRENCE_LIMIT
    assert limit > 1

    conn = kb.connect()
    db_path = kb.kanban_db_path()
    seen_in_transaction: list[bool] = []

    try:
        tid = _running_task(conn, title="loop-breaker")

        # Drive the unblock ↔ re-block loop up to (but not past) the limit.
        for _ in range(limit - 1):
            assert kb.block_task(
                conn, tid, reason="flaky again", kind=_LOOP_KIND,
            ) is True
            assert kb.unblock_task(conn, tid) is True
            assert kb.claim_task(conn, tid, claimer="worker") is not None

        # Preconditions: the counter really was built up, and the task is
        # blockable again — no hand-written endpoint state.
        pre = conn.execute(
            "SELECT status, block_kind, block_recurrences, current_run_id "
            "FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
        assert pre["status"] == "running"
        assert pre["block_kind"] == _LOOP_KIND
        assert pre["block_recurrences"] == limit - 1
        final_run_id = pre["current_run_id"]
        assert final_run_id is not None

        # Register only now, so the setup blocks above are not counted.
        hooks.record(
            "kanban_task_blocked",
            also=lambda **kw: seen_in_transaction.append(conn.in_transaction),
        )

        result = kb.block_task(
            conn, tid, reason="still flaky", kind=_LOOP_KIND,
        )
    finally:
        conn.close()

    assert result is True
    fired = hooks.blocked()
    assert len(fired) == 1
    assert fired[0]["task_id"] == tid
    assert seen_in_transaction == [False]

    durable = sqlite3.connect(str(db_path), timeout=5.0)
    try:
        durable.row_factory = sqlite3.Row
        task_row = durable.execute(
            "SELECT status, block_kind, block_recurrences, current_run_id, "
            "claim_lock, claim_expires, worker_pid FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
        loop_events = durable.execute(
            "SELECT payload, run_id FROM task_events "
            "WHERE task_id = ? AND kind = 'block_loop_detected' ORDER BY id",
            (tid,),
        ).fetchall()
        dependency_events = durable.execute(
            "SELECT COUNT(*) FROM task_events "
            "WHERE task_id = ? AND kind = 'dependency_wait'",
            (tid,),
        ).fetchone()[0]
        run_row = durable.execute(
            "SELECT status, outcome, summary, ended_at FROM task_runs "
            "WHERE id = ?",
            (final_run_id,),
        ).fetchone()
    finally:
        durable.close()

    # Loop-detected routing: triage for a human, never blocked and never todo.
    assert task_row["status"] == "triage"
    assert task_row["block_kind"] == _LOOP_KIND
    assert task_row["block_recurrences"] == limit
    assert task_row["current_run_id"] is None
    assert task_row["claim_lock"] is None
    assert task_row["claim_expires"] is None
    assert task_row["worker_pid"] is None

    # Exactly one block_loop_detected, with the full documented payload.
    assert len(loop_events) == 1
    assert json.loads(loop_events[0]["payload"]) == {
        "reason": "still flaky",
        "kind": _LOOP_KIND,
        "recurrences": limit,
        "limit": limit,
        "source_status": "ready",
    }
    assert loop_events[0]["run_id"] == final_run_id
    # No dependency-branch leakage into the loop-detected path.
    assert dependency_events == 0

    assert run_row["status"] == "blocked"
    assert run_row["outcome"] == "blocked"
    assert run_row["summary"] == "still flaky"
    assert run_row["ended_at"] is not None


# ---------------------------------------------------------------------------
# R1N-T11 — exact dependency-block callback kwargs
# ---------------------------------------------------------------------------


_PARITY_BOARD = "r1-parity-board"
_PARITY_ASSIGNEE = "parity-worker"
_PARITY_PROFILE = "r1-parity-profile"
_PARITY_REASON = "R-1 parity: waiting on parent task"


def test_dependency_callback_kwargs_parity(kanban_home, hooks, monkeypatch):
    """The dependency-block callback receives exactly the documented kwargs.

    Every value is deliberately non-default — a real non-``default`` board, a
    distinctive assignee and reason, a real run id, and a patched profile
    resolver — so a hardcoded fallback or a dropped argument cannot be masked
    by a value that happens to match the default.

    Preservation only: timing is R1N-T01's and durability is R1N-T02's, and
    neither is re-asserted here.
    """
    # A genuinely non-default board, created the normal way, so
    # get_current_board() resolves to it for real rather than by patching.
    kb.create_board(_PARITY_BOARD)
    monkeypatch.setenv("HERMES_KANBAN_BOARD", _PARITY_BOARD)
    assert kb.get_current_board() == _PARITY_BOARD

    # profile_name is resolved inside _fire_kanban_lifecycle_hook; make it
    # something no default could produce.
    monkeypatch.setattr(
        "hermes_cli.profiles.get_active_profile_name", lambda: _PARITY_PROFILE,
    )

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="kwarg parity", assignee=_PARITY_ASSIGNEE)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        assert kb.claim_task(conn, tid, claimer=_PARITY_ASSIGNEE) is not None

        run_id = conn.execute(
            "SELECT current_run_id FROM tasks WHERE id = ?", (tid,),
        ).fetchone()["current_run_id"]
        assert run_id is not None

        hooks.record("kanban_task_blocked")

        result = kb.block_task(
            conn, tid, reason=_PARITY_REASON, kind="dependency",
        )
    finally:
        conn.close()

    assert result is True

    fired = hooks.blocked()
    assert len(fired) == 1

    # Exact mapping: equality catches a missing key, an extra key, and any
    # changed value in one assertion.
    assert fired[0] == {
        "task_id": tid,
        "board": _PARITY_BOARD,
        "assignee": _PARITY_ASSIGNEE,
        "run_id": run_id,
        "reason": _PARITY_REASON,
        "profile_name": _PARITY_PROFILE,
        "telemetry_schema_version": OBSERVER_SCHEMA_VERSION,
    }


# ---------------------------------------------------------------------------
# R1N-T12 — dependency-block business behaviour is unchanged
# ---------------------------------------------------------------------------


_BUSINESS_REASON = "waiting on parent"


def _setup_claimed_with_prior_recurrence(conn):
    """Running, with a real non-zero block_recurrences already accumulated.

    The counter is built through production calls (ordinary block → unblock →
    re-claim) so that "the dependency branch does not touch it" is a
    meaningful assertion rather than 0 == 0.
    """
    tid = _running_task(conn, title="business-claimed")
    assert kb.block_task(
        conn, tid, reason="needs a human", kind="needs_input",
    ) is True
    assert kb.unblock_task(conn, tid) is True
    assert kb.claim_task(conn, tid, claimer="worker") is not None
    return tid


def _setup_never_claimed(conn):
    """Ready but never claimed — no active run for _end_run to close."""
    tid = kb.create_task(conn, title="business-unclaimed", assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    return tid


@pytest.mark.parametrize(
    "setup",
    [_setup_claimed_with_prior_recurrence, _setup_never_claimed],
    ids=["claimed_run_closed", "never_claimed_run_synthesized"],
)
def test_dependency_block_business_parity(kanban_home, setup):
    """Destination, event, loop counter, claim teardown and run handling.

    Registers no callbacks at all: hook timing is R1N-T01's and hook kwargs
    are R1N-T11's, and neither is re-asserted here.
    """
    conn = kb.connect()
    db_path = kb.kanban_db_path()

    try:
        tid = setup(conn)

        before = conn.execute(
            "SELECT status, block_kind, block_recurrences, current_run_id "
            "FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
        status_before = before["status"]
        recurrences_before = before["block_recurrences"]
        run_id_before = before["current_run_id"]
        assert status_before in ("running", "ready")

        before_event_ids = {
            row["id"] for row in conn.execute(
                "SELECT id FROM task_events WHERE task_id = ?", (tid,),
            )
        }
        before_run_ids = {
            row["id"] for row in conn.execute(
                "SELECT id FROM task_runs WHERE task_id = ?", (tid,),
            )
        }

        result = kb.block_task(
            conn, tid, reason=_BUSINESS_REASON, kind="dependency",
        )
    finally:
        conn.close()

    assert result is True

    durable = sqlite3.connect(str(db_path), timeout=5.0)
    try:
        durable.row_factory = sqlite3.Row
        task_row = durable.execute(
            "SELECT status, block_kind, block_recurrences, current_run_id, "
            "claim_lock, claim_expires, worker_pid FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
        all_events = durable.execute(
            "SELECT id, kind, payload, run_id FROM task_events "
            "WHERE task_id = ? ORDER BY id",
            (tid,),
        ).fetchall()
        all_runs = durable.execute(
            "SELECT id, status, outcome, summary, ended_at FROM task_runs "
            "WHERE task_id = ? ORDER BY id",
            (tid,),
        ).fetchall()
    finally:
        durable.close()

    # Destination: dependency waits in todo — never blocked, never triage.
    assert task_row["status"] == "todo"
    assert task_row["block_kind"] == "dependency"

    # The loop counter is the ordinary branch's business, not this one's.
    assert task_row["block_recurrences"] == recurrences_before

    # Claim teardown.
    assert task_row["current_run_id"] is None
    assert task_row["claim_lock"] is None
    assert task_row["claim_expires"] is None
    assert task_row["worker_pid"] is None

    # Run closed (claimed) or synthesized (never claimed), never both.
    new_runs = [row for row in all_runs if row["id"] not in before_run_ids]
    if run_id_before is not None:
        assert new_runs == []
        resolved_run_id = run_id_before
    else:
        assert len(new_runs) == 1
        resolved_run_id = new_runs[0]["id"]
    run_row = next(row for row in all_runs if row["id"] == resolved_run_id)
    assert run_row["status"] == "blocked"
    assert run_row["outcome"] == "blocked"
    assert run_row["summary"] == _BUSINESS_REASON
    assert run_row["ended_at"] is not None

    # Exactly one new event, and it is the dependency_wait row — nothing else
    # was appended by this call.
    new_events = [row for row in all_events if row["id"] not in before_event_ids]
    assert len(new_events) == 1
    assert new_events[0]["kind"] == "dependency_wait"
    assert json.loads(new_events[0]["payload"]) == {
        "reason": _BUSINESS_REASON,
        "kind": "dependency",
        "source_status": "ready",
    }
    assert new_events[0]["run_id"] == resolved_run_id
