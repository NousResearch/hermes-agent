"""C-1: generic ``kanban_task_event`` observer surface.

Tests are named ``C1N-*`` after the implementation packet's matrix. They assert
relationships and transaction behavior only — never line numbers, hook totals,
the production ``_append_event`` call-site count, or a frozen vocabulary size.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from sqlite3 import OperationalError as sqlite3_OperationalError

from hermes_cli import kanban_db as kb
from hermes_cli.plugins import VALID_HOOKS, get_plugin_manager

GENERIC_HOOK = "kanban_task_event"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """A private board root, with every ambient board override cleared.

    ``HERMES_KANBAN_DB`` in particular must be unset: ``kanban_db_path``
    consults it *ahead of* an explicit board slug, so a leaked value would
    silently redirect every ``connect(board=...)`` in this module.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def pinned_manager(monkeypatch):
    """The plugin-manager singleton, pinned for the duration of one test.

    ``get_plugin_manager`` is a lazily-created module singleton and the
    repo-wide autouse ``_hermetic_environment`` fixture resets it to ``None``
    (``tests/conftest.py``). Resolving the manager once at fixture setup is
    therefore not enough: if anything re-resolves it mid-test, a *different*
    manager answers the dispatch and the callback this fixture registered is
    simply not there.

    That is a pre-existing, C-1-independent hazard — on the unmodified base,
    ``test_kanban_lifecycle_hooks.py::test_claim_fires_hook`` fails for exactly
    this reason when ``test_kanban_cli_dispatch_passthrough.py`` runs before it
    in the same process. Pinning the module attribute makes these tests
    order-independent instead of inheriting it.
    """
    from hermes_cli import plugins as plugins_mod

    mgr = get_plugin_manager()
    monkeypatch.setattr(plugins_mod, "_plugin_manager", mgr)
    return mgr


@pytest.fixture
def events(pinned_manager):
    """Capture every ``kanban_task_event`` callback kwargs dict, in order."""
    mgr = pinned_manager
    seen: list[dict] = []
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    mgr._hooks.setdefault(GENERIC_HOOK, []).append(lambda **kw: seen.append(kw))
    try:
        yield seen
    finally:
        mgr._hooks = saved


@pytest.fixture
def hooks(pinned_manager):
    """Register arbitrary callbacks against the shared plugin manager.

    Yields a ``register(hook_name, callback)`` helper; the registry is restored
    afterwards so tests never leak callbacks into each other.
    """
    mgr = pinned_manager
    saved = {k: list(v) for k, v in mgr._hooks.items()}

    def register(hook_name, callback):
        mgr._hooks.setdefault(hook_name, []).append(callback)

    try:
        yield register
    finally:
        mgr._hooks = saved


def rows(conn, task_id=None):
    """Read committed ``task_events`` rows, oldest first."""
    sql = "SELECT id, task_id, run_id, kind, created_at FROM task_events"
    args: tuple = ()
    if task_id is not None:
        sql += " WHERE task_id = ?"
        args = (task_id,)
    return [dict(r) for r in conn.execute(sql + " ORDER BY id", args).fetchall()]


# --------------------------------------------------------------------------
# C1N-T01 — registration surface
# --------------------------------------------------------------------------


def test_c1n_t01_generic_hook_is_declared():
    """The generic observer hook is declared as a valid hook name."""
    assert "kanban_task_event" in VALID_HOOKS


def test_c1n_t01_legacy_hooks_unchanged():
    """The three legacy kanban hooks survive C-1 untouched.

    No assertion on ``len(VALID_HOOKS)`` — the total is exact-base evidence,
    never API.
    """
    for name in (
        "kanban_task_claimed",
        "kanban_task_completed",
        "kanban_task_blocked",
    ):
        assert name in VALID_HOOKS


# --------------------------------------------------------------------------
# C1N-T02 — one committed row -> exactly one callback, with the committed scalars
# --------------------------------------------------------------------------


def test_c1n_t02_committed_row_produces_one_matching_callback(kanban_home, events):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t")
        committed = rows(conn, tid)
    finally:
        conn.close()

    assert len(committed) == 1
    assert len(events) == 1
    kw = events[0]
    row = committed[0]
    assert kw["task_id"] == row["task_id"] == tid
    assert kw["kind"] == row["kind"] == "created"
    assert kw["core_event_seq"] == row["id"]
    assert kw["created_at_epoch_s"] == row["created_at"]
    assert kw["run_id"] == row["run_id"]
    assert kw["kanban_event_schema_version"] == "hermes.kanban.event.v1"
    assert "profile_name" in kw


# --------------------------------------------------------------------------
# C1N-T16 — the append seam fails closed
# --------------------------------------------------------------------------


def test_c1n_t16_append_event_without_a_frame_raises_before_inserting(kanban_home):
    """A durable new-event insert must never escape instrumentation."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t")
        before = len(rows(conn))
        with pytest.raises(RuntimeError):
            kb._append_event(conn, tid, "created")
        assert len(rows(conn)) == before
    finally:
        conn.close()


def test_c1n_t16_append_event_works_under_a_callers_frame(kanban_home, events):
    """The guard tests for an *active* frame, not a locally-opened one.

    ``_insert_completion_attachment`` never opens its own ``write_txn``; it runs
    inside ``complete_task``'s. Reproduce that shape directly.
    """
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t")
        events.clear()
        with kb.write_txn(conn):
            kb._insert_completion_attachment(
                conn,
                tid,
                filename="a.txt",
                stored_path="/tmp/a.txt",
                size=3,
                created_at=1,
            )
        committed = [r for r in rows(conn, tid) if r["kind"] == "attached"]
    finally:
        conn.close()

    assert len(committed) == 1
    assert [e["kind"] for e in events] == ["attached"]
    assert events[0]["core_event_seq"] == committed[0]["id"]


# --------------------------------------------------------------------------
# C1N-T03 / C1N-T05 — insertion ordering, including epoch-second ties
# --------------------------------------------------------------------------


def test_c1n_t03_multi_row_transaction_dispatches_in_insertion_order(
    kanban_home, events
):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t")
        events.clear()
        with kb.write_txn(conn):
            kb._append_event(conn, tid, "commented")
            kb._append_event(conn, tid, "edited")
            kb._append_event(conn, tid, "heartbeat")
            # Nothing may be dispatched while the transaction is still open.
            assert events == []
        committed = [r for r in rows(conn, tid) if r["kind"] != "created"]
    finally:
        conn.close()

    assert [e["kind"] for e in events] == ["commented", "edited", "heartbeat"]
    assert [e["core_event_seq"] for e in events] == [r["id"] for r in committed]


def test_c1n_t05_epoch_second_ties_still_order_by_core_event_seq(kanban_home, events):
    """Rows in one transaction routinely share ``created_at``; ids still order."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t")
        events.clear()
        with kb.write_txn(conn):
            for _ in range(5):
                kb._append_event(conn, tid, "heartbeat")
    finally:
        conn.close()

    stamps = {e["created_at_epoch_s"] for e in events}
    assert len(stamps) == 1, "fixture did not produce an epoch-second tie"
    seqs = [e["core_event_seq"] for e in events]
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)


# --------------------------------------------------------------------------
# C1N-T06 / C1N-T07 — callbacks run outside the transaction and on-path
# --------------------------------------------------------------------------


def test_c1n_t06_callback_runs_outside_the_transaction_and_write_lock(
    kanban_home, hooks
):
    observed: list[bool] = []
    other = kb.connect()

    def _cb(**kw):
        observed.append(other.in_transaction)
        # The write lock must already be released: a competing connection can
        # take BEGIN IMMEDIATE while this observer is still running.
        other.execute("BEGIN IMMEDIATE")
        other.execute("ROLLBACK")

    hooks(GENERIC_HOOK, _cb)
    conn = kb.connect()
    try:
        conn.execute("PRAGMA busy_timeout=2000")
        kb.create_task(conn, title="t")
    finally:
        conn.close()
        other.close()

    assert observed == [False]


def test_c1n_t07_callback_latency_is_off_lock_but_on_path(kanban_home, hooks):
    """A sleeping observer measurably delays the originating call, after commit."""
    import time as _time

    marks: list[tuple[str, float]] = []

    def _cb(**kw):
        marks.append(("cb_enter", _time.monotonic()))
        _time.sleep(0.25)
        marks.append(("cb_exit", _time.monotonic()))

    hooks(GENERIC_HOOK, _cb)
    conn = kb.connect()
    try:
        start = _time.monotonic()
        kb.create_task(conn, title="t")
        elapsed = _time.monotonic() - start
    finally:
        conn.close()

    assert [m[0] for m in marks] == ["cb_enter", "cb_exit"]
    assert elapsed >= 0.25


# --------------------------------------------------------------------------
# C1N-T08 / T09 / T10 / T11 — every suppression path
# --------------------------------------------------------------------------


def test_c1n_t08_body_rollback_suppresses_and_clears(kanban_home, events):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t")
        events.clear()
        with pytest.raises(RuntimeError):
            with kb.write_txn(conn):
                kb._append_event(conn, tid, "commented")
                raise RuntimeError("body blew up")
        assert events == []
        assert [r["kind"] for r in rows(conn, tid)] == ["created"]
        assert kb._TASK_EVENT_FRAME.get() is None
        assert conn.in_transaction is False
    finally:
        conn.close()


def test_c1n_t08_nested_savepoint_rollback_discards_only_its_own_records(
    kanban_home, events
):
    """A swallowed nested failure must not announce rows it rolled back.

    ``write_txn(allow_nested=True)`` composes via a SQLite savepoint, so a
    ``ROLLBACK TO`` undoes the inner rows while the OUTER transaction goes on
    to commit. The queue has to be truncated to its pre-savepoint mark or the
    outer commit would dispatch events for rows that never became durable —
    the same "no callbacks on rollback" rule the top-level path enforces.
    """
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t")
        events.clear()
        with kb.write_txn(conn):
            kb._append_event(conn, tid, "commented")
            # Swallowed on purpose: this is the whole point of a savepoint.
            with pytest.raises(RuntimeError):
                with kb.write_txn(conn, allow_nested=True):
                    kb._append_event(conn, tid, "attached")
                    raise RuntimeError("nested blew up")
            kb._append_event(conn, tid, "edited")
        durable = [r["kind"] for r in rows(conn, tid)]
        assert kb._TASK_EVENT_FRAME.get() is None
        assert conn.in_transaction is False
    finally:
        conn.close()

    # The rolled-back row is gone from the table, and from the fan-out.
    assert durable == ["created", "commented", "edited"]
    assert [e["kind"] for e in events] == ["commented", "edited"]
    assert "attached" not in {e["kind"] for e in events}


def test_c1n_t08_nested_savepoint_success_flushes_once_with_the_outer_commit(
    kanban_home, events
):
    """A released savepoint stays pending until the OUTER transaction commits."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t")
        events.clear()
        with kb.write_txn(conn):
            with kb.write_txn(conn, allow_nested=True):
                kb._append_event(conn, tid, "commented")
            # RELEASE only folds the savepoint in — nothing is durable, so
            # nothing may have been dispatched yet.
            assert events == []
        assert [e["kind"] for e in events] == ["commented"]
        assert [r["kind"] for r in rows(conn, tid)] == ["created", "commented"]
    finally:
        conn.close()


def test_c1n_t08_outer_rollback_still_discards_released_nested_records(
    kanban_home, events
):
    """A successful savepoint is not durable on its own.

    If the outer transaction later rolls back, the released inner rows go with
    it and must not be announced.
    """
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t")
        events.clear()
        with pytest.raises(RuntimeError):
            with kb.write_txn(conn):
                with kb.write_txn(conn, allow_nested=True):
                    kb._append_event(conn, tid, "commented")
                raise RuntimeError("outer blew up")
        assert events == []
        assert [r["kind"] for r in rows(conn, tid)] == ["created"]
        assert kb._TASK_EVENT_FRAME.get() is None
    finally:
        conn.close()


def test_c1n_t09_terminal_commit_failure_suppresses_and_clears(
    kanban_home, events, monkeypatch
):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t")
        events.clear()

        real = kb._execute_boundary_with_retry

        def _fail_commit(c, stmt, *a, **kw):
            if stmt.strip().upper().startswith("COMMIT"):
                raise sqlite3_OperationalError("commit exhausted retries")
            return real(c, stmt, *a, **kw)

        monkeypatch.setattr(kb, "_execute_boundary_with_retry", _fail_commit)
        with pytest.raises(Exception):
            with kb.write_txn(conn):
                kb._append_event(conn, tid, "commented")
        monkeypatch.undo()

        assert events == []
        assert kb._TASK_EVENT_FRAME.get() is None
        assert conn.in_transaction is False
        assert [r["kind"] for r in rows(conn, tid)] == ["created"]
    finally:
        conn.close()


def test_c1n_t10_post_commit_invariant_failure_suppresses(
    kanban_home, events, monkeypatch
):
    """The deliberate committed-row-without-callback window.

    Asserts the CALLBACK COUNT ONLY and explicitly tolerates a durable row: the
    invariant runs after COMMIT, so the data may well be on disk. The
    connection is put in rollback-journal mode first, because the invariant
    deliberately skips under WAL.
    """
    import hermes_cli.sqlite_safe_read as ssr

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t")
        conn.execute("PRAGMA journal_mode=DELETE")
        events.clear()
        monkeypatch.setattr(ssr, "file_length_matches_header", lambda _c: False)
        with pytest.raises(Exception):
            with kb.write_txn(conn):
                kb._append_event(conn, tid, "commented")
        monkeypatch.undo()

        assert events == []
        assert kb._TASK_EVENT_FRAME.get() is None
        # Deliberately NOT asserting the row is absent — it may be durable.
    finally:
        conn.close()


def test_c1n_t10_invariant_is_skipped_under_wal(kanban_home, events, monkeypatch):
    """D-11 control: the same forced mismatch is a no-op under WAL."""
    import hermes_cli.sqlite_safe_read as ssr

    conn = kb.connect()
    try:
        assert str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "wal"
        tid = kb.create_task(conn, title="t")
        events.clear()
        monkeypatch.setattr(ssr, "file_length_matches_header", lambda _c: False)
        with kb.write_txn(conn):
            kb._append_event(conn, tid, "commented")
        monkeypatch.undo()
        assert [e["kind"] for e in events] == ["commented"]
    finally:
        conn.close()


def test_c1n_t11_pre_begin_guard_installs_no_frame(kanban_home, events, monkeypatch):
    conn = kb.connect()
    try:
        events.clear()
        monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", "1")
        with pytest.raises(PermissionError):
            with kb.write_txn(conn):  # pragma: no cover - body never runs
                raise AssertionError("body must not run")
        monkeypatch.undo()
        assert events == []
        assert kb._TASK_EVENT_FRAME.get() is None
        assert conn.in_transaction is False
    finally:
        conn.close()


# --------------------------------------------------------------------------
# C1N-T12 / T13 / T14 — observer isolation, fan-out order, reentrancy
# --------------------------------------------------------------------------


def test_c1n_t12_raising_observer_is_isolated(kanban_home, hooks):
    seen: list[str] = []

    def _boom(**kw):
        seen.append("boom")
        raise RuntimeError("observer exploded")

    hooks(GENERIC_HOOK, _boom)
    hooks(GENERIC_HOOK, lambda **kw: seen.append("after"))

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t")
        committed = rows(conn, tid)
    finally:
        conn.close()

    assert seen == ["boom", "after"]
    assert [r["kind"] for r in committed] == ["created"]


def test_c1n_t13_fanout_follows_registration_order(kanban_home, hooks):
    order: list[str] = []
    for name in ("a", "b", "c"):
        hooks(GENERIC_HOOK, lambda _n=name, **kw: order.append(f"{_n}:{kw['kind']}"))

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t")
        with kb.write_txn(conn):
            kb._append_event(conn, tid, "commented")
    finally:
        conn.close()

    assert order == [
        "a:created", "b:created", "c:created",
        "a:commented", "b:commented", "c:commented",
    ]


def test_c1n_t14_observer_that_writes_the_board_uses_a_fresh_frame(kanban_home, hooks):
    depth: list[int] = []
    seen: list[str] = []

    def _cb(**kw):
        seen.append(kw["kind"])
        # The completed frame is already detached, so this write cannot append
        # to it or re-flush it.
        assert kb._TASK_EVENT_FRAME.get() is None
        if kw["kind"] == "created" and len(depth) < 3:
            depth.append(1)
            inner = kb.connect()
            try:
                with kb.write_txn(inner):
                    kb._append_event(inner, kw["task_id"], "commented")
            finally:
                inner.close()

    hooks(GENERIC_HOOK, _cb)
    conn = kb.connect()
    try:
        kb.create_task(conn, title="t")
    finally:
        conn.close()

    # One nested write, no unbounded recursion: 'commented' does not re-trigger.
    assert seen == ["created", "commented"]
    assert kb._TASK_EVENT_FRAME.get() is None


# --------------------------------------------------------------------------
# C1N-T35 — board comes from the CONNECTION, never from ambient state
# --------------------------------------------------------------------------


def test_c1n_t35a_explicit_board_beats_ambient_board(kanban_home, events, monkeypatch):
    """``HERMES_KANBAN_DB`` must stay cleared here.

    ``kanban_db_path`` consults that variable *ahead of* the explicit slug, so
    a set value would open the env-named file and this assertion would say
    nothing about ``board="B"``.
    """
    kb.create_board("board-b")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "board-a")
    kb.create_board("board-a")
    assert kb.get_current_board() == "board-a"

    conn = kb.connect(board="board-b")
    try:
        events.clear()
        kb.create_task(conn, title="t")
    finally:
        conn.close()

    assert [e["board"] for e in events] == ["board-b"]


def test_c1n_t35a_explicit_db_path_beats_ambient_board(kanban_home, events, monkeypatch):
    kb.create_board("board-b")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "board-a")
    kb.create_board("board-a")

    conn = kb.connect(db_path=kb.kanban_db_path(board="board-b"))
    try:
        events.clear()
        kb.create_task(conn, title="t")
    finally:
        conn.close()

    assert [e["board"] for e in events] == ["board-b"]


def test_c1n_t35a_default_board_legacy_path_resolves(kanban_home, events):
    conn = kb.connect(board="default")
    try:
        events.clear()
        kb.create_task(conn, title="t")
    finally:
        conn.close()

    assert [e["board"] for e in events] == ["default"]


def test_c1n_t35b_env_db_override_outside_layout_yields_none(
    kanban_home, events, tmp_path, monkeypatch
):
    """The resolver reports the connection's ACTUAL file.

    ``HERMES_KANBAN_DB`` overrode the explicit ``board="B"`` argument, so the
    connection is not writing board B at all. Reporting ``"board-b"`` here
    would be a lie; ``None`` is the honest answer.
    """
    kb.create_board("board-b")
    outside = tmp_path / "elsewhere" / "kanban.db"
    outside.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_KANBAN_DB", str(outside))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "board-a")

    conn = kb.connect(board="board-b")
    try:
        events.clear()
        kb.create_task(conn, title="t")
    finally:
        conn.close()

    assert [e["board"] for e in events] == [None]


def test_c1n_t35b_db_path_outside_layout_yields_none(
    kanban_home, events, tmp_path, monkeypatch
):
    outside = tmp_path / "elsewhere2" / "kanban.db"
    outside.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "board-a")

    conn = kb.connect(db_path=outside)
    try:
        events.clear()
        kb.create_task(conn, title="t")
    finally:
        conn.close()

    assert [e["board"] for e in events] == [None]


def test_c1n_t35_board_is_fixed_at_frame_install(kanban_home, events, monkeypatch):
    """Ambient state changing mid-transaction cannot re-attribute the board."""
    kb.create_board("board-b")
    conn = kb.connect(board="board-b")
    try:
        tid = kb.create_task(conn, title="t")
        events.clear()
        with kb.write_txn(conn):
            kb._append_event(conn, tid, "commented")
            monkeypatch.setenv("HERMES_KANBAN_BOARD", "board-a")
            kb._append_event(conn, tid, "edited")
    finally:
        conn.close()

    assert [e["board"] for e in events] == ["board-b", "board-b"]


# --------------------------------------------------------------------------
# C1N-T21 — capability declaration
# --------------------------------------------------------------------------


def test_c1n_t21_declaration_is_an_immutable_module_constant():
    assert isinstance(kb.KANBAN_EVENT_KINDS, frozenset)
    assert all(isinstance(k, str) for k in kb.KANBAN_EVENT_KINDS)


def test_c1n_t21_driven_writers_emit_declared_kinds(kanban_home, events):
    """Per-writer assertion. No assertion on the size of the declaration."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t", assignee="w")
        kb.assign_task(conn, tid, "w2")
        kb.add_comment(conn, tid, "me", "hello")
        claimed = kb.claim_task(conn, tid, claimer="w:1")
        assert claimed is not None
        kb.complete_task(conn, tid, result="done")
    finally:
        conn.close()

    emitted = [e["kind"] for e in events]
    assert "created" in emitted and "claimed" in emitted and "completed" in emitted
    for kind in emitted:
        assert kind in kb.KANBAN_EVENT_KINDS, f"{kind} is emitted but undeclared"


# --------------------------------------------------------------------------
# C1N-T17 / T18 / T34 — the five bundled-dashboard writers
# --------------------------------------------------------------------------


@pytest.fixture
def dash():
    """The bundled dashboard plugin API module (a third writing process)."""
    return pytest.importorskip("plugins.kanban.dashboard.plugin_api")


def test_c1n_t17_dashboard_reprioritized_and_edited_use_the_shared_seam(
    kanban_home, events, dash
):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t")
    finally:
        conn.close()
    events.clear()

    dash.update_task(tid, dash.UpdateTaskBody(priority=7), board=None)
    dash.update_task(tid, dash.UpdateTaskBody(title="new title"), board=None)

    conn = kb.connect()
    try:
        committed = [r for r in rows(conn, tid) if r["kind"] != "created"]
    finally:
        conn.close()

    assert [e["kind"] for e in events] == ["reprioritized", "edited"]
    for kw, row in zip(events, committed):
        assert kw["core_event_seq"] == row["id"]
        assert kw["created_at_epoch_s"] == row["created_at"]
        assert kw["run_id"] == row["run_id"] is None
        assert kw["task_id"] == tid
        assert kw["kanban_event_schema_version"] == "hermes.kanban.event.v1"
        assert kw["kind"] in kb.KANBAN_EVENT_KINDS
    # A dashboard edit carries no title/body content.
    assert "title" not in events[1] and "body" not in events[1]


def test_c1n_t17_dashboard_status_writer_uses_the_shared_seam(
    kanban_home, events, dash
):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t")
    finally:
        conn.close()
    events.clear()

    conn = kb.connect()
    try:
        assert dash._set_status_direct(conn, tid, "todo") is True
        committed = [r for r in rows(conn, tid) if r["kind"] == "status"]
    finally:
        conn.close()

    assert [e["kind"] for e in events] == ["status"]
    assert events[0]["core_event_seq"] == committed[0]["id"]
    assert events[0]["status_to"] == "todo"


def test_c1n_t17_dashboard_bulk_reprioritized_uses_the_shared_seam(
    kanban_home, events, dash
):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t")
    finally:
        conn.close()
    events.clear()

    dash.bulk_update(dash.BulkTaskBody(ids=[tid], priority=3), board=None)

    assert [e["kind"] for e in events] == ["reprioritized"]
    assert "priority" not in events[0]
    assert "status_to" not in events[0]


def test_c1n_t18_reopened_parent_demotion_carries_status_to_todo(
    kanban_home, events, dash
):
    conn = kb.connect()
    try:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child")
        kb.link_tasks(conn, parent, child)
        # Drive the parent to done so the child can sit in ready.
        assert dash._set_status_direct(conn, parent, "done") is True
        assert dash._set_status_direct(conn, child, "ready") is True
        events.clear()
        # Reopening the parent must demote the ready child back to todo.
        assert dash._set_status_direct(conn, parent, "todo") is True
        child_rows = [r for r in rows(conn, child) if r["kind"] == "status"]
    finally:
        conn.close()

    demotions = [e for e in events if e["task_id"] == child and e["kind"] == "status"]
    assert len(demotions) == 1
    assert demotions[0]["status_to"] == "todo"
    assert demotions[0]["core_event_seq"] == child_rows[-1]["id"]
    # No payload parsing and no post-commit read: nothing from the payload
    # ('reason', 'parent') leaks into the kwargs.
    assert "reason" not in demotions[0] and "parent" not in demotions[0]


def test_c1n_t34_dashboard_board_is_the_connection_not_ambient(
    kanban_home, events, dash, monkeypatch
):
    kb.create_board("board-b")
    kb.create_board("board-a")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "board-a")
    assert kb.get_current_board() == "board-a"

    conn = kb.connect(board="board-b")
    try:
        tid = kb.create_task(conn, title="t")
    finally:
        conn.close()
    events.clear()

    dash.update_task(tid, dash.UpdateTaskBody(priority=9), board="board-b")

    assert [e["kind"] for e in events] == ["reprioritized"]
    assert events[0]["board"] == "board-b"


# --------------------------------------------------------------------------
# C1N-T26 / C1N-T27 — the exact status_to matrix and conditional-transition honesty
# --------------------------------------------------------------------------


def only(events, kind, task_id=None):
    """Return the single callback for ``kind`` (optionally scoped to a task)."""
    hits = [
        e for e in events
        if e["kind"] == kind and (task_id is None or e["task_id"] == task_id)
    ]
    assert len(hits) == 1, f"expected exactly one {kind!r} callback, got {len(hits)}"
    return hits[0]


def test_c1n_t26_created_status_domain(kanban_home, events):
    """``created``'s domain is {ready, todo, triage, blocked} — never running."""
    conn = kb.connect()
    try:
        plain = kb.create_task(conn, title="plain")
        assert only(events, "created", plain)["status_to"] == "ready"

        events.clear()
        triage = kb.create_task(conn, title="triage", triage=True)
        assert only(events, "created", triage)["status_to"] == "triage"

        events.clear()
        blocked = kb.create_task(conn, title="b", initial_status="blocked")
        assert only(events, "created", blocked)["status_to"] == "blocked"

        events.clear()
        child = kb.create_task(conn, title="child", parents=[plain])
        assert only(events, "created", child)["status_to"] == "todo"
    finally:
        conn.close()


def test_c1n_t26_claim_complete_and_block_transitions(kanban_home, events):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t")
        events.clear()
        assert kb.claim_task(conn, tid, claimer="w:1") is not None
        assert only(events, "claimed")["status_to"] == "running"

        events.clear()
        assert kb.block_task(conn, tid, reason="need input", kind="needs_input")
        assert only(events, "blocked")["status_to"] == "blocked"

        events.clear()
        assert kb.unblock_task(conn, tid)
        assert only(events, "unblocked")["status_to"] == "ready"

        events.clear()
        assert kb.claim_task(conn, tid, claimer="w:2") is not None
        kb.complete_task(conn, tid, result="ok")
        assert only(events, "completed")["status_to"] == "done"
    finally:
        conn.close()


def test_c1n_t26_unblocked_reports_the_exact_local_new_status(kanban_home, events):
    """``unblocked`` reports ``todo`` when parents are still undone."""
    conn = kb.connect()
    try:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child", parents=[parent])
        # The child sits in 'todo' behind its parent; force it ready so
        # block_task's guarded UPDATE can match, then block it.
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (child,))
        conn.commit()
        assert kb.block_task(conn, child, reason="r")
        events.clear()
        assert kb.unblock_task(conn, child)
        assert only(events, "unblocked")["status_to"] == "todo"
    finally:
        conn.close()


def test_c1n_t26_dependency_wait_reports_todo(kanban_home, events):
    """A dependency block routes to ``todo``, never to ``blocked``."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t")
        events.clear()
        assert kb.block_task(conn, tid, reason="dep", kind="dependency") is True
        assert only(events, "dependency_wait")["status_to"] == "todo"
        assert kb.get_task(conn, tid).status == "todo"
    finally:
        conn.close()


def test_c1n_t26_block_loop_detected_reports_triage(kanban_home, events):
    """At BLOCK_RECURRENCE_LIMIT the block routes to ``triage``."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t")
        # First block for this cause records recurrence 1 and lands in
        # 'blocked'; the re-block after an unblock reaches the limit.
        assert kb.block_task(conn, tid, reason="same", kind="needs_input")
        assert kb.unblock_task(conn, tid)
        events.clear()
        assert kb.block_task(conn, tid, reason="same", kind="needs_input")
        assert only(events, "block_loop_detected")["status_to"] == "triage"
        assert kb.get_task(conn, tid).status == "triage"
    finally:
        conn.close()


def test_c1n_t26_promote_schedule_archive_specify_decompose(kanban_home, events):
    conn = kb.connect()
    try:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child", parents=[parent])
        events.clear()
        ok, _ = kb.promote_task(conn, child, actor="op", force=True)
        assert ok
        assert only(events, "promoted_manual")["status_to"] == "ready"

        events.clear()
        assert kb.schedule_task(conn, child, reason="later")
        assert only(events, "scheduled")["status_to"] == "scheduled"

        events.clear()
        assert kb.archive_task(conn, child)
        assert only(events, "archived")["status_to"] == "archived"

        tri = kb.create_task(conn, title="tri", triage=True)
        events.clear()
        assert kb.specify_triage_task(conn, tri, title="spec", body="b")
        assert only(events, "specified")["status_to"] == "todo"

        tri2 = kb.create_task(conn, title="tri2", triage=True)
        events.clear()
        kids = kb.decompose_triage_task(
            conn, tri2, root_assignee=None,
            children=[{"title": "k1"}, {"title": "k2"}],
        )
        assert kids
        assert only(events, "decomposed")["status_to"] == "todo"
        created = [e for e in events if e["kind"] == "created"]
        assert created and all(e["status_to"] == "todo" for e in created)
        # decompose's own `linked` writes establish no status.
        for e in events:
            if e["kind"] == "linked":
                assert "status_to" not in e
    finally:
        conn.close()


def test_c1n_t26_reclaim_and_recovery_writers_report_ready(kanban_home, events):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t")
        assert kb.claim_task(conn, tid, claimer="w:1") is not None
        events.clear()
        assert kb.reclaim_task(conn, tid, reason="operator")
        assert only(events, "reclaimed")["status_to"] == "ready"

        # reconcile_orphaned_running: running with broken claim bookkeeping.
        conn.execute(
            "UPDATE tasks SET status='running', claim_lock=NULL, "
            "claim_expires=NULL WHERE id=?", (tid,),
        )
        conn.commit()
        events.clear()
        assert kb.reconcile_orphaned_running(conn) == [tid]
        assert only(events, "reconciled")["status_to"] == "ready"
    finally:
        conn.close()


def test_c1n_t26_non_transition_writers_omit_status_to(kanban_home, events):
    conn = kb.connect()
    try:
        a = kb.create_task(conn, title="a")
        b = kb.create_task(conn, title="b")
        events.clear()
        kb.assign_task(conn, a, "worker")
        kb.set_model_override(conn, a, "gpt-x")
        kb.set_reasoning_effort(conn, a, "high")
        kb.add_comment(conn, a, "me", "hi")
        kb.link_tasks(conn, a, b)
        kb.unlink_tasks(conn, a, b)
        assert kb.claim_task(conn, a, claimer="w:1") is not None
        kb.heartbeat_worker(conn, a, note="tick")
    finally:
        conn.close()

    expected_omitted = {
        "assigned", "model_override_set", "reasoning_effort_set",
        "commented", "unlinked", "heartbeat",
    }
    seen = {e["kind"] for e in events}
    assert expected_omitted <= seen
    for e in events:
        if e["kind"] in expected_omitted:
            assert "status_to" not in e, f"{e['kind']} must omit status_to"


def test_c1n_t27_promoted_omits_status_to_when_the_update_matched_nothing(
    kanban_home, events, monkeypatch
):
    """D-9: ``recompute_ready`` appends without binding its UPDATE's rowcount.

    Move the row out of the guarded status between the SELECT and the UPDATE so
    the guarded UPDATE matches zero rows; ``status_to`` must be omitted rather
    than asserted from the SQL literal.
    """
    conn = kb.connect()
    try:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child", parents=[parent])
        assert kb.claim_task(conn, parent, claimer="w:1") is not None
        kb.complete_task(conn, parent, result="ok")
        # complete_task already ran its own recompute_ready pass; push the
        # child back to 'todo' so this test drives the promotion itself.
        conn.execute("UPDATE tasks SET status='todo' WHERE id=?", (child,))
        conn.commit()
        events.clear()
        real_execute = conn.execute
        state = {"sabotaged": False}

        def _sabotage(sql, *a, **kw):
            # The promotion target is now bound (the review lane resumes to
            # its source phase), so match the guarded 'todo' predicate plus
            # the parameterised SET rather than a 'ready' literal.
            if (
                not state["sabotaged"]
                and "SET status = ?" in sql
                and "status = 'todo'" in sql
            ):
                state["sabotaged"] = True
                real_execute(
                    "UPDATE tasks SET status = 'blocked' WHERE id = ?", (child,)
                )
            return real_execute(sql, *a, **kw)

        monkeypatch.setattr(conn, "execute", _sabotage)
        kb.recompute_ready(conn)
        monkeypatch.undo()
        assert state["sabotaged"]

        promoted = [e for e in events if e["kind"] == "promoted"]
        assert len(promoted) == 1
        assert "status_to" not in promoted[0]
    finally:
        conn.close()


def test_c1n_t27_promoted_reports_ready_when_the_update_matched(kanban_home, events):
    conn = kb.connect()
    try:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child", parents=[parent])
        assert kb.claim_task(conn, parent, claimer="w:1") is not None
        kb.complete_task(conn, parent, result="ok")
        conn.execute("UPDATE tasks SET status='todo' WHERE id=?", (child,))
        conn.commit()
        events.clear()
        kb.recompute_ready(conn)
        assert only(events, "promoted", child)["status_to"] == "ready"
    finally:
        conn.close()


def test_c1n_t27_linked_status_to_is_rowcount_gated(kanban_home, events):
    conn = kb.connect()
    try:
        parent = kb.create_task(conn, title="parent")
        ready_child = kb.create_task(conn, title="ready-child")
        events.clear()
        kb.link_tasks(conn, parent, ready_child)
        # The child WAS ready, so the guarded demotion matched: honest 'todo'.
        assert only(events, "linked", ready_child)["status_to"] == "todo"

        # A second link from another parent: the child is already 'todo', so
        # the guarded UPDATE matches nothing and status_to must be omitted.
        other_parent = kb.create_task(conn, title="parent2")
        events.clear()
        kb.link_tasks(conn, other_parent, ready_child)
        assert "status_to" not in only(events, "linked", ready_child)
    finally:
        conn.close()


def test_c1n_t27_claim_rejected_status_to_is_rowcount_gated(kanban_home, events):
    conn = kb.connect()
    try:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child")
        # Force the child ready even though its parent is not done, so
        # claim_task takes the parents_not_done branch and demotes it.
        kb.link_tasks(conn, parent, child)
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (child,))
        conn.commit()
        events.clear()
        assert kb.claim_task(conn, child, claimer="w:1") is None
        assert only(events, "claim_rejected", child)["status_to"] == "todo"

        # Now the child is already 'todo': the guarded demotion matches nothing.
        events.clear()
        assert kb.claim_task(conn, child, claimer="w:1") is None
        assert "status_to" not in only(events, "claim_rejected", child)
    finally:
        conn.close()


# --------------------------------------------------------------------------
# C1N-T28 / C1N-T24 — strict failure_count, and the dynamic outcome domain
# --------------------------------------------------------------------------


def test_c1n_t28_failure_count_acceptance_is_strict(kanban_home, events):
    """No parsing, coercion, rounding, clamping, or defaulting."""
    assert kb.FAILURE_COUNT_MAX == 1000
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t")
        events.clear()
        accepted = [0, 1, 7, kb.FAILURE_COUNT_MAX]
        rejected = [True, False, "3", 3.0, -1, kb.FAILURE_COUNT_MAX + 1, None]
        with kb.write_txn(conn):
            for value in accepted + rejected:
                kb._append_event(conn, tid, "heartbeat", failure_count=value)
    finally:
        conn.close()

    got = events[: len(accepted)]
    assert [e["failure_count"] for e in got] == accepted
    for e in events[len(accepted):]:
        assert "failure_count" not in e


def test_c1n_t28_failure_count_absent_by_default(kanban_home, events):
    conn = kb.connect()
    try:
        kb.create_task(conn, title="t")
    finally:
        conn.close()
    assert all("failure_count" not in e for e in events)


def test_c1n_t28_gave_up_carries_the_bounded_local_integer(kanban_home, events):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t")
        assert kb.claim_task(conn, tid, claimer="w:1") is not None
        events.clear()
        # force_trip drives the gave_up branch on the first failure.
        assert kb._record_task_failure(
            conn, tid, "boom", outcome="spawn_failed",
            force_trip=True, release_claim=True, end_run=True,
        ) is True
        gave_up = only(events, "gave_up")
        assert gave_up["failure_count"] == 1
        assert gave_up["status_to"] == "blocked"
        # No payload-derived content rides along.
        for banned in ("error", "payload", "trigger_outcome", "reason"):
            assert banned not in gave_up
    finally:
        conn.close()


def test_c1n_t24_dynamic_outcome_domain_reaches_the_seam(kanban_home, events):
    """``spawn_failed`` and ``timed_out`` reach the dynamic append; ``crashed``
    with ``end_run=False`` correctly does not (D-4)."""
    conn = kb.connect()
    try:
        for outcome in ("spawn_failed", "timed_out"):
            tid = kb.create_task(conn, title=outcome)
            assert kb.claim_task(conn, tid, claimer="w:1") is not None
            events.clear()
            assert kb._record_task_failure(
                conn, tid, "boom", outcome=outcome,
                failure_limit=99, release_claim=True, end_run=True,
            ) is False
            kw = only(events, outcome)
            assert kw["kind"] in kb.KANBAN_EVENT_KINDS
            assert kw["status_to"] == "ready"
            assert kw["failure_count"] == 1

        # end_run=False never reaches the dynamic append.
        tid = kb.create_task(conn, title="crashed")
        assert kb.claim_task(conn, tid, claimer="w:1") is not None
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        conn.commit()
        events.clear()
        kb._record_task_failure(
            conn, tid, "boom", outcome="crashed",
            failure_limit=99, release_claim=False, end_run=False,
        )
        assert [e["kind"] for e in events] == []
    finally:
        conn.close()


# --------------------------------------------------------------------------
# C1N-T04 — capture happens at INSERT time, never by post-commit re-read
# --------------------------------------------------------------------------


def test_c1n_t04_later_mutation_cannot_change_an_earlier_events_scalars(
    kanban_home, hooks
):
    """A competing post-commit mutation/deletion must not rewrite what was
    already captured — because nothing is re-read after commit."""
    captured: list[dict] = []

    def _cb(**kw):
        captured.append(dict(kw))
        # Mutate and then delete the very row/task this callback describes.
        other = kb.connect()
        try:
            other.execute("UPDATE tasks SET status='archived' WHERE id=?",
                          (kw["task_id"],))
            other.execute("UPDATE task_events SET kind='rewritten' WHERE id=?",
                          (kw["core_event_seq"],))
            other.execute("DELETE FROM task_events WHERE id=?",
                          (kw["core_event_seq"],))
            other.commit()
        finally:
            other.close()

    hooks(GENERIC_HOOK, _cb)
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t")
        assert rows(conn, tid) == []  # the callback deleted the row
    finally:
        conn.close()

    assert len(captured) == 1
    assert captured[0]["kind"] == "created"
    assert captured[0]["status_to"] == "ready"
    assert isinstance(captured[0]["core_event_seq"], int)
    assert captured[0]["task_id"] == tid


# --------------------------------------------------------------------------
# C1N-T19 — a writer outside kanban_db.py reaches the same seam
# --------------------------------------------------------------------------


def test_c1n_t19_out_of_module_iteration_budget_path(kanban_home, events):
    """``agent/turn_finalizer``'s iteration-budget path, reproduced exactly.

    The finalizer opens its own connection and relies on
    ``_record_task_failure`` to open ``write_txn``; it needs no C-1 edit but is
    inside the writer closure. This test drives that exact call shape rather
    than a whole agent turn; that the finalizer really is this caller is
    asserted statically in ``test_kanban_event_closure.py``.
    """
    setup = kb.connect()
    try:
        tid = kb.create_task(setup, title="t")
        assert kb.claim_task(setup, tid, claimer="w:1") is not None
    finally:
        setup.close()
    events.clear()

    conn = kb.connect()  # a fresh connection, as the finalizer does
    try:
        kb._record_task_failure(
            conn,
            tid,
            error="Iteration budget exhausted (40/40)",
            outcome="timed_out",
            failure_limit=99,
            release_claim=True,
            end_run=True,
            event_payload_extra={"budget_used": 40, "budget_max": 40},
        )
    finally:
        conn.close()

    kw = only(events, "timed_out")
    assert kw["status_to"] == "ready"
    assert kw["failure_count"] == 1
    assert "error" not in kw and "budget_used" not in kw


# --------------------------------------------------------------------------
# C1N-T20 — three process surfaces, each asserting LOCAL delivery only
# --------------------------------------------------------------------------


def test_c1n_t20_each_process_surface_delivers_locally(kanban_home, events, dash):
    """Dispatcher-, worker/manual- and web-server-side writers each deliver in
    the process that performed the write.

    Deliberately makes no claim about arrival order ACROSS surfaces — there is
    no cross-process broker, and none is requested.
    """
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t")

        # Dispatcher-side surface: recovery/claim bookkeeping.
        events.clear()
        assert kb.claim_task(conn, tid, claimer="w:1") is not None
        assert {e["kind"] for e in events} == {"claimed"}

        # Worker/manual surface: the terminal verb.
        events.clear()
        kb.complete_task(conn, tid, result="ok")
        assert "completed" in {e["kind"] for e in events}
    finally:
        conn.close()

    # Web-server surface: the bundled dashboard's plugin API.
    events.clear()
    conn = kb.connect()
    try:
        other = kb.create_task(conn, title="dash")
    finally:
        conn.close()
    events.clear()
    dash.update_task(other, dash.UpdateTaskBody(priority=2), board=None)
    assert [e["kind"] for e in events] == ["reprioritized"]


# --------------------------------------------------------------------------
# C1N-T22 / C1N-T23 — reconciled, and the dynamic crash domain
# --------------------------------------------------------------------------


def test_c1n_t22_reconciled_is_declared_and_reports_ready(kanban_home, events):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t")
        conn.execute(
            "UPDATE tasks SET status='running', claim_lock=NULL, "
            "claim_expires=NULL WHERE id=?", (tid,),
        )
        conn.commit()
        events.clear()
        assert kb.reconcile_orphaned_running(conn) == [tid]
    finally:
        conn.close()

    kw = only(events, "reconciled")
    assert kw["kind"] in kb.KANBAN_EVENT_KINDS
    assert kw["status_to"] == "ready"


@pytest.mark.parametrize(
    "exit_code, expected_kind",
    [
        (0, "protocol_violation"),
        (None, "rate_limited"),   # replaced with the quota sentinel below
        (3, "crashed"),
    ],
)
def test_c1n_t23_crash_classifier_kinds_dispatch_as_declared(
    kanban_home, events, monkeypatch, exit_code, expected_kind
):
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    if expected_kind == "rate_limited":
        exit_code = kb.KANBAN_RATE_LIMIT_EXIT_CODE

    conn = kb.connect()
    try:
        host = kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="c")
        assert kb.claim_task(conn, tid, claimer=f"{host}:w") is not None
        pid = 90001
        conn.execute("UPDATE tasks SET worker_pid=? WHERE id=?", (pid, tid))
        conn.commit()
        kb._record_worker_exit(pid, exit_code << 8)
        events.clear()
        kb.detect_crashed_workers(conn)
    finally:
        conn.close()

    kw = only(events, expected_kind)
    assert kw["kind"] in kb.KANBAN_EVENT_KINDS
    assert kw["status_to"] == "ready"


# --------------------------------------------------------------------------
# C1N-T25 / C1N-T39 — unknown kinds pass through, and what that costs
# --------------------------------------------------------------------------


def test_c1n_t25_unknown_future_kind_dispatches_unchanged(kanban_home, events):
    """Core never drops, rewrites, or coerces an undeclared kind."""
    kind = "some_future_kind_from_a_later_release"
    assert kind not in kb.KANBAN_EVENT_KINDS
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t")
        events.clear()
        with kb.write_txn(conn):
            kb._append_event(conn, tid, kind)
        committed = [r for r in rows(conn, tid) if r["kind"] == kind]
    finally:
        conn.close()

    assert [e["kind"] for e in events] == [kind]
    assert committed and committed[0]["kind"] == kind


def test_c1n_t39_unknown_kind_privacy_is_convention_bounded_not_enforced(
    kanban_home, events
):
    """DOCUMENTED LIMITATION, not a promise — do not "fix" this by coercing kinds.

    ``task_events.kind`` is an open TEXT column and undeclared kinds are
    dispatched unchanged. Those two facts compose: an out-of-tree writer that
    puts arbitrary prose in ``kind`` has that string delivered VERBATIM to every
    observer. C-1 cannot prevent this without dropping or rewriting kinds,
    which the contract forbids. The guarantee C-1 actually makes is narrower:
    it introduces no content channel of its own and copies no payload into one.
    """
    prose = "user said: my password is hunter2 /home/someone/secret.txt"
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t")
        events.clear()
        with kb.write_txn(conn):
            kb._append_event(conn, tid, prose, {"reason": "not delivered"})
    finally:
        conn.close()

    assert [e["kind"] for e in events] == [prose]
    # The payload channel stays closed even here.
    assert "reason" not in events[0] and "payload" not in events[0]


# --------------------------------------------------------------------------
# C1N-T29 — content exclusion, against the real prose-bearing writers
# --------------------------------------------------------------------------


BANNED_KWARGS = {
    "payload", "reason", "error", "summary", "summary_preview", "title",
    "body", "message", "note", "filename", "author", "claimer", "pid",
    "actor", "assignee", "by", "priority", "parent", "child_ids",
}


def test_c1n_t29_no_payload_content_reaches_the_callback(kanban_home, events):
    secret_reason = "SECRET-REASON-cbd41"
    secret_summary = "SECRET-SUMMARY-9f2aa"
    secret_comment = "SECRET-COMMENT-77bcd"

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="SECRET-TITLE", body="SECRET-BODY")
        kb.add_comment(conn, tid, "SECRET-AUTHOR", secret_comment)
        assert kb.claim_task(conn, tid, claimer="SECRET-CLAIMER:1") is not None
        assert kb.block_task(conn, tid, reason=secret_reason, kind="needs_input")
        assert kb.unblock_task(conn, tid)
        assert kb.claim_task(conn, tid, claimer="SECRET-CLAIMER:2") is not None
        kb.complete_task(conn, tid, result="SECRET-RESULT", summary=secret_summary)
    finally:
        conn.close()

    assert len(events) >= 5
    allowed = {
        "task_id", "kind", "core_event_seq", "created_at_epoch_s", "run_id",
        "board", "profile_name", "status_to", "failure_count",
        "kanban_event_schema_version", "telemetry_schema_version",
    }
    haystack = " ".join(
        str(v) for e in events for v in e.values()
    )
    for e in events:
        assert set(e) <= allowed, f"unexpected kwargs: {set(e) - allowed}"
        assert not (set(e) & BANNED_KWARGS)
    for secret in (
        "SECRET-TITLE", "SECRET-BODY", "SECRET-AUTHOR", "SECRET-CLAIMER",
        "SECRET-RESULT", secret_reason, secret_summary, secret_comment,
    ):
        assert secret not in haystack


# --------------------------------------------------------------------------
# C1N-T30 / C1N-T33 — legacy coexistence and R-1 independence
# --------------------------------------------------------------------------


def dependency_branch_is_precommit():
    """Probe THIS base: does the dependency branch dispatch inside write_txn?

    Parameterizing on the base is the whole point — hard-coding the frozen
    base's asymmetry would fail the moment R-1 lands independently, and
    hard-coding harmonized timing fails here. Either way the failure would be
    the test's, not C-1's.
    """
    import ast
    import inspect

    src = inspect.getsource(kb.block_task)
    tree = ast.parse(src.lstrip() if src.startswith(" ") else src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        opens_txn = any(
            isinstance(item.context_expr, ast.Call)
            and getattr(item.context_expr.func, "id", None) == "write_txn"
            for item in node.items
        )
        if not opens_txn:
            continue
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and getattr(child.func, "id", None) == "_fire_kanban_lifecycle_hook"
                and child.args
                and isinstance(child.args[0], ast.Constant)
                and child.args[0].value == "kanban_task_blocked"
            ):
                return True
    return False


def test_c1n_t30_legacy_timing_is_preserved_exactly_as_the_base_has_it(
    kanban_home, hooks
):
    """C-1 changes NO legacy timing — asserted parameterized on the base."""
    timeline: list[tuple[str, bool]] = []
    probe = kb.connect()

    def _in_txn():
        # The generic hook is post-commit on every path, so a competing
        # BEGIN IMMEDIATE succeeding proves the write lock is gone.
        try:
            probe.execute("BEGIN IMMEDIATE")
            probe.execute("ROLLBACK")
            return False
        except Exception:
            return True

    hooks("kanban_task_blocked", lambda **kw: timeline.append(("legacy", _in_txn())))
    hooks(GENERIC_HOOK, lambda **kw: timeline.append((f"generic:{kw['kind']}", _in_txn())))

    conn = kb.connect()
    try:
        conn.execute("PRAGMA busy_timeout=500")
        probe.execute("PRAGMA busy_timeout=500")
        tid = kb.create_task(conn, title="t")
        timeline.clear()
        assert kb.block_task(conn, tid, reason="dep", kind="dependency") is True
    finally:
        conn.close()
        probe.close()

    legacy = [t for t in timeline if t[0] == "legacy"]
    generic = [t for t in timeline if t[0].startswith("generic:")]
    assert len(legacy) == 1, "the legacy hook must keep firing exactly once"
    assert [t[0] for t in generic] == ["generic:dependency_wait"]

    # The generic hook is post-commit unconditionally, on every base.
    assert generic[0][1] is False

    if dependency_branch_is_precommit():
        # Frozen base, R-1 not applied: legacy fires while the lock is held.
        assert legacy[0][1] is True
        assert timeline.index(legacy[0]) < timeline.index(generic[0])
    else:
        # A base or fixture with R-1 applied: both are post-commit.
        assert legacy[0][1] is False


def test_c1n_t33_c1_contains_no_r1_change(kanban_home):
    """C-1 must stay droppable without disturbing R-1's published PR.

    It neither moves the dependency branch's dispatch site nor retimes any
    legacy hook — so this candidate's suite is valid on a base with or without
    R-1, which is exactly why C1N-T30 probes instead of hard-coding.
    """
    import inspect

    src = inspect.getsource(kb.block_task)
    # The dependency branch's legacy dispatch is untouched: still exactly one
    # legacy blocked-hook call per block path, and C-1 added none.
    assert src.count('_fire_kanban_lifecycle_hook(') == 2
    assert 'kanban_task_event' not in src
    # The generic hook is never dispatched from a legacy call site.
    for fn in (kb.claim_task, kb.complete_task, kb.block_task):
        assert 'kanban_task_event' not in inspect.getsource(fn)


def test_c1n_t30_legacy_hooks_keep_names_kwargs_and_cardinality(
    kanban_home, hooks, events
):
    seen: list[tuple[str, dict]] = []
    for name in ("kanban_task_claimed", "kanban_task_completed", "kanban_task_blocked"):
        hooks(name, lambda _n=name, **kw: seen.append((_n, kw)))

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t")
        assert kb.claim_task(conn, tid, claimer="w:1") is not None
        assert kb.block_task(conn, tid, reason="r", kind="needs_input")
        assert kb.unblock_task(conn, tid)
        assert kb.claim_task(conn, tid, claimer="w:2") is not None
        kb.complete_task(conn, tid, result="ok")
    finally:
        conn.close()

    fired = [n for n, _ in seen]
    assert fired.count("kanban_task_claimed") == 2
    assert fired.count("kanban_task_blocked") == 1
    assert fired.count("kanban_task_completed") == 1
    for name, kw in seen:
        for key in ("task_id", "board", "assignee", "run_id", "profile_name"):
            assert key in kw, f"{name} lost kwarg {key}"
        if name == "kanban_task_completed":
            assert "summary" in kw
        if name == "kanban_task_blocked":
            assert "reason" in kw
    # A row that fires a legacy hook also fires one generic notification; core
    # does not deduplicate. That exclusivity is the consumer's job.
    assert any(e["kind"] == "completed" for e in events)


# --------------------------------------------------------------------------
# C1N-T31 / T32 / T38 — sequence honesty
# --------------------------------------------------------------------------


def test_c1n_t31_gaps_after_deletion_and_gc_are_not_hook_loss(kanban_home, events):
    conn = kb.connect()
    try:
        keep = kb.create_task(conn, title="keep")
        doomed = kb.create_task(conn, title="doomed")
        with kb.write_txn(conn):
            kb._append_event(conn, doomed, "commented")
            kb._append_event(conn, keep, "commented")
            kb._append_event(conn, doomed, "edited")
        delivered = [e["core_event_seq"] for e in events]
        assert delivered == sorted(delivered)
        assert kb.archive_task(conn, doomed)
        assert kb.delete_archived_task(conn, doomed) is True
        surviving = [r["id"] for r in rows(conn)]
    finally:
        conn.close()

    # Every surviving row was delivered, but not every delivered row survived.
    assert set(surviving) < set(delivered)
    # Non-contiguous survivors: a gap is a deletion, never evidence of loss.
    assert surviving != list(range(surviving[0], surviving[0] + len(surviving)))


def test_c1n_t32_core_event_seq_is_not_stable_across_a_rebuild(
    kanban_home, events, tmp_path
):
    """``_rebuild_drifted_tables`` excludes the legacy id, so AUTOINCREMENT
    reassigns fresh values. No test may treat the seq as stable."""
    db = tmp_path / "drifted" / "kanban.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    import sqlite3

    raw = sqlite3.connect(db)
    raw.execute(
        "CREATE TABLE task_events (id TEXT PRIMARY KEY, task_id TEXT NOT NULL, "
        "run_id INTEGER, kind TEXT NOT NULL, payload TEXT, "
        "created_at INTEGER NOT NULL)"
    )
    raw.execute(
        "INSERT INTO task_events VALUES ('legacy-1','t1',NULL,'created',NULL,10)"
    )
    raw.commit()
    raw.close()

    events.clear()
    conn = kb.connect(db_path=db)
    try:
        after = [dict(r) for r in conn.execute(
            "SELECT id, kind FROM task_events"
        ).fetchall()]
    finally:
        conn.close()

    assert after == [{"id": 1, "kind": "created"}]  # id reassigned, not 'legacy-1'
    assert events == [], "a migration copy must emit nothing"


def test_c1n_t38_board_generation_resets_and_repeats_the_sequence(
    kanban_home, events
):
    """A repeat is a NEW BOARD GENERATION, not duplicate delivery or hook loss."""
    for archive in (True, False):
        slug = f"gen-{int(archive)}"
        kb.create_board(slug)
        conn = kb.connect(board=slug)
        try:
            events.clear()
            kb.create_task(conn, title="first")
            first = [e["core_event_seq"] for e in events]
        finally:
            conn.close()
        assert first == [1]

        kb.remove_board(slug, archive=archive)
        kb.create_board(slug)
        conn = kb.connect(board=slug)
        try:
            events.clear()
            kb.create_task(conn, title="second")
            second = [e["core_event_seq"] for e in events]
        finally:
            conn.close()
        assert second == [1], "a same-slug board is a new generation from 1"


# --------------------------------------------------------------------------
# C1N-T36 / C1N-T37 — migration writers are invisible to C-1
# --------------------------------------------------------------------------


def test_c1n_t36_rebuild_of_a_drifted_table_emits_nothing(
    kanban_home, events, tmp_path
):
    import sqlite3

    db = tmp_path / "rebuild" / "kanban.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    raw = sqlite3.connect(db)
    raw.execute(
        "CREATE TABLE task_events (id TEXT PRIMARY KEY, task_id TEXT NOT NULL, "
        "run_id INTEGER, kind TEXT NOT NULL, payload TEXT, "
        "created_at INTEGER NOT NULL)"
    )
    for i in range(5):
        raw.execute(
            "INSERT INTO task_events VALUES (?,?,NULL,'created',NULL,?)",
            (f"legacy-{i}", f"t{i}", 100 + i),
        )
    raw.commit()
    raw.close()

    events.clear()
    conn = kb.connect(db_path=db)
    try:
        moved = conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0]
        assert moved == 5, "the rebuild must have copied every row"
        assert kb._TASK_EVENT_FRAME.get() is None, "no frame may survive migration"
    finally:
        conn.close()

    assert events == []


def test_c1n_t37_event_kind_rename_emits_nothing_and_breaks_seq_kind_stability(
    kanban_home, events, tmp_path
):
    """``(core_event_seq, kind)`` is NOT stable across migration (D-18).

    The rename mutates existing rows in place, so the id is unchanged while the
    kind a consumer already observed reads back differently.
    """
    import sqlite3

    db = tmp_path / "renames" / "kanban.db"
    db.parent.mkdir(parents=True, exist_ok=True)

    # First open creates the current schema.
    conn = kb.connect(db_path=db)
    try:
        tid = kb.create_task(conn, title="t")
        events.clear()
        with kb.write_txn(conn):
            kb._append_event(conn, tid, "gave_up")
        seq = events[0]["core_event_seq"]
        # Put the row back into its legacy shape, as an old board would have it.
        conn.execute(
            "UPDATE task_events SET kind='spawn_auto_blocked' WHERE id=?", (seq,)
        )
        conn.commit()
    finally:
        conn.close()

    # Force the one-shot migration pass to run again on this path.
    kb._INITIALIZED_PATHS.discard(str(db.resolve()))
    events.clear()
    conn = kb.connect(db_path=db)
    try:
        after = conn.execute(
            "SELECT kind FROM task_events WHERE id=?", (seq,)
        ).fetchone()
    finally:
        conn.close()

    assert events == [], "an in-place kind rewrite must emit nothing"
    assert after["kind"] == "gave_up"
    # Same id, different kind than the row a consumer may have stored.


# --------------------------------------------------------------------------
# C1N-T27 (continued) — the two D-9 writers inside ``_record_task_failure``
#
# The packet names FOUR unchecked writers plus the dynamic-outcome UPDATE.
# ``promoted``, ``linked`` and ``claim_rejected`` are driven above; the two
# remaining ones both live in ``_record_task_failure`` and each has two
# guarded branches (``release_claim=True`` and the else branch), so all four
# combinations are driven here.
# --------------------------------------------------------------------------


def _park(conn, task_id, status):
    """Move a task out of a guarded UPDATE's ``WHERE`` status, out of band."""
    conn.execute("UPDATE tasks SET status=? WHERE id=?", (status, task_id))
    conn.commit()


def test_c1n_t27_gave_up_release_claim_branch_is_rowcount_gated(
    kanban_home, events
):
    """D-9: ``gave_up``'s ``release_claim=True`` UPDATE is guarded
    ``WHERE id = ? AND status IN ('running', 'ready')``.

    Positive and zero-match are driven through the *same* branch so the
    difference is the rowcount and nothing else.
    """
    conn = kb.connect()
    try:
        # Matched: the task really is 'running'.
        hit = kb.create_task(conn, title="hit")
        assert kb.claim_task(conn, hit, claimer="w:1") is not None
        events.clear()
        assert kb._record_task_failure(
            conn, hit, "boom", outcome="spawn_failed",
            force_trip=True, release_claim=True, end_run=True,
        ) is True
        assert only(events, "gave_up", hit)["status_to"] == "blocked"
        assert kb.get_task(conn, hit).status == "blocked"

        # Zero match: the task moved out of ('running', 'ready') first. The
        # append still happens — only the status claim must disappear.
        miss = kb.create_task(conn, title="miss")
        _park(conn, miss, "triage")
        events.clear()
        assert kb._record_task_failure(
            conn, miss, "boom", outcome="spawn_failed",
            force_trip=True, release_claim=True, end_run=True,
        ) is True
        gave_up = only(events, "gave_up", miss)
        assert "status_to" not in gave_up
        # The row genuinely did not move, which is exactly why the claim
        # would have been a lie.
        assert kb.get_task(conn, miss).status == "triage"
        # The bounded integer is unaffected by the rowcount gate.
        assert gave_up["failure_count"] == 1
    finally:
        conn.close()


def test_c1n_t27_gave_up_else_branch_is_rowcount_gated(kanban_home, events):
    """D-9: the ``release_claim=False`` branch is guarded
    ``WHERE id = ? AND status IN ('ready', 'running')`` — the timeout/crash
    path, where the caller has already parked the task at ``ready``.
    """
    conn = kb.connect()
    try:
        hit = kb.create_task(conn, title="hit")   # created 'ready'
        assert kb.get_task(conn, hit).status == "ready"
        events.clear()
        assert kb._record_task_failure(
            conn, hit, "boom", outcome="timed_out",
            force_trip=True, release_claim=False, end_run=False,
        ) is True
        assert only(events, "gave_up", hit)["status_to"] == "blocked"

        miss = kb.create_task(conn, title="miss")
        _park(conn, miss, "todo")
        events.clear()
        assert kb._record_task_failure(
            conn, miss, "boom", outcome="timed_out",
            force_trip=True, release_claim=False, end_run=False,
        ) is True
        assert "status_to" not in only(events, "gave_up", miss)
        assert kb.get_task(conn, miss).status == "todo"
    finally:
        conn.close()


def test_c1n_t27_dynamic_outcome_requeue_is_rowcount_gated(kanban_home, events):
    """D-9: the below-threshold dynamic-outcome append reports ``ready`` only
    when the ``WHERE id = ? AND status = 'running'`` requeue actually matched.
    """
    conn = kb.connect()
    try:
        miss = kb.create_task(conn, title="miss")
        assert kb.claim_task(conn, miss, claimer="w:1") is not None
        # Move it out of 'running' so the guarded requeue matches nothing,
        # while end_run=True still drives the dynamic append.
        _park(conn, miss, "review")
        events.clear()
        assert kb._record_task_failure(
            conn, miss, "boom", outcome="spawn_failed",
            failure_limit=99, release_claim=True, end_run=True,
        ) is False
        kw = only(events, "spawn_failed", miss)
        assert "status_to" not in kw
        assert kw["failure_count"] == 1
        assert kb.get_task(conn, miss).status == "review"
    finally:
        conn.close()


def test_c1n_t27_dynamic_outcome_non_release_branch_omits_status_to(
    kanban_home, events
):
    """The else branch bookkeeps the counter only — it establishes no status,
    so ``status_to`` is omitted even though the task is sitting at ``ready``.

    Asserting the writer's *own* honesty, not the row's current value: a
    status the writer did not establish is never reported.
    """
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t")
        assert kb.get_task(conn, tid).status == "ready"
        events.clear()
        assert kb._record_task_failure(
            conn, tid, "boom", outcome="timed_out",
            failure_limit=99, release_claim=False, end_run=True,
        ) is False
        kw = only(events, "timed_out", tid)
        assert "status_to" not in kw
        assert kw["failure_count"] == 1
        assert kb.get_task(conn, tid).status == "ready"
    finally:
        conn.close()


# --------------------------------------------------------------------------
# C1N-T26 (continued) — the §6.6 rows whose writers the matrix above does not
# reach: claim_review_task, release_stale_claims, enforce_max_runtime and
# detect_stale_running.
# --------------------------------------------------------------------------


def test_c1n_t26_claim_review_task_reports_running(kanban_home, events):
    """The second ``claimed`` writer (review → running) reports ``running``."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t")
        assert kb.claim_task(conn, tid, claimer="w:1") is not None
        _park(conn, tid, "review")
        conn.execute(
            "UPDATE tasks SET claim_lock=NULL, claim_expires=NULL WHERE id=?",
            (tid,),
        )
        conn.commit()
        events.clear()
        assert kb.claim_review_task(conn, tid, claimer="reviewer:1") is not None
        kw = only(events, "claimed", tid)
        assert kw["status_to"] == "running"
        assert kw["kind"] in kb.KANBAN_EVENT_KINDS
        # source_status rides only on the payload, never on the kwargs.
        assert "source_status" not in kw and "lock" not in kw
    finally:
        conn.close()


def test_c1n_t26_release_stale_claims_reports_ready(kanban_home, events):
    """The ``reclaimed`` writer inside ``release_stale_claims`` (:4591 in the
    packet's inventory) reports ``ready``; its ``claim_extended`` sibling in
    the same function establishes no status and must omit it."""
    conn = kb.connect()
    try:
        host = kb._claimer_id().split(":", 1)[0]

        # (a) expired claim, no live worker -> reclaimed
        dead = kb.create_task(conn, title="dead")
        assert kb.claim_task(conn, dead, claimer=f"{host}:w1") is not None
        conn.execute(
            "UPDATE tasks SET claim_expires=?, worker_pid=NULL WHERE id=?",
            (1, dead),
        )
        conn.commit()
        events.clear()
        assert kb.release_stale_claims(conn) == 1
        assert only(events, "reclaimed", dead)["status_to"] == "ready"
        assert kb.get_task(conn, dead).status == "ready"
    finally:
        conn.close()


def test_c1n_t26_claim_extended_omits_status_to(kanban_home, events, monkeypatch):
    """A live host-local worker gets its claim extended — no status changes."""
    conn = kb.connect()
    try:
        host = kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="live")
        assert kb.claim_task(conn, tid, claimer=f"{host}:w1") is not None
        conn.execute(
            "UPDATE tasks SET claim_expires=?, worker_pid=?, "
            "last_heartbeat_at=? WHERE id=?",
            (1, 4242, int(time.time()), tid),
        )
        conn.commit()
        monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)
        events.clear()
        assert kb.release_stale_claims(conn) == 0
        kw = only(events, "claim_extended", tid)
        assert "status_to" not in kw
        assert kb.get_task(conn, tid).status == "running"
        for banned in ("reason", "claim_lock", "worker_pid"):
            assert banned not in kw
    finally:
        conn.close()


def test_c1n_t26_enforce_max_runtime_reports_ready(kanban_home, events, monkeypatch):
    """``timed_out`` (:7284 in the inventory) reports ``ready``."""
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    conn = kb.connect()
    try:
        host = kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="slow")
        assert kb.claim_task(conn, tid, claimer=f"{host}:w1") is not None
        conn.execute(
            "UPDATE tasks SET worker_pid=?, max_runtime_seconds=1, "
            "started_at=? WHERE id=?",
            (4242, 1, tid),
        )
        conn.execute("UPDATE task_runs SET started_at=? WHERE task_id=?", (1, tid))
        conn.commit()
        events.clear()
        assert kb.enforce_max_runtime(conn, signal_fn=lambda *_a: None) == [tid]
        kw = only(events, "timed_out", tid)
        assert kw["status_to"] == "ready"
        assert kw["kind"] in kb.KANBAN_EVENT_KINDS
        for banned in ("pid", "elapsed_seconds", "sigkill"):
            assert banned not in kw
    finally:
        conn.close()


def test_c1n_t26_detect_stale_running_reports_ready(kanban_home, events, monkeypatch):
    """``stale`` (:7422 in the inventory) reports ``ready``."""
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    conn = kb.connect()
    try:
        host = kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="wedged")
        assert kb.claim_task(conn, tid, claimer=f"{host}:w1") is not None
        conn.execute(
            "UPDATE tasks SET started_at=?, last_heartbeat_at=NULL, "
            "worker_pid=NULL WHERE id=?", (1, tid),
        )
        conn.execute("UPDATE task_runs SET started_at=? WHERE task_id=?", (1, tid))
        conn.commit()
        events.clear()
        assert kb.detect_stale_running(
            conn, stale_timeout_seconds=60, signal_fn=lambda *_a: None,
        ) == [tid]
        assert only(events, "stale", tid)["status_to"] == "ready"
        assert kb.get_task(conn, tid).status == "ready"
    finally:
        conn.close()


def test_c1n_t26_reclaim_deferred_omits_status_to(kanban_home, events, monkeypatch):
    """A deferred reclaim holds the claim — the task stays ``running`` and the
    writer establishes nothing."""
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: True)
    conn = kb.connect()
    try:
        host = kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="survivor")
        assert kb.claim_task(conn, tid, claimer=f"{host}:w1") is not None
        conn.execute(
            "UPDATE tasks SET started_at=?, last_heartbeat_at=NULL, "
            "worker_pid=? WHERE id=?", (1, 4242, tid),
        )
        conn.execute("UPDATE task_runs SET started_at=? WHERE task_id=?", (1, tid))
        conn.commit()
        events.clear()
        assert kb.detect_stale_running(
            conn, stale_timeout_seconds=60, signal_fn=lambda *_a: None,
        ) == []
        kw = only(events, "reclaim_deferred", tid)
        assert "status_to" not in kw
        assert kb.get_task(conn, tid).status == "running"
    finally:
        conn.close()


# --------------------------------------------------------------------------
# C1N-T26 (continued) — the remaining §6.6 "omitted" writers, driven for real
# --------------------------------------------------------------------------


def test_c1n_t26_attachment_writers_omit_status_to(kanban_home, events):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t")
        events.clear()
        att = kb.add_attachment(
            conn, tid, filename="secret-notes.txt",
            stored_path=str(kanban_home / "secret-notes.txt"),
            size=11, uploaded_by="SECRET-UPLOADER",
        )
        assert kb.delete_attachment(conn, att) is not None
    finally:
        conn.close()

    assert [e["kind"] for e in events] == ["attached", "attachment_removed"]
    for e in events:
        assert "status_to" not in e
        assert "filename" not in e and "by" not in e
    assert "secret-notes.txt" not in " ".join(
        str(v) for e in events for v in e.values()
    )


def test_c1n_t26_edit_completed_result_omits_status_to(kanban_home, events):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t")
        assert kb.claim_task(conn, tid, claimer="w:1") is not None
        kb.complete_task(conn, tid, result="first")
        events.clear()
        assert kb.edit_completed_task_result(
            conn, tid, result="revised", summary="SECRET-EDIT-SUMMARY",
        ) is True
    finally:
        conn.close()

    kw = only(events, "edited", tid)
    assert "status_to" not in kw
    assert "summary" not in kw and "fields" not in kw
    assert "SECRET-EDIT-SUMMARY" not in " ".join(str(v) for v in kw.values())


def test_c1n_t26_spawn_and_scratch_tip_writers_omit_status_to(
    kanban_home, events, monkeypatch
):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t")
        assert kb.claim_task(conn, tid, claimer="w:1") is not None
        events.clear()
        kb._set_worker_pid(conn, tid, 4242)
        monkeypatch.setattr(kb, "_scratch_tip_shown", lambda: False)
        monkeypatch.setattr(kb, "_mark_scratch_tip_shown", lambda: None)
        kb._maybe_emit_scratch_tip(conn, tid, "scratch")
    finally:
        conn.close()

    kinds = [e["kind"] for e in events]
    assert "spawned" in kinds and "tip_scratch_workspace" in kinds
    for e in events:
        assert "status_to" not in e
        assert "pid" not in e and "message" not in e


def test_c1n_t26_hallucination_advisory_writers_omit_status_to(kanban_home, events):
    """Both completion-advisory writers establish no status of their own."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t")
        assert kb.claim_task(conn, tid, claimer="w:1") is not None
        events.clear()
        with pytest.raises(kb.HallucinatedCardsError):
            kb.complete_task(
                conn, tid, result="ok", created_cards=["t_deadbeefcafe"],
            )
        blocked = only(events, "completion_blocked_hallucination", tid)
        assert "status_to" not in blocked
        assert "phantom_cards" not in blocked and "summary_preview" not in blocked
        assert kb.get_task(conn, tid).status == "running"

        events.clear()
        kb.complete_task(
            conn, tid, result="see t_deadbeefcafe for the rest", summary="done",
        )
        suspected = only(events, "suspected_hallucinated_references", tid)
        assert "status_to" not in suspected
        assert "refs" not in suspected
        # The sibling 'completed' row in the same transaction still reports its
        # own honest destination.
        assert only(events, "completed", tid)["status_to"] == "done"
    finally:
        conn.close()


def test_c1n_t26_dashboard_reprioritized_and_edited_omit_status_to(
    kanban_home, events, dash
):
    """The two dashboard writers that change no status omit ``status_to``,
    unlike the dashboard ``status`` writer covered by C1N-T17/T18."""
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="t")
    finally:
        conn.close()
    events.clear()

    dash.update_task(tid, dash.UpdateTaskBody(priority=5), board=None)
    dash.update_task(tid, dash.UpdateTaskBody(body="new body"), board=None)

    assert [e["kind"] for e in events] == ["reprioritized", "edited"]
    for e in events:
        assert "status_to" not in e


# --------------------------------------------------------------------------
# C1N-T35 (continued) — the resolver fails SAFE, never fails the transaction
# --------------------------------------------------------------------------


class _NotASqliteConnection:
    """A connection double, as several existing kanban tests already use.

    ``execute`` returns ``None`` for anything it does not model, so a
    ``PRAGMA database_list`` probe gets an object with no ``fetchall``. The
    resolver must treat that as "this connection maps to no board", exactly
    like an erroring PRAGMA — it must not raise out of ``write_txn`` and take
    an unrelated write down with it.
    """

    def __init__(self):
        self.calls = []
        self.in_transaction = False

    def execute(self, sql, *args):
        self.calls.append(sql)
        return None


def test_c1n_t35_resolver_returns_none_when_the_pragma_probe_is_unusable():
    """§6.2a: ``None`` on ANY failure, not only on ``sqlite3.Error``.

    The resolver is best-effort board *attribution*. It runs at frame install,
    after BEGIN IMMEDIATE has already succeeded, so a failure to attribute must
    degrade to an honest ``None`` and never propagate — otherwise C-1 would
    convert a cosmetic attribution problem into a failed write.
    """
    conn = _NotASqliteConnection()
    assert kb._resolve_board_for_connection(conn) is None


def test_c1n_t35_resolver_returns_none_when_the_pragma_raises_any_exception():
    class _Boom:
        def execute(self, sql, *args):
            raise RuntimeError("connection wrapper blew up")

    assert kb._resolve_board_for_connection(_Boom()) is None


def test_c1n_t35_frame_install_survives_an_unattributable_connection(
    kanban_home, monkeypatch
):
    """A write_txn over a connection double still opens, commits, and clears.

    This is the shape several pre-existing kanban tests use to exercise
    ``write_txn``'s busy-retry boundary; C-1 must leave them working. The
    post-commit invariant is stubbed exactly as those tests already stub it —
    it reads its own PRAGMAs and is not what this test is about.
    """
    monkeypatch.setattr(kb, "_check_file_length_invariant", lambda conn: None)
    conn = _NotASqliteConnection()
    with kb.write_txn(conn):
        pass
    assert conn.calls[0] == "BEGIN IMMEDIATE"
    assert "COMMIT" in conn.calls
    assert kb._TASK_EVENT_FRAME.get() is None


# --------------------------------------------------------------------------
# C1N-T40 — zero-subscriber fast path
#
# Cost assertions here are DETERMINISTIC: they count calls to the expensive
# collaborators (board resolution, profile resolution, hook fan-out), never
# wall-clock time. Nothing in this section may assert a duration, a rate, or
# a frozen absolute call total for a production writer.
# --------------------------------------------------------------------------


@pytest.fixture
def no_generic_subscriber(pinned_manager):
    """Guarantee zero ``kanban_task_event`` subscribers for one test."""
    mgr = pinned_manager
    saved = {k: list(v) for k, v in mgr._hooks.items()}
    mgr._hooks.pop(GENERIC_HOOK, None)
    try:
        yield mgr
    finally:
        mgr._hooks = saved


def _count_resolver(monkeypatch):
    """Count ``_resolve_board_for_connection`` calls, preserving behavior."""
    calls: list[object] = []
    real = kb._resolve_board_for_connection

    def counting(conn):
        calls.append(conn)
        return real(conn)

    monkeypatch.setattr(kb, "_resolve_board_for_connection", counting)
    return calls


def _count_profile_resolution(monkeypatch):
    """Count ``get_active_profile_name`` calls made by the hook fan-out."""
    from hermes_cli import profiles as profiles_mod

    calls: list[int] = []
    real = profiles_mod.get_active_profile_name

    def counting(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(profiles_mod, "get_active_profile_name", counting)
    return calls


def test_c1n_t40_no_subscriber_skips_board_resolution(
    kanban_home, no_generic_subscriber, monkeypatch
):
    """With nothing subscribed, write_txn must not resolve the board.

    Board resolution runs ``PRAGMA database_list`` and several ``Path``
    resolutions per transaction. It exists only to attribute a generic
    notification, so with no subscriber it is pure waste on every kanban
    write in the process.
    """
    resolver_calls = _count_resolver(monkeypatch)
    conn = kb.connect()
    try:
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET priority = priority WHERE 1 = 0")
    finally:
        conn.close()

    assert resolver_calls == []


def test_c1n_t40_no_subscriber_skips_dispatch_and_profile_work(
    kanban_home, no_generic_subscriber, monkeypatch
):
    """With nothing subscribed, an appended event does no envelope work.

    ``_append_event`` must still enforce its transaction requirement, but it
    must not build a record, and the post-commit path must not resolve a
    profile or fan out.
    """
    resolver_calls = _count_resolver(monkeypatch)
    profile_calls = _count_profile_resolution(monkeypatch)
    fired: list[tuple] = []
    monkeypatch.setattr(
        kb,
        "_fire_kanban_lifecycle_hook",
        lambda *a, **kw: fired.append((a, kw)),
    )

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="unobserved")
        with kb.write_txn(conn):
            kb._append_event(conn, tid, "status", {"status": "ready"})
        committed = rows(conn, tid)
    finally:
        conn.close()

    # The row still commits — the fast path skips NOTIFICATION work only.
    assert any(r["kind"] == "status" for r in committed)
    assert resolver_calls == []
    assert profile_calls == []
    assert [f for f in fired if f[0][:1] == ("kanban_task_event",)] == []


def test_c1n_t40_no_subscriber_still_fails_closed_outside_a_txn(
    kanban_home, no_generic_subscriber
):
    """The fail-closed seam survives the fast path.

    The zero-subscriber shortcut must not become a back door that lets an
    out-of-transaction ``_append_event`` through.
    """
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="unobserved")
        with pytest.raises(RuntimeError, match="requires an active write_txn"):
            kb._append_event(conn, tid, "status", {"status": "ready"})
    finally:
        conn.close()


def test_c1n_t40_subscriber_still_gets_board_resolution_and_delivery(
    kanban_home, events, monkeypatch
):
    """The fast path must not damage the subscribed case.

    Same transaction as the skip tests, but with a subscriber registered:
    the resolver runs and the event is delivered with its attribution.
    """
    resolver_calls = _count_resolver(monkeypatch)
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="observed")
        with kb.write_txn(conn):
            kb._append_event(conn, tid, "status", {"status": "ready"})
    finally:
        conn.close()

    assert resolver_calls, "board resolution must run when someone subscribes"
    delivered = [e for e in events if e.get("kind") == "status"]
    assert len(delivered) == 1
    assert delivered[0]["task_id"] == tid
    assert delivered[0]["board"] == "default"
    assert delivered[0]["kanban_event_schema_version"] == (
        kb.KANBAN_EVENT_SCHEMA_VERSION
    )


def test_c1n_t41_swarm_root_completion_delivers_status_to_done(
    kanban_home, events
):
    """End-to-end: the swarm root's ``completed`` event carries status_to='done'.

    The AST guard in test_kanban_event_closure asserts the call site passes it;
    this asserts a real subscriber actually receives it, through the swarm's
    attribute-style seam call and its outer write_txn.
    """
    from hermes_cli.kanban_swarm import SwarmWorkerSpec, create_swarm

    conn = kb.connect()
    try:
        created = create_swarm(
            conn,
            goal="observe the root completion",
            workers=[SwarmWorkerSpec(profile="worker-a", title="A", body="A")],
            verifier_assignee="reviewer",
            synthesizer_assignee="writer",
        )
        root_id = created.root_id
    finally:
        conn.close()

    completed = [
        e for e in events
        if e.get("kind") == "completed" and e.get("task_id") == root_id
    ]
    assert len(completed) == 1, "the swarm root must emit exactly one completion"
    assert completed[0]["status_to"] == "done"


# --------------------------------------------------------------------------
# C1N-T42 — shell-hook interaction
# --------------------------------------------------------------------------


def test_c1n_t42_generic_hook_is_not_shell_excluded():
    """``kanban_task_event`` stays available to shell hooks, deliberately.

    ``SHELL_UNSUPPORTED_HOOKS`` is a RETURN-VALUE-CHANNEL exclusion, not a
    cost control: ``agent/shell_hooks._parse_hooks_block`` refuses those
    events because ``_parse_response`` has no channel for the directive they
    return, so registering would silently drop the hook's output. This hook
    is an observer whose return value is ignored by contract, so there is no
    directive to drop and nothing to refuse.

    Excluding it here to save a fork would also be inconsistent: the RFC
    #58548 observers fire just as often (every tick, every task write) and
    are not excluded either. A shell subscriber costs a fork per event only
    because a user explicitly registered one, and with none registered the
    zero-subscriber fast path means no work at all.
    """
    from hermes_cli.plugins import SHELL_UNSUPPORTED_HOOKS

    assert GENERIC_HOOK not in SHELL_UNSUPPORTED_HOOKS
    for peer in (
        "on_kanban_task_updated",
        "on_kanban_dispatch_tick",
        "kanban_task_completed",
    ):
        assert peer not in SHELL_UNSUPPORTED_HOOKS, (
            "peer kanban observers are not shell-excluded either — if that "
            "changes, revisit this hook's exclusion too"
        )


def test_c1n_t42_a_shell_style_subscriber_defeats_the_fast_path(
    kanban_home, hooks, monkeypatch
):
    """The fast path must not silently drop shell-hook subscribers.

    ``agent.shell_hooks.register_from_config`` wires its callbacks straight
    into ``manager._hooks[event]`` — the same dict ``has_hook`` reads. A
    subscriber registered that way must therefore make the event consumed and
    receive delivery, exactly like a Python plugin's.
    """
    delivered: list[dict] = []
    hooks(GENERIC_HOOK, lambda **kw: delivered.append(kw))

    resolver_calls = _count_resolver(monkeypatch)
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="shell-observed")
        with kb.write_txn(conn):
            kb._append_event(conn, tid, "status", {"status": "ready"})
    finally:
        conn.close()

    assert resolver_calls, "a registered shell hook must defeat the fast path"
    assert [e for e in delivered if e.get("kind") == "status"]
