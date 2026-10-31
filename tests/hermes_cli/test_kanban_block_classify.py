"""block_task() classifies an untyped breaker block in place (#117363)."""
import os
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _time_out_once(conn, tid: str) -> None:
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, os.getpid())
    started = int(time.time()) - 30
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET started_at = ? WHERE id = ?", (started, tid))
        conn.execute(
            "UPDATE task_runs SET started_at = ? "
            "WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
            (started, tid),
        )
    assert tid in kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: None)


def test_typed_block_classifies_untyped_breaker_block(kanban_home, monkeypatch):
    """Two timeouts trip the breaker untyped; the supervisor's typed block then
    attaches the kind without a status flap, a synthetic run, or lost evidence."""
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="ceiling", assignee="worker", max_runtime_seconds=1)
        _time_out_once(conn, tid)
        assert kb.get_task(conn, tid).status == "ready"
        _time_out_once(conn, tid)
        parked = kb.get_task(conn, tid)
        assert (parked.status, parked.block_kind, parked.block_recurrences) == ("blocked", None, 0)

        assert kb.block_task(conn, tid, reason="needs a human decision", kind="needs_input") is True

        after = kb.get_task(conn, tid)
        assert (after.status, after.block_kind, after.block_recurrences) == ("blocked", "needs_input", 1)
        assert after.consecutive_failures == parked.consecutive_failures
        assert after.last_failure_error == parked.last_failure_error
        assert after.current_run_id is None
        kinds = [e.kind for e in kb.list_events(conn, tid)]
        assert (kinds.count("blocked"), kinds.count("gave_up"), kinds.count("timed_out")) == (1, 1, 2)
        assert len(kb.list_runs(conn, tid)) == 2


def test_typed_block_still_refuses_typed_or_live_blocked_cards(kanban_home, monkeypatch):
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    with kbc.connect() as conn:
        parked = kb.create_task(conn, title="parked", assignee="worker", max_runtime_seconds=1)
        _time_out_once(conn, parked)
        _time_out_once(conn, parked)
        assert kb.get_task(conn, parked).status == "blocked"
        stale_run_id = conn.execute(
            "SELECT MAX(id) FROM task_runs WHERE task_id = ?", (parked,)).fetchone()[0]
        assert stale_run_id is not None
        # A worker asserting ownership of its (ended) run cannot classify a parked card.
        assert kb.block_task(conn, parked, reason="mine", kind="needs_input",
                             expected_run_id=stale_run_id) is False
        assert kb.get_task(conn, parked).block_kind is None

        typed = kb.create_task(conn, title="typed once", assignee="worker")
        kb.claim_task(conn, typed)
        assert kb.block_task(conn, typed, reason="decision", kind="needs_input") is True
        assert kb.block_task(conn, typed, reason="again", kind="capability") is False
        assert kb.get_task(conn, typed).block_kind == "needs_input"

        live = kb.create_task(conn, title="live run", assignee="worker")
        kb.claim_task(conn, live)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'blocked' WHERE id = ?", (live,))
        assert kb.block_task(conn, live, reason="race", kind="needs_input") is False
        assert kb.block_task(conn, live, reason="no kind") is False
        assert kb.get_task(conn, live).block_kind is None


def test_dependency_classify_parks_in_todo_and_promotes(kanban_home, monkeypatch):
    """Classifying a parked card as ``dependency`` while a parent is open parks
    it in ``todo`` (``dependency_wait``) so the parent's completion releases it —
    not a sticky ``blocked`` card no tick can promote (#129486)."""
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    with kbc.connect() as conn:
        parent = kb.create_task(conn, title="open-parent", assignee="worker")
        tid = kb.create_task(conn, title="waiter", assignee="worker", max_runtime_seconds=1)
        _time_out_once(conn, tid)
        _time_out_once(conn, tid)
        assert kb.get_task(conn, tid).status == "blocked"
        kb.link_tasks(conn, parent_id=parent, child_id=tid)

        assert kb.block_task(conn, tid, reason="waiting on parent", kind="dependency") is True

        after = kb.get_task(conn, tid)
        assert (after.status, after.block_kind, after.block_recurrences) == ("todo", "dependency", 0)
        wait = [e for e in kb.list_events(conn, tid) if e.kind == "dependency_wait"][-1]
        assert wait.payload.get("classified_in_place") is True
        # Gated in todo while the parent is open, released when it lands — no manual unblock.
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, tid).status == "todo"
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (parent,))
        kb.claim_task(conn, parent, claimer="worker")
        kb.complete_task(conn, parent, result="done")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, tid).status == "ready"


def test_dependency_classify_without_open_parent_rekinds_needs_input(kanban_home, monkeypatch):
    """Classifying a parked card as ``dependency`` with no open parent re-kinds
    to sticky ``needs_input`` exactly like the running path, keeping the rekind
    provenance instead of an unsatisfiable ``dependency`` kind (#129486)."""
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="solo", assignee="worker", max_runtime_seconds=1)
        _time_out_once(conn, tid)
        _time_out_once(conn, tid)
        assert kb.get_task(conn, tid).status == "blocked"

        assert kb.block_task(conn, tid, reason="waiting", kind="dependency") is True

        after = kb.get_task(conn, tid)
        assert (after.status, after.block_kind, after.block_recurrences) == ("blocked", "needs_input", 1)
        blocked = [e for e in kb.list_events(conn, tid) if e.kind == "blocked"][-1].payload
        assert (blocked["requested_kind"], blocked["rekind_reason"]) == ("dependency", "no_open_parent")
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, tid).status == "blocked"
