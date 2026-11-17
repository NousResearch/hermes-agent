"""Tests for kanban cancellation (B3): ``cancel_task`` / ``cancel_tasks_for_session``
and the ``hermes kanban cancel`` CLI.

Contract, from the B3 investigation: the ``cancelled`` value already exists in
the schema — this wires it. A live worker is signalled (fingerprint-checked)
before the card is marked; a card already terminal is an idempotent no-op;
cancellation never increments the circuit-breaker counter; a worker whose
result lands after the cancel never re-opens the card; stop-all cancels only
its own session's cards.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
from contextlib import contextmanager
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kbc.connect_closing() as c:
        yield c


@contextmanager
def _live_sleeper():
    """A real live process standing in for a worker (dies on SIGTERM)."""
    proc = subprocess.Popen(["sleep", "30"])
    try:
        yield proc
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def _claim_with_worker(conn, tid, proc):
    """Claim ``tid`` and publish ``proc`` as its worker (real pid + fingerprint)."""
    host = kb._claimer_id().split(":", 1)[0]
    assert kb.claim_task(conn, tid, claimer=f"{host}:A")
    kbd._set_worker_pid(conn, tid, proc.pid)


def _events(conn, tid):
    rows = conn.execute(
        "SELECT kind, payload FROM task_events WHERE task_id = ? ORDER BY id", (tid,)
    ).fetchall()
    return [(row["kind"], json.loads(row["payload"]) if row["payload"] else None) for row in rows]


def _recording_kill(calls):
    def spy(pid, sig):
        calls.append((pid, sig))
        os.kill(pid, sig)

    return spy


# ---------------------------------------------------------------------------
# Single-card cancellation
# ---------------------------------------------------------------------------


def test_cancel_todo_task_marks_cancelled_with_audit(conn):
    """A parked card cancels cleanly: status, timestamp, actor and event."""
    parent = kb.create_task(conn, title="parent", assignee="w")
    child = kb.create_task(conn, title="child", parents=[parent])
    assert kb.get_task(conn, child).status == "todo"

    outcome = kb.cancel_task(conn, child, reason="no longer needed", cancelled_by="tester")

    assert outcome.cancelled is True
    assert outcome.status == "cancelled"
    assert outcome.cancelled_ids == [child]
    assert outcome.workers_terminated == [] and outcome.workers_survived == []
    card = kb.get_task(conn, child)
    assert card.status == "cancelled"
    assert card.completed_at is not None
    assert card.cancelled_by == "tester"
    assert card.claim_lock is None and card.worker_pid is None
    kinds = [kind for kind, _ in _events(conn, child)]
    assert "cancelled" in kinds
    payload = dict(_events(conn, child))["cancelled"]
    assert payload["reason"] == "no longer needed"
    assert payload["cancelled_by"] == "tester"
    assert payload["prior_status"] == "todo"
    # The parent is untouched: cancelling a child never climbs.
    assert kb.get_task(conn, parent).status == "ready"


def test_cancel_does_not_touch_breaker_counter(conn):
    """Cancellation is not a failure: ``consecutive_failures`` never moves."""
    tid = kb.create_task(conn, title="flaky before", assignee="w")
    conn.execute("UPDATE tasks SET consecutive_failures = 1 WHERE id = ?", (tid,))
    conn.commit()

    kb.cancel_task(conn, tid, cancelled_by="tester")

    assert kb.get_task(conn, tid).consecutive_failures == 1


def test_cancel_done_is_idempotent_noop(conn):
    """A finished card is history: cancelling it changes nothing, errors nothing."""
    tid = kb.create_task(conn, title="done work", assignee="w")
    assert kb.complete_task(conn, tid, result="shipped")
    before = _events(conn, tid)

    outcome = kb.cancel_task(conn, tid, cancelled_by="tester")

    assert outcome.cancelled is False
    assert outcome.status == "done"
    assert outcome.cancelled_ids == []
    assert kb.get_task(conn, tid).status == "done"
    assert _events(conn, tid) == before


def test_cancel_unknown_id_reports_none(conn):
    outcome = kb.cancel_task(conn, "t_nope", cancelled_by="tester")
    assert outcome.cancelled is False
    assert outcome.status is None
    assert outcome.cancelled_ids == []


def test_cancel_twice_is_idempotent(conn):
    tid = kb.create_task(conn, title="twice", assignee="w")
    first = kb.cancel_task(conn, tid, cancelled_by="tester")
    second = kb.cancel_task(conn, tid, cancelled_by="tester")
    assert first.cancelled is True
    assert second.cancelled is False
    assert second.status == "cancelled"
    assert second.cancelled_ids == []
    assert [kind for kind, _ in _events(conn, tid)].count("cancelled") == 1


# ---------------------------------------------------------------------------
# Live worker
# ---------------------------------------------------------------------------


def test_cancel_running_worker_gets_sigterm(conn):
    """A live worker is signalled (fingerprint-checked) and the card ends cancelled."""
    tid = kb.create_task(conn, title="live work", assignee="w")
    calls: list = []
    with _live_sleeper() as sleeper:
        _claim_with_worker(conn, tid, sleeper)
        assert kb.get_task(conn, tid).status == "running"

        outcome = kb.cancel_task(
            conn, tid, reason="stop", cancelled_by="tester",
            signal_fn=_recording_kill(calls),
        )

        assert sleeper.poll() is not None, "worker survived the cancel"
        assert outcome.cancelled is True
        assert outcome.workers_terminated == [tid]
        assert outcome.workers_survived == []
        assert (sleeper.pid, signal.SIGTERM) in calls
        card = kb.get_task(conn, tid)
        assert card.status == "cancelled"
        assert card.worker_pid is None and card.claim_lock is None
        kinds = [kind for kind, _ in _events(conn, tid)]
        assert "cancelled" in kinds and "cancel_worker_termination" in kinds


def test_cancel_running_dead_pid_counts_as_terminated_without_signal(conn):
    """A worker already gone needs no signal: the card still cancels cleanly."""
    tid = kb.create_task(conn, title="dead worker", assignee="w")
    dead = subprocess.Popen(["true"])
    dead.wait()
    _claim_with_worker(conn, tid, dead)
    calls: list = []

    outcome = kb.cancel_task(conn, tid, cancelled_by="tester", signal_fn=_recording_kill(calls))

    assert outcome.cancelled is True
    assert outcome.workers_terminated == [tid]
    assert calls == []


def test_cancel_unverified_fingerprint_never_signalled(conn):
    """A live PID with no verified fingerprint is never signalled — but the
    card still cancels, and the survivor is reported for the reaper."""
    tid = kb.create_task(conn, title="unverified", assignee="w")
    calls: list = []
    with _live_sleeper() as sleeper:
        _claim_with_worker(conn, tid, sleeper)
        conn.execute(
            "UPDATE tasks SET worker_started_at = ? WHERE id = ?",
            (kbd.UNVERIFIED_WORKER_FINGERPRINT, tid),
        )
        conn.commit()

        outcome = kb.cancel_task(
            conn, tid, cancelled_by="tester", signal_fn=_recording_kill(calls),
        )

        assert calls == [], "an unverified PID must never be signalled"
        assert sleeper.poll() is None, "the test's sleeper must still be alive here"
        assert outcome.cancelled is True
        assert outcome.workers_survived == [tid]
        assert outcome.workers_terminated == []
        assert kb.get_task(conn, tid).status == "cancelled"


def test_late_worker_result_never_reopens_cancelled_card(conn):
    """A worker that finishes after the cancel finds completion refused."""
    tid = kb.create_task(conn, title="late finisher", assignee="w")
    with _live_sleeper() as sleeper:
        _claim_with_worker(conn, tid, sleeper)
        kb.cancel_task(conn, tid, cancelled_by="tester")

        assert kb.complete_task(conn, tid, result="finished anyway") is False
        assert kb.get_task(conn, tid).status == "cancelled"
        assert kbd.heartbeat_worker(conn, tid) is False


# ---------------------------------------------------------------------------
# Recursion and stop-all
# ---------------------------------------------------------------------------


def test_cancel_parent_cancels_children_not_siblings(conn):
    """Recursion follows ``task_links`` down only: siblings and done children stay."""
    parent = kb.create_task(conn, title="parent", assignee="w")
    child_a = kb.create_task(conn, title="child a", parents=[parent])
    child_b = kb.create_task(conn, title="child b standalone")
    grandchild = kb.create_task(conn, title="grandchild", parents=[child_a])
    sibling = kb.create_task(conn, title="sibling", assignee="w")
    # A finished child linked under the parent afterwards: terminal, so the
    # cancel must leave it alone (only the demotion rule touches ready cards).
    assert kb.complete_task(conn, child_b, result="already done")
    assert kb.link_tasks(conn, parent, child_b) is False

    outcome = kb.cancel_task(conn, parent, cancelled_by="tester")

    assert outcome.cancelled is True
    assert set(outcome.cancelled_ids) == {parent, child_a, grandchild}
    assert kb.get_task(conn, child_b).status == "done"
    assert kb.get_task(conn, sibling).status == "ready"
    assert kb.get_task(conn, grandchild).cancelled_by == "tester"


def test_stop_all_by_session_only_that_session(conn):
    """Stop-all keys on the origin session: other sessions and session-less
    cards are untouched, and finished cards of the same session stay finished."""
    mine_a = kb.create_task(conn, title="mine a", session_id="sess-1")
    mine_b = kb.create_task(conn, title="mine b", session_id="sess-1")
    mine_done = kb.create_task(conn, title="mine done", session_id="sess-1")
    assert kb.complete_task(conn, mine_done, result="shipped")
    theirs = kb.create_task(conn, title="theirs", session_id="sess-2")
    no_session = kb.create_task(conn, title="no session")

    cancelled = kb.cancel_tasks_for_session(conn, "sess-1", cancelled_by="tester")

    assert set(cancelled) == {mine_a, mine_b}
    assert kb.get_task(conn, mine_done).status == "done"
    assert kb.get_task(conn, theirs).status == "ready"
    assert kb.get_task(conn, no_session).status == "ready"


def test_cancelled_parent_does_not_release_child(conn):
    """Only ``done``/``archived`` release a child: a card linked under a
    cancelled parent stays parked instead of being promoted."""
    parent = kb.create_task(conn, title="parent", assignee="w")
    kb.cancel_task(conn, parent, cancelled_by="tester")
    child = kb.create_task(conn, title="late child", parents=[parent])

    assert kb.get_task(conn, child).status == "todo"
    kb.recompute_ready(conn)
    assert kb.get_task(conn, child).status == "todo"


# ---------------------------------------------------------------------------
# Vocabulary, migration, CLI
# ---------------------------------------------------------------------------


def test_cancelled_is_a_listable_status(conn):
    """``cancelled`` joins the status vocabulary (like ``archived``): the
    observation path ``list_tasks(status="cancelled")`` works, and an unknown
    status still raises."""
    tid = kb.create_task(conn, title="listable", assignee="w")
    kb.cancel_task(conn, tid, cancelled_by="tester")

    assert [t.id for t in kb.list_tasks(conn, status="cancelled")] == [tid]
    with pytest.raises(ValueError):
        kb.list_tasks(conn, status="bogus")


def test_cancelled_by_column_migrates_legacy_db(tmp_path, monkeypatch):
    """A board predating the column reads fine (``cancelled_by`` is NULL) and
    the migration re-adds the column so cancelling works afterwards."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect_closing() as c:
        tid = kb.create_task(c, title="legacy", assignee="w")
        c.execute("ALTER TABLE tasks DROP COLUMN cancelled_by")
        c.commit()
        # A row without the column still reads: the audit field is NULL.
        assert kb.get_task(c, tid).cancelled_by is None
    kb.init_db()  # re-runs the migration pass
    with kbc.connect_closing() as c:
        assert "cancelled_by" in {
            row["name"] for row in c.execute("PRAGMA table_info(tasks)")
        }
        assert kb.cancel_task(c, tid, cancelled_by="tester").cancelled is True
        assert kb.get_task(c, tid).cancelled_by == "tester"


def test_cancelled_is_a_restore_source_not_a_drag_target():
    """Manual moves mirror ``archived``: a human may drag a cancelled card back
    to the board (restore), but cancelling is never a drag — it is a verb with
    its own gates (worker kill, audit), so ``-> cancelled`` is always False."""
    from hermes_cli import kanban_workflow as kw

    assert kw.DEFAULT_WORKFLOW.can_move("cancelled", "todo") is True
    assert kw.DEFAULT_WORKFLOW.can_move("cancelled", "ready") is True
    assert kw.DEFAULT_WORKFLOW.can_move("cancelled", "cancelled") is False
    assert kw.DEFAULT_WORKFLOW.can_move("ready", "cancelled") is False
    assert kw.DEFAULT_WORKFLOW.can_move("cancelled", "archived") is True


def test_cli_cancel_single_task(kanban_home):
    with kbc.connect_closing() as c:
        tid = kb.create_task(c, title="cli one", assignee="w")
    out = kc.run_slash(f"cancel {tid} --by tester")
    assert f"Cancelled {tid}" in out
    with kbc.connect_closing() as c:
        card = kb.get_task(c, tid)
    assert card.status == "cancelled"
    assert card.cancelled_by == "tester"


def test_cli_cancel_unknown_id_fails(kanban_home):
    out = kc.run_slash("cancel t_nope")
    assert "cannot cancel t_nope" in out


def test_cli_cancel_requires_a_target(kanban_home):
    out = kc.run_slash("cancel")
    assert "at least one task_id or --session is required" in out


def test_cli_cancel_session_json(kanban_home):
    with kbc.connect_closing() as c:
        mine = kb.create_task(c, title="mine", session_id="sess-9")
        kb.create_task(c, title="theirs", session_id="sess-8")
    out = kc.run_slash("cancel --session sess-9 --json")
    payload = json.loads(out)
    assert payload["session_id"] == "sess-9"
    assert payload["cancelled"] == [mine]
    with kbc.connect_closing() as c:
        assert kb.get_task(c, mine).status == "cancelled"


def test_cli_cancel_ids_json_reports_failed(kanban_home):
    with kbc.connect_closing() as c:
        tid = kb.create_task(c, title="json one", assignee="w")
    out = kc.run_slash(f"cancel {tid} t_nope --json")
    payload = json.loads(out)
    assert payload["cancelled"] == [tid]
    assert payload["failed"] == ["t_nope"]


def test_cli_cancel_ids_and_session_conflict(kanban_home):
    out = kc.run_slash("cancel t_x --session sess-1")
    assert "choose either task_ids to cancel or --session" in out
