"""Iteration-budget exhaustion and dispatcher failure-limit tests (#79399).

Moved out of ``tests/hermes_cli/test_kanban_db.py`` to keep that file under
its FILE_LINES ratchet (moved tests keep their coverage)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_recompute_ready_honours_dispatcher_failure_limit(kanban_home):
    """The guard's effective limit must follow the same resolution order
    as the circuit breaker (#35072): per-task max_retries → dispatcher
    failure_limit → DEFAULT_FAILURE_LIMIT.

    Without threading the dispatcher's ``kanban.failure_limit`` through,
    the guard falls back to DEFAULT_FAILURE_LIMIT and disagrees with the
    breaker — sticking a task prematurely (config limit > default) or
    letting a tripped task escape (config limit < default).
    """
    with kbc.connect() as conn:
        # Config allows MORE retries than the default. A task blocked
        # with failures below the configured limit must still recover.
        t = kb.create_task(conn, title="lenient", assignee="a")
        conn.execute(
            "UPDATE tasks SET status='blocked', consecutive_failures=? "
            "WHERE id=?",
            (kb.DEFAULT_FAILURE_LIMIT, t),
        )
        conn.commit()
        # Default-limit call would stick it (failures >= default).
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, t).status == "blocked"
        # Dispatcher configured a higher limit → recover, preserve counter.
        promoted = kb.recompute_ready(
            conn, failure_limit=kb.DEFAULT_FAILURE_LIMIT + 2
        )
        assert promoted == 1
        task = kb.get_task(conn, t)
        assert task.status == "ready"
        assert task.consecutive_failures == kb.DEFAULT_FAILURE_LIMIT

        # Config allows FEWER retries than the default. A task at the
        # stricter limit must stay blocked even though it's below default.
        t2 = kb.create_task(conn, title="strict", assignee="a")
        conn.execute(
            "UPDATE tasks SET status='blocked', consecutive_failures=1 "
            "WHERE id=?",
            (t2,),
        )
        conn.commit()
        # Default-limit (2) would recover it (1 < 2).
        # Stricter config limit (1) must keep it blocked (1 >= 1).
        assert kb.recompute_ready(conn, failure_limit=1) == 0
        assert kb.get_task(conn, t2).status == "blocked"


# ---------------------------------------------------------------------------
# _record_task_failure metadata + late-finalizer guard (#79399)
# ---------------------------------------------------------------------------


def test_record_task_failure_below_threshold_preserves_budget_metadata(kanban_home):
    """Iteration-budget exhaustion keeps budget_used/budget_max in the run
    metadata AND the event payload on the below-threshold path (#79399)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="budget task")
        claimed = kb.claim_task(conn, tid, claimer="test:1")
        assert claimed is not None
        assert kb.get_task(conn, tid).status == "running"

        # Below the breaker threshold (failure_limit=5, this is failure 1).
        blocked = kbd._record_task_failure(
            conn, tid,
            error="Iteration budget exhausted (20/20)",
            outcome="timed_out",
            failure_limit=5,
            release_claim=True,
            end_run=True,
            event_payload_extra={"budget_used": 20, "budget_max": 20},
        )
        assert blocked is False
        # Task dropped back to ready for respawn.
        assert kb.get_task(conn, tid).status == "ready"

        run = conn.execute(
            "SELECT metadata FROM task_runs WHERE task_id = ? "
            "ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()
        assert run is not None
        meta = json.loads(run["metadata"] or "{}")
        assert meta.get("budget_used") == 20
        assert meta.get("budget_max") == 20

        ev = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? "
            "AND kind = 'timed_out' ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()
        assert ev is not None
        payload = json.loads(ev["payload"] or "{}")
        assert payload.get("budget_used") == 20
        assert payload.get("budget_max") == 20


def test_record_task_failure_noop_after_blocked_handoff(kanban_home):
    """A late iteration finalizer must not fail a task already blocked (#79399)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="blocked handoff")
        claimed = kb.claim_task(conn, tid, claimer="test:1")
        assert claimed is not None

        # Worker spends its final iteration on kanban_block -> task blocked.
        assert kb.block_task(conn, tid, reason="needs human input") is True
        assert kb.get_task(conn, tid).status == "blocked"

        # Late finalizer fires after the handoff: must be a no-op.
        result = kbd._record_task_failure(
            conn, tid,
            error="Iteration budget exhausted (20/20)",
            outcome="timed_out",
            failure_limit=5,
            release_claim=True,
            end_run=True,
            event_payload_extra={"budget_used": 20, "budget_max": 20},
        )
        assert result is False
        assert kb.get_task(conn, tid).status == "blocked"
        # No timed_out event appended on top of the handoff.
        kinds = [
            r["kind"]
            for r in conn.execute(
                "SELECT kind FROM task_events WHERE task_id = ?", (tid,)
            ).fetchall()
        ]
        assert "timed_out" not in kinds


def test_record_task_failure_noop_after_completed_handoff(kanban_home):
    """A late iteration finalizer must not fail a task already done (#79399)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="completed handoff")
        claimed = kb.claim_task(conn, tid, claimer="test:1")
        assert claimed is not None

        run_id = kb.get_task(conn, tid).current_run_id
        assert kb.complete_task(
            conn, tid, result="done", expected_run_id=run_id,
        ) is True
        assert kb.get_task(conn, tid).status == "done"

        result = kbd._record_task_failure(
            conn, tid,
            error="Iteration budget exhausted (20/20)",
            outcome="timed_out",
            failure_limit=5,
            release_claim=True,
            end_run=True,
            event_payload_extra={"budget_used": 20, "budget_max": 20},
        )
        assert result is False
        assert kb.get_task(conn, tid).status == "done"
        # No timed_out event appended on top of the handoff.
        kinds = [
            r["kind"]
            for r in conn.execute(
                "SELECT kind FROM task_events WHERE task_id = ?", (tid,)
            ).fetchall()
        ]
        assert "timed_out" not in kinds
