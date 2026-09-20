"""Tests for typed block reasons + the unblock-loop breaker.

Covers the built-in fix for the kanban "blocked loop" — a worker blocks a
task, a cron unblocks it, the worker re-blocks for the same reason, repeat
forever. The fix gives ``block_task`` a typed ``kind`` and a persistent
``block_recurrences`` counter:

* ``dependency`` blocks route to ``todo`` (parent-gated, auto-resumed) and
  never enter the human ``blocked`` bucket a cron would keep unblocking —
  unless no parent is open, in which case the wait can never be satisfied
  and the block is recorded as ``needs_input`` (sticky, loop-counted).
* ``needs_input`` / ``capability`` / un-typed blocks land in ``blocked``;
  each same-cause re-block after an unblock increments ``block_recurrences``,
  and at ``BLOCK_RECURRENCE_LIMIT`` the task routes to ``triage`` for a human.
* ``unblock_task`` deliberately does NOT reset ``block_recurrences`` (the
  amnesia that let the loop run unbounded).
* A successful ``complete_task`` resets the loop memory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from hermes_cli import kanban as kanban_cli
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


def _running_task(conn, title="t", parents=()):
    """Create a task (linked under ``parents`` first) and drive it to ``running`` so block_task can act."""
    tid = kb.create_task(conn, title=title, assignee="worker")
    for parent in parents:
        kb.link_tasks(conn, parent_id=parent, child_id=tid)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    claimed = kb.claim_task(conn, tid, claimer="worker")
    assert claimed is not None
    return tid


def _make_running_again(conn, tid):
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None


# ---------------------------------------------------------------------------
# Loop breaker
# ---------------------------------------------------------------------------










def test_block_loop_detected_event_emitted(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="x", kind="capability")
        kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        kb.block_task(conn, tid, reason="x", kind="capability")
        events = [e for e in kb.list_events(conn, tid)
                  if e.kind == "block_loop_detected"]
        assert events, "expected a block_loop_detected event"
        payload = events[-1].payload or {}
        assert payload.get("recurrences") == 2
        assert payload.get("kind") == "capability"


def test_machine_pr_feedback_block_returns_to_dispatcher(kanban_home: Path) -> None:
    """Recoverable PR/CI worker failures never become human-sticky blocks."""

    import hermes_cli.kanban_db_connect as _hermes_cli_kanban_db_connect
    with _hermes_cli_kanban_db_connect.connect_closing() as conn:
        tid = kb.create_task(
            conn,
            title="GitHub PR feedback: acme/widgets#17",
            assignee="pr-repair-steward",
            idempotency_key="github-pr-feedback:abc123",
            initial_status="running",
        )
        assert kb.claim_task(conn, tid, claimer="pr-repair-steward") is not None
        kb.block_task(conn, tid, reason="start validation", kind="needs_input")
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "ready"
        events = [event.kind for event in kb.list_events(conn, tid)]
        assert "machine_handoff" in events


def test_db_accepts_synthetic_assignee_for_internal_worker(kanban_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (kanban_home / "profiles").mkdir()
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="Review worker", assignee="worker")
        assert task_id.startswith("t_")


def test_intent_review_needs_input_triage_stays_human_gated(kanban_home: Path) -> None:
    import hermes_cli.kanban_db_connect as _hermes_cli_kanban_db_connect
    with _hermes_cli_kanban_db_connect.connect_closing() as conn:
        tid = kb.create_task(
            conn,
            title="Operator intent required: PR #17",
            body="Record only the operator intent decision.",
            assignee="task-orchestrator",
            idempotency_key="github-pr-feedback:intent-review:repo:17:abc",
            initial_status="running",
        )
        assert kb.claim_task(conn, tid, claimer="task-orchestrator") is not None
        kb.block_task(conn, tid, reason="operator intent", kind="needs_input")
        kb.unblock_task(conn, tid)
        assert kb.claim_task(conn, tid, claimer="task-orchestrator") is not None
        kb.block_task(conn, tid, reason="operator intent", kind="needs_input")
        assert kb.get_task(conn, tid).status == "triage"

        promoted = kb.recompute_ready(conn)

        assert promoted == 0
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "triage"


def test_live_home_preserves_external_assignee_lane(kanban_home: Path) -> None:
    (kanban_home / "profiles").mkdir()
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="Review worker", assignee="worker")
        assert kb.get_task(conn, task_id).assignee == "worker"


# ---------------------------------------------------------------------------
# Dependency routing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("finished_parent", [False, True])
def test_dependency_without_pending_parent_stays_blocked(kanban_home, finished_parent):
    with kbc.connect_closing() as conn:
        if finished_parent:
            parent = _running_task(conn, title="finished prerequisite")
            kb.complete_task(conn, parent, result="done")
            child = kb.create_task(conn, title="missing external prerequisite", assignee="worker")
            kb.link_tasks(conn, parent_id=parent, child_id=child)
            assert kb.claim_task(conn, child, claimer="worker") is not None
        else:
            child = _running_task(conn, title="missing external prerequisite")
        assert kb.block_task(conn, child, reason="source unavailable", kind="dependency")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "blocked"
        assert kb.claim_task(conn, child, claimer="worker") is None


def test_dependency_then_parent_done_promotes(kanban_home: Path) -> None:
    """A dependency-parked child becomes ready once its parent completes."""
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(
            conn,
            parent_id=parent,
            child_id=child,
            expected_child_run_id=kb.get_task(conn, child).current_run_id,
        )
        kb.block_task(conn, child, reason="wait", kind="dependency")
        assert kb.get_task(conn, child).status == "todo"
        # Finish the parent, then let recompute_ready run.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        kb.claim_task(conn, parent, claimer="worker")
        kb.complete_task(conn, parent, result="done")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"


def test_dependency_block_with_terminal_parents_parks_then_escalates(
    kanban_home: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A ``dependency`` block whose parents are all terminal can never be
    satisfied by ``recompute_ready``: it must park in ``blocked`` as
    ``needs_input`` (no ``dependency_wait``, no re-promotion), say so on the
    CLI, and count toward the loop breaker so a re-block after an unblock
    reaches ``triage``."""
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="already-done-parent", assignee="worker")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='done' WHERE id=?", (parent,))
        # The edge exists before the run: link_tasks refuses to gate a running child retroactively.
        child = _running_task(conn, title="child-of-done", parents=(parent,))

        # `hermes kanban block <child> --kind dependency waiting on upstream`
        args = argparse.Namespace(task_id=child, ids=None, reason=["waiting", "on", "upstream"], kind="dependency")
        assert kanban_cli._cmd_block(args) == 0
        assert f"Blocked {child} as needs_input (no open parent to wait on): waiting on upstream" in capsys.readouterr().out
        parked = kb.get_task(conn, child)
        assert (parked.status, parked.block_kind, parked.block_recurrences) == ("blocked", "needs_input", 1)
        events = kb.list_events(conn, child)
        assert not [e for e in events if e.kind == "dependency_wait"]
        blocked = [e for e in events if e.kind == "blocked"][-1].payload
        assert (blocked["requested_kind"], blocked["rekind_reason"]) == ("dependency", "no_open_parent")
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, child).status == "blocked"

        # A cron/human unblocks; the worker re-declares the same impossible wait.
        assert kb.unblock_task(conn, child)
        assert kb.claim_task(conn, child, claimer="worker") is not None
        assert kb.block_task(conn, child, reason="still waiting", kind="dependency")
        assert kb.get_task(conn, child).status == "triage"
        loop = [e for e in kb.list_events(conn, child) if e.kind == "block_loop_detected"][-1].payload
        assert loop["recurrences"] == kb.BLOCK_RECURRENCE_LIMIT


def test_dependency_block_with_open_parent_stays_parked_across_dispatch_tick(
    kanban_home: Path, all_assignees_spawnable,
) -> None:
    """Control: a genuine wait on an incomplete parent parks in ``todo``
    without counting a recurrence, survives a dispatch tick unspawned, and
    resumes once the parent finishes."""
    spawns: list[str] = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 4242

    with kbc.connect() as conn:
        parent = kb.create_task(conn, title="open-parent", assignee="alice")
        # Linked while todo (a running child cannot be gated retroactively); the parent then stays
        # open while the child runs — the reopened-parent shape, forced the same way the loop below does.
        child = kb.create_task(conn, title="waiter", assignee="worker")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='running' WHERE id=?", (child,))
        for _ in range(kb.BLOCK_RECURRENCE_LIMIT + 1):
            assert kb.block_task(conn, child, reason="wait", kind="dependency")
            parked = kb.get_task(conn, child)
            assert (parked.status, parked.block_kind, parked.block_recurrences) == ("todo", "dependency", 0)
            with kb.write_txn(conn):
                conn.execute("UPDATE tasks SET status='running' WHERE id=?", (child,))
        assert kb.block_task(conn, child, reason="wait", kind="dependency")
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn)
        assert kb.get_task(conn, child).status == "todo"
        assert child not in spawns
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        kb.claim_task(conn, parent, claimer="alice")
        kb.complete_task(conn, parent, result="done")
        res2 = kbd.dispatch_once(conn, spawn_fn=fake_spawn)
        assert child in [row[0] for row in res2.spawned]


# ---------------------------------------------------------------------------
# Completion resets loop memory
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Validation + back-compat
# ---------------------------------------------------------------------------
