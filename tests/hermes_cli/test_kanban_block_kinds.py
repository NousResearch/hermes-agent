"""Tests for typed block reasons + the unblock-loop breaker.

Covers the built-in fix for the kanban "blocked loop" — a worker blocks a
task, a cron unblocks it, the worker re-blocks for the same reason, repeat
forever. The fix gives ``block_task`` a typed ``kind`` and a persistent
``block_recurrences`` counter:

* ``dependency`` blocks route to ``todo`` (parent-gated, auto-resumed) and
  never enter the human ``blocked`` bucket a cron would keep unblocking.
* ``needs_input`` / ``capability`` / un-typed blocks land in ``blocked``;
  each same-cause re-block after an unblock increments ``block_recurrences``,
  and at ``BLOCK_RECURRENCE_LIMIT`` the task routes to ``triage`` for a human.
* ``unblock_task`` deliberately does NOT reset ``block_recurrences`` (the
  amnesia that let the loop run unbounded).
* A successful ``complete_task`` resets the loop memory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _running_task(conn, title="t"):
    """Create a task and drive it to ``running`` so block_task can act."""
    tid = kb.create_task(conn, title=title, assignee="worker")
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
    with kb.connect_closing() as conn:
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


# ---------------------------------------------------------------------------
# Dependency routing
# ---------------------------------------------------------------------------


def test_dependency_then_parent_done_promotes(kanban_home: Path) -> None:
    """A dependency-parked child becomes ready once its parent completes."""
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        kb.block_task(conn, child, reason="wait", kind="dependency")
        assert kb.get_task(conn, child).status == "todo"
        # Finish the parent, then let recompute_ready run.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        kb.claim_task(conn, parent, claimer="worker")
        kb.complete_task(conn, parent, result="done")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"


# ---------------------------------------------------------------------------
# Respawn guard: a dependency block without a live parent edge is refused
# ---------------------------------------------------------------------------


def test_dependency_block_without_edges_refused(kanban_home: Path) -> None:
    """No parent edges at all: block_task(kind='dependency') must fail closed.

    A parentless todo is re-promoted to ready by recompute_ready on the next
    tick, so parking one in todo respawns the worker immediately.
    """
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        with pytest.raises(ValueError) as excinfo:
            kb.block_task(conn, tid, reason="wait", kind="dependency")
        # The refusal must carry the actionable dependency_edge_missing code.
        assert "dependency_edge_missing" in str(excinfo.value)
        # Nothing was written: the task is still running, no todo park, no
        # dependency_wait event.
        task = kb.get_task(conn, tid)
        assert task.status == "running"
        assert task.block_kind != "dependency"
        assert not [
            e for e in kb.list_events(conn, tid) if e.kind == "dependency_wait"
        ]
        # And no phantom re-promotion can happen: recompute_ready must not
        # move it either.
        kb.recompute_ready(conn)
        assert kb.get_task(conn, tid).status == "running"


def test_dependency_block_all_parents_terminal_refused(
    kanban_home: Path,
) -> None:
    """Every parent already done/archived: fail closed instead of respawning."""
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        kb.claim_task(conn, parent, claimer="worker")
        kb.complete_task(conn, parent, result="done")
        with pytest.raises(ValueError):
            kb.block_task(conn, child, reason="wait", kind="dependency")
        assert kb.get_task(conn, child).status == "running"


def test_dependency_block_live_parent_still_parks(kanban_home: Path) -> None:
    """A live (non-terminal) parent edge keeps the documented todo parking."""
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        assert kb.block_task(conn, child, reason="wait", kind="dependency")
        assert kb.get_task(conn, child).status == "todo"
        # The parked child does NOT respawn while its parent is unfinished:
        # recompute_ready must keep it in todo.
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "todo"


def test_dependency_block_links_new_parents_transactionally(
    kanban_home: Path,
) -> None:
    """dependency_ids add edges inside the block's own write txn."""
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        assert kb.block_task(
            conn, child, reason="wait", kind="dependency",
            dependency_ids=[parent],
        )
        assert kb.get_task(conn, child).status == "todo"
        assert kb.parent_ids(conn, child) == [parent]
        events = [e for e in kb.list_events(conn, child) if e.kind == "linked"]
        assert events, "expected a linked event for the new edge"
        # Finish the parent: the parked child auto-resumes (the happy path).
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        kb.claim_task(conn, parent, claimer="worker")
        kb.complete_task(conn, parent, result="done")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"


def test_dependency_block_new_parent_terminal_still_refused(
    kanban_home: Path,
) -> None:
    """Linking an already-done parent yields no live edge -> fail closed."""
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        kb.claim_task(conn, parent, claimer="worker")
        kb.complete_task(conn, parent, result="done")
        child = _running_task(conn, title="child")
        with pytest.raises(ValueError):
            kb.block_task(
                conn, child, reason="wait", kind="dependency",
                dependency_ids=[parent],
            )
        assert kb.get_task(conn, child).status == "running"


def test_dependency_block_rejects_invalid_dependency_ids(
    kanban_home: Path,
) -> None:
    """Self, unknown, and cyclic dependency_ids fail with no partial write."""
    with kb.connect_closing() as conn:
        child = _running_task(conn, title="child")
        # Self-dependency.
        with pytest.raises(ValueError):
            kb.block_task(
                conn, child, reason="wait", kind="dependency",
                dependency_ids=[child],
            )
        # Unknown task id.
        with pytest.raises(ValueError):
            kb.block_task(
                conn, child, reason="wait", kind="dependency",
                dependency_ids=["t_nonexistent"],
            )
        # Cycle: child's child pointing back at child.
        grandchild = kb.create_task(conn, title="grandchild", assignee="worker")
        kb.link_tasks(conn, parent_id=child, child_id=grandchild)
        with pytest.raises(ValueError):
            kb.block_task(
                conn, child, reason="wait", kind="dependency",
                dependency_ids=[grandchild],
            )
        # No partial state: still running, no edges, no todo park.
        assert kb.get_task(conn, child).status == "running"
        assert kb.parent_ids(conn, child) == []


def test_dependency_ids_require_dependency_kind(kanban_home: Path) -> None:
    """dependency_ids only make sense with kind='dependency'."""
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        with pytest.raises(ValueError):
            kb.block_task(
                conn, child, reason="wait", kind="needs_input",
                dependency_ids=[parent],
            )
        assert kb.get_task(conn, child).status == "running"
        assert kb.parent_ids(conn, child) == []


def test_dependency_block_stale_expected_run_writes_nothing(
    kanban_home: Path,
) -> None:
    """A stale expected_run_id must refuse the whole call atomically.

    Regression: edge links used to run before the run-id check, so a stale
    run could commit parents while the park UPDATE matched nothing — leaving
    a half-blocked task. The run id is now validated before any write.
    """
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        run = kb.latest_run(conn, child)
        assert run is not None
        stale_run_id = run.id + 999
        assert not kb.block_task(
            conn, child, reason="wait", kind="dependency",
            expected_run_id=stale_run_id,
            dependency_ids=[parent],
        )
        # Zero side effects: state, edges, and events all unchanged.
        assert kb.get_task(conn, child).status == "running"
        assert kb.parent_ids(conn, child) == []
        assert not [
            e for e in kb.list_events(conn, child)
            if e.kind in ("linked", "dependency_wait", "blocked")
        ]


def test_dependency_promotion_idempotent_across_recomputes(
    kanban_home: Path,
) -> None:
    """Repeated parent completion / recompute calls must not double-promote.

    Once the parent finishes, the child promotes exactly once; extra
    recompute_ready ticks and a second complete_task call are no-ops.
    """
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        kb.block_task(conn, child, reason="wait", kind="dependency")
        assert kb.get_task(conn, child).status == "todo"
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        kb.claim_task(conn, parent, claimer="worker")
        kb.complete_task(conn, parent, result="done")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"
        # Repeat the promotion pass: no duplicate promotion, no status churn.
        promoted_again = kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"
        assert promoted_again == 0, (
            "child already promoted; a second recompute must be a no-op"
        )
        # Re-completing the finished parent must not re-park or re-promote.
        assert not kb.complete_task(conn, parent, result="done again")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"


def test_cli_dependency_block_refusal_leaves_no_blocked_comment(
    kanban_home: Path,
) -> None:
    """CLI: a refused dependency block must not stamp a BLOCKED comment.

    Regression: ``hermes kanban block`` wrote the comment before validating
    the transition, so a respawn-guard refusal left a misleading
    ``BLOCKED:`` note on a task that was never blocked.
    """
    import argparse

    from hermes_cli import kanban as kc

    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        args = argparse.Namespace(
            task_id=tid,
            reason=["waiting on nothing"],
            kind="dependency",
            ids=None,
            dependency_ids=None,
        )
        rc = kc._cmd_block(args)
        assert rc == 1, "refused dependency block must exit non-zero"
        assert kb.get_task(conn, tid).status == "running"
        comments = kb.list_comments(conn, tid)
        assert not [c for c in comments if "BLOCKED" in (c.body or "")], (
            "refused block must not leave a misleading BLOCKED comment"
        )
        # A successful block still records the comment.
        parent = kb.create_task(conn, title="parent", assignee="worker")
        args_ok = argparse.Namespace(
            task_id=tid,
            reason=["waiting on parent"],
            kind="dependency",
            ids=None,
            dependency_ids=[parent],
        )
        assert kc._cmd_block(args_ok) == 0
        assert kb.get_task(conn, tid).status == "todo"
        comments = kb.list_comments(conn, tid)
        assert [c for c in comments if "BLOCKED" in (c.body or "")], (
            "successful block must record its reason"
        )


# ---------------------------------------------------------------------------
# Completion resets loop memory
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Validation + back-compat
# ---------------------------------------------------------------------------


