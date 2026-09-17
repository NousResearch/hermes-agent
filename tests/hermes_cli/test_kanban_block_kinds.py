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


# ---------------------------------------------------------------------------
# Dependency routing
# ---------------------------------------------------------------------------


def test_dependency_then_parent_done_promotes(kanban_home: Path) -> None:
    """A dependency-parked child becomes ready once its parent completes."""
    with kbc.connect_closing() as conn:
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


def test_dependency_block_with_no_open_parent_does_not_auto_promote(
    kanban_home: Path,
) -> None:
    """A mislabelled ``kind='dependency'`` with no incomplete parent must
    not land in the auto-resume ``todo`` lane. ``recompute_ready`` would
    otherwise promote it on the next tick and a fresh worker would claim
    it with none of the prior context.
    """
    with kbc.connect_closing() as conn:
        tid = _running_task(conn, title="no-parent-refusal")
        assert kb.block_task(
            conn, tid,
            reason="The draft specification has failed review",
            kind="dependency",
        )
        parked = kb.get_task(conn, tid)
        assert parked is not None
        assert parked.status == "blocked"
        assert parked.block_kind == "needs_input"
        wait = [e for e in kb.list_events(conn, tid) if e.kind == "dependency_wait"]
        assert not wait, (
            "mislabelled dependency must not be recorded as dependency_wait; "
            f"got {[(e.kind, e.payload) for e in wait]}"
        )
        blocked = [e for e in kb.list_events(conn, tid) if e.kind == "blocked"]
        assert blocked, "re-kinded block must surface as a sticky blocked event"
        payload = blocked[-1].payload or {}
        assert payload.get("requested_kind") == "dependency"
        assert payload.get("rekind_reason") == "no_open_parent"
        assert payload.get("kind") == "needs_input"
        promoted = kb.recompute_ready(conn)
        assert promoted == 0
        after = kb.get_task(conn, tid)
        assert after is not None
        assert after.status == "blocked"
        assert after.status != "ready"


def test_dependency_block_with_already_done_parent_does_not_auto_promote(
    kanban_home: Path,
) -> None:
    """Incident shape: created with ``parents=[done_id]`` so there was never
    an open dependency, then blocked ``kind='dependency'``."""
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="already-done-parent", assignee="worker")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        kb.claim_task(conn, parent, claimer="worker")
        kb.complete_task(conn, parent, result="done")
        child = kb.create_task(
            conn, title="child-of-done", assignee="worker", parents=[parent],
        )
        claimed = kb.claim_task(conn, child, claimer="worker")
        assert claimed is not None
        assert kb.block_task(
            conn, child,
            reason="The draft specification has failed review",
            kind="dependency",
        )
        parked = kb.get_task(conn, child)
        assert parked is not None
        assert parked.status == "blocked"
        assert parked.block_kind == "needs_input"
        assert kb.recompute_ready(conn) == 0
        after = kb.get_task(conn, child)
        assert after is not None
        assert after.status == "blocked"


def test_mislabelled_dependency_block_is_not_reclaimed_on_next_dispatch_tick(
    kanban_home: Path, all_assignees_spawnable,
) -> None:
    """Full tick: block kind=dependency with no open parent, then
    ``dispatch_once`` must not promote+spawn a second worker."""
    spawns: list[str] = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 4242

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="refusal", assignee="alice")
        assert kb.claim_task(conn, tid, claimer="alice") is not None
        assert kb.block_task(
            conn, tid,
            reason="needs a human decision, filed as dependency",
            kind="dependency",
        )
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn)
        after = kb.get_task(conn, tid)
        assert after is not None
        spawned_ids = [row[0] for row in res.spawned]
        assert tid not in spawned_ids, (
            f"tick re-claimed the mislabelled card: spawned={res.spawned!r} "
            f"promoted={res.promoted} status={after.status!r}"
        )
        assert tid not in spawns
        assert res.promoted == 0
        assert after.status == "blocked"
        assert after.block_kind == "needs_input"


def test_dependency_block_with_open_parent_stays_parked_across_dispatch_tick(
    kanban_home: Path, all_assignees_spawnable,
) -> None:
    """Legitimate fan-in: an incomplete parent must still park in todo and
    auto-resume only after that parent finishes."""
    spawns: list[str] = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 4242

    with kbc.connect() as conn:
        parent = kb.create_task(conn, title="open-parent", assignee="alice")
        child = kb.create_task(conn, title="waiter", assignee="alice")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (child,))
        assert kb.claim_task(conn, child, claimer="alice") is not None
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        assert kb.block_task(conn, child, reason="wait", kind="dependency")
        parked = kb.get_task(conn, child)
        assert parked is not None
        assert parked.status == "todo"
        assert parked.block_kind == "dependency"
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn)
        still = kb.get_task(conn, child)
        assert still is not None
        assert still.status == "todo"
        assert child not in spawns
        assert child not in [row[0] for row in res.spawned]
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        kb.claim_task(conn, parent, claimer="alice")
        kb.complete_task(conn, parent, result="done")
        res2 = kbd.dispatch_once(conn, spawn_fn=fake_spawn)
        resumed = kb.get_task(conn, child)
        assert resumed is not None
        assert child in [row[0] for row in res2.spawned] or resumed.status in (
            "ready", "running",
        )


# ---------------------------------------------------------------------------
# Completion resets loop memory
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Validation + back-compat
# ---------------------------------------------------------------------------


