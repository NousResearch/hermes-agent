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
# Dependency-wait respawn cooldown (check_respawn_guard)
# ---------------------------------------------------------------------------


def test_dependency_wait_respawn_guard_defers_within_cooldown_across_ticks(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dependency-wait card with an UNCHANGED (still-blocked) parent must
    not respawn more than once per cooldown window across N simulated
    dispatcher ticks (brief A1/A4a)."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setenv("HERMES_KANBAN_DEPENDENCY_WAIT_COOLDOWN_SECONDS", "900")
    now = 10_000_000

    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='blocked' WHERE id=?", (parent,))
        child = _running_task(conn, title="child")
        kb.link_tasks(conn, parent_id=parent, child_id=child)

        monkeypatch.setattr(_kb.time, "time", lambda: now)
        kb.block_task(conn, child, reason="wait", kind="dependency")
        assert kb.get_task(conn, child).status == "todo"
        assert kb.get_task(conn, child).block_recurrences == 1

        # Simulate N dispatcher ticks, all inside the cooldown window, with
        # the parent's status never changing. The guard must defer every
        # single tick — never allowing a respawn.
        for tick in range(5):
            monkeypatch.setattr(_kb.time, "time", lambda t=tick: now + 60 * t)
            assert kb.check_respawn_guard(conn, child) == "dependency_wait_cooldown"


def test_dependency_wait_respawn_guard_bypasses_on_parent_completion(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SAME card must respawn IMMEDIATELY (next tick, guard returns
    None) once the parent transitions to 'done' (brief A3 — safety
    critical, must never wait out the cooldown)."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setenv("HERMES_KANBAN_DEPENDENCY_WAIT_COOLDOWN_SECONDS", "900")
    now = 10_000_000

    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='blocked' WHERE id=?", (parent,))
        child = _running_task(conn, title="child")
        kb.link_tasks(conn, parent_id=parent, child_id=child)

        monkeypatch.setattr(_kb.time, "time", lambda: now)
        kb.block_task(conn, child, reason="wait", kind="dependency")

        # Still well inside the cooldown window — guard defers.
        monkeypatch.setattr(_kb.time, "time", lambda: now + 30)
        assert kb.check_respawn_guard(conn, child) == "dependency_wait_cooldown"

        # Parent transitions blocked -> done, then recompute_ready promotes
        # the child (emits a 'promoted' event AFTER the dependency_wait).
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        kb.claim_task(conn, parent, claimer="worker")
        monkeypatch.setattr(_kb.time, "time", lambda: now + 31)
        kb.complete_task(conn, parent, result="done")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"

        # Guard must return None on the VERY NEXT tick — not wait out the
        # 900s cooldown — because a genuine promotion event now postdates
        # the dependency_wait event.
        monkeypatch.setattr(_kb.time, "time", lambda: now + 32)
        assert kb.check_respawn_guard(conn, child) is None


def test_dependency_wait_recurrence_escalates_past_limit(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-blocking the SAME dependency wait (same parent, unchanged state)
    across BLOCK_RECURRENCE_LIMIT+ cycles must increment block_recurrences
    (fixing block_task's dependency branch) and, once the limit is reached,
    the guard escalates to a hard stop instead of a timed cooldown
    (brief A2/A4c)."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setenv("HERMES_KANBAN_DEPENDENCY_WAIT_COOLDOWN_SECONDS", "900")
    now = 10_000_000

    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='blocked' WHERE id=?", (parent,))
        child = _running_task(conn, title="child")
        kb.link_tasks(conn, parent_id=parent, child_id=child)

        # First dependency block: recurrences == 1.
        monkeypatch.setattr(_kb.time, "time", lambda: now)
        kb.block_task(conn, child, reason="wait", kind="dependency")
        assert kb.get_task(conn, child).block_recurrences == 1

        # Repeatedly: re-verify the SAME still-unsatisfied parent and
        # re-block with the same 'dependency' kind. Each cycle increments
        # the counter. (We force status='running' directly via SQL rather
        # than through claim_task, since claim_task's parent-gating
        # invariant is a separate, out-of-scope concern here — this test
        # only exercises block_task's recurrence counting + the guard.)
        for expected in range(2, kb.BLOCK_RECURRENCE_LIMIT + 2):
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET status='running' WHERE id=?", (child,),
                )
            monkeypatch.setattr(_kb.time, "time", lambda t=expected: now + t)
            kb.block_task(conn, child, reason="wait", kind="dependency")
            assert kb.get_task(conn, child).block_recurrences == expected

        # Now past BLOCK_RECURRENCE_LIMIT — the guard must hard-stop
        # (escalated), not merely apply the timed cooldown, even far outside
        # the cooldown window.
        monkeypatch.setattr(
            _kb.time, "time", lambda: now + 100_000,
        )
        assert kb.check_respawn_guard(conn, child) == "dependency_wait_escalated"

        # A genuine promotion event (parent done) still bypasses the
        # escalated hard-stop.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        kb.claim_task(conn, parent, claimer="worker")
        kb.complete_task(conn, parent, result="done")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"
        assert kb.check_respawn_guard(conn, child) is None


def test_dependency_wait_cooldown_disabled_via_zero(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cooldown of 0 disables the guard (test-friendly escape hatch),
    mirroring the rate-limit cooldown's documented 0-disables behaviour."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setenv("HERMES_KANBAN_DEPENDENCY_WAIT_COOLDOWN_SECONDS", "0")
    now = 10_000_000

    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='blocked' WHERE id=?", (parent,))
        child = _running_task(conn, title="child")
        kb.link_tasks(conn, parent_id=parent, child_id=child)

        monkeypatch.setattr(_kb.time, "time", lambda: now)
        kb.block_task(conn, child, reason="wait", kind="dependency")

        monkeypatch.setattr(_kb.time, "time", lambda: now + 1)
        assert kb.check_respawn_guard(conn, child) is None


# ---------------------------------------------------------------------------
# Completion resets loop memory
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Validation + back-compat
# ---------------------------------------------------------------------------
