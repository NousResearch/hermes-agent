"""Regression tests for blocked kanban tasks staying blocked.

The bug: when a task reached ``status='blocked'``, the dispatcher's
``recompute_ready`` could promote it back to ``ready`` on the next tick
if it had no parents or all parents were done.  The next dispatcher pass
could then claim/spawn it without an explicit human unblock.

These tests pin down:

* Any task with ``status='blocked'`` survives ``recompute_ready``.
* No-parent approval gates and parent-completion paths do not silently
  become runnable again.
* Circuit-breaker blocks (``gave_up`` event, status flipped via
  ``_record_task_failure``) require explicit unblock like every other
  blocked task.
* An explicit ``kanban_unblock`` clears the blocked state.
* The full block → promote → crash → ``gave_up`` loop is broken after
  this fix: subsequent ticks leave the task blocked.

The tangentially related schema-init ordering bug originally reported
in #28712 (``init_db`` crashing on legacy DBs that pre-dated the
``session_id`` migration) is covered separately by
``test_kanban_db.py::test_connect_migrates_legacy_db_before_optional_column_indexes``,
landed via #28754 / #28781 ahead of this fix.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# Worker-initiated kanban_block must be sticky
# ---------------------------------------------------------------------------


def test_worker_block_is_not_auto_promoted_by_recompute_ready(kanban_home: Path) -> None:
    """A standalone task that a worker explicitly blocks for review
    must stay blocked across an arbitrary number of dispatcher ticks.
    Before #28712's fix, ``recompute_ready`` would silently flip it
    back to ``ready`` on the very next tick."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="needs human review")
        kb.claim_task(conn, tid)
        assert kb.block_task(
            conn, tid,
            reason="review-required: please verify ACL change",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        assert kb.get_task(conn, tid).status == "blocked"

        # Hammer the promotion code — exactly the dispatcher loop's
        # behaviour, just compressed in time.
        for _ in range(5):
            promoted = kb.recompute_ready(conn)
            assert promoted == 0, "worker-blocked task must not auto-promote"
            assert kb.get_task(conn, tid).status == "blocked"


def test_no_parent_blocked_task_is_not_auto_promoted_by_recompute_ready(kanban_home: Path) -> None:
    """A blocked approval gate with no parents must not auto-promote.

    The dangerous edge case is ``all([]) == True``: if ``recompute_ready``
    scans blocked rows, a standalone blocked task can become ready and be
    claimed/spawned on the same dispatcher tick.
    """
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="standalone approval gate", initial_status="blocked")
        assert kb.get_task(conn, tid).status == "blocked"

        promoted = kb.recompute_ready(conn)
        assert promoted == 0
        assert kb.get_task(conn, tid).status == "blocked"


def test_dispatch_once_does_not_claim_or_spawn_no_parent_blocked_task(kanban_home: Path) -> None:
    """Dispatcher ticks must not turn a blocked no-parent gate into work."""
    spawned: list[str] = []

    def spawn_fn(task: kb.Task, workspace: str) -> int:
        spawned.append(task.id)
        return 12345

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="blocked gate",
            assignee="default",
            initial_status="blocked",
        )

        result = kb.dispatch_once(conn, spawn_fn=spawn_fn)

        assert result.promoted == 0
        assert result.spawned == []
        assert spawned == []
        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "blocked"
        assert task.current_run_id is None


def test_worker_block_on_child_with_done_parents_is_still_sticky(kanban_home: Path) -> None:
    """The parent-completion path is the one ``recompute_ready`` was
    designed for, so it's the most dangerous false-positive: even when
    every parent is done, a worker-initiated block on the child must
    stay blocked."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child", parents=[parent])
        kb.complete_task(conn, parent, result="parent ok")

        kb.claim_task(conn, child)
        kb.block_task(
            conn, child,
            reason="review-required: child needs sign-off",
            expected_run_id=kb.get_task(conn, child).current_run_id,
        )
        assert kb.get_task(conn, child).status == "blocked"

        promoted = kb.recompute_ready(conn)
        assert promoted == 0
        assert kb.get_task(conn, child).status == "blocked"


# ---------------------------------------------------------------------------
# Circuit-breaker blocks require explicit unblock too
# ---------------------------------------------------------------------------


def test_circuit_breaker_block_does_not_auto_promote(kanban_home: Path) -> None:
    """A child that was put into ``blocked`` without a worker-issued
    ``kanban_block`` (e.g. circuit-breaker after repeated spawn failures,
    manual DB triage) must still require explicit unblock."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child", parents=[parent])
        kb.complete_task(conn, parent, result="ok")

        # Simulate a circuit-breaker / direct triage that flips status
        # without emitting a ``blocked`` event — exactly what
        # ``_record_task_failure`` does after a ``gave_up``.
        conn.execute(
            "UPDATE tasks SET status='blocked', consecutive_failures=5, "
            "last_failure_error='persistent error' WHERE id=?",
            (child,),
        )
        conn.commit()

        promoted = kb.recompute_ready(conn)
        assert promoted == 0
        task = kb.get_task(conn, child)
        assert task is not None
        assert task.status == "blocked"
        assert task.consecutive_failures == 5
        assert task.last_failure_error == "persistent error"


def test_gave_up_event_alone_still_leaves_blocked_task_blocked(kanban_home: Path) -> None:
    """The circuit-breaker emits ``gave_up`` (not ``blocked``), but the
    durable state is still ``status='blocked'`` and must not auto-promote."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child", parents=[parent])
        kb.complete_task(conn, parent, result="ok")

        # Status + event match what _record_task_failure writes when
        # the breaker trips.
        conn.execute(
            "UPDATE tasks SET status='blocked' WHERE id=?", (child,),
        )
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, 'gave_up', NULL, ?)",
            (child, int(time.time())),
        )
        conn.commit()

        promoted = kb.recompute_ready(conn)
        assert promoted == 0
        assert kb.get_task(conn, child).status == "blocked"


# ---------------------------------------------------------------------------
# unblock_task clears the blocked state
# ---------------------------------------------------------------------------


def test_unblock_clears_blocked_state_and_later_blocks_stay_blocked(kanban_home: Path) -> None:
    """``hermes kanban unblock`` (or the ``kanban_unblock`` tool) is
    the only legitimate way out of a blocked task.  A later block on the
    same task must also require explicit unblock."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="t")
        kb.claim_task(conn, tid)
        kb.block_task(
            conn, tid,
            reason="review-required: ...",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        assert kb.unblock_task(conn, tid)
        # After unblock the task is no longer blocked at all.
        assert kb.get_task(conn, tid).status == "ready"

        # Now simulate a *later* circuit-breaker block (no new
        # ``blocked`` event, just status flip).  A blocked task remains
        # blocked until the operator explicitly unblocks it again.
        conn.execute(
            "UPDATE tasks SET status='blocked' WHERE id=?", (tid,),
        )
        conn.commit()

        promoted = kb.recompute_ready(conn)
        assert promoted == 0
        assert kb.get_task(conn, tid).status == "blocked"


# ---------------------------------------------------------------------------
# Full bug-shaped loop: block → promote → crash → gave_up → next tick
# ---------------------------------------------------------------------------


def test_protocol_violation_loop_is_broken(kanban_home: Path) -> None:
    """Reproduces the exact #28712 loop and asserts the dispatcher
    leaves the task blocked instead of cycling.

    Loop shape from the issue:

    1. Worker calls ``kanban_block`` → status='blocked',
       ``task_runs.outcome='blocked'``, ``blocked`` event.
    2. (Bug) Dispatcher promotes back to ``ready``.
    3. Fresh worker exits cleanly without terminal tool call →
       ``protocol_violation`` event.
    4. ``_record_task_failure(failure_limit=1)`` → ``gave_up`` event,
       status='blocked' again.
    5. (Bug) Dispatcher promotes again → infinite loop.

    With the fix in place, step 2 never happens — the test simulates
    one would-be loop cycle by faking the crash-then-gave_up entries
    that *would* have been written and asserts the *next* tick still
    leaves the task blocked.
    """
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="loop reproducer")
        kb.claim_task(conn, tid)
        kb.block_task(
            conn, tid,
            reason="review-required: human eyes please",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        assert kb.get_task(conn, tid).status == "blocked"

        # First dispatcher tick — must NOT promote.
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, tid).status == "blocked"

        # Simulate the (hypothetical) protocol_violation + gave_up
        # entries that the dispatcher would have written if the bug
        # were still present.  Even with those event rows in place,
        # the worker-initiated ``blocked`` event is the most recent
        # of the ``{blocked, unblocked}`` pair, so the sticky guard
        # still fires.
        now = int(time.time())
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, 'protocol_violation', NULL, ?)",
            (tid, now),
        )
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, 'gave_up', NULL, ?)",
            (tid, now + 1),
        )
        conn.commit()

        # Subsequent ticks must still leave it blocked.
        for _ in range(3):
            promoted = kb.recompute_ready(conn)
            assert promoted == 0
            assert kb.get_task(conn, tid).status == "blocked"


# ---------------------------------------------------------------------------
# Schema-init recovery on legacy DBs is covered by
# tests/hermes_cli/test_kanban_db.py::test_connect_migrates_legacy_db_before_optional_column_indexes
# (landed via #28754 / #28781).  The original PR shipped a duplicate test
# here; dropped during salvage to avoid two assertions of the same contract.
# ---------------------------------------------------------------------------
