"""Regression tests for #28712 — kanban dispatcher must not auto-promote
worker-initiated ``kanban_block`` (sticky blocks), but must keep
auto-recovering circuit-breaker blocks.

The bug: when a worker called ``kanban_block(reason="review-required:
...")`` to hand off to a human, the dispatcher's ``recompute_ready``
would promote the task back to ``ready`` on the next tick.  The fresh
worker found nothing to do (work already applied), exited cleanly, and
got recorded as a ``protocol_violation`` → ``gave_up`` → promote → loop
until manual intervention.

These tests pin down:

* Worker / operator-initiated blocks are sticky and survive
  ``recompute_ready``.
* Circuit-breaker blocks (``gave_up`` event, status flipped via
  ``_record_task_failure``) still auto-recover — the original intent
  of #40c1decb3 is preserved.
* An explicit ``kanban_unblock`` clears the sticky state.
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
# Circuit-breaker blocks still auto-recover (preserve #40c1decb3 intent)
# ---------------------------------------------------------------------------


def test_circuit_breaker_block_still_auto_promotes(kanban_home: Path) -> None:
    """A child that was put into ``blocked`` *without* a worker-issued
    ``kanban_block`` (e.g. circuit-breaker after repeated spawn
    failures, manual DB triage) must still get auto-promoted when its
    parents complete — preserves the pre-#28712 recovery semantics."""
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
        assert promoted == 1
        task = kb.get_task(conn, child)
        assert task.status == "ready"
        assert task.consecutive_failures == 0
        assert task.last_failure_error is None


def test_gave_up_event_alone_does_not_make_block_sticky(kanban_home: Path) -> None:
    """The circuit-breaker emits ``gave_up`` (not ``blocked``).  Make
    sure ``_has_sticky_block`` doesn't accidentally treat ``gave_up``
    as sticky — otherwise we'd regress the safety net for genuinely
    transient crashes."""
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
        assert promoted == 1
        assert kb.get_task(conn, child).status == "ready"


# ---------------------------------------------------------------------------
# unblock_task clears the sticky state
# ---------------------------------------------------------------------------


def test_unblock_clears_sticky_state_and_lets_block_recover(kanban_home: Path) -> None:
    """``hermes kanban unblock`` (or the ``kanban_unblock`` tool) is
    the only legitimate way out of a worker-initiated block.  After
    unblock, a *subsequent* circuit-breaker block on the same task
    must again be eligible for auto-recovery."""
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
        # ``blocked`` event, just status flip).  The most recent
        # block/unblock event is ``unblocked`` → guard does not fire
        # → recompute can recover.
        conn.execute(
            "UPDATE tasks SET status='blocked' WHERE id=?", (tid,),
        )
        conn.commit()

        promoted = kb.recompute_ready(conn)
        assert promoted == 1
        assert kb.get_task(conn, tid).status == "ready"


# ---------------------------------------------------------------------------
# Full bug-shaped loop: block → promote → crash → gave_up → next tick
# ---------------------------------------------------------------------------


def test_dispatch_repairs_legacy_ready_task_with_sticky_block(kanban_home: Path) -> None:
    """If an older dispatcher already promoted a review-required block
    back to ``ready``, the next dispatcher tick must repair the task to
    ``blocked`` instead of leaving it ready-but-respawn-guarded."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="already corrupted review task")
        kb.claim_task(conn, tid)
        task = kb.get_task(conn, tid)
        assert task is not None
        kb.block_task(
            conn,
            tid,
            reason="review-required: review the branch",
            expected_run_id=task.current_run_id,
        )

        # Reproduce the live corrupt state: a sticky block event exists,
        # but a buggy promotion flipped the task back to ready afterward.
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, 'promoted', NULL, ?)",
            (tid, int(time.time())),
        )
        conn.commit()
        ready_task = kb.get_task(conn, tid)
        assert ready_task is not None
        assert ready_task.status == "ready"
        assert kb._has_sticky_block(conn, tid)

        promoted = kb.recompute_ready(conn)

        assert promoted == 0
        repaired = kb.get_task(conn, tid)
        assert repaired is not None
        assert repaired.status == "blocked"


def test_claim_refuses_ready_task_with_sticky_block(kanban_home: Path) -> None:
    """The ready->running claim boundary is the final defensive gate:
    review-required sticky blocks are blockers for this task, not new work."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="sticky ready must not claim")
        kb.claim_task(conn, tid)
        task = kb.get_task(conn, tid)
        assert task is not None
        kb.block_task(
            conn,
            tid,
            reason="review-required: needs approval",
            expected_run_id=task.current_run_id,
        )
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        conn.commit()

        claimed = kb.claim_task(conn, tid)

        assert claimed is None
        repaired = kb.get_task(conn, tid)
        assert repaired is not None
        assert repaired.status == "blocked"


def test_recompute_repairs_corrupted_sticky_todo_before_promotion(kanban_home: Path) -> None:
    """A sticky block that somehow landed in ``todo`` must be repaired
    before parent-completion promotion can make it runnable again."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child", parents=[parent])
        initial = kb.get_task(conn, child)
        assert initial is not None
        assert initial.status == "todo"
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, 'blocked', ?, ?)",
            (child, '{"reason":"review-required: todo repair"}', int(time.time())),
        )
        conn.commit()

        kb.complete_task(conn, parent, result="parent ok")
        assert kb.recompute_ready(conn) == 0

        repaired = kb.get_task(conn, child)
        assert repaired is not None
        assert repaired.status == "blocked"


def test_claim_repairs_sticky_ready_before_parent_demote(kanban_home: Path) -> None:
    """A sticky-ready child with unfinished parents must be restored to
    blocked, not demoted to ``todo`` and later promoted when parents finish."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child", parents=[parent])

        # Reproduce a corrupt intermediate state that should never become
        # runnable: the child has a sticky review block event but status was
        # incorrectly moved to ready while its parent is still open.
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (child,))
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, 'blocked', ?, ?)",
            (child, '{"reason":"review-required: parent-gated review"}', int(time.time())),
        )
        conn.commit()

        claimed = kb.claim_task(conn, child)

        assert claimed is None
        repaired = kb.get_task(conn, child)
        assert repaired is not None
        assert repaired.status == "blocked"

        kb.complete_task(conn, parent, result="parent ok")
        assert kb.recompute_ready(conn) == 0
        final = kb.get_task(conn, child)
        assert final is not None
        assert final.status == "blocked"


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
