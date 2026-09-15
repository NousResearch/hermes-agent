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
from hermes_cli import kanban_db_connect as kbc


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
    with kbc.connect() as conn:
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




# ---------------------------------------------------------------------------
# Circuit-breaker blocks still auto-recover (preserve #40c1decb3 intent)
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# unblock_task clears the sticky state
# ---------------------------------------------------------------------------


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
    with kbc.connect() as conn:
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


def test_created_with_initial_status_blocked_is_not_promoted_by_recompute_ready(kanban_home: Path) -> None:
    """Verify a task created with initial_status='blocked' remains blocked when parents complete."""
    with kbc.connect() as conn:
        parent_id = kb.create_task(conn, title="parent task")
        child_id = kb.create_task(
            conn, title="gated child task", parents=[parent_id], initial_status="blocked"
        )
        assert kb.get_task(conn, child_id).status == "blocked"

        # Complete parent task
        kb.claim_task(conn, parent_id)
        kb.complete_task(conn, parent_id, result="done")
        assert kb.get_task(conn, parent_id).status == "done"

        # recompute_ready must NOT promote the blocked child task
        promoted = kb.recompute_ready(conn)
        assert promoted == 0
        assert kb.get_task(conn, child_id).status == "blocked"


# ---------------------------------------------------------------------------
# Unattributed parks: a block with neither a failure record nor a block event
# ---------------------------------------------------------------------------


def test_direct_status_write_to_blocked_is_not_promoted_by_recompute_ready(kanban_home: Path) -> None:
    """A ``blocked`` row carrying no recorded failure and no ``blocked``
    event — the unattributed park an operator creates by writing the
    status directly — must stay blocked across arbitrary dispatcher
    ticks. Nothing attributes that block to the circuit breaker, so
    ``recompute_ready`` promoting it would spawn a worker on a card a
    human deliberately parked."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="parked by hand", assignee="a")
        assert kb.get_task(conn, tid).status == "ready"

        conn.execute("UPDATE tasks SET status='blocked' WHERE id=?", (tid,))
        conn.commit()
        assert kb.get_task(conn, tid).status == "blocked"

        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, tid).status == "blocked"

        # A second tick must hold it too — the dispatcher loops.
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, tid).status == "blocked"


def test_unblocked_then_status_write_blocked_is_not_promoted(kanban_home: Path) -> None:
    """The real-incident shape: a worker parks the card via
    ``block_task``, an operator clears the sticky block via
    ``unblock_task``, and the card is then re-parked with a direct
    status write. ``unblock_task`` resets ``consecutive_failures`` and
    leaves ``unblocked`` as the newest ``blocked``/``unblocked`` event,
    so nothing attributes the new park to the breaker —
    ``recompute_ready`` must leave it blocked."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="re-parked after unblock")
        kb.claim_task(conn, tid)
        assert kb.block_task(
            conn, tid,
            reason="needs_input: waiting on the operator",
            kind="needs_input",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        assert kb.get_task(conn, tid).status == "blocked"
        assert kb.unblock_task(conn, tid)
        assert kb.get_task(conn, tid).status == "ready"

        conn.execute("UPDATE tasks SET status='blocked' WHERE id=?", (tid,))
        conn.commit()
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, tid).status == "blocked"


def test_failure_recorded_block_still_auto_recovers_below_limit(kanban_home: Path) -> None:
    """The invariant the zero-evidence guard must not swallow: a block
    that IS attributable to the breaker — ``consecutive_failures >= 1``,
    below the effective limit — still auto-recovers to ``ready``. This
    is the path pinned by
    ``test_kanban_db.py::test_recompute_ready_honours_dispatcher_failure_limit``."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="breaker-blocked", assignee="a")
        conn.execute(
            "UPDATE tasks SET status='blocked', consecutive_failures=1 WHERE id=?",
            (tid,),
        )
        conn.commit()
        assert kb.recompute_ready(conn, failure_limit=3) == 1
        assert kb.get_task(conn, tid).status == "ready"

