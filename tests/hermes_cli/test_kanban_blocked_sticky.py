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


# ---------------------------------------------------------------------------
# Schema-init recovery on legacy DBs is covered by
# tests/hermes_cli/test_kanban_db.py::test_connect_migrates_legacy_db_before_optional_column_indexes
# (landed via #28754 / #28781).  The original PR shipped a duplicate test
# here; dropped during salvage to avoid two assertions of the same contract.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# initial_status="blocked" must be sticky from birth
# ---------------------------------------------------------------------------
#
# ``create_task(initial_status="blocked")`` is the documented way to park a
# task for human-ops review ("parks the task directly in the blocked column
# for human-ops review", commit fb96208892).  Before this fix it wrote only a
# ``created`` event, so ``recompute_ready`` classified the row as
# auto-recoverable (the "direct DB manipulation" path preserved by #28712)
# and promoted AND spawned it on the next dispatcher tick — the human gate
# the initial status exists to enforce never engaged.
#
# Contract: a task created directly in ``blocked`` behaves identically to one
# blocked via ``block_task`` — sticky across ticks, releasable only by an
# explicit ``unblock_task``.


def test_initial_status_blocked_survives_recompute_ready(kanban_home: Path) -> None:
    """A task created with initial_status="blocked" must not be auto-promoted.

    Red on base: ``recompute_ready`` promotes the fresh blocked row on the
    first tick because no ``blocked``/``unblocked`` event exists yet.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="human-ops gate", initial_status="blocked")
        assert kb.get_task(conn, tid).status == "blocked"

        for _ in range(5):
            promoted = kb.recompute_ready(conn)
            assert promoted == 0, (
                "task created in blocked must not auto-promote — "
                "the human-ops gate was bypassed"
            )
            assert kb.get_task(conn, tid).status == "blocked"


def test_initial_status_blocked_is_released_by_unblock_task(kanban_home: Path) -> None:
    """unblock_task is the only exit from an initial_status="blocked" park.

    Pins the release path of the human-ops gate: after an explicit unblock
    the task resumes its normal lifecycle (ready), like any other
    sticky-blocked task.
    """
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="human-ops gate", initial_status="blocked")

        assert kb.unblock_task(conn, tid) is True
        assert kb.get_task(conn, tid).status == "ready"

        # The release is durable: subsequent ticks keep it schedulable.
        conn.execute(
            "UPDATE tasks SET status = 'todo' WHERE id = ?", (tid,)
        )
        conn.commit()
        assert kb.recompute_ready(conn) == 1
        assert kb.get_task(conn, tid).status == "ready"
