"""Regression tests for fail-closed Kanban block gates.

The dispatcher may auto-promote dependency waits in ``todo`` after their
parents finish. It must never infer that a ``blocked`` card is runnable:
creation-time blocks, worker/operator blocks, circuit-breaker blocks, and
unknown legacy state all require an explicit ``unblock_task`` transition.
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
# Creation-time blocks must be sticky
# ---------------------------------------------------------------------------


def test_blocked_on_creation_stays_blocked_until_explicit_unblock(
    kanban_home: Path,
) -> None:
    """``initial_status=blocked`` is a human gate, not a dependency wait.

    Creation must persist the same sticky ``blocked`` event used by
    ``block_task`` so dispatcher recomputation cannot silently bypass it.
    Only an explicit unblock may return the card to the runnable queue.
    """
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="human release gate",
            initial_status="blocked",
        )

        events = [
            row["kind"]
            for row in conn.execute(
                "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id",
                (tid,),
            )
        ]
        assert events == ["created", "blocked"]
        assert kb.get_task(conn, tid).status == "blocked"

        for _ in range(3):
            assert kb.recompute_ready(conn) == 0
            assert kb.get_task(conn, tid).status == "blocked"

        assert kb.unblock_task(conn, tid)
        assert kb.get_task(conn, tid).status == "ready"


def test_unknown_block_state_fails_closed(kanban_home: Path) -> None:
    """A blocked row with no auditable gate event must never auto-run."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ambiguous legacy block")
        conn.execute("UPDATE tasks SET status = 'blocked' WHERE id = ?", (tid,))
        conn.commit()

        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, tid).status == "blocked"


def test_ambiguous_reblock_after_unblock_fails_closed(kanban_home: Path) -> None:
    """A blocked row without a matching new block event stays gated."""
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="ambiguous reblock", initial_status="blocked"
        )
        assert kb.unblock_task(conn, tid)

        # Simulate an unsupported/legacy writer changing only the status. The
        # old ``unblocked`` event cannot authorize a later ambiguous block.
        conn.execute("UPDATE tasks SET status = 'blocked' WHERE id = ?", (tid,))
        conn.commit()

        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, tid).status == "blocked"


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




# ---------------------------------------------------------------------------
# Circuit-breaker blocks still auto-recover (preserve #40c1decb3 intent)
# ---------------------------------------------------------------------------


def test_circuit_breaker_block_below_limit_still_fails_closed(
    kanban_home: Path,
) -> None:
    """Failure counters cannot silently release a blocked card."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child", parents=[parent])
        kb.complete_task(conn, parent, result="ok")

        conn.execute(
            "UPDATE tasks SET status='blocked', consecutive_failures=1, "
            "last_failure_error='transient error' WHERE id=?",
            (child,),
        )
        conn.commit()

        assert kb.recompute_ready(conn) == 0
        task = kb.get_task(conn, child)
        assert task.status == "blocked"
        assert task.consecutive_failures == 1

def test_gave_up_block_requires_explicit_unblock(kanban_home: Path) -> None:
    """A circuit-breaker ``gave_up`` event is not an automatic release."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child", parents=[parent])
        kb.complete_task(conn, parent, result="ok")

        conn.execute(
            "UPDATE tasks SET status='blocked' WHERE id=?", (child,),
        )
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, 'gave_up', NULL, ?)",
            (child, int(time.time())),
        )
        conn.commit()

        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, child).status == "blocked"
        assert kb.unblock_task(conn, child)
        assert kb.get_task(conn, child).status == "ready"

# ---------------------------------------------------------------------------
# unblock_task clears the sticky state
# ---------------------------------------------------------------------------


def test_unblock_clears_sticky_state(kanban_home: Path) -> None:
    """Explicit unblock is the only supported exit from blocked state."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="t")
        kb.claim_task(conn, tid)
        kb.block_task(
            conn, tid,
            reason="review-required: ...",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        assert kb.unblock_task(conn, tid)
        assert kb.get_task(conn, tid).status == "ready"
        latest = conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()
        assert latest["kind"] == "unblocked"

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
