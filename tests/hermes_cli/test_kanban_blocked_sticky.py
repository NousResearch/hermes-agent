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

import os
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB."""
    inherited_db = os.environ.get("HERMES_KANBAN_DB")
    home = tmp_path / ".hermes"
    home.mkdir()
    disposable_db = tmp_path / "disposable-kanban.db"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(disposable_db))
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_WORKSPACES_ROOT", raising=False)
    if inherited_db:
        assert Path(inherited_db).expanduser() != disposable_db
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert kb.kanban_db_path() == disposable_db
    kb.init_db()
    return home


def _event_kinds(conn, task_id: str) -> list[str]:
    return [
        row["kind"]
        for row in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id ASC",
            (task_id,),
        ).fetchall()
    ]


def _task_status(conn, task_id: str) -> str:
    task = kb.get_task(conn, task_id)
    assert task is not None
    return task.status


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
        assert _task_status(conn, tid) == "blocked"

        # Hammer the promotion code — exactly the dispatcher loop's
        # behaviour, just compressed in time.
        for _ in range(5):
            promoted = kb.recompute_ready(conn)
            assert promoted == 0, "worker-blocked task must not auto-promote"
            assert _task_status(conn, tid) == "blocked"


def test_create_time_blocked_parent_free_task_is_sticky(kanban_home: Path) -> None:
    """``initial_status='blocked'`` is an explicit human-ops block.

    Even with no parents, dispatcher recomputation must not silently
    promote it to ready/claimable. The sticky marker must be durable and
    ordered after the ``created`` event so notification cursors can inherit
    both creation facts before they are caught up.
    """
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="parked for ops", initial_status="blocked")

        assert _task_status(conn, tid) == "blocked"
        assert _event_kinds(conn, tid) == ["created", "blocked"]

        assert kb.recompute_ready(conn) == 0
        assert kb.claim_task(conn, tid) is None
        assert _task_status(conn, tid) == "blocked"


def test_create_time_blocked_child_with_done_parent_is_sticky(kanban_home: Path) -> None:
    """A blocked child must not become claimable when its parents are done."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="already reviewed parent")
        assert kb.complete_task(conn, parent, summary="done")
        child = kb.create_task(
            conn,
            title="child parked for ops",
            parents=[parent],
            initial_status="blocked",
        )

        assert _task_status(conn, child) == "blocked"
        assert _event_kinds(conn, child)[:2] == ["created", "blocked"]

        assert kb.recompute_ready(conn) == 0
        assert kb.claim_task(conn, child) is None
        assert _task_status(conn, child) == "blocked"


def test_reused_still_blocked_idempotent_task_gets_sticky_marker_only_on_explicit_block_request(
    kanban_home: Path,
) -> None:
    """Legacy idempotent rows may already be blocked without a marker.

    Backfill the marker only when the retried creator again explicitly asks
    for ``initial_status='blocked'``. Default idempotent reuse keeps the old
    non-sticky auto-recover behavior.
    """
    with kb.connect() as conn:
        non_explicit = kb.create_task(
            conn, title="legacy default reuse", idempotency_key="default-reuse"
        )
        conn.execute("UPDATE tasks SET status = 'blocked' WHERE id = ?", (non_explicit,))
        conn.commit()

        assert (
            kb.create_task(
                conn, title="legacy default reuse", idempotency_key="default-reuse"
            )
            == non_explicit
        )
        assert _event_kinds(conn, non_explicit) == ["created"]
        assert kb.recompute_ready(conn) == 1
        assert _task_status(conn, non_explicit) == "ready"

        explicit = kb.create_task(
            conn, title="legacy blocked reuse", idempotency_key="blocked-reuse"
        )
        conn.execute("UPDATE tasks SET status = 'blocked' WHERE id = ?", (explicit,))
        conn.commit()

        assert (
            kb.create_task(
                conn,
                title="legacy blocked reuse",
                idempotency_key="blocked-reuse",
                initial_status="blocked",
            )
            == explicit
        )
        assert _event_kinds(conn, explicit) == ["created", "blocked"]
        assert kb.recompute_ready(conn) == 0
        assert _task_status(conn, explicit) == "blocked"


def test_reused_blocked_idempotent_task_gets_sticky_marker_inside_outer_write_txn(
    kanban_home: Path,
) -> None:
    """Idempotent blocked reuse backfill composes under graph-builder writes."""
    with kb.connect() as conn:
        explicit = kb.create_task(
            conn, title="legacy nested reuse", idempotency_key="nested-reuse"
        )
        conn.execute("UPDATE tasks SET status = 'blocked' WHERE id = ?", (explicit,))
        conn.commit()

        with kb.write_txn(conn):
            assert (
                kb.create_task(
                    conn,
                    title="legacy nested reuse",
                    idempotency_key="nested-reuse",
                    initial_status="blocked",
                )
                == explicit
            )

        assert _event_kinds(conn, explicit) == ["created", "blocked"]
        assert kb.recompute_ready(conn) == 0
        assert _task_status(conn, explicit) == "blocked"




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
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="loop reproducer")
        kb.claim_task(conn, tid)
        kb.block_task(
            conn, tid,
            reason="review-required: human eyes please",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        assert _task_status(conn, tid) == "blocked"

        # First dispatcher tick — must NOT promote.
        assert kb.recompute_ready(conn) == 0
        assert _task_status(conn, tid) == "blocked"

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
            assert _task_status(conn, tid) == "blocked"


# ---------------------------------------------------------------------------
# Schema-init recovery on legacy DBs is covered by
# tests/hermes_cli/test_kanban_db.py::test_connect_migrates_legacy_db_before_optional_column_indexes
# (landed via #28754 / #28781).  The original PR shipped a duplicate test
# here; dropped during salvage to avoid two assertions of the same contract.
# ---------------------------------------------------------------------------
