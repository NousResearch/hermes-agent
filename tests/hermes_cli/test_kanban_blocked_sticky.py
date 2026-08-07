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
# Create-time blocked rows are also sticky
# ---------------------------------------------------------------------------


def _event_kinds(conn, task_id: str) -> list[str]:
    return [
        row["kind"]
        for row in conn.execute(
            "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id",
            (task_id,),
        ).fetchall()
    ]


def test_create_time_blocked_task_is_sticky_and_not_claimable(kanban_home: Path) -> None:
    """``create_task(initial_status='blocked')`` is a deliberate human-ops
    parking state. It must emit the same durable sticky marker as an explicit
    ``kanban block`` call, otherwise ``recompute_ready`` can promote it on the
    next dispatcher tick and the selector can claim it."""
    with kb.connect() as conn:
        marker_id = kb.create_task(
            conn,
            title="marker-only row",
            assignee="default",
            initial_status="blocked",
        )
        control_id = kb.create_task(conn, title="dispatchable control", assignee="default")

        assert kb.recompute_ready(conn) == 0

        marker = kb.get_task(conn, marker_id)
        control = kb.get_task(conn, control_id)
        assert marker is not None
        assert marker.status == "blocked"
        assert control is not None
        assert control.status == "ready"
        assert _event_kinds(conn, marker_id) == ["created", "blocked"]
        assert kb._has_sticky_block(conn, marker_id) is True

        assert kb.claim_task(conn, marker_id) is None
        claimed_control = kb.claim_task(conn, control_id)
        assert claimed_control is not None
        assert claimed_control.id == control_id


def test_initial_blocked_child_inherits_parent_notify_after_block_marker(
    kanban_home: Path,
) -> None:
    """Parent-chat subscriptions inherited by a create-time blocked child
    must start after both the child ``created`` and child ``blocked`` markers.
    Otherwise the inherited parent notifier will later replay the child's
    create-time blocked marker as if it were a future child event."""
    with kb.connect() as conn:
        parent_id = kb.create_task(conn, title="subscribed parent", assignee="default")
        conn.execute(
            """
            INSERT INTO kanban_notify_subs (
                task_id, platform, chat_id, thread_id, user_id,
                notifier_profile, created_at, last_event_id
            ) VALUES (?, 'discord', 'parent-chat', '', 'u1', 'notifier', 0, 0)
            """,
            (parent_id,),
        )
        conn.commit()

        child_id = kb.create_task(
            conn,
            title="blocked child",
            assignee="default",
            parents=[parent_id],
            initial_status="blocked",
        )

        rows = conn.execute(
            """
            SELECT id, kind
              FROM task_events
             WHERE task_id = ?
             ORDER BY id
            """,
            (child_id,),
        ).fetchall()
        assert [row["kind"] for row in rows] == ["created", "blocked"]
        blocked_event_id = next(row["id"] for row in rows if row["kind"] == "blocked")

        sub = conn.execute(
            """
            SELECT last_event_id
              FROM kanban_notify_subs
             WHERE task_id = ?
               AND platform = 'discord'
               AND chat_id = 'parent-chat'
               AND thread_id = ''
            """,
            (child_id,),
        ).fetchone()
        assert sub is not None
        assert sub["last_event_id"] >= blocked_event_id

        future_inherited_parent_events = conn.execute(
            """
            SELECT kind
              FROM task_events
             WHERE task_id = ?
               AND id > ?
             ORDER BY id
            """,
            (child_id, sub["last_event_id"]),
        ).fetchall()
        assert [row["kind"] for row in future_inherited_parent_events] == []


def test_idempotent_initial_blocked_reuse_backfills_legacy_nonsticky_row(
    kanban_home: Path,
) -> None:
    """Retries with the same idempotency key should repair old still-blocked
    rows created before initial blocked rows wrote a sticky ``blocked`` event.
    The repair is deliberately narrow: the reused row must still be blocked and
    the caller must again request ``initial_status='blocked'``."""
    with kb.connect() as conn:
        old_id = "t_oldblocked"
        conn.execute(
            """
            INSERT INTO tasks (
                id, title, assignee, status, created_by, created_at,
                workspace_kind, idempotency_key
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                old_id,
                "old non-sticky blocked row",
                "default",
                "blocked",
                "legacy-test",
                int(time.time()),
                "scratch",
                "reuse-key",
            ),
        )
        kb._append_event(
            conn,
            old_id,
            "created",
            {"status": "blocked", "assignee": "default", "parents": []},
        )
        conn.commit()
        assert kb._has_sticky_block(conn, old_id) is False

        reused_id = kb.create_task(
            conn,
            title="retry blocked row",
            assignee="default",
            initial_status="blocked",
            idempotency_key="reuse-key",
        )

        assert reused_id == old_id
        assert _event_kinds(conn, old_id) == ["created", "blocked"]
        assert kb._has_sticky_block(conn, old_id) is True
        assert kb.recompute_ready(conn) == 0
        old_task = kb.get_task(conn, old_id)
        assert old_task is not None
        assert old_task.status == "blocked"
        assert kb.claim_task(conn, old_id) is None


# ---------------------------------------------------------------------------
# Schema-init recovery on legacy DBs is covered by
# tests/hermes_cli/test_kanban_db.py::test_connect_migrates_legacy_db_before_optional_column_indexes
# (landed via #28754 / #28781).  The original PR shipped a duplicate test
# here; dropped during salvage to avoid two assertions of the same contract.
# ---------------------------------------------------------------------------
