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
* Out-of-process bridge blocks obey the same sticky lifecycle.
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

import json
import sqlite3
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
            kind="needs_input",
            expected_run_id=kb.get_task(conn, tid).current_run_id,
        )
        assert kb.get_task(conn, tid).status == "blocked"

        promoted_ok, promote_error = kb.promote_task(
            conn, tid, actor="test-operator", force=True,
        )
        assert not promoted_ok
        assert promote_error is not None and "use unblock" in promote_error

        # Hammer the promotion code — exactly the dispatcher loop's
        # behaviour, just compressed in time.
        for _ in range(5):
            promoted = kb.recompute_ready(conn)
            assert promoted == 0, "worker-blocked task must not auto-promote"
            assert kb.get_task(conn, tid).status == "blocked"

        # Recovery can preserve the classified blocked task row while losing
        # the event tail. Fail closed on that durable classification instead
        # of silently promoting the recovered task.
        conn.execute(
            "DELETE FROM task_events WHERE task_id=? AND kind='blocked'", (tid,),
        )
        conn.commit()
        assert kb.recompute_ready(conn) == 0
        recovered = kb.get_task(conn, tid)
        assert recovered is not None
        assert recovered.status == "blocked"
        assert kb.claim_task(conn, tid) is None


def test_init_backfills_recovered_canonical_block_without_claiming_bridge_block(
    kanban_home: Path,
) -> None:
    """Upgrade backfill protects event-loss rows but not the bridge channel."""
    with kb.connect() as conn:
        recovered = kb.create_task(conn, title="recovered canonical block")
        conn.execute(
            "UPDATE tasks SET status='blocked', block_kind='needs_input', "
            "operator_blocked=0 WHERE id=?",
            (recovered,),
        )
        conn.execute(
            "DELETE FROM task_events WHERE task_id=? AND kind IN "
            "('blocked', 'unblocked', 'bridge_blocked', "
            "'bridge_requeued', 'bridge_dispatched')",
            (recovered,),
        )

        bridge = kb.create_task(conn, title="bridge-owned block")
        conn.execute(
            "UPDATE tasks SET status='blocked', block_kind='needs_input', "
            "operator_blocked=0 WHERE id=?",
            (bridge,),
        )
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, 'bridge_blocked', NULL, ?)",
            (bridge, int(time.time())),
        )

    # Force the additive migration/backfill pass, as an upgraded install does
    # on its first connection after restart.
    kb.init_db()
    with kb.connect() as conn:
        flags = {
            row["id"]: row["operator_blocked"]
            for row in conn.execute(
                "SELECT id, operator_blocked FROM tasks WHERE id IN (?, ?)",
                (recovered, bridge),
            )
        }
        assert flags == {recovered: 1, bridge: 0}
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='trigger' AND name=?",
            (kb._OPERATOR_BLOCK_GUARD_TRIGGER,),
        ).fetchone() is not None

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "UPDATE tasks SET status='ready', block_kind=NULL WHERE id=?",
                (recovered,),
            )

        # A bridge-owned block remains on its independent lifecycle and is not
        # accidentally converted into a canonical operator block by backfill.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET status='ready', block_kind=NULL WHERE id=?",
                (bridge,),
            )
            conn.execute(
                "INSERT INTO task_events (task_id, kind, payload, created_at) "
                "VALUES (?, 'bridge_requeued', NULL, ?)",
                (bridge, int(time.time())),
            )
        bridge_task = kb.get_task(conn, bridge)
        assert bridge_task is not None
        assert bridge_task.status == "ready"


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
# Out-of-process bridge blocks must share the sticky lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("clear_event", ["bridge_requeued", "bridge_dispatched"])
def test_bridge_block_is_sticky_until_bridge_clear_transition(
    kanban_home: Path,
    clear_event: str,
) -> None:
    """A bridge-managed worker emits bridge-specific transition names.

    Treating ``bridge_blocked`` as ordinary direct DB manipulation lets the
    dispatcher immediately re-promote review-gated work.  A later explicit
    bridge requeue or dispatch must clear that sticky state, just like
    ``unblock_task`` clears a canonical worker block.
    """
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="bridge-managed review")
        now = int(time.time())
        conn.execute("UPDATE tasks SET status='blocked' WHERE id=?", (tid,))
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, 'bridge_blocked', NULL, ?)",
            (tid, now),
        )
        conn.commit()

        assert kb.recompute_ready(conn) == 0
        blocked_task = kb.get_task(conn, tid)
        assert blocked_task is not None
        assert blocked_task.status == "blocked"

        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, ?, NULL, ?)",
            (tid, clear_event, now + 1),
        )
        conn.commit()

        # A later transient status-only block is recoverable once the bridge
        # has explicitly cleared its prior review gate.
        assert kb.recompute_ready(conn) == 1
        ready_task = kb.get_task(conn, tid)
        assert ready_task is not None
        assert ready_task.status == "ready"

        # A canonical operator unblock is also authorized to clear an older
        # bridge block.
        canonical_clear = kb.create_task(conn, title="bridge block, operator clear")
        conn.execute(
            "UPDATE tasks SET status='blocked' WHERE id=?", (canonical_clear,),
        )
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, 'bridge_blocked', NULL, ?)",
            (canonical_clear, now + 2),
        )
        conn.commit()
        assert kb.unblock_task(conn, canonical_clear)
        assert not kb._has_sticky_block(conn, canonical_clear)

        # Bridge receipts only clear the bridge channel. They must never
        # substitute for the canonical `unblocked` event required to release
        # an operator block.
        canonical = kb.create_task(conn, title="canonical operator review")
        assert kb.claim_task(conn, canonical) is not None
        canonical_task = kb.get_task(conn, canonical)
        assert canonical_task is not None
        assert kb.block_task(
            conn,
            canonical,
            reason="review-required: operator approval",
            kind="needs_input",
            expected_run_id=canonical_task.current_run_id,
        )
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, ?, NULL, ?)",
            (canonical, clear_event, now + 2),
        )
        conn.commit()

        assert kb.recompute_ready(conn) == 0
        canonical_task = kb.get_task(conn, canonical)
        assert canonical_task is not None
        assert canonical_task.status == "blocked"
        assert kb.claim_task(conn, canonical) is None


def _apply_real_bridge_transition(
    conn: sqlite3.Connection,
    task_id: str,
    clear_event: str,
) -> None:
    """Execute the live bridge's exact UPDATE-then-event transaction."""
    now = int(time.time())
    with kb.write_txn(conn):
        if clear_event == "bridge_dispatched":
            conn.execute(
                """UPDATE tasks
                   SET status = 'running', assignee = ?,
                       started_at = COALESCE(started_at, ?), completed_at = NULL,
                       workspace_path = COALESCE(?, workspace_path),
                       branch_name = COALESCE(?, branch_name),
                       claim_lock = ?, claim_expires = ?, worker_pid = ?,
                       last_heartbeat_at = ?, current_run_id = NULL,
                       block_kind = NULL, last_failure_error = NULL
                   WHERE id = ?""",
                (
                    "orchestrator",
                    now,
                    None,
                    None,
                    "acp:test:external",
                    now + 7200,
                    None,
                    now,
                    task_id,
                ),
            )
        else:
            conn.execute(
                """UPDATE tasks
                   SET status = ?, assignee = ?, completed_at = ?,
                       workspace_path = COALESCE(?, workspace_path),
                       branch_name = COALESCE(?, branch_name),
                       claim_lock = NULL, claim_expires = NULL, worker_pid = NULL,
                       last_heartbeat_at = ?, current_run_id = NULL,
                       block_kind = CASE WHEN ? = 'blocked' THEN 'needs_input' ELSE NULL END,
                       last_failure_error = COALESCE(?, last_failure_error),
                       result = COALESCE(?, result)
                   WHERE id = ?""",
                (
                    "ready",
                    "orchestrator",
                    None,
                    None,
                    None,
                    now,
                    "ready",
                    None,
                    None,
                    task_id,
                ),
            )
        conn.execute(
            "INSERT INTO task_events(task_id, run_id, kind, payload, created_at) "
            "VALUES (?, NULL, ?, ?, ?)",
            (
                task_id,
                clear_event,
                json.dumps({"from": "blocked", "to": clear_event}),
                now,
            ),
        )


@pytest.mark.parametrize(
    "clear_event",
    ["bridge_requeued", "bridge_dispatched"],
)
def test_canonical_block_rejects_real_bridge_update_before_event(
    kanban_home: Path,
    clear_event: str,
) -> None:
    """The live bridge rewrites the task before appending its receipt.

    Model that exact update+event transaction for both ready and running
    transitions. SQLite authority must abort the rewrite atomically; checking
    an event predicate later in ``recompute_ready`` or ``claim_task`` is too
    late because the visible blocked state has already been erased.
    """
    with kb.connect() as conn:
        tid = kb.create_task(conn, title=f"canonical block vs {clear_event}")
        claimed = kb.claim_task(conn, tid, claimer="test:worker")
        assert claimed is not None
        assert kb.block_task(
            conn,
            tid,
            reason="review-required: operator approval",
            kind="needs_input",
            expected_run_id=claimed.current_run_id,
        )
        with pytest.raises(
            sqlite3.IntegrityError,
            match="operator-blocked task requires authoritative unblock",
        ):
            _apply_real_bridge_transition(conn, tid, clear_event)

        task = kb.get_task(conn, tid)
        assert task is not None
        assert task.status == "blocked"
        assert task.block_kind == "needs_input"
        assert conn.execute(
            "SELECT operator_blocked FROM tasks WHERE id = ?", (tid,),
        ).fetchone()["operator_blocked"] == 1
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM task_events "
            "WHERE task_id = ? AND kind = ?",
            (tid, clear_event),
        ).fetchone()["n"] == 0
        assert kb.claim_task(conn, tid, claimer="test:claim") is None


@pytest.mark.parametrize(
    "clear_event",
    ["bridge_requeued", "bridge_dispatched"],
)
def test_init_backfills_recovered_reblock_with_missing_latest_event(
    kanban_home: Path,
    clear_event: str,
) -> None:
    """A surviving blocked run must preserve a recovered canonical re-block."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title=f"recovered reblock vs {clear_event}")
        first = kb.claim_task(conn, tid, claimer="test:first")
        assert first is not None
        assert kb.block_task(
            conn,
            tid,
            reason="review-required: first operator gate",
            kind="needs_input",
            expected_run_id=first.current_run_id,
        )
        assert kb.unblock_task(conn, tid)

        second = kb.claim_task(conn, tid, claimer="test:second")
        assert second is not None
        assert kb.block_task(
            conn,
            tid,
            reason="capability gate after recovery",
            kind="capability",
            expected_run_id=second.current_run_id,
        )
        latest_run = conn.execute(
            "SELECT outcome FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()
        assert latest_run is not None and latest_run["outcome"] == "blocked"

        # Model an upgrade from a recovered DB: the new authority column starts
        # at its additive default, and only the newest canonical block event was
        # lost. Older blocked -> unblocked history still survives, which must not
        # outweigh the current classified row plus its latest blocked run.
        conn.execute(
            "DELETE FROM task_events WHERE id = ("
            "SELECT MAX(id) FROM task_events WHERE task_id=? AND kind='blocked')",
            (tid,),
        )
        conn.execute("UPDATE tasks SET operator_blocked=0 WHERE id=?", (tid,))
        conn.execute(f"DROP TRIGGER IF EXISTS {kb._OPERATOR_BLOCK_GUARD_TRIGGER}")
        conn.commit()
        lifecycle = [
            row["kind"]
            for row in conn.execute(
                "SELECT kind FROM task_events WHERE task_id=? "
                "AND kind IN ('blocked', 'unblocked') ORDER BY id",
                (tid,),
            )
        ]
        assert lifecycle == ["blocked", "unblocked"]

    kb.init_db()
    with kb.connect() as conn:
        row = conn.execute(
            "SELECT status, block_kind, operator_blocked FROM tasks WHERE id=?",
            (tid,),
        ).fetchone()
        assert row is not None
        assert (row["status"], row["block_kind"], row["operator_blocked"]) == (
            "blocked", "capability", 1,
        )
        with pytest.raises(
            sqlite3.IntegrityError,
            match="operator-blocked task requires authoritative unblock",
        ):
            _apply_real_bridge_transition(conn, tid, clear_event)
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM task_events WHERE task_id=? AND kind=?",
            (tid, clear_event),
        ).fetchone()["n"] == 0
        assert kb.claim_task(conn, tid, claimer="test:claim") is None


def test_init_does_not_claim_stale_block_kind_after_circuit_breaker(
    kanban_home: Path,
) -> None:
    """Known failure-run evidence keeps circuit-breaker recovery automatic."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="operator unblock then circuit breaker")
        first = kb.claim_task(conn, tid, claimer="test:first")
        assert first is not None
        assert kb.block_task(
            conn,
            tid,
            reason="review-required: prior operator gate",
            kind="needs_input",
            expected_run_id=first.current_run_id,
        )
        assert kb.unblock_task(conn, tid)

        second = kb.claim_task(conn, tid, claimer="test:second")
        assert second is not None
        assert kb._record_task_failure(
            conn,
            tid,
            "transient spawn failure",
            outcome="spawn_failed",
            failure_limit=1,
            release_claim=True,
            end_run=True,
        )
        row = conn.execute(
            "SELECT status, block_kind, operator_blocked FROM tasks WHERE id=?",
            (tid,),
        ).fetchone()
        assert row is not None
        assert (row["status"], row["block_kind"], row["operator_blocked"]) == (
            "blocked", "needs_input", 0,
        )
        assert conn.execute(
            "SELECT outcome FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1",
            (tid,),
        ).fetchone()["outcome"] == "gave_up"

    kb.init_db()
    with kb.connect() as conn:
        assert conn.execute(
            "SELECT operator_blocked FROM tasks WHERE id=?", (tid,),
        ).fetchone()["operator_blocked"] == 0
        assert kb.recompute_ready(conn, failure_limit=2) == 1
        recovered = kb.get_task(conn, tid)
        assert recovered is not None
        assert recovered.status == "ready"


# ---------------------------------------------------------------------------
# Circuit-breaker blocks still auto-recover (preserve #40c1decb3 intent)
# ---------------------------------------------------------------------------


def test_circuit_breaker_block_still_auto_promotes(kanban_home: Path) -> None:
    """A child that was put into ``blocked`` *without* a worker-issued
    ``kanban_block`` (e.g. a transient crash, manual DB triage) and whose
    ``consecutive_failures`` is still *below* the circuit-breaker limit
    must get auto-promoted when its parents complete — preserves the
    pre-#28712 recovery semantics for genuinely transient failures.

    The complementary case — a block whose failure count has *reached*
    the limit must stay blocked — is covered by
    ``test_kanban_db.py::test_recompute_ready_skips_tasks_at_failure_limit``
    (#35072).  Together they pin the contract: ``recompute_ready`` defers
    the give-up decision to the same effective limit the breaker uses, so
    the two never disagree.
    """
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child", parents=[parent])
        kb.complete_task(conn, parent, result="ok")

        # Simulate a transient circuit-breaker / direct triage that flips
        # status without emitting a ``blocked`` event — exactly what
        # ``_record_task_failure`` does below the limit.  One failure is
        # under the default limit (2), so recovery is still correct.
        conn.execute(
            "UPDATE tasks SET status='blocked', consecutive_failures=1, "
            "last_failure_error='transient error' WHERE id=?",
            (child,),
        )
        conn.commit()

        promoted = kb.recompute_ready(conn)
        assert promoted == 1
        task = kb.get_task(conn, child)
        assert task.status == "ready"
        # Counter is preserved across recovery (not reset) so the breaker
        # can still accumulate if the task keeps failing (#35072).
        assert task.consecutive_failures == 1


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
