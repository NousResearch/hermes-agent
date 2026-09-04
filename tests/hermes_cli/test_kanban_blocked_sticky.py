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
* Ordinary circuit-breaker blocks (``gave_up`` without protocol-breaker
  metadata) still auto-recover — the original intent of #40c1decb3 is
  preserved.
* A protocol breaker that exhausts its violation-only budget stays blocked.
* An explicit ``kanban_unblock`` clears either sticky state.
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
# Protocol-breaker blocks are sticky; ordinary failures keep retry semantics
# ---------------------------------------------------------------------------


def _drive_protocol_violation(
    conn, task_id: str, pid: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reap one clean worker exit without a terminal kanban call."""
    import hermes_cli.kanban_db as live_kb

    host = live_kb._claimer_id().split(":", 1)[0]
    claimed = live_kb.claim_task(conn, task_id, claimer=f"{host}:test")
    assert claimed is not None
    live_kb._set_worker_pid(conn, task_id, pid)
    live_kb._record_worker_exit(pid, 0)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(live_kb, "_pid_alive", lambda _pid: False)
    assert task_id in live_kb.detect_crashed_workers(conn)


def test_protocol_breaker_stays_blocked_until_explicit_unblock(
    kanban_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three clean protocol exits trip a sticky breaker.

    A later dispatcher tick must neither promote nor spawn the task.  An
    explicit operator unblock is still allowed to start a fresh attempt.
    """
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="protocol breaker", assignee="worker")

        for attempt in range(kb._PROTOCOL_VIOLATION_FAILURE_LIMIT):
            _drive_protocol_violation(conn, tid, 81000 + attempt, monkeypatch)
            expected = (
                "blocked"
                if attempt + 1 == kb._PROTOCOL_VIOLATION_FAILURE_LIMIT
                else "ready"
            )
            assert kb.get_task(conn, tid).status == expected

        gave_up = [e for e in kb.list_events(conn, tid) if e.kind == "gave_up"]
        assert len(gave_up) == 1
        assert (gave_up[0].payload or {})["protocol_violations"] == 3
        assert (gave_up[0].payload or {})["protocol_violation_limit"] == 3

        before_tick = conn.execute(
            "SELECT COALESCE(MAX(id), 0) AS id FROM task_events WHERE task_id = ?",
            (tid,),
        ).fetchone()["id"]
        spawn_calls = []
        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda *args, **kwargs: spawn_calls.append((args, kwargs)),
            reconcile_orphans=False,
        )

        assert result.promoted == 0
        assert result.spawned == []
        assert spawn_calls == []
        tick_kinds = {
            row["kind"]
            for row in conn.execute(
                "SELECT kind FROM task_events WHERE task_id = ? AND id > ?",
                (tid, before_tick),
            )
        }
        assert tick_kinds.isdisjoint({"promoted", "claimed", "spawned"})
        assert kb.get_task(conn, tid).status == "blocked"

        assert kb.unblock_task(conn, tid)
        assert kb.get_task(conn, tid).status == "ready"
        assert kb.claim_task(conn, tid, claimer="operator-approved-retry") is not None


def test_non_protocol_failures_keep_existing_retry_semantics(
    kanban_home: Path,
) -> None:
    """The protocol guard must not make unrelated transient failures sticky."""
    with kb.connect() as conn:
        retryable = kb.create_task(conn, title="transient", assignee="worker")
        assert kb.claim_task(conn, retryable) is not None
        assert not kb._record_task_failure(
            conn,
            retryable,
            "temporary worker failure",
            outcome="spawn_failed",
            failure_limit=5,
            release_claim=True,
            end_run=True,
        )
        task = kb.get_task(conn, retryable)
        assert task.status == "ready"
        assert task.consecutive_failures == 1
        assert kb.claim_task(conn, retryable) is not None


def test_ordinary_gave_up_recovers_when_failure_limit_increases(
    kanban_home: Path,
) -> None:
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ordinary breaker", assignee="worker")
        assert kb.claim_task(conn, tid) is not None
        assert kb._record_task_failure(
            conn,
            tid,
            "temporary worker failure",
            outcome="spawn_failed",
            failure_limit=1,
            release_claim=True,
            end_run=True,
        )
        assert kb.get_task(conn, tid).status == "blocked"

        gave_up = next(e for e in kb.list_events(conn, tid) if e.kind == "gave_up")
        assert {"protocol_violations", "protocol_violation_limit"}.isdisjoint(
            gave_up.payload or {}
        )
        assert kb.recompute_ready(conn, failure_limit=2) == 1
        assert kb.get_task(conn, tid).status == "ready"




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
# Schema-init recovery on legacy DBs is covered by
# tests/hermes_cli/test_kanban_db.py::test_connect_migrates_legacy_db_before_optional_column_indexes
# (landed via #28754 / #28781).  The original PR shipped a duplicate test
# here; dropped during salvage to avoid two assertions of the same contract.
# ---------------------------------------------------------------------------
