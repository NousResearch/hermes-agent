"""A terminal hold on a kanban task must bind at every promotion door.

Two holds are *status-independent* decisions about a task, not properties of
the column it happens to sit in:

* **sticky block** — a worker/operator ``kanban_block`` (or the unblock-loop
  breaker's ``block_loop_detected`` escalation) said "a human decides next".
* **breaker trip** — ``_record_task_failure`` recorded ``gave_up`` and
  persisted the limit it tripped against.

``recompute_ready`` honoured both, but only for rows it found in ``blocked``.
Every other door into the work pool re-derived its own, narrower rule:

* ``claim_task`` had no hold check at all, so any writer that put a held row
  in ``ready`` (a racy promote, a stale-claim release, an old dump, a
  dependency refresh) got a worker spawned on the next tick.
* ``specify_triage_task`` / ``decompose_triage_task`` gated only on the
  ``block_loop_detected`` escalation, so a sticky-blocked or gave-up card
  parked in ``triage`` was promoted straight back to ``todo``.
* ``reclaim_task`` flipped ``blocked -> ready`` and cleared the breaker
  decision, reviving a task nobody had unblocked.

The hold is released by exactly two explicit operator acts — ``unblock_task``
and ``promote_task`` — and by a successful completion. Everything else
(recompute, restart, reclaim, dependency refresh, re-specify) leaves it
standing.

Separately: a spawn failure on a task claimed out of the ``review`` column
requeued it to ``ready``, dropping it out of review and back into the
implementation pool where the next dispatcher tick spawns an ordinary worker
instead of the forced ``sdlc-review`` agent.
"""

from __future__ import annotations

import json
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
# helpers — build the two hold shapes through the real code paths
# ---------------------------------------------------------------------------


def _sticky_blocked(conn, *, title: str = "needs a human", **kwargs) -> str:
    """A task a worker explicitly blocked for review."""
    tid = kb.create_task(conn, title=title, **kwargs)
    kb.claim_task(conn, tid)
    assert kb.block_task(
        conn, tid,
        reason="review-required: verify before continuing",
        kind="needs_input",
        expected_run_id=kb.get_task(conn, tid).current_run_id,
    )
    assert kb.get_task(conn, tid).status == "blocked"
    return tid


def _gave_up(conn, *, title: str = "keeps failing", **kwargs) -> str:
    """A task the circuit breaker gave up on (spawn failures)."""
    tid = kb.create_task(conn, title=title, **kwargs)
    kb.claim_task(conn, tid)
    assert kb._record_spawn_failure(conn, tid, "boom", failure_limit=1) is True
    assert kb.get_task(conn, tid).status == "blocked"
    assert kb.recorded_breaker_limit(conn, tid) is not None
    return tid


def _force_status(conn, task_id: str, status: str) -> None:
    """Put a row in ``status`` behind the API's back.

    Stands in for every writer that can land a held row somewhere it does
    not belong: a racy promote, a restored dump, a dependency refresh, an
    operator's manual SQL.
    """
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = ? WHERE id = ?", (status, task_id),
        )


# ---------------------------------------------------------------------------
# claim_task — the last gate before a worker is spawned
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_held", [_sticky_blocked, _gave_up])
def test_claim_task_refuses_a_held_task_parked_in_ready(kanban_home, make_held):
    """A held row that reaches ``ready`` must not be claimable.

    ``recompute_ready`` is not the only writer of ``status='ready'``, so it
    cannot be the only place the hold is enforced. ``claim_task`` is the
    single door into ``running`` — it must re-check, exactly as it already
    re-checks parent completion.
    """
    with kb.connect() as conn:
        tid = make_held(conn)
        _force_status(conn, tid, "ready")

        assert kb.claim_task(conn, tid) is None
        # And it is demoted back to where a held task belongs, so the next
        # tick doesn't re-litigate the same decision.
        assert kb.get_task(conn, tid).status == "blocked"


@pytest.mark.parametrize("make_held", [_sticky_blocked, _gave_up])
def test_claim_review_task_refuses_a_held_task(kanban_home, make_held):
    """Same gate on the review column's claim door — but no demotion.

    ``claim_task`` demotes a held row out of ``ready`` because ``ready`` is
    the active work pool and ``recompute_ready`` would keep re-examining it.
    ``review`` is neither: ``recompute_ready`` only ever promotes out of
    ``todo``/``blocked``, so refusing the claim already binds the hold.

    Demoting here would be a one-way door. ``unblock_task`` restores
    ``ready`` (or ``todo`` behind open parents) and never ``review``, so the
    card would come back as ordinary implementation work with its PR already
    open — the exact failure ``_claim_source_status`` exists to prevent.
    """
    with kb.connect() as conn:
        tid = make_held(conn)
        _force_status(conn, tid, "review")

        assert kb.claim_review_task(conn, tid) is None
        assert kb.get_task(conn, tid).status == "review"
        # Still not promotable by any automatic door, and still not claimable.
        assert kb.recompute_ready(conn) == 0
        assert kb.claim_review_task(conn, tid) is None
        assert kb.get_task(conn, tid).status == "review"


def test_claim_task_still_works_for_an_unheld_ready_task(kanban_home):
    """The gate must not cost an ordinary task its claim."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ordinary work")
        _force_status(conn, tid, "ready")

        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        assert kb.get_task(conn, tid).status == "running"


def test_below_limit_spawn_failure_stays_claimable(kanban_home):
    """A failure that did NOT trip the breaker is not a hold.

    ``_record_task_failure`` requeues to ``ready`` with the counter bumped
    and no ``gave_up``; that task must still be retried.
    """
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="transient failure")
        kb.claim_task(conn, tid)
        assert kb._record_spawn_failure(conn, tid, "flaky", failure_limit=3) is False
        assert kb.get_task(conn, tid).status == "ready"
        assert kb.recorded_breaker_limit(conn, tid) is None

        assert kb.claim_task(conn, tid) is not None


# ---------------------------------------------------------------------------
# recompute_ready — restart / dependency refresh
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_held", [_sticky_blocked, _gave_up])
def test_recompute_ready_holds_a_held_todo_row(kanban_home, make_held):
    """A held row parked in ``todo`` must not be promoted either.

    ``recompute_ready`` selects ``todo`` and ``blocked`` and only checked
    the hold on the ``blocked`` branch — so anything that moved a held card
    to ``todo`` (specify, a dependency-flavoured re-block, manual SQL) got
    it promoted on the very next tick.
    """
    with kb.connect() as conn:
        tid = make_held(conn)
        _force_status(conn, tid, "todo")

        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, tid).status != "ready"


def test_held_task_survives_a_restart_with_a_laxer_limit(kanban_home):
    """A fresh connection + a laxer configured limit must not free it."""
    with kb.connect() as conn:
        tid = _gave_up(conn)

    # "Restart": new connection, dispatcher configured with a bigger budget.
    with kb.connect() as conn:
        for _ in range(3):
            assert kb.recompute_ready(conn, failure_limit=99) == 0
            assert kb.get_task(conn, tid).status == "blocked"


def test_dependency_refresh_does_not_free_a_held_child(kanban_home):
    """Parents completing is not an unblock.

    The dependency-refresh path exists so a task blocked purely on a parent
    frees itself. A task with its own terminal hold must stay put even when
    every parent lands ``done``.
    """
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        child = _sticky_blocked(conn, title="held child")
        kb.link_tasks(conn, parent, child)
        _force_status(conn, parent, "done")

        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, child).status == "blocked"


# ---------------------------------------------------------------------------
# triage promotion doors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_held", [_sticky_blocked, _gave_up])
def test_specify_triage_task_refuses_a_held_card(kanban_home, make_held):
    """``specify`` must not launder a held card into ``todo``.

    It already refuses the ``block_loop_detected`` escalation; a plain
    sticky block and a recorded give-up are the same class of human hold.
    """
    with kb.connect() as conn:
        tid = make_held(conn)
        _force_status(conn, tid, "triage")

        assert kb.specify_triage_task(
            conn, tid, title="rescoped", author="operator",
        ) is False
        assert kb.get_task(conn, tid).status == "triage"


@pytest.mark.parametrize("make_held", [_sticky_blocked, _gave_up])
def test_decompose_triage_task_refuses_a_held_card(kanban_home, make_held):
    """Neither may auto-decompose fan a held card out into the work pool."""
    with kb.connect() as conn:
        tid = make_held(conn)
        _force_status(conn, tid, "triage")

        assert kb.decompose_triage_task(
            conn, tid,
            root_assignee=None,
            children=[{"title": "step one"}, {"title": "step two"}],
        ) is None
        assert kb.get_task(conn, tid).status == "triage"
        assert kb.list_tasks(conn, limit=100) and len(
            [t for t in kb.list_tasks(conn, limit=100)]
        ) == 1


def test_specify_triage_task_still_promotes_an_unheld_card(kanban_home):
    """Ordinary triage flow is untouched."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="fresh idea", triage=True)
        assert kb.get_task(conn, tid).status == "triage"

        assert kb.specify_triage_task(
            conn, tid, title="scoped idea", author="operator",
        ) is True
        # No open parents, and specify runs a promotion pass on the way out.
        assert kb.get_task(conn, tid).status == "ready"


# ---------------------------------------------------------------------------
# reclaim — an operator recovery, not an unblock
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_held", [_sticky_blocked, _gave_up])
def test_reclaim_does_not_revive_a_held_task(kanban_home, make_held):
    """Reclaim releases a worker; it does not overrule a terminal hold.

    A held row that still carries a fence (the requeue path records one
    when a process group outlived its kill) passes reclaim's "is there
    anything to release?" gate — and the reclaim then flipped it to
    ``ready`` and cleared the breaker decision on the way out.
    """
    with kb.connect() as conn:
        tid = make_held(conn)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET worker_fence = ? WHERE id = ?",
                (
                    json.dumps({
                        "pid": 999999, "pgid": 999999, "run_id": 1,
                        "claim_lock": "gone", "identity": None,
                        "recorded_at": int(time.time()),
                        "reason": "unconfirmed_termination",
                    }),
                    tid,
                ),
            )

        kb.reclaim_task(conn, tid, reason="operator poke", signal_fn=lambda *a: None)

        assert kb.get_task(conn, tid).status == "blocked"
        assert kb.claim_task(conn, tid) is None
        assert kb.recompute_ready(conn) == 0


def test_reclaim_does_not_clear_a_trip_recorded_only_in_the_event_log(
    kanban_home,
):
    """The two representations of "the trip was reset" must agree.

    ``recorded_breaker_limit`` reads ``tasks.breaker_limit`` and falls back
    to the event log for boards written before that column existed (or
    restored from such a dump). Keeping the *column* across a reclaim is
    therefore only half the decision: if ``"reclaimed"`` still counts as a
    reset in the event-log fallback, the reclaim's own event clears the trip
    a tick later and the task promotes itself — the same resurrection by a
    slower route, on exactly the boards the fallback exists for.

    A reclaim recovers a worker. It is not an operator decision about the
    retry budget in either representation.
    """
    with kb.connect() as conn:
        tid = _gave_up(conn)
        # A board whose trip survives only as the ``gave_up`` event.
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET breaker_limit = NULL, worker_fence = ? "
                "WHERE id = ?",
                (
                    json.dumps({
                        "pid": 999999, "pgid": 999999, "run_id": 1,
                        "claim_lock": "gone", "identity": None,
                        "recorded_at": int(time.time()),
                        "reason": "unconfirmed_termination",
                    }),
                    tid,
                ),
            )
        assert kb.recorded_breaker_limit(conn, tid) is not None
        assert kb.terminal_hold_reason(conn, tid) == "breaker_trip"

        kb.reclaim_task(conn, tid, reason="operator poke", signal_fn=lambda *a: None)

        assert kb.terminal_hold_reason(conn, tid) == "breaker_trip"
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, tid).status == "blocked"
        assert kb.claim_task(conn, tid) is None


def test_unblock_still_clears_a_trip_recorded_only_in_the_event_log(
    kanban_home,
):
    """The operator escapes must keep working on those same boards."""
    with kb.connect() as conn:
        tid = _gave_up(conn)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET breaker_limit = NULL WHERE id = ?", (tid,),
            )
        assert kb.terminal_hold_reason(conn, tid) == "breaker_trip"

        assert kb.unblock_task(conn, tid) is True
        assert kb.terminal_hold_reason(conn, tid) is None
        assert kb.claim_task(conn, tid) is not None


def test_reclaim_still_releases_a_running_worker(kanban_home):
    """The recovery path itself keeps working for an unheld task."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="stuck worker")
        kb.claim_task(conn, tid)

        assert kb.reclaim_task(
            conn, tid, reason="hallucinating", signal_fn=lambda *a: None,
        ) is True
        assert kb.get_task(conn, tid).status == "ready"


# ---------------------------------------------------------------------------
# the two explicit operator escapes still work
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_held", [_sticky_blocked, _gave_up])
def test_unblock_frees_a_held_task(kanban_home, make_held):
    with kb.connect() as conn:
        tid = make_held(conn)

        assert kb.unblock_task(conn, tid) is True
        assert kb.get_task(conn, tid).status == "ready"
        assert kb.claim_task(conn, tid) is not None


@pytest.mark.parametrize("make_held", [_sticky_blocked, _gave_up])
def test_manual_promote_frees_a_held_task(kanban_home, make_held):
    """``hermes kanban promote`` is a deliberate operator override.

    It already clears the breaker decision and the worker fence; the claim
    gate must read it as a release too, or a promoted card would be demoted
    straight back to ``blocked`` by the very next tick and never dispatch.
    """
    with kb.connect() as conn:
        tid = make_held(conn)

        ok, err = kb.promote_task(conn, tid, actor="operator", reason="looked at it")
        assert (ok, err) == (True, None)
        assert kb.get_task(conn, tid).status == "ready"

        assert kb.recompute_ready(conn) == 0  # already ready, nothing to do
        assert kb.get_task(conn, tid).status == "ready"
        assert kb.claim_task(conn, tid) is not None


def test_parent_gated_todo_still_promotes(kanban_home):
    """The ordinary dependency promotion is untouched."""
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent")
        child = kb.create_task(conn, title="child", parents=(parent,))
        assert kb.get_task(conn, child).status == "todo"

        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, child).status == "todo"

        _force_status(conn, parent, "done")
        assert kb.recompute_ready(conn) == 1
        assert kb.get_task(conn, child).status == "ready"
        assert kb.claim_task(conn, child) is not None


# ---------------------------------------------------------------------------
# goal_mode
# ---------------------------------------------------------------------------


def test_goal_mode_budget_block_is_terminal(kanban_home):
    """A goal-loop worker that exhausts its turn budget sticky-blocks.

    That card must not come back through any automatic door — the goal loop
    would otherwise re-spawn against the same exhausted budget forever.
    """
    with kb.connect() as conn:
        tid = _sticky_blocked(
            conn, title="ralph loop", goal_mode=True, goal_max_turns=3,
        )
        assert kb.get_task(conn, tid).goal_mode is True

        for _ in range(3):
            assert kb.recompute_ready(conn) == 0
        _force_status(conn, tid, "ready")
        assert kb.claim_task(conn, tid) is None
        assert kb.get_task(conn, tid).status == "blocked"

        # The operator escape still resumes it.
        assert kb.unblock_task(conn, tid) is True
        assert kb.claim_task(conn, tid) is not None


# ---------------------------------------------------------------------------
# terminal columns stay terminal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("terminal_status", ["done", "archived"])
def test_completed_and_superseded_tasks_stay_unclaimable(
    kanban_home, terminal_status,
):
    """A finished or superseded card is never promoted or claimed."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="finished work")
        _force_status(conn, tid, terminal_status)

        assert kb.recompute_ready(conn) == 0
        assert kb.claim_task(conn, tid) is None
        assert kb.claim_review_task(conn, tid) is None
        assert kb.get_task(conn, tid).status == terminal_status


def test_completion_clears_the_hold_for_a_later_rerun(kanban_home):
    """A success is the third release: the decision it reversed is history."""
    with kb.connect() as conn:
        tid = _gave_up(conn)
        assert kb.unblock_task(conn, tid) is True
        kb.claim_task(conn, tid)
        kb.complete_task(conn, tid, summary="done at last")

        assert kb.recorded_breaker_limit(conn, tid) is None
        _force_status(conn, tid, "ready")
        assert kb.claim_task(conn, tid) is not None


# ---------------------------------------------------------------------------
# review column requeue
# ---------------------------------------------------------------------------


def test_review_spawn_failure_requeues_to_review(kanban_home):
    """A spawn failure on a review claim belongs back in ``review``.

    The dispatcher force-loads the ``sdlc-review`` skill for tasks it claims
    out of ``review``. Requeueing to ``ready`` drops that: the next tick
    picks the card up as ordinary work and spawns a plain implementation
    worker against a task whose PR is already open.
    """
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="PR awaiting review", assignee="reviewer")
        _force_status(conn, tid, "review")
        claimed = kb.claim_review_task(conn, tid)
        assert claimed is not None

        assert kb._record_spawn_failure(
            conn, tid, "workspace: no such worktree", failure_limit=3,
        ) is False
        assert kb.get_task(conn, tid).status == "review"


def test_review_spawn_failure_still_trips_the_breaker(kanban_home):
    """The requeue column must not cost the review card its breaker."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="PR awaiting review", assignee="reviewer")
        _force_status(conn, tid, "review")
        assert kb.claim_review_task(conn, tid) is not None

        assert kb._record_spawn_failure(
            conn, tid, "boom", failure_limit=1,
        ) is True
        assert kb.get_task(conn, tid).status == "blocked"
        assert kb.recorded_breaker_limit(conn, tid) is not None


def test_ready_spawn_failure_still_requeues_to_ready(kanban_home):
    """The ordinary spawn-failure requeue is unchanged."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="ordinary work")
        assert kb.claim_task(conn, tid) is not None

        assert kb._record_spawn_failure(
            conn, tid, "transient", failure_limit=3,
        ) is False
        assert kb.get_task(conn, tid).status == "ready"
