"""Tests for typed block reasons + the unblock-loop breaker.

Covers the built-in fix for the kanban "blocked loop" — a worker blocks a
task, a cron unblocks it, the worker re-blocks for the same reason, repeat
forever. The fix gives ``block_task`` a typed ``kind`` and a persistent
``block_recurrences`` counter:

* ``dependency`` blocks route to ``todo`` (parent-gated, auto-resumed) and
  never enter the human ``blocked`` bucket a cron would keep unblocking —
  unless no parent is open, in which case the wait can never be satisfied
  and the block is recorded as ``needs_input`` (sticky, loop-counted).
* ``needs_input`` / ``capability`` / un-typed blocks land in ``blocked``;
  each same-cause re-block after an unblock increments ``block_recurrences``,
  and at ``BLOCK_RECURRENCE_LIMIT`` the task routes to ``triage`` for a human.
* ``unblock_task`` deliberately does NOT reset ``block_recurrences`` (the
  amnesia that let the loop run unbounded).
* A successful ``complete_task`` resets the loop memory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from hermes_cli import kanban as kanban_cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _running_task(conn, title="t", parents=()):
    """Create a task (linked under ``parents`` first) and drive it to ``running`` so block_task can act."""
    tid = kb.create_task(conn, title=title, assignee="worker")
    for parent in parents:
        kb.link_tasks(conn, parent_id=parent, child_id=tid)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    claimed = kb.claim_task(conn, tid, claimer="worker")
    assert claimed is not None
    return tid


def _make_running_again(conn, tid):
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None


# ---------------------------------------------------------------------------
# Loop breaker
# ---------------------------------------------------------------------------










def test_block_loop_detected_event_emitted(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="x", kind="capability")
        kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        kb.block_task(conn, tid, reason="x", kind="capability")
        events = [e for e in kb.list_events(conn, tid)
                  if e.kind == "block_loop_detected"]
        assert events, "expected a block_loop_detected event"
        payload = events[-1].payload or {}
        assert payload.get("recurrences") == 2
        assert payload.get("kind") == "capability"


def test_reason_churn_does_not_dodge_the_breaker(kanban_home: Path) -> None:
    """#28712/#130239: the counter is reason-blind — free-text reasons can be
    reworded at zero cost, so keying the reset on the sentence lets a stalled
    claim loop run forever (20 cycles, eight sentences, loop_detected=0).
    Same kind keeps counting however the reason is phrased; the supervisor's
    escape hatch is the explicit triage re-park, not a reworded reason."""
    with kbc.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="awaiting api key", kind="needs_input")
        kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        # Second block: same kind, completely different sentence — must still
        # be recurrence 2, i.e. the breaker trips exactly like on a repeated
        # sentence (base behaviour).
        kb.block_task(conn, tid, reason="schema mismatch", kind="needs_input")
        task = kb.get_task(conn, tid)
        assert task.status == "triage"
        loops = [e for e in kb.list_events(conn, tid) if e.kind == "block_loop_detected"]
        assert loops and loops[-1].payload["recurrences"] == kb.BLOCK_RECURRENCE_LIMIT


def _poison_to_triage(conn, tid: str, reason: str = "same stall") -> None:
    """Drive one same-cause block/unblock/re-block cycle past the limit."""
    kb.block_task(conn, tid, reason=reason, kind="needs_input")
    kb.unblock_task(conn, tid)
    _make_running_again(conn, tid)
    kb.block_task(conn, tid, reason=reason, kind="needs_input")
    assert kb.get_task(conn, tid).status == "triage"


def test_poisoned_triage_card_reparks_idempotently(kanban_home: Path) -> None:
    """#130239: a breaker-poisoned triage card re-asserts its park instead of
    erroring forever; the re-park re-arms the counter at 1."""
    with kbc.connect_closing() as conn:
        tid = _running_task(conn)
        _poison_to_triage(conn, tid)
        # The failing receipt: the re-park used to return False forever.
        assert kb.block_task(conn, tid, reason="same stall", kind="needs_input") is True
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.block_kind == "needs_input"
        assert task.block_recurrences == 1
        reparks = [e for e in kb.list_events(conn, tid)
                   if e.kind == "blocked" and (e.payload or {}).get("reparked")]
        assert reparks and reparks[-1].payload["reason"] == "same stall"
        # The unlatched card closes the loop: unblock lands it back in the pool.
        assert kb.unblock_task(conn, tid) is True
        assert kb.get_task(conn, tid).status == "ready"


def test_triage_repark_refuses_ownership_asserting_caller(kanban_home: Path) -> None:
    """A worker's run cannot own a poisoned card; only the supervisor re-asserts."""
    with kbc.connect_closing() as conn:
        tid = _running_task(conn)
        _poison_to_triage(conn, tid)
        assert kb.block_task(conn, tid, reason="probe", kind="needs_input",
                             expected_run_id=1) is False
        assert kb.get_task(conn, tid).status == "triage"


def _reviewer_run_poisoned_to_triage(conn, tid: str) -> None:
    """Drive a *reviewer-blocked* card through block/unblock/re-block to triage.

    The reviewer lane's identity lives in the run's ``claimed`` event
    (``source_status=review`` via ``claim_review_task``), which
    ``_retry_status_for_run`` propagates into every block payload."""
    kb.request_review(conn, tid, force=True)
    assert kb.get_task(conn, tid).status == "review"
    assert kb.claim_review_task(conn, tid, claimer="reviewer") is not None
    assert kb.block_task(conn, tid, reason="changes requested", kind="needs_input")
    assert kb.unblock_task(conn, tid)
    assert kb.get_task(conn, tid).status == "review"
    assert kb.claim_review_task(conn, tid, claimer="reviewer") is not None
    assert kb.block_task(conn, tid, reason="changes requested", kind="needs_input")
    assert kb.get_task(conn, tid).status == "triage"


def test_triage_repark_preserves_reviewer_phase(kanban_home: Path) -> None:
    """A reviewer-blocked card that tripped the breaker and was re-parked must
    resume in ``review`` after the unblock, not silently drop to ``ready`` —
    ``unblock_task`` picks the landing lane from the newest lifecycle event's
    ``source_status``, so the re-park event has to carry the phase forward
    from the ``block_loop_detected`` payload it re-asserts."""
    with kbc.connect_closing() as conn:
        tid = _running_task(conn)
        _reviewer_run_poisoned_to_triage(conn, tid)
        assert kb.block_task(conn, tid, reason="changes requested", kind="needs_input") is True
        repark = [e for e in kb.list_events(conn, tid)
                  if e.kind == "blocked" and (e.payload or {}).get("reparked")][-1]
        assert repark.payload["source_status"] == "review"
        assert kb.unblock_task(conn, tid) is True
        assert kb.get_task(conn, tid).status == "review"


# ---------------------------------------------------------------------------
# Dependency routing
# ---------------------------------------------------------------------------


def test_dependency_then_parent_done_promotes(kanban_home: Path) -> None:
    """A dependency-parked child becomes ready once its parent completes."""
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(
            conn,
            parent_id=parent,
            child_id=child,
            expected_child_run_id=kb.get_task(conn, child).current_run_id,
        )
        kb.block_task(conn, child, reason="wait", kind="dependency")
        assert kb.get_task(conn, child).status == "todo"
        # Finish the parent, then let recompute_ready run.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        kb.claim_task(conn, parent, claimer="worker")
        kb.complete_task(conn, parent, result="done")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"


def test_dependency_block_with_terminal_parents_parks_then_escalates(
    kanban_home: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """A ``dependency`` block whose parents are all terminal can never be
    satisfied by ``recompute_ready``: it must park in ``blocked`` as
    ``needs_input`` (no ``dependency_wait``, no re-promotion), say so on the
    CLI, and count toward the loop breaker so a re-block after an unblock
    reaches ``triage``."""
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="already-done-parent", assignee="worker")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='done' WHERE id=?", (parent,))
        # The edge exists before the run: link_tasks refuses to gate a running child retroactively.
        child = _running_task(conn, title="child-of-done", parents=(parent,))

        # `hermes kanban block <child> --kind dependency waiting on upstream`
        args = argparse.Namespace(task_id=child, ids=None, reason=["waiting", "on", "upstream"], kind="dependency")
        assert kanban_cli._cmd_block(args) == 0
        assert "needs_input" in capsys.readouterr().out
        parked = kb.get_task(conn, child)
        assert (parked.status, parked.block_kind, parked.block_recurrences) == ("blocked", "needs_input", 1)
        events = kb.list_events(conn, child)
        assert not [e for e in events if e.kind == "dependency_wait"]
        blocked = [e for e in events if e.kind == "blocked"][-1].payload
        assert (blocked["requested_kind"], blocked["rekind_reason"]) == ("dependency", "no_open_parent")
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, child).status == "blocked"

        # A cron/human unblocks; the worker re-declares the same impossible wait.
        assert kb.unblock_task(conn, child)
        assert kb.claim_task(conn, child, claimer="worker") is not None
        assert kb.block_task(conn, child, reason="waiting on upstream", kind="dependency")
        assert kb.get_task(conn, child).status == "triage"
        loop = [e for e in kb.list_events(conn, child) if e.kind == "block_loop_detected"][-1].payload
        assert loop["recurrences"] == kb.BLOCK_RECURRENCE_LIMIT


def test_dependency_block_with_open_parent_stays_parked_across_dispatch_tick(
    kanban_home: Path, all_assignees_spawnable,
) -> None:
    """Control: a genuine wait on an incomplete parent parks in ``todo``
    without counting a recurrence, survives a dispatch tick unspawned, and
    resumes once the parent finishes."""
    spawns: list[str] = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 4242

    with kbc.connect() as conn:
        parent = kb.create_task(conn, title="open-parent", assignee="alice")
        # Linked while todo (a running child cannot be gated retroactively); the parent then stays
        # open while the child runs — the reopened-parent shape, forced the same way the loop below does.
        child = kb.create_task(conn, title="waiter", assignee="worker")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='running' WHERE id=?", (child,))
        for _ in range(kb.BLOCK_RECURRENCE_LIMIT + 1):
            assert kb.block_task(conn, child, reason="wait", kind="dependency")
            parked = kb.get_task(conn, child)
            assert (parked.status, parked.block_kind, parked.block_recurrences) == ("todo", "dependency", 0)
            with kb.write_txn(conn):
                conn.execute("UPDATE tasks SET status='running' WHERE id=?", (child,))
        assert kb.block_task(conn, child, reason="wait", kind="dependency")
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn)
        assert kb.get_task(conn, child).status == "todo"
        assert child not in spawns
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        kb.claim_task(conn, parent, claimer="alice")
        kb.complete_task(conn, parent, result="done")
        res2 = kbd.dispatch_once(conn, spawn_fn=fake_spawn)
        assert child in [row[0] for row in res2.spawned]


# ---------------------------------------------------------------------------
# Completion resets loop memory
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Validation + back-compat
# ---------------------------------------------------------------------------


