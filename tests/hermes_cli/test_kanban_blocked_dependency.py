"""Regression: a task waiting on a *blocked* parent must not be silently asleep.

Reproduced lifecycle (orchestrator root + one child, wired exactly like
``kanban_db_graph.decompose_triage_task``):

1. the child exhausts ``kanban.failure_limit`` attempts;
2. ``_record_task_failure`` trips the breaker: child status ``blocked``,
   ``gave_up`` event, ``consecutive_failures`` at the limit;
3. ``recompute_ready`` only promotes a waiter whose parents are all
   ``done``/``archived``, so the root stays ``todo`` forever;
4. no dispatcher lane reads ``todo``, and ``promote_task`` refuses while the
   parent is unsatisfied.

The root therefore carries no distress signal of its own even though its work
is permanently stalled. These tests pin the lifecycle facts (which must NOT
change: a failed child must never satisfy a success dependency) and the
``blocked_dependency`` diagnostic that makes the waiting side actionable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_diagnostics as kd

FAILURE_LIMIT = 2


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _build(conn) -> tuple[str, str]:
    """Root depends on child, as decompose_triage_task wires a fan-out."""
    child = kb.create_task(conn, title="child work", assignee="default")
    root = kb.create_task(conn, title="root goal", assignee="foreman", parents=[child])
    assert kb.get_task(conn, root).status == "todo"
    assert kb.get_task(conn, child).status == "ready"
    return root, child


def _fail_once(conn, child: str, attempt: int) -> bool:
    """One full attempt: claim, then record a non-success outcome."""
    kb.claim_task(conn, child)
    return kbd._record_task_failure(
        conn, child,
        error=f"deterministic boom {attempt}",
        outcome="crashed",
        failure_limit=FAILURE_LIMIT,
        release_claim=True,
        end_run=True,
    )


def _exhaust(conn, child: str) -> None:
    for attempt in range(1, FAILURE_LIMIT + 1):
        tripped = _fail_once(conn, child, attempt)
    assert tripped, "breaker must trip on the final permitted attempt"


def _diags(conn, task_id: str) -> list:
    return kd.compute_task_diagnostics(
        kb.get_task(conn, task_id),
        kb.list_events(conn, task_id),
        kb.list_runs(conn, task_id),
        config={"failure_limit": FAILURE_LIMIT},
        graph=kb.task_graph_context(conn, task_id),
    )


def _kinds(conn, task_id: str) -> set[str]:
    return {d.kind for d in _diags(conn, task_id)}


# ---------------------------------------------------------------------------
# 1. Happy path — unchanged
# ---------------------------------------------------------------------------


def test_successful_child_promotes_root_and_raises_nothing(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        root, child = _build(conn)
        kb.claim_task(conn, child)
        assert kb.complete_task(conn, child, result="shipped")
        assert kb.get_task(conn, child).status == "done"

        # complete_task runs recompute_ready itself, so the root is already
        # released; a further pass is a no-op, not a second promotion.
        assert kb.get_task(conn, root).status == "ready"
        assert kb.recompute_ready(conn, failure_limit=FAILURE_LIMIT) == 0
        assert "blocked_dependency" not in _kinds(conn, root)


# ---------------------------------------------------------------------------
# 2. Transient failure — retried, no distress signal on the waiter
# ---------------------------------------------------------------------------


def test_failure_below_limit_retries_and_does_not_flag_the_waiter(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        root, child = _build(conn)
        assert _fail_once(conn, child, 1) is False

        c = kb.get_task(conn, child)
        assert c.status == "ready", "below the limit the child goes back to the ready lane"
        assert c.consecutive_failures == 1
        assert kb.get_task(conn, root).status == "todo"
        assert "blocked_dependency" not in _kinds(conn, root)


# ---------------------------------------------------------------------------
# 3. Retry exhaustion — terminal, not successful, and does not spin
# ---------------------------------------------------------------------------


def test_retry_exhaustion_blocks_child_without_spinning(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        root, child = _build(conn)
        _exhaust(conn, child)

        c = kb.get_task(conn, child)
        assert c.status == "blocked"
        assert c.status != "done", "an exhausted child must never read as successful"
        assert c.consecutive_failures == FAILURE_LIMIT
        assert c.last_failure_error
        assert "gave_up" in [e.kind for e in kb.list_events(conn, child)]

        # No spawnable work anywhere: the child is breaker-blocked, the root is todo.
        assert kbd.has_spawnable_ready(conn) is False
        assert kbd.has_spawnable_review(conn) is False

        for _ in range(3):
            assert kb.recompute_ready(conn, failure_limit=FAILURE_LIMIT) == 0
            assert kb.get_task(conn, child).status == "blocked"
            assert kb.get_task(conn, child).consecutive_failures == FAILURE_LIMIT


# ---------------------------------------------------------------------------
# 4. The waiting root gets an actionable signal
# ---------------------------------------------------------------------------


def test_root_waiting_on_exhausted_child_is_flagged(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        root, child = _build(conn)
        _exhaust(conn, child)
        kb.recompute_ready(conn, failure_limit=FAILURE_LIMIT)

        # The lifecycle facts that make this invisible without a diagnostic.
        assert kb.get_task(conn, root).status == "todo"
        ok, err = kb.promote_task(conn, root, actor="operator")
        assert ok is False and child in (err or "")

        diags = _diags(conn, root)
        flagged = [d for d in diags if d.kind == "blocked_dependency"]
        assert len(flagged) == 1
        d = flagged[0]
        assert d.severity == "error"
        assert d.data["blocked_parent_ids"] == [child]
        assert d.data["waiting_task_id"] == root
        # Every action names a real recovery command for this exact pair.
        commands = [a.payload.get("command", "") for a in d.actions]
        assert any(child in c for c in commands)
        assert any(f"unlink {child} {root}" in c for c in commands)


def test_operator_recovery_clears_the_signal(kanban_home: Path) -> None:
    """Unblock + finish the child: the root promotes and the signal clears."""
    with kbc.connect() as conn:
        root, child = _build(conn)
        _exhaust(conn, child)
        assert "blocked_dependency" in _kinds(conn, root)

        assert kb.unblock_task(conn, child)
        kb.claim_task(conn, child)
        assert kb.complete_task(conn, child, result="fixed by hand")

        # complete_task runs recompute_ready itself, so the root is already
        # released; a further pass is a no-op, not a second promotion.
        assert kb.get_task(conn, root).status == "ready"
        assert kb.recompute_ready(conn, failure_limit=FAILURE_LIMIT) == 0
        assert "blocked_dependency" not in _kinds(conn, root)


# ---------------------------------------------------------------------------
# 5. A deliberate human block is not auto-promoted, and also flags the waiter
# ---------------------------------------------------------------------------


def test_sticky_operator_block_holds_and_flags_the_waiter(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        root, child = _build(conn)
        kb.claim_task(conn, child)
        assert kb.block_task(
            conn, child, reason="need the production credential", kind="needs_input",
            expected_run_id=kb.get_task(conn, child).current_run_id,
        )
        assert kb.get_task(conn, child).status == "blocked"

        for _ in range(3):
            assert kb.recompute_ready(conn, failure_limit=FAILURE_LIMIT) == 0
            assert kb.get_task(conn, child).status == "blocked"

        assert "blocked_dependency" in _kinds(conn, root)


def test_dependency_wait_parent_is_not_flagged(kanban_home: Path) -> None:
    """``kind=dependency`` parks in ``todo``, not ``blocked`` — a self-resolving
    wait must not raise a distress signal on its waiter."""
    with kbc.connect() as conn:
        root, child = _build(conn)
        kb.claim_task(conn, child)
        assert kb.block_task(
            conn, child, reason="waiting on a sibling", kind="dependency",
            expected_run_id=kb.get_task(conn, child).current_run_id,
        )
        assert kb.get_task(conn, child).status == "todo"
        assert "blocked_dependency" not in _kinds(conn, root)


# ---------------------------------------------------------------------------
# 6. Idempotent reconciliation
# ---------------------------------------------------------------------------


def test_reconciliation_is_idempotent_over_the_failed_graph(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        root, child = _build(conn)
        _exhaust(conn, child)

        def snapshot() -> tuple:
            c, r = kb.get_task(conn, child), kb.get_task(conn, root)
            return (
                c.status, c.consecutive_failures, c.last_failure_error,
                r.status,
                tuple(e.kind for e in kb.list_events(conn, child)),
                tuple(e.kind for e in kb.list_events(conn, root)),
            )

        first = snapshot()
        for _ in range(5):
            assert kb.recompute_ready(conn, failure_limit=FAILURE_LIMIT) == 0
            assert snapshot() == first, "repeated passes must not churn state"
            assert [d.kind for d in _diags(conn, root)].count("blocked_dependency") == 1


# ---------------------------------------------------------------------------
# 7. Restart durability
# ---------------------------------------------------------------------------


def test_failure_state_survives_reopening_the_board(kanban_home: Path) -> None:
    with kbc.connect() as conn:
        root, child = _build(conn)
        _exhaust(conn, child)
        before = kb.get_task(conn, child)
        expected = (before.status, before.consecutive_failures, before.last_failure_error)

    # Fresh connection — everything must come back off disk.
    with kbc.connect() as conn:
        after = kb.get_task(conn, child)
        assert (after.status, after.consecutive_failures, after.last_failure_error) == expected
        assert "gave_up" in [e.kind for e in kb.list_events(conn, child)]
        assert kb.get_task(conn, root).status == "todo"
        assert "blocked_dependency" in _kinds(conn, root)


# ---------------------------------------------------------------------------
# 8. A stale worker must not convert a breaker-tripped attempt into success
# ---------------------------------------------------------------------------


def test_stale_worker_cannot_complete_a_breaker_tripped_task(kanban_home: Path) -> None:
    """The run that lost its claim to the breaker is invalidated; a late
    ``kanban_complete`` carrying that run id must not flip the task to done.

    ``tools.kanban_tools`` passes ``expected_run_id=$HERMES_KANBAN_RUN_ID``, so
    this is the exact call a reclaimed worker would still make."""
    with kbc.connect() as conn:
        root, child = _build(conn)
        kb.claim_task(conn, child)
        stale_run_id = kb.get_task(conn, child).current_run_id
        assert stale_run_id is not None

        kbd._record_task_failure(
            conn, child, error="boom", outcome="crashed",
            failure_limit=1, release_claim=True, end_run=True,
        )
        assert kb.get_task(conn, child).status == "blocked"

        completed = kb.complete_task(
            conn, child, result="I finished after all", expected_run_id=stale_run_id,
        )
        assert completed is False, "an invalidated run must not complete the task"
        assert kb.get_task(conn, child).status == "blocked"
        assert kb.get_task(conn, root).status == "todo"
