"""Tests: ``reclaim_task`` releases a CLAIM, it never promotes a blocked card.

Regression for the dispatcher blocked-status bug: a card parked in ``blocked``
(via ``--initial-status blocked``, a worker's ``review-required`` handoff, or the
circuit breaker) that still carried stale claim residue could be laundered into
``ready`` by an operator reclaim. ``reclaim_task`` unconditionally wrote
``status='ready'`` while accepting ``blocked`` in its WHERE clause, so the next
dispatcher tick claimed the row and spawned a worker on a card that was supposed
to be held — defeating the block entirely and burning worker budget.

The reclaim also emitted only a ``reclaimed`` event, never ``unblocked``, so
``_has_sticky_block`` still reported True: the row ended up ``status='ready'``
AND sticky-blocked at the same time, an incoherent state no other writer can
produce.

``unblock_task`` / ``promote_task`` remain the only ways out of ``blocked``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home, monkeypatch):
    # The dispatcher refuses to spawn for an assignee that isn't a real Hermes
    # profile. Without this the "did not spawn" assertions below would pass
    # trivially (skipped_nonspawnable) instead of proving the status guard.
    import hermes_cli.profiles as _profiles

    monkeypatch.setattr(_profiles, "profile_exists", lambda name: True)
    with kb.connect() as c:
        yield c


def _status(conn, tid):
    return conn.execute(
        "SELECT status FROM tasks WHERE id = ?", (tid,)
    ).fetchone()["status"]


def _stamp_stale_claim(conn, tid, lock="ghost-host:4242"):
    """Leave a dead worker's claim residue on the row without touching status."""
    conn.execute(
        "UPDATE tasks SET claim_lock = ?, worker_pid = ? WHERE id = ?",
        (lock, 999999, tid),
    )
    conn.commit()


def test_reclaim_does_not_promote_initial_status_blocked_card(conn):
    """A card created ``--initial-status blocked`` stays blocked across a reclaim."""
    tid = kb.create_task(
        conn, title="held card", assignee="w", initial_status="blocked",
    )
    assert _status(conn, tid) == "blocked"
    _stamp_stale_claim(conn, tid)

    assert kb.reclaim_task(conn, tid) is True

    # The claim residue is gone...
    row = conn.execute(
        "SELECT status, claim_lock, worker_pid FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["claim_lock"] is None
    assert row["worker_pid"] is None
    # ...but the block held. Pre-fix this was 'ready'.
    assert row["status"] == "blocked"


def test_dispatcher_does_not_spawn_a_reclaimed_blocked_card(conn):
    """End-to-end: reclaim a blocked card, then run a real dispatcher pass."""
    tid = kb.create_task(
        conn, title="do not dispatch me", assignee="w", initial_status="blocked",
    )
    _stamp_stale_claim(conn, tid)
    kb.reclaim_task(conn, tid)

    spawned: list[str] = []

    def spawn_fn(task, workspace, board=None, **kwargs):
        spawned.append(task.id)
        return 4242

    kb.dispatch_once(conn, spawn_fn=spawn_fn)

    # Pre-fix: the reclaim left it 'ready', the dispatcher claimed it and
    # spawned a worker on a card the operator had explicitly held.
    assert spawned == []
    assert _status(conn, tid) == "blocked"


def test_reclaim_leaves_blocked_card_coherent_with_sticky_block(conn):
    """A reclaimed blocked card must not be 'ready' while still sticky-blocked."""
    tid = kb.create_task(
        conn, title="sticky", assignee="w", initial_status="blocked",
    )
    _stamp_stale_claim(conn, tid)
    kb.reclaim_task(conn, tid)

    # reclaim emits 'reclaimed', never 'unblocked' — so the task is still
    # sticky-blocked. Status must agree with that, or recompute_ready and the
    # dispatcher disagree about whether the card is available.
    assert kb._has_sticky_block(conn, tid) is True
    assert _status(conn, tid) == "blocked"


def test_reclaim_preserves_worker_review_required_block(conn):
    """The ``review-required`` handoff survives a reclaim (real incident shape)."""
    tid = kb.create_task(conn, title="review handoff", assignee="w")
    kb.claim_task(conn, tid)
    assert kb.block_task(
        conn, tid, reason="review-required: needs a human look",
    ) is True
    assert _status(conn, tid) == "blocked"

    _stamp_stale_claim(conn, tid)
    kb.reclaim_task(conn, tid)

    assert _status(conn, tid) == "blocked"


def test_reclaim_records_the_preserved_status_in_board_history(conn):
    """Operators can see why a reclaimed blocked card did not go ready."""
    tid = kb.create_task(
        conn, title="held", assignee="w", initial_status="blocked",
    )
    _stamp_stale_claim(conn, tid)
    kb.reclaim_task(conn, tid)

    payloads = [
        r["payload"] for r in conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'reclaimed'",
            (tid,),
        )
    ]
    assert payloads, "reclaim should append a 'reclaimed' event"
    assert any("status_preserved" in (p or "") for p in payloads)


def test_reclaim_still_returns_a_running_task_to_ready(conn):
    """The primary reclaim contract is unchanged: running -> ready."""
    tid = kb.create_task(conn, title="live worker", assignee="w")
    kb.claim_task(conn, tid)
    assert _status(conn, tid) == "running"

    assert kb.reclaim_task(conn, tid) is True
    assert _status(conn, tid) == "ready"

    spawned: list[str] = []

    def spawn_fn(task, workspace, board=None, **kwargs):
        spawned.append(task.id)
        return 4242

    kb.dispatch_once(conn, spawn_fn=spawn_fn)
    # A genuinely reclaimed running task MUST become dispatchable again.
    assert spawned == [tid]


def test_reclaim_does_not_promote_a_triage_card(conn):
    """Same bug class: ``triage`` is a pre-dispatch parking column too.

    ``reclaim_task``'s WHERE clause did not accept ``triage``, so a triage row
    carrying stale claim residue could not be reclaimed at ALL (returned
    False). It must now be reclaimable *without* being promoted -- the
    specifier/decomposer owns the triage -> todo transition.
    """
    tid = kb.create_task(conn, title="unspecified idea", assignee="w", triage=True)
    assert _status(conn, tid) == "triage"
    _stamp_stale_claim(conn, tid)

    assert kb.reclaim_task(conn, tid) is True
    row = conn.execute(
        "SELECT status, claim_lock, worker_pid FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["claim_lock"] is None and row["worker_pid"] is None
    assert row["status"] == "triage"


def test_no_claim_release_path_promotes_a_parked_row(conn):
    """Whole-bug-class invariant over the module source.

    A claim-release UPDATE that writes a LITERAL ``status = 'ready'`` while
    accepting a parking status (``blocked`` / ``triage`` / ``scheduled``) in its
    WHERE clause is the exact shape of this bug: it launders a held card into
    the dispatcher's queue. The sibling paths (``release_stale_claims``,
    ``detect_crashed_workers``, ``enforce_max_runtime``) are all scoped to
    ``status = 'running'``, so ``reclaim_task`` was the only offender -- pin
    that, so a future writer widening a sibling's WHERE clause has to confront
    this test instead of silently reintroducing the bug on another path.
    """
    import inspect
    import re

    src = inspect.getsource(kb)
    parked = ("blocked", "triage", "scheduled")
    offenders = []
    # Each claim-clearing UPDATE, from the SET through the end of the SQL string
    # group (the call's closing paren).
    for m in re.finditer(r'"UPDATE tasks SET status = ', src):
        chunk = src[m.start():m.start() + 700]
        chunk = chunk[:chunk.find(")\n") + 1] or chunk
        if "claim_lock = NULL" not in chunk:
            continue
        writes_ready_literal = "status = 'ready'" in chunk
        accepts_parked = any(f"'{p}'" in chunk.split("WHERE", 1)[-1] for p in parked)
        if writes_ready_literal and accepts_parked:
            offenders.append(chunk.splitlines()[0].strip())
    assert not offenders, (
        "claim-release UPDATE promotes a parked row to a literal 'ready': "
        + repr(offenders)
    )


def test_cli_reclaim_tells_the_operator_the_card_is_still_held(conn, capsys):
    """The CLI must not print a bare 'Reclaimed <id>' for a held card.

    A bare success line reads as 'it's queued again', so the operator waits for
    a worker that will never come. Point at ``unblock`` instead.
    """
    import argparse

    from hermes_cli import kanban as kcli

    tid = kb.create_task(
        conn, title="held", assignee="w", initial_status="blocked",
    )
    _stamp_stale_claim(conn, tid)

    rc = kcli._cmd_reclaim(argparse.Namespace(task_id=tid, reason=None))
    out = capsys.readouterr().out
    assert rc == 0
    assert "blocked" in out and "unblock" in out, out
