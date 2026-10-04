"""Invariant: ``complete_task`` never closes a live worker's run for a caller that
neither owns the run nor asked for an operator override (issue #111764).

A claim-less completion (a human at the CLI, an orchestrator session — anything
without ``HERMES_KANBAN_*`` env) used to be authorised by task status alone, so it
marked a ``running`` card done and ``_end_run`` closed the dispatcher worker's run
row while that worker kept executing. The guard mirrors ``request_review``'s: a
``running`` task under a live claim needs ``expected_run_id`` or ``force=True``.
"""

from __future__ import annotations

import os
from argparse import Namespace
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban as kb_cli
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as c:
        yield c


def _claimed_running_task(conn, *, live_worker: bool = True) -> tuple[str, int]:
    tid = kb.create_task(conn, title="live", assignee="coder")
    assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
    if live_worker:
        # This process stands in for the spawned worker: alive, fingerprinted.
        kbd._set_worker_pid(conn, tid, os.getpid())
    return tid, kb._current_run_id(conn, tid)


def test_claimless_complete_refuses_live_run_until_forced(conn):
    tid, run_id = _claimed_running_task(conn)

    with pytest.raises(kb.LiveClaimError):
        kb.complete_task(conn, tid, result="someone else says done")

    # Nothing moved: the worker's run is still open and it can still finish its own card.
    run = conn.execute("SELECT ended_at FROM task_runs WHERE id = ?", (run_id,)).fetchone()
    assert run["ended_at"] is None
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()["status"] == "running"
    assert kb.complete_task(conn, tid, result="worker done", expected_run_id=run_id) is True

    # Explicit operator override still closes a live run.
    tid2, run2 = _claimed_running_task(conn)
    assert kb.complete_task(conn, tid2, result="operator override", force=True) is True
    run = conn.execute("SELECT ended_at, outcome FROM task_runs WHERE id = ?", (run2,)).fetchone()
    assert run["ended_at"] is not None and run["outcome"] == "completed"


def test_claimless_complete_of_claim_without_live_worker_unchanged(conn):
    """A claim whose worker never spawned (or is gone) protects no live run: the
    library / CLI flow that claims and then completes keeps working."""
    tid, run_id = _claimed_running_task(conn, live_worker=False)
    assert kb.complete_task(conn, tid, result="done") is True
    run = conn.execute("SELECT ended_at FROM task_runs WHERE id = ?", (run_id,)).fetchone()
    assert run["ended_at"] is not None


def test_claimless_complete_of_unclaimed_card_unchanged(conn):
    """The legitimate manual flow — completing a card nobody is working on — needs no proof."""
    tid = kb.create_task(conn, title="admin", assignee="coder")
    assert kb.complete_task(conn, tid, result="done") is True
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()["status"] == "done"


def test_force_complete_unassigned_triage_parent_releases_children(conn):
    parent = kb.create_task(conn, title="unroutable", triage=True)
    child = kb.create_task(conn, title="finished work", assignee="coder", parents=(parent,))
    assert kb.get_task(conn, parent).assignee is None
    assert kb.get_task(conn, child).status == "todo"

    # Normal completion must not silently turn a triage card into a done card.
    assert kb.complete_task(conn, parent, summary="operator closes unroutable work") is False
    assert kb.get_task(conn, parent).status == "triage"
    assert kb.get_task(conn, child).status == "todo"

    assert kb.complete_task(conn, parent, summary="operator closes unroutable work", force=True)
    assert kb.get_task(conn, parent).status == "done"
    assert kb.get_task(conn, child).status == "ready"
    assert kb.complete_task(conn, child, summary="finished")
    assert kb.get_task(conn, child).status == "done"


def test_cli_triage_requires_explicit_force_and_evidence(conn, capsys):
    parent = kb.create_task(conn, title="unroutable", triage=True)
    args = Namespace(task_ids=[parent], result="operator closed unroutable work",
                     summary=None, metadata=None, force=False)
    assert kb_cli._cmd_complete(args) == 1
    assert "use --force" in capsys.readouterr().err
    assert kb.get_task(conn, parent).status == "triage"

    args.force = True
    args.result = None
    assert kb_cli._cmd_complete(args) == 1
    assert kb.get_task(conn, parent).status == "triage"
    args.result = "operator closed unroutable work"
    assert kb_cli._cmd_complete(args) == 0
    assert kb.get_task(conn, parent).status == "done"


def test_force_does_not_bypass_triage_parent_dependency(conn):
    root = kb.create_task(conn, title="first gate", triage=True)
    child = kb.create_task(conn, title="second gate", triage=True, parents=(root,))
    assert not kb.complete_task(conn, child, summary="premature", force=True)
    assert kb.get_task(conn, child).status == "triage"


def test_request_review_shares_the_live_worker_fence(conn):
    """``request_review`` keys on the same liveness as ``complete_task``: a claim
    without a live worker process is not a live claim (the human/library flow
    ``claim`` -> ``request_review`` works), a live worker's claim still is."""
    tid, _ = _claimed_running_task(conn, live_worker=False)
    assert kb.request_review(conn, tid, summary="handoff") is True
    assert kb.get_task(conn, tid).status == "review"

    tid2, run2 = _claimed_running_task(conn)
    ok, reason = kb.request_review(conn, tid2, summary="steal", with_reason=True)
    assert ok is False and "live claim" in reason
    assert kb.request_review(conn, tid2, summary="own", expected_run_id=run2) is True
