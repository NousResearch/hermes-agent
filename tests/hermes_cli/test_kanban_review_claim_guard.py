"""Invariant: ``complete_task`` never approves a ``review``-status card for a
caller that never claimed it (kanban board governance gap, 2026-11-19).

``claim_review_task`` is an atomic CAS lock (``review -> running``, same
pattern a build-run claim uses) and already stops two *dispatcher-spawned*
reviewers from racing each other. What it did NOT stop: an env-less caller —
an orchestrator session, a stray ``delegate_task`` child, a second human/CLI
invocation, literally any caller with no ``HERMES_KANBAN_TASK`` bound to this
card — completing the card directly while it still sits in ``review``. Such a
caller supplies no ``expected_run_id``, and a ``review``-status task is never
``running``, so the live-claim fence (``_claim_is_live`` / ``LiveClaimError``)
never even fires for it. That gap let a self-approval race ship a real bug
twice on the gmail_attachments chain (t_2f63744e, t_15aec0c6) before a
dispatcher-claimed reviewer's verdict ever landed.

The fix mirrors the existing live-claim fence exactly: completing FROM
``review`` now requires ``force=True`` (explicit human/operator override —
the dashboard and ``hermes kanban complete --force`` already pass it) unless
the caller already claimed the review (which flips status to ``running``,
where the pre-existing guard takes over).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


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


def _card_in_review(conn) -> str:
    """A task that has gone through the real implementer -> review handoff."""
    tid = kb.create_task(conn, title="needs review", assignee="implementer")
    assert kb.claim_task(conn, tid) is not None
    run_id = kb.get_task(conn, tid).current_run_id
    assert kb.request_review(conn, tid, summary="ready", expected_run_id=run_id, reviewer="reviewer")
    assert kb.get_task(conn, tid).status == "review"
    return tid


def test_bare_complete_of_unclaimed_review_is_refused(conn):
    """The exact gap: no claim, no force — must not silently approve."""
    tid = _card_in_review(conn)
    with pytest.raises(kb.ReviewNotClaimedError):
        kb.complete_task(conn, tid, summary="self-approved")
    # Nothing moved — the card is still sitting in review, untouched.
    assert kb.get_task(conn, tid).status == "review"


def test_second_caller_cannot_race_the_claimed_reviewer(conn):
    """The concrete race from the ticket: a dispatcher-claimed reviewer is the
    only run allowed to approve; a second, claim-less caller trying to win the
    same card must lose even though nothing else has completed it yet."""
    tid = _card_in_review(conn)

    # The dispatcher claims the review slot (the only legitimate path in) and
    # spawns a worker for it -- stand in for that spawned process the same way
    # the live-claim-guard tests do (alive, fingerprinted).
    reviewer_run = kb.claim_review_task(conn, tid)
    assert reviewer_run is not None
    assert kb.get_task(conn, tid).status == "running"
    kbd._set_worker_pid(conn, tid, os.getpid())

    # A second claim attempt on the same slot is already rejected by the
    # existing CAS lock (claim_lock IS NULL guard) -- this part worked before.
    assert kb.claim_review_task(conn, tid) is None

    # The real gap: a parallel, env-less caller with no claim and no proof of
    # run ownership tries to complete the SAME card while the reviewer is
    # live. This is covered by the pre-existing LiveClaimError fence once the
    # claim has landed -- confirm it still holds end-to-end with the new guard
    # in place (regression guard for the two fences interacting correctly).
    with pytest.raises(kb.LiveClaimError):
        kb.complete_task(conn, tid, summary="raced the reviewer")

    # Only the run that actually holds the claim may approve it.
    assert kb.complete_task(
        conn, tid, summary="approved by the real reviewer",
        expected_run_id=reviewer_run.current_run_id,
    )
    assert kb.get_task(conn, tid).status == "done"


def test_force_true_is_the_explicit_human_override(conn):
    """Dashboard 'Approve' and `hermes kanban complete --force` intentionally
    bypass the claim requirement -- that is the documented, auditable escape
    hatch, not a silent default."""
    tid = _card_in_review(conn)
    assert kb.complete_task(conn, tid, summary="manual approval", force=True)
    assert kb.get_task(conn, tid).status == "done"


def test_review_claim_then_complete_is_the_intended_flow(conn):
    """The corrected shape every real caller should use: claim first (proves
    ownership), then complete with that run's id."""
    tid = _card_in_review(conn)
    reviewer_run = kb.claim_review_task(conn, tid)
    assert reviewer_run is not None
    assert kb.complete_task(
        conn, tid, summary="approved", expected_run_id=reviewer_run.current_run_id,
    )
    assert kb.get_task(conn, tid).status == "done"
