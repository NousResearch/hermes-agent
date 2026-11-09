"""Tests: a reclaimed worker cannot write a terminal state to its old card.

Issue: the dispatcher reclaimed a task from a worker that was still running
and did not terminate it (``reclaim`` events recorded ``terminated: false``).
The surviving worker kept executing and could still comment on, block, and
complete the card it no longer owned. On t_f16a7cec that produced seven
concurrent audits and a published verdict that was already false.

Two independent defects, one per half of this module:

1. ``reclaim_task`` (the ``manual: true`` reclaim path behind the dashboard
   button and ``hermes kanban reclaim``) released the claim unconditionally.
   Its two siblings, ``release_stale_claims`` and the heartbeat-stale
   backstop, already refuse to release a claim whose host-local worker
   survived termination, because releasing it spawns a duplicate alongside
   the worker that is still running. ``reclaim_task`` skipped that check.

2. Nothing on the terminal write path rejected a worker whose claim had been
   revoked. ``complete_task`` / ``block_task`` enforced ``expected_run_id``
   only when the caller volunteered one, and a caller that omitted it could
   drive the card to a terminal state while another worker held the claim.
"""

from __future__ import annotations

import subprocess
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
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    with kb.connect() as c:
        yield c


@pytest.fixture
def live_worker():
    """Spawn a real, long-lived process and register it as the worker pid."""
    procs: list[subprocess.Popen] = []

    def _spawn() -> int:
        proc = subprocess.Popen(["sleep", "120"])
        procs.append(proc)
        return proc.pid

    yield _spawn
    for proc in procs:
        proc.terminate()
        proc.wait()


def _unkillable(_pid, _sig):
    """A kill that never lands.

    Reproduces the observed field condition: ``_terminate_reclaimed_worker``
    recorded ``termination_attempted: true, terminated: false, sigkill: false``
    on every reclaim of t_f16a7cec, which is the EPERM/ESRCH-shaped ``OSError``
    branch: the signal is attempted and the worker keeps running.
    """
    raise PermissionError(1, "Operation not permitted")


def _latest_event(conn, task_id, kind):
    return conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = ? "
        "ORDER BY id DESC LIMIT 1",
        (task_id, kind),
    ).fetchone()


# ---------------------------------------------------------------------------
# 1. reclaim_task must not release a claim whose worker survived termination
# ---------------------------------------------------------------------------


def test_reclaim_task_defers_when_worker_survives_termination(conn, live_worker):
    """The duplication source: releasing a claim beside a live worker."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="survivor", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:A")
    pid = live_worker()
    kb._set_worker_pid(conn, tid, pid)

    ok = kb.reclaim_task(conn, tid, reason="operator", signal_fn=_unkillable)

    assert ok is False, "reclaim reported success while the worker was alive"
    task = kb.get_task(conn, tid)
    assert task.status == "running", "claim released beside a live worker"
    assert task.claim_lock == f"{host}:A"
    assert task.worker_pid == pid
    assert task.current_run_id is not None, "run closed under a live worker"

    deferred = _latest_event(conn, tid, "reclaim_deferred")
    assert deferred is not None, "no reclaim_deferred event for the operator"
    assert "manual_reclaim_worker_alive" in deferred["payload"]
    assert _latest_event(conn, tid, "reclaimed") is None


def test_reclaim_task_force_releases_a_surviving_worker(conn, live_worker):
    """The operator escape hatch stays open, and says so in the event log."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="forced", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:A")
    kb._set_worker_pid(conn, tid, live_worker())

    ok = kb.reclaim_task(
        conn, tid, reason="wedged", force=True, signal_fn=_unkillable,
    )

    assert ok is True
    assert kb.get_task(conn, tid).status == "ready"
    event = _latest_event(conn, tid, "reclaimed")
    assert event is not None
    assert '"forced": true' in event["payload"]


def test_reclaim_task_still_reclaims_a_dead_worker(conn):
    """The ordinary path must keep working: a dead worker reclaims cleanly."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="dead", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:A")
    dead = subprocess.Popen(["true"])
    dead.wait()
    kb._set_worker_pid(conn, tid, dead.pid)

    assert kb.reclaim_task(conn, tid, reason="crashed") is True
    assert kb.get_task(conn, tid).status == "ready"


def test_reclaim_task_reclaims_when_worker_is_not_ours(conn, live_worker):
    """A claim held by another host is not ours to defer on, so release it.

    ``_worker_survived_termination`` deliberately ignores non-host-local
    claims: we cannot signal that worker, so holding the claim forever would
    strand the card.
    """
    tid = kb.create_task(conn, title="remote", assignee="w")
    kb.claim_task(conn, tid, claimer="other-host:9999")
    kb._set_worker_pid(conn, tid, live_worker())

    assert kb.reclaim_task(conn, tid, reason="remote") is True
    assert kb.get_task(conn, tid).status == "ready"


# ---------------------------------------------------------------------------
# 2. A revoked worker cannot write a terminal state
# ---------------------------------------------------------------------------


def _reclaim_and_rehand(conn, tid, host, monkeypatch=None):
    """Reclaim worker A's card and hand it to worker B. Returns A's run id.

    When ``monkeypatch`` is given, the process env is also made to look like
    worker A's: the terminal-write guard is scoped to kanban workers, so a
    bare library call is (correctly) not treated as a revoked worker.
    """
    run_a = kb.get_task(conn, tid).current_run_id
    kb.reclaim_task(conn, tid, reason="test", force=True)
    kb.claim_task(conn, tid, claimer=f"{host}:B")
    if monkeypatch is not None:
        monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_a))
        monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", f"{host}:A")
    return run_a


def test_revoked_worker_cannot_complete_with_its_own_run_id(conn, monkeypatch):
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="t", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:A")
    run_a = _reclaim_and_rehand(conn, tid, host, monkeypatch)

    assert kb.complete_task(conn, tid, summary="stale", expected_run_id=run_a) is False
    assert kb.get_task(conn, tid).status == "running"


def test_revoked_worker_cannot_complete_without_a_run_id(conn, monkeypatch):
    """The gap: an unguarded terminal write over another worker's live claim."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="t", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:A")
    _reclaim_and_rehand(conn, tid, host, monkeypatch)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)

    assert kb.complete_task(conn, tid, summary="stale") is False
    task = kb.get_task(conn, tid)
    assert task.status == "running", "revoked worker completed a live card"
    assert task.claim_lock == f"{host}:B"


def test_revoked_worker_cannot_block_without_a_run_id(conn, monkeypatch):
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="t", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:A")
    _reclaim_and_rehand(conn, tid, host, monkeypatch)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)

    assert kb.block_task(conn, tid, reason="stale", kind="needs_input") is False
    assert kb.get_task(conn, tid).status == "running"


def test_force_still_lets_a_human_complete_a_live_card(conn, monkeypatch):
    """A human closing a running card keeps working, via an explicit force."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="t", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:A")
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "999999")

    assert kb.complete_task(conn, tid, summary="human", force=True) is True
    assert kb.get_task(conn, tid).status == "done"


def test_force_still_lets_a_human_block_a_live_card(conn, monkeypatch):
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="t", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:A")
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "999999")

    assert kb.block_task(conn, tid, reason="human", force=True) is True
    assert kb.get_task(conn, tid).status == "blocked"


def test_non_worker_caller_is_not_gated(conn):
    """A plain library/CLI caller with no worker env keeps its old behaviour.

    The guard must not turn every direct ``complete_task`` on a running task
    into a force-only operation; only dispatcher-spawned workers are scoped.
    """
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="t", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:A")

    assert kb.complete_task(conn, tid, summary="library call") is True
    assert kb.get_task(conn, tid).status == "done"


def test_unclaimed_card_still_completes_without_force(conn):
    """No claim, no guard: the plain CLI path on a ready card is untouched."""
    tid = kb.create_task(conn, title="t", assignee="w")
    assert kb.complete_task(conn, tid, summary="cli") is True
    assert kb.get_task(conn, tid).status == "done"


def test_owning_worker_completes_normally(conn):
    """The guard must not break the worker it was written to protect."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="t", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:A")
    run_a = kb.get_task(conn, tid).current_run_id

    assert kb.complete_task(conn, tid, summary="ok", expected_run_id=run_a) is True
    assert kb.get_task(conn, tid).status == "done"


def test_reviewer_can_still_approve_a_review_card(conn):
    """A card parked in review has no live claim, so approval needs no force."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="t", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:A")
    run_a = kb.get_task(conn, tid).current_run_id
    assert kb.request_review(conn, tid, summary="impl", expected_run_id=run_a)
    assert kb.get_task(conn, tid).status == "review"

    assert kb.complete_task(conn, tid, summary="approved") is True
    assert kb.get_task(conn, tid).status == "done"


# ---------------------------------------------------------------------------
# 3. ownership evidence: only a contradiction refuses a worker
# ---------------------------------------------------------------------------


def test_owner_without_run_id_env_still_completes(conn, monkeypatch):
    """The guard must not strand a worker spawned without a run-id env var.

    ``_default_spawn`` only exports ``HERMES_KANBAN_RUN_ID`` when the task
    had a ``current_run_id`` at spawn time. Such a worker still carries
    ``HERMES_KANBAN_CLAIM_LOCK``, which matches, so its handoff goes through.
    """
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="t", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:A")

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", f"{host}:A")

    assert kb.complete_task(conn, tid, summary="ok") is True
    assert kb.get_task(conn, tid).status == "done"


def test_worker_without_any_ownership_evidence_is_allowed(conn, monkeypatch):
    """Absence of evidence must not be treated as evidence of revocation.

    A worker carrying only ``HERMES_KANBAN_TASK`` (no run id, no claim lock)
    cannot be distinguished from the legitimate owner. Refusing it would
    strand its own handoff, which is worse than the duplicate write the
    guard exists to stop, so the call goes through unchanged.
    """
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="t", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:A")

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_CLAIM_LOCK", raising=False)

    assert kb.complete_task(conn, tid, summary="own handoff") is True
    assert kb.get_task(conn, tid).status == "done"


def test_revoked_worker_detected_by_claim_lock_alone(conn, monkeypatch):
    """Claim lock alone is enough evidence to refuse a revoked worker."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="t", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:A")
    kb.reclaim_task(conn, tid, reason="test", force=True)
    kb.claim_task(conn, tid, claimer=f"{host}:B")

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", f"{host}:A")

    assert kb.complete_task(conn, tid, summary="stale") is False
    assert kb.get_task(conn, tid).status == "running"


def test_worker_scoped_to_another_card_is_not_gated(conn, monkeypatch):
    """Only the worker's OWN card is gated; foreign ids fall through.

    Cross-task mutation is already refused upstream by
    ``_enforce_worker_task_ownership`` in tools/kanban_tools.py; this guard
    deliberately does not duplicate that policy.
    """
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="t", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:A")

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_someoneelse")
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", f"{host}:ZZZ")

    assert kb.complete_task(conn, tid, summary="other worker") is True
