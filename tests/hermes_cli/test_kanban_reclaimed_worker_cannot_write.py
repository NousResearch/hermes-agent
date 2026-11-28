"""Tests: a reclaimed worker cannot write a terminal state to its old card.

Issue t_1f3b0108, raised from t_f16a7cec, where the dispatcher reclaimed a
task from a worker that was still running and did not terminate it (every
``reclaimed`` event recorded ``terminated: false``). The surviving worker
kept executing and could still block and complete the card it no longer
owned: seven concurrent audits of one deliverable, and a published verdict
that was already false when posted.

Two gaps remained after the live-claim fence landed on ``complete_task`` and
``request_review``:

1. ``reclaim_task`` (the ``manual: true`` path behind the dashboard button
   and ``hermes kanban reclaim``) released the claim unconditionally. Its two
   siblings, ``reclaim_stale_tasks`` and the heartbeat-stale backstop, already
   defer when a host-local worker survives termination, because releasing the
   claim lets the dispatcher spawn a second worker beside the first.
   ``reclaim_task`` skipped that check, so each manual reclaim of a worker
   that ignored SIGTERM produced a duplicate.

2. ``block_task`` had no live-claim fence, so a revoked-but-still-running
   worker could block the successor's live run even though the same worker
   was correctly refused by ``complete_task``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


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
    with kbc.connect() as c:
        yield c


@pytest.fixture
def live_worker():
    """Spawn a real, long-lived process to stand in for a running worker."""
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

    Reproduces the observed field condition: every reclaim of t_f16a7cec
    recorded ``termination_attempted: true, terminated: false, sigkill:
    false``, which is the ``OSError`` branch. The signal is attempted and the
    worker keeps running.
    """
    raise PermissionError(1, "Operation not permitted")


def _claim_with_live_worker(conn, tid, lock, pid):
    kb.claim_task(conn, tid, claimer=lock)
    kbd._set_worker_pid(conn, tid, pid)


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
    pid = live_worker()
    _claim_with_live_worker(conn, tid, f"{host}:A", pid)

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
    _claim_with_live_worker(conn, tid, f"{host}:A", live_worker())

    ok = kb.reclaim_task(
        conn, tid, reason="wedged", force=True, signal_fn=_unkillable,
    )

    assert ok is True
    assert kb.get_task(conn, tid).status == "ready"
    event = _latest_event(conn, tid, "reclaimed")
    assert event is not None
    assert '"forced": true' in event["payload"]


def test_reclaim_task_still_reclaims_a_dead_worker(conn):
    """The ordinary path keeps working: a dead worker reclaims cleanly."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="dead", assignee="w")
    dead = subprocess.Popen(["true"])
    dead.wait()
    _claim_with_live_worker(conn, tid, f"{host}:A", dead.pid)

    assert kb.reclaim_task(conn, tid, reason="crashed") is True
    assert kb.get_task(conn, tid).status == "ready"


def test_reclaim_task_reclaims_a_claim_with_no_worker(conn):
    """A library/CLI claim that never spawned a worker reclaims normally."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="noworker", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:A")

    assert kb.reclaim_task(conn, tid, reason="no worker") is True
    assert kb.get_task(conn, tid).status == "ready"


def test_reclaim_task_reclaims_when_worker_is_not_ours(conn, live_worker):
    """A claim held by another host is not ours to defer on, so release it.

    ``_worker_survived_termination`` deliberately ignores non-host-local
    claims: we cannot signal that worker, so holding the claim forever would
    strand the card.
    """
    tid = kb.create_task(conn, title="remote", assignee="w")
    _claim_with_live_worker(conn, tid, "other-host:9999", live_worker())

    assert kb.reclaim_task(conn, tid, reason="remote") is True
    assert kb.get_task(conn, tid).status == "ready"


def test_reclaim_task_releases_an_unsignallable_worker(conn, live_worker):
    """A worker we REFUSED to signal is released, not deferred forever.

    An UNVERIFIED spawn fingerprint means killing by bare PID could hit a
    recycled number, so ``_terminate_reclaimed_worker`` declines to signal.
    Deferring on that would never clear, because the next attempt refuses
    for the same reason; the operator reclaim is the override for it.
    """
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="unverified", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:A")
    pid = live_worker()
    kbd._set_worker_pid(conn, tid, pid)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET worker_started_at = ? WHERE id = ?",
            (kbd.UNVERIFIED_WORKER_FINGERPRINT, tid),
        )

    killed: list = []
    ok = kb.reclaim_task(
        conn, tid, reason="operator", signal_fn=lambda p, s: killed.append((p, s)),
    )

    assert ok is True, "an unsignallable worker must not strand the card"
    assert killed == [], "signalled a PID we could not verify"
    assert kb.get_task(conn, tid).status == "ready"


# ---------------------------------------------------------------------------
# 2. block_task gets the live-claim fence complete_task already had
# ---------------------------------------------------------------------------


def test_block_task_refuses_a_live_claim_without_ownership(conn, live_worker):
    """A revoked worker must not block the run its successor is executing."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="t", assignee="w")
    _claim_with_live_worker(conn, tid, f"{host}:A", live_worker())

    with pytest.raises(kb.LiveClaimError):
        kb.block_task(conn, tid, reason="stale", kind="needs_input")
    assert kb.get_task(conn, tid).status == "running"


def test_block_task_allows_the_owning_run(conn, live_worker):
    """The guard must not break the worker it was written to protect."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="t", assignee="w")
    _claim_with_live_worker(conn, tid, f"{host}:A", live_worker())
    run_id = kb.get_task(conn, tid).current_run_id

    assert kb.block_task(
        conn, tid, reason="needs a decision", kind="needs_input",
        expected_run_id=run_id,
    ) is True
    assert kb.get_task(conn, tid).status == "blocked"


def test_block_task_force_overrides_a_live_claim(conn, live_worker):
    """An operator can still block a running card explicitly."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="t", assignee="w")
    _claim_with_live_worker(conn, tid, f"{host}:A", live_worker())

    assert kb.block_task(conn, tid, reason="human", force=True) is True
    assert kb.get_task(conn, tid).status == "blocked"


def test_block_task_unaffected_without_a_live_worker(conn):
    """A claim with no live worker has no run to protect, so no force needed.

    This is the ordinary CLI/library path and must behave exactly as before.
    """
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="t", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:A")

    assert kb.block_task(conn, tid, reason="cli block") is True
    assert kb.get_task(conn, tid).status == "blocked"


def test_block_task_refuses_a_superseded_run_id(conn, live_worker):
    """A revoked worker passing its OWN (stale) run id is still refused.

    Regression cover for the CAS on ``current_run_id``, which already held
    before this change: the setup deliberately uses a claim with no worker
    process, so it needs no ``force`` and runs unchanged against main.
    """
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="t", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:A")
    stale_run = kb.get_task(conn, tid).current_run_id
    # No worker pid on A's claim, so this reclaim has nothing to terminate.
    kb.reclaim_task(conn, tid, reason="test")
    _claim_with_live_worker(conn, tid, f"{host}:B", live_worker())

    assert kb.block_task(
        conn, tid, reason="stale", kind="needs_input", expected_run_id=stale_run,
    ) is False
    task = kb.get_task(conn, tid)
    assert task.status == "running", "revoked worker blocked a live card"
    assert task.claim_lock == f"{host}:B"


# ---------------------------------------------------------------------------
# 3. the terminal writes a revoked worker must not land (end to end)
# ---------------------------------------------------------------------------


def test_revoked_worker_cannot_complete_the_successors_run(conn, live_worker):
    """Both refusals on the completion path, with a force-free setup.

    The CAS half already held before this change; the ``LiveClaimError`` half
    is the fence. Keeping the setup force-free means this file runs against
    main unchanged, so its RED/GREEN state is about behaviour, not kwargs.
    """
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="t", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:A")
    stale_run = kb.get_task(conn, tid).current_run_id
    kb.reclaim_task(conn, tid, reason="test")
    _claim_with_live_worker(conn, tid, f"{host}:B", live_worker())

    # With its own stale run id: refused by the CAS on current_run_id.
    assert kb.complete_task(
        conn, tid, summary="stale", expected_run_id=stale_run,
    ) is False
    # With no run id at all: refused by the live-claim fence.
    with pytest.raises(kb.LiveClaimError):
        kb.complete_task(conn, tid, summary="stale")

    task = kb.get_task(conn, tid)
    assert task.status == "running"
    assert task.claim_lock == f"{host}:B"


def test_owning_worker_completes_normally(conn, live_worker):
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="t", assignee="w")
    _claim_with_live_worker(conn, tid, f"{host}:A", live_worker())
    run_id = kb.get_task(conn, tid).current_run_id

    assert kb.complete_task(conn, tid, summary="ok", expected_run_id=run_id) is True
    assert kb.get_task(conn, tid).status == "done"


def test_reviewer_can_still_approve_a_review_card(conn, live_worker):
    """A card parked in review has no live claim, so approval needs no force."""
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="t", assignee="w")
    _claim_with_live_worker(conn, tid, f"{host}:A", live_worker())
    run_id = kb.get_task(conn, tid).current_run_id
    assert kb.request_review(conn, tid, summary="impl", expected_run_id=run_id)
    assert kb.get_task(conn, tid).status == "review"

    assert kb.complete_task(conn, tid, summary="approved") is True
    assert kb.get_task(conn, tid).status == "done"
