"""Regressions for #140342: a status flip on a running card must not let the
dispatcher spawn a worker beside the still-running one.

``_set_status_direct`` (dashboard drag/bulk) releases the claim and closes the
run in one txn, then kills the worker AFTER the commit. A dispatcher tick in
that window used to re-claim the card freely — three concurrent workers were
observed on one card. ``claim_task``/``claim_review_task`` now hold the claim
off while the prior ended run's host-local worker is still alive (the closed
run row keeps ``worker_pid``/``worker_started_at``/``claim_lock`` for exactly
this kind of liveness evidence, like ``reap_terminal_workers``).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def conn(tmp_path: Path):
    db = kbc.connect(tmp_path / "kanban.db")
    try:
        yield db
    finally:
        db.close()


def _flip_running_to_ready(conn, task_id: str) -> None:
    """DB-equivalent of the dashboard running->ready drag (``_set_status_direct``
    txn half): status to ready, claim cleared, run closed as reclaimed — the
    worker kill that follows the commit is what the claim guard must not
    pre-assume."""
    with kb.write_txn(conn):
        assert kb._end_run(
            conn, task_id, outcome="reclaimed", status="reclaimed",
            summary="status changed to ready (dashboard/direct)",
        )
        conn.execute(
            "UPDATE tasks SET status = 'ready', claim_lock = NULL, claim_expires = NULL, worker_pid = NULL, worker_started_at = NULL WHERE id = ?",
            (task_id,),
        )


def _last_claim_rejection(conn, task_id: str) -> str | None:
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'claim_rejected' ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchall()
    return json.loads(rows[0]["payload"])["reason"] if rows else None


def test_flip_with_live_worker_blocks_reclaim(conn, monkeypatch):
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="flipped", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:A")
    worker = subprocess.Popen(["sleep", "30"])
    try:
        kbd._set_worker_pid(conn, tid, worker.pid)

        # The flip's post-commit kill did not land (foreign container, lost
        # signal): the card reads ready/unclaimed while the worker still runs.
        monkeypatch.setattr(kb, "_terminate_reclaimed_worker", lambda *a, **k: {})
        _flip_running_to_ready(conn, tid)

        assert kb.claim_task(conn, tid, claimer=f"{host}:B") is None
        row = conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()
        assert row["status"] == "ready"
        assert _last_claim_rejection(conn, tid) == "prior_worker_alive"

        # Worker gone (the flip's SIGKILL / the reaper finally landed): the very
        # next claim goes through — the card is not wedged.
        worker.terminate()
        worker.wait()
        claimed = kb.claim_task(conn, tid, claimer=f"{host}:B")
        assert claimed is not None and claimed.status == "running"
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait()


def test_flip_after_worker_died_claims_normally(conn):
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="retried", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:A")
    dead = subprocess.Popen(["true"])
    dead.wait()
    kbd._set_worker_pid(conn, tid, dead.pid)
    _flip_running_to_ready(conn, tid)

    assert kb.claim_task(conn, tid, claimer=f"{host}:B") is not None


def test_review_claim_held_while_implementer_alive(conn):
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="review handoff", assignee="w")
    kb.claim_task(conn, tid, claimer=f"{host}:A")
    worker = subprocess.Popen(["sleep", "30"])
    try:
        kbd._set_worker_pid(conn, tid, worker.pid)
        _flip_running_to_ready(conn, tid)
        # The flip's retry-status resolution routed it to review (the run had
        # come from review); the reviewer must not start beside a live worker.
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (tid,))
        conn.commit()
        assert kb.claim_review_task(conn, tid, claimer=f"{host}:R") is None
        assert _last_claim_rejection(conn, tid) == "prior_worker_alive"

        worker.terminate()
        worker.wait()
        assert kb.claim_review_task(conn, tid, claimer=f"{host}:R") is not None
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait()


def test_foreign_host_run_never_holds_the_claim(conn):
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="remote", assignee="w")
    kb.claim_task(conn, tid, claimer="otherhost:9:Z")
    worker = subprocess.Popen(["sleep", "30"])
    try:
        kbd._set_worker_pid(conn, tid, worker.pid)
        _flip_running_to_ready(conn, tid)

        # A prior worker we cannot observe (claim_lock from another host) is the
        # TTL path's business; a local claim is not held hostage to it.
        assert kb.claim_task(conn, tid, claimer=f"{host}:B") is not None
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait()
