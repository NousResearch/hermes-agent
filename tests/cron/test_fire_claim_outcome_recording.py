"""What a fire-claim ownership loss records — and what it must not.

Two gateways must never double-run a job, so a run that loses its durable fire claim is
interrupted and recorded as interrupted. What went wrong (Sep 2026: 28 executions across 14
jobs, every one with a session ending ``cron_complete``) is that the *signal* was treated as a
*verdict*: the in-memory ``lost_ownership`` event also fires when the renewal cannot be confirmed
(still-held fence, transient store error past the grace window), so runs that had already reached
terminal completion and shipped their output were stamped
``Interrupted by shutdown before terminal completion.`` — the failure streak and the streak nudge
then accumulated on jobs that never failed.

Contract pinned here:
  * a run that reached terminal completion and delivered is recorded from its own outcome;
  * a run whose side-effect fence was actually refused is still interrupted and still recorded —
    with a message naming the fire claim, never a shutdown that never happened.
"""

from __future__ import annotations

import contextlib
import threading

import pytest


@pytest.fixture()
def fire_store(tmp_path, monkeypatch):
    """Real jobs.json + real execution ledger in temp dirs (no mocks for the bookkeeping)."""
    import cron.executions as executions
    import cron.jobs as jobs

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    with jobs.use_cron_store(tmp_path):
        job = jobs.create_job(
            prompt="audit the world", schedule="every 5m", name="audit", deliver="telegram")
        assert jobs.claim_job_for_fire(job["id"]) is True
        yield jobs, executions, job["id"]


def _latest_execution(executions, job_id: str) -> dict:
    rows = executions.list_executions(job_id=job_id)
    assert rows, "no execution row was written"
    return rows[0]


@contextlib.contextmanager
def _scoped():
    from unittest.mock import patch

    with patch("agent.secret_scope.set_secret_scope", return_value=None), \
         patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
         patch("agent.secret_scope.reset_secret_scope"):
        yield


def test_delivered_run_is_recorded_from_its_own_outcome(fire_store, monkeypatch):
    """A renewal that cannot be confirmed *during* a long delivery must not turn an already
    delivered run into a recorded failure (the Sep-2026 shape).

    The store still names this run as the claim owner — renewal uncertainty, not a takeover.
    """
    import cron.scheduler as scheduler

    jobs, executions, job_id = fire_store
    delivery_in_flight = threading.Event()
    delivery_done = threading.Event()
    renewal_failed = threading.Event()
    delivered: list = []

    def _run_job(job, **_kwargs):
        return True, "saved output", "the audit finished", None

    def _deliver(job, content, **_kwargs):
        delivered.append(content)
        delivery_in_flight.set()
        assert renewal_failed.wait(timeout=5), "heartbeat never failed during delivery"
        delivery_done.set()
        return None

    def _heartbeat(_job_id, *, expected_owner):
        # Unconfirmable only while the send is in flight; the claim is still ours after.
        if delivery_in_flight.is_set() and not delivery_done.is_set():
            renewal_failed.set()
            raise OSError("store unavailable")
        return True

    monkeypatch.setattr(scheduler, "run_job", _run_job)
    monkeypatch.setattr(scheduler, "_deliver_result", _deliver)
    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", _heartbeat)
    monkeypatch.setattr(scheduler, "_resolve_delivery_targets",
                        lambda job, for_failure=False: [{"platform": "telegram", "chat_id": "-1"}])
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.02)
    monkeypatch.setattr(scheduler, "_FIRE_CLAIM_HEARTBEAT_GRACE_SECONDS", 0.02)

    with _scoped():
        assert scheduler.run_one_job(jobs.get_job(job_id)) is True

    assert delivered, "the run's output was never delivered"
    record = _latest_execution(executions, job_id)
    assert record["status"] == "completed", record
    assert record["delivery_outcome"] == "delivered", record
    assert not (record["error"] or ""), record
    stored = jobs.get_job(job_id)
    assert stored["last_status"] == "ok"
    assert not stored.get("failure_streak")


def test_refused_side_effect_fence_is_still_recorded_as_interrupted(fire_store, monkeypatch):
    """The protection stands: when the fence is actually refused, the run is interrupted, its
    result is neither saved nor delivered, and the record names the fire claim — not a shutdown."""
    import cron.scheduler as scheduler

    jobs, executions, job_id = fire_store
    delivered: list = []

    @contextlib.contextmanager
    def refused_fence(*_args, **_kwargs):
        yield False

    monkeypatch.setattr(
        scheduler, "run_job",
        lambda job, **_kw: (True, "saved output", "the audit finished", None))
    monkeypatch.setattr(scheduler, "_deliver_result",
                        lambda job, content, **_kw: delivered.append(content))
    monkeypatch.setattr(scheduler, "fire_claim_fence", refused_fence)

    with _scoped():
        assert scheduler.run_one_job(jobs.get_job(job_id)) is True

    assert delivered == [], "a refused fence must not deliver"
    record = _latest_execution(executions, job_id)
    assert record["status"] == "failed", record
    error = record["error"] or ""
    assert "fire claim ownership lost" in error.lower(), error
    assert "shutdown" not in error.lower(), error
    stored = jobs.get_job(job_id)
    assert stored["last_status"] != "ok"
    assert stored["failure_streak"] == 1
