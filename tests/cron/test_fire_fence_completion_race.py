"""Regression tests for the cron fire-fence ownership/lock-timeout conflation.

Bug (t_0ce0b31e): a cron fire whose per-job fence (``_fire_job_lock``) was held
— e.g. by its own slow delivery — for longer than the fence lock timeout could
be misrecorded as failed *after* the agent turn and delivery both succeeded.
Before the fix, ``heartbeat_fire_claim`` returned False both when the claim was
genuinely taken over AND when the fence lock merely could not be acquired
within its timeout (``_fire_job_lock`` fails closed with ``yield False``). The
heartbeat loop treated both identically, set ``lost_ownership``, and
``run_one_job``'s post-delivery bookkeeping then wrote ``mark_job_run(False,
"Interrupted by shutdown before terminal completion.")`` over a delivered,
successful run — flipping ``last_status`` to ``error`` and incrementing
``failure_streak`` in jobs.json.

Contract under test:

1. An unconfirmable heartbeat (fence lock unavailable, claim provably still
   ours in the store) must NOT be reported as ownership loss. Ownership cannot
   have changed while the fence is busy: every takeover/revocation path
   serializes through that same fence.
2. A store I/O error inside the heartbeat must propagate as uncertainty, not
   surface as a loss verdict.
3. A successful, delivered run whose heartbeat hit a contention blip keeps its
   success bookkeeping (last_status "ok", failure_streak 0, ledger completed).
4. A genuine claim takeover still fails closed: the stale result is discarded
   and never recorded as success.
"""

import contextlib
import threading
import time
from unittest.mock import patch

import pytest

import cron.scheduler as scheduler


@contextlib.contextmanager
def _owned_fence(*_args, **_kwargs):
    """Stand-in fence that always grants ownership (for mocked pipelines)."""
    yield True


@pytest.fixture
def temp_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so jobs.json doesn't touch the real store."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


def _make_claimed_job(jobs, name):
    job = jobs.create_job(prompt="x", schedule="every 5m", name=name)
    assert jobs.claim_job_for_fire(job["id"]) is True
    stored = jobs.get_job(job["id"])
    assert stored is not None and stored.get("fire_claim")
    owner = stored["fire_claim"]["by"]
    return job, owner


def test_heartbeat_lock_timeout_does_not_report_ownership_loss(
    temp_home, monkeypatch
):
    """A contended fence is not an ownership loss: the claim is still ours.

    Cross-thread contention against the real fence (local RLock + flock), with
    the fence lock timeout shortened so the probe fails fast. The old code
    returned False here — the exact false 'ownership lost' verdict.
    """
    import cron.jobs as jobs

    monkeypatch.setattr(jobs, "_JOBS_LOCK_TIMEOUT_SECONDS", 0.05)

    job, owner = _make_claimed_job(jobs, "hb-contended")
    jid = job["id"]

    acquired = threading.Event()
    release = threading.Event()

    def holder():
        with jobs._fire_job_lock(jid) as got:
            assert got, "holder could not acquire the fence"
            acquired.set()
            release.wait(timeout=5)

    holder_thread = threading.Thread(target=holder, daemon=True)
    holder_thread.start()
    assert acquired.wait(timeout=5)
    try:
        verdict = {}

        def probe():
            try:
                jobs.heartbeat_fire_claim(jid, expected_owner=owner)
                verdict["ok"] = True
            except jobs.FireFenceUnavailableError:
                # The fixed contract: contention raises uncertainty instead
                # of returning a loss verdict.
                verdict["ok"] = "raised"

        probe_thread = threading.Thread(target=probe, daemon=True)
        probe_thread.start()
        probe_thread.join(timeout=5)
        assert not probe_thread.is_alive(), "heartbeat probe deadlocked"
        assert verdict["ok"] == "raised", (
            "fence contention must raise uncertainty, not report loss "
            f"(got {verdict.get('ok')!r})"
        )
    finally:
        release.set()
        holder_thread.join(timeout=5)

    # The claim was never mutated: still owned by the original owner.
    assert jobs.get_job(jid)["fire_claim"]["by"] == owner


def test_heartbeat_store_error_is_not_swallowed_as_loss(temp_home, monkeypatch):
    """A store I/O failure inside the heartbeat must propagate as uncertainty.

    Swallowing it into a False (loss) verdict would let a transient jobs.json
    error discard a healthy run's result; callers treat a raised error with
    bounded grace instead of an immediate loss.
    """
    import cron.jobs as jobs

    job, owner = _make_claimed_job(jobs, "hb-io-error")

    def _boom(*_a, **_kw):
        raise OSError("jobs.json temporarily unwritable")

    monkeypatch.setattr(jobs, "save_jobs", _boom)

    try:
        jobs.heartbeat_fire_claim(job["id"], expected_owner=owner)
    except OSError:
        pass
    else:
        raise AssertionError("expected the store error to propagate")


def test_successful_long_delivery_is_not_demoted_after_heartbeat_grace(
    temp_home, monkeypatch
):
    """A delivery holding its own fence past heartbeat grace stays successful."""
    import cron.jobs as jobs
    from cron.executions import latest_execution

    monkeypatch.setattr(jobs, "_JOBS_LOCK_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.005)
    monkeypatch.setattr(scheduler, "_FIRE_CLAIM_HEARTBEAT_GRACE_SECONDS", 0.02)

    job, owner = _make_claimed_job(jobs, "long-delivery")
    jid = job["id"]

    delivered = []

    def slow_delivery(_job, content, **_kw):
        # _deliver_result runs under the real side-effect fence. Keep it held
        # long enough for repeated heartbeat lock timeouts to exceed grace.
        time.sleep(0.12)
        delivered.append(content)
        return None

    monkeypatch.setattr(scheduler, "claim_dispatch", lambda *_a, **_kw: True)
    monkeypatch.setattr(scheduler, "mark_execution_running", lambda *_a: None)
    monkeypatch.setattr(
        scheduler,
        "run_job",
        lambda *_a, **_kw: (True, "out", "final response", None),
    )
    monkeypatch.setattr(scheduler, "save_job_output", lambda *_a: "out.md")
    monkeypatch.setattr(scheduler, "_deliver_result", slow_delivery)

    with patch("agent.secret_scope.set_secret_scope", return_value=None), \
         patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
         patch("agent.secret_scope.reset_secret_scope"):
        assert scheduler.run_one_job(
            {
                "id": jid,
                "name": "long-delivery",
                "deliver": "all",
                "fire_claim": {"at": "2026-09-01T00:00:00+00:00", "by": owner},
            }
        ) is True

    assert delivered == ["final response"]
    final = jobs.get_job(jid)
    assert final["last_status"] == "ok", final
    assert not final.get("last_error"), final
    assert not final.get("failure_streak"), final
    row = latest_execution(jid)
    assert row is not None and row["status"] == "completed", row


def test_shutdown_during_long_delivery_still_fails_closed(temp_home, monkeypatch):
    """A real transport cancellation is not mistaken for fence self-contention."""
    import cron.jobs as jobs
    from cron.executions import latest_execution

    monkeypatch.setattr(jobs, "_JOBS_LOCK_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.005)
    monkeypatch.setattr(scheduler, "_FIRE_CLAIM_HEARTBEAT_GRACE_SECONDS", 0.02)

    job, owner = _make_claimed_job(jobs, "cancelled-long-delivery")
    jid = job["id"]
    external_cancel = threading.Event()

    def cancelled_delivery(*_args, **_kwargs):
        time.sleep(0.12)
        external_cancel.set()
        return None

    monkeypatch.setattr(scheduler, "claim_dispatch", lambda *_a, **_kw: True)
    monkeypatch.setattr(scheduler, "mark_execution_running", lambda *_a: None)
    monkeypatch.setattr(
        scheduler,
        "run_job",
        lambda *_a, **_kw: (True, "out", "final response", None),
    )
    monkeypatch.setattr(scheduler, "save_job_output", lambda *_a: "out.md")
    monkeypatch.setattr(scheduler, "_deliver_result", cancelled_delivery)

    with patch("agent.secret_scope.set_secret_scope", return_value=None), \
         patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
         patch("agent.secret_scope.reset_secret_scope"):
        assert scheduler.run_one_job(
            {
                "id": jid,
                "name": "cancelled-long-delivery",
                "deliver": "all",
                "fire_claim": {"at": "2026-09-01T00:00:00+00:00", "by": owner},
            },
            cancel_event=external_cancel,
        ) is True

    final = jobs.get_job(jid)
    assert final is not None
    assert final["last_status"] == "error", final
    assert final["failure_streak"] == 1, final
    assert final["last_error"] == "Interrupted by shutdown before terminal completion."
    row = latest_execution(jid)
    assert row is not None and row["status"] == "failed", row


def test_genuine_ownership_loss_still_fails_closed(monkeypatch):
    """A real claim takeover must still discard the stale result — no fix may
    open the fail-closed path that protects at-most-once semantics."""
    finished = []

    monkeypatch.setattr(
        scheduler, "create_execution", lambda *_a, **_kw: {"id": "exec-loss"}
    )
    monkeypatch.setattr(scheduler, "claim_dispatch", lambda *_a, **_kw: True)
    monkeypatch.setattr(scheduler, "mark_execution_running", lambda *_a: None)
    monkeypatch.setattr(
        scheduler, "run_job", lambda *_a, **_kw: (True, "out", "stale", None)
    )
    monkeypatch.setattr(
        scheduler, "fire_claim_fence", _owned_fence, raising=False
    )
    monkeypatch.setattr(scheduler, "save_job_output", lambda *_a: "out.md")
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *_a, **_kw: None)

    # Initial validation confirms; the loop probe then reports a REAL loss
    # (claim taken over — heartbeat says False and stays False).
    probes = {"n": 0}

    def heartbeat(job_id, *, expected_owner):
        probes["n"] += 1
        return probes["n"] == 1

    monkeypatch.setattr(scheduler, "heartbeat_fire_claim", heartbeat)
    monkeypatch.setattr(scheduler, "_RUN_CLAIM_HEARTBEAT_SECONDS", 0.01)
    monkeypatch.setattr(scheduler, "mark_job_run", lambda *_a, **_kw: True)
    monkeypatch.setattr(
        scheduler,
        "finish_execution",
        lambda eid, **kw: finished.append((eid, kw)),
    )

    assert scheduler.run_one_job(
        {
            "id": "taken-over",
            "name": "taken-over",
            "fire_claim": {"at": "2026-09-01T00:00:00+00:00", "by": "owner-A"},
        }
    ) is True

    # No success write landed anywhere: the stale result was discarded.
    assert finished == [
        ("exec-loss", {"success": False, "error": "Fire claim ownership lost; "
         "stale result was discarded."})
    ]
