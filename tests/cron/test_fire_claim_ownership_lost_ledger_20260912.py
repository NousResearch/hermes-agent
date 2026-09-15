"""K3 case-1 regression tests (2026-09-12): fire-claim ownership lost ledger branches.

Covers the three regression designs in upstream_fix_cron_ownership_lost_20260912.md:
1. delivered-then-lost -> completed-with-warning, never the shutdown string (12/20 misrecords)
2. pre-delivery lost -> distinct pre-delivery reason, distinguishable from shutdown (K3 term 1)
3. original shutdown-string path preserved (transport-level cancel via owner-fenced write)
"""

import threading
from unittest.mock import MagicMock, patch

import pytest


def _mk_job(eid):
    # No fire_claim key -> owner=None -> fence.lost() reduces to fire_claim_lost.is_set().
    return {"id": "job-" + eid, "name": "job " + eid, "prompt": "work", "execution_id": eid}


def _patch_common(monkeypatch, recorded):
    monkeypatch.setattr("cron.scheduler.finish_execution",
                        lambda eid, *, success, error=None, **kw:
                        recorded.update(finish_success=success, finish_error=error))
    monkeypatch.setattr("cron.scheduler.mark_job_run",
                        lambda job_id, success, error=None, *a, **kw:
                        (recorded.update(success=success, error=error), True)[1])
    monkeypatch.setattr("cron.scheduler.claim_dispatch", lambda job_id: True)
    monkeypatch.setattr("cron.scheduler.mark_execution_running", lambda eid: {})


class _FireOwnershipStub:
    """Replaces scheduler._FireOwnership: lost() reads the injected event, fence is a nullcontext."""

    def __init__(self, job, fire_claim_lost):
        self.owner = None
        self._lost_ev = fire_claim_lost if fire_claim_lost is not None else threading.Event()

    def side_effect_fence(self):
        import contextlib
        return contextlib.nullcontext(True)

    def lost(self):
        return self._lost_ev.is_set()


def _run_body(monkeypatch, job, run_job_impl, *, lost, deliver=None):
    """Drive _run_one_job_body with _FireOwnership patched to the stub.

    lost=True: claim already lost when run_job returns (pre-delivery path).
    lost="after-delivery": claim lost DURING delivery (delivered-then-lost path) —
    the stub flips lost() to True only after _save_compose_deliver has run, matching
    the real fence-timeout-during-delivery sequence.
    """
    import cron.scheduler as scheduler

    lost_ev = threading.Event()

    def _fake_scd(d, fence, final_response, output, **kw):
        if lost == "after-delivery":
            lost_ev.set()  # claim expires mid-delivery
        if deliver is not None:
            deliver(d)

    monkeypatch.setattr("cron.scheduler.run_job", run_job_impl)
    monkeypatch.setattr("cron.scheduler._save_compose_deliver", _fake_scd)
    monkeypatch.setattr("cron.scheduler._FireOwnership",
                        lambda j, fcl: _FireOwnershipStub(j, fcl))

    if lost is True:
        lost_ev.set()
    with patch("agent.secret_scope.set_secret_scope", return_value=None), \
         patch("agent.secret_scope.build_profile_secret_scope", return_value=None), \
         patch("agent.secret_scope.reset_secret_scope"):
        return scheduler._run_one_job_body(
            job,
            execution_token=object(),
            fire_claim_lost=lost_ev,
        )


def test_delivered_then_ownership_lost_records_completed_with_warning(monkeypatch):
    """Delivery already succeeded + claim lost -> completed-with-warning, NEVER shutdown string.

    Real-world anchor: qc-bot 09-11 21:30 -- fence timeout 21:32:49, delivered 21:39:22,
    yet executions.db recorded failed "Interrupted by shutdown..." (the misrecord).
    """
    import cron.scheduler as scheduler
    recorded = {}
    _patch_common(monkeypatch, recorded)

    def _run_job(job, **kw):
        return True, "output", "final response", None

    def _deliver(d):
        d.delivery_attempted = True
        d.delivery_error = None

    assert _run_body(monkeypatch, _mk_job("exec-1"), _run_job, lost="after-delivery", deliver=_deliver) is True

    assert recorded["success"] is True
    assert "Interrupted by shutdown" not in (recorded["error"] or "")
    assert "delivery confirmed" in (recorded["error"] or "")
    assert recorded["finish_success"] is True


def test_pre_delivery_ownership_lost_records_distinct_reason(monkeypatch):
    """Loss BEFORE the delivery phase -> distinct pre-delivery reason (K3 term 1)."""
    import cron.scheduler as scheduler
    recorded = {}
    _patch_common(monkeypatch, recorded)

    def _run_job(job, **kw):
        return True, "output", "final response", None

    assert _run_body(monkeypatch, _mk_job("exec-2"), _run_job, lost=True) is True

    assert recorded["success"] is False
    assert "Interrupted by shutdown" not in (recorded["error"] or "")
    assert "not a shutdown" in (recorded["error"] or "")
    assert "before delivery" in (recorded["error"] or "")


def test_shutdown_string_path_preserved(monkeypatch):
    """Transport-level cancel: heartbeat still holds -> original interrupted semantics kept."""
    import cron.scheduler as scheduler
    recorded = {}
    monkeypatch.setattr("cron.scheduler.heartbeat_fire_claim",
                        lambda job_id, expected_owner=None: True)
    monkeypatch.setattr("cron.scheduler.finish_execution",
                        lambda eid, *, success, error=None, **kw:
                        recorded.update(finish_success=success))
    monkeypatch.setattr("cron.scheduler.mark_job_run",
                        lambda job_id, success, error=None, *a, **kw:
                        (recorded.update(success=success, error=error), True)[1])

    scheduler._record_fire_ownership_lost("job-x", "owner-1", "exec-3")

    assert recorded["success"] is False
    assert recorded["error"] == scheduler._OWNERSHIP_LOST_INTERRUPTED
    assert recorded["finish_success"] is False


def test_delivered_with_delivery_error_keeps_failure_bookkeeping(monkeypatch):
    """Delivery attempted but failed + claim lost -> failure record, never shutdown string."""
    import cron.scheduler as scheduler
    recorded = {}
    _patch_common(monkeypatch, recorded)
    monkeypatch.setattr("cron.scheduler.heartbeat_fire_claim",
                        lambda job_id, expected_owner=None: True)

    def _run_job(job, **kw):
        return True, "output", "final response", None

    def _deliver(d):
        d.delivery_attempted = True
        d.delivery_error = "gateway down"

    assert _run_body(monkeypatch, _mk_job("exec-4"), _run_job, lost=True, deliver=_deliver) is True

    assert recorded["success"] is False
    assert "Interrupted by shutdown" not in (recorded["error"] or "")


def test_heartbeat_interval_configurable_fail_closed_default(monkeypatch):
    """cron.heartbeat_seconds configures cadence; missing/invalid -> 60s default (fail-closed)."""
    import cron.scheduler as scheduler
    import hermes_cli.config as hc

    monkeypatch.setattr(hc, "load_config", lambda: {"cron": {"heartbeat_seconds": 120}})
    assert scheduler._cron_heartbeat_seconds() == 120.0

    monkeypatch.setattr(hc, "load_config", lambda: {"cron": {"heartbeat_seconds": -5}})
    assert scheduler._cron_heartbeat_seconds() == 60.0

    monkeypatch.setattr(hc, "load_config", lambda: {"cron": {}})
    assert scheduler._cron_heartbeat_seconds() == 60.0
