"""#122813 follow-up: forced/split-fire count changes must also notify lifecycle
observers, or gateway_state.json goes stale exactly where recovery matters.

Two seams the first PR missed (flagged in review of #123064):

1. ``sweep_stale_inflight`` force-releases claims under ``_running_lock``
   without ``_notify_job_lifecycle()`` — the recovery path for wedged claims,
   where the file kept reading busy forever.
2. ``run_one_job`` registers/pops ``_running_fire_owners`` directly (never via
   ``try_register_running_job``/``release_running_job``), so the split-fire
   path moved the count with no notify — reachable from dashboard "Run now"
   and the NAS webhook.
"""
import time
from unittest.mock import patch

import cron.scheduler as sched


def _clean(job_id, home):
    sched._running_job_ids.discard(sched._inflight_key(job_id, home))
    sched._running_since.pop(sched._inflight_key(job_id, home), None)
    sched._running_futures.pop(sched._inflight_key(job_id, home), None)
    sched._running_fire_owners.pop(sched._inflight_key(job_id, None), None)


class TestSweepNotifiesLifecycle:
    def _sweep_releases(self, tmp_path):
        """Inject an aged, future-less claim and sweep it; returns (released,
        observer-snapshot list). Home is pinned so the sweep actually sees the
        claim (the sweep only considers keys for _get_hermes_home())."""
        job_id = "sweep-notify-test"
        key = sched._inflight_key(job_id, tmp_path)
        calls = []

        def cb():
            calls.append(set(sched.get_running_job_ids()))

        sched._running_job_ids.add(key)
        sched._running_since[key] = time.time() - 6 * 60 * 60  # 6h: past allowance
        sched.register_job_lifecycle_callback(cb)
        try:
            with patch.object(sched, "_get_hermes_home", return_value=tmp_path):
                released = sched.sweep_stale_inflight([])
            return released, calls
        finally:
            sched.unregister_job_lifecycle_callback(cb)
            _clean(job_id, tmp_path)

    def test_sweep_notifies_when_it_releases(self, tmp_path):
        released, calls = self._sweep_releases(tmp_path)
        assert released == ["sweep-notify-test"], (
            f"sweep did not release the aged claim: {released}")
        # RED pre-fix: the release moved the count with zero notifies. GREEN:
        # the observer fired with the claim gone.
        assert any("sweep-notify-test" not in c for c in calls), (
            f"forced release fired no lifecycle notify: {calls}")

    def test_sweep_with_no_stale_claims_does_not_notify(self, tmp_path):
        calls = []

        def cb():
            calls.append(1)

        sched.register_job_lifecycle_callback(cb)
        try:
            assert sched.sweep_stale_inflight([]) == []
        finally:
            sched.unregister_job_lifecycle_callback(cb)
        assert calls == []


class TestSplitFireNotifiesLifecycle:
    def test_run_one_job_registers_and_releases_with_notify(self, monkeypatch, tmp_path):
        """run_one_job's fire-owner registration and its finally-pop must each
        fire lifecycle observers (the gateway active_agents persist)."""
        calls = []

        def cb():
            calls.append(len(sched.get_running_job_ids()))

        job = {"id": "split-fire-notify", "name": "sf", "_scheduled_instant": None}
        monkeypatch.setattr(sched, "create_execution",
                            lambda jid, source, scheduled_instant=None: {"id": "exec-1"})
        monkeypatch.setattr(sched, "_launch_external_cron_worker", lambda _job: False)
        monkeypatch.setattr(sched, "_run_with_fire_claim_heartbeat",
                            lambda job, body, **kw: True)
        monkeypatch.setenv("_HERMES_CRON_EXTERNAL_WORKER", "exec-1")

        sched.register_job_lifecycle_callback(cb)
        try:
            sched.run_one_job(job)
        finally:
            sched.unregister_job_lifecycle_callback(cb)
            _clean(job["id"], None)

        # RED on the pre-fix scheduler: zero notifies. GREEN: >= 2 (register
        # fire, release fire). The count itself must return to baseline.
        assert len([c for c in calls]) >= 2

    def test_run_one_job_notify_survives_body_exception(self, monkeypatch):
        """The finally-side notify must fire even when the body raises."""
        calls = []

        def cb():
            calls.append(1)

        job = {"id": "split-fire-boom", "name": "sfb", "_scheduled_instant": None}
        monkeypatch.setattr(sched, "create_execution",
                            lambda jid, source, scheduled_instant=None: {"id": "exec-2"})
        monkeypatch.setattr(sched, "_launch_external_cron_worker", lambda _job: False)

        def boom(job, body, **kw):
            raise RuntimeError("body failed")

        monkeypatch.setattr(sched, "_run_with_fire_claim_heartbeat", boom)
        monkeypatch.setenv("_HERMES_CRON_EXTERNAL_WORKER", "exec-2")

        sched.register_job_lifecycle_callback(cb)
        try:
            import pytest
            with pytest.raises(RuntimeError):
                sched.run_one_job(job)
        finally:
            sched.unregister_job_lifecycle_callback(cb)
            _clean(job["id"], None)

        assert len(calls) >= 2  # register + finally-pop both notified
