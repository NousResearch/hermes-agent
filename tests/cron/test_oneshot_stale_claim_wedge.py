"""Regression tests for est-20mb — wedged one-shot parked forever by a STALE claim.

A finite one-shot whose tick died after stamping ``run_claim`` kept
``state=scheduled`` + ``next_run_at`` in the past + ``last_run_at=None`` with
``repeat.completed`` consumed, because ``_retire_expired_oneshot`` held ANY
claim-bearing record ("a run may be in flight elsewhere") on presence alone —
never LIVENESS. With the run_claim TTL long expired and no live run, the record
sat OVERDUE indefinitely (keeper jobs 4aa8422fa577 / 8e814eb0c6ad, 2026-10-03:
21h overdue while the ticker heartbeat was 35s fresh): never fired, never
removed, no diagnostic.

The fix gates the hold on ``_claim_is_live`` (TTL + owner-pid probe). A stale
claim no longer holds: the consumed-budget case falls through to
``_oneshot_dispatch_limit_reached`` (retire + wedged diagnostic, fail loud);
other stale/expired records retire with the missed-run diagnostic; a genuinely
LIVE claim still holds so its ``mark_job_run`` can land.
"""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cron.jobs as jobs_mod


@pytest.fixture
def cron_store(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    (hermes_home / "cron").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(jobs_mod, "HERMES_DIR", hermes_home)
    monkeypatch.setattr(jobs_mod, "CRON_DIR", hermes_home / "cron")
    monkeypatch.setattr(jobs_mod, "JOBS_FILE", hermes_home / "cron" / "jobs.json")
    monkeypatch.setattr(jobs_mod, "OUTPUT_DIR", hermes_home / "cron" / "output")
    return hermes_home


def _wedged_oneshot(**overrides) -> dict:
    now = jobs_mod._hermes_now()
    run_at = (now - timedelta(hours=21)).isoformat()  # far outside grace
    rec = {
        "id": "wedge", "name": "wedge", "enabled": True, "state": "scheduled",
        "schedule": {"kind": "once", "run_at": run_at},
        "next_run_at": run_at, "last_run_at": None, "last_status": None,
        "repeat": {"times": 1, "completed": 1},
        "deliver": "local", "prompt": "est-20mb", "created_at": run_at,
    }
    rec.update(overrides)
    jobs_mod.save_jobs([rec])
    return rec


def _disk_ids():
    return {j["id"] for j in jobs_mod.load_jobs()}


def _diag_written(job_id):
    d = jobs_mod.OUTPUT_DIR / job_id
    return d.is_dir() and any(d.iterdir())


class TestStaleClaimNoLongerParks:
    def test_stale_run_claim_consumed_budget_removed_with_diagnostic(self, cron_store):
        now = jobs_mod._hermes_now()
        _wedged_oneshot(run_claim={"at": (now - timedelta(hours=20)).isoformat(),
                                   "by": "dead-tick"})
        assert jobs_mod.get_due_jobs() == []
        assert "wedge" not in _disk_ids()          # no longer sits scheduled forever
        assert _diag_written("wedge")              # fails loud, not silently

    def test_stale_fire_claim_removed(self, cron_store):
        now = jobs_mod._hermes_now()
        _wedged_oneshot(fire_claim={"at": (now - timedelta(hours=9)).isoformat(),
                                    "by": "dead-host:tok"})
        assert jobs_mod.get_due_jobs() == []
        assert "wedge" not in _disk_ids()
        assert _diag_written("wedge")

    def test_stale_claim_unconsumed_budget_retires_never_fires(self, cron_store):
        # Completed < times: no agent side effect can have run (claim_dispatch
        # increments before dispatch), but the instant is outside the never-fire
        # grace window — retire with the missed diagnostic, do NOT re-fire hours late.
        now = jobs_mod._hermes_now()
        _wedged_oneshot(repeat={"times": 1, "completed": 0},
                        run_claim={"at": (now - timedelta(hours=20)).isoformat(),
                                   "by": "dead-tick"})
        assert jobs_mod.get_due_jobs() == []
        assert "wedge" not in _disk_ids()
        assert _diag_written("wedge")


class TestLiveClaimStillHolds:
    def test_live_run_claim_record_kept(self, cron_store):
        now = jobs_mod._hermes_now()
        _wedged_oneshot(run_claim={"at": now.isoformat(), "by": "live-tick"})
        assert jobs_mod.get_due_jobs() == []       # not re-dispatched while claimed
        assert "wedge" in _disk_ids()              # kept for its mark_job_run
        assert not _diag_written("wedge")

    def test_live_fire_claim_record_kept(self, cron_store):
        now = jobs_mod._hermes_now()
        _wedged_oneshot(fire_claim={"at": now.isoformat(), "by": "host:tok"})
        assert jobs_mod.get_due_jobs() == []
        assert "wedge" in _disk_ids()
