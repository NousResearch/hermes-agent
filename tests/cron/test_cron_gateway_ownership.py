"""Regression tests for live-gateway ownership of continuable cron delivery."""

from __future__ import annotations

import time


class _RunningLoop:
    def is_running(self) -> bool:
        return True


def _continuable_discord_job(job_id: str = "cron-gateway-owner") -> dict:
    return {
        "id": job_id,
        "name": "continuable discord",
        "prompt": "brief me",
        "deliver": "discord:123456789",
        "attach_to_session": True,
        "schedule": {"kind": "interval", "seconds": 3600},
        "next_run_at": "2000-01-01T00:00:00+00:00",
    }


def test_should_yield_cron_to_live_gateway_for_adapterless_continuable_job(monkeypatch):
    from cron import scheduler as sched

    monkeypatch.setattr(sched, "_live_gateway_running_for_profile", lambda: True)

    assert sched._should_yield_cron_to_live_gateway(_continuable_discord_job()) is True


def test_should_not_yield_when_current_process_has_live_adapter_context(monkeypatch):
    from cron import scheduler as sched

    monkeypatch.setattr(sched, "_live_gateway_running_for_profile", lambda: True)

    assert (
        sched._should_yield_cron_to_live_gateway(
            _continuable_discord_job(),
            adapters={"discord": object()},
            loop=_RunningLoop(),
        )
        is False
    )


def test_tick_leaves_due_jobs_for_live_gateway_before_advancing_claims(
    tmp_path, monkeypatch
):
    from cron import scheduler as sched

    due_job = _continuable_discord_job()
    monkeypatch.setattr(
        sched, "_get_lock_paths", lambda: (tmp_path, tmp_path / ".tick.lock")
    )
    monkeypatch.setattr(sched, "_last_dead_owner_reap_at", time.monotonic())
    monkeypatch.setattr(sched, "get_due_jobs", lambda: [due_job])
    monkeypatch.setattr(sched, "_live_gateway_running_for_profile", lambda: True)

    advanced: list[list[str]] = []
    claimed: list[str] = []
    monkeypatch.setattr(
        sched, "advance_next_runs", lambda ids: advanced.append(list(ids)) or 1
    )
    monkeypatch.setattr(
        sched, "claim_job_for_fire", lambda job_id, **_: claimed.append(job_id) or True
    )

    assert sched.tick(verbose=False, adapters=None, loop=None, sync=True) == 0
    assert advanced == []
    assert claimed == []
