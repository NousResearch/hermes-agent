"""The stop drain and the cron leash must be fitted to the ONE deadline the shutdown watchdog is armed
with, measured from when it was armed -- not to a second formula (ExitTimeOut - reserve) that ignores
both the watchdog's dump margin and the time stop() spent before the drain started."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

import gateway.run_shutdown as rs
from gateway.restart import LAUNCHD_STOP_CLEANUP_RESERVE_S, fit_drain_to_armed_leash
from gateway.run import GatewayRunner


def test_fit_leaves_the_cleanup_reserve_before_the_armed_exit():
    # ExitTimeOut 60 under launchd: drain capped to 50, watchdog armed at min(50 + 60, 60 - 2) = 58.
    assert fit_drain_to_armed_leash(50.0, 58.0, 0.0) == 58.0 - LAUNCHD_STOP_CLEANUP_RESERVE_S
    # Time already spent in stop() before the drain comes out of the drain, not the teardown.
    assert fit_drain_to_armed_leash(50.0, 58.0, 5.0) == pytest.approx(43.0)
    # Never extends a shorter configured drain; never negative; no armed watchdog -> unchanged.
    assert fit_drain_to_armed_leash(20.0, 58.0, 5.0) == 20.0
    assert fit_drain_to_armed_leash(50.0, 58.0, 100.0) == 0.0
    assert fit_drain_to_armed_leash(50.0, None, 5.0) == 50.0


def _stop_runner():
    return SimpleNamespace(
        _restart_drain_timeout=50.0, _launchd_exit_timeout_s=60.0, _stop_requested_by_signal=True,
        _restart_requested=False, _active_deferred_agent_worker_count=lambda: 0,
    )


def test_stop_fits_the_drain_to_the_armed_leash_and_the_elapsed_time(monkeypatch):
    seen = {}

    async def begin(self, ctx):
        ctx.started_at = time.monotonic()
        await asyncio.sleep(0.3)  # stop() work before the drain (notify sessions, ...)

    async def drain(self, timeout, ctx):
        seen["timeout"], seen["leash"] = timeout, ctx.armed_leash_s
        ctx.timed_out = False

    async def noop(self, *a, **k):
        return None

    monkeypatch.setattr(GatewayRunner, "_stop_begin_teardown", begin, raising=False)
    monkeypatch.setattr(GatewayRunner, "_stop_drain_active_work", drain, raising=False)
    for name in ("_stop_interrupt_remaining_work", "_stop_finalize_agents_and_adapters",
                 "_stop_release_runtime_state", "_stop_persist_exit_state"):
        monkeypatch.setattr(GatewayRunner, name, noop, raising=False)
    monkeypatch.setattr(GatewayRunner, "_stop_quiesce_and_close_session_dbs", lambda *a, **k: None, raising=False)

    asyncio.run(rs.GatewayShutdownMixin._stop_impl(_stop_runner()))

    assert seen["leash"] == 58.0  # min(50 + 60 grace, 60 - 2 dump margin)
    expected = 58.0 - LAUNCHD_STOP_CLEANUP_RESERVE_S - 0.3
    assert expected - 0.25 <= seen["timeout"] <= expected + 0.01, seen["timeout"]


def test_cron_leash_is_the_armed_leash(monkeypatch):
    captured = {}

    def spy(timeout, cfg, *, watchdog_delay, elapsed):
        captured.update(watchdog_delay=watchdog_delay, elapsed=elapsed)
        return timeout

    monkeypatch.setattr(rs, "resolve_cron_drain_budget", spy)

    async def mark(*a, **k):
        return []

    async def drain_agents(self, timeout, cron_timeout):
        return {}, True  # timed out: return right after the drain

    monkeypatch.setattr(GatewayRunner, "_mark_running_sessions_resume_pending", mark, raising=False)
    runner = SimpleNamespace(
        _cron_drain_timeout=30.0, _active_cron_job_count=lambda: 1, _active_api_run_count=lambda: 0,
        _running_agent_count=lambda: 0, _stop_requested_by_signal=True, _launchd_exit_timeout_s=60.0,
        _restart_drain_timeout=50.0,
    )
    runner._drain_active_agents = lambda t, c: drain_agents(runner, t, c)
    ctx = rs.GatewayShutdownMixin._StopContext(deferred_count=lambda: 0)
    ctx.started_at = time.monotonic()
    ctx.armed_leash_s, ctx.armed_at = 58.0, time.monotonic() - 4.0

    asyncio.run(rs.GatewayShutdownMixin._stop_drain_active_work(runner, 20.0, ctx))

    # Re-deriving from the (already fitted) drain would give min(20 + 60, 58) and elapsed ~0.
    assert captured["watchdog_delay"] == 58.0
    assert captured["elapsed"] >= 4.0
