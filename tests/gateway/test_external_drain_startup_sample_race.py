"""Shutdown across a false off-loop marker read must not publish running.

Retains independent_review_probe.py::shutdown_during_clear_second_sample's
real startup/wait/abort methods and inert service/transport boundaries.
"""
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import gateway.drain_control as dc
from gateway.run_shutdown import GatewayShutdownMixin
from gateway.run_startup import GatewayStartupMixin


def _inert_runner(monkeypatch):
    runner = SimpleNamespace(_running=False, _draining=False, _restart_requested=False,
                             _external_drain_active=False, _shutdown_event=asyncio.Event(), _stop_task=None,
                             _startup_parked_platforms=False, adapters={},
                             delivery_router=SimpleNamespace(adapters={}))
    runner._startup_should_abort = lambda: GatewayStartupMixin._startup_should_abort(runner)
    runner._abort_startup_if_shutdown_requested = lambda: GatewayStartupMixin._abort_startup_if_shutdown_requested(runner)
    runner._start_wait_external_drain = lambda: GatewayStartupMixin._start_wait_external_drain(runner)
    runner._exit_external_drain = lambda: GatewayShutdownMixin._exit_external_drain(runner)
    runner._enter_external_drain = lambda: GatewayShutdownMixin._enter_external_drain(runner)
    runner._active_work_count = lambda: 0
    runner._start_install_faulthandler = lambda: None
    runner._start_log_startup_environment = AsyncMock()
    runner._start_check_access_policy = lambda: False
    runner._start_recover_previous_run = AsyncMock()
    runner._start_startup_warmup = lambda: None
    runner._start_prefilter_platforms = AsyncMock(return_value=(False, 0, [], []))
    runner._start_connect_pending = AsyncMock(return_value=[])
    runner._start_aggregate_connect_results = AsyncMock(return_value=0)
    runner._start_secondary_profiles = AsyncMock(return_value=(False, 0))
    runner._start_handle_no_connections = lambda *a: False
    runner._serving_state = lambda: 'running'
    runner._wire_teams_pipeline_runtime = lambda: None
    runner._install_plugin_message_injector = lambda: None
    runner._update_runtime_status = lambda *a: None
    observed = {'restoration_called': False, 'watchers_started': False}

    async def restoration_boundary(*args):
        observed.update(restoration_called=True, running_at_restore=runner._running,
                        draining_at_restore=runner._draining, shutdown_at_restore=runner._shutdown_event.is_set())

    def start_watchers():
        observed['watchers_started'] = True

    runner._start_finish_wiring = restoration_boundary
    runner._start_spawn_background_watchers = start_watchers
    monkeypatch.setattr('hermes_cli.observability.shared_metrics_startup.record_process_ready', lambda *a, **k: None)
    return runner, observed


@pytest.mark.asyncio
@pytest.mark.parametrize('sample', ['second-absent', 'second-cleared', 'first-absent'])
async def test_shutdown_during_clear_marker_sample(monkeypatch, sample):
    runner, observed = _inert_runner(monkeypatch)
    reached = asyncio.Event()
    permit_read = threading.Event()
    loop = asyncio.get_running_loop()
    calls = 0
    held_sample = {'first-absent': 1, 'second-absent': 2, 'second-cleared': 3}[sample]

    def held_absent_read():
        nonlocal calls
        calls += 1
        if sample == 'second-cleared' and calls == 2:
            return True
        if calls == held_sample:
            loop.call_soon_threadsafe(reached.set)
            if not permit_read.wait(3):
                raise TimeoutError('private marker read release missing')
        return False

    monkeypatch.setattr(dc, 'drain_requested', held_absent_read)
    task = asyncio.create_task(GatewayStartupMixin._start_impl(runner))
    try:
        await asyncio.wait_for(reached.wait(), 3)
        if sample == 'second-cleared':
            assert runner._external_drain_active is True
        # Completed synthetic shutdown while the actual off-loop read is pending.
        runner._draining = True
        runner._running = False
        runner._shutdown_event.set()
        permit_read.set()
        assert await asyncio.wait_for(task, 3) is True
        assert not observed['restoration_called'], 'startup restored after shutdown during a clear marker read'
        assert runner._running is False
        assert runner._draining is True and runner._shutdown_event.is_set()
        assert observed['watchers_started'] is False and calls == held_sample
        if sample == 'first-absent':
            runner._start_recover_previous_run.assert_not_awaited()
    finally:
        permit_read.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_clear_second_sample_without_shutdown_starts_normally(monkeypatch):
    runner, observed = _inert_runner(monkeypatch)
    monkeypatch.setattr(dc, 'drain_requested', lambda: False)
    assert await GatewayStartupMixin._start_impl(runner) is True
    assert observed == {'restoration_called': True, 'watchers_started': True,
                        'running_at_restore': True, 'draining_at_restore': False, 'shutdown_at_restore': False}
