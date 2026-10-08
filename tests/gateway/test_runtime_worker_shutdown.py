"""Shutdown counts managed claims and retains profile ownership until actual threads stop."""
import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from gateway import run_runtime
from gateway.run_shutdown import GatewayShutdownMixin
from gateway.session_authorities import SessionAuthorities


@pytest.mark.asyncio
async def test_managed_claims_participate_in_zero_and_nonzero_drain(tmp_path):
    release = asyncio.Event()
    task = asyncio.create_task(release.wait())
    stopped = []
    live = SimpleNamespace(task=task, route='route', event_stream=SimpleNamespace(execution={'execution_generation': 2}))
    authority = SimpleNamespace(sessions={'s': live}, _managed_workers={
        's': SimpleNamespace(control=lambda frame: stopped.append(frame), closed=threading.Event())})
    runner = SimpleNamespace(session_authority=authority, _running_agents={},
        _running_agent_count=lambda: 0, _active_cron_job_count=lambda: 0, _active_api_run_count=lambda: 0,
        _active_deferred_agent_worker_count=lambda: 0, _snapshot_running_agents=lambda: {},
        _update_runtime_status=lambda *a: None)
    runner._drain_work_counts = GatewayShutdownMixin._drain_work_counts.__get__(runner)
    authority.runner = runner
    authority.pending_stops = {}
    try:
        assert GatewayShutdownMixin._active_work_count(runner) == 1
        _, timed_out = await GatewayShutdownMixin._drain_active_agents(runner, timeout=0)
        assert timed_out is True
        run_runtime.stop_managed_turns(runner)
        assert stopped == [{'type': 'stop'}]
        drain = asyncio.create_task(GatewayShutdownMixin._drain_active_agents(runner, timeout=5))
        await asyncio.sleep(0)
        assert not drain.done()
        authority._managed_workers['s'].closed.set()
        release.set()
        _, timed_out = await asyncio.wait_for(drain, 5)
        assert timed_out is False
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_profile_timeout_keeps_authority_until_physical_worker_exits(tmp_path, monkeypatch):
    started, release, done, stopped = (threading.Event() for _ in range(4))
    def turn():
        started.set()
        try:
            assert release.wait(10)
        finally:
            done.set()
    task = asyncio.create_task(asyncio.to_thread(turn))
    assert await asyncio.to_thread(started.wait, 5)
    launch, beta = tmp_path, tmp_path / 'profiles' / 'beta'
    beta.mkdir(parents=True)
    registry = SessionAuthorities(launch)
    registry.add(launch, SimpleNamespace(profile_id=str(launch)))
    agent = SimpleNamespace(hard_interrupt=lambda message: stopped.set())
    live = SimpleNamespace(task=task, route='beta-route', event_stream=SimpleNamespace(execution={'execution_generation': 2}))
    authority = SimpleNamespace(sessions={'s': live}, profile_id=str(beta), pending_stops={},
        agent=lambda ref: agent, _turn_workers={1: (SimpleNamespace(worker_done=done), [agent])})
    registry.add(beta, authority, name='beta')
    runner = SimpleNamespace(session_authorities=registry, session_runtime_descriptor={},
                             session_ticket_store=SimpleNamespace(profile_ids=registry.profile_ids()))
    unbound = []
    monkeypatch.setattr('gateway.session_cron.unbind_owner', lambda a: unbound.append(a))
    authority.runner = runner
    monkeypatch.setattr(run_runtime, 'TURN_SETTLE_SECONDS', 0)
    try:
        assert await run_runtime.unserve_profile_runtime(runner, beta) is False
        assert stopped.is_set() and not done.is_set()
        assert registry.for_home(beta) is authority
        assert not unbound
        assert str(beta) in runner.session_ticket_store.profile_ids
        release.set()
        await asyncio.wait_for(task, 5)
        await run_runtime.unserve_profile_runtime(runner, beta)
        assert registry.for_home(beta) is None
        assert str(beta) not in runner.session_ticket_store.profile_ids
        assert unbound == [authority]
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_shutdown_lets_an_interrupted_thread_unwind_within_the_settle_budget(tmp_path):
    stopped, done = threading.Event(), threading.Event()
    def unwind():
        assert stopped.wait(5)
        time.sleep(0.2)
        done.set()
    thread = threading.Thread(target=unwind, daemon=True)
    thread.start()
    registry = SessionAuthorities(tmp_path)
    agent = SimpleNamespace(hard_interrupt=lambda message: stopped.set())
    registry.add(tmp_path, SimpleNamespace(sessions={}, profile_id=str(tmp_path), pending_stops={}, runner=SimpleNamespace(),
        _turn_workers={1: (SimpleNamespace(worker_done=done), [agent])}))
    await asyncio.wait_for(run_runtime.settle_gateway_runtime(SimpleNamespace(session_authorities=registry)), 5)
    assert stopped.is_set() and done.is_set()
    thread.join(5)
